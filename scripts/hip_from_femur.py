"""Rebuild the human greater trochanter (GTR) in exo trials from the femoral epicondyles (LE, ME).

In the exo trials the GTR markers sit on the exo motors, so the human GTR is reconstructed:
LE and ME fix the epicondylar axis and the hip joint of the scaled human model (on the pelvis)
fixes the rest of the femur. IK is run with both GTR markers at zero weight and the model GTR
marker (placed on the femur from the no-exo static) is read from that pose.

Subject-specific correction (USE_CORRECTION): the GTR skin marker slides over the bone as the hip
and knee move. On the no-exo walking trial (GTR visible) the reconstruction error, in femur axes,
is fitted as  error = c0 + c1 * hip_flexion + c2 * knee_angle  per axis and side, and subtracted
from the exo-trial reconstruction.

Inputs: the no-exo static (its scaled model, made by scale_model.py), the no-exo walking trial
and the exo trials. Shows the corrected vs uncorrected error on the no-exo walking trial and
writes, for every exo trial, Results/<session>/hip_from_femur/<trial>_with_GTR_est.trc: the trial
with RGTR_est / LGTR_est added. Edit the settings below, then run:
    python scripts/hip_from_femur.py
"""
import contextlib
import io
import os
import tempfile

import numpy as np
import matplotlib.pyplot as plt
import opensim as osm

import config as cfg
from hip_from_cluster import read_trc
from run_ik_id import run_ik

# ---------- Settings ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = '10_16_2026 AB01'                       # folder name under Data/ and Results/
CALIBRATION_STATIC = 'Static_NoExo_shoes'         # no-exo static: its scaled model is used for everything
CALIBRATION_WALK = 'LG_1p2_NoExo_shoes'           # no-exo walking (GTR visible): fits the correction
EXO_TRIALS = ['Static_Exo', 'LG_1p2_ExoOn']       # trials that get the reconstructed GTR
USE_CORRECTION = True                             # False = write the uncorrected reconstruction
IK_SETUP = os.path.join(_ROOT, 'setup', 'iktemplate.xml')   # IK marker weights (GTR set to 0 below)
SIDES = ('R', 'L')
TARGET = 'GTR'

DATA_DIR = os.path.join(_ROOT, 'Data', SESSION)
SCALED_MODEL = os.path.join(_ROOT, 'Results', SESSION, 'model', f'{CALIBRATION_STATIC}_scaled.osim')
OUT_DIR = os.path.join(_ROOT, 'Results', SESSION, 'hip_from_femur')

SIDE_NAME = {'R': 'Right', 'L': 'Left'}
AXIS_COLORS = ('tab:red', 'tab:green', 'tab:purple')


def read_coordinates(ik_file, model):
    """IK .mot as (times, {coordinate: values in model units (rad / m)})."""
    table = osm.TimeSeriesTable(ik_file)
    times = np.array(table.getIndependentColumn())
    in_degrees = table.getTableMetaDataAsString('inDegrees') == 'yes'
    coords = model.getCoordinateSet()
    values = {}
    for i in range(coords.getSize()):
        c = coords.get(i)
        col = table.getDependentColumn(c.getName()).to_numpy()
        values[c.getName()] = np.deg2rad(col) if in_degrees and c.getMotionType() == 1 else col
    return times, values


def reconstruct(model, trial, work_dir):
    """GTR-free IK of <trial>.trc, then per frame and side: model GTR in the lab (mm), femur
    orientation (columns = femur axes in the lab) and [hip flexion, knee angle] (deg)."""
    ik_file = os.path.join(work_dir, f'ik_noGTR_{trial}.mot')
    weights = {**cfg.ik_marker_weights(IK_SETUP), **{s + TARGET: 0.0 for s in SIDES}}
    run_ik(SCALED_MODEL, os.path.join(DATA_DIR, f'{trial}.trc'), weights, ik_file,
           os.path.join(work_dir, 'ik_errors.sto'), f'{trial} (no GTR)')

    state = model.initSystem()
    times, values = read_coordinates(ik_file, model)
    coords = model.getCoordinateSet()
    unit = [osm.Vec3(1, 0, 0), osm.Vec3(0, 1, 0), osm.Vec3(0, 0, 1)]
    markers = {s: model.getMarkerSet().get(s + TARGET) for s in SIDES}
    femurs = {s: model.getBodySet().get(f'femur_{s.lower()}') for s in SIDES}
    gtr = {s: np.zeros((len(times), 3)) for s in SIDES}
    rot = {s: np.zeros((len(times), 3, 3)) for s in SIDES}
    for k in range(len(times)):
        for name, v in values.items():
            coords.get(name).setValue(state, v[k], False)
        model.realizePosition(state)
        for s in SIDES:
            gtr[s][k] = markers[s].getLocationInGround(state).to_numpy() * 1000.0
            for i in range(3):
                rot[s][k][:, i] = femurs[s].expressVectorInGround(state, unit[i]).to_numpy()
    angles = {s: np.column_stack([np.rad2deg(values[f'hip_flexion_{s.lower()}']),
                                  np.rad2deg(values[f'knee_angle_{s.lower()}'])]) for s in SIDES}
    return times, gtr, rot, angles


def correction_inputs(angles):
    return np.column_stack([np.ones(len(angles)), angles])


def fit_correction(gtr, rot, angles, measured):
    """Least-squares fit of the femur-frame error on [1, hip flexion, knee angle], per side.
    Returns {side: (3 inputs x 3 femur axes)}."""
    coef = {}
    for s in SIDES:
        error_femur = np.einsum('nji,nj->ni', rot[s], gtr[s] - measured[s + TARGET])
        ok = np.isfinite(error_femur).all(axis=1)
        coef[s], *_ = np.linalg.lstsq(correction_inputs(angles[s])[ok], error_femur[ok], rcond=None)
    return coef


def apply_correction(gtr, rot, angles, coef):
    return {s: gtr[s] - np.einsum('nij,nj->ni', rot[s], correction_inputs(angles[s]) @ coef[s]) for s in SIDES}


def write_trc_with_markers(src_trc, dst_trc, new_markers):
    """Copy src_trc to dst_trc with extra markers appended. new_markers: {name: (n_frames, 3) mm}."""
    with open(src_trc) as f:
        lines = f.read().splitlines()
    data = [l for l in lines[5:] if l.strip()]
    n_old = int(lines[2].split('\t')[3])
    if any(len(xyz) != len(data) for xyz in new_markers.values()):
        raise ValueError('New marker length does not match the number of TRC frames')

    lines[0] = '\t'.join(lines[0].split('\t')[:3] + [os.path.basename(dst_trc)])
    header = lines[2].split('\t')
    header[3] = str(n_old + len(new_markers))
    lines[2] = '\t'.join(header)
    lines[3] = lines[3].rstrip('\t') + '\t\t\t' + ''.join(f'{name}\t\t\t' for name in new_markers)
    lines[4] = lines[4].rstrip('\t') + ''.join(
        f'\tX{i}\tY{i}\tZ{i}' for i in range(n_old + 1, n_old + len(new_markers) + 1)) + '\t'
    fmt = lambda v: '' if np.isnan(v) else f'{v:.5f}'
    for k, row in enumerate(data):
        extra = '\t'.join(fmt(v) for xyz in new_markers.values() for v in xyz[k])
        data[k] = row.rstrip('\t') + '\t' + extra
    with open(dst_trc, 'w', newline='\n') as f:
        f.write('\n'.join(lines[:5] + data) + '\n')


def error_stats(error):
    dist = np.linalg.norm(error, axis=1)
    return f'mean {np.nanmean(dist):.1f} mm, RMS {np.sqrt(np.nanmean(dist ** 2)):.1f} mm, max {np.nanmax(dist):.1f} mm'


def plot_calibration(times, raw, corrected, measured):
    """2 x 2: 3D error (uncorrected vs corrected) on top, corrected X/Y/Z error below; left / right legs."""
    fig, axes = plt.subplots(2, 2, sharex=True, figsize=(14, 7))
    for col, s in enumerate(('L', 'R')):
        uncorrected_err = raw[s] - measured[s + TARGET]
        ax = axes[0, col]
        ax.plot(times, np.linalg.norm(uncorrected_err, axis=1), color='0.65', linewidth=0.8, label='uncorrected')
        if corrected is not None:
            ax.plot(times, np.linalg.norm(corrected[s] - measured[s + TARGET], axis=1),
                    color='black', linewidth=0.8, label='corrected')
        ax.set_title(f'{SIDE_NAME[s]} {TARGET}')
        ax.set_ylabel('3D distance error (mm)')
        ax.set_ylim(bottom=0)
        ax.legend(loc='upper right')

        err = (corrected[s] if corrected is not None else raw[s]) - measured[s + TARGET]
        ax = axes[1, col]
        for k, (axis_name, color) in enumerate(zip('XYZ', AXIS_COLORS)):
            ax.plot(times, err[:, k], color=color, linewidth=0.8, label=axis_name)
        ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
        ax.set_ylabel(f'Error (mm), {"corrected" if corrected is not None else "uncorrected"}')
        ax.set_xlabel('Time (s)')
        ax.legend(loc='upper right')
    fig.suptitle(f'{SESSION} {CALIBRATION_WALK}: reconstructed minus measured {TARGET}'
                 + (' (correction fitted on this trial)' if corrected is not None else ''))
    fig.tight_layout()


def main():
    if not os.path.isfile(SCALED_MODEL):
        raise FileNotFoundError(f'Scaled model not found: {SCALED_MODEL}\nRun scale_model.py on {CALIBRATION_STATIC} first.')
    os.makedirs(OUT_DIR, exist_ok=True)
    model = osm.Model(SCALED_MODEL)

    with tempfile.TemporaryDirectory() as work_dir:
        # calibration: no-exo walking, GTR visible
        cal_time, cal_raw, cal_rot, cal_angles = reconstruct(model, CALIBRATION_WALK, work_dir)
        _, cal_measured = read_trc(os.path.join(DATA_DIR, f'{CALIBRATION_WALK}.trc'))
        coef = fit_correction(cal_raw, cal_rot, cal_angles, cal_measured) if USE_CORRECTION else None
        cal_corrected = apply_correction(cal_raw, cal_rot, cal_angles, coef) if coef else None

        print(f'\nCalibration on {CALIBRATION_WALK} (model {os.path.basename(SCALED_MODEL)}):')
        for s in SIDES:
            line = f'  {SIDE_NAME[s]} {TARGET}: uncorrected {error_stats(cal_raw[s] - cal_measured[s + TARGET])}'
            if coef:
                line += f'  ->  corrected {error_stats(cal_corrected[s] - cal_measured[s + TARGET])}'
            print(line)

        # exo trials: reconstruct and write the .trc with the calculated GTR
        for trial in EXO_TRIALS:
            _, raw, rot, angles = reconstruct(model, trial, work_dir)
            estimate = apply_correction(raw, rot, angles, coef) if coef else raw
            src = os.path.join(DATA_DIR, f'{trial}.trc')
            dst = os.path.join(OUT_DIR, f'{trial}_with_{TARGET}_est.trc')
            write_trc_with_markers(src, dst, {f'{s}{TARGET}_est': estimate[s] for s in SIDES})
            print(f'  {trial}: {", ".join(f"{s}{TARGET}_est" for s in SIDES)} '
                  f'({"corrected" if coef else "uncorrected"}) added -> {dst}')

    plot_calibration(cal_time, cal_raw, cal_corrected, cal_measured)
    plt.show()


if __name__ == '__main__':
    main()
