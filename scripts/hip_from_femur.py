"""Rebuild the greater trochanter (GTR) position from the femoral epicondyles (LE, ME).

LE and ME alone only fix the epicondylar axis; the femur can still rotate about it, so the
third point comes from the hip joint of the scaled model. IK is re-run on the test trial with
both GTR markers at zero weight: the femur pose then follows from LE / ME at the knee and the
hip joint on the pelvis (ASIS and the rest of the body). The model GTR marker, placed on the
femur from the static trial during scaling, is read from that pose and compared with the
measured GTR marker.

Uses the model scaled by scale_model.py and the trial's .trc; scaling and the regular IK / ID
(run_ik_id.py) are not redone here. Edit the settings below, then run:
    python scripts/hip_from_femur.py
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import opensim as osm

import config as cfg
from hip_from_cluster import read_trc
from run_ik_id import run_ik

# ---------- Settings ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = '10_16_2026 AB01'         # folder name under Data/ and Results/
TEST_TRIAL = 'LG_1p2_NoExo_shoes'
SCALED_MODEL = os.path.join(_ROOT, 'Results', SESSION, 'model', 'Static_NoExo_shoes_scaled.osim')
IK_SETUP = os.path.join(_ROOT, 'setup', 'iktemplate.xml')   # IK marker weights (GTR set to 0 below)
IK_MARKER_WEIGHTS = cfg.ik_marker_weights(IK_SETUP)
SIDES = ('R', 'L')
TARGET = 'GTR'

TRC_FILE = os.path.join(_ROOT, 'Data', SESSION, f'{TEST_TRIAL}.trc')
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


def model_marker_paths(model, ik_file, marker_names):
    """Lab position (mm) of model markers for every frame of an IK solution."""
    state = model.initSystem()
    times, values = read_coordinates(ik_file, model)
    coords = model.getCoordinateSet()
    markers = {n: model.getMarkerSet().get(n) for n in marker_names}
    out = {n: np.zeros((len(times), 3)) for n in marker_names}
    for k in range(len(times)):
        for name, v in values.items():
            coords.get(name).setValue(state, v[k], False)
        model.realizePosition(state)
        for n, marker in markers.items():
            out[n][k] = marker.getLocationInGround(state).to_numpy() * 1000.0
    return times, out


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


def main():
    if not os.path.isfile(SCALED_MODEL):
        raise FileNotFoundError(f"Scaled model not found: {SCALED_MODEL}\nRun scale_model.py first.")
    os.makedirs(OUT_DIR, exist_ok=True)
    ik_file = os.path.join(OUT_DIR, f'ik_noGTR_{TEST_TRIAL}.mot')
    no_gtr_weights = {**IK_MARKER_WEIGHTS, **{side + TARGET: 0.0 for side in SIDES}}
    run_ik(SCALED_MODEL, TRC_FILE, no_gtr_weights, ik_file,
           os.path.join(OUT_DIR, f'ik_marker_errors_noGTR_{TEST_TRIAL}.sto'), f'{TEST_TRIAL} (no GTR)')

    model = osm.Model(SCALED_MODEL)
    names = [side + TARGET for side in SIDES]
    ik_time, derived = model_marker_paths(model, ik_file, names)
    trc_time, measured = read_trc(TRC_FILE)
    if len(ik_time) != len(trc_time) or not np.allclose(ik_time, trc_time, atol=1e-6):
        raise ValueError('IK and TRC time columns do not match')

    est_trc = os.path.join(OUT_DIR, f'{TEST_TRIAL}_with_{TARGET}_est.trc')
    write_trc_with_markers(TRC_FILE, est_trc, {f'{name}_est': derived[name] for name in names})
    print(f'Markers {", ".join(f"{n}_est" for n in names)} added to {est_trc}')

    for side in SIDES:
        name = side + TARGET
        error = derived[name] - measured[name]
        dist = np.linalg.norm(error, axis=1)
        print(f'{SIDE_NAME[side]} {TARGET}: 3D error mean {np.nanmean(dist):.1f} mm, '
              f'RMS {np.sqrt(np.nanmean(dist ** 2)):.1f} mm, max {np.nanmax(dist):.1f} mm; '
              f'mean X/Y/Z {np.nanmean(error[:, 0]):.1f} / {np.nanmean(error[:, 1]):.1f} / '
              f'{np.nanmean(error[:, 2]):.1f} mm')

        fig, (ax_dist, ax_xyz) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
        ax_dist.plot(trc_time, dist, color='black', linewidth=0.8)
        ax_dist.set_ylabel('3D distance error (mm)')
        ax_dist.set_ylim(bottom=0)
        for k, (axis_name, color) in enumerate(zip('XYZ', AXIS_COLORS)):
            ax_xyz.plot(trc_time, error[:, k], color=color, linewidth=0.8, label=axis_name)
        ax_xyz.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
        ax_xyz.set_ylabel('Error (mm)')
        ax_xyz.set_xlabel('Time (s)')
        ax_xyz.legend(loc='upper right')
        fig.suptitle(f'{SESSION} {TEST_TRIAL}: {SIDE_NAME[side]} {TARGET}, derived minus measured\n'
                     f'(femur from LE / ME + model hip joint, IK without {TARGET} markers)')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'gtr_error_{SIDE_NAME[side].lower()}_{TEST_TRIAL}.png'), dpi=150)

    print(f'Plots saved to {OUT_DIR}')
    plt.show()


if __name__ == '__main__':
    main()
