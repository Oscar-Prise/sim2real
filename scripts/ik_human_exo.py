"""One .mot for the exo model: human joint angles from the calculated GTR, exo angles from the motor GTR.

Input: the trial .trc written by hip_from_femur.py (exo markers, RGTR / LGTR on the exo motors,
plus RGTR_est / LGTR_est = calculated human GTR).
  1. human angles: IK with the human (no-exo) scaled model, RGTR_est / LGTR_est used as its GTR
  2. exo angles:   IK with the exo scaled model, RGTR / LGTR (motors) as in the exo model, with every
                   human coordinate held at its value from step 1 - so the exo pelvis / exo hip
                   angles are relative to the same human pose that is written out
The output has every coordinate of the exo model. Load it onto the exo model in OpenSim.
The intermediate IK files are not kept. Edit the settings below, then run:
    python scripts/ik_human_exo.py
"""
import math
import os
import tempfile

import numpy as np
import pandas as pd
import opensim as osm

import config as cfg
from hip_from_cluster import read_trc
from run_ik_id import run_ik, write_opensim_table
from util_marker_report import print_ik_report
from util_progress_bar import ProgressBar

# ---------- Settings ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = '10_16_2026 AB01'
TRIAL = 'LG_1p2_ExoOn'
EST_TRC = os.path.join(_ROOT, 'Results', SESSION, 'hip_from_femur', f'{TRIAL}_with_GTR_est.trc')  # from hip_from_femur.py
HUMAN_MODEL = os.path.join(_ROOT, 'Results', SESSION, 'model', 'Static_NoExo_shoes_scaled.osim')
HUMAN_IK_SETUP = os.path.join(_ROOT, 'setup', 'iktemplate.xml')
EXO_MODEL = os.path.join(_ROOT, 'Results', SESSION, 'model', 'Static_Exo_scaled.osim')
EXO_IK_SETUP = os.path.join(_ROOT, 'setup', 'ik38mktemplate.xml')
OUT_MOT = os.path.join(_ROOT, 'Results', SESSION, TRIAL, f'ik_human_exo_{TRIAL}.mot')
HOLD_WEIGHT = 1e6                   # weight holding the human coordinates in step 2 (markers are ~1-30)
SIDES = ('R', 'L')


def replace_trc_markers(src_trc, dst_trc, new_xyz):
    """Copy src_trc to dst_trc with the data of existing markers replaced. new_xyz: {name: (n, 3) mm}."""
    with open(src_trc) as f:
        lines = f.read().splitlines()
    names = [n for n in lines[3].split('\t')[2:] if n]
    cols = {name: 2 + 3 * names.index(name) for name in new_xyz}
    data = [l for l in lines[5:] if l.strip()]
    for k, row in enumerate(data):
        values = row.split('\t')
        for name, c in cols.items():
            values[c:c + 3] = [f'{v:.5f}' for v in new_xyz[name][k]]
        data[k] = '\t'.join(values)
    with open(dst_trc, 'w', newline='\n') as f:
        f.write('\n'.join(lines[:5] + data) + '\n')


def ik_holding_coordinates(model_file, trc_file, marker_weights, held_mot, held_names, label):
    """IK frame by frame with the coordinates `held_names` pinned to their values in held_mot.
    Returns (times, coordinate names, values in degrees / m)."""
    model = osm.Model(model_file)
    state = model.initSystem()
    weights = osm.SetMarkerWeights()
    for name, w in marker_weights.items():
        weights.cloneAndAppend(osm.MarkerWeight(name, w))
    markers_ref = osm.MarkersReference(trc_file, weights)
    times = list(markers_ref.getMarkerTable().getIndependentColumn())

    held = osm.Storage(held_mot)
    if held.isInDegrees():
        model.getSimbodyEngine().convertDegreesToRadians(held)
    splines = osm.GCVSplineSet(5, held)
    coord_refs = osm.SimTKArrayCoordinateReference()
    for name in held_names:
        ref = osm.CoordinateReference(name, splines.get(name))
        ref.setWeight(HOLD_WEIGHT)
        coord_refs.push_back(ref)

    solver = osm.InverseKinematicsSolver(model, markers_ref, coord_refs, math.inf)
    solver.setAccuracy(1e-5)
    coords = model.getCoordinateSet()
    names = [coords.get(i).getName() for i in range(coords.getSize())]
    to_deg = np.array([180 / math.pi if coords.get(i).getMotionType() == 1 else 1.0 for i in range(coords.getSize())])
    bar = ProgressBar(len(times), label)
    state.setTime(times[0])
    solver.assemble(state)
    marker_names = [solver.getMarkerNameForIndex(i) for i in range(solver.getNumMarkersInUse())]
    q = np.zeros((len(times), len(names)))
    marker_err = np.zeros((len(times), len(marker_names)))
    err = osm.SimTKArrayDouble()
    for k, t in enumerate(times):
        state.setTime(t)
        solver.track(state)
        q[k] = [coords.get(i).getValue(state) for i in range(len(names))]
        solver.computeCurrentMarkerErrors(err)
        marker_err[k] = [err.getElt(j) for j in range(err.size())]
        bar.update(k + 1)
    bar.finish()
    print_ik_report(label.strip(), marker_names, 100 * marker_err)
    return np.array(times), names, q * to_deg


def read_mot(path):
    with open(path) as f:
        header_end = next(i for i, line in enumerate(f) if 'endheader' in line)
    return pd.read_csv(path, sep='\t', skiprows=header_end + 1, index_col=False)


def main():
    for f, hint in [(EST_TRC, 'hip_from_femur.py'), (HUMAN_MODEL, 'scale_model.py (no-exo static)'),
                    (EXO_MODEL, 'scale_model.py (exo static)')]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f'{f} not found - run {hint} first.')
    os.makedirs(os.path.dirname(OUT_MOT), exist_ok=True)

    with tempfile.TemporaryDirectory() as work:
        _, markers = read_trc(EST_TRC)
        human_trc = os.path.join(work, 'human.trc')
        replace_trc_markers(EST_TRC, human_trc, {f'{s}GTR': markers[f'{s}GTR_est'] for s in SIDES})
        human_mot = os.path.join(work, 'human.mot')
        run_ik(HUMAN_MODEL, human_trc, cfg.ik_marker_weights(HUMAN_IK_SETUP), human_mot,
               os.path.join(work, 'human_err.sto'), f'{TRIAL} human (GTR_est)')
        human = read_mot(human_mot)

        exo_model = osm.Model(EXO_MODEL)
        exo_coords = exo_model.getCoordinateSet()
        exo_model_names = [exo_coords.get(i).getName() for i in range(exo_coords.getSize())]
        held = [c for c in human.columns if c != 'time' and c in exo_model_names]
        times, names, q = ik_holding_coordinates(EXO_MODEL, EST_TRC, cfg.ik_marker_weights(EXO_IK_SETUP),
                                                 human_mot, held, f'IK  {TRIAL} exo (GTR on motor)')

    if len(times) != len(human) or not np.allclose(times, human.time, atol=1e-6):
        raise ValueError('Human and exo IK time columns do not match')
    result = pd.DataFrame(q, columns=names)
    drift = (result[held] - human[held].to_numpy()).abs().max()
    result[held] = human[held].to_numpy()          # write the human IK values exactly
    write_opensim_table(OUT_MOT, 'Coordinates', times, names, result[names].to_numpy(), in_degrees=True)

    print(f'\nHuman coordinates (IK with GTR_est): {", ".join(held)}')
    print(f'Exo coordinates (IK with motor GTR, human pose held): {", ".join(n for n in names if n not in held)}')
    print(f'Largest deviation of a held human coordinate in the exo IK: {drift.max():.4f} ({drift.idxmax()})')
    print(f'Written: {OUT_MOT}')


if __name__ == '__main__':
    main()
