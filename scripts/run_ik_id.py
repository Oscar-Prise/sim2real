"""
Inverse Kinematics and Inverse Dynamics with the OpenSim Python API, using a model already
scaled by scale_model.py. Set the paths below, then run:

    python scripts/run_ik_id.py
"""
import math
import os

import numpy as np
import opensim as osm

import config as cfg
from util_marker_report import print_ik_report
from util_progress_bar import ProgressBar

# ---------- Data to process ----------
# SCALED_MODEL is the output of scale_model.py.
# DATA_DIR holds the trials: every <trial>.trc gets IK, and <trial>.mot (force plates) gets ID.
# Results go to RESULTS_DIR/<trial>/.
# Any path can be replaced by a full path, e.g. r'C:\Users\oscar\Documents\some_session'.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SCALED_MODEL = os.path.join(_ROOT, 'Results', '10_16_2026 AB01', 'model', 'Static_Exo_scaled.osim')
DATA_DIR = os.path.join(_ROOT, 'Data', '10_16_2026 AB01')
RESULTS_DIR = os.path.join(_ROOT, 'Results', '10_16_2026 AB01')
TRIALS = ['LG_1p2_ExoOn', 'Static_Exo']   # None = every .trc in DATA_DIR
PROCESS_STATIC_TRIALS = True    # with TRIALS = None: also trials with 'standing' or 'static' in the name

# IK marker weights must match the marker protocol of the session:
#   26-marker protocol:                  cfg.ik_marker_weights(<IK setup .xml saved in the OpenSim GUI>)
#   old protocol (PSIS, thigh clusters): cfg.IK_MARKER_WEIGHTS
IK_SETUP = os.path.join(_ROOT, 'setup', 'ik38mktemplate.xml')
IK_MARKER_WEIGHTS = cfg.ik_marker_weights(IK_SETUP)

osm.Logger.setLevelString('warn')
osm.Logger.removeFileSink()
if os.path.isdir(cfg.GEOMETRY_DIR):
    osm.ModelVisualizer.addDirToGeometrySearchPaths(cfg.GEOMETRY_DIR)


def write_external_loads_xml(xml_output_path, datafile_path, data_source_name):
    """Write an ExternalLoads XML file following the forces.xml structure."""
    xml_text = f"""<?xml version=\"1.0\" encoding=\"UTF-8\" ?>
<OpenSimDocument Version=\"40500\">
	<ExternalLoads name=\"externalloads\">
		<objects>
			<ExternalForce name=\"externalforce_r\">
				<applied_to_body>calcn_r</applied_to_body>
				<force_expressed_in_body>ground</force_expressed_in_body>
				<point_expressed_in_body>ground</point_expressed_in_body>
				<force_identifier>ground_force1_v</force_identifier>
				<point_identifier>ground_force1_p</point_identifier>
				<torque_identifier>ground_torque1_</torque_identifier>
				<data_source_name>{data_source_name}</data_source_name>
			</ExternalForce>
			<ExternalForce name=\"externalforce_l\">
				<applied_to_body>calcn_l</applied_to_body>
				<force_expressed_in_body>ground</force_expressed_in_body>
				<point_expressed_in_body>ground</point_expressed_in_body>
				<force_identifier>ground_force2_v</force_identifier>
				<point_identifier>ground_force2_p</point_identifier>
				<torque_identifier>ground_torque2_</torque_identifier>
				<data_source_name>{data_source_name}</data_source_name>
			</ExternalForce>
		</objects>
		<groups />
		<datafile>{datafile_path}</datafile>
	</ExternalLoads>
</OpenSimDocument>
"""

    with open(xml_output_path, "w", encoding="utf-8") as f:
        f.write(xml_text)


def write_opensim_table(path, title, times, labels, data, in_degrees):
    """Write a .mot / .sto file in the same format the OpenSim tools use."""
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"{title}\nversion=1\nnRows={len(times)}\nnColumns={len(labels) + 1}\n"
                f"inDegrees={'yes' if in_degrees else 'no'}\nendheader\n")
        f.write('time\t' + '\t'.join(labels) + '\n')
        for t, row in zip(times, data):
            f.write(f'{t:.8f}\t' + '\t'.join(f'{v:.8f}' for v in row) + '\n')


def run_ik(model_file, trc_file, marker_weights, ik_file, errors_file, label):
    """Same as InverseKinematicsTool (accuracy 1e-5, hard constraints) but frame by frame."""
    model = osm.Model(model_file)
    state = model.initSystem()

    weights = osm.SetMarkerWeights()
    for name, w in marker_weights.items():
        weights.cloneAndAppend(osm.MarkerWeight(name, w))
    markers_ref = osm.MarkersReference(trc_file, weights)
    times = list(markers_ref.getMarkerTable().getIndependentColumn())

    solver = osm.InverseKinematicsSolver(model, markers_ref, osm.SimTKArrayCoordinateReference(), math.inf)
    solver.setAccuracy(1e-5)

    coords = model.getCoordinateSet()
    names = [coords.get(i).getName() for i in range(coords.getSize())]
    to_deg = np.array([180 / math.pi if coords.get(i).getMotionType() == 1 else 1.0
                       for i in range(coords.getSize())])

    bar = ProgressBar(len(times), f'IK  {label}')
    state.setTime(times[0])
    solver.assemble(state)
    # Only valid after assemble(); querying the solver before that crashes the process.
    marker_names = [solver.getMarkerNameForIndex(i) for i in range(solver.getNumMarkersInUse())]
    q_out = np.zeros((len(times), len(names)))
    per_marker = np.zeros((len(times), len(marker_names)))
    marker_err = osm.SimTKArrayDouble()
    for k, t in enumerate(times):
        state.setTime(t)
        solver.track(state)
        q_out[k] = [coords.get(i).getValue(state) for i in range(len(names))]
        solver.computeCurrentMarkerErrors(marker_err)
        per_marker[k] = [marker_err.getElt(j) for j in range(marker_err.size())]
        bar.update(k + 1)

    err_out = np.column_stack([np.sum(per_marker ** 2, axis=1), np.sqrt(np.mean(per_marker ** 2, axis=1)),
                               per_marker.max(axis=1)])
    write_opensim_table(ik_file, 'Coordinates', times, names, q_out * to_deg, in_degrees=True)
    write_opensim_table(errors_file, 'Model Marker Errors from IK', times,
                        ['total_squared_error', 'marker_error_RMS', 'marker_error_max'], err_out, False)
    bar.finish(f'marker RMS {100 * err_out[:, 1].mean():.1f} cm (max {100 * err_out[:, 2].max():.1f} cm)')
    print_ik_report(label, marker_names, 100 * per_marker)


def run_id(model_file, ik_file, loads_xml, id_file, lowpass_hz, label):
    """Same as InverseDynamicsTool (muscles excluded, coordinates low-pass filtered) but frame by frame."""
    model = osm.Model(model_file)
    external_loads = osm.ExternalLoads(loads_xml, True)
    model.addModelComponent(external_loads)
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscles.get(i).set_appliesForce(False)
    state = model.initSystem()

    # The solver expects coordinates in mobility (U) order; recover it from the state
    coord_set = model.getCoordinateSet()
    for i in range(coord_set.getSize()):
        coord_set.get(i).setSpeedValue(state, i + 1.0)
    coords = [coord_set.get(int(round(u)) - 1) for u in state.getU().to_numpy()]

    q = osm.Storage(ik_file)
    time_col = osm.ArrayDouble()
    q.getTimeColumn(time_col)
    times = [time_col.get(i) for i in range(time_col.getSize())]
    q.pad(q.getSize() // 2)
    q.lowpassIIR(lowpass_hz)
    if q.isInDegrees():
        model.getSimbodyEngine().convertDegreesToRadians(q)
    splines = osm.GCVSplineSet(5, q)
    functions = osm.FunctionSet()
    for c in coords:
        name = c.getName()
        functions.cloneAndAppend(splines.get(name) if splines.contains(name) else osm.Constant(c.getDefaultValue()))

    solver = osm.InverseDynamicsSolver(model)
    labels = [c.getName() + ('_moment' if c.getMotionType() == 1 else '_force') for c in coords]
    tau = np.zeros((len(times), len(coords)))

    bar = ProgressBar(len(times), f'ID  {label}')
    for k, t in enumerate(times):
        tau[k] = solver.solve(state, functions, t).to_numpy()
        bar.update(k + 1)

    write_opensim_table(id_file, 'Inverse Dynamics Generalized Forces', times, labels, tau, in_degrees=False)
    bar.finish()


def list_trials(data_dir, include_static):
    """All trials with a .trc file in data_dir; static (standing) trials only if include_static."""
    trials = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.lower().endswith('.trc')]
    is_static = lambda t: 'standing' in t.lower() or 'static' in t.lower()
    return sorted(t for t in trials if include_static or not is_static(t))


def main():
    if not os.path.isfile(SCALED_MODEL):
        raise FileNotFoundError(f"Scaled model not found: {SCALED_MODEL}\nRun scale_model.py first.")
    trials = TRIALS or list_trials(DATA_DIR, PROCESS_STATIC_TRIALS)
    print(f"\n=== {DATA_DIR}\n    model: {os.path.basename(SCALED_MODEL)}\n    trials: {', '.join(trials)}")

    for trial in trials:
        out_dir = os.path.join(RESULTS_DIR, trial)
        os.makedirs(out_dir, exist_ok=True)
        trc_file = os.path.join(DATA_DIR, f'{trial}.trc')
        force_mot = os.path.join(DATA_DIR, f'{trial}.mot')
        ik_file = os.path.join(out_dir, f'ik_{trial}.mot')
        loads_xml = os.path.join(out_dir, f'forces_{trial}.xml')

        run_ik(SCALED_MODEL, trc_file, IK_MARKER_WEIGHTS, ik_file,
               os.path.join(out_dir, f'ik_marker_errors_{trial}.sto'), trial)
        if not os.path.isfile(force_mot):
            print(f"  ID  {trial}: no force file ({os.path.basename(force_mot)}), skipped")
            continue
        write_external_loads_xml(loads_xml, force_mot, trial)
        run_id(SCALED_MODEL, ik_file, loads_xml, os.path.join(out_dir, f'id_{trial}.sto'),
               cfg.ID_LOWPASS_HZ, trial)

    print(f"Results in {RESULTS_DIR}")


if __name__ == '__main__':
    main()
