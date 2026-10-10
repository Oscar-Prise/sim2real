"""
Scale the model, then run Inverse Kinematics and Inverse Dynamics with the OpenSim Python API
for every trial in a session (static / standing trials are skipped).

    python scripts/run_ik_id.py                                # session set in config.py
    python scripts/run_ik_id.py "09_21_2026 AB02"              # another session
    python scripts/run_ik_id.py --all                          # every session in Data/
    python scripts/run_ik_id.py "10_08_2026 AB01" --trials lg_1p2_walking
"""
import argparse
import math
import os
import time

import numpy as np
import opensim as osm

import config as cfg
from progress import ProgressBar
from scale_model import scale_model

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

    q_out = np.zeros((len(times), len(names)))
    err_out = np.zeros((len(times), 3))
    marker_err = osm.SimTKArrayDouble()

    bar = ProgressBar(len(times), f'IK  {label}')
    state.setTime(times[0])
    solver.assemble(state)
    for k, t in enumerate(times):
        state.setTime(t)
        solver.track(state)
        q_out[k] = [coords.get(i).getValue(state) for i in range(len(names))]
        solver.computeCurrentMarkerErrors(marker_err)
        e = np.array([marker_err.getElt(j) for j in range(marker_err.size())])
        err_out[k] = [np.sum(e ** 2), np.sqrt(np.mean(e ** 2)), e.max()]
        bar.update(k + 1)

    write_opensim_table(ik_file, 'Coordinates', times, names, q_out * to_deg, in_degrees=True)
    write_opensim_table(errors_file, 'Model Marker Errors from IK', times,
                        ['total_squared_error', 'marker_error_RMS', 'marker_error_max'], err_out, False)
    bar.finish(f'marker RMS {100 * err_out[:, 1].mean():.1f} cm (max {100 * err_out[:, 2].max():.1f} cm)')


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


def process_session(session, trials=None):
    subject = cfg.subject_info(session)
    paths = cfg.session_paths(session)
    trials = trials or cfg.list_trials(session)

    print(f"\n=== {session}  |  {subject.name} ({subject.id}), {subject.mass_kg} kg, {subject.height_m} m ===")
    print(f"Trials: {', '.join(trials)}")

    t0 = time.time()
    scale_model(
        setup_template_xml=cfg.SCALE_SETUP_TEMPLATE,
        generic_model_file=cfg.GENERIC_MODEL,
        static_trc_file=paths.static_trc,
        subject_mass_kg=subject.mass_kg,
        output_model_file=paths.scaled_model,
        t_start=paths.static_t_start,
        t_end=paths.static_t_end,
        subject_height_m=subject.height_m,
        subject_age=subject.age,
    )

    for trial in trials:
        tp = cfg.trial_paths(session, trial)
        os.makedirs(tp.out_dir, exist_ok=True)
        run_ik(paths.scaled_model, tp.trc, cfg.IK_MARKER_WEIGHTS, tp.ik, tp.ik_errors, trial)
        if not os.path.isfile(tp.mot):
            print(f"  ID  {trial}: no force file ({os.path.basename(tp.mot)}), skipped")
            continue
        write_external_loads_xml(tp.loads_xml, tp.mot, trial)
        run_id(paths.scaled_model, tp.ik, tp.loads_xml, tp.id, cfg.ID_LOWPASS_HZ, trial)

    print(f"Done in {time.time() - t0:.0f}s. Results in {paths.results_dir}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Scale + IK + ID for every trial in a session.")
    p.add_argument('sessions', nargs='*', help=f"Session folder(s) in Data/ (default: {cfg.SESSION})")
    p.add_argument('--all', action='store_true', help='Process every session in Data/')
    p.add_argument('--trials', nargs='+', help='Only these trials (default: all except standing)')
    a = p.parse_args()

    sessions = cfg.list_sessions() if a.all else (a.sessions or [cfg.SESSION])
    for s in sessions:
        process_session(s, a.trials)
