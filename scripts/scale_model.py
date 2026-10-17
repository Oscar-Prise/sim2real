"""
Automatic model scaling with the OpenSim Python API (replaces the GUI "Scale Model" step).

One-time setup (GUI):
    1. Open OpenSim GUI -> Tools -> Scale Model.
    2. Pick your generic model + marker set, define the measurements (marker pairs),
       and the marker-placer weights exactly as you normally do.
    3. Click "Save" (NOT run) to export a setup .xml file, e.g. scale_setup_template.xml.

After that, set the paths below and run:
    python scripts/scale_model.py
Input: the unscaled model and the static .trc. Output: the scaled model (OUTPUT_MODEL), which
run_ik_id.py and hip_from_femur.py then load.
"""
import os

import numpy as np
import opensim as osm

import config as cfg
from marker_report import print_scale_report, read_scale_factors

osm.Logger.setLevelString('warn')

# ---------- Model to scale ----------
# STATIC_TRC is the standing trial used to scale the model. STATIC_TIME_WINDOW is the
# (start, end) in seconds averaged for scaling, or None to use the whole trial.
# SUBJECT_ID is looked up in 'Subject Information.txt' for mass / height / age.
# The scaled model is written to OUTPUT_MODEL; in-between files go next to it in intermediate/.
# Any path can be replaced by a full path, e.g. r'C:\Users\oscar\Documents\some_session\standing.trc'.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_TRC = os.path.join(_ROOT, 'Data', '10_16_2026 AB01', 'Static_NoExo_shoes.trc')
STATIC_TIME_WINDOW = (7.93, 8.13)
SUBJECT_ID = 'AB01'
OUTPUT_MODEL = os.path.join(_ROOT, 'Results', '10_16_2026 AB01', 'model', 'Static_NoExo_shoes_scaled.osim')

# Model and template must match the marker protocol of the session:
#   old protocol (PSIS, thigh clusters): gait2392_MovedPCIS.osim, scale_setup_template.xml
#   26-marker protocol:                  gait2392_Oz_NoPsis_26mk.osim, scale_26mk_template.xml
GENERIC_MODEL = os.path.join(_ROOT, 'unscaled model', 'gait2392_Oz_NoPsis_26mk.osim')
SCALE_SETUP_TEMPLATE = os.path.join(_ROOT, 'setup', 'scale_26mk_template.xml')


def _time_range(trc_file, t_start=None, t_end=None):
    """Return an ArrayDouble [start, end]. Defaults to the full range of the TRC file."""
    marker_data = osm.MarkerData(trc_file)
    t0 = marker_data.getStartFrameTime() if t_start is None else t_start
    t1 = marker_data.getLastFrameTime() if t_end is None else t_end
    arr = osm.ArrayDouble()
    arr.append(float(t0))
    arr.append(float(t1))
    return arr


def scale_model(setup_template_xml, generic_model_file, static_trc_file,
                subject_mass_kg, output_model_file,
                t_start=None, t_end=None, marker_set_file=None,
                subject_height_m=None, subject_age=None):
    """
    Scale `generic_model_file` to the subject using the static trial and return the
    path of the scaled model (.osim).

    setup_template_xml : Scale setup .xml exported once from the OpenSim GUI
                         (holds the measurement set and marker-placer weights).
    t_start / t_end    : time window of the static trial to use (default: whole file).
    marker_set_file    : optional marker set .xml to attach to the generic model. Default: the
                         template's <marker_set_file>, relative paths taken from the template's folder.
    """
    # Absolute paths avoid OpenSim resolving them relative to the template's folder
    setup_template_xml = os.path.abspath(setup_template_xml)
    generic_model_file = os.path.abspath(generic_model_file)
    static_trc_file = os.path.abspath(static_trc_file)
    output_model_file = os.path.abspath(output_model_file)

    # Only the final model sits next to output_model_file; in-between files go to intermediate/
    intermediate_dir = os.path.join(os.path.dirname(output_model_file), 'intermediate')
    os.makedirs(intermediate_dir, exist_ok=True)
    stage1_model_file = os.path.join(intermediate_dir, 'stage1_segments_scaled.osim')
    scale_factors_file = os.path.join(intermediate_dir, 'stage1_scale_factors.xml')
    placed_markers_file = os.path.join(intermediate_dir, 'stage2_placed_markers.xml')
    time_range = _time_range(static_trc_file, t_start, t_end)

    tool = osm.ScaleTool(setup_template_xml)
    # Otherwise OpenSim prefixes every file with the template's folder, even absolute paths
    tool.setPathToSubject('')

    # --- Subject ---
    tool.setSubjectMass(float(subject_mass_kg))
    if subject_height_m:
        tool.setSubjectHeight(1000.0 * subject_height_m)  # OpenSim stores height in mm
    if subject_age:
        tool.setSubjectAge(float(subject_age))

    # --- Generic model ---
    gmm = tool.getGenericModelMaker()
    gmm.setModelFileName(generic_model_file)
    if marker_set_file:
        marker_set_file = os.path.abspath(marker_set_file)
    elif gmm.getMarkerSetFileName() not in ('', 'Unassigned'):
        marker_set_file = os.path.join(os.path.dirname(setup_template_xml), gmm.getMarkerSetFileName())
    if marker_set_file:
        if not os.path.isfile(marker_set_file):
            raise FileNotFoundError(f"Marker set file not found: {marker_set_file}")
        gmm.setMarkerSetFileName(marker_set_file)

    # --- Step 1: scale body segments from marker-pair distances ---
    scaler = tool.getModelScaler()
    scaler.setMarkerFileName(static_trc_file)
    scaler.setTimeRange(time_range)
    scaler.setOutputModelFileName(stage1_model_file)
    scaler.setOutputScaleFileName(scale_factors_file)

    # --- Step 2: move markers to match the static pose (this is the final model) ---
    placer = tool.getMarkerPlacer()
    placer.setStaticPoseFileName(static_trc_file)
    placer.setTimeRange(time_range)
    placer.setOutputModelFileName(output_model_file)
    placer.setOutputMarkerFileName(placed_markers_file)
    placer.setOutputMotionFileName(os.path.join(intermediate_dir, 'stage2_static_pose.mot'))

    if marker_set_file:
        print(f"  Marker set {os.path.basename(marker_set_file)} replaces the generic model's markers")
    print(f"  Scale  {os.path.basename(generic_model_file)} with "
          f"{os.path.basename(static_trc_file)} ({time_range.get(0):.2f}-{time_range.get(1):.2f} s, "
          f"{subject_mass_kg} kg)...", end=' ', flush=True)
    if not tool.run():
        raise RuntimeError("ScaleTool.run() failed - check the OpenSim log above.")
    print(f"done -> {os.path.basename(output_model_file)}")

    names, static_err, moved = _static_check(tool, stage1_model_file, output_model_file,
                                             static_trc_file, time_range.get(0), time_range.get(1))
    print_scale_report(names, static_err, moved, read_scale_factors(scale_factors_file))
    return output_model_file


def _static_check(tool, stage1_model_file, final_model_file, static_trc_file, t_start, t_end):
    """Per-marker fit of the scaled model to the static window before markers are moved (cm,
    same marker weights as the marker placer) and how far the placer then moved each marker (cm)."""
    tasks = tool.getMarkerPlacer().getIKTaskSet()
    weights_by_name = {}
    for i in range(tasks.getSize()):
        task = osm.IKMarkerTask.safeDownCast(tasks.get(i))
        if task is not None:
            weights_by_name[task.getName()] = task.getWeight() if task.getApply() else 0.0

    model = osm.Model(stage1_model_file)
    state = model.initSystem()
    marker_set = model.getMarkerSet()
    weights = osm.SetMarkerWeights()
    for i in range(marker_set.getSize()):
        name = marker_set.get(i).getName()
        weights.cloneAndAppend(osm.MarkerWeight(name, weights_by_name.get(name, 0.0)))
    reference = osm.MarkersReference(static_trc_file, weights)
    table = reference.getMarkerTable()
    times = np.array(table.getIndependentColumn())
    frames = np.flatnonzero((times >= t_start - 1e-9) & (times <= t_end + 1e-9))
    names = [marker_set.get(i).getName() for i in range(marker_set.getSize())
             if marker_set.get(i).getName() in table.getColumnLabels()]
    columns = [table.getColumnIndex(n) for n in names]

    solver = osm.InverseKinematicsSolver(model, reference, osm.SimTKArrayCoordinateReference(), float('inf'))
    solver.setAccuracy(1e-5)
    err = np.zeros((len(frames), len(names)))
    for k, frame in enumerate(frames):
        state.setTime(times[frame])
        solver.assemble(state) if k == 0 else solver.track(state)
        model.realizePosition(state)
        row = table.getRowAtIndex(int(frame))
        for j, (name, col) in enumerate(zip(names, columns)):
            measured = row.getElt(0, col).to_numpy()
            err[k, j] = np.linalg.norm(marker_set.get(name).getLocationInGround(state).to_numpy() - measured)

    final_model = osm.Model(final_model_file)
    final_markers = final_model.getMarkerSet()
    moved = [np.linalg.norm(final_markers.get(n).get_location().to_numpy()
                            - marker_set.get(n).get_location().to_numpy()) for n in names]
    return names, 100 * np.nanmean(err, axis=0), 100 * np.array(moved)


if __name__ == '__main__':
    subject = cfg.subject_info(SUBJECT_ID)
    t_start, t_end = STATIC_TIME_WINDOW or (None, None)
    scale_model(
        setup_template_xml=SCALE_SETUP_TEMPLATE,
        generic_model_file=GENERIC_MODEL,
        static_trc_file=STATIC_TRC,
        subject_mass_kg=subject.mass_kg,
        output_model_file=OUTPUT_MODEL,
        t_start=t_start,
        t_end=t_end,
        subject_height_m=subject.height_m,
        subject_age=subject.age,
    )
