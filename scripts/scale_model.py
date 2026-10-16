"""
Automatic model scaling with the OpenSim Python API (replaces the GUI "Scale Model" step).

One-time setup (GUI):
    1. Open OpenSim GUI -> Tools -> Scale Model.
    2. Pick your generic model + marker set, define the measurements (marker pairs),
       and the marker-placer weights exactly as you normally do.
    3. Click "Save" (NOT run) to export a setup .xml file, e.g. scale_setup_template.xml.

After that, set the paths below and run:
    python scripts/scale_model.py
run_ik_id.py calls scale_model() itself, using the paths at the top of that script.
"""
import os

import opensim as osm

import config as cfg

# ---------- Model to scale ----------
# STATIC_TRC is the standing trial used to scale the model. STATIC_TIME_WINDOW is the
# (start, end) in seconds averaged for scaling, or None to use the whole trial.
# SUBJECT_ID is looked up in 'Subject Information.txt' for mass / height / age.
# The scaled model is written to OUTPUT_MODEL; in-between files go next to it in intermediate/.
# Any path can be replaced by a full path, e.g. r'C:\Users\oscar\Documents\some_session\standing.trc'.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STATIC_TRC = os.path.join(_ROOT, 'Data', '10_08_2026 AB01', 'standing.trc')
STATIC_TIME_WINDOW = (7.0, 7.2)
SUBJECT_ID = 'AB01'
OUTPUT_MODEL = os.path.join(_ROOT, 'Results', '10_08_2026 AB01', 'model', 'standing_scaled.osim')

GENERIC_MODEL = os.path.join(_ROOT, 'unscaled model', 'gait2392_MovedPCIS.osim')
SCALE_SETUP_TEMPLATE = os.path.join(_ROOT, 'setup', 'scale_setup_template.xml')


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
    marker_set_file    : optional marker set .xml to attach to the generic model.
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
        gmm.setMarkerSetFileName(os.path.abspath(marker_set_file))

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

    print(f"  Scale  {os.path.basename(generic_model_file)} with "
          f"{os.path.basename(static_trc_file)} ({time_range.get(0):.2f}-{time_range.get(1):.2f} s, "
          f"{subject_mass_kg} kg)...", end=' ', flush=True)
    if not tool.run():
        raise RuntimeError("ScaleTool.run() failed - check the OpenSim log above.")
    print(f"done -> {os.path.basename(output_model_file)}")
    return output_model_file


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
