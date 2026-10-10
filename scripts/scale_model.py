"""
Automatic model scaling with the OpenSim Python API (replaces the GUI "Scale Model" step).

One-time setup (GUI):
    1. Open OpenSim GUI -> Tools -> Scale Model.
    2. Pick your generic model + marker set, define the measurements (marker pairs),
       and the marker-placer weights exactly as you normally do.
    3. Click "Save" (NOT run) to export a setup .xml file, e.g. scale_setup_template.xml.

After that, every subject / session is just:
    scale_model(
        setup_template_xml = 'scale_setup_template.xml',
        generic_model_file = 'path/to/generic.osim',
        static_trc_file    = 'path/to/standing.trc',
        subject_mass_kg    = 70.0,
        output_model_file  = 'path/to/subject_scaled.osim',
    )
or from the command line:
    python scale_model.py --template scale_setup_template.xml --generic generic.osim ^
        --static standing.trc --mass 70 --out subject_scaled.osim
"""
import argparse
import os

import opensim as osm


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

    out_dir = os.path.dirname(output_model_file)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(output_model_file)[0]
    scale_factors_file = base + '_scaleSet.xml'
    placed_markers_file = base + '_markers.xml'
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
    scaler.setOutputModelFileName(base + '_scaled_only.osim')
    scaler.setOutputScaleFileName(scale_factors_file)

    # --- Step 2: move markers to match the static pose (this is the final model) ---
    placer = tool.getMarkerPlacer()
    placer.setStaticPoseFileName(static_trc_file)
    placer.setTimeRange(time_range)
    placer.setOutputModelFileName(output_model_file)
    placer.setOutputMarkerFileName(placed_markers_file)

    print(f"  Scale  {os.path.basename(generic_model_file)} with "
          f"{os.path.basename(static_trc_file)} ({time_range.get(0):.2f}-{time_range.get(1):.2f} s, "
          f"{subject_mass_kg} kg)...", end=' ', flush=True)
    if not tool.run():
        raise RuntimeError("ScaleTool.run() failed - check the OpenSim log above.")
    print(f"done -> {os.path.basename(output_model_file)}")
    return output_model_file


if __name__ == '__main__':
    p = argparse.ArgumentParser(description="Scale an OpenSim model from a static trial.")
    p.add_argument('--template', required=True, help='Scale setup .xml exported from the GUI')
    p.add_argument('--generic', required=True, help='Generic (unscaled) .osim model')
    p.add_argument('--static', required=True, help='Static trial .trc')
    p.add_argument('--mass', required=True, type=float, help='Subject mass (kg)')
    p.add_argument('--out', required=True, help='Output scaled .osim')
    p.add_argument('--t0', type=float, default=None, help='Static window start (s)')
    p.add_argument('--t1', type=float, default=None, help='Static window end (s)')
    a = p.parse_args()
    scale_model(a.template, a.generic, a.static, a.mass, a.out, a.t0, a.t1)
