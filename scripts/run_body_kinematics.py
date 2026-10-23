"""
OpenSim BodyKinematics analysis (Analyze tool) of every trial, using the IK results of run_ik_id.py.
Gives position / orientation, velocity and acceleration of every body in the ground frame, which
util_BodyKinematicsPlot.py turns into segment-to-segment angles (e.g. femur vs pelvis, femur vs torso).

Each condition (e.g. no exo / exo on) has its own scaled model and static trial. The static trial
is run too (it is the zero reference of the plots); trials without an IK result yet (usually the
static) get IK first with the condition's IK setup. Edit the settings below, then run:

    python scripts/run_body_kinematics.py

Writes Results/<session>/<trial>/<trial>_BodyKinematics_{pos,vel,acc}_global.sto.
"""
import os
import tempfile
import xml.etree.ElementTree as ET

import opensim as osm

import config as cfg
from run_ik_id import run_ik

# ---------- Settings ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = '10_18_2026 AB01'
# condition: scaled model (Results/<session>/model/), static trial, IK setup (only used for trials
# without an IK result yet), walking trials. Any number of conditions and trials.
CONDITIONS = {
    'NoExo': dict(model='NoExo_Static_scaled.osim', static='NoExo_Static', ik_setup='iktemplate.xml',
                  trials=['NoExo_LG_1p2']),
    'ExoOn': dict(model='ExoOn_Static_scaled.osim', static='ExoOn_Static', ik_setup='ik40mk_PSIS1_ASIS30.xml',
                  trials=['ExoOn_LG_1p2_loose', 'ExoOn_LG_1p2_moderate', 'ExoOn_LG_1p2_tight']),
}
ANALYZE_TEMPLATE = os.path.join(_ROOT, 'setup', 'bodyKinematics_Analyze.xml')
LOWPASS_HZ = cfg.ID_LOWPASS_HZ   # on the IK coordinates before differentiating; None = no filter

DATA_DIR = os.path.join(_ROOT, 'Data', SESSION)
RESULTS_DIR = os.path.join(_ROOT, 'Results', SESSION)

osm.Logger.setLevelString('warn')
if os.path.isdir(cfg.GEOMETRY_DIR):
    osm.ModelVisualizer.addDirToGeometrySearchPaths(cfg.GEOMETRY_DIR)


def ik_time_range(ik_file):
    times = osm.TimeSeriesTable(ik_file).getIndependentColumn()
    return times[0], times[-1]


def run_body_kinematics(model_file, ik_file, out_dir, name):
    """Analyze tool with the BodyKinematics template on one IK result."""
    t0, t1 = ik_time_range(ik_file)
    root = ET.parse(ANALYZE_TEMPLATE).getroot()
    tool = root.find('AnalyzeTool')
    tool.set('name', name)
    values = {'model_file': model_file, 'results_directory': out_dir, 'coordinates_file': ik_file,
              'initial_time': t0, 'final_time': t1,
              'lowpass_cutoff_frequency_for_coordinates': LOWPASS_HZ if LOWPASS_HZ else -1}
    for tag, value in values.items():
        tool.find(tag).text = str(value)
    for analysis in tool.iter('BodyKinematics'):
        analysis.find('start_time').text = str(t0)
        analysis.find('end_time').text = str(t1)

    # The tool resolves relative paths from the setup file's folder; all paths above are absolute
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        setup = os.path.join(tmp, f'analyze_{name}.xml')
        ET.ElementTree(root).write(setup, encoding='UTF-8', xml_declaration=True)
        print(f"  BodyKinematics  {name} ({t0:.2f}-{t1:.2f} s)...", end=' ', flush=True)
        if not osm.AnalyzeTool(setup).run():
            raise RuntimeError(f"AnalyzeTool failed for {name} - check the OpenSim log")
    print('done')


def main():
    failed = []
    for condition, c in CONDITIONS.items():
        model_file = os.path.join(RESULTS_DIR, 'model', c['model'])
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Scaled model for {condition} not found: {model_file}\nRun run_ik_id.py first.")
        weights = cfg.ik_marker_weights(os.path.join(_ROOT, 'setup', c['ik_setup']))
        print(f"\n=== {condition}: model {c['model']}")
        for trial in [c['static']] + c['trials']:
            out_dir = os.path.join(RESULTS_DIR, trial)
            ik_file = os.path.join(out_dir, f'ik_{trial}.mot')
            try:
                if not os.path.isfile(ik_file):
                    os.makedirs(out_dir, exist_ok=True)
                    print(f"  no IK result for {trial} yet, running IK with {c['ik_setup']}")
                    run_ik(model_file, os.path.join(DATA_DIR, f'{trial}.trc'), weights, ik_file,
                           os.path.join(out_dir, f'ik_marker_errors_{trial}.sto'), trial)
                run_body_kinematics(model_file, ik_file, out_dir, trial)
            except Exception as e:
                print(f"  FAILED {trial}: {e}")
                failed.append(trial)
    print(f"\nResults in {RESULTS_DIR}" + (f"\nFailed: {', '.join(failed)}" if failed else ''))


if __name__ == '__main__':
    main()
