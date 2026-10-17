"""Processing settings for run_ik_id.py. Data paths are set at the top of run_ik_id.py,
plot files at the top of plot_ik_id.py.

Subject mass / height / age are read from 'Subject Information.txt' by subject id (e.g. AB01).
"""
import os
import re
import xml.etree.ElementTree as ET
from types import SimpleNamespace

ID_LOWPASS_HZ = 6.0

# IK marker weights for the old protocol (PSIS, thigh clusters). The 26-marker protocol takes its
# weights from an IK setup file saved in the OpenSim GUI (setup/iktemplate.xml).
IK_MARKER_WEIGHTS = {
    "LSHO": 1.0, "C7": 1.0, "RSHO": 1.0, "JN": 1.0,
    "LPSIS": 20.0, "RPSIS": 20.0, "RASIS": 20.0, "LASIS": 20.0,
    "RGTR": 20.0, "RLKNE": 20.0, "RMKNE": 20.0, "LGTR": 20.0,
    "LMKNE": 20.0, "LLKNE": 20.0, "RHEEL": 1.0, "RLANK": 20.0,
    "RMANK": 20.0, "RMT1": 1.0, "RMT5": 1.0, "LMANK": 20.0,
    "LHEEL": 1.0, "LLANK": 1.0, "LMT1": 1.0, "LMT5": 1.0,
}


def ik_marker_weights(ik_setup_xml):
    """{marker: weight} from the IKMarkerTasks of an IK setup file saved in the OpenSim GUI.
    Tasks that are not applied get weight 0."""
    weights = {}
    for task in ET.parse(ik_setup_xml).getroot().iter('IKMarkerTask'):
        applied = task.findtext('apply', 'true').strip().lower() == 'true'
        weights[task.get('name')] = float(task.findtext('weight', '1')) if applied else 0.0
    return weights

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUBJECT_INFO_FILE = os.path.join(ROOT, 'Subject Information.txt')
GEOMETRY_DIR = r'C:\OpenSim 4.5\Geometry'


def subject_info(subject_id):
    """Look up a subject (e.g. 'AB01' or 'AB 01') in Subject Information.txt."""
    subject_id = subject_id.replace(' ', '').upper()
    subjects, current = {}, None
    with open(SUBJECT_INFO_FILE, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            header = re.match(r'^([A-Za-z]+)\s*(\d+)\s+(.+)$', line)
            if header and ':' not in line:
                current = f'{header.group(1).upper()}{header.group(2)}'
                subjects[current] = {'id': current, 'name': header.group(3).strip()}
            elif current and ':' in line:
                key, value = (s.strip() for s in line.split(':', 1))
                number = re.search(r'[-+]?\d*\.?\d+', value)
                subjects[current][key.lower()] = float(number.group()) if number and key.lower() != 'gender' else value

    if subject_id not in subjects:
        raise KeyError(f"Subject '{subject_id}' not found in {SUBJECT_INFO_FILE}")
    info = subjects[subject_id]
    return SimpleNamespace(id=info['id'], name=info['name'], mass_kg=info.get('weight'),
                           height_m=info.get('height'), age=info.get('age'), gender=info.get('gender'))
