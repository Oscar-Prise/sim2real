"""Settings shared by run_ik_id.py and plot_ik_id.py.

Data layout expected:
    Data/<MM_DD_YYYY SUBJECT>/standing.trc         static trial (used for scaling)
    Data/<MM_DD_YYYY SUBJECT>/<trial>.trc/.mot/.csv every other trial gets IK + ID
Subject mass / height / age are read from 'Subject Information.txt' using the SUBJECT id
at the end of the session folder name (e.g. '10_08_2026 AB01' -> AB01).
"""
import os
import re
from types import SimpleNamespace

# ---------- Defaults ----------
SESSION = '10_08_2026 AB01'     # session processed by default / plotted
TRIAL = 'lg_1p2_walking'        # trial plotted by plot_ik_id.py
STATIC_TRIAL = 'standing'       # trials containing this word are never run through IK / ID

# Time window of the static trial used for scaling, per session (default: whole trial)
STATIC_TIME_WINDOW = {
    '10_08_2026 AB01': (7.0, 7.2),
}

ID_LOWPASS_HZ = 6.0

IK_MARKER_WEIGHTS = {
    "LSHO": 1.0, "C7": 1.0, "RSHO": 1.0, "JN": 1.0,
    "LPSIS": 20.0, "RPSIS": 20.0, "RASIS": 20.0, "LASIS": 20.0,
    "RGTR": 20.0, "RLKNE": 20.0, "RMKNE": 20.0, "LGTR": 20.0,
    "LMKNE": 20.0, "LLKNE": 20.0, "RHEEL": 1.0, "RLANK": 20.0,
    "RMANK": 20.0, "RMT1": 1.0, "RMT5": 1.0, "LMANK": 20.0,
    "LHEEL": 1.0, "LLANK": 1.0, "LMT1": 1.0, "LMT5": 1.0,
}

# ---------- Folder layout ----------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.path.join(ROOT, 'Data')
RESULTS_ROOT = os.path.join(ROOT, 'Results')
GENERIC_MODEL = os.path.join(ROOT, 'unscaled model', 'gait2392.osim')
SCALE_SETUP_TEMPLATE = os.path.join(ROOT, 'setup', 'scale_setup_template.xml')
SUBJECT_INFO_FILE = os.path.join(ROOT, 'Subject Information.txt')
GEOMETRY_DIR = r'C:\OpenSim 4.5\Geometry'


def list_sessions():
    return sorted(d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d)))


def list_trials(session):
    """All trials with a .trc file in the session folder, except static (standing) trials."""
    data_dir = os.path.join(DATA_ROOT, session)
    trials = [os.path.splitext(f)[0] for f in os.listdir(data_dir) if f.lower().endswith('.trc')]
    return sorted(t for t in trials if STATIC_TRIAL not in t.lower())


def subject_info(session):
    """Look up the subject (last word of the session name, e.g. AB01) in Subject Information.txt."""
    subject_id = session.split()[-1].replace(' ', '').upper()
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
        raise KeyError(f"Subject '{subject_id}' (from session '{session}') not found in {SUBJECT_INFO_FILE}")
    info = subjects[subject_id]
    return SimpleNamespace(id=info['id'], name=info['name'], mass_kg=info.get('weight'),
                           height_m=info.get('height'), age=info.get('age'), gender=info.get('gender'))


def session_paths(session):
    data_dir = os.path.join(DATA_ROOT, session)
    results_dir = os.path.join(RESULTS_ROOT, session)
    t_start, t_end = STATIC_TIME_WINDOW.get(session, (None, None))
    return SimpleNamespace(
        data_dir=data_dir,
        results_dir=results_dir,
        static_trc=os.path.join(data_dir, f'{STATIC_TRIAL}.trc'),
        scaled_model=os.path.join(results_dir, 'model', f'{STATIC_TRIAL}_scaled.osim'),
        static_t_start=t_start,
        static_t_end=t_end,
    )


def trial_paths(session, trial):
    data_dir = os.path.join(DATA_ROOT, session)
    out_dir = os.path.join(RESULTS_ROOT, session, trial)
    return SimpleNamespace(
        trc=os.path.join(data_dir, f'{trial}.trc'),
        mot=os.path.join(data_dir, f'{trial}.mot'),
        csv=os.path.join(data_dir, f'{trial}.csv'),
        out_dir=out_dir,
        ik=os.path.join(out_dir, f'ik_{trial}.mot'),
        ik_errors=os.path.join(out_dir, f'ik_marker_errors_{trial}.sto'),
        id=os.path.join(out_dir, f'id_{trial}.sto'),
        loads_xml=os.path.join(out_dir, f'forces_{trial}.xml'),
    )
