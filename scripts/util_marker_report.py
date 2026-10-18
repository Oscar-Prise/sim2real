"""Terminal tables that summarise marker errors by body region, with a quick usable / check verdict.

IK thresholds follow the OpenSim best-practice guide: marker errors under ~2 cm are good, up to
~4 cm acceptable. Region status uses the 95th percentile so one bad frame does not flag a region.
The static check is looser: generic marker positions are often a few cm off the subject's
(e.g. GTR ~4 cm), while a mislabelled marker shows up as a 20+ cm move.
"""
import re

import numpy as np

GOOD_CM, LIMIT_CM = 2.0, 4.0
STATIC_FIT_GOOD_CM, STATIC_FIT_LIMIT_CM = 2.0, 5.0
MOVED_GOOD_CM, MOVED_LIMIT_CM = 5.0, 10.0

REGION_ORDER = ('torso', 'pelvis', 'hip', 'thigh', 'knee', 'shank', 'ankle', 'foot', 'exoP', 'exoT', 'other')
_REGION_OF_SUFFIX = {
    'C7': 'torso', 'JN': 'torso', 'S': 'torso', 'SHO': 'torso',
    'ASIS': 'pelvis', 'PSIS': 'pelvis',
    'GTR': 'hip',
    'CLUST1': 'thigh', 'CLUST2': 'thigh', 'CLUST3': 'thigh',
    'LE': 'knee', 'ME': 'knee', 'LKNE': 'knee', 'MKNE': 'knee',
    'HF': 'shank', 'TT': 'shank',
    'LA': 'ankle', 'MA': 'ankle', 'LANK': 'ankle', 'MANK': 'ankle',
    'HEEL': 'foot', 'MT1': 'foot', 'MT5': 'foot',
    'MP': 'exoP', 'PP': 'exoP', 'DP': 'exoP',               # exo pelvis plate
    'PT': 'exoT', 'DTF': 'exoT', 'DTB': 'exoT',             # exo thigh segments
}


def region_of(marker):
    if marker in _REGION_OF_SUFFIX:
        return _REGION_OF_SUFFIX[marker]
    if marker[:1] in 'RL' and marker[1:] in _REGION_OF_SUFFIX:
        return _REGION_OF_SUFFIX[marker[1:]]
    return 'other'


def _status(value, good, limit):
    return 'good' if value <= good else 'ok' if value <= limit else 'HIGH'


def _marker_types(names, idx):
    """'LE ME' instead of 'RLE RME LLE LME'."""
    types = []
    for j in idx:
        n = names[j]
        t = n[1:] if n[:1] in 'RL' and n[1:] in _REGION_OF_SUFFIX else n
        if t not in types:
            types.append(t)
    return ' '.join(types)


def _by_region(names):
    groups = {}
    for j, n in enumerate(names):
        groups.setdefault(region_of(n), []).append(j)
    return [(r, groups[r]) for r in REGION_ORDER if r in groups]


def print_ik_report(label, names, err_cm):
    """err_cm: (n_frames, n_markers) marker errors in cm for one IK trial."""
    err_cm = np.asarray(err_cm)
    frame_rms = np.sqrt(np.mean(err_cm ** 2, axis=1))
    p95 = np.percentile(err_cm, 95, axis=0)
    worst_share = np.bincount(err_cm.argmax(axis=1), minlength=len(names)) / len(err_cm)

    print(f'\n  Marker check: {label}  (status from 95th percentile: good <= {GOOD_CM:g} cm, ok <= {LIMIT_CM:g} cm)')
    print(f'    {"region":8s} {"markers":24s} {"mean":>6s} {"p95":>6s} {"max":>6s}  status')
    high = []
    for region, idx in _by_region(names):
        rp95 = p95[idx].max()
        status = _status(rp95, GOOD_CM, LIMIT_CM)
        if status == 'HIGH':
            high.append(region)
        markers = _marker_types(names, idx)
        print(f'    {region:8s} {markers[:24]:24s} {err_cm[:, idx].mean():6.2f} {rp95:6.2f} '
              f'{err_cm[:, idx].max():6.2f}  {status}')

    print('    worst markers: ' + ', '.join(
        f'{names[j]} (p95 {p95[j]:.2f} cm, worst in {100 * worst_share[j]:.0f}% of frames)'
        for j in np.argsort(-p95)[:3]))
    print(f'    overall: RMS {frame_rms.mean():.2f} cm (worst frame {frame_rms.max():.2f}), '
          f'max {err_cm.max():.2f} cm, frames with a marker > {LIMIT_CM:g} cm: '
          f'{100 * np.mean(err_cm.max(axis=1) > LIMIT_CM):.1f}%')

    if frame_rms.mean() > GOOD_CM or [r for r in high if r != 'foot']:
        print(f'    VERDICT: CHECK - high errors in {", ".join(high) or "overall RMS"}')
    elif high:
        print('    VERDICT: USABLE - only foot markers are high (rigid model foot; low IK weight)')
    else:
        print('    VERDICT: USABLE')


def print_scale_report(names, static_err_cm, moved_cm, scale_factors):
    """static_err_cm / moved_cm: per marker, in cm. scale_factors: {body: (sx, sy, sz)}."""
    static_err_cm, moved_cm = np.asarray(static_err_cm), np.asarray(moved_cm)

    print('\n  Scale factors (flag outside 0.8-1.25):')
    for body, s in scale_factors.items():
        flag = '' if all(0.8 <= v <= 1.25 for v in s) else '  <-- check'
        print(f'    {body:10s} {s[0]:.3f} {s[1]:.3f} {s[2]:.3f}{flag}')

    print(f'\n  Marker check: static trial  (fit = scaled model before markers are moved, '
          f'good <= {STATIC_FIT_GOOD_CM:g} / ok <= {STATIC_FIT_LIMIT_CM:g} cm;')
    print(f'                               moved = how far each marker was moved, '
          f'good <= {MOVED_GOOD_CM:g} / ok <= {MOVED_LIMIT_CM:g} cm)')
    print(f'    {"region":8s} {"markers":24s} {"fit":>6s} {"moved":>6s}  status')
    flagged = []
    for region, idx in _by_region(names):
        fit, moved = static_err_cm[idx].max(), moved_cm[idx].max()
        s1 = _status(fit, STATIC_FIT_GOOD_CM, STATIC_FIT_LIMIT_CM)
        s2 = _status(moved, MOVED_GOOD_CM, MOVED_LIMIT_CM)
        status = 'HIGH' if 'HIGH' in (s1, s2) else 'ok' if 'ok' in (s1, s2) else 'good'
        if status == 'HIGH':
            flagged.append(region)
        markers = _marker_types(names, idx)
        print(f'    {region:8s} {markers[:24]:24s} {fit:6.2f} {moved:6.2f}  {status}')
    print('    most moved: ' + ', '.join(f'{names[j]} {moved_cm[j]:.1f} cm' for j in np.argsort(-moved_cm)[:3]))
    if flagged:
        print(f'    VERDICT: CHECK {", ".join(flagged)} - large fit error or marker move '
              f'(mislabelled / misplaced marker?)')
    else:
        print('    VERDICT: USABLE')


def read_scale_factors(scale_factors_xml):
    """{body: (sx, sy, sz)} from the ScaleTool's scale-factor file, offsets and ground left out."""
    text = open(scale_factors_xml, encoding='utf-8').read()
    pairs = re.findall(r'<scales>([^<]+)</scales>\s*(?:<!--.*?-->\s*)*<segment>([^<]+)</segment>', text, re.S)
    return {seg.strip(): tuple(float(v) for v in s.split()) for s, seg in pairs
            if seg.strip() != 'ground' and not seg.strip().endswith('_offset')}
