"""Rebuild the greater trochanter (GTR) position from the pelvis and thigh cluster markers.

A static window of the standing trial defines where the GTR sits relative to the pelvis
markers (ASIS + PSIS) and to the thigh cluster (CLUST1-3). In the test trial the rigid
motion of each marker set is fitted every frame (least squares / Kabsch) and applied to the
GTR, giving one pelvis-based and one cluster-based estimate. The derived GTR is their
weighted average (PELVIS_WEIGHT), compared with the measured GTR marker and plotted.

Edit the settings below, then run:  python scripts/hip_from_cluster.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------- Settings ----------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SESSION = '09_21_2026 AB02'         # folder name under Data/
STATIC_TRIAL = 'standing'           # reference trial (subject standing still)
TEST_TRIAL = 'walking_maria'        # e.g. 'lg_1p2_walking' for 10_08_2026 AB01
REF_DURATION_S = 0.3                # length of the reference window
REF_START_S = None                  # start of the window in seconds; None = stillest window
SIDES = ('R', 'L')
TARGET = 'GTR'
PELVIS_WEIGHT = 0.65                # weight of the pelvis estimate; the cluster gets 1 - PELVIS_WEIGHT

DATA_DIR = os.path.join(_ROOT, 'Data', SESSION)
OUT_DIR = os.path.join(_ROOT, 'Results', SESSION, 'hip_from_cluster')

PELVIS = ('RASIS', 'LASIS', 'RPSIS', 'LPSIS')
CLUSTER = ('CLUST1', 'CLUST2', 'CLUST3')

SIDE_NAME = {'R': 'Right', 'L': 'Left'}
AXIS_COLORS = ('tab:red', 'tab:green', 'tab:purple')


def read_trc(path):
    """Returns (time, {marker: (n_frames, 3) array in mm}). Missing samples are NaN."""
    with open(path) as f:
        lines = f.readlines()
    names = [n for n in lines[3].rstrip('\n').split('\t')[2:] if n]
    data = pd.read_csv(path, sep='\t', skiprows=5, header=None).apply(pd.to_numeric, errors='coerce')
    data = data.dropna(how='all', axis=1)
    time = data.iloc[:, 1].to_numpy()
    markers = {name: data.iloc[:, 2 + 3 * i: 5 + 3 * i].to_numpy() for i, name in enumerate(names)}
    return time, markers


def cluster_markers(side):
    return [side + m for m in CLUSTER]


def reference_window(time, markers):
    """Index slice of the REF_DURATION_S window: REF_START_S, or where the markers move least."""
    n = int(round(REF_DURATION_S / np.median(np.diff(time))))
    if REF_START_S is not None:
        start = int(np.searchsorted(time, REF_START_S))
        return slice(start, start + n)
    names = list(PELVIS) + [n for s in SIDES for n in cluster_markers(s) + [s + TARGET]]
    speed = sum(np.linalg.norm(np.diff(markers[m], axis=0), axis=1) for m in names)
    window_motion = np.convolve(speed, np.ones(n - 1), mode='valid')
    start = int(np.nanargmin(window_motion))
    return slice(start, start + n)


def fit_rigid(ref, cur):
    """Rotation R and translation t with cur ≈ R @ ref + t (least squares, Kabsch).

    Markers missing in cur are left out; needs at least 3 visible markers.
    """
    ok = ~np.isnan(cur).any(axis=1)
    if ok.sum() < 3:
        return None, None
    ref, cur = ref[ok], cur[ok]
    ref_c, cur_c = ref.mean(axis=0), cur.mean(axis=0)
    h = (ref - ref_c).T @ (cur - cur_c)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, cur_c - rot @ ref_c


def carry_point(static_ref, test, names, point):
    """Lab position, every test frame, of a point rigidly attached to the marker set `names`."""
    ref = np.array([static_ref[n] for n in names])
    frames = np.stack([test[n] for n in names], axis=1)
    out = np.full((len(frames), 3), np.nan)
    for i, cur in enumerate(frames):
        rot, t = fit_rigid(ref, cur)
        if rot is not None:
            out[i] = rot @ point + t
    return out


def derive_target(static_ref, test, side):
    target_ref = static_ref[side + TARGET]
    from_pelvis = carry_point(static_ref, test, list(PELVIS), target_ref)
    from_cluster = carry_point(static_ref, test, cluster_markers(side), target_ref)
    return PELVIS_WEIGHT * from_pelvis + (1 - PELVIS_WEIGHT) * from_cluster


def main():
    static_time, static = read_trc(os.path.join(DATA_DIR, f'{STATIC_TRIAL}.trc'))
    test_time, test = read_trc(os.path.join(DATA_DIR, f'{TEST_TRIAL}.trc'))

    window = reference_window(static_time, static)
    print(f'Reference window: {static_time[window][0]:.2f}-{static_time[window][-1]:.2f} s of {STATIC_TRIAL}')
    static_ref = {name: np.nanmean(xyz[window], axis=0) for name, xyz in static.items()}

    os.makedirs(OUT_DIR, exist_ok=True)
    for side in SIDES:
        error = derive_target(static_ref, test, side) - test[side + TARGET]
        dist = np.linalg.norm(error, axis=1)
        print(f'{SIDE_NAME[side]} {TARGET}: 3D error mean {np.nanmean(dist):.1f} mm, '
              f'RMS {np.sqrt(np.nanmean(dist ** 2)):.1f} mm, max {np.nanmax(dist):.1f} mm')

        fig, (ax_dist, ax_xyz) = plt.subplots(2, 1, sharex=True, figsize=(11, 7))
        ax_dist.plot(test_time, dist, color='black', linewidth=0.8)
        ax_dist.set_ylabel('3D distance error (mm)')
        ax_dist.set_ylim(bottom=0)
        for k, (axis_name, color) in enumerate(zip('XYZ', AXIS_COLORS)):
            ax_xyz.plot(test_time, error[:, k], color=color, linewidth=0.8, label=axis_name)
        ax_xyz.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
        ax_xyz.set_ylabel('Error (mm)')
        ax_xyz.set_xlabel('Time (s)')
        ax_xyz.legend(loc='upper right')
        fig.suptitle(f'{SESSION} {TEST_TRIAL}: {SIDE_NAME[side]} {TARGET}, derived minus measured\n'
                     f'(pelvis weight {PELVIS_WEIGHT:.2f}, cluster weight {1 - PELVIS_WEIGHT:.2f})')
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f'gtr_error_{SIDE_NAME[side].lower()}_{TEST_TRIAL}.png'), dpi=150)

    print(f'Plots saved to {OUT_DIR}')
    plt.show()


if __name__ == '__main__':
    main()
