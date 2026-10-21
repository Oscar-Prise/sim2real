"""
Hip, knee and ankle angles (IK) and moments (ID) over the gait cycle, left and right side.
Edit the settings below, then run:

    python scripts/util_IKIDGaitPlot.py

For every trial in TRIALS, two figures are shown and saved: one for IK, one for ID. Each has 3 rows
(hip, knee, ankle) and 2 columns (left side on the left, right side on the right), mean ± 1 SD
across gait cycles.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

import config as cfg

# ======================= Settings you change often =======================
SESSION = '10_18_2026 AB01'     # folder name under Data/ and Results/
# Any number of trials (Data/<SESSION>/<trial>.mot, Results/<SESSION>/<trial>/), each gets its own figures
TRIALS = [
    'ExoOn_LG_1p2_loose',
    'ExoOn_LG_1p2_moderate',
    'ExoOn_LG_1p2_tight',
]
SUBJECT_ID = 'AB01'             # looked up in Subject Information.txt for body mass
SUBJECT_MASS_KG = None          # None = read from Subject Information.txt; or a number to override

SHOW_FIGS = True
SAVE_FIGS = True
FIG_FORMAT = 'png'
FIG_DPI = 300

LOWPASS_HZ = 6.0                # low-pass on IK and ID before cutting cycles; None = no filter
GAIT_POINTS = 101               # samples per gait cycle (0 to 100 %)
CYCLE_TOLERANCE = 0.2           # drop cycles more than 20 % longer or shorter than the median stride
MIN_SWING_TIME = 0.1            # s the COP must stay frozen (foot off the plate) before a new heel strike
FORCE_PLATE = {'r': 1, 'l': 2}  # plate number under each foot (see forces_<trial>.xml)

# Exo hip vs human hip flexion, one row per entry (trials whose IK has the exo coordinates only):
#   'exo_pelvis':   exo thigh relative to the exo pelvis (exo_hip_Rrz / Lrz, what the exo hinge sees)
#   'human_pelvis': exo thigh relative to the human pelvis (exo pelvis tilt included)
EXO_HIP_VERSIONS = ['exo_pelvis', 'human_pelvis']   # [] = no exo hip figure

# ======================= Paths (built from the settings above) =======================
# Any folder can be replaced by a full path, e.g. r'C:\Users\oscar\Documents\some_session'.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, 'Data', SESSION)
RESULTS_DIR = os.path.join(_ROOT, 'Results', SESSION)


def trial_paths(trial):
    """Inputs and figure outputs of one trial; figures go next to its IK / ID results."""
    results = os.path.join(RESULTS_DIR, trial)
    return dict(
        ik=os.path.join(results, f'ik_{trial}.mot'),
        id=os.path.join(results, f'id_{trial}.sto'),
        force_mot=os.path.join(DATA_DIR, f'{trial}.mot'),
        ik_fig=os.path.join(results, f'IK_gait_{trial}.{FIG_FORMAT}'),
        id_fig=os.path.join(results, f'ID_gait_{trial}.{FIG_FORMAT}'),
        exo_hip_fig=os.path.join(results, f'Hip_exo_gait_{trial}.{FIG_FORMAT}'),
    )

# (row title, IK column without side suffix, sign, positive direction, negative direction)
# sign = -1 for the knee: gait2392 knee_angle is negative in flexion; flipped so flexion is up.
JOINTS = [
    ('Hip', 'hip_flexion', 1, 'Flexion', 'Extension'),
    ('Knee', 'knee_angle', -1, 'Flexion', 'Extension'),
    ('Ankle', 'ankle_angle', 1, 'Dorsiflexion', 'Plantarflexion'),
]
SIDES = [('l', 'Left', 'tab:orange'), ('r', 'Right', 'tab:blue')]   # column order: left, right
EXO_COLOR = 'tab:green'


def exo_hip_vs_exo_pelvis(ik, side):
    return ik[f'exo_hip_{side.upper()}rz'].values


def exo_hip_vs_human_pelvis(ik, side):
    """Exo thigh orientation in the human pelvis frame (Pelvis2ExoP rotates about X, Y, Z, then the
    exo hip pin about Z), as the flexion angle of the Z-X-Y sequence the human hip joint uses."""
    angles = np.column_stack([ik['exoP_rx'], ik['exoP_ry'], ik['exoP_rz'] + ik[f'exo_hip_{side.upper()}rz']])
    return Rotation.from_euler('XYZ', angles, degrees=True).as_euler('ZXY', degrees=True)[:, 0]


# version: (row title, exo hip angle in deg from the IK table and side)
EXO_HIP_DEFINITIONS = {
    'exo_pelvis': ('Exo thigh vs exo pelvis', exo_hip_vs_exo_pelvis),
    'human_pelvis': ('Exo thigh vs human pelvis', exo_hip_vs_human_pelvis),
}


def read_opensim_table(path):
    """.mot / .sto file as a DataFrame (header lines up to 'endheader' are skipped)."""
    with open(path, 'r') as f:
        header_end = next(i for i, line in enumerate(f) if 'endheader' in line)
    # index_col=False: force .mot rows end with a tab, which would otherwise shift every column
    return pd.read_csv(path, sep='\t', skiprows=header_end + 1, index_col=False)


def lowpass(signal, time, cutoff_hz, order=4):
    if cutoff_hz is None:
        return signal
    fs = 1.0 / np.median(np.diff(time))
    b, a = butter(order, cutoff_hz / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)


def detect_heel_strikes_from_cop(time, cop_x, cop_z, min_swing_time):
    """Times when the COP starts moving again after being frozen for at least min_swing_time.

    While the foot is off the plate, this Vicon export holds the COP at a fixed value, so a
    frozen COP means swing and the first sample where it changes again is heel strike.
    """
    moving = np.r_[True, (np.diff(cop_x) != 0) | (np.diff(cop_z) != 0)]
    strikes = []
    frozen_since = None
    for i in range(len(time)):
        if not moving[i]:
            if frozen_since is None:
                frozen_since = time[i - 1] if i > 0 else time[i]
        else:
            if frozen_since is not None and time[i] - frozen_since >= min_swing_time:
                strikes.append(time[i])
            frozen_since = None
    return np.array(strikes)


def segment_gait_cycles(time, signal, heel_strikes):
    """Cut signal into heel-strike-to-heel-strike cycles resampled to GAIT_POINTS samples.

    Cycles far from the median stride (gait start / stop, missed steps) are left out.
    Returns an array of shape (n_cycles, GAIT_POINTS).
    """
    if len(heel_strikes) < 2:
        return np.empty((0, GAIT_POINTS))
    durations = np.diff(heel_strikes)
    median = np.median(durations)
    cycles = []
    for start, end, duration in zip(heel_strikes[:-1], heel_strikes[1:], durations):
        if abs(duration - median) > CYCLE_TOLERANCE * median:
            continue
        if start < time[0] or end > time[-1]:
            continue
        cycles.append(np.interp(np.linspace(start, end, GAIT_POINTS), time, signal))
    return np.array(cycles).reshape(-1, GAIT_POINTS)


def label_direction(ax, positive, negative):
    """Dashed line at 0 plus the meaning of up / down written in the top / bottom left corners."""
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    text_kw = dict(transform=ax.transAxes, ha='left', fontsize=8, color='dimgray')
    ax.text(0.01, 0.98, f'{positive} \u2191', va='top', **text_kw)
    ax.text(0.01, 0.02, f'{negative} \u2193', va='bottom', **text_kw)


def plot_gait_figure(table, column_suffix, scale, heel_strikes, title, ylabel_unit):
    """3 x 2 figure: rows hip / knee / ankle, columns left / right, mean ± 1 SD over the gait cycle."""
    time = table['time'].values
    percent = np.linspace(0, 100, GAIT_POINTS)
    fig, axes = plt.subplots(len(JOINTS), len(SIDES), sharex=True, sharey='row', figsize=(11, 9))
    fig.suptitle(title, fontsize=14)

    for row, (joint, column, sign, positive, negative) in enumerate(JOINTS):
        for col, (side, side_name, color) in enumerate(SIDES):
            ax = axes[row, col]
            signal = lowpass(table[f'{column}_{side}{column_suffix}'].values, time, LOWPASS_HZ) * scale * sign
            cycles = segment_gait_cycles(time, signal, heel_strikes[side])
            if len(cycles) == 0:
                raise ValueError(f"No gait cycles found for side '{side}' - check the force plate .mot")
            mean, sd = cycles.mean(axis=0), cycles.std(axis=0)
            ax.plot(percent, mean, color=color, label=f'n = {len(cycles)} cycles')
            ax.fill_between(percent, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0)

            ax.set_xlim(0, 100)
            ax.legend(loc='upper right', fontsize=8)
            if row == 0:
                ax.set_title(side_name, fontsize=12)
            if row == len(JOINTS) - 1:
                ax.set_xlabel('Gait cycle (%)')
            if col == 0:
                ax.set_ylabel(f'{joint} {ylabel_unit}')
            label_direction(ax, positive, negative)

    fig.tight_layout()
    return fig


def plot_exo_hip_figure(ik, heel_strikes, title):
    """One row per EXO_HIP_VERSIONS entry, columns left / right: human hip flexion and exo hip angle,
    mean ± 1 SD over the gait cycle. The legend gives the mean and RMS of exo - human."""
    time = ik['time'].values
    percent = np.linspace(0, 100, GAIT_POINTS)
    fig, axes = plt.subplots(len(EXO_HIP_VERSIONS), len(SIDES), sharex=True, sharey=True,
                             figsize=(11, 3.2 * len(EXO_HIP_VERSIONS) + 0.8), squeeze=False)
    fig.suptitle(title, fontsize=14)

    for row, version in enumerate(EXO_HIP_VERSIONS):
        row_title, exo_angle = EXO_HIP_DEFINITIONS[version]
        for col, (side, side_name, color) in enumerate(SIDES):
            ax = axes[row, col]
            human = segment_gait_cycles(time, lowpass(ik[f'hip_flexion_{side}'].values, time, LOWPASS_HZ),
                                        heel_strikes[side])
            exo = segment_gait_cycles(time, lowpass(exo_angle(ik, side), time, LOWPASS_HZ), heel_strikes[side])
            diff = exo - human
            for cycles, c, label in [(human, color, f'human hip (n = {len(human)})'),
                                     (exo, EXO_COLOR, f'exo: diff {diff.mean():+.1f}, RMS {np.sqrt(np.mean(diff ** 2)):.1f} deg')]:
                mean, sd = cycles.mean(axis=0), cycles.std(axis=0)
                ax.plot(percent, mean, color=c, label=label)
                ax.fill_between(percent, mean - sd, mean + sd, color=c, alpha=0.25, linewidth=0)

            ax.set_xlim(0, 100)
            ax.legend(loc='upper right', fontsize=8)
            if row == 0:
                ax.set_title(side_name, fontsize=12)
            if row == len(EXO_HIP_VERSIONS) - 1:
                ax.set_xlabel('Gait cycle (%)')
            if col == 0:
                ax.set_ylabel(f'{row_title}\nangle (deg)')
            label_direction(ax, 'Flexion', 'Extension')

    fig.tight_layout()
    return fig


def plot_trial(trial, mass):
    paths = trial_paths(trial)
    forces = read_opensim_table(paths['force_mot'])
    force_time = forces['time'].values
    heel_strikes = {
        side: detect_heel_strikes_from_cop(force_time, forces[f'ground_force{plate}_px'].values,
                                           forces[f'ground_force{plate}_pz'].values, MIN_SWING_TIME)
        for side, plate in FORCE_PLATE.items()
    }
    print(f"{trial}: {len(heel_strikes['l'])} left / {len(heel_strikes['r'])} right heel strikes")

    ik = read_opensim_table(paths['ik'])
    id_ = read_opensim_table(paths['id'])
    fig_ik = plot_gait_figure(ik, '', 1.0, heel_strikes,
                              f'Joint angles from inverse kinematics - {trial} (mean ± 1 SD)', 'angle (deg)')
    fig_id = plot_gait_figure(id_, '_moment', 1.0 / mass, heel_strikes,
                              f'Joint moments from inverse dynamics - {trial} (mean ± 1 SD)', 'moment (Nm/kg)')
    figs = {'ik_fig': fig_ik, 'id_fig': fig_id}
    if EXO_HIP_VERSIONS and 'exo_hip_Rrz' in ik.columns:
        figs['exo_hip_fig'] = plot_exo_hip_figure(ik, heel_strikes,
                                                  f'Exo vs human hip angle - {trial} (mean ± 1 SD)')

    for key, fig in figs.items():
        if SAVE_FIGS:
            fig.savefig(paths[key], dpi=FIG_DPI, bbox_inches='tight')
            print(f"  saved {os.path.basename(paths[key])}")
        if not SHOW_FIGS:
            plt.close(fig)


def main():
    mass = SUBJECT_MASS_KG or cfg.subject_info(SUBJECT_ID).mass_kg
    print(f"{SESSION}: {len(TRIALS)} trial(s), mass {mass} kg")
    for trial in TRIALS:
        plot_trial(trial, mass)
    if SHOW_FIGS:
        plt.show()


if __name__ == '__main__':
    main()
