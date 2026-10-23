"""
Segment-to-segment angles over the gait cycle from the BodyKinematics results of
run_body_kinematics.py: femur vs pelvis and femur vs torso for every trial on the same axes, and
for each exo trial a hip figure with the exo thigh next to the human femur. Edit the settings
below, then run:

    python scripts/util_BodyKinematicsPlot.py

Angles are the rotation of a segment relative to a reference segment, zeroed on the static trial
(the static pose is 0 deg on every axis) and decomposed like the gait2392 hip joint: flexion (+),
adduction (+), internal rotation (+). Figures go to Results/<session>/bodykinematics/.
"""
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

import util_IKIDGaitPlot as gait

# ======================= Settings =======================
SESSION = '10_18_2026 AB01'
# trial: (static trial whose pose is the zero, legend label, color). Any number of trials.
TRIALS = {
    'NoExo_LG_1p2': ('NoExo_Static', 'NoExo', 'black'),
    'ExoOn_LG_1p2_loose': ('ExoOn_Static', 'ExoOn loose', 'tab:red'),
    'ExoOn_LG_1p2_moderate': ('ExoOn_Static', 'ExoOn moderate', 'tab:purple'),
    'ExoOn_LG_1p2_tight': ('ExoOn_Static', 'ExoOn tight', 'tab:green'),
}
STATIC_WINDOW = (3.0, 3.03)   # s of the static trial averaged as the zero pose; None = whole trial

# (figure name, segment, reference segment); {s} = r / l, {S} = R / L
COMPARISONS = [
    ('Femur_vs_Pelvis', 'femur_{s}', 'pelvis'),
    ('Femur_vs_Torso', 'femur_{s}', 'torso'),
]
# Hip_<trial> figures (exo trials only), flexion. One row per human femur pair; every row also
# shows all EXO_CURVES (exo thigh vs exo pelvis and vs torso).
HIP_ROWS = [('vs pelvis', ('femur_{s}', 'pelvis')), ('vs torso', ('femur_{s}', 'torso'))]
EXO_CURVES = [(('ExoT{S}', 'ExoP'), 'tab:orange'), (('ExoT{S}', 'torso'), 'tab:green')]
NOEXO_REFERENCE = 'NoExo_LG_1p2'   # its human femur is drawn as a dashed mean
NOEXO_COLOR, EXO_ON_HUMAN_COLOR = 'black', 'tab:blue'

# Global_<mode>_<trial> figures (exo trials only): segments vs the lab frame, NOEXO_REFERENCE vs the trial.
# Rows: (title, segment, side whose heel strikes cut the cycles, (positive, negative) label per axis).
# Same Z-X-Y decomposition as the gait2392 pelvis joint (absolute pelvis = pelvis_tilt / list / rotation).
_TRUNK_AXES = [('Posterior tilt', 'Anterior tilt'), ('Right side down', 'Left side down'),
               ('Turn left', 'Turn right')]
_FEMUR_AXES = [('Flexion', 'Extension'), ('Adduction', 'Abduction'), ('Internal rot.', 'External rot.')]
GLOBAL_ROWS = [
    ('Torso', 'torso', 'r', _TRUNK_AXES),
    ('Pelvis', 'pelvis', 'r', _TRUNK_AXES),
    ('Left femur', 'femur_l', 'l', _FEMUR_AXES),
    ('Right femur', 'femur_r', 'r', _FEMUR_AXES),
]
GLOBAL_COLUMNS = ['Sagittal (about Z)', 'Frontal (about X)', 'Transverse (about Y)']
# 'static' = 0 is the static pose, 'absolute' = orientation in the lab frame; one figure per mode
GLOBAL_MODES = ['static', 'absolute']

# Fixed y-axis limits (deg) so all figures share the same scale; None = automatic
AXES_YLIM = [(-40, 40), (-20, 20), (-20, 20)]   # Femur_vs_* rows: flexion, adduction, rotation
HIP_YLIM = (-40, 40)                            # Hip_<trial> figures

SHOW_FIGS = True
FIG_FORMAT = 'png'
FIG_DPI = 300

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, 'Data', SESSION)
RESULTS_DIR = os.path.join(_ROOT, 'Results', SESSION)
FIG_DIR = os.path.join(RESULTS_DIR, 'bodykinematics')

AXES = [('Flexion', 'Extension'), ('Adduction', 'Abduction'), ('Internal rot.', 'External rot.')]
SIDES = [('l', 'Left'), ('r', 'Right')]   # column order: left, right


def read_body_orientations(trial):
    """BodyKinematics position file of a trial as a DataFrame (body_Ox/Oy/Oz = body-fixed XYZ, deg)."""
    path = os.path.join(RESULTS_DIR, trial, f'{trial}_BodyKinematics_pos_global.sto')
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}\nRun run_body_kinematics.py first.")
    return gait.read_opensim_table(path)


def has_body(table, body):
    return f'{body}_Ox' in table.columns


def orientation(table, body):
    return Rotation.from_euler('XYZ', table[[f'{body}_Ox', f'{body}_Oy', f'{body}_Oz']].values, degrees=True)


def relative_orientation(table, segment, reference):
    rel = orientation(table, segment)
    return rel if reference == 'ground' else orientation(table, reference).inv() * rel


def relative_angles(table, static, segment, reference, side, zero_on_static=True):
    """(n, 3) Z-X-Y angles of segment vs reference ('ground' = lab frame), 0 = static pose unless
    zero_on_static is False. For a femur: flexion / adduction / internal rotation."""
    rel = relative_orientation(table, segment, reference)
    if zero_on_static:
        static_rel = relative_orientation(static, segment, reference)
        if STATIC_WINDOW:
            t = static['time'].values
            static_rel = static_rel[(t >= STATIC_WINDOW[0] - 1e-9) & (t <= STATIC_WINDOW[1] + 1e-9)]
        rel = static_rel.mean().inv() * rel
    angles = rel.as_euler('ZXY', degrees=True)
    if side == 'l':   # left hip adduction / rotation axes point the other way (gait2392)
        angles[:, 1:] *= -1
    return angles


def heel_strikes(trial):
    forces = gait.read_opensim_table(os.path.join(DATA_DIR, f'{trial}.mot'))
    t = forces['time'].values
    return {side: gait.detect_heel_strikes_from_cop(t, forces[f'ground_force{plate}_px'].values,
                                                    forces[f'ground_force{plate}_pz'].values, gait.MIN_SWING_TIME)
            for side, plate in gait.FORCE_PLATE.items()}


def names(pair, side):
    return [p.format(s=side, S=side.upper()) for p in pair]


def load_trials():
    """{trial: (table, static table, heel strikes)}; statics are read once."""
    statics, data = {}, {}
    for trial, (static, _, _) in TRIALS.items():
        if static not in statics:
            statics[static] = read_body_orientations(static)
        data[trial] = (read_body_orientations(trial), statics[static], heel_strikes(trial))
    return data


def plot_band(ax, percent, cycles, color, label):
    mean, sd = cycles.mean(axis=0), cycles.std(axis=0)
    ax.plot(percent, mean, color=color, label=label)
    ax.fill_between(percent, mean - sd, mean + sd, color=color, alpha=0.15, linewidth=0)


def gait_cycles(table, angle, side, strikes):
    return gait.segment_gait_cycles(table['time'].values, angle, strikes[side])


def plot_comparison(data, fig_name, segment, reference):
    percent = np.linspace(0, 100, gait.GAIT_POINTS)
    fig, axes = plt.subplots(3, 2, sharex=True, sharey='row', figsize=(11, 9))
    fig.suptitle(f'{segment.format(s="").rstrip("_").capitalize()} relative to {reference} '
                 f'(0 = static pose, mean ± 1 SD)', fontsize=13)
    print(f'\n{fig_name}: peak flexion / extension (deg)')
    for col, (side, side_name) in enumerate(SIDES):
        seg, ref = names((segment, reference), side)
        for trial, (table, static, strikes) in data.items():
            _, label, color = TRIALS[trial]
            angles = relative_angles(table, static, seg, ref, side)
            for row in range(3):
                cycles = gait_cycles(table, angles[:, row], side, strikes)
                plot_band(axes[row, col], percent, cycles, color, f'{label} (n = {len(cycles)})')
                if row == 0:
                    m = cycles.mean(axis=0)
                    print(f'   {side_name:5s} {label:15s} {m.max():6.1f} / {m.min():6.1f}')
        for row, (pos, neg) in enumerate(AXES):
            ax = axes[row, col]
            ax.set_xlim(0, 100)
            if AXES_YLIM and AXES_YLIM[row]:
                ax.set_ylim(*AXES_YLIM[row])
            gait.label_direction(ax, pos, neg)
            if row == 0:
                ax.set_title(side_name, fontsize=12)
                ax.legend(loc='upper right', fontsize=7)
            if row == 2:
                ax.set_xlabel('Gait cycle (%)')
            if col == 0:
                ax.set_ylabel(f'{pos.split()[0]} (deg)')
    fig.tight_layout()
    return fig


def plot_hip_figure(trial, data):
    """Hip flexion of one exo trial: rows = HIP_ROWS (femur vs pelvis, femur vs torso), columns
    left / right. Each row: dashed NOEXO_REFERENCE femur mean, this trial's femur and every
    EXO_CURVES exo thigh as mean ± 1 SD."""
    percent = np.linspace(0, 100, gait.GAIT_POINTS)
    table, static, strikes = data[trial]
    ref_table, ref_static, ref_strikes = data[NOEXO_REFERENCE]
    label, ref_label = TRIALS[trial][1], TRIALS[NOEXO_REFERENCE][1]
    flexion = lambda t, s, seg, ref, side, hs: gait_cycles(t, relative_angles(t, s, seg, ref, side)[:, 0], side, hs)

    fig, axes = plt.subplots(len(HIP_ROWS), 2, sharex=True, sharey=True,
                             figsize=(11, 3.2 * len(HIP_ROWS) + 0.8), squeeze=False)
    fig.suptitle(f'Hip flexion - {label} vs {ref_label} (0 = static pose, mean ± 1 SD)', fontsize=13)
    for col, (side, side_name) in enumerate(SIDES):
        exo = [(names(pair, side), color, flexion(table, static, *names(pair, side), side, strikes))
               for pair, color in EXO_CURVES]
        for row, (title, human_pair) in enumerate(HIP_ROWS):
            ax = axes[row, col]
            seg, ref = names(human_pair, side)
            noexo = flexion(ref_table, ref_static, seg, ref, side, ref_strikes)
            ax.plot(percent, noexo.mean(axis=0), color=NOEXO_COLOR, ls='--', lw=1.2,
                    label=f'{ref_label}: {seg} vs {ref} (n = {len(noexo)})')
            human = flexion(table, static, seg, ref, side, strikes)
            plot_band(ax, percent, human, EXO_ON_HUMAN_COLOR, f'{label}: {seg} vs {ref} (n = {len(human)})')
            for (exo_seg, exo_ref), color, cycles in exo:
                plot_band(ax, percent, cycles, color, f'{label}: {exo_seg} vs {exo_ref}')

            ax.set_xlim(0, 100)
            if HIP_YLIM:
                ax.set_ylim(*HIP_YLIM)
            gait.label_direction(ax, 'Flexion', 'Extension')
            ax.legend(loc='upper right', fontsize=7)
            if row == 0:
                ax.set_title(side_name, fontsize=12)
            if row == len(HIP_ROWS) - 1:
                ax.set_xlabel('Gait cycle (%)')
            if col == 0:
                ax.set_ylabel(f'Thigh {title}\nflexion (deg)')
    fig.tight_layout()
    return fig


def plot_global_figure(trial, data, mode):
    """GLOBAL_ROWS segments vs the lab frame, columns = GLOBAL_COLUMNS: NOEXO_REFERENCE and this
    trial, mean ± 1 SD. mode 'static' zeroes on each condition's static pose, 'absolute' does not."""
    percent = np.linspace(0, 100, gait.GAIT_POINTS)
    zero = mode == 'static'
    fig, axes = plt.subplots(len(GLOBAL_ROWS), 3, sharex=True, figsize=(14, 2.6 * len(GLOBAL_ROWS) + 0.8),
                             squeeze=False)
    fig.suptitle(f'Segments vs lab frame - {TRIALS[trial][1]} vs {TRIALS[NOEXO_REFERENCE][1]} '
                 f'({"0 = static pose" if zero else "absolute"}, mean ± 1 SD)', fontsize=13)
    for row, (title, segment, side, labels) in enumerate(GLOBAL_ROWS):
        # Only femur_l uses the left-side sign convention
        sign_side = 'l' if segment.endswith('_l') else 'r'
        for name, color in [(NOEXO_REFERENCE, NOEXO_COLOR), (trial, EXO_ON_HUMAN_COLOR)]:
            table, static, strikes = data[name]
            angles = relative_angles(table, static, segment, 'ground', sign_side, zero)
            for col in range(3):
                cycles = gait_cycles(table, angles[:, col], side, strikes)
                plot_band(axes[row, col], percent, cycles, color, f'{TRIALS[name][1]} (n = {len(cycles)})')
        for col, (pos, neg) in enumerate(labels):
            ax = axes[row, col]
            ax.set_xlim(0, 100)
            if AXES_YLIM and AXES_YLIM[col]:
                ax.set_ylim(*AXES_YLIM[col])
            gait.label_direction(ax, pos, neg)
            ax.legend(loc='upper right', fontsize=7)
            if row == 0:
                ax.set_title(GLOBAL_COLUMNS[col], fontsize=11)
            if row == len(GLOBAL_ROWS) - 1:
                ax.set_xlabel('Gait cycle (%)')
            if col == 0:
                ax.set_ylabel(f'{title}\n(deg, cycle from {side.upper()} HS)')
    fig.tight_layout()
    return fig


def save(fig, name):
    path = os.path.join(FIG_DIR, f'{name}.{FIG_FORMAT}')
    # An image open in a viewer / the IDE preview can briefly lock the file on Windows
    for attempt in range(10):
        try:
            fig.savefig(path, dpi=FIG_DPI, bbox_inches='tight')
            break
        except OSError:
            if attempt == 9:
                raise
            time.sleep(0.5)
    print(f'  saved {os.path.relpath(path, RESULTS_DIR)}')
    if not SHOW_FIGS:
        plt.close(fig)


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    data = load_trials()
    for fig_name, segment, reference in COMPARISONS:
        save(plot_comparison(data, fig_name, segment, reference), fig_name)
    for trial, (table, _, _) in data.items():
        if NOEXO_REFERENCE in data and has_body(table, 'ExoP'):
            if HIP_ROWS:
                save(plot_hip_figure(trial, data), f'Hip_{trial}')
            for mode in GLOBAL_MODES:
                save(plot_global_figure(trial, data, mode), f'Global_{mode}_{trial}')
    if SHOW_FIGS:
        plt.show()


if __name__ == '__main__':
    main()
