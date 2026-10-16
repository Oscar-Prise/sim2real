import os

import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import numpy as np
from scipy.signal import butter, filtfilt

# ---------- Files to plot ----------
# Edit these, then run:  python scripts/plot_ik_id.py
# IK_FILE is the inverse-kinematics .mot (joint angles, degrees).
# ID_FILE is the inverse-dynamics .sto (joint moments, N·m).
# FORCE_MOT is the trial's force-plate .mot from Data/, used to find heel strikes.
# SUBJECT_MASS_KG divides the moments so the plot is in N·m/kg.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

IK_FILE = os.path.join(_ROOT, 'Results', '10_08_2026 AB01', 'lg_1p2_walking', 'ik_lg_1p2_walking.mot')
ID_FILE = os.path.join(_ROOT, 'Results', '10_08_2026 AB01', 'lg_1p2_walking', 'id_lg_1p2_walking.sto')
FORCE_MOT = os.path.join(_ROOT, 'Data', '10_08_2026 AB01', 'lg_1p2_walking.mot')
SUBJECT_MASS_KG = 74.5
SIDE = 'both'        # 'both', 'r', or 'l'
LAYOUT = 'together'  # 'together' (angle and moment as two subplots) or 'separate' (one figure each)

GAIT_POINTS = 101           # samples per gait cycle (0 to 100 %)
CYCLE_TOLERANCE = 0.2       # drop cycles more than 20 % longer or shorter than the median stride


def filter(data, cutoff=4, fs=100, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

def load_force_data(path):
        """
        Loads the force data from a CSV file.
        """
        filepath = path
        force_data = pd.read_csv(filepath, skiprows=5, nrows=3200, header=None)
        force_data = force_data.apply(pd.to_numeric, errors='coerce').dropna()
        time = np.arange(0, len(force_data) * 0.001, 0.001)[:len(force_data)]
        return time, force_data

def get_gait_indicies(self, grf):
        threshold = self.subject_mass * 9.81 *0.1 #convert to N from kg

        grf_copy = grf.copy()
        grf_copy[grf_copy < threshold] = 0
        grf_copy[grf_copy >= threshold] = 1
        grf_copy = np.diff(grf_copy)
        heel_strikes = np.where(grf_copy == 1)[0]

        # Remove heel strikes that are less than 50 indices apart
        heel_strikes = heel_strikes[np.insert(np.diff(heel_strikes) >= 500, 0, True)]
        return heel_strikes

def find_row_with_string(filepath, search_string):
        with open(filepath, 'r') as f:
            for i, line in enumerate(f):
                if search_string in line:
                    return i
        raise ValueError(f"'{search_string}' not found in {filepath}")


def upsample_signal(self, signal, upsample_factor=10):
    original_indices = np.linspace(0, 1, len(signal))
    new_indices = np.linspace(0, 1, len(signal) * upsample_factor)
    return np.interp(new_indices, original_indices, signal)

def load_ik_id(path):
        """
        Loads the inverse kinematics and inverse dynamics data from a CSV file.
        """
        filepath = path
        search_string = "Model Outputs"  # Replace with the actual string you want to find
        row_idx = find_row_with_string(filepath, search_string)
        ik_id = pd.read_csv(filepath, skiprows=row_idx+6, nrows=1590, header=None)
        ik_id = ik_id.apply(pd.to_numeric, errors='coerce').dropna(how='all', axis=1)
        return ik_id

def segment_by_gait_cycles(self, data, heel_strikes,target_length=1000):
    """
    Segments the data into gait cycles using heel strike indices and upsamples each cycle.
    Returns segmented data (each column is a cycle) and the average cycle.
    """
    segmented_data = []
    # Use consecutive pairs of heel strikes as gait cycle boundaries
    for i in range(len(heel_strikes) - 1):
        start = heel_strikes[i]
        end = heel_strikes[i + 1]
        current_gc = data[start:end]
        if len(current_gc) < 2:  # Skip if not enough data to interpolate
            continue
        # Upsample to target_length using linear interpolation
        original_indices = np.linspace(0, 100, len(current_gc))
        new_indices = np.linspace(0, 100, target_length)
        upsampled_gc = np.interp(new_indices, original_indices, current_gc)
        segmented_data.append(upsampled_gc)
    if not segmented_data:
        return None, None
    segmented_data = np.array(segmented_data).T  # shape: (target_length, num_cycles)
    #avg_data = np.mean(segmented_data, axis=1)
    return segmented_data 

def read_opensim_mot(file_path):
    # Open the file to find the line where the header ends
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # Find the index of 'endheader'
    header_end_idx = 0
    for i, line in enumerate(lines):
        if 'endheader' in line:
            header_end_idx = i
            break
            
    # Load the data into a DataFrame
    # OpenSim files typically use tabs (\t) as separators
    # index_col=False: force .mot rows end with a tab, which would otherwise shift every column
    df = pd.read_csv(file_path, sep='\t', skiprows=header_end_idx + 1, index_col=False)
    return df


def detect_heel_strikes(time, fz, threshold, min_off_time=0.1):
    """Times when the foot goes from off the plate to on it (same rule as utils_gait_seg.py).

    Contact comes from vertical force, not COP: this Vicon export holds COP at the plate
    centre instead of 0 while the foot is off. The foot must stay off for min_off_time
    so force noise around the threshold is not counted as a new step.
    """
    on_plate = fz > threshold
    strikes = []
    off_since = None
    for i in range(1, len(fz)):
        if not on_plate[i]:
            if off_since is None:
                off_since = time[i]
        else:
            if off_since is not None and time[i] - off_since >= min_off_time:
                strikes.append(time[i])
            off_since = None
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


SIDE_STYLE = {'r': ('Right', 'tab:blue'), 'l': ('Left', 'tab:orange')}


def sides_to_plot():
    side = SIDE.strip().lower()
    if side not in ('both', 'r', 'l'):
        raise ValueError("SIDE must be 'both', 'r', or 'l'")
    return ['r', 'l'] if side == 'both' else [side]


def layout_is_together():
    layout = LAYOUT.strip().lower()
    if layout == 'together':
        return True
    if layout in ('separate', 'seperate'):
        return False
    raise ValueError("LAYOUT must be 'together' or 'separate'")


def make_axes():
    """Angle and moment axes: two stacked subplots in one figure, or one figure each."""
    if layout_is_together():
        fig, (ax_angle, ax_moment) = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
        fig.subplots_adjust(hspace=0.3)
        return ax_angle, ax_moment
    return plt.figure().gca(), plt.figure().gca()


def plot_time(ax, time, series, name):
    """series is {'r': array, 'l': array}."""
    for s in sides_to_plot():
        label, color = SIDE_STYLE[s]
        ax.plot(time, series[s], color=color, label=f'{label} {name}')


def plot_gait(ax, cycles, name):
    """Mean across cycles as a line, ±1 standard deviation shaded. cycles is {'r': ..., 'l': ...}."""
    percent = np.linspace(0, 100, GAIT_POINTS)
    for s in sides_to_plot():
        if len(cycles[s]) == 0:
            raise ValueError(f"No gait cycles found for side '{s}' - check FORCE_MOT")
        label, color = SIDE_STYLE[s]
        mean = cycles[s].mean(axis=0)
        sd = cycles[s].std(axis=0)
        ax.plot(percent, mean, color=color, label=f'{label} {name} (n={len(cycles[s])})')
        ax.fill_between(percent, mean - sd, mean + sd, color=color, alpha=0.25, linewidth=0)


def finish_axes(ax, title, xlabel, ylabel, show_xlabel=True):
    ax.set_title(title)
    if show_xlabel:
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    label_direction(ax)
    ax.legend()


def label_direction(ax, positive='Flexion', negative='Extension'):
    """Dashed line at 0 plus the meaning of up / down written along the y-axis."""
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    text_kw = dict(transform=ax.transAxes, rotation=90, ha='right', fontsize=10, color='dimgray')
    ax.text(-0.09, 1.0, positive, va='top', **text_kw)
    ax.text(-0.09, 0.0, negative, va='bottom', **text_kw)
    ax.yaxis.labelpad = 22
    ax.figure.subplots_adjust(left=0.2)



#load force plate data (.mot): plate 1 is the right foot, plate 2 the left (see forces_<trial>.xml)
forces = read_opensim_mot(FORCE_MOT)
force_time = forces['time'].values
contact_threshold = 0.1 * SUBJECT_MASS_KG * 9.81  # 10 % body weight, in N
heel_strikes = {
    'r': detect_heel_strikes(force_time, forces['ground_force1_vy'].values, contact_threshold),
    'l': detect_heel_strikes(force_time, forces['ground_force2_vy'].values, contact_threshold),
}

#Load IK results from .mot file
IK = read_opensim_mot(IK_FILE)

# 2. Explicitly save each column as an individual variable
ik_time = IK['time'].values

# Pelvis translations and rotations
pelvis_tilt = filter(IK['pelvis_tilt'].values)
pelvis_list = filter(IK['pelvis_list'].values)
pelvis_rotation = filter(IK['pelvis_rotation'].values)
pelvis_tx = filter(IK['pelvis_tx'].values)
pelvis_ty = filter(IK['pelvis_ty'].values)
pelvis_tz = filter(IK['pelvis_tz'].values)

# Right side joints
hip_flexion_r = filter(IK['hip_flexion_r'].values)
hip_adduction_r = filter(IK['hip_adduction_r'].values)
hip_rotation_r = filter(IK['hip_rotation_r'].values)
knee_angle_r = filter(IK['knee_angle_r'].values)
ankle_angle_r = filter(IK['ankle_angle_r'].values)
subtalar_angle_r = filter(IK['subtalar_angle_r'].values)
mtp_angle_r = filter(IK['mtp_angle_r'].values)

# Left side joints
hip_flexion_l = filter(IK['hip_flexion_l'].values)
hip_adduction_l = filter(IK['hip_adduction_l'].values) 
hip_rotation_l = filter(IK['hip_rotation_l'].values)
knee_angle_l = filter(IK['knee_angle_l'].values)
ankle_angle_l = filter(IK['ankle_angle_l'].values)
subtalar_angle_l = filter(IK['subtalar_angle_l'].values)
mtp_angle_l = filter(IK['mtp_angle_l'].values)

# Lumbar (lower back)
lumbar_extension = filter(IK['lumbar_extension'].values)
lumbar_bending = filter(IK['lumbar_bending'].values)
lumbar_rotation = filter(IK['lumbar_rotation'].values)

#load ID results from .sto file
ID = read_opensim_mot(ID_FILE)

# 2. Explicitly save each column to an individual variable
id_time = ID['time'].values

# Pelvis Moments and Forces
pelvis_tilt_moment = filter(ID['pelvis_tilt_moment'].values)
pelvis_list_moment = filter(ID['pelvis_list_moment'].values)
pelvis_rotation_moment = filter(ID['pelvis_rotation_moment'].values)
pelvis_tx_force = filter(ID['pelvis_tx_force'].values)
pelvis_ty_force = filter(ID['pelvis_ty_force'].values)
pelvis_tz_force = filter(ID['pelvis_tz_force'].values)

# Hip Moments
hip_flexion_r_moment = filter(ID['hip_flexion_r_moment'].values)
hip_adduction_r_moment = filter(ID['hip_adduction_r_moment'].values)
hip_rotation_r_moment = filter(ID['hip_rotation_r_moment'].values)
hip_flexion_l_moment = filter(ID['hip_flexion_l_moment'].values)
hip_adduction_l_moment = filter(ID['hip_adduction_l_moment'].values)
hip_rotation_l_moment = filter(ID['hip_rotation_l_moment'].values)

# Lumbar (Back) Moments
lumbar_extension_moment = filter(ID['lumbar_extension_moment'].values)
lumbar_bending_moment = filter(ID['lumbar_bending_moment'].values)
lumbar_rotation_moment = filter(ID['lumbar_rotation_moment'].values)

# Knee, Ankle, Subtalar, and MTP Moments
knee_angle_r_moment = filter(ID['knee_angle_r_moment'].values)
knee_angle_l_moment = filter(ID['knee_angle_l_moment'].values)
ankle_angle_r_moment = filter(ID['ankle_angle_r_moment'].values)
ankle_angle_l_moment = filter(ID['ankle_angle_l_moment'].values)
subtalar_angle_r_moment = filter(ID['subtalar_angle_r_moment'].values)
subtalar_angle_l_moment = filter(ID['subtalar_angle_l_moment'].values)
mtp_angle_r_moment = filter(ID['mtp_angle_r_moment'].values)
mtp_angle_l_moment = filter(ID['mtp_angle_l_moment'].values)

hip_angle = {'r': hip_flexion_r, 'l': hip_flexion_l}
hip_moment = {'r': hip_flexion_r_moment / SUBJECT_MASS_KG, 'l': hip_flexion_l_moment / SUBJECT_MASS_KG}
together = layout_is_together()

# Hip angle and moment over time
ax_angle, ax_moment = make_axes()
plot_time(ax_angle, ik_time, hip_angle, 'hip flexion')
finish_axes(ax_angle, 'Hip joint angle from inverse kinematics', 'Time (s)', 'Hip flexion (deg)',
            show_xlabel=not together)
plot_time(ax_moment, id_time, hip_moment, 'hip flexion')
finish_axes(ax_moment, 'Hip joint moment from inverse dynamics', 'Time (s)', 'Hip flexion moment (Nm/kg)')

# Hip angle and moment over the gait cycle (each side cut at its own heel strikes)
angle_cycles = {s: segment_gait_cycles(ik_time, hip_angle[s], heel_strikes[s]) for s in ('r', 'l')}
moment_cycles = {s: segment_gait_cycles(id_time, hip_moment[s], heel_strikes[s]) for s in ('r', 'l')}

ax_angle, ax_moment = make_axes()
plot_gait(ax_angle, angle_cycles, 'hip flexion')
finish_axes(ax_angle, 'Hip joint angle over the gait cycle (mean ± 1 SD)', 'Gait cycle (%)',
            'Hip flexion (deg)', show_xlabel=not together)
plot_gait(ax_moment, moment_cycles, 'hip flexion')
finish_axes(ax_moment, 'Hip joint moment over the gait cycle (mean ± 1 SD)', 'Gait cycle (%)',
            'Hip flexion moment (Nm/kg)')

plt.show()
