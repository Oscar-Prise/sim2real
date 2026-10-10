import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import resample
import numpy as np
from scipy.signal import butter, filtfilt
import os

import config as cfg


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
    df = pd.read_csv(file_path, sep='\t', skiprows=header_end_idx + 1)
    return df


def label_direction(ax, positive='Flexion', negative='Extension'):
    """Dashed line at 0 plus the meaning of up / down written along the y-axis."""
    ax.axhline(0, color='gray', linestyle='--', linewidth=1, zorder=0)
    text_kw = dict(transform=ax.transAxes, rotation=90, ha='right', fontsize=10, color='dimgray')
    ax.text(-0.09, 1.0, positive, va='top', **text_kw)
    ax.text(-0.09, 0.0, negative, va='bottom', **text_kw)
    ax.yaxis.labelpad = 22
    ax.figure.subplots_adjust(left=0.2)



#Session / trial to plot: set in config.py, or pass them:  python plot_ik_id.py "10_08_2026 AB01" lg_1p2_walking
import sys
session = sys.argv[1] if len(sys.argv) > 1 else cfg.SESSION
trial = sys.argv[2] if len(sys.argv) > 2 else cfg.TRIAL
paths = cfg.trial_paths(session, trial)
subject_mass_kg = cfg.subject_info(session).mass_kg

#load force data (vicon File .csv)
time, force_data = load_force_data(paths.csv)

grfR = abs(force_data.iloc[:, 4].to_numpy())
grfL = abs(force_data.iloc[:, 13].to_numpy())

#Load IK results from .mot file
IK = read_opensim_mot(paths.ik)

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
ID = read_opensim_mot(paths.id)

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

# Hip joint angle over time from IK
plt.figure()
plt.plot(ik_time, hip_flexion_r, label='Right hip flexion')
plt.plot(ik_time, hip_flexion_l, label='Left hip flexion')
plt.xlabel('Time (s)')
plt.ylabel('Hip flexion (deg)')
plt.title('Hip joint angle from inverse kinematics')
label_direction(plt.gca())
plt.legend()

# Hip flexion moment over time from ID
plt.figure()
plt.plot(id_time, hip_flexion_r_moment / subject_mass_kg, label='Right hip flexion')
plt.plot(id_time, hip_flexion_l_moment / subject_mass_kg, label='Left hip flexion')
plt.xlabel('Time (s)')
plt.ylabel('Hip flexion moment (Nm/kg)')
plt.title('Hip joint moment from inverse dynamics')
label_direction(plt.gca())
plt.legend()

plt.show()
