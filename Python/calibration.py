import pandas as pd
import numpy as np
import scipy.signal as signal

from pathlib import Path

def calculate_gravity(g_interp_x, g_interp_y, g_interp_z):
    g = np.array([g_interp_x, g_interp_y, g_interp_z], dtype=float)
    g /= np.linalg.norm(g, axis=0)
    return g

def calculate_raw_peaks(acc, g, dt_sampling):
    # Accelerometer Magnitude
    a = np.array([acc['x'].values, acc['y'].values, acc['z'].values], dtype=float)
    acc_vertical = np.sum(a * g, axis=0) * g
    acc_linear = a - acc_vertical
    acc_mag = np.linalg.norm(acc_linear, axis=0)

    # Low-pass filter magnitude
    b, a_filter = signal.butter(4, 2.5 / (0.5 * (1.0 / dt_sampling)), btype='low')
    acc_mag_filt = signal.filtfilt(b, a_filter, acc_mag)

    # Find raw candidate peaks
    raw_peaks, _ = signal.find_peaks(acc_mag_filt, prominence=0.2, distance=int(0.35 / dt_sampling))

    return raw_peaks, acc_mag

def walk_cadence_verification(acc, g, dt_sampling, t_acc):
    valid_peaks = []
    raw_peaks, acc_mag = calculate_raw_peaks(acc, g, dt_sampling)

    for i in range(len(raw_peaks)):
        idx = raw_peaks[i]
        t_curr = t_acc[idx]
        
        prev_dt = (t_curr - t_acc[raw_peaks[i-1]]) if i > 0 else None
        next_dt = (t_acc[raw_peaks[i+1]] - t_curr) if i < len(raw_peaks)-1 else None
        
        is_rhythmic = False
        if prev_dt is not None and 0.35 <= prev_dt <= 1.3:
            is_rhythmic = True
        if next_dt is not None and 0.35 <= next_dt <= 1.3:
            is_rhythmic = True
            
        if is_rhythmic or len(raw_peaks) == 1:
            valid_peaks.append(idx)
    
    peaks = np.array(valid_peaks)
    
    return peaks, acc_mag

def calibrate_weinberg_constant(actual_distance, peaks, acc_mag):
    step_feature_sum = 0
    
    for i in range(len(peaks)):
        idx = peaks[i]
        
        w_start = max(0, idx - 10)
        w_end = min(len(acc_mag), idx + 10)
        a_max = np.max(acc_mag[w_start:w_end])
        a_min = np.min(acc_mag[w_start:w_end])
        
        feature = (a_max - a_min) ** 0.25 # K = 1
        step_feature_sum += feature

    if step_feature_sum == 0:
        return 0

    # K = Actual Distance / Sum of (a_max - a_min)^0.25
    calibrated_k = actual_distance / step_feature_sum
    return calibrated_k

def calibrate_cadence_adaptive_k(actual_distance, peaks, acc_mag, step_times):
    step_feature_sum = 0
    
    for i in range(len(peaks)):
        idx = peaks[i]

        if i == 0:
            dt = step_times[1] - step_times[0] if len(peaks) > 1 else 0.55
        else:
            dt = step_times[i] - step_times[i-1]
            
        dt = np.clip(dt, 0.35, 1.3)
        cadence_hz = 1.0 / dt
        
        w_start = max(0, idx - 10)
        w_end = min(len(acc_mag), idx + 10)
        a_max = np.max(acc_mag[w_start:w_end])
        a_min = np.min(acc_mag[w_start:w_end])

        cadence_scaling = np.sqrt(cadence_hz / 1.8)
        feature = cadence_scaling * ((a_max - a_min) ** 0.25)

        step_feature_sum += feature

    if step_feature_sum == 0:
        return 0

    # K = Actual Distance / Sum of (a_max - a_min)^0.25
    calibrated_k = actual_distance / step_feature_sum
    return calibrated_k

if __name__ == '__main__':
    ground_truth_distance = 60 # Ground Truth (m)
    
    # Folders
    parent_folder_path = Path(__file__).parent.resolve()
    dataset_folder = parent_folder_path / "SensorLogger-Cali"
    input_data_name = "2026-09-06_06-04-41"
    input_data_path = dataset_folder / input_data_name

    # Load Data (Assuming a straight line walk)
    acc = pd.read_csv(input_data_path / "Accelerometer.csv").sort_values('seconds_elapsed').reset_index(drop=True)
    # gyro = pd.read_csv(input_data_path / "Gyroscope.csv").sort_values('seconds_elapsed').reset_index(drop=True)
    grav = pd.read_csv(input_data_path / "Gravity.csv").sort_values('seconds_elapsed').reset_index(drop=True)

    # Timestamps and sampling rate
    t_acc = acc['seconds_elapsed'].values
    t_grav = grav['seconds_elapsed'].values
    dt_sampling = np.mean(np.diff(t_acc))

    # Calculate gravity interpolation
    g_interp_x = np.interp(t_acc, t_grav, grav['x'].values)
    g_interp_y = np.interp(t_acc, t_grav, grav['y'].values)
    g_interp_z = np.interp(t_acc, t_grav, grav['z'].values)

    # Process forces
    g = calculate_gravity(g_interp_x, g_interp_y, g_interp_z)

    # Find Steps
    peaks, acc_mag = walk_cadence_verification(acc, g, dt_sampling, t_acc)

    # Calibrate Cadence-Adaptive Constant
    # recommended_k = calibrate_weinberg_constant(ground_truth_distance, peaks, acc_mag)
    recommended_k = calibrate_cadence_adaptive_k(ground_truth_distance, peaks, acc_mag, t_acc[peaks])

    # If K is 0, it means no valid steps were detected.
    if recommended_k == 0:
        print("No valid steps detected.")

    print(f"Ground Truth Distance: {ground_truth_distance} m")
    print(f"Total Steps Detected:  {len(peaks)}")
    # print(f"Recommended Weinberg Constant (K): {recommended_k:.4f}")
    print(f"Recommended Cadence-Adaptive Constant (K): {recommended_k:.4f}")