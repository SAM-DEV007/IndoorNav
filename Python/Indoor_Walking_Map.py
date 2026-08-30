import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

from scipy.integrate import cumulative_trapezoid
from pathlib import Path

def calculate_gravity(g_interp_x, g_interp_y, g_interp_z):
    g = np.array([g_interp_x, g_interp_y, g_interp_z], dtype=float)
    g /= np.linalg.norm(g, axis=0) # Normalize
    
    return g

def calculate_gyro(gz_interp_x, gz_interp_y, gz_interp_z):
    gyro = np.array([gz_interp_x, gz_interp_y, gz_interp_z], dtype=float)
    gz_interp = np.sum(g * gyro, axis=0)

    return gz_interp

def calculate_mag_heading(mag_interp_x, mag_interp_y, mag_interp_z, g):
    m = np.array([mag_interp_x, mag_interp_y, mag_interp_z], dtype=float)

    horizontal = m - np.sum(m * g, axis = 0) * g
    horizontal /= np.linalg.norm(horizontal, axis=0)

    forward = np.array([np.zeros_like(g[0]), np.ones_like(g[0]), np.zeros_like(g[0])]) # [0, 1, 0] - +Y forward default
    forward_h = (forward - np.sum(forward * g, axis=0) * g)
    forward_h /= np.linalg.norm(forward_h, axis=0)

    right_h = np.cross(g.T, forward_h.T).T
    right_h /= np.linalg.norm(right_h, axis=0)

    north_component = np.sum(horizontal * forward_h, axis=0)
    east_component  = np.sum(horizontal * right_h, axis=0)

    mag_heading = np.degrees(np.arctan2(east_component, north_component)) % 360

    return mag_heading

def calculate_raw_peaks(acc, g, dt_sampling):
    # Accelerometer Magnitude
    a = np.array([acc['x'].values, acc['y'].values, acc['z'].values], dtype=float)
    acc_vertical = np.sum(a * g, axis=0) * g
    acc_linear = a - acc_vertical
    acc_mag = np.linalg.norm(acc_linear, axis=0)

    # Low-pass filter magnitude
    b, a = signal.butter(4, 2.5 / (0.5 * (1.0 / dt_sampling)), btype='low')
    acc_mag_filt = signal.filtfilt(b, a, acc_mag)

    # Find raw candidate peaks (Threshold & Minimum distance)
    raw_peaks, _ = signal.find_peaks(acc_mag_filt, height=np.mean(acc_mag_filt) + 0.35, distance=int(0.35 / dt_sampling))

    return raw_peaks, acc_mag

def calculate_yaw_gyro(gz_interp, gyro_threshold, t_acc):
    gz_gated = np.where(np.abs(gz_interp) < gyro_threshold, 0.0, gz_interp)
    yaw_gyro = cumulative_trapezoid(gz_gated, t_acc, initial=0)

    return yaw_gyro

def walk_cadence_verification(acc, g, dt_sampling, t_acc):
    # Human walking cadence is 1.2 Hz - 2.5 Hz (step interval ~0.35s to 1.3s). 
    # Isolated vertical movements (lifting phone, drops, elevator jolts) lack rhythmic step neighbours.
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
    
    return peaks, t_acc[peaks], acc_mag

def weinberg_stride_calculation(peaks, step_times, acc_mag, K, use_turn_attenuation, gz_interp, x_pos, y_pos):
    # Only valid for walking cadence
    strides = []
    step_durations = []

    for i in range(len(peaks)):
        idx = peaks[i]
        if i == 0:
            dt = step_times[1] - step_times[0] if len(peaks) > 1 else 0.5
        else:
            dt = step_times[i] - step_times[i-1]
        
        w_start = max(0, idx - 10)
        w_end = min(len(acc_mag), idx + 10)
        a_max = np.max(acc_mag[w_start:w_end])
        a_min = np.min(acc_mag[w_start:w_end])
        
        base_stride = K * ((a_max - a_min) ** 0.25)
        stride = base_stride
        
        # Turn-rate attenuation factor (Decays stride length during sharp turns)
        if use_turn_attenuation is True:
            turn_rate = np.abs(gz_interp[idx])
            turn_attenuation = np.exp(-1.8 * turn_rate)
            stride *= turn_attenuation
        
        strides.append(stride)
        step_durations.append(dt)
        
        heading = step_yaws_unwrapped[i]
        x_pos.append(x_pos[-1] + stride * np.cos(heading))
        y_pos.append(y_pos[-1] + stride * np.sin(heading))

    x_pos = np.array(x_pos)
    y_pos = np.array(y_pos)
    strides = np.array(strides)
    step_durations = np.array(step_durations)
    step_speeds = strides / step_durations
    cum_distance = np.cumsum(strides)

    return x_pos, y_pos, strides, step_durations, step_speeds, cum_distance

def create_distance_logs(peaks, step_times, strides, step_speeds, cum_distance, x_pos, y_pos, step_mag_heading, step_heading_deg, step_heading_unwrapped_deg, step_heading_change_deg, gz_interp):
    pdr_df = pd.DataFrame({
        'Step': np.arange(1, len(peaks) + 1),
        'Seconds_Elapsed': np.round(step_times, 2),
        'Stride_Length_m': np.round(strides, 3),
        'Speed_m_s': np.round(step_speeds, 3),
        'Cumulative_Distance_m': np.round(cum_distance, 2),
        'X_m': np.round(x_pos[1:], 2),
        'Y_m': np.round(y_pos[1:], 2),
        'Magnetic_Heading_deg': np.round(step_mag_heading, 2),
        'Gyro_Heading_deg': np.round(step_heading_deg, 1),
        'Gyro_Heading_Unwrapped_deg': np.round(step_heading_unwrapped_deg, 1),
        'Heading_Change_deg': np.round(step_heading_change_deg, 1),
        'Gyro_Rate_rad_s': np.round(gz_interp[peaks], 3)
    })

    return pdr_df

def extract_low_speed(step_speeds):
    low_speed_indices = np.where(step_speeds < 0.3)[0]

    low_speed_events = []
    if len(low_speed_indices) > 0:
        curr = [low_speed_indices[0]]
        for idx in low_speed_indices[1:]:
            if idx == curr[-1] + 1:
                curr.append(idx)
            else:
                low_speed_events.append(curr)
                curr = [idx]
        low_speed_events.append(curr)
    
    return low_speed_indices, low_speed_events

def create_landmark_logs(low_speed_events, cum_distance):
    landmarks = []
    prev_dist = 0.0

    for i, grp in enumerate(low_speed_events):
        s_idx = grp[0]
        e_idx = grp[-1]
        cdist = cum_distance[s_idx]
        interval = cdist - prev_dist
        landmarks.append({
            'Landmark_ID': f"LM_Pause_{i+1}",
            'Step_Range': f"Steps {s_idx+1}-{e_idx+1}",
            'Time_s': f"{step_times[s_idx]:.2f}s - {step_times[e_idx]:.2f}s",
            'Cumulative_Dist_m': round(cdist, 2),
            'Distance_From_Prev_Landmark_m': round(interval, 2)
        })
        prev_dist = cdist

    return pd.DataFrame(landmarks)

def detect_turns(peaks, step_heading_unwrapped_deg, step_gyro_rate, gyro_turn_threshold, min_turn_angle_deg, max_gap_steps):
    local_heading_change_deg = np.zeros(len(peaks))
    for i in range(len(peaks)):
        # Look one step before and one step after
        left = max(0, i - 1)
        right = min(len(peaks) - 1, i + 1)

        local_heading_change_deg[i] = step_heading_unwrapped_deg[right] - step_heading_unwrapped_deg[left]

    # Candidate turn steps
    turn_candidate_mask = ((np.abs(step_gyro_rate) >= gyro_turn_threshold) & (np.abs(local_heading_change_deg) >= min_turn_angle_deg / 2))
    turn_indices = np.where(turn_candidate_mask)[0]

    # Group nearby candidate steps into physical turn events
    turn_groups = []
    if len(turn_indices) > 0:
        curr = [turn_indices[0]]
        for idx in turn_indices[1:]:
            # Only merge genuinely nearby steps
            if idx <= curr[-1] + max_gap_steps:
                curr.append(idx)
            else:
                turn_groups.append(curr)
                curr = [idx]
        turn_groups.append(curr)

    # Validate each group by TOTAL accumulated heading change
    valid_turn_groups = []
    for grp in turn_groups:
        start_idx = max(0, grp[0] - 1)
        end_idx = min(len(peaks) - 1, grp[-1] + 1)

        # Total gyro-derived heading change across the turn
        total_turn_angle = step_heading_unwrapped_deg[end_idx] - step_heading_unwrapped_deg[start_idx]

        # Keep only meaningful turns
        if abs(total_turn_angle) >= min_turn_angle_deg:
            valid_turn_groups.append(grp)

    return valid_turn_groups, turn_indices

def create_turn_logs(turn_groups, step_gyro_rate, peaks, step_heading_unwrapped_deg, cum_distance, step_times, step_heading_deg):
    # Build turn event log
    turns = []
    prev_turn_end_dist = 0.0

    for i, grp in enumerate(turn_groups):
        start_idx = grp[0]
        end_idx = grp[-1]

        # Peak angular-rate step inside the turn
        peak_idx = grp[np.argmax(np.abs(step_gyro_rate[grp]))]

        # Total heading change over the turn
        angle_start_idx = max(0, start_idx - 1)
        angle_end_idx = min(len(peaks) - 1, end_idx + 1)
        turn_angle_deg = step_heading_unwrapped_deg[angle_end_idx] - step_heading_unwrapped_deg[angle_start_idx]

        # Determine turn direction
        if turn_angle_deg > 0:
            turn_direction = "Left"
        elif turn_angle_deg < 0:
            turn_direction = "Right"
        else:
            turn_direction = "Straight"

        # Straight distance BEFORE this turn:
        # Previous turn end -> current turn start
        turn_start_dist = cum_distance[start_idx]
        straight_segment_length = (
            turn_start_dist - prev_turn_end_dist
        )

        turns.append({
            'Turn_ID': f"Turn_{i+1}",
            'Start_Step': start_idx + 1,
            'End_Step': end_idx + 1,
            'Peak_Turn_Step': peak_idx + 1,
            'Start_Time_s': round(step_times[start_idx], 2),
            'End_Time_s': round(step_times[end_idx], 2),
            'Start_Dist_m': round(cum_distance[start_idx], 2),
            'End_Dist_m': round(cum_distance[end_idx], 2),
            'Start_Heading_deg': round(step_heading_deg[start_idx], 1),
            'End_Heading_deg': round(step_heading_deg[end_idx], 1),
            'Turn_Angle_deg': round(turn_angle_deg, 1),
            'Turn_Direction': turn_direction,
            'Peak_Gyro_Rate_rad_s': round(step_gyro_rate[peak_idx], 3),
            'Straight_Segment_Before_Turn_m': round(straight_segment_length, 2)
        })

        # Next straight segment begins AFTER this turn
        prev_turn_end_dist = cum_distance[end_idx]

    turns_df = pd.DataFrame(turns)

    # Final straight segment after the last turn
    if len(turn_groups) > 0:
        final_straight_len = (cum_distance[-1] - prev_turn_end_dist)
    else:
        final_straight_len = cum_distance[-1]

    return turns_df, round(final_straight_len, 2)

def plot_walking_map(x_pos, y_pos, low_speed_indices, turn_indices, title, save_name, output_data_save):
    plt.figure(figsize=(10, 8))
    plt.plot(x_pos, y_pos, linestyle='--', color='gray', zorder=1, label='Trajectory')

    sc = plt.scatter(x_pos[1:], y_pos[1:], c=step_speeds, cmap='viridis', s=50, zorder=2, edgecolor='k')
    cbar = plt.colorbar(sc)
    cbar.set_label('Step Speed (m/s)')

    plt.scatter(x_pos[0], y_pos[0], color='green', marker='o', s=150, label='Start', zorder=3)
    plt.scatter(x_pos[-1], y_pos[-1], color='red', marker='X', s=150, label='End', zorder=3)

    if len(low_speed_indices) > 0:
        plt.scatter(x_pos[low_speed_indices+1], y_pos[low_speed_indices+1], color='orange', marker='s', s=100, label='<0.3 m/s Landmark', zorder=4)

    if len(turn_indices) > 0:
        plt.scatter(x_pos[turn_indices+1], y_pos[turn_indices+1], color='red', marker='^', s=50, label='Turns Landmark', zorder=4)

    plt.title(title)
    plt.xlabel('X Position (meters)')
    plt.ylabel('Y Position (meters)')
    plt.axis('equal')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_data_save / f'{save_name}.png', dpi=300)
    plt.close()

def save_logs(pdr_df, landmarks_df, turns_df, output_data_save):
    pdr_df.to_csv(output_data_save / f"pdr_distance_log.csv", index=False)
    landmarks_df.to_csv(output_data_save / f"pdr_landmarks_log.csv", index=False)
    turns_df.to_csv(output_data_save / f"pdr_turns_log.csv", index=False)

if __name__ == '__main__':
    # Load Data
    parent_folder_path = Path(__file__).parent.resolve()
    dataset_folder = parent_folder_path / "SensorLogger"

    save_folder = parent_folder_path / "Output"

    input_data_name = "2026-08-30_01-03-47"
    input_data_path = dataset_folder / input_data_name

    output_data_save = save_folder / input_data_name
    output_data_save.mkdir(parents=True, exist_ok=True)

    acc = pd.read_csv(input_data_path / "Accelerometer.csv").sort_values('seconds_elapsed').reset_index(drop=True)
    gyro = pd.read_csv(input_data_path / "Gyroscope.csv").sort_values('seconds_elapsed').reset_index(drop=True)
    mag = pd.read_csv(input_data_path / "Magnetometer.csv").sort_values('seconds_elapsed').reset_index(drop=True)
    grav = pd.read_csv(input_data_path / "Gravity.csv").sort_values('seconds_elapsed').reset_index(drop=True)

    # Config
    gyro_threshold = 0.12 # rad/s threshold

    K = 0.55  # Weinberg constant
    x_pos, y_pos = [0.0], [0.0]
    use_turn_attenuation = False

    gyro_turn_threshold = 0.15 # rad/s
    min_turn_angle_deg = 15.0
    max_gap_steps = 1

    # Timestamps and sampling rate
    t_acc = acc['seconds_elapsed'].values
    t_gyro = gyro['seconds_elapsed'].values
    t_mag = mag['seconds_elapsed'].values
    t_grav = grav['seconds_elapsed'].values
    dt_sampling = np.mean(np.diff(t_acc))

    # Calculate Yaw (Heading)
    gz_interp_x = np.interp(t_acc, t_gyro, gyro['x'].values)
    gz_interp_y = np.interp(t_acc, t_gyro, gyro['y'].values)
    gz_interp_z = np.interp(t_acc, t_gyro, gyro['z'].values)

    # Calculate magnetic fields
    mag_interp_x = np.interp(t_acc, t_mag, mag['x'].values)
    mag_interp_y = np.interp(t_acc, t_mag, mag['y'].values)
    mag_interp_z = np.interp(t_acc, t_mag, mag['z'].values)

    # Calculate gravity
    g_interp_x = np.interp(t_acc, t_grav, grav['x'].values)
    g_interp_y = np.interp(t_acc, t_grav, grav['y'].values)
    g_interp_z = np.interp(t_acc, t_grav, grav['z'].values)

    g = calculate_gravity(g_interp_x, g_interp_y, g_interp_z)
    gx, gy, gz = g

    # Gyro rotation with respect to Earth's gravity
    gz_interp = calculate_gyro(gz_interp_x, gz_interp_y, gz_interp_z)

    # Calculate tilt
    mag_heading = calculate_mag_heading(mag_interp_x, mag_interp_y, mag_interp_z, g)
    print(f"Initial Magnetic Heading: {mag_heading[0]}")

    # VERTICAL FALSE STEP FILTERING (Cadence & Rhythm Verification)
    peaks, step_times, acc_mag = walk_cadence_verification(acc, g, dt_sampling, t_acc)

    # Heading Gyro
    yaw_gyro = calculate_yaw_gyro(gz_interp, gyro_threshold, t_acc)

    step_yaws_unwrapped = yaw_gyro[peaks]
    step_heading_unwrapped_deg = np.rad2deg(step_yaws_unwrapped)
    step_heading_deg = np.mod(step_heading_unwrapped_deg, 360.0)
    step_heading_change_deg = np.diff(step_heading_unwrapped_deg, prepend=step_heading_unwrapped_deg[0]) # Per-step heading change from gyro

    step_mag_heading = mag_heading[peaks] # Peaks magnetic heading

    # Trajectory & Stride Calculation
    x_pos, y_pos, strides, step_durations, step_speeds, cum_distance = weinberg_stride_calculation(peaks, step_times, acc_mag, K, use_turn_attenuation, gz_interp, x_pos, y_pos)
    pdr_df = create_distance_logs(peaks, step_times, strides, step_speeds, cum_distance, x_pos, y_pos, step_mag_heading, step_heading_deg, step_heading_unwrapped_deg, step_heading_change_deg, gz_interp) # Create distance logs

    # Extract Low-Speed (< 0.3 m/s) Landmark Events
    low_speed_indices, low_speed_events = extract_low_speed(step_speeds)
    landmarks_df = create_landmark_logs(low_speed_events, cum_distance)

    # Extract Turn Events & Straight Segment Lengths
    step_gyro_rate = gz_interp[peaks]
    turn_groups, turn_indices = detect_turns(peaks, step_heading_unwrapped_deg, step_gyro_rate, gyro_turn_threshold, min_turn_angle_deg, max_gap_steps)
    turns_df, final_straight_len = create_turn_logs(turn_groups, step_gyro_rate, peaks, step_heading_unwrapped_deg, cum_distance, step_times, step_heading_deg)

    # Plot Walking Map with Landmarks
    plot_walking_map(x_pos, y_pos, low_speed_indices, turn_indices, 'Pedestrian Dead Reckoning with Landmark', 'pdr_map', output_data_save)

    # Plot rotated magnetic map
    theta = np.deg2rad(90.0 - step_mag_heading[0])
    x_rot = x_pos * np.cos(theta) - y_pos * np.sin(theta)
    y_rot = x_pos * np.sin(theta) + y_pos * np.cos(theta)

    plot_walking_map(x_rot, y_rot, low_speed_indices, turn_indices, 'Pedestrian Dead Reckoning with Landmark - Aligned with Magnetic North', 'pdr_map_magnetic', output_data_save)

    # Save logs
    save_logs(pdr_df, landmarks_df, turns_df, output_data_save)

    # Print Results
    print(f"Initial Magnetic Heading (step): {step_mag_heading[0]}\n")

    print("DISTANCE TRAVELLED WITH SECONDS ELAPSED")
    print(pdr_df.to_string(index=False))

    print("\nDISTANCE INTERVALS BETWEEN LOW-SPEED (<0.3 m/s) STEPS")
    print(landmarks_df.to_string(index=False))

    print("\nDISTANCE INTERVALS BETWEEN TURNS")
    print(turns_df.to_string(index=False))
    print(f"Final Straight Segment (Post Last Turn -> End): {final_straight_len} m")