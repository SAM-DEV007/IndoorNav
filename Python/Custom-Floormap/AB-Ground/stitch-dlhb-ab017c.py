import ast
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.distance import cdist

def rotate_points(x_pos, y_pos, theta):
    x_rot = x_pos * np.cos(theta) - y_pos * np.sin(theta)
    y_rot = x_pos * np.sin(theta) + y_pos * np.cos(theta)

    return x_rot, y_rot

def add_door_coordinates(rooms_df, door_offset=1.1):
    rooms_df = rooms_df.copy()

    def door_position(x_col, y_col, tx_col, ty_col):
        vx = rooms_df[x_col] - rooms_df[tx_col]
        vy = rooms_df[y_col] - rooms_df[ty_col]
        distance = np.hypot(vx, vy)
        valid = distance > 0.01
        door_x = rooms_df[x_col].copy()
        door_y = rooms_df[y_col].copy()
        door_x.loc[valid] -= vx.loc[valid] / distance.loc[valid] * door_offset
        door_y.loc[valid] -= vy.loc[valid] / distance.loc[valid] * door_offset
        return np.round(door_x, 2), np.round(door_y, 2)

    calculated_values = [
        ("Door_X_rot", "Door_Y_rot", "Coords_X_rot", "Coords_Y_rot", "Trajectory_X_rot", "Trajectory_Y_rot"),
        ("Door_X", "Door_Y", "Coords_X", "Coords_Y", "Trajectory_X", "Trajectory_Y"),
    ]
    for door_x_col, door_y_col, x_col, y_col, tx_col, ty_col in calculated_values:
        calculated_x, calculated_y = door_position(x_col, y_col, tx_col, ty_col)
        if door_x_col not in rooms_df:
            rooms_df[door_x_col], rooms_df[door_y_col] = calculated_x, calculated_y
        else:
            missing = rooms_df[door_x_col].isna() | rooms_df[door_y_col].isna()
            rooms_df.loc[missing, door_x_col] = calculated_x.loc[missing]
            rooms_df.loc[missing, door_y_col] = calculated_y.loc[missing]
    return rooms_df

def ab015_correct_washrooms(path, offset_m = 2.5):
    dist_df = pd.read_csv(path / "pdr_distance_log.csv").set_index("Step")
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv").iloc[:-1].copy()

    if rooms_df[rooms_df['Room_ID'] == 'AB018 Washroom'].empty:
        return # No correction needed if the washrooms are already present

    original_step = rooms_df[rooms_df['Room_ID'] == 'AB018 Washroom'][['Matched_Step']].values[0][0]
    modified_step = original_step - 3

    data = dist_df.loc[modified_step]
    rx, ry = data["X_m"], data["Y_m"]
    heading = np.deg2rad(data["Gyro_Heading_Unwrapped_deg"])

    angle_shift = np.pi / 2
    door_x = rx + offset_m * np.cos(heading + angle_shift)
    door_y = ry + offset_m * np.sin(heading + angle_shift)

    new_washroom_row = {
        "Room_ID": "AB018 A - Ladies Washroom",
        "Time_s": "MANUAL",
        "Matched_Step": modified_step,
        "Brightness": "MANUAL",
        "Direction": "Left",
        "Coords": str([(door_x, door_y)]),
        "Trajectory_X": rx,
        "Trajectory_Y": ry,
    }

    rooms_df = pd.concat([rooms_df, pd.DataFrame([new_washroom_row])], ignore_index=True)
    rooms_df.loc[rooms_df['Room_ID'] == 'AB018 Washroom', 'Room_ID'] = 'AB018 B - Gents Washroom'

    rooms_df.to_csv(path / "pdr_rooms_log.csv", index=False)

    return rooms_df

def process_data(path):
    dist_df = pd.read_csv(path / "pdr_distance_log.csv")

    initial_heading = dist_df["Magnetic_Heading_deg"].iloc[0]
    theta = np.deg2rad(90.0 - initial_heading)

    # Rotate the coordinates based on the initial heading
    x_rot, y_rot = rotate_points(dist_df["X_m"].values, dist_df["Y_m"].values, theta)
    dist_df["X_rot"] = np.round(x_rot, 2)
    dist_df["Y_rot"] = np.round(y_rot, 2)

    dist_df["X_m"] = np.round(dist_df["X_m"], 2)
    dist_df["Y_m"] = np.round(dist_df["Y_m"], 2)

    # Read the rooms data and rotate the coordinates
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv")

    rooms_df["Trajectory_X"] = np.round(rooms_df["Trajectory_X"], 2)
    rooms_df["Trajectory_Y"] = np.round(rooms_df["Trajectory_Y"], 2)

    traj_x_rot, traj_y_rot = rotate_points(rooms_df["Trajectory_X"].values, rooms_df["Trajectory_Y"].values, theta)
    rooms_df["Trajectory_X_rot"] = np.round(traj_x_rot, 2)
    rooms_df["Trajectory_Y_rot"] = np.round(traj_y_rot, 2)

    coords_list = [eval(row)[0] for row in rooms_df["Coords"]]
    x, y = np.array(coords_list).T

    x_r_rot, y_r_rot = rotate_points(np.array(x), np.array(y), theta)
    rooms_df["Coords_X_rot"] = np.round(x_r_rot, 2)
    rooms_df["Coords_Y_rot"] = np.round(y_r_rot, 2)

    rooms_df["Coords_X"] = np.round(x, 2)
    rooms_df["Coords_Y"] = np.round(y, 2)

    rooms_df.drop(columns=["Coords"], inplace=True)

    return dist_df, rooms_df

def load_unified_data(dir_path, version):
    dist_df = pd.read_csv(dir_path / f"unified_fp_distance_{version}.csv")
    rooms_df = pd.read_csv(dir_path / f"unified_fp_rooms_{version}.csv")

    return dist_df, rooms_df

def stitch_floorplan(ab017_dist_df, ab017_rooms_df, dlhb_rooms_df):
    # Stitching ab017c to dlhb unified

    # Ab017c intersect point - End step
    # Dlhb intersect point - Infront of Ab 023

    ab017_intersect = ab017_dist_df.iloc[-1][["X_rot", "Y_rot"]].values
    dlhb_intersect = dlhb_rooms_df[(dlhb_rooms_df["Room_ID"] == "Ab 023") & (dlhb_rooms_df["Path_ID"] == "DLHB")][["Trajectory_X_rot", "Trajectory_Y_rot"]].values[0]

    # Calculate the offset between the two intersect points
    offset = dlhb_intersect - ab017_intersect

    # Apply the offset to the ab017 dataset
    ab017_dist_df["X_rot"] = ab017_dist_df["X_rot"] + offset[0]
    ab017_dist_df["Y_rot"] = ab017_dist_df["Y_rot"] + offset[1]

    ab017_rooms_df["Trajectory_X_rot"] = ab017_rooms_df["Trajectory_X_rot"] + offset[0]
    ab017_rooms_df["Trajectory_Y_rot"] = ab017_rooms_df["Trajectory_Y_rot"] + offset[1]
    ab017_rooms_df["Coords_X_rot"] = ab017_rooms_df["Coords_X_rot"] + offset[0]
    ab017_rooms_df["Coords_Y_rot"] = ab017_rooms_df["Coords_Y_rot"] + offset[1]

    return ab017_dist_df, ab017_rooms_df

def drop_unaligned_rows(dist_df, rooms_df):
    dist_df.drop(columns=["X_m", "Y_m"], inplace=True)
    rooms_df.drop(columns=["Trajectory_X", "Trajectory_Y", "Coords_X", "Coords_Y"], inplace=True)

    return dist_df, rooms_df

def generate_unified_floorplan(ab017_dist_path, ab017_rooms_path, unified_dist_path, unified_rooms_path, output_dir, new_path_id='AB017'):
    # Load the new corridor data and the existing unified floorplan
    ab017_dist = pd.read_csv(ab017_dist_path)
    ab017_rooms = pd.read_csv(ab017_rooms_path)
    unified_dist = pd.read_csv(unified_dist_path)
    unified_rooms = pd.read_csv(unified_rooms_path)

    ab017_dist['Path_ID'] = new_path_id
    ab017_rooms['Path_ID'] = new_path_id

    # Initialize intersection columns on the new path
    ab017_dist['Is_Intersection'] = False
    ab017_dist['Intersection_ID'] = None
    ab017_dist['Connected_Path'] = None
    ab017_dist['Connected_Step'] = None

    # Ensure existing unified dataset retains its prior intersections
    for col in ['Is_Intersection', 'Intersection_ID', 'Connected_Path', 'Connected_Step']:
        if col not in unified_dist.columns:
            unified_dist[col] = False if col == 'Is_Intersection' else None

    coords_new = ab017_dist[['X_rot', 'Y_rot']].values
    coords_uni = unified_dist[['X_rot', 'Y_rot']].values

    dists = cdist(coords_new, coords_uni)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    new_idx, uni_idx = min_idx[0], min_idx[1]

    # Extract metadata from the matched step in the unified floorplan
    intersected_row = unified_dist.iloc[uni_idx]
    connected_path = intersected_row['Path_ID']
    connected_step = int(intersected_row['Step'])
    new_step = int(ab017_dist.iloc[new_idx]['Step'])

    junction_id = 'Junction'

    # Populate reciprocal intersection links
    ab017_dist.loc[new_idx, 'Is_Intersection'] = True
    ab017_dist.loc[new_idx, 'Intersection_ID'] = junction_id
    ab017_dist.loc[new_idx, 'Connected_Path'] = connected_path
    ab017_dist.loc[new_idx, 'Connected_Step'] = connected_step

    unified_dist.loc[uni_idx, 'Is_Intersection'] = True
    unified_dist.loc[uni_idx, 'Intersection_ID'] = junction_id
    unified_dist.loc[uni_idx, 'Connected_Path'] = new_path_id
    unified_dist.loc[uni_idx, 'Connected_Step'] = new_step

    # Concatenate datasets
    final_unified_dist = pd.concat([unified_dist, ab017_dist], ignore_index=True)

    # Filter corridor markers from the new rooms file and merge
    ab017_rooms_clean = ab017_rooms[~ab017_rooms['Room_ID'].str.contains('corridor', case=False, na=False)].copy()
    final_unified_rooms = pd.concat([unified_rooms, ab017_rooms_clean], ignore_index=True)
    final_unified_rooms = add_door_coordinates(final_unified_rooms)

    # Save updated unified datasets
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    final_unified_dist.to_csv(output_path / 'unified_fp_distance_2.csv', index=False)
    final_unified_rooms.to_csv(output_path / 'unified_fp_rooms_2.csv', index=False)

    return final_unified_dist, final_unified_rooms

def render_floorplan(unified_dist, unified_rooms, output_path):
    back_gate = unified_dist[(unified_dist['Path_ID'] == 'BGFG') & (unified_dist['Step'] == 1)]
    bg_x = back_gate['X_rot'].iloc[0]
    bg_y = back_gate['Y_rot'].iloc[0]

    intersections = unified_dist[unified_dist['Is_Intersection'] == True]

    # Both clean floorplan (show_axes=False) and coordinate grid (show_axes=True)
    for show_axes in [False, True]:
        fig, ax = plt.subplots(figsize=(14, 10))

        # Hallway Corridors (Outer wall border + Inner walkway floor)
        for path_id, group in unified_dist.groupby('Path_ID'):
            ax.plot(group['X_rot'], group['Y_rot'], color='#95A5A6', linewidth=16, solid_capstyle='round', zorder=1)
            ax.plot(group['X_rot'], group['Y_rot'], color='#ECF0F1', linewidth=12, solid_capstyle='round', zorder=2)

        # Intersection Marker
        if not intersections.empty:
            j_y = intersections['Y_rot'].iloc[::2]
            j_x = intersections['X_rot'].iloc[::2]
            ax.scatter(j_x, j_y, color='yellow', marker='o', s=120, edgecolor='black', linewidth=1.5, zorder=4, label='Intersection')

        # Room Door Blocks
        ax.scatter(unified_rooms['Coords_X_rot'], unified_rooms['Coords_Y_rot'], color='#3498DB', marker='s', s=140, edgecolor='#2C3E50', linewidth=1.5, zorder=4, label='Rooms')

        door_x = unified_rooms["Door_X_rot"]
        door_y = unified_rooms["Door_Y_rot"]

        ax.scatter(door_x, door_y, color="purple", marker="o", s=60, edgecolor="white", linewidth=0.9, zorder=5, label="Entrances")

        # Outward Dynamic Label Placement
        for _, r in unified_rooms.iterrows():
            cx, cy = r['Coords_X_rot'], r['Coords_Y_rot']
            tx, ty = r['Trajectory_X_rot'], r['Trajectory_Y_rot']
            vx = cx - tx
            vy = cy - ty

            # Determine outward push direction based on corridor orientation
            if abs(vx) >= abs(vy):
                # Vertical hallway segment means push outward left or right
                if vx >= 0:
                    ha, va = 'left', 'center'
                    text_x, text_y = cx + 1.0, cy
                else:
                    ha, va = 'right', 'center'
                    text_x, text_y = cx - 1.0, cy
            else:
                # Horizontal hallway segment means push outward top or bottom
                if vy >= 0:
                    ha, va = 'center', 'bottom'
                    text_x, text_y = cx, cy + 1.0
                else:
                    ha, va = 'center', 'top'
                    text_x, text_y = cx, cy - 1.0

            ax.text(text_x, text_y, str(r['Room_ID']), fontsize=7.5, color='#2C3E50', weight='bold', ha=ha, va=va, zorder=5,
                bbox=dict(
                    facecolor='white',
                    alpha=0.8,
                    edgecolor='none',
                    boxstyle='round,pad=0.2',
                ),
            )

        ax.scatter(0, 0, color='red', marker='o', s=250, edgecolor='black', zorder=6, label='Front Gate')
        ax.scatter(bg_x, bg_y, color='black', marker='o', s=250, edgecolor='black', zorder=6, label='Back Gate')

        ax.set_aspect('equal')

        if not show_axes:
            ax.axis('off')
            ax.set_title('AB Ground Floor', fontsize=15, weight='bold', pad=20)
            ax.legend(loc='lower right', frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / 'floorplan_2.png', dpi=300, bbox_inches='tight')
            plt.close()
        else:
            ax.set_xlabel('X Position (meters)', fontsize=11, weight='bold', labelpad=8)
            ax.set_ylabel('Y Position (meters)', fontsize=11, weight='bold', labelpad=8)
            ax.grid(True, linestyle='--', alpha=0.5, color='#BDC3C7')
            ax.axhline(0, color='#7F8C8D', linestyle=':', linewidth=0.8, alpha=0.7)
            ax.axvline(0, color='#7F8C8D', linestyle=':', linewidth=0.8, alpha=0.7)
            ax.set_title('AB Ground Floor (With Coordinates)', fontsize=15, weight='bold', pad=20)
            ax.legend(loc='lower right', frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / 'floorplan_coords_2.png', dpi=300, bbox_inches='tight')
            plt.close()

def save_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "fp_distance.csv", index=False)
    add_door_coordinates(rooms_df).to_csv(path / "fp_rooms.csv", index=False)

def save_stitched_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "stitched_fp_distance.csv", index=False)
    add_door_coordinates(rooms_df).to_csv(path / "stitched_fp_rooms.csv", index=False)

if __name__ == "__main__":
    raw_dir = Path(__file__).parent.resolve() / "Raw"
    ab017_dir = raw_dir / "G-AB017C-AB020TIHAN"

    save_dir = Path(__file__).parent.resolve() / "Processed"
    ab017_save_dir = save_dir / "G-AB017C-AB020TIHAN"
    ab017_save_dir.mkdir(parents=True, exist_ok=True)
    
    ab015_correct_washrooms(ab017_dir)

    ab017_dist_df, ab017_rooms_df = process_data(ab017_dir)
    dlhb_dist_df, dlhb_rooms_df = load_unified_data(save_dir, 1) # Unified data load for DLHB

    ab017_stitched_dist_df, ab017_stitched_rooms_df = stitch_floorplan(ab017_dist_df.copy(), ab017_rooms_df.copy(), dlhb_rooms_df.copy())

    save_data(ab017_dist_df, ab017_rooms_df, ab017_save_dir)
    save_stitched_data(ab017_stitched_dist_df, ab017_stitched_rooms_df, ab017_save_dir)

    unified_dist, unified_rooms = generate_unified_floorplan(ab017_save_dir / "stitched_fp_distance.csv", ab017_save_dir / "stitched_fp_rooms.csv",
                               save_dir / "unified_fp_distance_1.csv", save_dir / "unified_fp_rooms_1.csv",
                               save_dir)
    render_floorplan(unified_dist, unified_rooms, save_dir)