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

    calculated_rot_x, calculated_rot_y = door_position(
        "Coords_X_rot", "Coords_Y_rot", "Trajectory_X_rot", "Trajectory_Y_rot"
    )
    calculated_x, calculated_y = door_position(
        "Coords_X", "Coords_Y", "Trajectory_X", "Trajectory_Y"
    )
    has_saved_door_coordinates = (
        all(column in rooms_df for column in ["Door_X_rot", "Door_Y_rot", "Door_X", "Door_Y"])
        and rooms_df[["Door_X_rot", "Door_Y_rot", "Door_X", "Door_Y"]].notna().any().any()
    )

    for column, values in [
        ("Door_X_rot", calculated_rot_x),
        ("Door_Y_rot", calculated_rot_y),
        ("Door_X", calculated_x),
        ("Door_Y", calculated_y),
    ]:
        if column not in rooms_df:
            rooms_df[column] = values
        else:
            rooms_df.loc[rooms_df[column].isna(), column] = values

    if not has_saved_door_coordinates:
        discussion_room = rooms_df["Room_ID"].eq("Discussion room")
        rooms_df.loc[discussion_room, "Door_X_rot"] = rooms_df.loc[discussion_room, "Coords_X_rot"] + door_offset
        rooms_df.loc[discussion_room, "Door_Y_rot"] = rooms_df.loc[discussion_room, "Coords_Y_rot"]
        rooms_df.loc[discussion_room, "Door_X"] = rooms_df.loc[discussion_room, "Coords_X"] + door_offset
        rooms_df.loc[discussion_room, "Door_Y"] = rooms_df.loc[discussion_room, "Coords_Y"]

    return rooms_df

def add_gate_rooms(rooms_df, dist_df):
    rooms_df = rooms_df.copy()
    gate_rows = []

    gates = [
        ("Front Gate", 0.0, 0.0, 0.0, 0.0, "BGFG", 1),
    ]
    back_gate = dist_df[(dist_df["Path_ID"] == "BGFG") & (dist_df["Step"] == 1)]
    if not back_gate.empty:
        row = back_gate.iloc[0]
        gates.append(("Back Gate", row["X_m"], row["Y_m"], row["X_rot"], row["Y_rot"], "BGFG", 1))

    for room_id, x, y, x_rot, y_rot, path_id, matched_step in gates:
        if rooms_df["Room_ID"].eq(room_id).any():
            continue
        gate_rows.append({
            "Room_ID": room_id,
            "Time_s": "MANUAL",
            "Matched_Step": matched_step,
            "Brightness": "MANUAL",
            "Direction": "",
            "Trajectory_X": x,
            "Trajectory_Y": y,
            "Trajectory_X_rot": x_rot,
            "Trajectory_Y_rot": y_rot,
            "Coords_X_rot": x_rot,
            "Coords_Y_rot": y_rot,
            "Coords_X": x,
            "Coords_Y": y,
            "Path_ID": path_id,
        })

    if gate_rows:
        rooms_df = pd.concat([rooms_df, pd.DataFrame(gate_rows)], ignore_index=True)
    return rooms_df

def dlhb_correct_rooms(path: Path, offset_m: float = 2.5):
    dist_df = pd.read_csv(path / "pdr_distance_log.csv").set_index("Step")
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv").iloc[:-1].copy()

    if rooms_df.iloc[-1]["Room_ID"].lower() == "lift":
        return # No correction needed if the last second room is already "Lift"

    corrections = [
        ("Staircase", 143, "Left"),
        ("Lift", 154, "Left (Both)"),
        ("Ab 022 cabins", 154, "Right (Both)"),
    ]

    new_rows = []
    for room_id, step, direction in corrections:
        step_data = dist_df.loc[step]
        rx, ry = step_data["X_m"], step_data["Y_m"]
        heading = np.deg2rad(step_data["Gyro_Heading_Unwrapped_deg"])

        angle_shift = (
            np.pi / 2 if "left" in direction.lower() else -np.pi / 2
        )
        door_x = rx + offset_m * np.cos(heading + angle_shift)
        door_y = ry + offset_m * np.sin(heading + angle_shift)

        new_rows.append(
            {
                "Room_ID": room_id,
                "Time_s": "MANUAL",
                "Matched_Step": step,
                "Brightness": "MANUAL",
                "Direction": direction,
                "Coords": str([(door_x, door_y)]),
                "Trajectory_X": rx,
                "Trajectory_Y": ry,
            }
        )

    corrected_rooms_df = pd.concat(
        [rooms_df, pd.DataFrame(new_rows)], ignore_index=True
    )
    corrected_rooms_df.to_csv(path / "pdr_rooms_log.csv", index=False)

    return corrected_rooms_df

def process_data(path, reset_coords=False):
    # Reset coords is only for the S-BG-FG dataset
    # Simply means that the coordinates are reset to the origin (0,0) for the S-BG-FG dataset (Front gate is at the origin)

    dist_df = pd.read_csv(path / "pdr_distance_log.csv")

    initial_heading = dist_df["Magnetic_Heading_deg"].iloc[0]
    theta = np.deg2rad(90.0 - initial_heading)

    if reset_coords is True:
        x_fg, y_fg = dist_df["X_m"].iloc[-1], dist_df["Y_m"].iloc[-1]
        dist_df["X_m"] = dist_df["X_m"] - x_fg
        dist_df["Y_m"] = dist_df["Y_m"] - y_fg

    # Rotate the coordinates based on the initial heading
    x_rot, y_rot = rotate_points(dist_df["X_m"].values, dist_df["Y_m"].values, theta)
    dist_df["X_rot"] = np.round(x_rot, 2)
    dist_df["Y_rot"] = np.round(y_rot, 2)

    dist_df["X_m"] = np.round(dist_df["X_m"], 2)
    dist_df["Y_m"] = np.round(dist_df["Y_m"], 2)

    # Read the rooms data and rotate the coordinates
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv")

    if reset_coords is True:
        rooms_df["Trajectory_X"] = rooms_df["Trajectory_X"] - x_fg
        rooms_df["Trajectory_Y"] = rooms_df["Trajectory_Y"] - y_fg

    rooms_df["Trajectory_X"] = np.round(rooms_df["Trajectory_X"], 2)
    rooms_df["Trajectory_Y"] = np.round(rooms_df["Trajectory_Y"], 2)

    traj_x_rot, traj_y_rot = rotate_points(rooms_df["Trajectory_X"].values, rooms_df["Trajectory_Y"].values, theta)
    rooms_df["Trajectory_X_rot"] = np.round(traj_x_rot, 2)
    rooms_df["Trajectory_Y_rot"] = np.round(traj_y_rot, 2)

    coords_list = [eval(row)[0] for row in rooms_df["Coords"]]
    x, y = np.array(coords_list).T

    if reset_coords is True:
        x = x - x_fg
        y = y - y_fg

    x_r_rot, y_r_rot = rotate_points(np.array(x), np.array(y), theta)
    rooms_df["Coords_X_rot"] = np.round(x_r_rot, 2)
    rooms_df["Coords_Y_rot"] = np.round(y_r_rot, 2)

    rooms_df["Coords_X"] = np.round(x, 2)
    rooms_df["Coords_Y"] = np.round(y, 2)

    rooms_df.drop(columns=["Coords"], inplace=True)

    return dist_df, rooms_df

def stitch_floorplan(bgfg_rooms_df, dlhb_dist_df, dlhb_rooms_df):
    # Stitching dlhb to bgfg

    # Bgfg intersect point - Corridor (rooms trajectory)
    # Dlhb intersect point - Discussion room (rooms trajectory) + some offset in steps (5 steps)

    bgfg_intersect = bgfg_rooms_df[bgfg_rooms_df["Room_ID"] == "Corridor"][["Trajectory_X_rot", "Trajectory_Y_rot"]].values[0]

    dlhb_intersect_primary = dlhb_rooms_df[dlhb_rooms_df["Room_ID"] == "Discussion room"][["Matched_Step"]].values[0][0]
    extra_offset = 5 # Shift towards left (in steps) to avoid overlap with the discussion room

    dlhb_intersect = dlhb_dist_df[dlhb_dist_df["Step"] == (dlhb_intersect_primary + extra_offset)][["X_rot", "Y_rot"]].values[0]

    # Calculate the offset between the two intersect points
    offset = bgfg_intersect - dlhb_intersect

    # Apply the offset to the dlhb dataset
    dlhb_dist_df["X_rot"] = dlhb_dist_df["X_rot"] + offset[0]
    dlhb_dist_df["Y_rot"] = dlhb_dist_df["Y_rot"] + offset[1]

    dlhb_rooms_df["Trajectory_X_rot"] = dlhb_rooms_df["Trajectory_X_rot"] + offset[0]
    dlhb_rooms_df["Trajectory_Y_rot"] = dlhb_rooms_df["Trajectory_Y_rot"] + offset[1]
    dlhb_rooms_df["Coords_X_rot"] = dlhb_rooms_df["Coords_X_rot"] + offset[0]
    dlhb_rooms_df["Coords_Y_rot"] = dlhb_rooms_df["Coords_Y_rot"] + offset[1]

    return dlhb_dist_df, dlhb_rooms_df

def drop_unaligned_rows(dist_df, rooms_df):
    dist_df.drop(columns=["X_m", "Y_m"], inplace=True)
    rooms_df.drop(columns=["Trajectory_X", "Trajectory_Y", "Coords_X", "Coords_Y"], inplace=True)

    return dist_df, rooms_df

def generate_unified_floorplan(bgfg_dist_path, bgfg_rooms_path, dlhb_dist_path, dlhb_rooms_path, output_dir):
    bgfg_dist = pd.read_csv(bgfg_dist_path)
    bgfg_rooms = pd.read_csv(bgfg_rooms_path)
    dlhb_dist = pd.read_csv(dlhb_dist_path)
    dlhb_rooms = pd.read_csv(dlhb_rooms_path)

    bgfg_dist['Path_ID'] = 'BGFG'
    dlhb_dist['Path_ID'] = 'DLHB'
    bgfg_rooms['Path_ID'] = 'BGFG'
    dlhb_rooms['Path_ID'] = 'DLHB'

    for df in [bgfg_dist, dlhb_dist]:
        df['Is_Intersection'] = False
        df['Intersection_ID'] = None
        df['Connected_Path'] = None
        df['Connected_Step'] = None

    coords_bg = bgfg_dist[['X_rot', 'Y_rot']].values
    coords_dl = dlhb_dist[['X_rot', 'Y_rot']].values
    dists = cdist(coords_bg, coords_dl)
    min_idx = np.unravel_index(np.argmin(dists), dists.shape)
    bg_idx, dl_idx = min_idx[0], min_idx[1]

    # Populate intersection links (Step 85 on BGFG <-> Step 83 on DLHB)
    junction_id = "Junction"
    bg_step = bgfg_dist.iloc[bg_idx]['Step']
    dl_step = dlhb_dist.iloc[dl_idx]['Step']

    bgfg_dist.loc[bg_idx, 'Is_Intersection'] = True
    bgfg_dist.loc[bg_idx, 'Intersection_ID'] = junction_id
    bgfg_dist.loc[bg_idx, 'Connected_Path'] = 'DLHB'
    bgfg_dist.loc[bg_idx, 'Connected_Step'] = int(dl_step)

    dlhb_dist.loc[dl_idx, 'Is_Intersection'] = True
    dlhb_dist.loc[dl_idx, 'Intersection_ID'] = junction_id
    dlhb_dist.loc[dl_idx, 'Connected_Path'] = 'BGFG'
    dlhb_dist.loc[dl_idx, 'Connected_Step'] = int(bg_step)

    unified_dist = pd.concat([bgfg_dist, dlhb_dist], ignore_index=True)
    unified_rooms = pd.concat([bgfg_rooms, dlhb_rooms], ignore_index=True)

    unified_rooms = unified_rooms[~unified_rooms['Room_ID'].str.contains('corridor', case=False, na=False)].copy()
    unified_rooms = add_gate_rooms(unified_rooms, unified_dist)
    unified_rooms = add_door_coordinates(unified_rooms)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    unified_dist.to_csv(output_path / "unified_fp_distance_1.csv", index=False)
    unified_rooms.to_csv(output_path / "unified_fp_rooms_1.csv", index=False)

    return unified_dist, unified_rooms

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
            j_x = intersections['X_rot'].iloc[0]
            j_y = intersections['Y_rot'].iloc[0]
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
            plt.savefig(output_path / 'floorplan_1.png', dpi=300, bbox_inches='tight')
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
            plt.savefig(output_path / 'floorplan_coords_1.png', dpi=300, bbox_inches='tight')
            plt.close()

def save_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "fp_distance.csv", index=False)
    add_door_coordinates(rooms_df).to_csv(path / "fp_rooms.csv", index=False)

def save_stitched_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "stitched_fp_distance.csv", index=False)
    add_door_coordinates(rooms_df).to_csv(path / "stitched_fp_rooms.csv", index=False)

if __name__ == "__main__":
    raw_dir = Path(__file__).parent.resolve() / "Raw"
    bgfg_dir = raw_dir / "S-BG-FG"
    dlhb_dir = raw_dir / "S-DLAK-HARB-CABINS"

    save_dir = Path(__file__).parent.resolve() / "Processed"
    bgfg_save_dir = save_dir / "S-BG-FG"
    dlhb_save_dir = save_dir / "S-DLAK-HARB-CABINS"

    bgfg_save_dir.mkdir(parents=True, exist_ok=True)
    dlhb_save_dir.mkdir(parents=True, exist_ok=True)

    bgfg_dist_df, bgfg_rooms_df = process_data(bgfg_dir, reset_coords=True)

    dlhb_correct_rooms(dlhb_dir) # Correct the room coordinates (AB022 and Lift)
    dlhb_dist_df, dlhb_rooms_df = process_data(dlhb_dir)

    dlhb_stitched_dist_df, dlhb_stitched_rooms_df = stitch_floorplan(bgfg_rooms_df.copy(), dlhb_dist_df.copy(), dlhb_rooms_df.copy())

    save_data(bgfg_dist_df, bgfg_rooms_df, bgfg_save_dir)
    save_data(dlhb_dist_df, dlhb_rooms_df, dlhb_save_dir)

    save_stitched_data(dlhb_stitched_dist_df, dlhb_stitched_rooms_df, dlhb_save_dir)
    save_stitched_data(bgfg_dist_df, bgfg_rooms_df, bgfg_save_dir)

    unified_dist, unified_rooms = generate_unified_floorplan(bgfg_save_dir / "stitched_fp_distance.csv", bgfg_save_dir / "stitched_fp_rooms.csv",
                               dlhb_save_dir / "stitched_fp_distance.csv", dlhb_save_dir / "stitched_fp_rooms.csv",
                               save_dir)
    render_floorplan(unified_dist, unified_rooms, save_dir)