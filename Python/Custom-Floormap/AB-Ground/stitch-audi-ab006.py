import shutil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path
from scipy.spatial.distance import cdist

def rotate_points(x_pos, y_pos, theta):
    x_rot = x_pos * np.cos(theta) - y_pos * np.sin(theta)
    y_rot = x_pos * np.sin(theta) + y_pos * np.cos(theta)

    return x_rot, y_rot

def correct_room_names(path):
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv")

    if rooms_df[rooms_df['Room_ID'] == 'W-AB-009'].empty:
            return # No correction needed

    # Correct room names for consistency
    rooms_df.loc[rooms_df['Room_ID'] == 'W-AB-009', 'Room_ID'] = 'AB009 Washroom'
    rooms_df.loc[rooms_df['Room_ID'] == 'S-AB-008', 'Room_ID'] = 'AB008 Storage'
    rooms_df.loc[rooms_df['Room_ID'] == 'CS-AB-007', 'Room_ID'] = 'AB007 Computer Science'
    rooms_df.loc[rooms_df['Room_ID'] == 'B-AB-006', 'Room_ID'] = 'AB006 Bank'

    rooms_df.to_csv(path / "pdr_rooms_log.csv", index=False)

def process_data(path, target_room_offset=2.50):
    dist_df = pd.read_csv(path / "pdr_distance_log.csv")
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv")

    initial_heading = dist_df["Magnetic_Heading_deg"].iloc[0]
    theta = np.deg2rad(90.0 - initial_heading)

    # Rotate distance coordinates
    x_rot, y_rot = rotate_points(
        dist_df["X_m"].values, dist_df["Y_m"].values, theta
    )
    dist_df["X_rot"] = np.round(x_rot, 2)
    dist_df["Y_rot"] = np.round(y_rot, 2)
    dist_df["X_m"] = np.round(dist_df["X_m"], 2)
    dist_df["Y_m"] = np.round(dist_df["Y_m"], 2)

    # Rotate room trajectory centers
    traj_x_rot, traj_y_rot = rotate_points(
        rooms_df["Trajectory_X"].values, rooms_df["Trajectory_Y"].values, theta
    )
    rooms_df["Trajectory_X_rot"] = np.round(traj_x_rot, 2)
    rooms_df["Trajectory_Y_rot"] = np.round(traj_y_rot, 2)
    rooms_df["Trajectory_X"] = np.round(rooms_df["Trajectory_X"], 2)
    rooms_df["Trajectory_Y"] = np.round(rooms_df["Trajectory_Y"], 2)

    # Parse and rotate room door coordinates
    coords_list = [eval(row)[0] for row in rooms_df["Coords"]]
    x, y = np.array(coords_list).T

    x_r_rot, y_r_rot = rotate_points(np.array(x), np.array(y), theta)
    rooms_df["Coords_X_rot"] = np.round(x_r_rot, 2)
    rooms_df["Coords_Y_rot"] = np.round(y_r_rot, 2)
    rooms_df["Coords_X"] = np.round(x, 2)
    rooms_df["Coords_Y"] = np.round(y, 2)

    # Standardize room box offset distance to match other paths (2.50m)
    if target_room_offset is not None:
        vx = rooms_df["Coords_X_rot"] - rooms_df["Trajectory_X_rot"]
        vy = rooms_df["Coords_Y_rot"] - rooms_df["Trajectory_Y_rot"]
        curr_dists = np.hypot(vx, vy)
        scale = np.where(curr_dists > 0.01, target_room_offset / curr_dists, 1.0)
        rooms_df["Coords_X_rot"] = np.round(
            rooms_df["Trajectory_X_rot"] + vx * scale, 2
        )
        rooms_df["Coords_Y_rot"] = np.round(
            rooms_df["Trajectory_Y_rot"] + vy * scale, 2
        )

        vx_raw = rooms_df["Coords_X"] - rooms_df["Trajectory_X"]
        vy_raw = rooms_df["Coords_Y"] - rooms_df["Trajectory_Y"]
        curr_dists_raw = np.hypot(vx_raw, vy_raw)
        scale_raw = np.where(
            curr_dists_raw > 0.01, target_room_offset / curr_dists_raw, 1.0
        )
        rooms_df["Coords_X"] = np.round(
            rooms_df["Trajectory_X"] + vx_raw * scale_raw, 2
        )
        rooms_df["Coords_Y"] = np.round(
            rooms_df["Trajectory_Y"] + vy_raw * scale_raw, 2
        )

    rooms_df.drop(columns=["Coords"], inplace=True)
    return dist_df, rooms_df

def save_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "fp_distance.csv", index=False)
    rooms_df.to_csv(path / "fp_rooms.csv", index=False)

def save_stitched_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "stitched_fp_distance.csv", index=False)
    rooms_df.to_csv(path / "stitched_fp_rooms.csv", index=False)

def stitch_floorplan(unified_dist_df, unified_rooms_df, new_dist_df, new_rooms_df, start_path_id="BANK", start_landmark=r"right.*corridor", end_landmark=r"ab\s*003"):
    dist_df = new_dist_df.copy()
    rooms_df = new_rooms_df.copy()

    start_match = unified_rooms_df[(unified_rooms_df["Path_ID"] == start_path_id) & (unified_rooms_df["Room_ID"].str.contains(start_landmark, case=False, na=False))]
    t_start_target = np.array([start_match.iloc[0]["Trajectory_X_rot"], start_match.iloc[0]["Trajectory_Y_rot"]])

    step_1_row = dist_df[dist_df["Step"] == 1]

    p_start_new = np.array([step_1_row.iloc[0]["X_rot"], step_1_row.iloc[0]["Y_rot"]])
    start_offset = t_start_target - p_start_new

    dist_df["X_rot"] = dist_df["X_rot"] + start_offset[0]
    dist_df["Y_rot"] = dist_df["Y_rot"] + start_offset[1]

    for column, value in [
        ("Trajectory_X_rot", start_offset[0]),
        ("Trajectory_Y_rot", start_offset[1]),
        ("Coords_X_rot", start_offset[0]),
        ("Coords_Y_rot", start_offset[1]),
    ]:
        rooms_df[column] = rooms_df[column] + value

    end_match = unified_rooms_df[unified_rooms_df["Room_ID"].str.contains(end_landmark, case=False, na=False)]

    t_end_target = np.array([end_match.iloc[0]["Trajectory_X_rot"], end_match.iloc[0]["Trajectory_Y_rot"],])
    end_point = dist_df[["X_rot", "Y_rot"]].iloc[-1].to_numpy(float)
    end_offset = t_end_target - end_point
    alphas = np.linspace(0.0, 1.0, len(dist_df))

    # Ease the correction in and out so the two connected ends remain
    # visually aligned without a sharp kink at either junction.
    correction = (alphas**2)[:, None] * end_offset
    dist_df[["X_rot", "Y_rot"]] = np.round(dist_df[["X_rot", "Y_rot"]].to_numpy(float) + correction, 2)

    original_room_points = rooms_df[["Trajectory_X_rot", "Trajectory_Y_rot"]].to_numpy(float)
    room_steps = rooms_df["Matched_Step"].to_numpy(float)
    room_alphas = np.clip((room_steps - float(dist_df["Step"].iloc[0]))  / (float(dist_df["Step"].iloc[-1]) - float(dist_df["Step"].iloc[0])), 0.0, 1.0)

    room_correction = (room_alphas**2)[:, None] * end_offset
    rooms_df[["Trajectory_X_rot", "Trajectory_Y_rot"]] = np.round(original_room_points + room_correction, 2)
    rooms_df[["Coords_X_rot", "Coords_Y_rot"]] = np.round(rooms_df[["Coords_X_rot", "Coords_Y_rot"]].to_numpy(float) + room_correction, 2)

    return dist_df, rooms_df

def detect_and_stitch_intersections(dist_df, dist_threshold=1.5, fixed_path_id="AUDI"):
    dist_df = dist_df.copy()
    dist_df["Is_Intersection"] = False
    dist_df["Intersection_ID"] = None
    dist_df["Connected_Path"] = None
    dist_df["Connected_Step"] = None

    path_ids = [p for p in dist_df["Path_ID"].dropna().unique()]
    junction_counter = 1

    for i in range(len(path_ids)):
        for j in range(i + 1, len(path_ids)):
            p1, p2 = path_ids[i], path_ids[j]

            df1 = dist_df[dist_df["Path_ID"] == p1]
            df2 = dist_df[dist_df["Path_ID"] == p2]

            coords1 = df1[["X_rot", "Y_rot"]].values
            coords2 = df2[["X_rot", "Y_rot"]].values

            if len(coords1) == 0 or len(coords2) == 0:
                continue

            dists = cdist(coords1, coords2)
            min_dist = np.min(dists)

            if min_dist <= dist_threshold:
                idx1, idx2 = np.unravel_index(np.argmin(dists), dists.shape)
                actual_idx1 = df1.index[idx1]
                actual_idx2 = df2.index[idx2]

                j_id = f"Junction_{junction_counter}"
                junction_counter += 1

                step1 = int(dist_df.loc[actual_idx1, "Step"])
                step2 = int(dist_df.loc[actual_idx2, "Step"])

                if p1 == fixed_path_id and p2 != fixed_path_id:
                    dist_df.loc[actual_idx2, ["X_rot", "Y_rot"]] = dist_df.loc[actual_idx1, ["X_rot", "Y_rot"]].values
                elif p2 == fixed_path_id and p1 != fixed_path_id:
                    dist_df.loc[actual_idx1, ["X_rot", "Y_rot"]] = dist_df.loc[actual_idx2, ["X_rot", "Y_rot"]].values
                elif min_dist > 0 and min_dist <= 0.05:
                    dist_df.loc[actual_idx1, ["X_rot", "Y_rot"]] = dist_df.loc[actual_idx2, ["X_rot", "Y_rot"]].values

                dist_df.loc[actual_idx1, "Is_Intersection"] = True
                dist_df.loc[actual_idx1, "Intersection_ID"] = j_id
                dist_df.loc[actual_idx1, "Connected_Path"] = p2
                dist_df.loc[actual_idx1, "Connected_Step"] = step2

                dist_df.loc[actual_idx2, "Is_Intersection"] = True
                dist_df.loc[actual_idx2, "Intersection_ID"] = j_id
                dist_df.loc[actual_idx2, "Connected_Path"] = p1
                dist_df.loc[actual_idx2, "Connected_Step"] = step1

    return dist_df

def calculate_cumulative_distance_rot(dist_df):
    df = dist_df.copy()
    cum_dist = pd.Series(0.0, index=df.index, dtype=float)

    for _, group in df.groupby("Path_ID", sort=False):
        if group.empty:
            continue
        x = group["X_rot"].to_numpy(dtype=float)
        y = group["Y_rot"].to_numpy(dtype=float)
        dx = np.diff(x, prepend=x[0])
        dy = np.diff(y, prepend=y[0])
        cum_dist.loc[group.index] = np.cumsum(np.hypot(dx, dy))

    df["Cumulative_Distance_Rot_m"] = np.round(cum_dist, 2)

    # Update position
    cols = [c for c in df.columns if c not in ("Cumulative_Distance_Rot_m", "Path_ID")]
    y_idx = cols.index("Y_rot")
    insert_cols = ["Cumulative_Distance_Rot_m"]
    if "Path_ID" in df.columns:
        insert_cols.append("Path_ID")
    cols[y_idx + 1 : y_idx + 1] = insert_cols

    return df[cols]

def generate_unified_floorplan(unified_dist_path, unified_rooms_path, new_dist_df, new_rooms_df, output_dir, new_path_id="BANK"):
    unified_dist = pd.read_csv(unified_dist_path)
    unified_rooms = pd.read_csv(unified_rooms_path)

    # Drop 'Right turn corridor' placeholder now that the actual corridor branch is attached
    unified_rooms = unified_rooms[~unified_rooms["Room_ID"].str.contains(r"right.*corridor", case=False, na=False)].copy()

    new_dist = new_dist_df.copy()
    new_rooms = new_rooms_df.copy()

    new_dist["Path_ID"] = new_path_id
    new_rooms["Path_ID"] = new_path_id

    # Combine distance data and register junctions
    combined_dist = pd.concat([unified_dist, new_dist], ignore_index=True)
    unified_dist_new = detect_and_stitch_intersections(combined_dist, dist_threshold=1.5, fixed_path_id="AUDI")

    # Filter duplicate landmark placeholders from the new path
    def filter_new_rooms(row):
        r_name = str(row["Room_ID"]).lower()
        if "audi door" in r_name or "discussion" in r_name:
            return False
        if any(k in r_name for k in ["corridor", "right turn", "left turn"]):
            return False
        return True

    new_rooms_clean = new_rooms[new_rooms.apply(filter_new_rooms, axis=1)].copy()
    unified_rooms_new = pd.concat([unified_rooms, new_rooms_clean], ignore_index=True)

    unified_dist_new = calculate_cumulative_distance_rot(unified_dist_new)

    unified_dist_new.to_csv(output_dir / "unified_fp_distance_4.csv", index=False)
    unified_rooms_new.to_csv(output_dir / "unified_fp_rooms_4.csv", index=False)

    return unified_dist_new, unified_rooms_new

def render_floorplan(unified_dist, unified_rooms, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    back_gate = unified_dist[(unified_dist["Path_ID"] == "BGFG") & (unified_dist["Step"] == 1)]
    bg_x = back_gate["X_rot"].iloc[0] if not back_gate.empty else -17.77
    bg_y = back_gate["Y_rot"].iloc[0] if not back_gate.empty else 56.14

    plot_intersections = unified_dist[unified_dist["Is_Intersection"] == True].drop_duplicates(subset=["Intersection_ID"]).copy()

    for show_axes in [False, True]:
        fig, ax = plt.subplots(figsize=(15, 11))

        # Hallways (Border wall + Walkway)
        for path_id, group in unified_dist.groupby("Path_ID"):
            ax.plot(group["X_rot"], group["Y_rot"], color="#95A5A6", linewidth=16, solid_capstyle="round", zorder=1)
            ax.plot(group["X_rot"], group["Y_rot"], color="#ECF0F1", linewidth=12, solid_capstyle="round", zorder=2)

        # Intersections
        if not plot_intersections.empty:
            ax.scatter(plot_intersections["X_rot"], plot_intersections["Y_rot"], color="yellow", marker="o", s=120, edgecolor="black", linewidth=1.5, zorder=4, label="Intersection")

        # Room Door Blocks
        ax.scatter(unified_rooms["Coords_X_rot"], unified_rooms["Coords_Y_rot"], color="#3498DB", marker="s", s=140, edgecolor="#2C3E50", linewidth=1.5, zorder=4, label="Rooms")

        # Door markers sit just inside each room, toward the hallway.
        room_to_path_x = unified_rooms["Coords_X_rot"] - unified_rooms["Trajectory_X_rot"]
        room_to_path_y = unified_rooms["Coords_Y_rot"] - unified_rooms["Trajectory_Y_rot"]

        room_to_path_distance = np.hypot(room_to_path_x, room_to_path_y)
        valid_direction = room_to_path_distance > 0.01

        door_x = unified_rooms["Coords_X_rot"].copy()
        door_y = unified_rooms["Coords_Y_rot"].copy()

        door_x.loc[valid_direction] += (-room_to_path_x.loc[valid_direction] / room_to_path_distance.loc[valid_direction] * 1.1)
        door_y.loc[valid_direction] += (-room_to_path_y.loc[valid_direction] / room_to_path_distance.loc[valid_direction] * 1.1)

        ax.scatter(door_x, door_y, color="purple", marker="o", s=60, edgecolor="white", linewidth=0.9, zorder=5, label="Entrances")

        # Dynamic Label Placement with Collision Avoidance
        for _, r in unified_rooms.iterrows():
            cx, cy = r["Coords_X_rot"], r["Coords_Y_rot"]
            tx, ty = r["Trajectory_X_rot"], r["Trajectory_Y_rot"]
            vx, vy = cx - tx, cy - ty

            room_name_clean = str(r["Room_ID"]).strip().lower()

            # Custom label placement for specific rooms to avoid overlap
            if room_name_clean == "stair":
                ha, va = ("right", "center")
                text_x = cx - 1.0
                text_y = cy
            elif room_name_clean == "ab006 bank":
                ha, va = ("center", "bottom")
                text_x = cx
                text_y = cy + 1.0
            elif room_name_clean == "ab 005 electrical panel":
                ha, va = ("center", "top")
                text_x = cx
                text_y = cy + 2.0
            elif abs(vx) >= abs(vy):
                ha, va = ("left", "center") if vx >= 0 else ("right", "center")
                text_x = cx + (1.0 if vx >= 0 else -1.0)
                text_y = cy
            else:
                ha, va = ("center", "bottom") if vy >= 0 else ("center", "top")
                text_x = cx
                text_y = cy + (1.0 if vy >= 0 else -1.0)

            ax.text(text_x, text_y, str(r["Room_ID"]), fontsize=7.2, color="#2C3E50", weight="bold", ha=ha, va=va, zorder=5,
                bbox=dict(
                    facecolor="white",
                    alpha=0.8,
                    edgecolor="none",
                    boxstyle="round,pad=0.2",
                )
            )

        # Front Gate and Back Gate Markers
        ax.scatter(0, 0, color="red", marker="o", s=250, edgecolor="black", zorder=6, label="Front Gate")
        ax.scatter(bg_x, bg_y, color="black", marker="o", s=250, edgecolor="black", zorder=6, label="Back Gate")

        ax.set_aspect("equal")

        if not show_axes:
            ax.axis("off")
            ax.set_title("AB Ground Floor", fontsize=15, weight="bold", pad=20)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / "floorplan_4.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            ax.set_xlabel("X Position (meters)", fontsize=11, weight="bold", labelpad=8)
            ax.set_ylabel("Y Position (meters)", fontsize=11, weight="bold", labelpad=8)
            ax.grid(True, linestyle="--", alpha=0.5, color="#BDC3C7")
            ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axvline(0, color="#7F8C8D", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_title("AB Ground Floor (Coordinates)", fontsize=15, weight="bold", pad=20)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / "floorplan_coords_4.png", dpi=300, bbox_inches="tight")
            plt.close()

def remove_useless_columns(dist_path, rooms_path):
    dist_df = pd.read_csv(dist_path)
    rooms_df = pd.read_csv(rooms_path)

    dist_df.drop(columns=["Seconds_Elapsed", "Stride_Length_m", "Speed_m_s", "Cumulative_Distance_m", "X_m", "Y_m", "Cadence_Adaptive_Constant_K"], inplace=True, errors="ignore")
    rooms_df.drop(columns=["Time_s", "Brightness", "Trajectory_X", "Trajectory_Y", "Coords_X", "Coords_Y"], inplace=True, errors="ignore")

    dist_df.to_csv(dist_path, index=False)
    rooms_df.to_csv(rooms_path, index=False)

def transfer_final_data(source_dir, target_dir):
    source_dist_path = source_dir / "unified_fp_distance_4.csv"
    source_rooms_path = source_dir / "unified_fp_rooms_4.csv"
    source_floorplan_path = source_dir / "floorplan_4.png"
    source_floorplan_coords_path = source_dir / "floorplan_coords_4.png"

    target_dist_path = target_dir / "ab_ground_unified_fp_distance.csv"
    target_rooms_path = target_dir / "ab_ground_unified_fp_rooms.csv"
    target_floorplan_path = target_dir / "ab_ground_floorplan.png"
    target_floorplan_coords_path = target_dir / "ab_ground_floorplan_coords.png"

    shutil.copy2(source_dist_path, target_dist_path)
    shutil.copy2(source_rooms_path, target_rooms_path)
    shutil.copy2(source_floorplan_path, target_floorplan_path)
    shutil.copy2(source_floorplan_coords_path, target_floorplan_coords_path)

    remove_useless_columns(target_dist_path, target_rooms_path)

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    raw_dir = base_dir / "Raw"
    save_dir = base_dir / "Processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    old_unified_dist_path = save_dir / "unified_fp_distance_3.csv"
    old_unified_rooms_path = save_dir / "unified_fp_rooms_3.csv"

    new_data_name = "G-AB010-AB006"
    new_path_dir = raw_dir / new_data_name
    new_save_dir = save_dir / new_data_name
    new_save_dir.mkdir(parents=True, exist_ok=True)

    correct_room_names(new_path_dir)

    new_dist_df, new_rooms_df = process_data(new_path_dir, target_room_offset=2.50)
    save_data(new_dist_df, new_rooms_df, new_save_dir)

    old_unified_dist_current = pd.read_csv(old_unified_dist_path)
    old_unified_rooms_current = pd.read_csv(old_unified_rooms_path)

    stitched_dist_df, stitched_rooms_df = stitch_floorplan(old_unified_dist_current, old_unified_rooms_current, new_dist_df, new_rooms_df, start_path_id="AUDI", start_landmark=r"right.*corridor", end_landmark=r"ab\s*003")
    save_stitched_data(stitched_dist_df, stitched_rooms_df, new_save_dir)

    unified_dist, unified_rooms = generate_unified_floorplan(old_unified_dist_path, old_unified_rooms_path, stitched_dist_df, stitched_rooms_df, save_dir, new_path_id="BANK")
    render_floorplan(unified_dist, unified_rooms, save_dir)

    transfer_final_data(save_dir, base_dir)