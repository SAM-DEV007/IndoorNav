import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.spatial.distance import cdist

def rotate_points(x_pos, y_pos, theta):
    x_rot = x_pos * np.cos(theta) - y_pos * np.sin(theta)
    y_rot = x_pos * np.sin(theta) + y_pos * np.cos(theta)
    return x_rot, y_rot

def process_data(path):
    dist_df = pd.read_csv(path / "pdr_distance_log.csv")
    rooms_df = pd.read_csv(path / "pdr_rooms_log.csv")

    initial_heading = dist_df["Magnetic_Heading_deg"].iloc[0]
    theta = np.deg2rad(90.0 - initial_heading)

    # Rotate distance coordinates
    x_rot, y_rot = rotate_points(dist_df["X_m"].values, dist_df["Y_m"].values, theta)
    dist_df["X_rot"] = np.round(x_rot, 2)
    dist_df["Y_rot"] = np.round(y_rot, 2)
    dist_df["X_m"] = np.round(dist_df["X_m"], 2)
    dist_df["Y_m"] = np.round(dist_df["Y_m"], 2)

    # Rotate room trajectory centers
    traj_x_rot, traj_y_rot = rotate_points(rooms_df["Trajectory_X"].values, rooms_df["Trajectory_Y"].values, theta)
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

    # Swap stair and audi backdoor room names
    stair_mask = rooms_df["Room_ID"].str.strip().str.lower() == "stair"
    backdoor_mask = rooms_df["Room_ID"].str.strip().str.lower() == "audi backdoor"

    if stair_mask.any() and backdoor_mask.any():
        s_idx = rooms_df[stair_mask].index[0]
        b_idx = rooms_df[backdoor_mask].index[0]
        stair_name = rooms_df.loc[s_idx, "Room_ID"]
        backdoor_name = rooms_df.loc[b_idx, "Room_ID"]
        rooms_df.loc[s_idx, "Room_ID"] = backdoor_name
        rooms_df.loc[b_idx, "Room_ID"] = stair_name

    rooms_df.drop(columns=["Coords"], inplace=True)

    return dist_df, rooms_df

def save_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "fp_distance.csv", index=False)
    rooms_df.to_csv(path / "fp_rooms.csv", index=False)

def save_stitched_data(dist_df, rooms_df, path):
    dist_df.to_csv(path / "stitched_fp_distance.csv", index=False)
    rooms_df.to_csv(path / "stitched_fp_rooms.csv", index=False)

def shrink_ab017_in_memory(unified_dist_df, unified_rooms_df):
    # Applies the step cuts (30-35 and 50-55) to the AB017 partition entirely in memory.
    # Re-rotates and re-anchors the bottom terminus cleanly to DLHB Step 132 (25.68, 13.75)

    dist_df = unified_dist_df.copy()
    rooms_df = unified_rooms_df.copy()

    ab_mask = dist_df["Path_ID"] == "AB017"
    if not ab_mask.any():
        return dist_df, rooms_df

    ab_dist = dist_df[ab_mask].copy().reset_index(drop=True)
    other_dist = dist_df[~ab_mask].copy()

    # Sequential in-memory step removals (evaluated on raw trajectory coordinates)
    # (start_step, end_step, min_len) - min_len to prevent repeated cuts on already-shrunk data
    cuts = [
        (30, 35, 72),  # Between W-AB018 and AB019
        (50, 55, 66),  # Between AB019 and AB020
    ]

    for start_step, end_step, min_len in cuts:
        if len(ab_dist) < min_len:
            continue

        prev_step = start_step - 1
        x_prev = ab_dist.loc[ab_dist["Step"] == prev_step, "X_m"].values[0]
        y_prev = ab_dist.loc[ab_dist["Step"] == prev_step, "Y_m"].values[0]
        dist_prev = ab_dist.loc[ab_dist["Step"] == prev_step, "Cumulative_Distance_m"].values[0]
        time_prev = ab_dist.loc[ab_dist["Step"] == prev_step, "Seconds_Elapsed"].values[0]

        x_end = ab_dist.loc[ab_dist["Step"] == end_step, "X_m"].values[0]
        y_end = ab_dist.loc[ab_dist["Step"] == end_step, "Y_m"].values[0]
        dist_end = ab_dist.loc[ab_dist["Step"] == end_step, "Cumulative_Distance_m"].values[0]
        time_end = ab_dist.loc[ab_dist["Step"] == end_step, "Seconds_Elapsed"].values[0]

        delta_x = x_end - x_prev
        delta_y = y_end - y_prev
        delta_dist = dist_end - dist_prev
        delta_time = time_end - time_prev
        steps_dropped = end_step - start_step + 1

        after_mask = ab_dist["Step"] > end_step
        ab_dist.loc[after_mask, "X_m"] = np.round(ab_dist.loc[after_mask, "X_m"] - delta_x, 2)
        ab_dist.loc[after_mask, "Y_m"] = np.round(ab_dist.loc[after_mask, "Y_m"] - delta_y, 2)
        ab_dist.loc[after_mask, "Cumulative_Distance_m"] = np.round(ab_dist.loc[after_mask, "Cumulative_Distance_m"] - delta_dist, 2)
        ab_dist.loc[after_mask, "Seconds_Elapsed"] = np.round(ab_dist.loc[after_mask, "Seconds_Elapsed"] - delta_time, 2)

        ab_dist = ab_dist[~ab_dist["Step"].between(start_step, end_step)].copy()
        ab_dist["Step"] = np.arange(1, len(ab_dist) + 1)
        ab_dist.reset_index(drop=True, inplace=True)

        # Shift room step indices in AB017
        ab_r_mask = rooms_df["Path_ID"] == "AB017"
        after_r_mask = ab_r_mask & (rooms_df["Matched_Step"] > end_step)
        rooms_df.loc[after_r_mask, "Matched_Step"] -= steps_dropped

    # Re-rotate AB017 using its original heading
    initial_heading = ab_dist["Magnetic_Heading_deg"].iloc[0]
    theta = np.deg2rad(90.0 - initial_heading)
    x_rot, y_rot = rotate_points(ab_dist["X_m"].values, ab_dist["Y_m"].values, theta)
    ab_dist["X_rot"] = np.round(x_rot, 2)
    ab_dist["Y_rot"] = np.round(y_rot, 2)

    # Re-anchor the bottom step cleanly to DLHB Step 132 (25.68, 13.75)
    dlhb_target_row = other_dist[(other_dist["Path_ID"] == "DLHB") & (other_dist["Step"] == 132)]
    dlhb_target = dlhb_target_row[["X_rot", "Y_rot"]].iloc[0].values if not dlhb_target_row.empty else np.array([25.68, 13.75])

    ab_last_pos = ab_dist[["X_rot", "Y_rot"]].iloc[-1].values
    offset = dlhb_target - ab_last_pos

    ab_dist["X_rot"] = np.round(ab_dist["X_rot"] + offset[0], 2)
    ab_dist["Y_rot"] = np.round(ab_dist["Y_rot"] + offset[1], 2)

    # Update intersection connection in DLHB and AB017
    new_last_step = int(ab_dist.iloc[-1]["Step"])
    ab_dist.loc[len(ab_dist) - 1, "Is_Intersection"] = True
    ab_dist.loc[len(ab_dist) - 1, "Intersection_ID"] = "Junction"
    ab_dist.loc[len(ab_dist) - 1, "Connected_Path"] = "DLHB"
    ab_dist.loc[len(ab_dist) - 1, "Connected_Step"] = 132.0

    dlhb_idx = other_dist[(other_dist["Path_ID"] == "DLHB") & (other_dist["Step"] == 132)].index
    if len(dlhb_idx) > 0:
        other_dist.loc[dlhb_idx, "Connected_Step"] = float(new_last_step)

    # Update AB017 room coordinates to match their shifted steps
    ab_r_mask = rooms_df["Path_ID"] == "AB017"
    for r_i in rooms_df[ab_r_mask].index:
        m_step = int(rooms_df.loc[r_i, "Matched_Step"])
        step_rows = ab_dist[ab_dist["Step"] == m_step]
        if not step_rows.empty:
            tx_rot = step_rows.iloc[0]["X_rot"]
            ty_rot = step_rows.iloc[0]["Y_rot"]
            dx = rooms_df.loc[r_i, "Coords_X_rot"] - rooms_df.loc[r_i, "Trajectory_X_rot"]
            dy = rooms_df.loc[r_i, "Coords_Y_rot"] - rooms_df.loc[r_i, "Trajectory_Y_rot"]
            rooms_df.loc[r_i, "Trajectory_X_rot"] = tx_rot
            rooms_df.loc[r_i, "Trajectory_Y_rot"] = ty_rot
            rooms_df.loc[r_i, "Coords_X_rot"] = np.round(tx_rot + dx, 2)
            rooms_df.loc[r_i, "Coords_Y_rot"] = np.round(ty_rot + dy, 2)

    merged_dist = pd.concat([other_dist, ab_dist], ignore_index=True)
    return merged_dist, rooms_df

def stitch_floorplan(unified_dist_df, unified_rooms_df, new_dist_df, new_rooms_df, bgfg_step_offset_back=7):
    dist_df = new_dist_df.copy()
    rooms_df = new_rooms_df.copy()

    # Local Anchor Point on New Corridor: Discussion room / Left turn corridor
    p1_rows = rooms_df[rooms_df["Room_ID"].str.contains(r"discussion|left turn", case=False, na=False)]
    p1_row = p1_rows.iloc[0] if not p1_rows.empty else rooms_df.iloc[len(rooms_df) // 2]
    p1_new = np.array([p1_row["Trajectory_X_rot"], p1_row["Trajectory_Y_rot"]])

    bgfg_disc = unified_rooms_df[(unified_rooms_df["Path_ID"] == "BGFG") & (unified_rooms_df["Room_ID"].str.contains(r"discussion|conference", case=False, na=False))]

    bgfg_matched_step = int(bgfg_disc.iloc[0]["Matched_Step"])
    target_step = max(1, bgfg_matched_step - bgfg_step_offset_back)
    target_step_row = unified_dist_df[(unified_dist_df["Path_ID"] == "BGFG") & (unified_dist_df["Step"] == target_step)]

    t1_target = np.array([target_step_row.iloc[0]["X_rot"], target_step_row.iloc[0]["Y_rot"]])

    # Pure Translation Offset
    offset = t1_target - p1_new

    dist_df["X_rot"] = np.round(dist_df["X_rot"] + offset[0], 2)
    dist_df["Y_rot"] = np.round(dist_df["Y_rot"] + offset[1], 2)

    rooms_df["Trajectory_X_rot"] = np.round(rooms_df["Trajectory_X_rot"] + offset[0], 2)
    rooms_df["Trajectory_Y_rot"] = np.round(rooms_df["Trajectory_Y_rot"] + offset[1], 2)
    rooms_df["Coords_X_rot"] = np.round(rooms_df["Coords_X_rot"] + offset[0], 2)
    rooms_df["Coords_Y_rot"] = np.round(rooms_df["Coords_Y_rot"] + offset[1], 2)

    return dist_df, rooms_df

def detect_and_stitch_intersections(dist_df, dist_threshold=1.5, fixed_path_id="AUDI"):
    dist_df = dist_df.copy()
    dist_df["Is_Intersection"] = False
    dist_df["Intersection_ID"] = None
    dist_df["Connected_Path"] = None
    dist_df["Connected_Step"] = None

    path_ids = dist_df["Path_ID"].unique()
    junction_counter = 1

    for i in range(len(path_ids)):
        for j in range(i + 1, len(path_ids)):
            p1, p2 = path_ids[i], path_ids[j]

            df1 = dist_df[dist_df["Path_ID"] == p1]
            df2 = dist_df[dist_df["Path_ID"] == p2]

            coords1 = df1[["X_rot", "Y_rot"]].values
            coords2 = df2[["X_rot", "Y_rot"]].values

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

                # Stitch separated endpoints together without modifying the Audi corridor
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

def generate_unified_floorplan(unified_dist_path, unified_rooms_path, new_dist_df, new_rooms_df, output_dir, new_path_id="AUDI"):
    unified_dist_raw = pd.read_csv(unified_dist_path)
    unified_rooms_raw = pd.read_csv(unified_rooms_path)

    # Non-destructive in-memory shrinkage of AB017
    unified_dist, unified_rooms = shrink_ab017_in_memory(unified_dist_raw, unified_rooms_raw)

    new_dist = new_dist_df.copy()
    new_rooms = new_rooms_df.copy()

    new_dist["Path_ID"] = new_path_id
    new_rooms["Path_ID"] = new_path_id

    # Combine Distance Data and Stitch All Junctions Seamlessly
    combined_dist = pd.concat([unified_dist, new_dist], ignore_index=True)
    unified_dist_new = detect_and_stitch_intersections(combined_dist, dist_threshold=1.5, fixed_path_id=new_path_id)

    # Filter New Rooms Data
    rt_corr_indices = new_rooms[new_rooms["Room_ID"].str.contains(r"right turn corridor", case=False, na=False)].index
    first_rt_idx = rt_corr_indices[0] if len(rt_corr_indices) > 0 else None

    def filter_new_rooms(row):
        r_name = str(row["Room_ID"]).lower()
        r_idx = row.name

        if "audi door" in r_name:
            return False
        if "discussion" in r_name:
            return False
        if r_idx == first_rt_idx:
            return True
        if any(k in r_name for k in ["corridor", "right turn", "left turn"]):
            return False

        return True

    # unified_rooms = unified_rooms[~unified_rooms["Room_ID"].str.contains(r"main audi ab012", case=False, na=False)].copy()
    cols = ["Coords_X_rot", "Coords_Y_rot"]
    unified_rooms.loc[unified_rooms["Room_ID"].str.contains(r"main audi ab012", case=False, na=False), cols] = new_rooms.loc[new_rooms["Room_ID"].str.contains(r"audi door", case=False, na=False), cols].values

    new_rooms_clean = new_rooms[new_rooms.apply(filter_new_rooms, axis=1)].copy()
    unified_rooms_new = pd.concat([unified_rooms, new_rooms_clean], ignore_index=True)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    unified_dist_new.to_csv(output_path / "unified_fp_distance_3.csv", index=False)
    unified_rooms_new.to_csv(output_path / "unified_fp_rooms_3.csv", index=False)

    return unified_dist_new, unified_rooms_new

def render_floorplan(unified_dist, unified_rooms, output_path):
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    back_gate = unified_dist[(unified_dist["Path_ID"] == "BGFG") & (unified_dist["Step"] == 1)]
    bg_x = back_gate["X_rot"].iloc[0] if not back_gate.empty else -17.77
    bg_y = back_gate["Y_rot"].iloc[0] if not back_gate.empty else 56.14

    # Group by unique Intersection_ID to plot exactly one yellow marker per physical junction
    plot_intersections = unified_dist[unified_dist["Is_Intersection"] == True].drop_duplicates(subset=["Intersection_ID"]).copy()

    for show_axes in [False, True]:
        fig, ax = plt.subplots(figsize=(15, 11))

        # Hallways (Outer wall border + Inner walkway)
        for path_id, group in unified_dist.groupby("Path_ID"):
            ax.plot(group["X_rot"], group["Y_rot"], color="#95A5A6", linewidth=16, solid_capstyle="round", zorder=1)
            ax.plot(group["X_rot"], group["Y_rot"], color="#ECF0F1", linewidth=12, solid_capstyle="round", zorder=2)

        # Intersections
        ax.scatter(plot_intersections["X_rot"], plot_intersections["Y_rot"], color="yellow", marker="o", s=120, edgecolor="black", linewidth=1.5, zorder=4, label="Intersection")

        # Room Door Blocks
        ax.scatter(unified_rooms["Coords_X_rot"], unified_rooms["Coords_Y_rot"], color="#3498DB", marker="s", s=140, edgecolor="#2C3E50", linewidth=1.5, zorder=4, label="Rooms")

        # Outward Dynamic Label Placement
        for _, r in unified_rooms.iterrows():
            cx, cy = r["Coords_X_rot"], r["Coords_Y_rot"]
            tx, ty = r["Trajectory_X_rot"], r["Trajectory_Y_rot"]
            vx, vy = cx - tx, cy - ty

            if abs(vx) >= abs(vy):
                ha, va = ("left", "center") if vx >= 0 else ("right", "center")
                text_x = cx + (1.0 if vx >= 0 else -1.0)
                text_y = cy
            else:
                ha, va = ("center", "bottom") if vy >= 0 else ("center", "top")
                text_x = cx
                text_y = cy + (1.0 if vy >= 0 else -1.0)

            ax.text(text_x, text_y, str(r["Room_ID"]), fontsize=7.2, color="#2C3E50", weight="bold", ha=ha, va=va, zorder=5, bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", boxstyle="round,pad=0.2"))

        # Front Gate and Back Gate Markers
        ax.scatter(0, 0, color="red", marker="o", s=250, edgecolor="black", zorder=6, label="Front Gate")
        ax.scatter(bg_x, bg_y, color="black", marker="o", s=250, edgecolor="black", zorder=6, label="Back Gate")

        ax.set_aspect("equal")

        if not show_axes:
            ax.axis("off")
            ax.set_title("AB Ground Floor", fontsize=15, weight="bold", pad=20)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / "floorplan_3.png", dpi=300, bbox_inches="tight")
            plt.close()
        else:
            ax.set_xlabel("X Position (meters)", fontsize=11, weight="bold", labelpad=8)
            ax.set_ylabel("Y Position (meters)", fontsize=11, weight="bold", labelpad=8)
            ax.grid(True, linestyle="--", alpha=0.5, color="#BDC3C7")
            ax.axhline(0, color="#7F8C8D", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.axvline(0, color="#7F8C8D", linestyle=":", linewidth=0.8, alpha=0.7)
            ax.set_title("AB Ground Floor (With Coordinates)", fontsize=15, weight="bold", pad=20)
            ax.legend(loc="lower right", frameon=True, fontsize=9.5, labelspacing=1.3, handletextpad=1.0, borderpad=1.0)
            plt.tight_layout()
            plt.savefig(output_path / "floorplan_coords_3.png", dpi=300, bbox_inches="tight")
            plt.close()

if __name__ == "__main__":
    base_dir = Path(__file__).parent.resolve()
    raw_dir = base_dir / "Raw"
    save_dir = base_dir / "Processed"
    save_dir.mkdir(parents=True, exist_ok=True)

    old_unified_dist_path = save_dir / "unified_fp_distance_2.csv"
    old_unified_rooms_path = save_dir / "unified_fp_rooms_2.csv"

    new_data_name = "G-AUL-AB015L"
    new_path_dir = raw_dir / new_data_name
    new_save_dir = save_dir / new_data_name
    new_save_dir.mkdir(parents=True, exist_ok=True)

    new_dist_df, new_rooms_df = process_data(new_path_dir)
    save_data(new_dist_df, new_rooms_df, new_save_dir)

    old_unified_dist_current = pd.read_csv(old_unified_dist_path)
    old_unified_rooms_current = pd.read_csv(old_unified_rooms_path)

    stitched_dist_df, stitched_rooms_df = stitch_floorplan(old_unified_dist_current, old_unified_rooms_current, new_dist_df, new_rooms_df, bgfg_step_offset_back=7)
    save_stitched_data(stitched_dist_df, stitched_rooms_df, new_save_dir)

    unified_dist, unified_rooms = generate_unified_floorplan(old_unified_dist_path, old_unified_rooms_path, stitched_dist_df, stitched_rooms_df, save_dir, new_path_id="AUDI")
    render_floorplan(unified_dist, unified_rooms, save_dir)