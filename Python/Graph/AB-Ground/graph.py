from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx


def edges(rooms_df, intersections_df):
    rooms_df = rooms_df.loc[:, ["Room_ID", "Trajectory_X_rot", "Trajectory_Y_rot", "Coords_X_rot", "Coords_Y_rot"]].copy()
    intersections_df = intersections_df.loc[:, ["Intersection_ID", "X_rot", "Y_rot"]].copy()

    rooms_id = {
        # Middle corridor
        1: "Front Gate",
        2: "Reception",
        3: "Discussion room",
        4: "Staircase",
        5: "Waiting room",
        6: "Admission office AB 026",
        7: "Trustee",
        8: "Chancelor",
        9: "Secretary to VC",
        10: "Vice president",
        11: "Vice chancellor",
        12: "Discussion room",
        13: "Conference room",
        14: "Main Audi AB012",
        15: "Back Gate",

        # Bottom corridor
        16: "Ab 001",
        17: "Ab 002",
        18: "ATM",
        19: "Ab 003",
        20: "Ab 005 electrical panel",
        21: "Staircase",
        22: "Ab 004",
        23: "Lift",
        24: "Ab 025",
        25: "Ab 024",
        26: "Ab 023",
        27: "Ab 021",
        28: "Staircase",
        29: "Ab 022 cabins",
        30: "Lift",

        # Audi corridor
        31: "Ab011 music room",
        32: "Stair",
        33: "Audi backdoor",
        34: "Lift",
        35: "Ab013 board room",
        36: "Ab014 audi 2",
        37: "Stairs",
        38: "Ab015 dsw office",

        # Bank corridor
        39: "AB-010",
        40: "AB009 Washroom",
        41: "AB008 Storage",
        42: "AB007 Computer Science",
        43: "AB006 Bank",

        # Right corridor
        44: "AB019 Cabin",
        45: "AB018 B - Gents Washroom",
        46: "AB018 A - Ladies Washroom",
        47: "AB017 Cabin",
    }

    connection_id = {
        # Middle corridor
        101: "RT",
        102: "Junction_1",
        103: "DRFT",
        104: "ST",
        105: "WT",
        106: "AB026T",
        107: "TT",
        108: "CT",
        109: "SVCT",
        110: "VPVCT",
        111: "DRBCRT",
        112: "Junction_2",
        113: "AUDIT",

        # Bottom corridor
        114: "AB001T",
        115: "AB002T",
        116: "ATMT",
        117: "Junction_4",
        118: "AB005T",
        119: "SLT",
        120: "AB004LT",
        121: "AB025T",
        122: "AB024T",
        123: "Junction_3",
        124: "AB021T",
        125: "SRT",
        126: "AB022LT",

        # Audi corridor
        127: "Junction_6",
        128: "AB011T",
        129: "ABST",
        130: "LT",
        131: "AB013BT",
        132: "AB014T",
        133: "Junction_5",
        134: "ST",
        135: "AB015T",

        # Bank corridor
        136: "AB010T",
        137: "AB009T",
        138: "AB008T",
        139: "AB007T",
        140: "AB006T",

        # Right corridor
        141: "AB019T",
        142: "AB018BT",
        143: "AB018AT",
        144: "AB017T",
    }

    main_edges = [
        # Middle corridor
        (1, 101),
        (101, 102),
        (102, 103),
        (103, 104),
        (104, 105),
        (105, 106),
        (106, 107),
        (107, 108),
        (108, 109),
        (109, 110),
        (110, 111),
        (111, 112),
        (112, 113),
        (113, 15),

        # Bottom corridor
        (102, 114),
        (114, 115),
        (115, 116),
        (116, 117),
        (117, 118),
        (118, 119),
        (119, 120),
        (102, 121),
        (121, 122),
        (122, 123),
        (123, 124),
        (124, 125),
        (125, 126),

        # Audi corridor
        (112, 127),
        (127, 128),
        (128, 129),
        (129, 130),
        (112, 131),
        (131, 132),
        (132, 133),
        (133, 134),
        (134, 135),

        # Bank corridor
        (127, 136),
        (136, 137),
        (137, 138),
        (138, 139),
        (139, 140),
        (140, 117),

        # Right corridor
        (123, 141),
        (141, 142),
        (142, 143),
        (143, 144),
        (144, 133),
    ]

    room_edges = [
        # Middle corridor
        (101, 2),
        (103, 3),
        (104, 4),
        (105, 5),
        (106, 6),
        (107, 7),
        (108, 8),
        (109, 9),
        (110, 10),
        (110, 11),
        (111, 12),
        (111, 13),
        (113, 14),

        # Bottom corridor
        (114, 16),
        (115, 17),
        (116, 18),
        (117, 19),
        (118, 20),
        (119, 21),
        (120, 22),
        (120, 23),
        (121, 24),
        (122, 25),
        (123, 26),
        (124, 27),
        (125, 28),
        (126, 29),
        (126, 30),

        # Audi corridor
        (128, 31),
        (129, 32),
        (129, 33),
        (130, 34),
        (131, 35),
        (132, 36),
        (134, 37),
        (135, 38),

        # Bank corridor
        (136, 39),
        (137, 40),
        (138, 41),
        (139, 42),
        (140, 43),

        # Right corridor
        (141, 44),
        (142, 45),
        (143, 46),
        (144, 47),
    ]

    # Manual positions to deal with same names
    # 1 & 2 is traj | 3 & 4 is coords from iloc output
    coords_manual_pos = {
        3: rooms_df.iloc[20].values[3:5],
        4: rooms_df.iloc[10].values[3:5],
        12: rooms_df.iloc[2].values[3:5],
        21: rooms_df.iloc[14].values[3:5],
        23: rooms_df.iloc[12].values[3:5],
        28: rooms_df.iloc[25].values[3:5],
        30: rooms_df.iloc[26].values[3:5],
        32: rooms_df.iloc[36].values[3:5],
        34: rooms_df.iloc[34].values[3:5],
        37: rooms_df.iloc[40].values[3:5],
    }

    traj_manual_pos = {
        3: np.linspace(
            rooms_df.iloc[10].values[1:3],
            intersections_df.loc[
                intersections_df["Intersection_ID"] == "Junction_1", ["X_rot", "Y_rot"]
            ].values[0],
            num=3,
        )[1], # Interpolate trajectory coordinates
        4: rooms_df.iloc[10].values[1:3],
        12: rooms_df.iloc[2].values[1:3],
        21: rooms_df.iloc[14].values[1:3],
        23: rooms_df.iloc[12].values[1:3],
        28: rooms_df.iloc[25].values[1:3],
        30: rooms_df.iloc[26].values[1:3],
        32: rooms_df.iloc[36].values[1:3],
        34: rooms_df.iloc[34].values[1:3],
        37: rooms_df.iloc[40].values[1:3],
    }

    connection_pos = {
        room_edges[i][0]: (
            traj_manual_pos[room_edges[i][1]]
            if room_edges[i][1] in traj_manual_pos
            else np.squeeze(rooms_df.loc[rooms_df["Room_ID"] == rooms_id[room_edges[i][1]], ["Trajectory_X_rot", "Trajectory_Y_rot"]].values)
        )
        for i in range(len(room_edges))
    }

    junction_pos = {
        i: np.squeeze(intersections_df.loc[intersections_df["Intersection_ID"] == connection_id[i], ["X_rot", "Y_rot"]].values)
        for i in (set(connection_id.keys()) - set(connection_pos.keys()))
    }

    rooms_pos = {
        room_id: (
            coords_manual_pos[room_id]
            if room_id in coords_manual_pos
            else np.squeeze(rooms_df.loc[rooms_df["Room_ID"] == room_name, ["Coords_X_rot", "Coords_Y_rot"]].values)
        )
        for room_id, room_name in rooms_id.items()
    }

    final_edges = main_edges + room_edges
    final_pos = connection_pos | junction_pos | rooms_pos

    return rooms_id, connection_id, final_edges, final_pos


def create_save_unweighted_graph(rooms_df, intersections_df, output_dir):
    G = nx.Graph()
    rooms_id, connection_id, all_edges, pos = edges(rooms_df, intersections_df)

    G.add_edges_from(all_edges)

    plt.figure(figsize=(12, 8))
    
    nx.draw(G, pos=pos, with_labels=True, node_size=100, font_size=5, font_color="black", node_color="lightblue", edge_color="gray")
    
    plt.title("AB-Ground Unweighted Graph")
    plt.savefig(output_dir / "ab_ground_unweighted_graph.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    main_dir = Path(__file__).parent.parent.parent.resolve()

    floorplan_dir = main_dir / "Custom-Floormap" / "AB-Ground"
    output_dir = Path(__file__).parent.resolve()

    rooms_info = floorplan_dir / "ab_ground_unified_fp_rooms.csv"
    rooms_df = pd.read_csv(rooms_info)

    intersections_info = floorplan_dir / "ab_ground_unified_fp_intersections.csv"
    intersections_df = pd.read_csv(intersections_info)

    create_save_unweighted_graph(rooms_df, intersections_df, output_dir)