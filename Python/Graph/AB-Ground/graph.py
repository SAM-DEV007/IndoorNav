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
        13: "Conference room - Sangam",
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
        31: "Audi door",
        32: "Ab011 music room",
        33: "Stair",
        34: "Audi backdoor",
        35: "Lift",
        36: "Ab013 board room",
        37: "Ab014 audi 2",
        38: "Stairs",
        39: "Ab015 dsw office",

        # Bank corridor
        40: "AB-010",
        41: "AB009 A - Boys Restroom",
        42: "AB009 B - Boys Restroom",
        43: "AB008 Storage",
        44: "AB007 Computer Studio - 1",
        45: "AB006 Bank",

        # Right corridor
        46: "AB019 Cabin",
        47: "AB018 B - Gents Washroom",
        48: "AB018 A - Ladies Washroom",
        49: "AB017 Cabin",
    }

    connection_id = {
        # Middle corridor
        1001: "RT",
        1002: "Junction_1",
        1003: "DRFT",
        1004: "ST",
        1005: "WT",
        1006: "AB026T",
        1007: "TT",
        1008: "CT",
        1009: "SVCT",
        1010: "VPVCT",
        1011: "DRBCRT",
        1012: "Junction_2",
        1013: "AUDIT",

        # Bottom corridor
        1014: "AB001T",
        1015: "AB002T",
        1016: "ATMT",
        1017: "Junction_4",
        1018: "AB005T",
        1019: "SLT",
        1020: "AB004LT",
        1021: "AB025T",
        1022: "AB024T",
        1023: "Junction_3",
        1024: "AB021T",
        1025: "SRT",
        1026: "AB022LT",

        # Audi corridor
        1027: "AudiDoorT",
        1028: "Junction_6",
        1029: "AB011T",
        1030: "ABST",
        1031: "LT",
        1032: "AB013BT",
        1033: "AB014T",
        1034: "Junction_5",
        1035: "ST",
        1036: "AB015T",

        # Bank corridor
        1037: "AB010T",
        1038: "AB009AT",
        1039: "AB009BT",
        1040: "AB008T",
        1041: "AB007T",
        1042: "AB006T",

        # Right corridor
        1043: "AB019T",
        1044: "AB018BT",
        1045: "AB018AT",
        1046: "AB017T",
    }

    main_edges = [
        # Middle corridor
        (1, 1001),
        (1001, 1002),
        (1002, 1003),
        (1003, 1004),
        (1004, 1005),
        (1005, 1006),
        (1006, 1007),
        (1007, 1008),
        (1008, 1009),
        (1009, 1010),
        (1010, 1011),
        (1011, 1012),
        (1012, 1013),
        (1013, 15),

        # Bottom corridor
        (1002, 1014),
        (1014, 1015),
        (1015, 1016),
        (1016, 1017),
        (1017, 1018),
        (1018, 1019),
        (1019, 1020),
        (1002, 1021),
        (1021, 1022),
        (1022, 1023),
        (1023, 1024),
        (1024, 1025),
        (1025, 1026),

        # Audi corridor
        (1012, 1027),
        (1027, 1028),
        (1028, 1029),
        (1029, 1030),
        (1030, 1031),
        (1012, 1032),
        (1032, 1033),
        (1033, 1034),
        (1034, 1035),
        (1035, 1036),

        # Bank corridor
        (1028, 1037),
        (1037, 1038),
        (1038, 1039),
        (1039, 1040),
        (1040, 1041),
        (1041, 1042),
        (1042, 1017),

        # Right corridor
        (1023, 1043),
        (1043, 1044),
        (1044, 1045),
        (1045, 1046),
        (1046, 1034),
    ]


    room_edges = [
        # Middle corridor
        (1001, 2),
        (1003, 3),
        (1004, 4),
        (1005, 5),
        (1006, 6),
        (1007, 7),
        (1008, 8),
        (1009, 9),
        (1010, 10),
        (1010, 11),
        (1011, 12),
        (1011, 13),
        (1013, 14),

        # Bottom corridor
        (1014, 16),
        (1015, 17),
        (1016, 18),
        (1017, 19),
        (1018, 20),
        (1019, 21),
        (1020, 22),
        (1020, 23),
        (1021, 24),
        (1022, 25),
        (1023, 26),
        (1024, 27),
        (1025, 28),
        (1026, 29),
        (1026, 30),

        # Audi corridor
        (1027, 31),
        (1029, 32),
        (1030, 33),
        (1030, 34),
        (1031, 35),
        (1032, 36),
        (1033, 37),
        (1035, 38),
        (1036, 39),

        # Bank corridor
        (1037, 40),
        (1038, 41),
        (1039, 42),
        (1040, 43),
        (1041, 44),
        (1042, 45),

        # Right corridor
        (1043, 46),
        (1044, 47),
        (1045, 48),
        (1046, 49),
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
        33: rooms_df.iloc[36].values[3:5],
        35: rooms_df.iloc[34].values[3:5],
        38: rooms_df.iloc[41].values[3:5],
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
        33: rooms_df.iloc[36].values[1:3],
        35: rooms_df.iloc[34].values[1:3],
        38: rooms_df.iloc[41].values[1:3],
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

    return rooms_id, connection_id, main_edges, room_edges, connection_pos, junction_pos, rooms_pos


def plot_graph(path, weighted=False):
    G = nx.read_graphml(f"{path}.graphml")
    G = nx.relabel_nodes(G, lambda x: int(x)) # The nodes are read as strings, so we convert them back to integers

    pos = {
        node: (
            float(data["pos_x"]),
            float(data["pos_y"])
        )
        for node, data in G.nodes(data=True)
    }

    plt.figure(figsize=(12, 8))
    
    nx.draw(G, pos=pos, with_labels=True, node_size=100, font_size=5, font_color="black", node_color="lightblue", edge_color="gray")

    if weighted is False:
        plt.title("AB-Ground Unweighted Graph")
    else:
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels, font_size=5, font_color="black")

        plt.title("AB-Ground Weighted Graph")
    
    plt.savefig(output_dir / f"{path}.png", dpi=300)
    plt.close()


def add_edge_weights(G, final_pos):
    # Add weights to edges based on Euclidean distance
    for edge in G.edges():
        node1, node2 = edge

        pos1 = final_pos[node1]
        pos2 = final_pos[node2]

        weight = np.linalg.norm(pos1 - pos2)
        G.edges[edge]['weight'] = np.round(weight, 2)

    return G


def create_save_weighted_graph(rooms_df, intersections_df, output_dir):
    _, _, main_edges, room_edges, connection_pos, junction_pos, rooms_pos = edges(rooms_df, intersections_df)
    
    final_edges = main_edges + room_edges
    final_pos = connection_pos | junction_pos | rooms_pos

    G = nx.Graph()
    G.add_edges_from(final_edges)

    G = add_edge_weights(G, final_pos) # Add weights to edges
    for node, (x, y) in final_pos.items():
        G.nodes[node]["pos_x"] = float(x)
        G.nodes[node]["pos_y"] = float(y)

    nx.write_graphml(G, output_dir / "ab_ground_weighted.graphml")
    plot_graph(output_dir / "ab_ground_weighted", weighted=True)


def create_save_unweighted_graph(rooms_df, intersections_df, output_dir):
    G = nx.Graph()
    _, _, main_edges, room_edges, connection_pos, junction_pos, rooms_pos = edges(rooms_df, intersections_df)

    final_edges = main_edges + room_edges
    final_pos = connection_pos | junction_pos | rooms_pos

    G.add_edges_from(final_edges)
    for node, (x, y) in final_pos.items():
        G.nodes[node]["pos_x"] = float(x)
        G.nodes[node]["pos_y"] = float(y)

    nx.write_graphml(G, output_dir / "ab_ground_unweighted.graphml")
    plot_graph(output_dir / "ab_ground_unweighted")


def save_ids(rooms_df, intersections_df, output_dir):
    rooms_id, connection_id, _, _, _, _, _ = edges(rooms_df, intersections_df)

    rooms_id_df = pd.DataFrame(list(rooms_id.items()), columns=["Room_ID", "Room_Name"])
    rooms_id_df.to_csv(output_dir / "ab_ground_rooms_id.csv", index=False)

    connection_id_df = pd.DataFrame(list(connection_id.items()), columns=["Connection_ID", "Connection_Name"])
    connection_id_df.to_csv(output_dir / "ab_ground_connections_id.csv", index=False)


if __name__ == "__main__":
    main_dir = Path(__file__).parent.parent.parent.resolve()

    floorplan_dir = main_dir / "Custom-Floormap" / "AB-Ground"
    output_dir = Path(__file__).parent.resolve()

    rooms_info = floorplan_dir / "ab_ground_unified_fp_rooms.csv"
    rooms_df = pd.read_csv(rooms_info)

    intersections_info = floorplan_dir / "ab_ground_unified_fp_intersections.csv"
    intersections_df = pd.read_csv(intersections_info)

    save_ids(rooms_df, intersections_df, output_dir)

    create_save_unweighted_graph(rooms_df, intersections_df, output_dir)
    create_save_weighted_graph(rooms_df, intersections_df, output_dir)