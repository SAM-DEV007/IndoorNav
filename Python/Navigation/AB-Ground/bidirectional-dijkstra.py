import ast

from pathlib import Path
from tqdm import tqdm

import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt


def room_permutations(rooms_df):
    room_ids = rooms_df["Room_ID"].tolist()
    room_permutations = [(room1, room2) for i, room1 in enumerate(room_ids) for j, room2 in enumerate(room_ids) if i < j]

    return room_permutations


def shortest_path(G, source, target):
    distance, path = nx.bidirectional_dijkstra(G, source, target, weight="weight")
    return distance, path


def load_cache(cache_file):
    cache_df = pd.read_csv(cache_file)
    cache_df["Path"] = cache_df["Path"].apply(ast.literal_eval)

    return cache_df

    
def plot_shortest_path(G, source, target, pos, path, distance, plot_dir):
    path_edges = list(zip(path[:-1], path[1:]))
    
    plt.figure(figsize=(12, 8))

    nx.draw_networkx_edges(G, pos=pos, edge_color="lightgray", width=1)
    nx.draw_networkx_nodes(G, pos=pos, node_size=100, node_color="lightblue")
    nx.draw_networkx_labels(G, pos=pos, font_size=5, font_color="black")

    nx.draw_networkx_edges(G, pos=pos, edgelist=path_edges, edge_color="red", width=3)
    nx.draw_networkx_nodes(G, pos=pos, nodelist=path, node_size=100, node_color="orange")

    nx.draw_networkx_nodes(G, pos=pos, nodelist=[source], node_size=100, node_color="green")
    nx.draw_networkx_nodes(G, pos=pos, nodelist=[target], node_size=100, node_color="red")

    plt.title(f"Shortest Path: Room {source} to Room {target} (Distance: {distance:.2f})")
    plt.savefig(plot_dir / f"{source}_{target}.png", dpi=300, bbox_inches="tight")
    plt.close()


def create_plots(G, cache_file, plot_dir):
    cache_df = load_cache(cache_file)

    pos = {
        node: (float(data["pos_x"]), float(data["pos_y"]))
        for node, data in G.nodes(data=True)
    }

    sample_df = cache_df.sample(n=min(20, len(cache_df)), random_state=42)

    for _, row in tqdm(sample_df.iterrows(), desc="Creating plots", total=len(sample_df)):
        source = row["Source"]
        target = row["Target"]
        distance = row["Distance"]
        path = row["Path"]

        plot_shortest_path(G, source, target, pos, path, distance, plot_dir)


def create_cache(G, room_permutations, cache_file):
    results = []

    for source, target in tqdm(room_permutations, desc="Creating cache", total=len(room_permutations)):
        distance, path = shortest_path(G, source, target)
        results.append({"Source": source, "Target": target, "Distance": distance, "Path": path})

    results_df = pd.DataFrame(results)
    results_df.to_csv(cache_file, index=False)


def construct_graph(path):
    G = nx.read_graphml(path)
    G = nx.relabel_nodes(G, lambda x: int(x))

    return G


if __name__ == "__main__":
    main_dir = Path(__file__).parent.parent.parent.resolve()
    parent_dir = Path(__file__).parent.resolve()

    output_dir = parent_dir / "Cache"
    plot_output_dir = output_dir / "Plots"
    plot_output_dir.mkdir(parents=True, exist_ok=True)

    cache_file = output_dir / "shortest_paths_cache.csv"

    graph_dir = main_dir / "Graph" / "AB-Ground"
    directed_graph_path = graph_dir / "ab_ground_weighted.graphml"

    rooms_info = graph_dir / "ab_ground_rooms_id.csv"
    rooms_df = pd.read_csv(rooms_info)

    room_permutations = room_permutations(rooms_df)

    G = construct_graph(directed_graph_path)
    create_cache(G, room_permutations, cache_file)
    create_plots(G, cache_file, plot_output_dir)