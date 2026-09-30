import ast
import math
from pathlib import Path

import networkx as nx
import pandas as pd
from streamlit_echarts import st_echarts
import streamlit as st


APP_DIR = Path(__file__).resolve().parent
GRAPH_PATH = APP_DIR.parent.parent / "Graph" / "AB-Ground" / "ab_ground_weighted.graphml"
ROOMS_PATH = APP_DIR.parent.parent / "Graph" / "AB-Ground" / "ab_ground_rooms_id.csv"
CACHE_PATH = APP_DIR / "Cache" / "shortest_paths_cache.csv"


@st.cache_data
def load_map_data():
    graph = nx.read_graphml(GRAPH_PATH)
    graph = nx.relabel_nodes(graph, lambda node: int(node))
    positions = {
        node: (float(data["pos_x"]), float(data["pos_y"]))
        for node, data in graph.nodes(data=True)
    }
    rooms = pd.read_csv(ROOMS_PATH)
    rooms["Room_ID"] = rooms["Room_ID"].astype(int)
    rooms["label"] = rooms.apply(lambda row: f"{row.Room_Name}", axis=1)
    
    cache = pd.read_csv(CACHE_PATH)
    cache["Path"] = cache["Path"].map(ast.literal_eval)

    routes = {}
    for row in cache.itertuples(index=False):
        path = [int(node) for node in row.Path]
        routes[(int(row.Source), int(row.Target))] = (float(row.Distance), path)
        routes[(int(row.Target), int(row.Source))] = (float(row.Distance), path[::-1])
    return graph, positions, rooms, routes


def nearest_room(x, y, rooms, positions):
    return min(
        rooms.Room_ID,
        key=lambda room_id: (positions[int(room_id)][0] - x) ** 2
        + (positions[int(room_id)][1] - y) ** 2,
    )


def wrap_label(label, width=19):
    words = label.split()
    lines, current = [], ""
    for word in words:
        if current and len(current) + len(word) + 1 > width:
            lines.append(current)
            current = word
        else:
            current = f"{current} {word}".strip()
    if current:
        lines.append(current)
    return "<br>".join(lines)


def room_label_layout(room_id, graph, positions):
    if room_id == 18:
        return {"xshift": 20, "yshift": 0, "xanchor": "left", "yanchor": "middle", "align": "left", "angle": 0}
    if room_id == 20:
        return {"xshift": 0, "yshift": 14, "xanchor": "center", "yanchor": "bottom", "align": "center", "angle": 0}
    if room_id == 21:
        return {"xshift": 0, "yshift": 14, "xanchor": "center", "yanchor": "bottom", "align": "center", "angle": 0}
    if room_id == 43:
        return {"xshift": 0, "yshift": 14, "xanchor": "center", "yanchor": "bottom", "align": "center", "angle": 0}
    if room_id == 32:
        return {"xshift": -12, "yshift": 0, "xanchor": "right", "yanchor": "middle", "align": "right", "angle": 0}

    x, y = positions[room_id]
    neighbors = list(graph.neighbors(room_id))
    if not neighbors:
        return {"xshift": 0, "yshift": 12, "xanchor": "center", "yanchor": "bottom", "align": "center", "angle": 0}

    neighbor_x, neighbor_y = positions[neighbors[0]]
    away_x, away_y = x - neighbor_x, y - neighbor_y
    angle = math.degrees(math.atan2(away_y, away_x))
    if abs(away_x) >= abs(away_y):
        return {
            "xshift": 30 if away_x > 0 else -30,
            "yshift": 0,
            "xanchor": "left" if away_x > 0 else "right",
            "yanchor": "middle",
            "align": "left" if away_x > 0 else "right",
            "angle": angle,
        }
    return {
        "xshift": 0,
        "yshift": 20 if away_y > 0 else -20,
        "xanchor": "center",
        "yanchor": "bottom" if away_y > 0 else "top",
        "align": "center",
        "angle": angle,
    }


def get_echarts_options(graph, positions, rooms, route, origin, destination):
    all_x, all_y = zip(*positions.values())
    padding = 8

    min_x, max_x = min(all_x) - padding, max(all_x) + padding
    min_y, max_y = min(all_y) - padding, max(all_y) + padding

    series = []

    base_lines = []
    for left, right in graph.edges:
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        base_lines.append({"coords": [[x1, y1], [x2, y2]]})

    series.append({
        "name": "Walkable path",
        "type": "lines",
        "coordinateSystem": "cartesian2d",
        "data": base_lines,
        "lineStyle": {"color": "#aebabc", "width": 15},
        "silent": False,
        "tooltip": {"show": False}
    })

    base_nodes = [{"value": [positions[n][0], positions[n][1]]} for n in graph.nodes]
    series.append({
        "type": "scatter",
        "data": base_nodes,
        "symbolSize": 15,
        "itemStyle": {"color": "#aebabc"},
        "silent": False,
        "tooltip": {"show": False}
    })

    series.append({
        "type": "lines",
        "coordinateSystem": "cartesian2d",
        "data": base_lines,
        "lineStyle": {"color": "#edf1ee", "width": 9},
        "silent": False,
        "tooltip": {"show": False}
    })

    series.append({
        "type": "scatter",
        "data": base_nodes,
        "symbolSize": 9,
        "itemStyle": {"color": "#edf1ee"},
        "silent": False,
        "tooltip": {"show": False}
    })

    if route:
        route_coords = [[positions[node][0], positions[node][1]] for node in route]
        series.append({
            "name": "Shortest route",
            "type": "line",
            "data": route_coords,
            "lineStyle": {"color": "#e4572e", "width": 5},
            "symbol": "none",
            "z": 10,
            "silent": True,
            "tooltip": {"show": False}
        })

    room_records = rooms[~rooms.Room_ID.isin([1, 15])]
    room_data = []
    for row in room_records.itertuples(index=False):
        x, y = positions[int(row.Room_ID)]
        layout = room_label_layout(int(row.Room_ID), graph, positions)
        room_data.append({
            "value": [x, y],
            "roomId": int(row.Room_ID),
            "name": row.Room_Name,
            "label": {
                "show": True,
                "formatter": row.Room_Name,
                "color": "#263238",
                "fontSize": 9,
                "offset": [layout["xshift"], -layout["yshift"]],
                "align": "center",
                "verticalAlign": "middle"
            }
        })
    
    series.append({
        "name": "Rooms",
        "type": "scatter",
        "symbol": "rect",
        "symbolSize": 10,
        "itemStyle": {"color": "#2f8fbd", "borderColor": "#17465d", "borderWidth": 1},
        "data": room_data,
        "z": 20
    })

    gate_data = [
        {
            "value": [positions[1][0], positions[1][1]],
            "roomId": 1,
            "name": "Front Gate",
            "label": {"show": True, "formatter": "Front Gate", "position": "bottom", "color": "#263238", "fontSize": 10},
            "itemStyle": {"color": "#e63946"}
        },
        {
            "value": [positions[15][0], positions[15][1]],
            "roomId": 15,
            "name": "Back Gate",
            "label": {"show": True, "formatter": "Back Gate", "position": "top", "color": "#263238", "fontSize": 10},
            "itemStyle": {"color": "#111111"}
        }
    ]
    series.append({
        "name": "Gates",
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 15,
        "itemStyle": {"borderColor": "#202a2e", "borderWidth": 2},
        "data": gate_data,
        "z": 20
    })

    series.append({
        "name": "Start",
        "type": "scatter",
        "data": [],
        "symbol": "circle",
        "symbolSize": 10,
        "itemStyle": {"color": "#ffffff", "borderColor": "#202a2e", "borderWidth": 1}
    })
    series.append({
        "name": "Destination",
        "type": "scatter",
        "data": [],
        "symbol": "circle",
        "symbolSize": 10,
        "itemStyle": {"color": "#2ca25f", "borderColor": "#202a2e", "borderWidth": 1}
    })

    marker_data = []
    for room_id, color in ((origin, "#ffffff"), (destination, "#2ca25f")):
        if int(room_id) in (1, 15):
            continue
        x, y = positions[int(room_id)]
        marker_data.append({
            "value": [x, y],
            "itemStyle": {"color": color, "borderColor": "#202a2e", "borderWidth": 2}
        })
    series.append({
        "type": "scatter",
        "symbolSize": 10,
        "data": marker_data,
        "z": 30,
        "silent": True,
        "tooltip": {"show": False}
    })

    options = {
        "backgroundColor": "#fbfaf6",
        "grid": {"left": 10, "right": 10, "top": 40, "bottom": 10},
        "legend": {
            "data": ["Rooms", "Gates", "Start", "Destination"],
            "top": 0,
            "left": 10,
            "textStyle": {"color": "#263238", "fontSize": 12},
            "itemGap": 15
        },
        "xAxis": {
            "show": False,
            "type": "value",
            "min": min_x,
            "max": max_x,
            "scale": True
        },
        "yAxis": {
            "show": False,
            "type": "value",
            "min": min_y,
            "max": max_y,
            "scale": True
        },
        "tooltip": {
            "show": True,
            "trigger": "item",
            "formatter": "{b}"
        },
        "dataZoom": [
            {
                "type": "inside",
                "xAxisIndex": 0,
                "yAxisIndex": 0,
                "zoomOnMouseWheel": True,
                "moveOnMouseMove": True
            }
        ],
        "series": series,
        "animation": False
    }

    return options


def handle_map_click(clicked_data):
    if not clicked_data:
        return False
        
    x, y = None, None
    new_dest = None
    
    if clicked_data.get("roomId"):
        new_dest = int(clicked_data["roomId"])
    elif clicked_data.get("value") and len(clicked_data["value"]) >= 2:
        x, y = float(clicked_data["value"][0]), float(clicked_data["value"][1])
    elif clicked_data.get("coords") and len(clicked_data["coords"]) > 0:
        x, y = float(clicked_data["coords"][0][0]), float(clicked_data["coords"][0][1])
        
    if new_dest is None and x is not None and y is not None:
        new_dest = int(nearest_room(x, y, st.session_state.rooms, st.session_state.positions))
        
    if new_dest is not None and st.session_state.get("destination") != new_dest:
        st.session_state.destination = new_dest
        return True
        
    return False


def main():
    st.set_page_config(page_title="AB Ground Navigation", layout="wide")
    
    graph, positions, rooms, routes = load_map_data()

    st.session_state.rooms = rooms
    st.session_state.positions = positions

    labels = dict(zip(rooms.Room_ID, rooms.label))

    st.title("AB Ground Floor Navigation")
    st.caption("Choose rooms or click a corridor to place the destination location.")
    
    controls, map_column = st.columns([1, 2.7], gap="large")
    
    with controls:
        room_options = list(rooms.Room_ID)
        default_origin = int(st.session_state.get("origin", room_options[0]))
        
        fallback_dest = room_options[1] if len(room_options) > 1 else room_options[0]
        destination_default = int(st.session_state.get("destination", fallback_dest))
        
        origin = st.selectbox("Starting room", room_options, index=room_options.index(default_origin), format_func=labels.get)
        destination = st.selectbox("Destination room", room_options, index=room_options.index(destination_default), format_func=labels.get)
        
        if origin == destination:
            st.info("Choose two different rooms to show a route.")
            route_data = None
        else:
            route_data = routes.get((origin, destination))
            if route_data:
                st.metric("Route distance", f"{route_data[0]:.2f} m")
            else:
                st.warning("No cached route exists for this pair.")

    with map_column:
        options = get_echarts_options(graph, positions, rooms, route_data[1] if route_data else None, origin, destination)
        
        events = {
            "click": """function(params) {
                return {
                    roomId: params.data ? params.data.roomId : null,
                    value: params.value,
                    coords: params.data ? params.data.coords : null
                }
            }"""
        }
        
        clicked_data = st_echarts(
            options=options,
            events=events,
            height="650px",
            key="floorplan"
        )
        
        if handle_map_click(clicked_data):
            st.rerun()


if __name__ == "__main__":
    main()