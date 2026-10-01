import ast
import math

from pathlib import Path
from streamlit_echarts import st_echarts

import networkx as nx
import pandas as pd
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
        key=lambda room_id: (positions[int(room_id)][0] - x) ** 2 + (positions[int(room_id)][1] - y) ** 2
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
    return " ".join(lines)


def room_label_layout(room_id, graph, positions):
	if room_id == 18:
		return {"position": "right", "distance": 8, "offset": [0, -8]}
	if room_id == 20:
		return {"position": "top", "distance": 8, "offset": [0, 0]}
	if room_id == 21:
		return {"position": "top", "distance": 5, "offset": [0, 0]}
	if room_id == 43:
		return {"position": "top", "distance": 8, "offset": [0, 0]}
	if room_id == 32:
		return {"position": "left", "distance": 8, "offset": [0, 0]}

	x, y = positions[room_id]
	neighbors = list(graph.neighbors(room_id))
	if not neighbors:
		return {"position": "top", "distance": 8, "offset": [0, 0]}

	neighbor_x, neighbor_y = positions[neighbors[0]]
	away_x, away_y = x - neighbor_x, y - neighbor_y
	if abs(away_x) >= abs(away_y):
		pos = "right" if away_x > 0 else "left"
	else:
		pos = "top" if away_y > 0 else "bottom"

	return {"position": pos, "distance": 8, "offset": [0, 0]}


def get_echarts_options(graph, positions, rooms, route, origin, destination):
    all_x, all_y = zip(*positions.values())
    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    x_pad = x_span * 0.08
    y_pad = y_span * 0.08

    min_x, max_x = min(all_x) - x_pad, max(all_x) + x_pad
    min_y, max_y = min(all_y) - y_pad, max(all_y) + y_pad

    mz = st.session_state.get("map_zoom")
    zoom_x_start = mz[0]["start"] if mz and len(mz) > 0 and "start" in mz[0] else 0
    zoom_x_end = mz[0]["end"] if mz and len(mz) > 0 and "end" in mz[0] else 100
    zoom_y_start = mz[1]["start"] if mz and len(mz) > 1 and "start" in mz[1] else 0
    zoom_y_end = mz[1]["end"] if mz and len(mz) > 1 and "end" in mz[1] else 100

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
		"lineStyle": {
			"color": "#d2dbde",
			"width": 12,
			"opacity": 1,
			"cap": "round",
			"join": "round"
		},
		"clip": False,
		"silent": False,
		"tooltip": {"show": False}
	})

    series.append({
        "type": "lines",
        "coordinateSystem": "cartesian2d",
        "data": base_lines,
        "lineStyle": {
            "color": "#d2dbde",
            "width": 9,
            "cap": "round",
            "join": "round"
        },
        "clip": False,
        "silent": False,
        "tooltip": {"show": False}
    })

    if route:
        route_coords = [[positions[node][0], positions[node][1]] for node in route]
        series.append({
            "name": "Shortest route",
            "type": "lines",
            "coordinateSystem": "cartesian2d",
            "polyline": True,
            "data": [{"coords": route_coords}],
            "lineStyle": {
                "color": "#e4572e",
                "width": 5,
                "opacity": 1,
                "cap": "round",
                "join": "round"
            },
            "clip": False,
            "z": 10,
            "silent": True,
            "tooltip": {"show": False}
        })

    room_records = rooms[~rooms.Room_ID.isin([1, 15])]
    room_boxes = []
    room_labels = []
    for row in room_records.itertuples(index=False):
        x, y = positions[int(row.Room_ID)]
        layout = room_label_layout(int(row.Room_ID), graph, positions)
        text = wrap_label(str(row.Room_Name))
        room_boxes.append({
            "value": [x, y],
            "roomId": int(row.Room_ID),
            "name": row.Room_Name
        })
        room_labels.append({
            "value": [x, y],
            "roomId": int(row.Room_ID),
            "name": row.Room_Name,
            "label": {
                "show": True,
                "formatter": text,
                "position": layout["position"],
                "distance": layout["distance"],
                "offset": layout["offset"],
                "color": "#1f2937",
                "fontWeight": "bold",
                "fontSize": 8,
                "lineHeight": 10
            }
        })

    series.append({
        "name": "Rooms",
        "type": "scatter",
        "symbol": "rect",
        "symbolSize": 10,
        "itemStyle": {"color": "#2f8fbd", "borderColor": "#17465d", "borderWidth": 1},
        "data": room_boxes,
        "z": 20
    })

    series.append({
        "name": "Room names",
        "type": "scatter",
        "symbol": "rect",
        "symbolSize": 0,
        "itemStyle": {"color": "#2f8fbd"},
        "data": room_labels,
        "z": 21
    })

    gate_boxes = [
        {
            "value": [positions[1][0], positions[1][1]],
            "roomId": 1,
            "name": "Front Gate",
            "itemStyle": {"color": "#e63946"}
        },
        {
            "value": [positions[15][0], positions[15][1]],
            "roomId": 15,
            "name": "Back Gate",
            "itemStyle": {"color": "#111111"}
        }
    ]

    gate_labels = [
        {
            "value": [positions[1][0], positions[1][1]],
            "roomId": 1,
            "name": "Front Gate",
            "label": {
                "show": True,
                "formatter": "Front Gate",
                "position": "bottom",
                "distance": 8,
                "color": "#1f2937",
                "fontWeight": "bold",
                "fontSize": 9
            }
        },
        {
            "value": [positions[15][0], positions[15][1]],
            "roomId": 15,
            "name": "Back Gate",
            "label": {
                "show": True,
                "formatter": "Back Gate",
                "position": "top",
                "distance": 8,
                "color": "#1f2937",
                "fontWeight": "bold",
                "fontSize": 9
            }
        }
    ]

    series.append({
        "name": "Gates",
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 15,
        "itemStyle": {"borderColor": "#202a2e", "borderWidth": 2},
        "data": gate_boxes,
        "z": 20
    })

    series.append({
        "name": "Gate names",
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 0,
        "itemStyle": {"color": "#202a2e"},
        "data": gate_labels,
        "z": 21
    })

    marker_data = []
    for room_id, color, size in ((origin, "#ffffff", 10), (destination, "#2ca25f", 15)):
        if room_id in (None, "-"):
            continue
        x, y = positions[int(room_id)]
        marker_data.append({
            "value": [x, y],
            "symbolSize": size,
            "itemStyle": {"color": color, "borderColor": "#202a2e", "borderWidth": 2}
        })

    series.append({
        "type": "scatter",
        "data": marker_data,
        "z": 30,
        "silent": True,
        "tooltip": {"show": False}
    })

    options = {
        "backgroundColor": "#fbfaf6",
        "grid": {
            "show": True,
            "borderColor": "#b0bec5",
            "borderWidth": 1.5,
            "left": 40,
            "right": 40,
            "top": 60,
            "bottom": 30
        },
        "legend": {
            "data": ["Rooms", "Room names", "Gates", "Gate names", "Start", "Destination"],
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
                "filterMode": "none",
                "start": zoom_x_start,
                "end": zoom_x_end,
                "zoomOnMouseWheel": True,
                "moveOnMouseMove": True
            },
            {
                "type": "inside",
                "yAxisIndex": 0,
                "filterMode": "none",
                "start": zoom_y_start,
                "end": zoom_y_end,
                "zoomOnMouseWheel": True,
                "moveOnMouseMove": True
            }
        ],
        "series": series,
        "animation": False
    }

    return options


def handle_map_click(clicked_data):
    if not clicked_data or not isinstance(clicked_data, dict):
        return False

    event = clicked_data.get("chart_event", clicked_data)
    if not isinstance(event, dict):
        return False

    dz = event.get("dataZoom")
    if dz and isinstance(dz, list) and len(dz) >= 2 and dz[0] and dz[1]:
        st.session_state.map_zoom = dz

    new_dest = None

    if event.get("roomId"):
        new_dest = int(event["roomId"])
    elif event.get("value") and isinstance(event["value"], list) and len(event["value"]) >= 2:
        x, y = float(event["value"][0]), float(event["value"][1])
        new_dest = int(nearest_room(x, y, st.session_state.rooms, st.session_state.positions))
    elif event.get("coords") and isinstance(event["coords"], list) and len(event["coords"]) > 0:
        x, y = float(event["coords"][0][0]), float(event["coords"][0][1])
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
    labels["-"] = "-"
    room_options = ["-"] + list(rooms.Room_ID)

    if "origin" not in st.session_state:
        st.session_state.origin = "-"
    if "destination" not in st.session_state:
        st.session_state.destination = "-"

    st.title("AB Ground Floor Navigation")
    st.caption("Choose rooms or click a corridor to place the destination location.")
    
    controls, map_column = st.columns([1, 2.7], gap="large")
    
    with controls:
        default_origin = st.session_state.get("origin", "-")
        default_dest = st.session_state.get("destination", "-")

        origin = st.selectbox(
            "Starting room (From)",
            room_options,
            index=room_options.index(default_origin) if default_origin in room_options else 0,
            format_func=labels.get
        )
        destination = st.selectbox(
            "Destination room (To)",
            room_options,
            index=room_options.index(default_dest) if default_dest in room_options else 0,
            format_func=labels.get
        )

        if origin == "-" or destination == "-":
            route_data = None
        elif origin == destination:
            st.info("Choose two different rooms to show a route.")
            route_data = None
        else:
            route_data = routes.get((int(origin), int(destination)))
            if route_data:
                st.metric("Route distance", f"{route_data[0]:.2f} m")
            else:
                st.warning("No cached route exists for this pair.")

    with map_column:
        options = get_echarts_options(graph, positions, rooms, route_data[1] if route_data else None, origin, destination)

        events = {
            "datazoom": """function(p) {
                window._mapZoom = window._mapZoom || [{start: 0, end: 100}, {start: 0, end: 100}];
                var b = p.batch || [p];
                for (var i = 0; i < b.length; i++) {
                    var idx = (b[i].dataZoomIndex !== undefined) ? b[i].dataZoomIndex : i;
                    if (idx < 2) {
                        window._mapZoom[idx] = { start: b[i].start, end: b[i].end };
                    }
                }
            }""",
            "click": """function(params) {
                return {
                    roomId: params.data ? params.data.roomId : null,
                    value: params.value,
                    coords: params.data ? params.data.coords : null,
                    dataZoom: window._mapZoom || null
                };
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