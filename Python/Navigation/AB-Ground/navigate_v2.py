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


def project_point_on_segment(p, a, b):
    px, py = p
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    line_len_sq = dx * dx + dy * dy

    if line_len_sq == 0:
        return ax, ay
    
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / line_len_sq))

    return ax + t * dx, ay + t * dy


def euclidean_dist(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


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


def get_echarts_options(graph, positions, rooms, route_coords, origin, destination):
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

    is_active = bool(st.session_state.get("click_target"))
    cursor_style = "pointer" if is_active else "grab"
    is_silent = not is_active

    series = []

    base_lines = []
    for left, right in graph.edges:
        x1, y1 = positions[left]
        x2, y2 = positions[right]
        base_lines.append({
            "coords": [[x1, y1], [x2, y2]],
            "edge": [int(left), int(right)]
        })

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
        "clip": True,
        "cursor": cursor_style,
        "silent": is_silent,
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
        "cursor": cursor_style,
        "silent": is_silent,
        "tooltip": {"show": False}
    })

    if route_coords:
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
        "cursor": cursor_style,
        "z": 20
    })

    series.append({
        "name": "Room names",
        "type": "scatter",
        "symbol": "rect",
        "symbolSize": 0,
        "itemStyle": {"color": "#2f8fbd"},
        "data": room_labels,
        "cursor": cursor_style,
        "silent": is_silent,
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
        "cursor": cursor_style,
        "z": 20
    })

    series.append({
        "name": "Gate names",
        "type": "scatter",
        "symbol": "circle",
        "symbolSize": 0,
        "itemStyle": {"color": "#202a2e"},
        "data": gate_labels,
        "cursor": cursor_style,
        "silent": is_silent,
        "z": 21
    })

    marker_data = []
    if origin == "Custom" and st.session_state.get("custom_origin"):
        cx, cy = st.session_state.custom_origin["point"]
        marker_data.append({
            "value": [cx, cy],
            "symbolSize": 10,
            "itemStyle": {"color": "#ffffff", "borderColor": "#202a2e", "borderWidth": 2}
        })
    elif origin not in (None, "-"):
        ox, oy = positions[int(origin)]
        marker_data.append({
            "value": [ox, oy],
            "symbolSize": 10,
            "itemStyle": {"color": "#ffffff", "borderColor": "#202a2e", "borderWidth": 2}
        })

    if destination not in (None, "-"):
        dx, dy = positions[int(destination)]
        marker_data.append({
            "value": [dx, dy],
            "symbolSize": 15,
            "itemStyle": {"color": "#2ca25f", "borderColor": "#202a2e", "borderWidth": 2}
        })

    series.append({
        "type": "scatter",
        "data": marker_data,
        "z": 30,
        "cursor": cursor_style,
        "silent": True,
        "tooltip": {"show": False}
    })

    click_target = st.session_state.get("click_target")

    # Start button styling
    start_active = click_target == "origin"
    start_bg = "#2ca25f" if start_active else "#ffffff"
    start_border = "#2ca25f" if start_active else "#b0bec5"
    start_text_color = "#ffffff" if start_active else "#263238"
    start_label = "Start: Active" if start_active else "Set Start"

    # Destination button styling
    dest_active = click_target == "destination"
    dest_bg = "#2ca25f" if dest_active else "#ffffff"
    dest_border = "#2ca25f" if dest_active else "#b0bec5"
    dest_text_color = "#ffffff" if dest_active else "#263238"
    dest_label = "Destination: Active" if dest_active else "Set Destination"

    btn_h = 30
    btn1_w = 105
    btn2_w = 125
    gap = 10

    graphic_buttons = [
        {
            "type": "group",
            "right": 20 + btn2_w + gap,
            "top": 14,
            "width": btn1_w,
            "height": btn_h,
            "cursor": "pointer",
            "info": "toggle_origin",
            "z": 100,
            "children": [
                {
                    "type": "rect",
                    "left": "center",
                    "top": "middle",
                    "shape": {"width": btn1_w, "height": btn_h, "r": 6},
                    "style": {
                        "fill": start_bg,
                        "stroke": start_border,
                        "lineWidth": 1.5,
                        "shadowBlur": 4,
                        "shadowColor": "rgba(0,0,0,0.12)",
                        "shadowOffsetY": 2
                    },
                    "cursor": "pointer",
                    "info": "toggle_origin"
                },
                {
                    "type": "text",
                    "left": "center",
                    "top": "middle",
                    "style": {
                        "text": start_label,
                        "fill": start_text_color,
                        "font": "600 11px sans-serif"
                    },
                    "cursor": "pointer",
                    "info": "toggle_origin"
                }
            ]
        },
        {
            "type": "group",
            "right": 20,
            "top": 14,
            "width": btn2_w,
            "height": btn_h,
            "cursor": "pointer",
            "info": "toggle_dest",
            "z": 100,
            "children": [
                {
                    "type": "rect",
                    "left": "center",
                    "top": "middle",
                    "shape": {"width": btn2_w, "height": btn_h, "r": 6},
                    "style": {
                        "fill": dest_bg,
                        "stroke": dest_border,
                        "lineWidth": 1.5,
                        "shadowBlur": 4,
                        "shadowColor": "rgba(0,0,0,0.12)",
                        "shadowOffsetY": 2
                    },
                    "cursor": "pointer",
                    "info": "toggle_dest"
                },
                {
                    "type": "text",
                    "left": "center",
                    "top": "middle",
                    "style": {
                        "text": dest_label,
                        "fill": dest_text_color,
                        "font": "600 11px sans-serif"
                    },
                    "cursor": "pointer",
                    "info": "toggle_dest"
                }
            ]
        }
    ]

    options = {
        "backgroundColor": "#fbfaf6",
        "graphic": graphic_buttons,
        "grid": {
            "show": True,
            "borderColor": "#b0bec5",
            "borderWidth": 1.5,
            "left": 35,
            "right": 35,
            "top": 55,
            "bottom": 48
        },
        "legend": {
            "data": ["Rooms", "Room names", "Gates", "Gate names"],
            "bottom": 6,
            "left": "center",
            "orient": "horizontal",
            "itemGap": 10,
            "itemWidth": 14,
            "itemHeight": 10,
            "textStyle": {
                "color": "#263238",
                "fontSize": 10
            }
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

    action = event.get("graphicAction")
    if action == "toggle_origin":
        curr = st.session_state.get("click_target")
        st.session_state.click_target = None if curr == "origin" else "origin"
        return True
    elif action == "toggle_dest":
        curr = st.session_state.get("click_target")
        st.session_state.click_target = None if curr == "destination" else "destination"
        return True

    target_key = st.session_state.get("click_target")
    if not target_key:
        return False

    click_pt = event.get("clickCoord")
    edge = event.get("edge")

    if target_key == "origin":
        if edge and click_pt:
            u, v = edge
            pos_u = st.session_state.positions[u]
            pos_v = st.session_state.positions[v]
            snapped_x, snapped_y = project_point_on_segment(click_pt, pos_u, pos_v)
            st.session_state.custom_origin = {
                "point": (snapped_x, snapped_y),
                "edge": (u, v)
            }
            st.session_state.origin = "Custom"
            st.session_state.click_target = "destination"
            return True
        elif event.get("roomId"):
            st.session_state.origin = int(event["roomId"])
            st.session_state.custom_origin = None
            st.session_state.click_target = "destination"
            return True

    elif target_key == "destination":
        selected_room = None
        if event.get("roomId"):
            selected_room = int(event["roomId"])
        elif click_pt:
            selected_room = int(nearest_room(click_pt[0], click_pt[1], st.session_state.rooms, st.session_state.positions))

        if selected_room is not None:
            st.session_state.destination = selected_room
            st.session_state.click_target = None
            return True

    return False


def main():
    st.set_page_config(page_title="AB Ground Navigation", layout="wide")
    
    graph, positions, rooms, routes = load_map_data()

    st.session_state.rooms = rooms
    st.session_state.positions = positions

    labels = dict(zip(rooms.Room_ID, rooms.label))
    labels["-"] = "-"
    labels["Custom"] = "Custom (Path)"

    room_options = ["-"] + (["Custom"] if st.session_state.get("origin") == "Custom" else []) + list(rooms.Room_ID)

    if "origin" not in st.session_state:
        st.session_state.origin = "-"
    if "destination" not in st.session_state:
        st.session_state.destination = "-"
    if "click_target" not in st.session_state:
        st.session_state.click_target = "origin"

    st.title("AB Ground Floor Navigation")
    st.caption("Choose rooms or click a corridor to place the destination location.")
    
    controls, map_column = st.columns([1, 2.7], gap="large")
    
    with controls:
        default_origin = st.session_state.get("origin", "-")
        default_dest = st.session_state.get("destination", "-")

        origin = st.selectbox(
            "Starting room",
            room_options,
            index=room_options.index(default_origin) if default_origin in room_options else 0,
            format_func=labels.get
        )
        if origin != default_origin:
            st.session_state.origin = origin
            if origin != "-":
                st.session_state.click_target = "destination"
            st.rerun()

        destination = st.selectbox(
            "Destination room",
            room_options,
            index=room_options.index(default_dest) if default_dest in room_options else 0,
            format_func=labels.get
        )
        if destination != default_dest:
            st.session_state.destination = destination
            if destination != "-":
                st.session_state.click_target = None
            st.rerun()

        route_coords = None
        route_distance = None

        if origin == "-" or destination == "-" or origin is None or destination is None:
            pass

        elif origin == destination:
            st.info("Choose two different rooms to show a route.")

        elif origin == "Custom":
            custom = st.session_state.get("custom_origin")
            dest_node = int(destination)
            if custom and dest_node in graph:
                pt = custom["point"]
                u, v = custom["edge"]

                dist_p_u = euclidean_dist(pt, positions[u])
                dist_p_v = euclidean_dist(pt, positions[v])

                d_u, path_u = nx.bidirectional_dijkstra(graph, u, dest_node, weight="weight")
                total_u = dist_p_u + d_u

                d_v, path_v = nx.bidirectional_dijkstra(graph, v, dest_node, weight="weight")
                total_v = dist_p_v + d_v

                if total_u < total_v and path_u:
                    route_distance = total_u
                    route_coords = [list(pt)] + [[positions[n][0], positions[n][1]] for n in path_u]
                elif path_v:
                    route_distance = total_v
                    route_coords = [list(pt)] + [[positions[n][0], positions[n][1]] for n in path_v]
                else:
                    st.warning("No path found to destination.")

        else:
            orig_node = int(origin)
            dest_node = int(destination)

            cached = routes.get((orig_node, dest_node))
            if cached:
                route_distance, best_path = cached[0], cached[1]
            else:
                route_distance, best_path = nx.bidirectional_dijkstra(graph, orig_node, dest_node, weight="weight")

            if best_path:
                route_coords = [[positions[n][0], positions[n][1]] for n in best_path]
            else:
                st.warning("No path found between selected rooms.")

        if route_distance is not None:
            st.metric(label="Route distance", value=f"{route_distance:.2f} m")

    with map_column:
        options = get_echarts_options(graph, positions, rooms, route_coords, origin, destination)

        all_x, all_y = zip(*positions.values())
        x_span = max(all_x) - min(all_x)
        y_span = max(all_y) - min(all_y)
        min_x, max_x = min(all_x) - x_span * 0.08, max(all_x) + x_span * 0.08
        min_y, max_y = min(all_y) - y_span * 0.08, max(all_y) + y_span * 0.08

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
            "click": f"""function(params) {{
                var gInfo = params.info || (params.target && params.target.info);
                if (params.componentType === 'graphic' || gInfo) {{
                    return {{
                        graphicAction: gInfo,
                        dataZoom: window._mapZoom || null
                    }};
                }}

                var dom = document.querySelector('div[_echarts_instance_]') || document.querySelector('.echarts-for-react');
                var clickPt = null;

                var pe = params.event || {{}};
                var nativeEvt = pe.event || pe;
                var rect = dom ? dom.getBoundingClientRect() : null;

                var touch = (nativeEvt.changedTouches && nativeEvt.changedTouches[0]) ||
                            (nativeEvt.touches && nativeEvt.touches[0]) ||
                            nativeEvt;

                var clientX = touch.clientX !== undefined ? touch.clientX : null;
                var clientY = touch.clientY !== undefined ? touch.clientY : null;

                var px = (pe.zrX !== undefined) ? pe.zrX : (
                        (pe.offsetX !== undefined) ? pe.offsetX : (
                        (rect && clientX !== null) ? clientX - rect.left : null));

                var py = (pe.zrY !== undefined) ? pe.zrY : (
                        (pe.offsetY !== undefined) ? pe.offsetY : (
                        (rect && clientY !== null) ? clientY - rect.top : null));

                if (px !== null && py !== null && dom) {{
                    var zx = (window._mapZoom && window._mapZoom[0]) ? window._mapZoom[0] : {{start: 0, end: 100}};
                    var zy = (window._mapZoom && window._mapZoom[1]) ? window._mapZoom[1] : {{start: 0, end: 100}};

                    var minX = {min_x}, maxX = {max_x};
                    var minY = {min_y}, maxY = {max_y};

                    var curMinX = minX + (maxX - minX) * (zx.start / 100.0);
                    var curMaxX = minX + (maxX - minX) * (zx.end / 100.0);
                    var curMinY = minY + (maxY - minY) * (zy.start / 100.0);
                    var curMaxY = minY + (maxY - minY) * (zy.end / 100.0);

                    var gridLeft = 40;
                    var gridRight = (dom.clientWidth || (rect ? rect.width : 0)) - 40;
                    var gridTop = 60;
                    var gridBottom = (dom.clientHeight || (rect ? rect.height : 0)) - 30;

                    var normX = (px - gridLeft) / (gridRight - gridLeft);
                    var normY = (gridBottom - py) / (gridBottom - gridTop);

                    clickPt = [
                        curMinX + normX * (curMaxX - curMinX),
                        curMinY + normY * (curMaxY - curMinY)
                    ];
                }}

                return {{
                    roomId: params.data ? params.data.roomId : null,
                    value: params.value,
                    coords: (params.data && params.data.coords) ? params.data.coords : null,
                    edge: (params.data && params.data.edge) ? params.data.edge : null,
                    clickCoord: clickPt,
                    dataZoom: window._mapZoom || null
                }};
            }}"""
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