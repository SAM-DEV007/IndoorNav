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

    target_key = st.session_state.get("click_target")
    if not target_key:
        return False

    click_pt = event.get("clickCoord")
    edge = event.get("edge")
    coords = event.get("coords")

    if not edge and coords and len(coords) == 2:
        c1, c2 = coords[0], coords[1]
        matched_u, matched_v = None, None
        for node, pos in st.session_state.positions.items():
            if math.isclose(pos[0], c1[0], abs_tol=1e-3) and math.isclose(pos[1], c1[1], abs_tol=1e-3):
                matched_u = node
            elif math.isclose(pos[0], c2[0], abs_tol=1e-3) and math.isclose(pos[1], c2[1], abs_tol=1e-3):
                matched_v = node
        if matched_u is not None and matched_v is not None:
            edge = [matched_u, matched_v]

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

    st.markdown(
        """
        <style>
        button[kind="primary"] {
            background-color: #2ca25f !important;
            border-color: #2ca25f !important;
            color: #ffffff !important;
        }
        button[kind="primary"]:hover {
            background-color: #238b50 !important;
            border-color: #238b50 !important;
            color: #ffffff !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
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
        click_target = st.session_state.get("click_target")

        c1, c2 = st.columns([3.5, 1.5])
        with c1:
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

        with c2:
            st.write("")
            st.write("")
            is_origin_active = click_target == "origin"
            if st.button(
                "Active" if is_origin_active else "Set on map",
                key="btn_origin",
                type="primary" if is_origin_active else "secondary"
            ):
                st.session_state.click_target = None if is_origin_active else "origin"
                st.rerun()

        c3, c4 = st.columns([3.5, 1.5])
        with c3:
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

        with c4:
            st.write("")
            st.write("")
            is_dest_active = click_target == "destination"
            if st.button(
                "Active" if is_dest_active else "Set on map",
                key="btn_dest",
                type="primary" if is_dest_active else "secondary"
            ):
                st.session_state.click_target = None if is_dest_active else "destination"
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
                var dom = document.querySelector('div[_echarts_instance_]') || document.querySelector('.echarts-for-react');
                var clickPt = null;

                var pe = params.event || {{}};
                var nativeEvt = pe.event || pe;
                var rect = dom ? dom.getBoundingClientRect() : null;
                var px = (pe.zrX !== undefined) ? pe.zrX : ((pe.offsetX !== undefined) ? pe.offsetX : (rect && nativeEvt.clientX !== undefined ? nativeEvt.clientX - rect.left : null));
                var py = (pe.zrY !== undefined) ? pe.zrY : ((pe.offsetY !== undefined ? pe.offsetY : (rect && nativeEvt.clientY !== undefined ? nativeEvt.clientY - rect.top : null)));

                var ec = null;
                if (window.echarts && dom) {{
                    try {{ ec = window.echarts.getInstanceByDom(dom); }} catch(e) {{}}
                }}
                if (!ec && dom) {{
                    for (var k in dom) {{
                        if (k.indexOf('__reactFiber') === 0 || k.indexOf('__reactInternalInstance') === 0) {{
                            var f = dom[k];
                            while (f) {{
                                if (f.stateNode) {{
                                    if (typeof f.stateNode.getEchartsInstance === 'function') {{
                                        ec = f.stateNode.getEchartsInstance();
                                        break;
                                    }}
                                    if (f.stateNode.echartsInstance) {{
                                        ec = f.stateNode.echartsInstance;
                                        break;
                                    }}
                                }}
                                f = f.return;
                            }}
                        }}
                        if (ec) break;
                    }}
                }}

                if (ec && px !== null && py !== null) {{
                    try {{
                        clickPt = ec.convertFromPixel({{gridIndex: 0}}, [px, py]);
                    }} catch(e) {{}}
                }}

                if (!clickPt && px !== null && py !== null && dom) {{
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