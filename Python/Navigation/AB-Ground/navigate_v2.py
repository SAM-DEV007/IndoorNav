import ast
import math

from pathlib import Path
from streamlit_echarts import st_echarts
from scipy.spatial import cKDTree

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
    room_ids = rooms["Room_ID"].tolist()
    room_tree = cKDTree([positions[room_id] for room_id in room_ids])
    
    cache = pd.read_csv(CACHE_PATH)
    cache["Path"] = cache["Path"].map(ast.literal_eval)

    routes = {}
    for row in cache.itertuples(index=False):
        path = [int(node) for node in row.Path]
        routes[(int(row.Source), int(row.Target))] = (float(row.Distance), path)
        routes[(int(row.Target), int(row.Source))] = (float(row.Distance), path[::-1])
    return graph, positions, rooms, routes, room_ids, room_tree


def is_mobile_device():
    if st.session_state.get("is_mobile") is not None:
        return st.session_state.is_mobile
    try:
        if hasattr(st, "context") and hasattr(st.context, "headers"):
            ua = st.context.headers.get("user-agent", "").lower()
            sec_mob = st.context.headers.get("sec-ch-ua-mobile", "")
            if sec_mob == "?1" or any(k in ua for k in ["mobile", "android", "iphone", "ipad"]):
                return True
    except Exception:
        pass
    return False


def calculate_turn_direction(v1, v2):
    dx1, dy1 = v1
    dx2, dy2 = v2
    
    ang1 = math.atan2(dy1, dx1)
    ang2 = math.atan2(dy2, dx2)
    diff = math.degrees(ang2 - ang1)
    diff = (diff + 180) % 360 - 180 

    if -25 <= diff <= 25:
        return "straight", "Continue straight", "↑"
    elif 25 < diff <= 65:
        return "slight_left", "Slight left", "↖"
    elif 65 < diff <= 120:
        return "left", "Turn left", "←"
    elif 120 < diff <= 165:
        return "sharp_left", "Sharp left", "↙"
    elif -65 <= diff < -25:
        return "slight_right", "Slight right", "↗"
    elif -120 <= diff < -65:
        return "right", "Turn right", "→"
    elif -165 <= diff < -120:
        return "sharp_right", "Sharp right", "↘"
    else:
        return "uturn", "Make a U-turn", "↩"


def get_intersection_rooms(pt, rooms, positions, labels, start_label, dest_label, graph=None, room_ids=None, room_tree=None, proximity_radius=5.0):
    connected = []
    
    curr_node = None
    for n, pos in positions.items():
        if math.hypot(pos[0] - pt[0], pos[1] - pt[1]) < 0.15:
            curr_node = n
            break

    if graph and curr_node is not None and curr_node in graph:
        for nbr in graph.neighbors(curr_node):
            if nbr in labels:
                r_name = labels.get(nbr, f"Room {nbr}")
                if r_name not in connected and r_name not in (start_label, dest_label):
                    connected.append(r_name)

    if not connected:
        if room_tree is None or room_ids is None:
            return connected

        nearby_indexes = room_tree.query_ball_point(pt, proximity_radius)
        connected_with_distance = []
        for index in nearby_indexes:
            r_id = room_ids[index]
            r_name = labels.get(r_id, f"Room {r_id}")
            if r_name in (start_label, dest_label):
                continue

            pos = positions[r_id]
            distance = math.hypot(pt[0] - pos[0], pt[1] - pos[1])
            connected_with_distance.append((r_name, distance))

        connected_with_distance.sort(key=lambda item: item[1])
        connected = [r_name for r_name, _ in connected_with_distance]

    return connected


def generate_directions(path_coords, rooms, positions, labels, start_label="Start", dest_label="Destination", graph=None, room_ids=None, room_tree=None):
    if not path_coords or len(path_coords) < 2:
        return []

    start_marker_html = '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#ffffff; border:3px solid #000000; box-sizing:border-box;"></span>'
    dest_marker_html = '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#2ca25f; border:2px solid #ffffff; outline:1.5px solid #2ca25f; box-sizing:border-box;"></span>'

    directions = []

    start_pt = path_coords[0]
    next_pt = path_coords[1]
    first_v = (next_pt[0] - start_pt[0], next_pt[1] - start_pt[1])
    first_dist = math.hypot(first_v[0], first_v[1])
    
    start_conn = get_intersection_rooms(start_pt, rooms, positions, labels, start_label, dest_label, graph=graph, room_ids=room_ids, room_tree=room_tree)
    if start_conn and start_conn[0] != start_label:
        start_desc = f"Near {start_conn[0]}"
    else:
        start_desc = "Head down the hallway"

    curr_instruction = {
        "icon": start_marker_html,
        "action": f"Depart from {start_label}",
        "landmark": start_desc,
        "distance": first_dist
    }

    for i in range(1, len(path_coords) - 1):
        p_prev = path_coords[i - 1]
        p_curr = path_coords[i]
        p_next = path_coords[i + 1]

        v_in = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v_out = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        seg_dist = math.hypot(v_out[0], v_out[1])

        m_code, m_label, m_icon = calculate_turn_direction(v_in, v_out)

        if m_code == "straight":
            curr_instruction["distance"] += seg_dist
        else:
            directions.append(curr_instruction)

            conn_rooms = get_intersection_rooms(p_curr, rooms, positions, labels, start_label, dest_label, graph=graph, room_ids=room_ids, room_tree=room_tree)
            if conn_rooms:
                if len(conn_rooms) == 1:
                    room_phrase = f"near {conn_rooms[0]}"
                elif len(conn_rooms) == 2:
                    room_phrase = f"near {conn_rooms[0]} and {conn_rooms[1]}"
                else:
                    room_phrase = f"near {conn_rooms[0]}"

                action_title = f"{m_label} {room_phrase}"
                sub_text = "Follow hallway corridor"
            else:
                action_title = m_label
                sub_text = "Follow hallway corridor"

            arrow_html = f'<span style="font-size: 14px; font-weight: 800; color: #1a1a1a; line-height: 1;">{m_icon}</span>'

            curr_instruction = {
                "icon": arrow_html,
                "action": action_title,
                "landmark": sub_text,
                "distance": seg_dist
            }

    directions.append(curr_instruction)

    directions.append({
        "icon": dest_marker_html,
        "action": f"Arrive at {dest_label}",
        "landmark": "Destination is ahead",
        "distance": 0
    })

    return directions


def render_directions_ui(directions):
    if not directions:
        return

    items_html = []
    for i, step in enumerate(directions):
        is_last = (i == len(directions) - 1)
        border_style = "" if is_last else "border-left: 2px dashed #b0bec5;"
        dist_badge = (
            f'<span style="font-size: 11px; font-weight: 600; color: #1976d2; background: #e3f2fd; padding: 2px 6px; border-radius: 4px; white-space: nowrap; flex-shrink: 0;">{step["distance"]:.2f} m</span>'
            if step["distance"] > 0 else ""
        )

        item = f"""
<div style="position: relative; padding-left: 24px; padding-bottom: {'4px' if is_last else '14px'}; {border_style} margin-left: 10px;">
    <div style="position: absolute; left: -11px; top: -1px; width: 20px; height: 20px; border-radius: 50%; background: #ffffff; display: flex; align-items: center; justify-content: center; box-shadow: 0 1px 3px rgba(0,0,0,0.12); border: 1px solid #cfd8dc;">
        {step["icon"]}
    </div>
    <div style="display: flex; justify-content: space-between; align-items: baseline;">
        <span style="font-weight: 600; font-size: 13px; color: #212121;">{step["action"]}</span>
        {dist_badge}
    </div>
    <div style="font-size: 11px; color: #616161; margin-top: 2px;">
        {step["landmark"]}
    </div>
</div>"""
        items_html.append(item)

    full_html = f"""
<div style="margin-top: 15px; border-radius: 8px; background-color: #ffffff; border: 1px solid #e0e0e0; padding: 12px; font-family: sans-serif;">
    <div style="font-weight: 700; font-size: 14px; color: #263238; margin-bottom: 12px; display: flex; align-items: center; gap: 6px;">
        Step-by-Step Navigation
    </div>
    {''.join(items_html)}
</div>"""

    if hasattr(st, "html"):
        st.html(full_html)
    else:
        st.markdown(full_html, unsafe_allow_html=True)


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
            "clip": True,
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

    start_active = click_target == "origin"
    start_bg = "#2ca25f" if start_active else "#ffffff"
    start_border = "#2ca25f" if start_active else "#b0bec5"
    start_text_color = "#ffffff" if start_active else "#263238"
    start_label = "Start: Active" if start_active else "Set Start"

    dest_active = click_target == "destination"
    dest_bg = "#2ca25f" if dest_active else "#ffffff"
    dest_border = "#2ca25f" if dest_active else "#b0bec5"
    dest_text_color = "#ffffff" if dest_active else "#263238"
    dest_label = "Destination: Active" if dest_active else "Set Destination"

    btn_h = 28
    btn_clear_w = 85
    btn_start_w = 95
    btn_dest_w = 125
    gap = 8

    total_w = btn_clear_w + gap + btn_start_w + gap + btn_dest_w

    graphic_buttons = [
        {
            "type": "group",
            "left": "center",
            "top": 12,
            "width": total_w,
            "height": btn_h,
            "z": 100,
            "children": [
                {
                    "type": "group",
                    "left": 0,
                    "top": 0,
                    "width": btn_clear_w,
                    "height": btn_h,
                    "cursor": "pointer",
                    "info": "clear_path",
                    "children": [
                        {
                            "type": "rect",
                            "left": "center",
                            "top": "middle",
                            "shape": {"width": btn_clear_w, "height": btn_h, "r": 5},
                            "style": {
                                "fill": "#ffffff",
                                "stroke": "#e63946",
                                "lineWidth": 1.5,
                                "shadowBlur": 4,
                                "shadowColor": "rgba(0,0,0,0.12)",
                                "shadowOffsetY": 2
                            },
                            "cursor": "pointer",
                            "info": "clear_path"
                        },
                        {
                            "type": "text",
                            "left": "center",
                            "top": "middle",
                            "style": {
                                "text": "Clear Path",
                                "fill": "#e63946",
                                "font": "600 11px sans-serif"
                            },
                            "cursor": "pointer",
                            "info": "clear_path"
                        }
                    ]
                },
                {
                    "type": "group",
                    "left": btn_clear_w + gap,
                    "top": 0,
                    "width": btn_start_w,
                    "height": btn_h,
                    "cursor": "pointer",
                    "info": "toggle_origin",
                    "children": [
                        {
                            "type": "rect",
                            "left": "center",
                            "top": "middle",
                            "shape": {"width": btn_start_w, "height": btn_h, "r": 5},
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
                    "left": btn_clear_w + gap + btn_start_w + gap,
                    "top": 0,
                    "width": btn_dest_w,
                    "height": btn_h,
                    "cursor": "pointer",
                    "info": "toggle_dest",
                    "children": [
                        {
                            "type": "rect",
                            "left": "center",
                            "top": "middle",
                            "shape": {"width": btn_dest_w, "height": btn_h, "r": 5},
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
            "top": 50,
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


def handle_map_click(clicked_data, rooms, positions):
    if not clicked_data or not isinstance(clicked_data, dict):
        return False

    event = clicked_data.get("chart_event", clicked_data)
    if not isinstance(event, dict):
        return False

    dz = event.get("dataZoom")
    if dz and isinstance(dz, list) and len(dz) >= 2 and dz[0] and dz[1]:
        st.session_state.map_zoom = dz

    action = event.get("graphicAction")
    if action == "clear_path":
        st.session_state.origin = "-"
        st.session_state.destination = "-"
        st.session_state.custom_origin = None
        st.session_state.click_target = None
        return True
    elif action == "toggle_origin":
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
            pos_u = positions[u]
            pos_v = positions[v]
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
            selected_room = int(nearest_room(click_pt[0], click_pt[1], rooms, positions))

        if selected_room is not None:
            st.session_state.destination = selected_room
            st.session_state.click_target = None
            return True

    return False


def main():
    st.set_page_config(page_title="AB Ground Navigation", layout="wide")
    
    graph, positions, rooms, routes, room_ids, room_tree = load_map_data()

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

                try:
                    d_u, path_u = nx.bidirectional_dijkstra(graph, u, dest_node, weight="weight")
                    total_u = dist_p_u + d_u
                except nx.NetworkXNoPath:
                    total_u, path_u = float('inf'), None
                try:
                    d_v, path_v = nx.bidirectional_dijkstra(graph, v, dest_node, weight="weight")
                    total_v = dist_p_v + d_v
                except nx.NetworkXNoPath:
                    total_v, path_v = float('inf'), None

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
                try:
                    route_distance, best_path = nx.bidirectional_dijkstra(graph, orig_node, dest_node, weight="weight")
                except nx.NetworkXNoPath:
                    route_distance, best_path = None, None

            if best_path:
                route_coords = [[positions[n][0], positions[n][1]] for n in best_path]
            else:
                st.warning("No path found between selected rooms.")

        if route_distance is not None and route_coords:
            st.metric(label="Route distance", value=f"{route_distance:.2f} m")

            if origin == "Custom":
                start_lbl = "Custom Pin"
            else:
                start_lbl = labels.get(origin, labels.get(int(origin) if str(origin).isdigit() else origin, "Start"))

            dest_lbl = labels.get(destination, labels.get(int(destination) if str(destination).isdigit() else destination, "Destination"))

            active_rooms = st.session_state.get("rooms", rooms if "rooms" in locals() else [])
            active_positions = st.session_state.get("positions", positions if "positions" in locals() else {})

            directions = generate_directions(
                path_coords=route_coords,
                rooms=active_rooms,
                positions=active_positions,
                labels=labels,
                start_label=start_lbl,
                dest_label=dest_lbl,
                graph=graph,
                room_ids=room_ids,
                room_tree=room_tree,
            )

            render_directions_ui(directions)

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

                var gridLeft = 35;
                var gridRight = (dom.clientWidth || (rect ? rect.width : 0)) - 35;
                var gridTop = 55;
                var gridBottom = (dom.clientHeight || (rect ? rect.height : 0)) - 48;

                if (px === null || py === null || px < gridLeft || px > gridRight || py < gridTop || py > gridBottom) {{
                    return {{
                        roomId: null,
                        value: null,
                        coords: null,
                        edge: null,
                        clickCoord: null,
                        dataZoom: window._mapZoom || null
                    }};
                }}

                var zx = (window._mapZoom && window._mapZoom[0]) ? window._mapZoom[0] : {{start: 0, end: 100}};
                var zy = (window._mapZoom && window._mapZoom[1]) ? window._mapZoom[1] : {{start: 0, end: 100}};

                var minX = {min_x}, maxX = {max_x};
                var minY = {min_y}, maxY = {max_y};

                var curMinX = minX + (maxX - minX) * (zx.start / 100.0);
                var curMaxX = minX + (maxX - minX) * (zx.end / 100.0);
                var curMinY = minY + (maxY - minY) * (zy.start / 100.0);
                var curMaxY = minY + (maxY - minY) * (zy.end / 100.0);

                var normX = (px - gridLeft) / (gridRight - gridLeft);
                var normY = (gridBottom - py) / (gridBottom - gridTop);

                var clickPt = [
                    curMinX + normX * (curMaxX - curMinX),
                    curMinY + normY * (curMaxY - curMinY)
                ];

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
            height="350px" if is_mobile_device() else "650px",
            key="floorplan"
        )

        if handle_map_click(clicked_data, rooms, positions):
            st.rerun()


if __name__ == "__main__":
    main()