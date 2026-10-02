import ast
import json
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

    manual_labels = {
        3: "Discussion room (near Waiting room)",
        4: "Staircase (near Waiting room)",
        12: "Discussion room (near Main Audi AB012)",
        21: "Staircase (near Ab 004)",
        23: "Lift (near Ab 004)",
        28: "Staircase (near Ab 021)",
        30: "Lift (near Ab 022 cabins)",
        32: "Stair (near Audi backdoor)",
        34: "Lift (near Audi backdoor)",
        37: "Stairs (near Ab 015 dsw office)",
    }

    for room_id, label in manual_labels.items():
        rooms.loc[rooms["Room_ID"] == room_id, "label"] = label

    room_ids = rooms["Room_ID"].tolist()
    room_tree = cKDTree([positions[room_id] for room_id in room_ids])

    node_ids = list(positions.keys())
    node_tree = cKDTree([positions[node_id] for node_id in node_ids])
    
    cache = pd.read_csv(CACHE_PATH)
    cache["Path"] = cache["Path"].map(ast.literal_eval)

    routes = {}
    for row in cache.itertuples(index=False):
        path = [int(node) for node in row.Path]
        routes[(int(row.Source), int(row.Target))] = (float(row.Distance), path)
        routes[(int(row.Target), int(row.Source))] = (float(row.Distance), path[::-1])

    return graph, positions, rooms, routes, room_ids, room_tree, node_ids, node_tree


def hash_networkx_graph(graph: nx.Graph):
    return (len(graph.nodes), len(graph.edges))


@st.cache_data(hash_funcs={nx.Graph: hash_networkx_graph})
def build_static_map_data(graph: nx.Graph, positions, rooms):
    base_lines = []

    for left, right in graph.edges:
        x1, y1 = positions[left]
        x2, y2 = positions[right]

        base_lines.append({
            "coords": [[x1, y1], [x2, y2]],
            "edge": [int(left), int(right)]
        })

    room_records = rooms[~rooms.Room_ID.isin([1, 15])]
    room_boxes = []
    room_labels = []

    for row in room_records.itertuples(index=False):
        room_id = int(row.Room_ID)
        x, y = positions[room_id]

        layout = room_label_layout(room_id, graph, positions)
        text = wrap_label(str(row.Room_Name))
        lines = text.split("\n")
        text_w = max(len(l) for l in lines) * 4.0
        text_h = len(lines) * 9.0

        room_boxes.append({
            "id": f"room_box_{room_id}",
            "value": [x, y],
            "roomId": room_id,
            "name": row.Room_Name,
            "textW": text_w,
            "textH": text_h,
            "layoutPos": layout["position"],
            "layoutDist": layout["distance"],
            "layoutOffset": layout["offset"]
        })

        room_labels.append({
            "id": f"room_label_{room_id}",
            "value": [x, y],
            "roomId": room_id,
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

    def priority(r):
        name = str(r["name"]).lower()
        if any(k in name for k in ["audi", "waiting", "dsw", "office", "entry", "hall"]):
            return 0
        if any(k in name for k in ["lab", "discussion", "stair", "lift"]):
            return 1
        return 2

    room_boxes.sort(key=lambda r: (priority(r), r["roomId"]))
    room_labels.sort(key=lambda r: (priority(r), r["roomId"]))

    gate_boxes = [
        {
            "id": "gate_box_1",
            "value": [positions[1][0], positions[1][1]],
            "roomId": 1,
            "name": "Front Gate",
            "itemStyle": {"color": "#e63946"}
        },
        {
            "id": "gate_box_15",
            "value": [positions[15][0], positions[15][1]],
            "roomId": 15,
            "name": "Back Gate",
            "itemStyle": {"color": "#111111"}
        }
    ]

    gate_labels = [
        {
            "id": "gate_label_1",
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
            "id": "gate_label_15",
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

    all_x, all_y = zip(*positions.values())
    x_span = max(all_x) - min(all_x)
    y_span = max(all_y) - min(all_y)
    x_pad = x_span * 0.08
    y_pad = y_span * 0.08

    bounds = {
        "min_x": min(all_x) - x_pad,
        "max_x": max(all_x) + x_pad,
        "min_y": min(all_y) - y_pad,
        "max_y": max(all_y) + y_pad
    }

    return base_lines, room_boxes, room_labels, gate_boxes, gate_labels, bounds


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


def get_intersection_rooms(pt, room_names, positions, start_label, dest_label, graph=None, room_ids=None, room_tree=None, node_ids=None, node_tree=None, proximity_radius=5.0):
    if node_tree is None or node_ids is None:
        return []
    
    connected = []
    
    distance, index = node_tree.query(pt)
    curr_node = node_ids[index] if distance < 0.15 else None

    if graph and curr_node is not None and curr_node in graph:
        for nbr in graph.neighbors(curr_node):
            if nbr in room_names:
                r_name = room_names[nbr]
                if r_name not in connected and r_name not in (start_label, dest_label):
                    connected.append(r_name)

    if not connected:
        if room_tree is None or room_ids is None:
            return connected

        nearby_indexes = room_tree.query_ball_point(pt, proximity_radius)
        connected_with_distance = []
        for index in nearby_indexes:
            r_id = room_ids[index]
            r_name = room_names.get(r_id, f"Room {r_id}")
            if r_name in (start_label, dest_label):
                continue

            pos = positions[r_id]
            distance = math.hypot(pt[0] - pos[0], pt[1] - pos[1])
            connected_with_distance.append((r_name, distance))

        connected_with_distance.sort(key=lambda item: item[1])
        connected = [r_name for r_name, _ in connected_with_distance]

    return connected


def generate_directions(path_coords, room_names, positions, start_label="Start", dest_label="Destination", graph=None, room_ids=None, room_tree=None, node_ids=None, node_tree=None):
    if not path_coords or len(path_coords) < 2:
        return []

    start_marker_html = '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#ffffff; border:3px solid #000000; box-sizing:border-box;"></span>'
    dest_marker_html = '<span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#2ca25f; border:2px solid #ffffff; outline:1.5px solid #2ca25f; box-sizing:border-box;"></span>'

    directions = []

    start_pt = path_coords[0]
    next_pt = path_coords[1]
    first_v = (next_pt[0] - start_pt[0], next_pt[1] - start_pt[1])
    first_dist = math.hypot(first_v[0], first_v[1])
    
    start_conn = get_intersection_rooms(start_pt, room_names, positions, start_label, dest_label, graph=graph, room_ids=room_ids, room_tree=room_tree, node_ids=node_ids, node_tree=node_tree)
    if start_conn and start_conn[0] != start_label:
        start_desc = f"Near {start_conn[0]}. Head down the hallway"
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

            conn_rooms = get_intersection_rooms(p_curr, room_names, positions, start_label, dest_label, graph=graph, room_ids=room_ids, room_tree=room_tree, node_ids=node_ids, node_tree=node_tree)
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


def nearest_room(x, y, room_ids, room_tree):
    _, index = room_tree.query((x, y))

    return room_ids[index]


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
    return "\n".join(lines)


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


def get_room_parts_bbox(r, scale_x, scale_y, cur_min_x, cur_max_y, box_r, grid_top):
    px = 35.0 + (r["value"][0] - cur_min_x) * scale_x
    py = grid_top + (cur_max_y - r["value"][1]) * scale_y

    box_bbox = (px - box_r, px + box_r, py - box_r, py + box_r)

    tw = r["textW"]
    th = r["textH"]
    pos = r["layoutPos"]
    dist = max(2.0, r["layoutDist"] * 0.4)
    ox = r["layoutOffset"][0] if r["layoutOffset"] else 0
    oy = r["layoutOffset"][1] if r["layoutOffset"] else 0

    if pos == "top":
        lx1 = px - tw / 2.0 + ox
        lx2 = px + tw / 2.0 + ox
        ly2 = py - box_r - dist
        ly1 = ly2 - th
    elif pos == "bottom":
        lx1 = px - tw / 2.0 + ox
        lx2 = px + tw / 2.0 + ox
        ly1 = py + box_r + dist
        ly2 = ly1 + th
    elif pos == "left":
        lx2 = px - box_r - dist
        lx1 = lx2 - tw
        ly1 = py - th / 2.0 + oy
        ly2 = py + th / 2.0 + oy
    else:
        lx1 = px + box_r + dist
        lx2 = lx1 + tw
        ly1 = py - th / 2.0 + oy
        ly2 = py + th / 2.0 + oy

    return {"box": box_bbox, "label": (lx1, lx2, ly1, ly2)}


def get_marker_parts_bbox(pt, name, scale_x, scale_y, cur_min_x, cur_max_y, marker_r, grid_top):
    px = 35.0 + (pt[0] - cur_min_x) * scale_x
    py = grid_top + (cur_max_y - pt[1]) * scale_y
    lines = name.split("\n")
    tw = max(len(l) for l in lines) * 4.0 + 4.0
    th = len(lines) * 9.0 + 2.0
    box_bbox = (px - marker_r, px + marker_r, py - marker_r, py + marker_r)
    ly2 = py - marker_r - 2.0
    ly1 = ly2 - th
    label_bbox = (px - tw / 2.0, px + tw / 2.0, ly1, ly2)
    return {"box": box_bbox, "label": label_bbox}


def boxes_overlap(b1, b2):
    return b1[0] < b2[1] and b1[1] > b2[0] and b1[2] < b2[3] and b1[3] > b2[2]


def room_collides_with(parts_a, parts_b):
    return (
        boxes_overlap(parts_a["box"], parts_b["box"])
        or boxes_overlap(parts_a["box"], parts_b["label"])
        or boxes_overlap(parts_a["label"], parts_b["box"])
        or boxes_overlap(parts_a["label"], parts_b["label"])
    )


def get_echarts_options(positions, route_coords, origin, destination, static_map_data):
    base_lines, room_boxes, room_labels, gate_boxes, gate_labels, bounds = static_map_data

    is_mobile = is_mobile_device()
    show_all = st.session_state.get("show_all", False)

    room_label_font_size = 6 if is_mobile else 8
    room_label_line_height = 8 if is_mobile else 10
    room_label_scale = 0.75 if is_mobile else 1.0

    path_width = 5 if is_mobile else 9
    route_width = 3 if is_mobile else 4.5
    room_size = 4.5 if is_mobile else 8
    gate_size = 9 if is_mobile else 13
    marker_orig_size = 7 if is_mobile else 10
    marker_dest_size = 9 if is_mobile else 13
    label_font_size = 7 if is_mobile else 8

    min_x = bounds["min_x"]
    max_x = bounds["max_x"]
    min_y = bounds["min_y"]
    max_y = bounds["max_y"]

    mz = st.session_state.get("map_zoom")

    zoom_x_start = mz[0]["start"] if mz and len(mz) > 0 and "start" in mz[0] else 0
    zoom_x_end = mz[0]["end"] if mz and len(mz) > 0 and "end" in mz[0] else 100
    zoom_y_start = mz[1]["start"] if mz and len(mz) > 1 and "start" in mz[1] else 0
    zoom_y_end = mz[1]["end"] if mz and len(mz) > 1 and "end" in mz[1] else 100

    name_lookup = {r["roomId"]: r["name"] for r in room_boxes}
    name_lookup[1] = "Front Gate"
    name_lookup[15] = "Back Gate"

    selected_ids = set()
    if str(origin).isdigit():
        selected_ids.add(int(origin))
    if str(destination).isdigit():
        selected_ids.add(int(destination))

    grid_top = 34 if is_mobile else 38
    grid_bottom = 50 if is_mobile else 38

    if show_all:
        visible_boxes = room_boxes
        visible_labels = [r for r in room_labels if r.get("roomId") not in selected_ids]
    else:
        chart_w = st.session_state.get("chart_width") or (375.0 if is_mobile else 1150.0)
        chart_h = st.session_state.get("chart_height") or (350.0 if is_mobile else 650.0)
        grid_w = chart_w - 70.0
        grid_h = chart_h - (grid_top + grid_bottom)
        span_x = max_x - min_x
        span_y = max_y - min_y
        cur_min_x = min_x + span_x * (zoom_x_start / 100.0)
        cur_max_x = min_x + span_x * (zoom_x_end / 100.0)
        cur_min_y = min_y + span_y * (zoom_y_start / 100.0)
        cur_max_y = min_y + span_y * (zoom_y_end / 100.0)

        scale_x = grid_w / max(cur_max_x - cur_min_x, 1e-5)
        scale_y = grid_h / max(cur_max_y - cur_min_y, 1e-5)

        kept_parts = []
        kept_ids = set()

        if origin == "Custom" and st.session_state.get("custom_origin"):
            m_pt = st.session_state.custom_origin["point"]
            kept_parts.append(get_marker_parts_bbox(m_pt, "Custom Start", scale_x, scale_y, cur_min_x, cur_max_y, marker_orig_size / 2.0, grid_top))
        elif str(origin).isdigit() and int(origin) in positions:
            m_pt = positions[int(origin)]
            m_name = wrap_label(str(name_lookup.get(int(origin), f"Room {origin}")))
            kept_parts.append(get_marker_parts_bbox(m_pt, m_name, scale_x, scale_y, cur_min_x, cur_max_y, marker_orig_size / 2.0, grid_top))

        if str(destination).isdigit() and int(destination) in positions:
            m_pt = positions[int(destination)]
            m_name = wrap_label(str(name_lookup.get(int(destination), f"Room {destination}")))
            kept_parts.append(get_marker_parts_bbox(m_pt, m_name, scale_x, scale_y, cur_min_x, cur_max_y, marker_dest_size / 2.0, grid_top))

        box_r = room_size / 2.0
        for r in room_boxes:
            rid = r["roomId"]
            if rid in selected_ids:
                continue

            r_parts = get_room_parts_bbox(r, scale_x, scale_y, cur_min_x, cur_max_y, box_r, grid_top)

            collides = False
            for kp in kept_parts:
                if room_collides_with(r_parts, kp):
                    collides = True
                    break

            if not collides:
                kept_ids.add(rid)
                kept_parts.append(r_parts)

        visible_boxes = [r for r in room_boxes if r["roomId"] in kept_ids]
        visible_labels = [r for r in room_labels if (r["roomId"] in kept_ids) and (r["roomId"] not in selected_ids)]

    visible_gate_labels = [g for g in gate_labels if g.get("roomId") not in selected_ids]
    visible_labels = [
        {
            **label,
            "label": {
                **label["label"],
                "fontSize": room_label_font_size,
                "lineHeight": room_label_line_height,
            }
        }
        for label in visible_labels
    ]

    click_target = st.session_state.get("click_target")
    is_active = bool(click_target)
    cursor_style = "pointer" if is_active else "grab"
    is_silent = not is_active

    series = [
        {
            "name": "Walkable path",
            "type": "lines",
            "coordinateSystem": "cartesian2d",
            "data": base_lines,
            "lineStyle": {
                "color": "#d2dbde",
                "width": path_width,
                "opacity": 1,
                "cap": "round",
                "join": "round"
            },
            "clip": True,
            "cursor": cursor_style,
            "silent": is_silent,
            "tooltip": {"show": False}
        },
        {
            "name": "Rooms",
            "type": "scatter",
            "symbol": "rect",
            "symbolSize": room_size,
            "itemStyle": {
                "color": "#2f8fbd",
                "borderColor": "#17465d",
                "borderWidth": 1
            },
            "clip": True,
            "data": visible_boxes,
            "cursor": cursor_style,
            "silent": False,
            "z": 20
        },
        {
            "name": "Room names",
            "type": "scatter",
            "symbol": "rect",
            "symbolSize": 0,
            "clip": True,
            "itemStyle": {"color": "#2f8fbd"},
            "labelLayout": {
                "hideOverlap": False
            },
            "data": visible_labels,
            "cursor": cursor_style,
            "silent": is_silent,
            "z": 21
        },
        {
            "name": "Gates",
            "type": "scatter",
            "symbol": "circle",
            "symbolSize": gate_size,
            "itemStyle": {
                "borderColor": "#202a2e",
                "borderWidth": 2
            },
            "clip": True,
            "data": gate_boxes,
            "cursor": cursor_style,
            "z": 22
        },
        {
            "name": "Gate names",
            "type": "scatter",
            "symbol": "circle",
            "symbolSize": 0,
            "clip": True,
            "itemStyle": {"color": "#202a2e"},
            "labelLayout": {
                "hideOverlap": False
            },
            "data": visible_gate_labels,
            "cursor": cursor_style,
            "silent": is_silent,
            "z": 23
        }
    ]

    if route_coords:
        series.append({
            "name": "Shortest route",
            "type": "lines",
            "coordinateSystem": "cartesian2d",
            "polyline": True,
            "data": [{"coords": route_coords}],
            "lineStyle": {
                "color": "#e4572e",
                "width": route_width,
                "opacity": 1,
                "cap": "round",
                "join": "round"
            },
            "clip": True,
            "z": 10,
            "silent": True,
            "tooltip": {"show": False}
        })

    marker_data = []

    if origin == "Custom" and st.session_state.get("custom_origin"):
        cx, cy = st.session_state.custom_origin["point"]
        marker_data.append({
            "id": "marker_custom_origin",
            "value": [cx, cy],
            "symbolSize": marker_orig_size,
            "itemStyle": {
                "color": "#ffffff",
                "borderColor": "#202a2e",
                "borderWidth": 2
            },
            "label": {
                "show": True,
                "formatter": "Custom Start",
                "position": "top",
                "distance": 4,
                "color": "#1f2937",
                "fontWeight": "bold",
                "fontSize": label_font_size,
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "borderColor": "#202a2e",
                "borderWidth": 1,
                "borderRadius": 3,
                "padding": [1, 3]
            }
        })
    elif origin not in (None, "-"):
        orig_id = int(origin)
        ox, oy = positions[orig_id]
        orig_name = wrap_label(str(name_lookup.get(orig_id, f"Room {orig_id}")))
        marker_data.append({
            "id": f"marker_origin_{orig_id}",
            "value": [ox, oy],
            "symbolSize": marker_orig_size,
            "itemStyle": {
                "color": "#ffffff",
                "borderColor": "#202a2e",
                "borderWidth": 2
            },
            "label": {
                "show": True,
                "formatter": orig_name,
                "position": "top",
                "distance": 4,
                "color": "#1f2937",
                "fontWeight": "bold",
                "fontSize": label_font_size,
                "backgroundColor": "rgba(255, 255, 255, 0.95)",
                "borderColor": "#202a2e",
                "borderWidth": 1,
                "borderRadius": 3,
                "padding": [1, 3]
            }
        })

    if destination not in (None, "-"):
        dest_id = int(destination)
        dx, dy = positions[dest_id]
        dest_name = wrap_label(str(name_lookup.get(dest_id, f"Room {dest_id}")))
        marker_data.append({
            "id": f"marker_dest_{dest_id}",
            "value": [dx, dy],
            "symbolSize": marker_dest_size,
            "itemStyle": {
                "color": "#2ca25f",
                "borderColor": "#202a2e",
                "borderWidth": 2
            },
            "label": {
                "show": True,
                "formatter": dest_name,
                "position": "top",
                "distance": 4,
                "color": "#0d3c26",
                "fontWeight": "bold",
                "fontSize": label_font_size,
                "backgroundColor": "#e8f5e9",
                "borderColor": "#2ca25f",
                "borderWidth": 1.5,
                "borderRadius": 3,
                "padding": [1, 3]
            }
        })

    series.append({
        "type": "scatter",
        "data": marker_data,
        "clip": True,
        "z": 35,
        "cursor": cursor_style,
        "silent": True,
        "tooltip": {"show": False}
    })

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

    btn_h = 17 if is_mobile else 22
    btn_start_w = 55 if is_mobile else 70
    btn_dest_w = 85 if is_mobile else 100
    btn_clear_w = 60 if is_mobile else 70
    gap = 4 if is_mobile else 6
    btn_font = f"600 {8 if is_mobile else 9.5}px sans-serif"
    btn_top = 7 if is_mobile else 8

    left_group_w = btn_start_w + gap + btn_dest_w

    show_all_bg = "#2ca25f" if show_all else "#ffffff"
    show_all_border = "#2ca25f" if show_all else "#b0bec5"
    show_all_text_color = "#ffffff" if show_all else "#263238"
    show_all_label = "Show All: ON" if show_all else "Show All: OFF"
    show_all_w = 60 if is_mobile else 68
    show_all_h = 16 if is_mobile else 19

    graphic_buttons = [
        {
            "type": "group",
            "left": 10 if is_mobile else 35,
            "top": btn_top,
            "width": left_group_w,
            "height": btn_h,
            "z": 100,
            "children": [
                {
                    "type": "group",
                    "left": 0,
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
                            "shape": {
                                "width": btn_start_w,
                                "height": btn_h,
                                "r": 3 if is_mobile else 4
                            },
                            "style": {
                                "fill": start_bg,
                                "stroke": start_border,
                                "lineWidth": 1.2,
                                "shadowBlur": 2,
                                "shadowColor": "rgba(0,0,0,0.1)",
                                "shadowOffsetY": 1
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
                                "font": btn_font
                            },
                            "cursor": "pointer",
                            "info": "toggle_origin"
                        }
                    ]
                },
                {
                    "type": "group",
                    "left": btn_start_w + gap,
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
                            "shape": {
                                "width": btn_dest_w,
                                "height": btn_h,
                                "r": 3 if is_mobile else 4
                            },
                            "style": {
                                "fill": dest_bg,
                                "stroke": dest_border,
                                "lineWidth": 1.2,
                                "shadowBlur": 2,
                                "shadowColor": "rgba(0,0,0,0.1)",
                                "shadowOffsetY": 1
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
                                "font": btn_font
                            },
                            "cursor": "pointer",
                            "info": "toggle_dest"
                        }
                    ]
                }
            ]
        },
        {
            "type": "group",
            "right": 10 if is_mobile else 35,
            "top": btn_top,
            "width": btn_clear_w,
            "height": btn_h,
            "z": 100,
            "cursor": "pointer",
            "info": "clear_path",
            "children": [
                {
                    "type": "rect",
                    "left": "center",
                    "top": "middle",
                    "shape": {
                        "width": btn_clear_w,
                        "height": btn_h,
                        "r": 3 if is_mobile else 4
                    },
                    "style": {
                        "fill": "#ffffff",
                        "stroke": "#e63946",
                        "lineWidth": 1.2,
                        "shadowBlur": 2,
                        "shadowColor": "rgba(0,0,0,0.1)",
                        "shadowOffsetY": 1
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
                        "font": btn_font
                    },
                    "cursor": "pointer",
                    "info": "clear_path"
                }
            ]
        },
        {
            "type": "group",
            "left": "center" if is_mobile else None,
            "right": None if is_mobile else 35,
            "bottom": 6,
            "width": show_all_w,
            "height": show_all_h,
            "z": 100,
            "cursor": "pointer",
            "info": "toggle_show_all",
            "children": [
                {
                    "type": "rect",
                    "left": "center",
                    "top": "middle",
                    "shape": {
                        "width": show_all_w,
                        "height": show_all_h,
                        "r": 3
                    },
                    "style": {
                        "fill": show_all_bg,
                        "stroke": show_all_border,
                        "lineWidth": 1.2,
                        "shadowBlur": 2,
                        "shadowColor": "rgba(0,0,0,0.1)",
                        "shadowOffsetY": 1
                    },
                    "cursor": "pointer",
                    "info": "toggle_show_all"
                },
                {
                    "type": "text",
                    "left": "center",
                    "top": "middle",
                    "style": {
                        "text": show_all_label,
                        "fill": show_all_text_color,
                        "font": f"600 {7.5 if is_mobile else 8.5}px sans-serif"
                    },
                    "cursor": "pointer",
                    "info": "toggle_show_all"
                }
            ]
        }
    ]

    return {
        "backgroundColor": "#fbfaf6",
        "graphic": graphic_buttons,
        "grid": {
            "show": False,
            "borderColor": "#b0bec5",
            "borderWidth": 1.5,
            "left": 35,
            "right": 35,
            "top": grid_top,
            "bottom": grid_bottom
        },
        "legend": {
            "data": [
                "Rooms",
                "Room names",
                "Gates",
                "Gate names"
            ],
            "bottom": 26 if is_mobile else 6,
            "left": "center" if is_mobile else 35,
            "orient": "horizontal",
            "itemGap": 8 if is_mobile else 12,
            "itemWidth": 12 if is_mobile else 14,
            "itemHeight": 8 if is_mobile else 10,
            "textStyle": {
                "color": "#263238",
                "fontSize": 9 if is_mobile else 10
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


def compute_auto_zoom(coords, bounds, min_span=25.0, padding=0.25):
    xs = [pt[0] for pt in coords]
    ys = [pt[1] for pt in coords]

    c_min_x, c_max_x = min(xs), max(xs)
    c_min_y, c_max_y = min(ys), max(ys)

    center_x = (c_min_x + c_max_x) / 2.0
    center_y = (c_min_y + c_max_y) / 2.0

    box_w = max(c_max_x - c_min_x, min_span)
    box_h = max(c_max_y - c_min_y, min_span)

    half_w = (box_w / 2.0) * (1.0 + padding)
    half_h = (box_h / 2.0) * (1.0 + padding)

    b_min_x, b_max_x = bounds["min_x"], bounds["max_x"]
    b_min_y, b_max_y = bounds["min_y"], bounds["max_y"]

    total_w = b_max_x - b_min_x
    total_h = b_max_y - b_min_y

    target_min_x = max(b_min_x, center_x - half_w)
    target_max_x = min(b_max_x, center_x + half_w)
    target_min_y = max(b_min_y, center_y - half_h)
    target_max_y = min(b_max_y, center_y + half_h)

    start_x = (target_min_x - b_min_x) / total_w * 100.0
    end_x = (target_max_x - b_min_x) / total_w * 100.0
    start_y = (target_min_y - b_min_y) / total_h * 100.0
    end_y = (target_max_y - b_min_y) / total_h * 100.0

    return [
        {"start": max(0.0, min(100.0, start_x)), "end": max(0.0, min(100.0, end_x))},
        {"start": max(0.0, min(100.0, start_y)), "end": max(0.0, min(100.0, end_y))}
    ]


def handle_map_click(clicked_data, positions, room_ids, room_tree):
    if not clicked_data or not isinstance(clicked_data, dict):
        return False

    event = clicked_data.get("chart_event", clicked_data)
    if not isinstance(event, dict):
        return False

    cw = event.get("chartWidth")
    ch = event.get("chartHeight")
    if cw and ch:
        st.session_state.chart_width = float(cw)
        st.session_state.chart_height = float(ch)

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
    elif action == "toggle_show_all":
        st.session_state.show_all = not st.session_state.get("show_all", False)
        return True

    target_key = st.session_state.get("click_target")
    if not target_key:
        return False

    click_pt = event.get("clickCoord")
    edge = event.get("edge")

    if target_key == "origin":
        if event.get("roomId"):
            selected_room = int(event["roomId"])
            st.session_state.origin = selected_room
            st.session_state.custom_origin = None
            st.session_state.click_target = "destination"
            return True
        elif edge and click_pt:
            u, v = edge
            pos_u = positions[u]
            pos_v = positions[v]
            snapped_x, snapped_y = project_point_on_segment(click_pt, pos_u, pos_v)

            dist, idx = room_tree.query((snapped_x, snapped_y))
            if dist <= 2.0:
                st.session_state.origin = int(room_ids[idx])
                st.session_state.custom_origin = None
            else:
                st.session_state.custom_origin = {
                    "point": (snapped_x, snapped_y),
                    "edge": (u, v)
                }
                st.session_state.origin = "Custom"

            st.session_state.click_target = "destination"
            return True
        elif click_pt:
            selected_room = int(nearest_room(click_pt[0], click_pt[1], room_ids, room_tree))
            st.session_state.origin = selected_room
            st.session_state.custom_origin = None
            st.session_state.click_target = "destination"
            return True

    elif target_key == "destination":
        selected_room = None
        if event.get("roomId"):
            selected_room = int(event["roomId"])
        elif click_pt:
            selected_room = int(nearest_room(click_pt[0], click_pt[1], room_ids, room_tree))

        if selected_room is not None:
            st.session_state.destination = selected_room
            st.session_state.click_target = None
            return True

    return False


def on_origin_change():
    st.session_state.origin = st.session_state.origin_select

    if st.session_state.origin != "-":
        st.session_state.click_target = "destination"


def on_destination_change():
    st.session_state.destination = st.session_state.destination_select

    if st.session_state.destination != "-":
        st.session_state.click_target = None


def main():
    graph, positions, rooms, routes, room_ids, room_tree, node_ids, node_tree = load_map_data()

    st.set_page_config(page_title="AB Ground Navigation", layout="wide")
    st.markdown("""
        <style>
        [data-testid="stAppViewContainer"],
        [data-testid="stAppViewBlockContainer"],
        [data-testid="stMain"],
        [data-testid="stHeader"] {
            opacity: 1 !important;
        }

        [data-stale="false"],
        [data-stale="true"] {
            opacity: 1 !important;
        }

        h1 {
            font-size: 2.2rem !important;
            margin-bottom: 0.2rem !important;
        }

        @media (max-width: 768px) {
            h1 {
                font-size: 1.8rem !important;
                margin-bottom: 0.1rem !important;
            }
            [data-testid="stCaptionContainer"] {
                font-size: 0.78rem !important;
                margin-bottom: 0.4rem !important;
            }
            [data-testid="stAppViewBlockContainer"] {
                padding-top: 1.5rem !important;
                padding-bottom: 1rem !important;
            }
        }
        </style>
        """, 
        unsafe_allow_html=True
    )

    labels = dict(zip(rooms.Room_ID, rooms.label))
    labels["-"] = "-"
    labels["Custom"] = "Custom (Path)"

    room_names = dict(zip(rooms.Room_ID, rooms.Room_Name))
    room_names["-"] = "-"
    room_names["Custom"] = "Custom (Path)"

    room_options_origin = ["-"] + (
        ["Custom"] if st.session_state.get("origin") == "Custom" else []
    ) + list(rooms.Room_ID)

    room_options_destination = ["-"] + list(rooms.Room_ID)

    if "origin" not in st.session_state:
        st.session_state.origin = "-"

    if "destination" not in st.session_state:
        st.session_state.destination = "-"

    if "click_target" not in st.session_state:
        st.session_state.click_target = "origin"

    if "show_all" not in st.session_state:
        st.session_state.show_all = False

    st.session_state.origin_select = st.session_state.origin
    st.session_state.destination_select = st.session_state.destination

    st.title("AB Ground Floor Navigation")
    st.caption("Choose rooms or click a corridor to place the destination location.")
    
    controls, map_column = st.columns([1, 2.7], gap="large")
    
    with controls:
        st.selectbox(
            "Starting room",
            room_options_origin,
            key="origin_select",
            format_func=labels.get,
            on_change=on_origin_change,
        )

        st.selectbox(
            "Destination room",
            room_options_destination,
            key="destination_select",
            format_func=labels.get,
            on_change=on_destination_change,
        )

        origin = st.session_state.origin
        destination = st.session_state.destination

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
                start_lbl = room_names.get(origin, room_names.get(int(origin) if str(origin).isdigit() else origin, "Start"))

            dest_lbl = room_names.get(destination, room_names.get(int(destination) if str(destination).isdigit() else destination, "Destination"))

            active_positions = st.session_state.get("positions", positions if "positions" in locals() else {})

            directions = generate_directions(
                path_coords=route_coords,
                room_names=room_names,
                positions=active_positions,
                start_label=start_lbl,
                dest_label=dest_lbl,
                graph=graph,
                room_ids=room_ids,
                room_tree=room_tree,
                node_ids=node_ids,
                node_tree=node_tree
            )

            render_directions_ui(directions)

    with map_column:
        if "render_token" not in st.session_state:
            st.session_state.render_token = 0
        st.session_state.render_token += 1

        static_map_data = build_static_map_data(graph, positions, rooms)
        options = get_echarts_options(positions, route_coords, origin, destination, static_map_data)

        _, all_boxes, all_labels, _, _, bounds = static_map_data
        min_x = bounds["min_x"]
        max_x = bounds["max_x"]
        min_y = bounds["min_y"]
        max_y = bounds["max_y"]

        selected_ids = []
        if str(origin).isdigit():
            selected_ids.append(int(origin))
        if str(destination).isdigit():
            selected_ids.append(int(destination))

        marker_items = []
        if origin == "Custom" and st.session_state.get("custom_origin"):
            marker_items.append({"pt": st.session_state.custom_origin["point"], "name": "Custom Start", "r": 3.5 if is_mobile_device() else 5.0})
        elif str(origin).isdigit() and int(origin) in positions:
            m_name = wrap_label(str(rooms.loc[rooms["Room_ID"] == int(origin), "Room_Name"].values[0] if (rooms["Room_ID"] == int(origin)).any() else f"Room {origin}"))
            marker_items.append({"pt": positions[int(origin)], "name": m_name, "r": 3.5 if is_mobile_device() else 5.0})
        if str(destination).isdigit() and int(destination) in positions:
            m_name = wrap_label(str(rooms.loc[rooms["Room_ID"] == int(destination), "Room_Name"].values[0] if (rooms["Room_ID"] == int(destination)).any() else f"Room {destination}"))
            marker_items.append({"pt": positions[int(destination)], "name": m_name, "r": 4.5 if is_mobile_device() else 6.5})

        boxes_json = json.dumps(all_boxes)
        labels_json = json.dumps(all_labels)
        selected_ids_json = json.dumps(selected_ids)
        marker_items_json = json.dumps(marker_items)
        show_all_val = "true" if st.session_state.get("show_all", False) else "false"

        grid_top = 34 if is_mobile_device() else 38
        grid_bottom = 50 if is_mobile_device() else 38

        filter_fn_body = f"""
            var isShowAll = {show_all_val};
            if (isShowAll) return;

            window._masterBoxes = {boxes_json};
            window._masterLabels = {labels_json};

            var dom = document.querySelector('div[_echarts_instance_]') || document.querySelector('.echarts-for-react');
            var chart = dom ? echarts.getInstanceByDom(dom) : null;
            if (!chart) return;

            var opt = chart.getOption();
            var dz = (opt && opt.dataZoom) ? opt.dataZoom : null;
            var zx = (dz && dz[0]) ? dz[0] : ((window._mapZoom && window._mapZoom[0]) ? window._mapZoom[0] : {{start: 0, end: 100}});
            var zy = (dz && dz[1]) ? dz[1] : ((window._mapZoom && window._mapZoom[1]) ? window._mapZoom[1] : {{start: 0, end: 100}});
            window._mapZoom = [{{start: zx.start, end: zx.end}}, {{start: zy.start, end: zy.end}}];

            var w = (dom.clientWidth || 1150) - 70;
            var h = (dom.clientHeight || 650) - {grid_top + grid_bottom};

            var curToken = {st.session_state.render_token};
            if (window._lastRenderToken !== curToken) {{
                window._lastRenderToken = curToken;
                window._lastProcessedZoomKey = null;
                window._lastShowAllState = null;
            }}

            var zoomKey = (zx ? zx.start.toFixed(2) + "_" + zx.end.toFixed(2) : "0_100") + "_" + (zy ? zy.start.toFixed(2) + "_" + zy.end.toFixed(2) : "0_100") + "_" + Math.round(w);
            if (window._lastProcessedZoomKey === zoomKey && window._lastShowAllState === isShowAll) return;
            window._lastProcessedZoomKey = zoomKey;
            window._lastShowAllState = isShowAll;

            var minX = {min_x}, maxX = {max_x};
            var minY = {min_y}, maxY = {max_y};
            var curMinX = minX + (maxX - minX) * ((zx ? zx.start : 0) / 100.0);
            var curMaxX = minX + (maxX - minX) * ((zx ? zx.end : 100) / 100.0);
            var curMinY = minY + (maxY - minY) * ((zy ? zy.start : 0) / 100.0);
            var curMaxY = minY + (maxY - minY) * ((zy ? zy.end : 100) / 100.0);

            var scaleX = w / Math.max(curMaxX - curMinX, 0.0001);
            var scaleY = h / Math.max(curMaxY - curMinY, 0.0001);

            var isMob = {"true" if is_mobile_device() else "false"};
            var fontScale = isMob ? 0.75 : 1.0;
            var boxR = (isMob ? 4.5 : 8.0) / 2.0;

            function getRoomParts(r) {{
                var px = 35 + (r.value[0] - curMinX) * scaleX;
                var py = {grid_top} + (curMaxY - r.value[1]) * scaleY;
                var bBox = [px - boxR, px + boxR, py - boxR, py + boxR];
                var tw = (r.textW || 18) * fontScale;
                var th = (r.textH || 9) * fontScale;
                var pos = r.layoutPos || "top", dist = Math.max(2, (r.layoutDist || 8) * 0.4);
                var ox = (r.layoutOffset && r.layoutOffset[0]) || 0;
                var oy = (r.layoutOffset && r.layoutOffset[1]) || 0;
                var lx1, lx2, ly1, ly2;
                if (pos === "top") {{
                    lx1 = px - tw / 2 + ox;
                    lx2 = px + tw / 2 + ox;
                    ly2 = py - boxR - dist;
                    ly1 = ly2 - th;
                }} else if (pos === "bottom") {{
                    lx1 = px - tw / 2 + ox;
                    lx2 = px + tw / 2 + ox;
                    ly1 = py + boxR + dist;
                    ly2 = ly1 + th;
                }} else if (pos === "left") {{
                    lx2 = px - boxR - dist;
                    lx1 = lx2 - tw;
                    ly1 = py - th / 2 + oy;
                    ly2 = py + th / 2 + oy;
                }} else {{
                    lx1 = px + boxR + dist;
                    lx2 = lx1 + tw;
                    ly1 = py - th / 2 + oy;
                    ly2 = py + th / 2 + oy;
                }}
                return {{ box: bBox, label: [lx1, lx2, ly1, ly2] }};
            }}

            function getMarkerParts(m) {{
                var px = 35 + (m.pt[0] - curMinX) * scaleX;
                var py = {grid_top} + (curMaxY - m.pt[1]) * scaleY;
                var mr = m.r;
                var lines = (m.name || "").split("\\n");
                var maxL = 0;
                for (var i = 0; i < lines.length; i++) {{
                    if (lines[i].length > maxL) maxL = lines[i].length;
                }}
                var tw = maxL * 4.0 + 4;
                var th = lines.length * 9.0 + 2;
                var bBox = [px - mr, px + mr, py - mr, py + mr];
                var ly2 = py - mr - 2;
                var ly1 = ly2 - th;
                return {{ box: bBox, label: [px - tw / 2, px + tw / 2, ly1, ly2] }};
            }}

            function isOverlap(a, b) {{
                return a[0] < b[1] && a[1] > b[0] && a[2] < b[3] && a[3] > b[2];
            }}

            function partsCollide(pA, pB) {{
                return isOverlap(pA.box, pB.box) ||
                       isOverlap(pA.box, pB.label) ||
                       isOverlap(pA.label, pB.box) ||
                       isOverlap(pA.label, pB.label);
            }}

            var keptParts = [];
            var mItems = {marker_items_json};
            for (var mi = 0; mi < mItems.length; mi++) {{
                keptParts.push(getMarkerParts(mItems[mi]));
            }}

            var selIds = {selected_ids_json};
            var kept = {{}};
            var mBoxes = window._masterBoxes || [];

            for (var i = 0; i < mBoxes.length; i++) {{
                var rid = mBoxes[i].roomId;
                if (selIds.indexOf(rid) !== -1) continue;

                var parts = getRoomParts(mBoxes[i]);
                var collides = false;
                for (var k = 0; k < keptParts.length; k++) {{
                    if (partsCollide(parts, keptParts[k])) {{
                        collides = true;
                        break;
                    }}
                }}

                if (!collides) {{
                    kept[rid] = true;
                    keptParts.push(parts);
                }}
            }}

            if (window._masterLabels) {{
                var fb = mBoxes.filter(function(r) {{ return kept[r.roomId]; }});
                var fl = window._masterLabels.filter(function(r) {{
                    return kept[r.roomId] && selIds.indexOf(r.roomId) === -1;
                }});
                chart.setOption({{
                    series: [
                        {{ name: "Walkable path" }},
                        {{ name: "Rooms", data: fb }},
                        {{ name: "Room names", data: fl }}
                    ]
                }});
            }}
        """

        events = {
            "finished": f"""function() {{
                {filter_fn_body}
            }}""",
            "datazoom": f"""function(p) {{
                {filter_fn_body}
            }}""",
            "click": f"""function(params) {{
                function getGraphicInfo(p) {{
                    if (!p) return null;
                    if (p.info) return p.info;
                    var t = p.target;
                    while (t) {{
                        if (t.info) return t.info;
                        t = t.parent;
                    }}
                    return null;
                }}

                var dom = document.querySelector('div[_echarts_instance_]') || document.querySelector('.echarts-for-react');
                var w = dom ? dom.clientWidth : null;
                var h = dom ? dom.clientHeight : null;

                var gInfo = getGraphicInfo(params);
                if (params.componentType === 'graphic' || gInfo) {{
                    if (gInfo === 'clear_path') {{
                        window._lastProcessedZoomKey = null;
                    }}
                    return {{
                        graphicAction: gInfo,
                        dataZoom: window._mapZoom || null,
                        chartWidth: w,
                        chartHeight: h
                    }};
                }}

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
                var gridTop = {grid_top};
                var gridBottom = (dom.clientHeight || (rect ? rect.height : 0)) - {grid_bottom};

                if (px === null || py === null || px < gridLeft || px > gridRight || py < gridTop || py > gridBottom) {{
                    return {{
                        roomId: null,
                        value: null,
                        coords: null,
                        edge: null,
                        clickCoord: null,
                        dataZoom: window._mapZoom || null,
                        chartWidth: w,
                        chartHeight: h
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
                    dataZoom: window._mapZoom || null,
                    chartWidth: w,
                    chartHeight: h
                }};
            }}"""
        }

        clicked_data = st_echarts(
            options=options,
            events=events,
            height="350px" if is_mobile_device() else "650px",
            key="floorplan"
        )

        if handle_map_click(clicked_data, positions, room_ids, room_tree):
            st.rerun()


if __name__ == "__main__":
    main()