import ast
import math
from pathlib import Path

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
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


def click_position(event):
	if not event or not getattr(event, "selection", None):
		return None
	points = event.selection.get("points", [])
	if not points:
		return None
	point = points[-1]
	if point.get("x") is None or point.get("y") is None:
		return None
	return float(point["x"]), float(point["y"])


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
		return {"xshift": 14, "yshift": 0, "xanchor": "left", "yanchor": "middle", "align": "left", "angle": 0}
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
			"xshift": 14 if away_x > 0 else -14,
			"yshift": 0,
			"xanchor": "left" if away_x > 0 else "right",
			"yanchor": "middle",
			"align": "left" if away_x > 0 else "right",
			"angle": angle,
		}
	return {
		"xshift": 0,
		"yshift": 12 if away_y > 0 else -12,
		"xanchor": "center",
		"yanchor": "bottom" if away_y > 0 else "top",
		"align": "center",
		"angle": angle,
	}


def make_map(graph, positions, rooms, route, origin, destination, clicked):
	figure = go.Figure()

	base_x, base_y = [], []

	for left, right in graph.edges:
		x1, y1 = positions[left]
		x2, y2 = positions[right]
		base_x.extend([x1, x2, None])
		base_y.extend([y1, y2, None])
        
	node_x = [positions[node][0] for node in graph.nodes]
	node_y = [positions[node][1] for node in graph.nodes]

	figure.add_trace(
		go.Scatter(
			x=base_x, y=base_y, mode="lines",
			line={"color": "#aebabc", "width": 15},
			hoverinfo="skip", name="Walkable path",
		)
	)

	figure.add_trace(
		go.Scatter(
			x=node_x, y=node_y, mode="markers",
			marker={"color": "#aebabc", "size": 15, "line": {"width": 0}},
			hoverinfo="skip", showlegend=False,
		)
	)

	figure.add_trace(
		go.Scatter(
			x=base_x, y=base_y, mode="lines",
			line={"color": "#edf1ee", "width": 9},
			hoverinfo="skip", showlegend=False,
		)
	)
    
	figure.add_trace(
		go.Scatter(
			x=node_x, y=node_y, mode="markers",
			marker={"color": "#edf1ee", "size": 9, "line": {"width": 0}},
			hoverinfo="skip", showlegend=False,
		)
	)

	if route:
		route_x = [positions[node][0] for node in route] if route else [None]
		route_y = [positions[node][1] for node in route] if route else [None]
		figure.add_trace(
			go.Scatter(
				x=route_x, y=route_y, mode="lines",
				line={"color": "#e4572e", "width": 5},
				name="Shortest route", hoverinfo="skip",
			)
		)

	room_records = rooms[~rooms.Room_ID.isin([1, 15])]
	room_x = [positions[int(room)][0] for room in room_records.Room_ID]
	room_y = [positions[int(room)][1] for room in room_records.Room_ID]
	figure.add_trace(
		go.Scatter(
			x=room_x, y=room_y, mode="markers",
			marker={"symbol": "square", "size": 10, "color": "#2f8fbd", "line": {"color": "#17465d", "width": 1}},
			customdata=room_records.Room_ID, text=room_records.Room_Name, name="Rooms",
			legendgroup="rooms",
			hovertemplate="%{text}<extra></extra>",
		)
	)

	all_x, all_y = zip(*positions.values())
	x_units_per_pixel = (max(all_x) - min(all_x)) / 900
	y_units_per_pixel = (max(all_y) - min(all_y)) / 650

	label_x_values, label_y_values, label_values = [], [], []
	for room in room_records.itertuples(index=False):
		x, y = positions[int(room.Room_ID)]

		label_layout = room_label_layout(int(room.Room_ID), graph, positions)
		label_x = x + label_layout["xshift"] * x_units_per_pixel
		label_y = y + label_layout["yshift"] * y_units_per_pixel

		label_x_values.append(label_x)
		label_y_values.append(label_y)
		label_values.append(room.Room_Name)

	figure.add_trace(
		go.Scatter(
			x=label_x_values, y=label_y_values, mode="text", text=label_values,
			textposition="middle center", textfont={"size": 7, "color": "#263238"},
			name="Room names", legendgroup="rooms", showlegend=False, hoverinfo="skip",
		)
	)

	gate_x = [positions[1][0], positions[15][0]]
	gate_y = [positions[1][1], positions[15][1]]
	figure.add_trace(
		go.Scatter(
			x=gate_x, y=gate_y, mode="markers+text", text=["Front Gate", "Back Gate"],
			textposition=["bottom center", "top center"], name="Gates",
			marker={"symbol": "circle", "size": 15, "color": ["#e63946", "#111111"], "line": {"color": "#202a2e", "width": 2}},
			textfont={"size": 8, "color": "#263238"}, 
			hovertemplate="%{text}<extra></extra>",
		)
	)

	samples_x, samples_y = [], []
	for left, right in graph.edges:
		x1, y1 = positions[left]
		x2, y2 = positions[right]
		for fraction in (0.25, 0.5, 0.75):
			samples_x.append(x1 + fraction * (x2 - x1))
			samples_y.append(y1 + fraction * (y2 - y1))
	figure.add_trace(
		go.Scatter(
			x=samples_x, y=samples_y, mode="markers",
			marker={"size": 18, "color": "rgba(0,0,0,0.01)"},
			name="Click map", hoverinfo="skip", showlegend=False,
		)
	)

	for room_id, color, label in ((origin, "#ffffff", "Start"), (destination, "#2ca25f", "Destination")):
		if int(room_id) in (1, 15):
			continue
		x, y = positions[int(room_id)]
		figure.add_trace(
			go.Scatter(
				x=[x], y=[y], mode="markers",
				marker={"size": 10, "color": color, "line": {"color": "#202a2e", "width": 2}},
				hoverinfo="skip", showlegend=False,
			)
		)
	for label, color in (("Start", "#ffffff"), ("Destination", "#2ca25f")):
		figure.add_trace(
			go.Scatter(
				x=[None], y=[None], mode="markers", name=label,
				marker={"size": 10, "color": color, "line": {"color": "#202a2e", "width": 1}},
				hoverinfo="skip",
			)
		)
	figure.update_layout(
		height=650, clickmode="event", dragmode="pan", uirevision="ab-ground",
		margin={"l": 10, "r": 10, "t": 10, "b": 10},
		plot_bgcolor="#fbfaf6", paper_bgcolor="#fbfaf6",
		legend={"orientation": "h", "y": 1.02, "x": 0, "groupclick": "togglegroup", "font": {"color": "#263238", "size": 10}},
		xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1, "fixedrange": False},
		yaxis={"visible": False, "fixedrange": False},
	)

	padding = 8

	min_x, max_x = min(all_x) - padding, max(all_x) + padding
	min_y, max_y = min(all_y) - padding, max(all_y) + padding

	figure.update_xaxes(range=[min_x, max_x])
	figure.update_yaxes(range=[min_y, max_y])

	figure.update_traces(
		selected={"marker": {"opacity": 1}, "textfont": {"color": "#263238"}},
		unselected={"marker": {"opacity": 1}, "textfont": {"color": "#263238"}}
	)

	return figure


def main():
	st.set_page_config(page_title="AB Ground Navigation", layout="wide")
	st.markdown(
		"""
		<style>
		.js-plotly-plot .plotly .nsewdrag {
			cursor: move !important;
		}
		.js-plotly-plot:has(.hoverlayer .hovertext) .plotly .nsewdrag {
			cursor: pointer !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	st.markdown(
		"""
		<style>
		.js-plotly-plot .plotly .modebar-container {
			top: auto !important;
			bottom: 55px !important;
			right: 12px !important;
			left: auto !important;
			z-index: 1001 !important;
		}
		.js-plotly-plot .plotly .modebar {
			opacity: 1 !important;
			background: rgba(251, 250, 246, 0.85) !important;
			border-radius: 4px !important;
		}
		</style>
		""",
		unsafe_allow_html=True,
	)
	graph, positions, rooms, routes = load_map_data()
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

	clicked = st.session_state.get("clicked")
	with map_column:
		event = st.plotly_chart(
			make_map(graph, positions, rooms, route_data[1] if route_data else None, origin, destination, clicked),
			width="stretch", on_select="rerun", key="floorplan",
			config={"scrollZoom": True, "displaylogo": False, "responsive": True},
		)

	selected_point = click_position(event)
	if selected_point and selected_point != st.session_state.get("clicked"):
		st.session_state.clicked = selected_point

		selected_room = int(nearest_room(*selected_point, rooms, positions))

		if selected_room != st.session_state.get("destination", None):
			st.session_state.destination = selected_room
			st.rerun()


if __name__ == "__main__":
	main()

