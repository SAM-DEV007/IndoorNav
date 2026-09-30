import ast
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
	rooms["label"] = rooms.apply(
		lambda row: f"{int(row.Room_ID)} - {row.Room_Name}", axis=1
	)
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


def make_map(graph, positions, rooms, route, origin, destination, clicked):
	figure = go.Figure()
	for left, right in graph.edges:
		x1, y1 = positions[left]
		x2, y2 = positions[right]
		figure.add_trace(
			go.Scatter(
				x=[x1, x2], y=[y1, y2], mode="lines",
				line={"color": "#b7c4c5", "width": 12},
				hoverinfo="skip", showlegend=False,
			)
		)

	if route:
		route_x = [positions[node][0] for node in route]
		route_y = [positions[node][1] for node in route]
		figure.add_trace(
			go.Scatter(
				x=route_x, y=route_y, mode="lines+markers",
				line={"color": "#e4572e", "width": 5},
				marker={"size": 7, "color": "#e4572e"},
				name="Shortest route", hoverinfo="skip",
			)
		)

	room_x = [positions[int(room)][0] for room in rooms.Room_ID]
	room_y = [positions[int(room)][1] for room in rooms.Room_ID]
	figure.add_trace(
		go.Scatter(
			x=room_x, y=room_y, mode="markers+text",
			text=rooms.Room_Name, textposition="top center",
			textfont={"size": 9, "color": "#263238"},
			marker={"symbol": "square", "size": 10, "color": "#2f8fbd", "line": {"color": "#17465d", "width": 1}},
			customdata=rooms.Room_ID, name="Rooms",
			hovertemplate="%{customdata}: %{text}<extra></extra>",
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

	for room_id, color, label in ((origin, "#f4c542", "Start"), (destination, "#e63946", "Destination")):
		x, y = positions[int(room_id)]
		figure.add_trace(
			go.Scatter(
				x=[x], y=[y], mode="markers+text", text=[label], textposition="bottom center",
				marker={"size": 17, "color": color, "line": {"color": "#202a2e", "width": 2}},
				hoverinfo="skip", showlegend=False,
			)
		)
	if clicked:
		figure.add_trace(
			go.Scatter(x=[clicked[0]], y=[clicked[1]], mode="markers",
					   marker={"size": 11, "color": "#f4c542", "symbol": "x"},
					   name="Clicked location", hoverinfo="skip")
		)

	figure.update_layout(
		height=720, clickmode="event+select", dragmode="pan",
		margin={"l": 10, "r": 10, "t": 10, "b": 10},
		plot_bgcolor="#fbfaf6", paper_bgcolor="#fbfaf6",
		legend={"orientation": "h", "y": 1.02, "x": 0},
		xaxis={"visible": False, "scaleanchor": "y", "scaleratio": 1},
		yaxis={"visible": False},
	)
	return figure


def main():
	st.set_page_config(page_title="AB Ground Navigation", page_icon="📍", layout="wide")
	graph, positions, rooms, routes = load_map_data()
	labels = dict(zip(rooms.Room_ID, rooms.label))

	st.title("AB Ground Floor Navigation")
	st.caption("Choose rooms or click a corridor to place the starting location.")
	controls, map_column = st.columns([1, 2.7], gap="large")
	with controls:
		room_options = list(rooms.Room_ID)
		default_origin = int(st.session_state.get("origin", room_options[0]))
		destination_default = room_options[1] if len(room_options) > 1 else room_options[0]
		origin = st.selectbox("Starting room", room_options, index=room_options.index(default_origin), format_func=labels.get)
		destination = st.selectbox("Destination room", room_options, index=room_options.index(destination_default), format_func=labels.get)
		if origin == destination:
			st.info("Choose two different rooms to show a route.")
			route_data = None
		else:
			route_data = routes.get((origin, destination))
			if route_data:
				st.metric("Route distance", f"{route_data[0]:.2f} m")
				st.write(f"{len(route_data[1]) - 1} segments")
			else:
				st.warning("No cached route exists for this pair.")
		st.markdown("**Map click**")
		st.write("Click a corridor or room marker to snap the start to the nearest room.")
		st.button("Reset clicked start", on_click=lambda: st.session_state.pop("clicked", None))

	clicked = st.session_state.get("clicked")
	with map_column:
		event = st.plotly_chart(
			make_map(graph, positions, rooms, route_data[1] if route_data else None, origin, destination, clicked),
			use_container_width=True, on_select="rerun", selection_mode="points", key="floorplan",
		)
	selected_point = click_position(event)
	if selected_point:
		selected_room = nearest_room(*selected_point, rooms, positions)
		st.session_state.clicked = selected_point
		st.session_state.origin = int(selected_room)
		st.rerun()


if __name__ == "__main__":
	main()

