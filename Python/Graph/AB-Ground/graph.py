from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import networkx as nx


if __name__ == "__main__":
    main_dir = Path(__file__).parent.parent.parent.resolve()

    floorplan_dir = main_dir / "Custom-Floormap" / "AB-Ground"
    output_dir = Path(__file__).parent.resolve()

    rooms_info = floorplan_dir / "ab_ground_unified_fp_rooms.csv"
    rooms_df = pd.read_csv(rooms_info)

    intersections_info = floorplan_dir / "ab_ground_unified_fp_intersections.csv"
    intersections_df = pd.read_csv(intersections_info)