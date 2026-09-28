from pathlib import Path


if __name__ == "__main__":
    main_dir = Path(__file__).parent.parent.resolve()
    parent_dir = Path(__file__).parent.resolve()

    floorplan_dir = main_dir / "Custom-Floormap" / "AB-Ground"
    output_dir = parent_dir / "AB-Ground"