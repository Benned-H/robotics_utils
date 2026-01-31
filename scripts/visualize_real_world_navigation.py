"""Visualize navigation planning on a real-world occupancy grid.

This script loads an occupancy grid from a YAML + image file pair and visualizes
path planning between specified start and goal poses using Spot's footprint.

To run this script:

    uv run scripts/visualize_real_world_navigation.py /path/to/occupancy_grid.yaml

The script expects both a YAML file and an image file (PNG/JPG) with the same stem.
"""

from __future__ import annotations

from pathlib import Path

import click

from robotics_utils.io import console
from robotics_utils.io.pydantic_schemata import OccupancyGrid2DSchema
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.motion_planning import NavigationQuery, RectangularFootprint, plan_se2_path
from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import Pose2D
from robotics_utils.visualization import display_in_window
from robotics_utils.visualization.navigation_visualization import NavigationVisualization

# Navigation query poses (Spot body frame, x-y-yaw)
START_POSE = Pose2D(x=4.337, y=-6.922, yaw_rad=1.387)
GOAL_POSE = Pose2D(x=6.498, y=-4.496, yaw_rad=1.514)
ERASE_WAYPOINT = GOAL_POSE  # Goal pose also serves as the erase waypoint

SPOT_FOOTPRINT = RectangularFootprint(max_x_m=0.63, min_x_m=-0.49, half_length_y_m=0.25)


def load_occupancy_grid(yaml_path: Path) -> OccupancyGrid2D:
    """Load an occupancy grid from a YAML file and its associated image.

    :param yaml_path: Path to the YAML file containing grid metadata
    :return: Loaded OccupancyGrid2D instance
    """
    yaml_data = load_yaml_data(yaml_path)

    # Resolve the image path relative to the YAML file's directory
    if "image_path" in yaml_data:
        image_path = Path(yaml_data["image_path"])
        if not image_path.is_absolute():
            yaml_data["image_path"] = yaml_path.parent / image_path

    schema = OccupancyGrid2DSchema.model_validate(yaml_data)
    return OccupancyGrid2D.from_schema(schema)


@click.command()
@click.argument("filepath", type=click.Path(exists=True, path_type=Path))
def main(filepath: Path) -> None:
    """Visualize navigation planning on a real-world occupancy grid.

    FILEPATH: Path to the occupancy grid YAML file.
    """
    console.print("[bold blue]Real-World Navigation Visualization[/bold blue]")
    console.print()

    # Load the occupancy grid
    console.print(f"[bold]Step 1:[/] Loading occupancy grid from {filepath}...")
    occ_grid = load_occupancy_grid(filepath)

    grid = occ_grid.grid
    console.print(f"  Grid size: {grid.height_cells} rows x {grid.width_cells} columns")
    console.print(f"  Grid resolution: {grid.resolution_m:.3f} m/cell")
    console.print(f"  Grid origin: ({grid.origin.x:.3f}, {grid.origin.y:.3f}) m")
    console.print()

    # Display robot footprint info
    console.print("[bold]Step 2:[/] Using Spot robot footprint...")
    console.print(
        f"  Footprint: {SPOT_FOOTPRINT.min_x_m:.2f} m to {SPOT_FOOTPRINT.max_x_m:.2f} m (x), "
        f"+/-{SPOT_FOOTPRINT.half_length_y_m:.2f} m (y)",
    )
    console.print()

    # Display query poses
    console.print("[bold]Step 3:[/] Navigation query poses...")
    console.print(
        f"  Start: ({START_POSE.x:.3f}, {START_POSE.y:.3f}, {START_POSE.yaw_rad:.3f} rad)",
    )
    console.print(f"  Goal:  ({GOAL_POSE.x:.3f}, {GOAL_POSE.y:.3f}, {GOAL_POSE.yaw_rad:.3f} rad)")
    console.print(f"  Erase waypoint: ({ERASE_WAYPOINT.x:.3f}, {ERASE_WAYPOINT.y:.3f})")
    console.print()

    # Check if start and goal are collision-free
    console.print("[bold]Step 4:[/] Checking pose validity...")
    start_valid = SPOT_FOOTPRINT.is_collision_free(START_POSE, occ_grid)
    goal_valid = SPOT_FOOTPRINT.is_collision_free(GOAL_POSE, occ_grid)
    console.print(f"  Start pose collision-free: {start_valid}")
    console.print(f"  Goal pose collision-free: {goal_valid}")
    console.print()

    # Attempt path planning
    console.print("[bold]Step 5:[/] Planning path...")
    query = NavigationQuery(
        start_pose=START_POSE,
        goal_pose=GOAL_POSE,
        occupancy_grid=occ_grid,
        robot_footprint=SPOT_FOOTPRINT,
    )
    path = plan_se2_path(query)

    if path is None:
        console.print("  [red]FAILED![/] No path found.")
    else:
        console.print(f"  [green]SUCCESS![/] Found path with {len(path)} waypoints.")
    console.print()

    # Visualize the result
    console.print("[bold]Step 6:[/] Visualizing navigation planning result...")
    viz = NavigationVisualization(occ_grid, SPOT_FOOTPRINT)
    viz.draw_path(path)
    viz.draw_query_endpoints(START_POSE, GOAL_POSE)
    display_in_window(viz.image, "Navigation Planning Result")


if __name__ == "__main__":
    main()
