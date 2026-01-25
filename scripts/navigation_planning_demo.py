"""Demonstrate navigation planning with occupancy grids and robot footprint collision checking.

This script creates a two-room environment with a doorway, demonstrates path planning,
and shows how adding/removing a door obstacle affects navigation feasibility.

To run this script, use the commands:

    uv venv --clear && uv sync --extra vision
    uv run scripts/navigation_planning_demo.py

"""

from __future__ import annotations

import click
import cv2
import numpy as np
from numpy.typing import NDArray

from robotics_utils.io import console
from robotics_utils.motion_planning import (
    DiscreteGrid2D,
    NavigationQuery,
    RectangularFootprint,
    plan_se2_path,
)
from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import Pose2D
from robotics_utils.states import ObjectCentricState
from robotics_utils.vision import RGBImage
from robotics_utils.visualization import display_in_window

# Grid parameters
GRID_RESOLUTION_M = 0.05  # 5 cm per cell
GRID_WIDTH_M = 10.0  # 10 meters wide
GRID_HEIGHT_M = 8.0  # 8 meters tall
GRID_WIDTH_CELLS = int(GRID_WIDTH_M / GRID_RESOLUTION_M)
GRID_HEIGHT_CELLS = int(GRID_HEIGHT_M / GRID_RESOLUTION_M)

# Room layout parameters (in meters)
WALL_THICKNESS_M = 0.1  # 10 cm thick walls
DIVIDING_WALL_X_M = 5.0  # Wall dividing the two rooms at x=5m
DOORWAY_Y_MIN_M = 3.0  # Doorway from y=3m to y=5m
DOORWAY_Y_MAX_M = 5.0

# Robot footprint (Spot-like dimensions)
ROBOT_MAX_X_M = 0.5  # Front of robot
ROBOT_MIN_X_M = -0.5  # Back of robot
ROBOT_HALF_WIDTH_M = 0.3  # Half-width of robot

# Visualization scale factor (pixels per grid cell)
VIZ_SCALE = 4

# Log-odds value for occupied cells
OCCUPIED_LOG_ODDS = 10.0


def create_two_room_occupancy_grid() -> OccupancyGrid2D:
    """Create an occupancy grid with two rectangular rooms and a doorway.

    The grid has:
    - Outer walls forming a rectangular boundary
    - A vertical dividing wall with a doorway in the middle
    """
    grid = DiscreteGrid2D(
        origin=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
        resolution_m=GRID_RESOLUTION_M,
        width_cells=GRID_WIDTH_CELLS,
        height_cells=GRID_HEIGHT_CELLS,
    )
    occ_grid = OccupancyGrid2D(grid)

    # Helper function to fill a rectangular region with walls
    def fill_wall(x_min_m: float, x_max_m: float, y_min_m: float, y_max_m: float) -> None:
        col_min = int(x_min_m / GRID_RESOLUTION_M)
        col_max = int(x_max_m / GRID_RESOLUTION_M)
        row_min = int(y_min_m / GRID_RESOLUTION_M)
        row_max = int(y_max_m / GRID_RESOLUTION_M)

        col_min = max(0, min(col_min, GRID_WIDTH_CELLS))
        col_max = max(0, min(col_max, GRID_WIDTH_CELLS))
        row_min = max(0, min(row_min, GRID_HEIGHT_CELLS))
        row_max = max(0, min(row_max, GRID_HEIGHT_CELLS))

        occ_grid.log_odds[row_min:row_max, col_min:col_max] = OCCUPIED_LOG_ODDS

    # Outer walls
    fill_wall(0, GRID_WIDTH_M, 0, WALL_THICKNESS_M)  # Bottom wall
    fill_wall(0, GRID_WIDTH_M, GRID_HEIGHT_M - WALL_THICKNESS_M, GRID_HEIGHT_M)  # Top wall
    fill_wall(0, WALL_THICKNESS_M, 0, GRID_HEIGHT_M)  # Left wall
    fill_wall(GRID_WIDTH_M - WALL_THICKNESS_M, GRID_WIDTH_M, 0, GRID_HEIGHT_M)  # Right wall

    # Dividing wall with doorway (wall below and above the doorway)
    fill_wall(
        DIVIDING_WALL_X_M - WALL_THICKNESS_M / 2,
        DIVIDING_WALL_X_M + WALL_THICKNESS_M / 2,
        0,
        DOORWAY_Y_MIN_M,
    )  # Below doorway
    fill_wall(
        DIVIDING_WALL_X_M - WALL_THICKNESS_M / 2,
        DIVIDING_WALL_X_M + WALL_THICKNESS_M / 2,
        DOORWAY_Y_MAX_M,
        GRID_HEIGHT_M,
    )  # Above doorway

    return occ_grid


def stamp_door_into_occupancy(occ_grid: OccupancyGrid2D) -> None:
    """Add a door obstacle that blocks the doorway."""
    col_min = int((DIVIDING_WALL_X_M - WALL_THICKNESS_M / 2) / GRID_RESOLUTION_M)
    col_max = int((DIVIDING_WALL_X_M + WALL_THICKNESS_M / 2) / GRID_RESOLUTION_M)
    row_min = int(DOORWAY_Y_MIN_M / GRID_RESOLUTION_M)
    row_max = int(DOORWAY_Y_MAX_M / GRID_RESOLUTION_M)

    occ_grid.log_odds[row_min:row_max, col_min:col_max] = OCCUPIED_LOG_ODDS


def remove_door_from_occupancy(occ_grid: OccupancyGrid2D) -> None:
    """Remove the door obstacle from the doorway."""
    col_min = int((DIVIDING_WALL_X_M - WALL_THICKNESS_M / 2) / GRID_RESOLUTION_M)
    col_max = int((DIVIDING_WALL_X_M + WALL_THICKNESS_M / 2) / GRID_RESOLUTION_M)
    row_min = int(DOORWAY_Y_MIN_M / GRID_RESOLUTION_M)
    row_max = int(DOORWAY_Y_MAX_M / GRID_RESOLUTION_M)

    occ_grid.log_odds[row_min:row_max, col_min:col_max] = -10.0  # Free space


def create_visualization_image(
    occ_grid: OccupancyGrid2D,
    robot_footprint: RectangularFootprint,
    path: list[Pose2D] | None = None,
    start_pose: Pose2D | None = None,
    goal_pose: Pose2D | None = None,
    door_closed: bool = False,
) -> RGBImage:
    """Create a scaled-up visualization of the occupancy grid with robot footprints.

    :param occ_grid: The occupancy grid to visualize
    :param robot_footprint: Robot footprint for drawing poses
    :param path: Optional navigation path to visualize
    :param start_pose: Optional start pose to highlight
    :param goal_pose: Optional goal pose to highlight
    :param door_closed: Whether the door is currently closed
    :return: RGB image for visualization
    """
    # Get occupied mask and create base image
    occupied_mask = occ_grid.get_occupied_mask()

    # Scale up the occupancy grid
    h, w = occupied_mask.shape
    scaled_h, scaled_w = h * VIZ_SCALE, w * VIZ_SCALE

    # Create RGB image (white background, black walls)
    img = np.ones((scaled_h, scaled_w, 3), dtype=np.uint8) * 255

    # Draw occupied cells as dark gray
    for row in range(h):
        for col in range(w):
            if occupied_mask[row, col]:
                y1 = (h - 1 - row) * VIZ_SCALE
                y2 = y1 + VIZ_SCALE
                x1 = col * VIZ_SCALE
                x2 = x1 + VIZ_SCALE
                img[y1:y2, x1:x2] = (40, 40, 40)  # Dark gray for walls

    def world_to_pixel(x_m: float, y_m: float) -> tuple[int, int]:
        """Convert world coordinates to pixel coordinates."""
        col = int(x_m / GRID_RESOLUTION_M)
        row = int(y_m / GRID_RESOLUTION_M)
        px = col * VIZ_SCALE + VIZ_SCALE // 2
        py = (h - 1 - row) * VIZ_SCALE + VIZ_SCALE // 2
        return (px, py)

    def draw_robot_footprint(pose: Pose2D, color: tuple[int, int, int], thickness: int = 2) -> None:
        """Draw the robot footprint at the given pose."""
        cos_yaw = np.cos(pose.yaw_rad)
        sin_yaw = np.sin(pose.yaw_rad)

        # Footprint corners in body frame
        corners_body = [
            (robot_footprint.max_x_m, robot_footprint.half_length_y_m),
            (robot_footprint.max_x_m, -robot_footprint.half_length_y_m),
            (robot_footprint.min_x_m, -robot_footprint.half_length_y_m),
            (robot_footprint.min_x_m, robot_footprint.half_length_y_m),
        ]

        # Transform to world frame and then to pixels
        corners_px = []
        for bx, by in corners_body:
            wx = pose.x + cos_yaw * bx - sin_yaw * by
            wy = pose.y + sin_yaw * bx + cos_yaw * by
            corners_px.append(world_to_pixel(wx, wy))

        # Draw the footprint polygon
        pts = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw heading direction (arrow from center to front)
        center_px = world_to_pixel(pose.x, pose.y)
        front_x = pose.x + robot_footprint.max_x_m * 0.7 * cos_yaw
        front_y = pose.y + robot_footprint.max_x_m * 0.7 * sin_yaw
        front_px = world_to_pixel(front_x, front_y)
        cv2.arrowedLine(img, center_px, front_px, color, thickness=max(1, thickness - 1))

    # Draw path if provided
    if path is not None and len(path) > 1:
        # Draw path line
        path_points = [world_to_pixel(p.x, p.y) for p in path]
        for i in range(len(path_points) - 1):
            cv2.line(img, path_points[i], path_points[i + 1], (0, 180, 0), thickness=2)

        # Draw robot footprints along path (every few waypoints)
        step = max(1, len(path) // 10)
        for i in range(0, len(path), step):
            draw_robot_footprint(path[i], (100, 200, 100), thickness=1)

    # Draw start pose
    if start_pose is not None:
        draw_robot_footprint(start_pose, (0, 0, 255), thickness=3)  # Blue
        center = world_to_pixel(start_pose.x, start_pose.y)
        cv2.putText(img, "S", (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Draw goal pose
    if goal_pose is not None:
        draw_robot_footprint(goal_pose, (255, 0, 0), thickness=3)  # Red
        center = world_to_pixel(goal_pose.x, goal_pose.y)
        cv2.putText(img, "G", (center[0] - 10, center[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add status text
    status = "Door: CLOSED" if door_closed else "Door: OPEN"
    path_status = f"Path: {len(path)} waypoints" if path else "Path: NOT FOUND"
    cv2.putText(img, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img, path_status, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return RGBImage(data=img)


def sample_pose_in_room(rng: np.random.Generator, left_room: bool) -> Pose2D:
    """Sample a random pose in one of the two rooms.

    :param rng: Random number generator
    :param left_room: If True, sample in left room; otherwise sample in right room
    :return: Random Pose2D within the specified room
    """
    # Leave margin from walls for robot footprint
    margin = max(ROBOT_MAX_X_M, ROBOT_HALF_WIDTH_M) + 0.2

    if left_room:
        x_min = WALL_THICKNESS_M + margin
        x_max = DIVIDING_WALL_X_M - margin
    else:
        x_min = DIVIDING_WALL_X_M + margin
        x_max = GRID_WIDTH_M - WALL_THICKNESS_M - margin

    y_min = WALL_THICKNESS_M + margin
    y_max = GRID_HEIGHT_M - WALL_THICKNESS_M - margin

    x = rng.uniform(x_min, x_max)
    y = rng.uniform(y_min, y_max)
    yaw = rng.uniform(-np.pi, np.pi)

    return Pose2D(x=x, y=y, yaw_rad=yaw)


def main() -> None:
    """Run the navigation planning demonstration."""
    rng = np.random.default_rng()

    console.print("[bold blue]Navigation Planning Demo[/bold blue]")
    console.print("=" * 50)
    console.print()

    # Create robot footprint
    robot_footprint = RectangularFootprint(
        max_x_m=ROBOT_MAX_X_M,
        min_x_m=ROBOT_MIN_X_M,
        half_length_y_m=ROBOT_HALF_WIDTH_M,
    )
    console.print(f"Robot footprint: {ROBOT_MIN_X_M}m to {ROBOT_MAX_X_M}m (x), +/-{ROBOT_HALF_WIDTH_M}m (y)")

    # Create object-centric state
    env_state = ObjectCentricState(robot_names=["spot"], object_names=["door"], root_frame="world")
    console.print("Created ObjectCentricState with robot 'spot' and object 'door'")
    console.print()

    while True:
        # Step 1: Create occupancy grid with two rooms
        console.print("[bold]Step 1:[/bold] Creating occupancy grid with two rooms and a doorway...")
        occ_grid = create_two_room_occupancy_grid()
        console.print(f"  Grid size: {GRID_WIDTH_CELLS} x {GRID_HEIGHT_CELLS} cells")
        console.print(f"  Resolution: {GRID_RESOLUTION_M}m per cell")
        console.print(f"  World size: {GRID_WIDTH_M}m x {GRID_HEIGHT_M}m")
        console.print()

        # Step 2: Sample start and goal poses
        console.print("[bold]Step 2:[/bold] Sampling start pose (left room) and goal pose (right room)...")
        start_pose = sample_pose_in_room(rng, left_room=True)
        goal_pose = sample_pose_in_room(rng, left_room=False)
        console.print(f"  Start: ({start_pose.x:.2f}, {start_pose.y:.2f}, {np.degrees(start_pose.yaw_rad):.1f} deg)")
        console.print(f"  Goal:  ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {np.degrees(goal_pose.yaw_rad):.1f} deg)")
        console.print()

        # Step 3: Plan path with door open
        console.print("[bold]Step 3:[/bold] Computing navigation plan (door OPEN)...")
        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=occ_grid,
            robot_footprint=robot_footprint,
        )
        path = plan_se2_path(query)

        if path is not None:
            console.print(f"  [green]SUCCESS![/green] Found path with {len(path)} waypoints")
        else:
            console.print("  [red]FAILED![/red] No path found (try again with different poses)")
            if click.confirm("Sample new poses and try again?"):
                continue
            break

        # Step 4: Visualize initial trajectory
        console.print()
        console.print("[bold]Step 4:[/bold] Visualizing trajectory (press any key to continue)...")
        viz_image = create_visualization_image(
            occ_grid, robot_footprint, path=path, start_pose=start_pose, goal_pose=goal_pose, door_closed=False
        )
        display_in_window(viz_image, title="Navigation Plan - Door Open")

        # Step 5: Add door to object-centric state
        console.print()
        console.print("[bold]Step 5:[/bold] Adding door object to ObjectCentricState...")
        door_pose = Pose2D(x=DIVIDING_WALL_X_M, y=(DOORWAY_Y_MIN_M + DOORWAY_Y_MAX_M) / 2, yaw_rad=0.0)
        env_state.set_known_object_pose("door", door_pose.to_3d())
        console.print(f"  Door pose: ({door_pose.x:.2f}, {door_pose.y:.2f})")
        console.print()

        # Step 6: Stamp door into occupancy and verify no path
        console.print("[bold]Step 6:[/bold] Stamping door into occupancy grid (door CLOSED)...")
        stamp_door_into_occupancy(occ_grid)

        query_blocked = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=occ_grid,
            robot_footprint=robot_footprint,
        )
        path_blocked = plan_se2_path(query_blocked)

        if path_blocked is None:
            console.print("  [green]VERIFIED:[/green] No path found with door closed")
        else:
            console.print(f"  [yellow]UNEXPECTED:[/yellow] Path found with {len(path_blocked)} waypoints")

        # Visualize blocked state
        console.print()
        console.print("Visualizing blocked state (press any key to continue)...")
        viz_image_blocked = create_visualization_image(
            occ_grid, robot_footprint, path=path_blocked, start_pose=start_pose, goal_pose=goal_pose, door_closed=True
        )
        display_in_window(viz_image_blocked, title="Navigation Plan - Door Closed")

        # Step 7: Remove door and verify path can be found again
        console.print()
        console.print("[bold]Step 7:[/bold] Removing door from occupancy grid (door OPEN)...")
        remove_door_from_occupancy(occ_grid)
        env_state.clear_object_pose("door")

        query_unblocked = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=occ_grid,
            robot_footprint=robot_footprint,
        )
        path_unblocked = plan_se2_path(query_unblocked)

        if path_unblocked is not None:
            console.print(f"  [green]VERIFIED:[/green] Path found again with {len(path_unblocked)} waypoints")
        else:
            console.print("  [red]UNEXPECTED:[/red] No path found after removing door")

        # Visualize restored state
        console.print()
        console.print("Visualizing restored state (press any key to continue)...")
        viz_image_restored = create_visualization_image(
            occ_grid,
            robot_footprint,
            path=path_unblocked,
            start_pose=start_pose,
            goal_pose=goal_pose,
            door_closed=False,
        )
        display_in_window(viz_image_restored, title="Navigation Plan - Door Removed")

        console.print()
        console.print("=" * 50)
        if not click.confirm("Run another iteration with new random poses?"):
            console.print("[green]Demo complete![/green]")
            break


if __name__ == "__main__":
    main()
