"""Demonstrate path planning over 2D occupancy grids with robot footprint collision checking.

This script creates a two-room environment with a doorway, demonstrates path planning,
and shows how adding/removing a door obstacle affects navigation feasibility.

To run this script, use the commands:

    uv venv --clear && uv sync --extra vision
    uv run scripts/path_planning_demo.py

"""

from __future__ import annotations

from dataclasses import dataclass

import click
import numpy as np

from robotics_utils.collision_models import Box, CollisionModel, CollisionModelRasterizer
from robotics_utils.geometry import Point2D
from robotics_utils.io import console
from robotics_utils.motion_planning import (
    DiscreteGrid2D,
    NavigationQuery,
    RectangularFootprint,
    plan_se2_path,
)
from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import Pose2D
from robotics_utils.states import ObjectKinematicState
from robotics_utils.visualization import display_in_window
from robotics_utils.visualization.navigation_visualization import NavigationVisualization

# Grid parameters
GRID_RESOLUTION_M = 0.1  # 5 cm per cell
GRID_WIDTH_M = 12.0  # 12 meters wide (extra space for rotated rooms)
GRID_HEIGHT_M = 12.0  # 12 meters tall (extra space for rotated rooms)
GRID_WIDTH_CELLS = int(GRID_WIDTH_M / GRID_RESOLUTION_M)
GRID_HEIGHT_CELLS = int(GRID_HEIGHT_M / GRID_RESOLUTION_M)

WORLD_T_GRID_X_M = -GRID_WIDTH_M / 2
WORLD_T_GRID_Y_M = -GRID_HEIGHT_M / 2

# Room layout parameters (in meters, relative to room center)
ROOM_WIDTH_M = 8.0  # Total width of both rooms
ROOM_HEIGHT_M = 6.0  # Total height of the rooms
WALL_THICKNESS_M = 0.2  # 20 cm thick walls
WALL_HEIGHT_M = 2.5  # Height of walls

# Dividing wall and doorway (relative to room center)
DIVIDING_WALL_X_M = 0.0  # Wall dividing the two rooms at center
DOORWAY_HALF_WIDTH_M = 1.0  # Doorway is 2 m wide (from -1 m to +1 m in y)

# Robot footprint (Spot-like dimensions)
ROBOT_MAX_X_M = 0.6  # Front of robot
ROBOT_MIN_X_M = -0.5  # Back of robot
ROBOT_HALF_WIDTH_M = 0.3  # Half-width of robot

# Door dimensions (in meters)
DOOR_THICKNESS_M = 0.05  # Thickness of the door (its x-dimension)
DOOR_WIDTH_M = 2 * DOORWAY_HALF_WIDTH_M  # Width matches doorway
DOOR_HEIGHT_M = 2.0  # Height of the door

# Rasterizer height range (robot navigation height)
RASTERIZE_MIN_HEIGHT_M = 0.0
RASTERIZE_MAX_HEIGHT_M = 1.0  # Capture obstacles up to 1 m height

VISUALIZATION = False


@dataclass
class RoomEnvironment:
    """Container for all collision objects in the two-room environment."""

    walls: list[ObjectKinematicState]
    door: ObjectKinematicState
    world_t_room: Pose2D
    """Transform from the world frame to the room reference frame."""


def create_wall_kinematic_state(
    name: str,
    center_r: Point2D,
    length_x: float,
    length_y: float,
    world_t_room: Pose2D,
) -> ObjectKinematicState:
    """Create an ObjectKinematicState for a wall.

    :param name: Name of the wall object
    :param center_r: Position of wall center in the room frame (r)
    :param length_x: Length of wall along x-axis (room frame)
    :param length_y: Length of wall along y-axis (room frame)
    :param world_t_room: Transform from the world frame to local room coordinates
    :return: ObjectKinematicState for the wall
    """
    wall_box = Box(x_m=length_x, y_m=length_y, z_m=WALL_HEIGHT_M)
    wall_collision_model = CollisionModel(primitives=[wall_box])

    # Local pose (before rotation, relative to room environment center)
    room_t_wall = Pose2D(x=center_r.x, y=center_r.y, yaw_rad=0.0)
    world_t_wall = world_t_room @ room_t_wall

    return ObjectKinematicState(name, world_t_wall.to_3d(), wall_collision_model)


def create_door_kinematic_state(world_t_room: Pose2D) -> ObjectKinematicState:
    """Create an ObjectKinematicState for a door that blocks the doorway.

    :param world_t_room: World-to-room frame transform
    :return: ObjectKinematicState for the door
    """
    door_box = Box(
        x_m=DOOR_THICKNESS_M,  # Thickness along x-axis (perpendicular to doorway)
        y_m=DOOR_WIDTH_M,  # Width along y-axis (spans the doorway)
        z_m=DOOR_HEIGHT_M,  # Height
    )
    door_collision_model = CollisionModel(primitives=[door_box])

    # Door pose in local coordinates: centered in the doorway
    room_t_door = Pose2D(x=DIVIDING_WALL_X_M, y=0.0, yaw_rad=0.0)
    world_t_door = world_t_room @ room_t_door

    return ObjectKinematicState(
        name="door",
        pose=world_t_door.to_3d(),
        collision_model=door_collision_model,
    )


def create_environment(rotation_rad: float) -> RoomEnvironment:
    """Create all collision objects for the two-room environment with the given rotation.

    :param rotation_rad: Rotation angle (radians) to apply to the entire environment
    :return: RoomsEnvironment containing all walls and the door
    """
    walls: list[ObjectKinematicState] = []

    # Half dimensions for positioning
    half_width_m = ROOM_WIDTH_M / 2
    half_height_m = ROOM_HEIGHT_M / 2
    world_t_room = Pose2D(x=0.0, y=0.0, yaw_rad=rotation_rad)

    # Bottom wall (spans full width)
    walls.append(
        create_wall_kinematic_state(
            name="wall_bottom",
            center_r=Point2D(x=0.0, y=-half_height_m - WALL_THICKNESS_M / 2),
            length_x=ROOM_WIDTH_M,
            length_y=WALL_THICKNESS_M,
            world_t_room=world_t_room,
        ),
    )

    # Top wall (spans full width)
    walls.append(
        create_wall_kinematic_state(
            name="wall_top",
            center_r=Point2D(x=0.0, y=half_height_m + WALL_THICKNESS_M / 2),
            length_x=ROOM_WIDTH_M,
            length_y=WALL_THICKNESS_M,
            world_t_room=world_t_room,
        ),
    )

    # Left wall (spans full height)
    walls.append(
        create_wall_kinematic_state(
            name="wall_left",
            center_r=Point2D(x=-half_width_m - WALL_THICKNESS_M / 2, y=0.0),
            length_x=WALL_THICKNESS_M,
            length_y=ROOM_HEIGHT_M,
            world_t_room=world_t_room,
        ),
    )

    # Right wall (spans full height)
    walls.append(
        create_wall_kinematic_state(
            name="wall_right",
            center_r=Point2D(x=half_width_m + WALL_THICKNESS_M / 2, y=0.0),
            length_x=WALL_THICKNESS_M,
            length_y=ROOM_HEIGHT_M,
            world_t_room=world_t_room,
        ),
    )

    # Dividing wall - bottom section (below doorway)
    divider_height_m = half_height_m - DOORWAY_HALF_WIDTH_M
    walls.append(
        create_wall_kinematic_state(
            name="wall_divider_bottom",
            center_r=Point2D(x=DIVIDING_WALL_X_M, y=-half_height_m + divider_height_m / 2),
            length_x=WALL_THICKNESS_M,
            length_y=divider_height_m,
            world_t_room=world_t_room,
        ),
    )

    # Dividing wall - top section (above doorway)
    walls.append(
        create_wall_kinematic_state(
            name="wall_divider_top",
            center_r=Point2D(x=DIVIDING_WALL_X_M, y=half_height_m - divider_height_m / 2),
            length_x=WALL_THICKNESS_M,
            length_y=divider_height_m,
            world_t_room=world_t_room,
        ),
    )

    door = create_door_kinematic_state(world_t_room)

    return RoomEnvironment(walls=walls, door=door, world_t_room=world_t_room)


def sample_base_pose(
    rng: np.random.Generator,
    occ_grid: OccupancyGrid2D,
    robot_footprint: RectangularFootprint,
    world_t_room: Pose2D,
    *,
    left_room: bool,
) -> tuple[Pose2D, list[Pose2D]]:
    """Sample a random non-colliding base pose in the specified room.

    :param rng: Random number generator
    :param occ_grid: Occupancy grid used to identify non-colliding base poses
    :param robot_footprint: Rectangular robot footprint used for collision checking
    :param world_t_room: Transform from the world frame to the room frame
    :param left_room: Whether to sample in the left room (True) or right room (False)
    :return: Tuple of (Non-colliding base pase, list of rejected pose samples)
    """
    half_width_m = ROOM_WIDTH_M / 2
    half_height_m = ROOM_HEIGHT_M / 2
    half_wall_thickness_m = WALL_THICKNESS_M / 2
    min_robot_radius_m = min(
        robot_footprint.max_x_m,
        abs(robot_footprint.min_x_m),
        robot_footprint.half_length_y_m,
    )  # Minimum radius (m) of the robot in any orientation
    min_robot_radius_m += GRID_RESOLUTION_M  # Add one cell of extra spacing for rounding

    if left_room:  # Left room: x from -half_width to divider
        x_min = -half_width_m + half_wall_thickness_m + min_robot_radius_m
        x_max = DIVIDING_WALL_X_M - half_wall_thickness_m - min_robot_radius_m
    else:  # Right room: x from divider to half_width
        x_min = DIVIDING_WALL_X_M + half_wall_thickness_m + min_robot_radius_m
        x_max = half_width_m - half_wall_thickness_m - min_robot_radius_m

    y_min = -half_height_m + half_wall_thickness_m + min_robot_radius_m
    y_max = half_height_m - half_wall_thickness_m - min_robot_radius_m

    rejects: list[Pose2D] = []
    while True:  # Rejection sample until a valid pose is found
        # Sample in room-frame coordinates
        room_x = rng.uniform(x_min, x_max)
        room_y = rng.uniform(y_min, y_max)
        room_yaw = rng.uniform(-np.pi, np.pi)
        pose_r = Pose2D(room_x, room_y, room_yaw, ref_frame="room")

        pose_w = world_t_room @ pose_r  # Transform to world frame

        if robot_footprint.is_collision_free(robot_pose=pose_w, occupancy_grid=occ_grid):
            console.print(
                f"Rejected {len(rejects)} samples before finding a collision-free base pose.",
            )
            return pose_w, rejects

        rejects.append(pose_w)


def main() -> None:
    """Run the path planning demonstration."""
    rng = np.random.default_rng()
    console.print("[bold blue]Path Planning Demo[/bold blue]")
    console.print()

    footprint = RectangularFootprint(
        max_x_m=ROBOT_MAX_X_M,
        min_x_m=ROBOT_MIN_X_M,
        half_length_y_m=ROBOT_HALF_WIDTH_M,
    )
    console.print(
        f"Robot footprint: {footprint.min_x_m} m to {footprint.max_x_m} m (x), "
        f"+/-{footprint.half_length_y_m} m (y)",
    )

    rasterizer = CollisionModelRasterizer(
        min_height_m=RASTERIZE_MIN_HEIGHT_M,
        max_height_m=RASTERIZE_MAX_HEIGHT_M,
    )

    while True:
        # Step 1: Sample a random rotation and create the environment
        rotation_rad = rng.uniform(-np.pi, np.pi)
        degrees_str = f"{np.rad2deg(rotation_rad):.1f} deg"
        console.print(f"[bold]Step 1:[/] Creating environment with rotation {degrees_str}...")

        room_env = create_environment(rotation_rad=rotation_rad)
        world_t_room = room_env.world_t_room

        # Step 2: Create an empty occupancy grid, stamp the walls as occupied, and visualize
        console.print("[bold]Step 2:[/] Rasterizing walls into occupancy grid...")

        world_t_grid = Pose2D(x=WORLD_T_GRID_X_M, y=WORLD_T_GRID_Y_M, yaw_rad=0.0)

        grid = DiscreteGrid2D(
            origin=world_t_grid,
            resolution_m=GRID_RESOLUTION_M,
            width_cells=GRID_WIDTH_CELLS,
            height_cells=GRID_HEIGHT_CELLS,
        )
        occ_grid = OccupancyGrid2D(grid)

        for wall in room_env.walls:
            occ_grid = occ_grid.stamp_as_occupied(wall, rasterizer)

        console.print(f"  Grid size: {grid.height_cells} rows by {grid.width_cells} columns.")
        console.print(f"  Grid resolution: {grid.resolution_m:.3f} m/cell.")
        console.print()

        if VISUALIZATION:
            just_walls_viz = NavigationVisualization(occ_grid, footprint)
            display_in_window(just_walls_viz.image, "Initial Occupancy Grid")

        # Step 3: Sample start and goal poses (sample in room frame, collision check in world)
        console.print("[bold]Step 3:[/] Sampling start pose and goal pose...")

        start_left = bool(rng.integers(low=0, high=1))  # Should the robot start in the left room?

        start_pose, start_rejects = sample_base_pose(
            rng,
            occ_grid,
            footprint,
            world_t_room,
            left_room=start_left,
        )
        goal_pose, goal_rejects = sample_base_pose(
            rng,
            occ_grid,
            footprint,
            world_t_room,
            left_room=not start_left,
        )

        console.print(f"  Start pose: {start_pose}")
        console.print(f"  Goal pose: {goal_pose}")
        console.print()

        if VISUALIZATION:
            rejection_viz = NavigationVisualization(occ_grid, footprint)
            rejection_viz.draw_query_endpoints(start=start_pose, goal=goal_pose)
            for reject in start_rejects + goal_rejects:
                rejection_viz.draw_base_pose(pose_w_b=reject, color=(255, 0, 0), thickness=1)
            display_in_window(rejection_viz.image, "Initial and Goal Poses (red = rejected poses)")

        # Step 4: Plan path with the door open
        console.print("[bold]Step 4:[/] Planning a path when door is OPEN...")
        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=occ_grid,
            robot_footprint=footprint,
        )
        path = plan_se2_path(query)

        if path is None:
            console.print("  [red]FAILED![/] No path found.")
        else:
            console.print(f"  [green]SUCCESS![/] Found path with {len(path)} waypoints.")

        # Step 5: Visualize the planner result when the door is open
        console.print()
        console.print("[bold]Step 5:[/] Visualizing planner result (press any key to continue)...")

        if VISUALIZATION:
            door_open_viz = NavigationVisualization(occ_grid, footprint)
            door_open_viz.draw_path(path)
            door_open_viz.draw_query_endpoints(start_pose, goal_pose)
            display_in_window(door_open_viz.image, "Planned Path (Door Open)")

        if path is None:
            if click.confirm("Sample new poses and try again?", default=True):
                continue
            break  # Otherwise, exit the demo loop

        console.print()

        # Step 6: Stamp the door object as occupied
        console.print("[bold]Step 6:[/] Adding door collision model to occupancy grid...")

        blocked_occ_grid = occ_grid.stamp_as_occupied(room_env.door, rasterizer)

        # Step 7: Compute a plan when the door is closed
        console.print()
        console.print("[bold]Step 7:[/] Planning a path when door is CLOSED...")
        query_blocked = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=blocked_occ_grid,
            robot_footprint=footprint,
        )
        path_blocked = plan_se2_path(query_blocked)

        if path_blocked is None:
            console.print("  [green]VERIFIED:[/] No path found when door is closed.")
        else:
            console.print("  [yellow]UNEXPECTED:[/] Found path when door is closed.")
        console.print()

        # Step 8: Visualize the planner result when the door is closed
        console.print("[bold]Step 8:[/] Visualizing planning result after door is closed...")

        if VISUALIZATION:
            blocked_viz = NavigationVisualization(blocked_occ_grid, footprint)
            blocked_viz.draw_path(path_blocked)
            blocked_viz.draw_query_endpoints(start_pose, goal_pose)
            display_in_window(blocked_viz.image, title="Planned Path (Door Closed)")

        # Step 9: Remove the door using the rasterizer and verify a path can be found again
        console.print("[bold]Step 9:[/] Removing door from occupancy grid and replanning...")

        unblocked_occ = blocked_occ_grid.stamp_as_free(obj=room_env.door, rasterizer=rasterizer)
        query_unblocked = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=unblocked_occ,
            robot_footprint=footprint,
        )
        path_unblocked = plan_se2_path(query_unblocked)

        if path_unblocked is None:
            console.print("  [yellow]UNEXPECTED:[/] No path found after removing door.")
        else:
            console.print(
                f"  [green]VERIFIED:[/] Path found again with {len(path_unblocked)} waypoints.",
            )
        console.print()

        # Step 10: Visualize the planner result after the door has been removed
        console.print("[bold]Step 10:[/] Visualizing planning result after door is cleared...")

        if VISUALIZATION:
            unblocked_viz = NavigationVisualization(unblocked_occ, footprint)
            unblocked_viz.draw_path(path_unblocked)
            unblocked_viz.draw_query_endpoints(start_pose, goal_pose)
            display_in_window(unblocked_viz.image, title="Planned Path (Door Removed)")

        console.print()
        if not click.confirm("Run another iteration with new sampled values?", default=None):
            console.print("[green]Demo complete![/]")
            break


if __name__ == "__main__":
    main()
