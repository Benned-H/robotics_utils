"""Demonstrate navigation planning with occupancy grids and robot footprint collision checking.

This script creates a two-room environment with a doorway, demonstrates path planning,
and shows how adding/removing a door obstacle affects navigation feasibility.

To run this script, use the commands:

    uv venv --clear && uv sync --extra vision
    uv run scripts/navigation_planning_demo.py

"""

from __future__ import annotations

from dataclasses import dataclass

import click
import cv2
import numpy as np
from numpy.typing import NDArray

from robotics_utils.collision_models import Box, CollisionModel
from robotics_utils.io import console
from robotics_utils.motion_planning import (
    DiscreteGrid2D,
    NavigationQuery,
    RectangularFootprint,
    plan_se2_path,
)
from robotics_utils.perception import CollisionModelRasterizer, OccupancyGrid2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D, Pose3D
from robotics_utils.states import ObjectCentricState, ObjectKinematicState
from robotics_utils.vision import RGBImage
from robotics_utils.visualization import display_in_window

# Grid parameters
GRID_RESOLUTION_M = 0.05  # 5 cm per cell
GRID_WIDTH_M = 12.0  # 12 meters wide (extra space for rotated rooms)
GRID_HEIGHT_M = 12.0  # 12 meters tall (extra space for rotated rooms)
GRID_WIDTH_CELLS = int(GRID_WIDTH_M / GRID_RESOLUTION_M)
GRID_HEIGHT_CELLS = int(GRID_HEIGHT_M / GRID_RESOLUTION_M)

# Room layout parameters (in meters, relative to room center)
ROOM_WIDTH_M = 8.0  # Total width of both rooms
ROOM_HEIGHT_M = 6.0  # Total height of the rooms
WALL_THICKNESS_M = 0.1  # 10 cm thick walls
WALL_HEIGHT_M = 2.5  # Height of walls

# Dividing wall and doorway (relative to room center)
DIVIDING_WALL_X_M = 0.0  # Wall dividing the two rooms at center
DOORWAY_HALF_WIDTH_M = 1.0  # Doorway is 2m wide (from -1m to +1m in y)

# Robot footprint (Spot-like dimensions)
ROBOT_MAX_X_M = 0.5  # Front of robot
ROBOT_MIN_X_M = -0.5  # Back of robot
ROBOT_HALF_WIDTH_M = 0.3  # Half-width of robot

# Door dimensions (in meters)
DOOR_THICKNESS_M = 0.1  # Thickness of the door (x-dimension)
DOOR_WIDTH_M = 2 * DOORWAY_HALF_WIDTH_M  # Width matches doorway
DOOR_HEIGHT_M = 2.0  # Height of the door

# Rasterizer height range (robot navigation height)
RASTERIZE_MIN_HEIGHT_M = 0.0
RASTERIZE_MAX_HEIGHT_M = 1.0  # Capture obstacles up to 1m height

# Visualization scale factor (pixels per grid cell)
VIZ_SCALE = 3

# Log-odds value for occupied cells
OCCUPIED_LOG_ODDS = 10.0

# Center of the environment (for rotation)
ENV_CENTER_X_M = GRID_WIDTH_M / 2
ENV_CENTER_Y_M = GRID_HEIGHT_M / 2


@dataclass
class RoomEnvironment:
    """Container for all collision objects in the two-room environment."""

    walls: list[ObjectKinematicState]
    door: ObjectKinematicState
    rotation_rad: float
    center_x: float
    center_y: float


def create_rotation_transform(yaw_rad: float, center_x: float, center_y: float) -> Pose3D:
    """Create a rotation transform around a specified center point.

    :param yaw_rad: Rotation angle in radians
    :param center_x: X coordinate of rotation center
    :param center_y: Y coordinate of rotation center
    :return: Pose3D representing the rotation
    """
    return Pose3D.from_xyz_rpy(x=center_x, y=center_y, z=0.0, yaw_rad=yaw_rad, ref_frame=DEFAULT_FRAME)


def transform_pose(local_pose: Pose3D, rotation_rad: float, center_x: float, center_y: float) -> Pose3D:
    """Transform a pose from local (room-centered) coordinates to world coordinates with rotation.

    :param local_pose: Pose in local coordinates (centered at origin)
    :param rotation_rad: Rotation angle to apply
    :param center_x: X coordinate of the environment center
    :param center_y: Y coordinate of the environment center
    :return: Transformed pose in world coordinates
    """
    # Create rotation transform around the center
    rotation_transform = create_rotation_transform(rotation_rad, center_x, center_y)
    # Apply rotation to the local pose (which is relative to center)
    return rotation_transform @ local_pose


def create_wall_kinematic_state(
    name: str,
    center_x: float,
    center_y: float,
    length_x: float,
    length_y: float,
    rotation_rad: float,
    env_center_x: float,
    env_center_y: float,
) -> ObjectKinematicState:
    """Create an ObjectKinematicState for a wall.

    :param name: Name of the wall object
    :param center_x: X position of wall center (in local/room coordinates)
    :param center_y: Y position of wall center (in local/room coordinates)
    :param length_x: Length of wall along x-axis
    :param length_y: Length of wall along y-axis
    :param rotation_rad: Global rotation to apply
    :param env_center_x: X coordinate of environment center
    :param env_center_y: Y coordinate of environment center
    :return: ObjectKinematicState for the wall
    """
    wall_box = Box(x_m=length_x, y_m=length_y, z_m=WALL_HEIGHT_M)
    wall_collision_model = CollisionModel(primitives=[wall_box])

    # Local pose (before rotation, relative to environment center)
    local_pose = Pose3D.from_xyz_rpy(
        x=center_x,
        y=center_y,
        z=WALL_HEIGHT_M / 2,
        ref_frame=DEFAULT_FRAME,
    )

    # Apply rotation
    world_pose = transform_pose(local_pose, rotation_rad, env_center_x, env_center_y)

    return ObjectKinematicState(name=name, pose=world_pose, collision_model=wall_collision_model)


def create_room_environment(rotation_rad: float) -> RoomEnvironment:
    """Create all collision objects for the two-room environment with the given rotation.

    :param rotation_rad: Rotation angle to apply to the entire environment
    :return: RoomEnvironment containing all walls and the door
    """
    walls: list[ObjectKinematicState] = []

    # Half dimensions for positioning
    half_width = ROOM_WIDTH_M / 2
    half_height = ROOM_HEIGHT_M / 2

    # Bottom wall (spans full width)
    walls.append(
        create_wall_kinematic_state(
            name="wall_bottom",
            center_x=0.0,
            center_y=-half_height + WALL_THICKNESS_M / 2,
            length_x=ROOM_WIDTH_M,
            length_y=WALL_THICKNESS_M,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Top wall (spans full width)
    walls.append(
        create_wall_kinematic_state(
            name="wall_top",
            center_x=0.0,
            center_y=half_height - WALL_THICKNESS_M / 2,
            length_x=ROOM_WIDTH_M,
            length_y=WALL_THICKNESS_M,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Left wall (spans full height)
    walls.append(
        create_wall_kinematic_state(
            name="wall_left",
            center_x=-half_width + WALL_THICKNESS_M / 2,
            center_y=0.0,
            length_x=WALL_THICKNESS_M,
            length_y=ROOM_HEIGHT_M,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Right wall (spans full height)
    walls.append(
        create_wall_kinematic_state(
            name="wall_right",
            center_x=half_width - WALL_THICKNESS_M / 2,
            center_y=0.0,
            length_x=WALL_THICKNESS_M,
            length_y=ROOM_HEIGHT_M,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Dividing wall - bottom section (below doorway)
    divider_bottom_height = half_height - DOORWAY_HALF_WIDTH_M
    walls.append(
        create_wall_kinematic_state(
            name="wall_divider_bottom",
            center_x=DIVIDING_WALL_X_M,
            center_y=-half_height + divider_bottom_height / 2,
            length_x=WALL_THICKNESS_M,
            length_y=divider_bottom_height,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Dividing wall - top section (above doorway)
    divider_top_height = half_height - DOORWAY_HALF_WIDTH_M
    walls.append(
        create_wall_kinematic_state(
            name="wall_divider_top",
            center_x=DIVIDING_WALL_X_M,
            center_y=half_height - divider_top_height / 2,
            length_x=WALL_THICKNESS_M,
            length_y=divider_top_height,
            rotation_rad=rotation_rad,
            env_center_x=ENV_CENTER_X_M,
            env_center_y=ENV_CENTER_Y_M,
        )
    )

    # Create door
    door = create_door_kinematic_state(rotation_rad)

    return RoomEnvironment(
        walls=walls,
        door=door,
        rotation_rad=rotation_rad,
        center_x=ENV_CENTER_X_M,
        center_y=ENV_CENTER_Y_M,
    )


def create_empty_occupancy_grid() -> OccupancyGrid2D:
    """Create an empty occupancy grid."""
    grid = DiscreteGrid2D(
        origin=Pose2D(x=0.0, y=0.0, yaw_rad=0.0),
        resolution_m=GRID_RESOLUTION_M,
        width_cells=GRID_WIDTH_CELLS,
        height_cells=GRID_HEIGHT_CELLS,
    )
    return OccupancyGrid2D(grid)


def stamp_walls_into_occupancy(occ_grid: OccupancyGrid2D, walls: list[ObjectKinematicState]) -> None:
    """Stamp all wall collision models into the occupancy grid.

    :param occ_grid: Occupancy grid to modify
    :param walls: List of wall kinematic states to stamp
    """
    for wall in walls:
        wall_mask = CollisionModelRasterizer.rasterize_object(
            obj=wall,
            grid=occ_grid.grid,
            min_height_m=RASTERIZE_MIN_HEIGHT_M,
            max_height_m=RASTERIZE_MAX_HEIGHT_M,
        )
        occ_grid.log_odds[wall_mask] = OCCUPIED_LOG_ODDS


def create_door_kinematic_state(rotation_rad: float) -> ObjectKinematicState:
    """Create an ObjectKinematicState for a door that blocks the doorway.

    :param rotation_rad: Rotation angle to apply to the door pose
    :return: ObjectKinematicState for the door
    """
    # Door collision model: a box primitive
    door_box = Box(
        x_m=DOOR_THICKNESS_M,  # Thickness along x-axis (perpendicular to doorway)
        y_m=DOOR_WIDTH_M,  # Width along y-axis (spans the doorway)
        z_m=DOOR_HEIGHT_M,  # Height
    )
    door_collision_model = CollisionModel(primitives=[door_box])

    # Door pose in local coordinates: centered in the doorway
    local_pose = Pose3D.from_xyz_rpy(
        x=DIVIDING_WALL_X_M,
        y=0.0,  # Centered in doorway
        z=DOOR_HEIGHT_M / 2,  # Box is centered, so place origin at half-height
        ref_frame=DEFAULT_FRAME,
    )

    # Apply rotation
    world_pose = transform_pose(local_pose, rotation_rad, ENV_CENTER_X_M, ENV_CENTER_Y_M)

    return ObjectKinematicState(
        name="door",
        pose=world_pose,
        collision_model=door_collision_model,
    )


def stamp_door_into_occupancy(occ_grid: OccupancyGrid2D, door_state: ObjectKinematicState) -> None:
    """Stamp a door obstacle into the occupancy grid using the rasterizer.

    :param occ_grid: Occupancy grid to modify
    :param door_state: Kinematic state of the door to stamp
    """
    door_mask = CollisionModelRasterizer.rasterize_object(
        obj=door_state,
        grid=occ_grid.grid,
        min_height_m=RASTERIZE_MIN_HEIGHT_M,
        max_height_m=RASTERIZE_MAX_HEIGHT_M,
    )
    occ_grid.log_odds[door_mask] = OCCUPIED_LOG_ODDS


def remove_door_from_occupancy(occ_grid: OccupancyGrid2D, door_state: ObjectKinematicState) -> None:
    """Remove a door obstacle from the occupancy grid using the rasterizer.

    :param occ_grid: Occupancy grid to modify
    :param door_state: Kinematic state of the door to remove
    """
    door_mask = CollisionModelRasterizer.rasterize_object(
        obj=door_state,
        grid=occ_grid.grid,
        min_height_m=RASTERIZE_MIN_HEIGHT_M,
        max_height_m=RASTERIZE_MAX_HEIGHT_M,
    )
    occ_grid.log_odds[door_mask] = -10.0  # Mark as free space


def create_visualization_image(
    occ_grid: OccupancyGrid2D,
    robot_footprint: RectangularFootprint,
    path: list[Pose2D] | None = None,
    start_pose: Pose2D | None = None,
    goal_pose: Pose2D | None = None,
    door_closed: bool = False,
    rotation_deg: float = 0.0,
) -> RGBImage:
    """Create a scaled-up visualization of the occupancy grid with robot footprints.

    :param occ_grid: The occupancy grid to visualize
    :param robot_footprint: Robot footprint for drawing poses
    :param path: Optional navigation path to visualize
    :param start_pose: Optional start pose to highlight
    :param goal_pose: Optional goal pose to highlight
    :param door_closed: Whether the door is currently closed
    :param rotation_deg: Environment rotation angle in degrees
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
    rotation_status = f"Rotation: {rotation_deg:.1f} deg"
    door_status = "Door: CLOSED" if door_closed else "Door: OPEN"
    path_status = f"Path: {len(path)} waypoints" if path else "Path: NOT FOUND"
    cv2.putText(img, rotation_status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, door_status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    cv2.putText(img, path_status, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return RGBImage(data=img)


def sample_pose_in_room(rng: np.random.Generator, left_room: bool, rotation_rad: float) -> Pose2D:
    """Sample a random pose in one of the two rooms, with rotation applied.

    :param rng: Random number generator
    :param left_room: If True, sample in left room; otherwise sample in right room
    :param rotation_rad: Rotation angle to apply to the sampled pose
    :return: Random Pose2D within the specified room (in world coordinates)
    """
    # Leave margin from walls for robot footprint
    margin = max(ROBOT_MAX_X_M, ROBOT_HALF_WIDTH_M) + 0.3

    # Room dimensions (in local coordinates, centered at origin)
    half_width = ROOM_WIDTH_M / 2
    half_height = ROOM_HEIGHT_M / 2

    if left_room:
        # Left room: x from -half_width to divider
        x_min = -half_width + WALL_THICKNESS_M + margin
        x_max = DIVIDING_WALL_X_M - margin
    else:
        # Right room: x from divider to +half_width
        x_min = DIVIDING_WALL_X_M + margin
        x_max = half_width - WALL_THICKNESS_M - margin

    y_min = -half_height + WALL_THICKNESS_M + margin
    y_max = half_height - WALL_THICKNESS_M - margin

    # Sample in local coordinates
    local_x = rng.uniform(x_min, x_max)
    local_y = rng.uniform(y_min, y_max)
    local_yaw = rng.uniform(-np.pi, np.pi)

    # Transform to world coordinates with rotation
    local_pose = Pose3D.from_xyz_rpy(x=local_x, y=local_y, z=0.0, yaw_rad=local_yaw, ref_frame=DEFAULT_FRAME)
    world_pose = transform_pose(local_pose, rotation_rad, ENV_CENTER_X_M, ENV_CENTER_Y_M)

    return world_pose.to_2d()


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
    wall_names = ["wall_bottom", "wall_top", "wall_left", "wall_right", "wall_divider_bottom", "wall_divider_top"]
    env_state = ObjectCentricState(robot_names=["spot"], object_names=wall_names + ["door"], root_frame=DEFAULT_FRAME)
    console.print("Created ObjectCentricState with robot 'spot', walls, and door")
    console.print()

    while True:
        # Step 1: Sample random rotation and create room environment
        rotation_rad = rng.uniform(-np.pi, np.pi)
        console.print(f"[bold]Step 1:[/bold] Creating room environment with rotation {np.degrees(rotation_rad):.1f} deg...")
        room_env = create_room_environment(rotation_rad)
        console.print(f"  Created {len(room_env.walls)} wall segments as CollisionModel objects")
        console.print(f"  Environment center: ({ENV_CENTER_X_M:.1f}, {ENV_CENTER_Y_M:.1f})")
        console.print()

        # Step 2: Create empty occupancy grid and stamp walls using rasterizer
        console.print("[bold]Step 2:[/bold] Rasterizing walls into occupancy grid...")
        occ_grid = create_empty_occupancy_grid()
        stamp_walls_into_occupancy(occ_grid, room_env.walls)

        # Register walls in object-centric state
        for wall in room_env.walls:
            env_state.set_known_object_pose(wall.name, wall.pose)
            env_state.kinematic_tree.set_collision_model(wall.name, wall.collision_model)

        console.print(f"  Grid size: {GRID_WIDTH_CELLS} x {GRID_HEIGHT_CELLS} cells")
        console.print(f"  Resolution: {GRID_RESOLUTION_M}m per cell")
        console.print(f"  World size: {GRID_WIDTH_M}m x {GRID_HEIGHT_M}m")
        console.print()

        # Step 3: Sample start and goal poses (with rotation applied)
        console.print("[bold]Step 3:[/bold] Sampling start pose (left room) and goal pose (right room)...")
        start_pose = sample_pose_in_room(rng, left_room=True, rotation_rad=rotation_rad)
        goal_pose = sample_pose_in_room(rng, left_room=False, rotation_rad=rotation_rad)
        console.print(f"  Start: ({start_pose.x:.2f}, {start_pose.y:.2f}, {np.degrees(start_pose.yaw_rad):.1f} deg)")
        console.print(f"  Goal:  ({goal_pose.x:.2f}, {goal_pose.y:.2f}, {np.degrees(goal_pose.yaw_rad):.1f} deg)")
        console.print()

        # Step 4: Plan path with door open
        console.print("[bold]Step 4:[/bold] Computing navigation plan (door OPEN)...")
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

        # Step 5: Visualize initial trajectory
        console.print()
        console.print("[bold]Step 5:[/bold] Visualizing trajectory (press any key to continue)...")
        rotation_deg = np.degrees(rotation_rad)
        viz_image = create_visualization_image(
            occ_grid, robot_footprint, path=path, start_pose=start_pose, goal_pose=goal_pose,
            door_closed=False, rotation_deg=rotation_deg
        )
        display_in_window(viz_image, title="Navigation Plan - Door Open")

        # Step 6: Add door to object-centric state and stamp into occupancy
        console.print()
        console.print("[bold]Step 6:[/bold] Adding door with CollisionModel (door CLOSED)...")
        door_state = room_env.door
        env_state.set_known_object_pose("door", door_state.pose)
        env_state.kinematic_tree.set_collision_model("door", door_state.collision_model)
        console.print(f"  Door pose: ({door_state.pose.position.x:.2f}, {door_state.pose.position.y:.2f}, {door_state.pose.position.z:.2f})")
        console.print(f"  Door collision model: Box({DOOR_THICKNESS_M}m x {DOOR_WIDTH_M}m x {DOOR_HEIGHT_M}m)")
        console.print()

        # Step 7: Stamp door into occupancy using rasterizer and verify no path
        console.print("[bold]Step 7:[/bold] Rasterizing door into occupancy grid...")
        stamp_door_into_occupancy(occ_grid, door_state)

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

        # Step 8: Visualize blocked state
        console.print()
        console.print("[bold]Step 8:[/bold] Visualizing blocked state (press any key to continue)...")
        viz_image_blocked = create_visualization_image(
            occ_grid, robot_footprint, path=path_blocked, start_pose=start_pose, goal_pose=goal_pose,
            door_closed=True, rotation_deg=rotation_deg
        )
        display_in_window(viz_image_blocked, title="Navigation Plan - Door Closed")

        # Step 9: Remove door using rasterizer and verify path can be found again
        console.print()
        console.print("[bold]Step 9:[/bold] Removing door from occupancy grid using rasterizer (door OPEN)...")
        remove_door_from_occupancy(occ_grid, door_state)
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

        # Step 10: Visualize restored state
        console.print()
        console.print("[bold]Step 10:[/bold] Visualizing restored state (press any key to continue)...")
        viz_image_restored = create_visualization_image(
            occ_grid,
            robot_footprint,
            path=path_unblocked,
            start_pose=start_pose,
            goal_pose=goal_pose,
            door_closed=False,
            rotation_deg=rotation_deg,
        )
        display_in_window(viz_image_restored, title="Navigation Plan - Door Removed")

        console.print()
        console.print("=" * 50)
        if not click.confirm("Run another iteration with new random poses?"):
            console.print("[green]Demo complete![/green]")
            break


if __name__ == "__main__":
    main()
