"""Define a function to visualize a 2D path over an occupancy grid."""

from __future__ import annotations

import cv2
import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.motion_planning import NavigationQuery
from robotics_utils.spatial import Pose2D
from robotics_utils.vision import RGB, RGBImage

OCCUPIED_RGB: RGB = (50, 50, 50)
PATH_RGB: RGB = (0, 180, 0)
FOOTPRINT_RGB: RGB = (120, 240, 120)

START_RGB: RGB = (0, 255, 0)
GOAL_RGB: RGB = (255, 0, 0)

FOOTPRINT_EVERY_N_WAYPOINTS = 5


def visualize_path(query: NavigationQuery, path: list[Pose2D] | None, scale: int = 3) -> RGBImage:
    """Visualize an SE(2) path over an occupancy grid.

    :param query: Path planning query (start/goal poses + occupancy grid + robot footprint)
    :param path: SE(2) path found to solve the query, or None if no path was found
    :param scale: Ratio (pixels/grid cell) used to scale up the occupancy grid (default: 3)
    :return: RGB image visualizing the path over the occupancy grid
    """
    occupied_mask = query.occupancy_grid.get_occupied_mask()

    # Scale up the occupancy grid to create the base image
    h, w = occupied_mask.shape
    scaled_h, scaled_w = h * scale, w * scale
    m_per_pixel = query.occupancy_grid.grid.resolution_m / scale  # m/cell / (px/cell) = m/px

    image_data = np.ones((scaled_h, scaled_w, 3), dtype=np.uint8) * 255  # All-white base

    for g_row in range(h):  # Rows of the grid increase bottom-to-top
        for g_col in range(w):  # Columns of the grid increase left-to-right
            if occupied_mask[g_row, g_col]:
                y1 = (h - 1 - g_row) * scale  # Pixel rows increase top-to-bottom
                y2 = y1 + scale
                x1 = g_col * scale  # Pixel columns increase left-to-right
                x2 = x1 + scale
                image_data[y1:y2, x1:x2] = OCCUPIED_RGB

    def world_to_pixel(position_w: Point2D) -> tuple[int, int]:
        """Convert a world-frame position into image pixel coordinates."""
        position_g = query.occupancy_grid.grid.to_grid_frame(position_w)
        px = int(position_g.x / m_per_pixel)
        py = int(scaled_h - 1 - (position_g.y / m_per_pixel))
        return px, py

    def draw_robot_footprint(pose_w_b: Pose2D, color: RGB, thickness: int = 2) -> None:
        """Draw the robot footprint at the given world-frame body pose."""
        corners_px: list[tuple[int, int]] = []
        for corner_b in query.robot_footprint.corners:
            corner_w = pose_w_b @ corner_b
            corners_px.append(world_to_pixel(corner_w))

        pts = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(image_data, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw the robot heading as an arrow
        center_px = world_to_pixel(pose_w_b.position)
        front_x = pose_w_b.x + query.robot_footprint.max_x_m * 0.7 * np.cos(pose_w_b.yaw_rad)
        front_y = pose_w_b.y + query.robot_footprint.max_x_m * 0.7 * np.sin(pose_w_b.yaw_rad)
        front_px = world_to_pixel(Point2D(front_x, front_y))
        cv2.arrowedLine(image_data, center_px, front_px, color, thickness=max(1, thickness - 1))

    if path is not None:
        if len(path) > 1:  # Draw path by connecting its waypoints
            path_pxs = [world_to_pixel(p.position) for p in path]
            for i in range(len(path_pxs) - 1):
                cv2.line(image_data, path_pxs[i], path_pxs[i + 1], color=PATH_RGB, thickness=2)

        # Draw footprints spaced along the path
        step = max(1, len(path) // FOOTPRINT_EVERY_N_WAYPOINTS)
        for i in range(0, len(path), step):
            draw_robot_footprint(pose_w_b=path[i], color=FOOTPRINT_RGB)

    # Draw start and goal poses from the query
    draw_robot_footprint(query.start_pose, color=START_RGB, thickness=3)
    draw_robot_footprint(query.goal_pose, color=GOAL_RGB, thickness=3)

    # Annotate the image with status text
    path_status = f"Path: {len(path)} waypoints" if path else "Path: NOT FOUND"
    cv2.putText(image_data, path_status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    return RGBImage(data=image_data)


#         cv2.putText(
#             img,
#             "S",
#             (center[0] - 10, center[1] + 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (0, 0, 255),
#             2,
#         )

#         cv2.putText(
#             img,
#             "G",
#             (center[0] - 10, center[1] + 5),
#             cv2.FONT_HERSHEY_SIMPLEX,
#             0.5,
#             (255, 0, 0),
#             2,
#         )
