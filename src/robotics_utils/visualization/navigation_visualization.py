"""Define a class to visualize navigation plans, occupancy grids, and robot footprints."""

from __future__ import annotations

import cv2
import numpy as np

from robotics_utils.collision_models import Box
from robotics_utils.geometry import Point2D
from robotics_utils.motion_planning import RectangularFootprint
from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import Pose2D
from robotics_utils.vision import RGB, PixelXY, RGBImage

OCCUPIED_RGB: RGB = (50, 50, 50)
PATH_RGB: RGB = (51, 153, 255)
FOOTPRINT_RGB: RGB = (102, 178, 255)

START_RGB: RGB = (0, 255, 0)
GOAL_RGB: RGB = (255, 0, 0)

OBJECT_OUTLINE_RGB = (153, 50, 204)


class NavigationVisualization:
    """Visualizes navigation queries, plans, and robot footprints."""

    def __init__(
        self,
        occ_grid: OccupancyGrid2D,
        robot_footprint: RectangularFootprint,
        scale: int = 5,
    ) -> None:
        """Initialize the visualizer with an occupancy grid, robot footprint, and scaling ratio.

        :param occ_grid: Occupancy grid forming the background of the visualization
        :param robot_footprint: Rectangular model of the robot's base
        :param scale: Scaling (pixels/grid cell) from the occupancy grid to the image (default: 3)
        """
        self.occ_grid = occ_grid
        self.robot_footprint = robot_footprint
        self.scale = scale
        """Scaling (pixels/grid cell) used to scale the occupancy grid into an image."""

        # Scale up the occupancy grid to create the base image
        occupied_mask = self.occ_grid.get_occupied_mask()
        h, w = occupied_mask.shape

        # Derivation: (meters per cell) / (pixels per cell) = meters per pixel
        self._m_per_pixel = self.occ_grid.grid.resolution_m / scale

        self._rgb_data = np.ones((self.scaled_h_px, self.scaled_w_px, 3), dtype=np.uint8) * 255

        for grid_row in range(h):  # Rows of the grid increase bottom-to-top in the image
            for grid_col in range(w):  # Columns of the grid increase left-to-right in the image
                if occupied_mask[grid_row, grid_col]:
                    y1 = (h - 1 - grid_row) * scale  # Pixel rows increase top-to-bottom
                    y2 = y1 + scale
                    x1 = grid_col * scale  # Pixel columns increase left-to-right
                    x2 = x1 + scale
                    self._rgb_data[y1:y2, x1:x2] = OCCUPIED_RGB

        self._text_rows_drawn = 0
        """Number of rows of text that have been drawn on the visualization."""

    @property
    def scaled_h_px(self) -> int:
        """Retrieve the height of the visualized grid in pixels."""
        return self.occ_grid.grid.height_cells * self.scale

    @property
    def scaled_w_px(self) -> int:
        """Retrieve the width of the visualized grid in pixels."""
        return self.occ_grid.grid.width_cells * self.scale

    @property
    def image(self) -> RGBImage:
        """Retrieve the visualization as an RGB image."""
        return RGBImage(data=self._rgb_data)

    def _world_to_pixel(self, position_w: Point2D) -> PixelXY:
        """Convert a world-frame position into image pixel coordinates."""
        position_g = self.occ_grid.grid.to_grid_frame(position_w)
        px = int(position_g.x / self._m_per_pixel)
        py = int(self.scaled_h_px - 1 - (position_g.y / self._m_per_pixel))
        return PixelXY((px, py))

    def draw_base_pose(self, pose_w_b: Pose2D, color: RGB, thickness: int) -> None:
        """Draw the robot footprint at the given world-frame robot body pose."""
        corners_px: list[tuple[int, int]] = []
        for corner_b in self.robot_footprint.corners:
            corner_w = pose_w_b @ corner_b
            corners_px.append(self._world_to_pixel(corner_w).to_tuple())

        pts = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(self._rgb_data, [pts], isClosed=True, color=color, thickness=thickness)

        # Draw the robot heading as an arrow
        center_px = self._world_to_pixel(pose_w_b.position).to_tuple()
        front_x = pose_w_b.x + self.robot_footprint.max_x_m * 0.7 * np.cos(pose_w_b.yaw_rad)
        front_y = pose_w_b.y + self.robot_footprint.max_x_m * 0.7 * np.sin(pose_w_b.yaw_rad)
        front_px = self._world_to_pixel(Point2D(front_x, front_y)).to_tuple()
        cv2.arrowedLine(self._rgb_data, center_px, front_px, color, thickness=thickness)

    def draw_box(
        self,
        box: Box,
        pose_w_b: Pose2D,
        color: RGB = OBJECT_OUTLINE_RGB,
        thickness: int = 2,
    ) -> None:
        """Draw a box primitive shape onto the visualization."""
        corners_box = [
            Point2D(x=box.x_m / 2, y=box.y_m / 2),
            Point2D(x=box.x_m / 2, y=-box.y_m / 2),
            Point2D(x=-box.x_m / 2, y=-box.y_m / 2),
            Point2D(x=-box.x_m / 2, y=box.y_m / 2),
        ]

        corners_px: list[tuple[int, int]] = []
        for corner_b in corners_box:
            corner_w = pose_w_b @ corner_b
            corners_px.append(self._world_to_pixel(corner_w).to_tuple())

        pts = np.array(corners_px, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(self._rgb_data, [pts], isClosed=True, color=color, thickness=thickness)

    def draw_path(self, path: list[Pose2D] | None) -> None:
        """Draw the given path of SE(2) waypoints onto the visualization."""
        if path is None:
            return

        # Draw footprints along the path before drawing path lines
        for pose in path:
            self.draw_base_pose(pose_w_b=pose, color=FOOTPRINT_RGB, thickness=1)

        if len(path) > 1:  # Draw path by connecting its waypoints
            path_pxs = [self._world_to_pixel(p.position).to_tuple() for p in path]
            for i in range(len(path_pxs) - 1):
                cv2.line(self._rgb_data, path_pxs[i], path_pxs[i + 1], color=PATH_RGB, thickness=2)

    def draw_query_endpoints(self, start: Pose2D, goal: Pose2D, thickness: int = 2) -> None:
        """Draw the given start pose and goal pose onto the visualization."""
        self.draw_base_pose(pose_w_b=start, color=START_RGB, thickness=thickness)
        self.draw_base_pose(pose_w_b=goal, color=GOAL_RGB, thickness=thickness)

    def draw_text(self, text_row: str, color: RGB = (0, 0, 0)) -> None:
        """Draw the given row of text onto the visualization."""
        origin = (10, 25 + (25 * self._text_rows_drawn))
        cv2.putText(self._rgb_data, text_row, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        self._text_rows_drawn += 1
