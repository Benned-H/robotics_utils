"""Define a class representing navigation feasibility queries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from robotics_utils.spatial import Pose2D

from .footprint import RectangularFootprint


@dataclass(frozen=True)
class NavigationQuery:
    """Navigation feasibility query for 2D path planning.

    This data structure encapsulates all information needed to perform a
    navigation feasibility check or path planning query on a 2D occupancy grid.
    """

    occupancy_mask: np.ndarray  # Boolean mask (True = occupied)
    start_pose: Pose2D  # Start pose in world frame
    goal_pose: Pose2D  # Goal pose in world frame
    resolution_m: float  # Grid resolution
    origin: Pose2D  # Grid origin
    robot_footprint: RectangularFootprint  # Robot footprint for collision checking

    def to_image(self) -> np.ndarray:
        """Convert occupancy mask to image format for python-pathfinding.

        python-pathfinding expects a 2D array where:
        - 0 = free space (can traverse)
        - 1 = occupied space (cannot traverse)

        :return: 2D uint8 array for python-pathfinding
        """
        return self.occupancy_mask.astype(np.uint8)

    def get_start_grid_coords(self) -> tuple[int, int]:
        """Get start pose as grid coordinates.

        :return: (grid_x, grid_y) tuple
        """
        return self._pose_to_grid(self.start_pose)

    def get_goal_grid_coords(self) -> tuple[int, int]:
        """Get goal pose as grid coordinates.

        :return: (grid_x, grid_y) tuple
        """
        return self._pose_to_grid(self.goal_pose)

    def _pose_to_grid(self, pose: Pose2D) -> tuple[int, int]:
        """Convert pose to grid coordinates.

        :param pose: Pose in world frame
        :return: (grid_x, grid_y) tuple
        """
        # Transform point relative to grid origin
        dx = pose.x - self.origin.x
        dy = pose.y - self.origin.y

        # Rotate by -origin.yaw to align with grid axes
        cos_yaw = np.cos(-self.origin.yaw_rad)
        sin_yaw = np.sin(-self.origin.yaw_rad)

        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw

        # Convert to grid indices
        grid_x = int(np.floor(local_x / self.resolution_m))
        grid_y = int(np.floor(local_y / self.resolution_m))

        return grid_x, grid_y
