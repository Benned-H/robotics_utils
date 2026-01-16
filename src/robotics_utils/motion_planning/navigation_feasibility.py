"""Define a high-level service for checking navigation feasibility with hypothetical worlds."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.perception import LaserScan2D, OccupancyGrid2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D

from .navigation_query import NavigationQuery
from .rectangular_footprint import RectangularFootprint
from .se2_planner import plan_se2_path

if TYPE_CHECKING:
    from robotics_utils.states import ObjectKinematicState


class NavigationFeasibilityChecker:
    """High-level service for checking navigation feasibility with hypothetical worlds.

    This class maintains a persistent occupancy grid built from LiDAR scans and
    provides navigation feasibility checks that can account for hypothetical
    object removals (e.g., for TAMP planning).
    """

    def __init__(
        self,
        robot_footprint: RectangularFootprint,
        resolution_m: float = 0.1,
        grid_size_m: float = 30.0,
        ref_frame: str = DEFAULT_FRAME,
    ) -> None:
        """Initialize the navigation feasibility checker.

        :param robot_footprint: Robot footprint for collision checking
        :param resolution_m: Grid resolution in meters (default: 0.1m)
        :param grid_size_m: Grid size in meters (default: 30m x 30m)
        :param ref_frame: Reference frame name (default: "map")
        """
        self.robot_footprint = robot_footprint
        self.resolution_m = resolution_m
        self.grid_size_m = grid_size_m
        self.ref_frame = ref_frame

        # Persistent occupancy grid (updated from LiDAR)
        self.base_grid: OccupancyGrid2D | None = None

    def _create_grid(self) -> OccupancyGrid2D:
        """Create a new occupancy grid centered at origin."""
        width_cells = int(self.grid_size_m / self.resolution_m)
        height_cells = int(self.grid_size_m / self.resolution_m)

        origin_x = -self.grid_size_m / 2.0
        origin_y = -self.grid_size_m / 2.0
        origin = Pose2D(x=origin_x, y=origin_y, yaw_rad=0.0, ref_frame=self.ref_frame)

        return OccupancyGrid2D(
            origin=origin,
            resolution_m=self.resolution_m,
            width_cells=width_cells,
            height_cells=height_cells,
            ref_frame=self.ref_frame,
        )

    def build_map_from_scans(self, scans: list[LaserScan2D]) -> None:
        """Build occupancy grid from multiple laser scans (offline mapping).

        :param scans: List of laser scans to incorporate
        """
        if len(scans) == 0:
            return

        self.base_grid = self._create_grid()

        for scan in scans:
            self.base_grid.update(scan)

    def update_map(self, scan: LaserScan2D) -> None:
        """Update occupancy grid with new laser scan (online update).

        :param scan: Laser scan to incorporate
        """
        if self.base_grid is None:
            self.base_grid = self._create_grid()

        self.base_grid.update(scan)

    def _get_grid_for_query(
        self,
        removed_objects: list[ObjectKinematicState] | None = None,
    ) -> OccupancyGrid2D:
        """Get occupancy grid, optionally with objects removed.

        :param removed_objects: Optional list of objects to remove for hypothetical world
        :return: Occupancy grid (copy if objects removed, original otherwise)
        """
        if self.base_grid is None:
            msg = "No occupancy grid has been built yet"
            raise ValueError(msg)

        if not removed_objects:
            return self.base_grid

        grid_copy = self.base_grid.copy()
        for obj in removed_objects:
            grid_copy = grid_copy.remove_object(obj, min_height_m=0.1, max_height_m=2.0)
        return grid_copy

    def check_feasibility(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        removed_objects: list[ObjectKinematicState] | None = None,
    ) -> bool:
        """Check if navigation from start to goal is feasible.

        :param start_pose: Start pose in world frame
        :param goal_pose: Goal pose in world frame
        :param removed_objects: Optional list of objects to remove for hypothetical world
        :return: True if navigation is feasible, False otherwise
        """
        grid = self._get_grid_for_query(removed_objects)

        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=grid,
            robot_footprint=self.robot_footprint,
        )

        path = plan_se2_path(query)
        return path is not None

    def plan_path(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        removed_objects: list[ObjectKinematicState] | None = None,
        num_headings: int = 8,
    ) -> list[Pose2D] | None:
        """Plan a path from start to goal.

        :param start_pose: Start pose in world frame
        :param goal_pose: Goal pose in world frame
        :param removed_objects: Optional list of objects to remove for hypothetical world
        :param num_headings: Number of discrete headings to consider (default 8)
        :return: List of waypoints from start to goal, or None if no path exists
        """
        grid = self._get_grid_for_query(removed_objects)

        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=grid,
            robot_footprint=self.robot_footprint,
        )

        return plan_se2_path(query, num_headings)

    def get_occupancy_grid(
        self,
        removed_objects: list[ObjectKinematicState] | None = None,
    ) -> tuple[OccupancyGrid2D, np.ndarray]:
        """Get occupancy grid and mask for visualization or analysis.

        :param removed_objects: Optional list of objects to remove
        :return: Tuple of (grid, occupancy_mask)
        """
        grid = self._get_grid_for_query(removed_objects)
        return grid, grid.get_occupied_mask()
