"""Define a high-level service for checking navigation feasibility with hypothetical worlds."""

from __future__ import annotations

from copy import deepcopy

from robotics_utils.perception import LaserScan2D, OccupancyGrid2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D
from robotics_utils.states import ObjectKinematicState

from .footprint import Footprint
from .grid_planner import GridPlanner2D
from .navigation_query import NavigationQuery


class NavigationFeasibilityChecker:
    """High-level service for checking navigation feasibility with hypothetical worlds.

    This class maintains a persistent occupancy grid built from LiDAR scans and
    provides navigation feasibility checks that can account for hypothetical
    object removals (e.g., for TAMP planning).
    """

    def __init__(
        self,
        resolution_m: float = 0.1,
        grid_size_m: float = 30.0,
        ref_frame: str = DEFAULT_FRAME,
    ):
        """Initialize the navigation feasibility checker.

        :param resolution_m: Grid resolution in meters (default: 0.1m)
        :param grid_size_m: Grid size in meters (default: 30m × 30m)
        :param ref_frame: Reference frame name (default: "map")
        """
        self.resolution_m = resolution_m
        self.grid_size_m = grid_size_m
        self.ref_frame = ref_frame

        # Persistent occupancy grid (updated from LiDAR)
        self.base_grid: OccupancyGrid2D | None = None

        # Planner with Spot footprint
        self.planner = GridPlanner2D(footprint=Footprint.spot_footprint())

    def build_map_from_scans(self, scans: list[LaserScan2D]) -> None:
        """Build occupancy grid from multiple laser scans (offline mapping).

        :param scans: List of laser scans to incorporate
        """
        if len(scans) == 0:
            return

        # Initialize grid centered at origin
        width_cells = int(self.grid_size_m / self.resolution_m)
        height_cells = int(self.grid_size_m / self.resolution_m)

        # Grid origin is at lower-left corner
        origin_x = -self.grid_size_m / 2.0
        origin_y = -self.grid_size_m / 2.0
        origin = Pose2D(x=origin_x, y=origin_y, yaw_rad=0.0, ref_frame=self.ref_frame)

        self.base_grid = OccupancyGrid2D(
            origin=origin,
            resolution_m=self.resolution_m,
            width_cells=width_cells,
            height_cells=height_cells,
            ref_frame=self.ref_frame,
        )

        # Update with each scan
        for scan in scans:
            self.base_grid.update(scan)

    def update_map(self, scan: LaserScan2D) -> None:
        """Update occupancy grid with new laser scan (online update).

        :param scan: Laser scan to incorporate
        """
        if self.base_grid is None:
            # Initialize grid if not already created
            width_cells = int(self.grid_size_m / self.resolution_m)
            height_cells = int(self.grid_size_m / self.resolution_m)

            origin_x = -self.grid_size_m / 2.0
            origin_y = -self.grid_size_m / 2.0
            origin = Pose2D(x=origin_x, y=origin_y, yaw_rad=0.0, ref_frame=self.ref_frame)

            self.base_grid = OccupancyGrid2D(
                origin=origin,
                resolution_m=self.resolution_m,
                width_cells=width_cells,
                height_cells=height_cells,
                ref_frame=self.ref_frame,
            )

        self.base_grid.update(scan)

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
        if self.base_grid is None:
            raise ValueError("No occupancy grid has been built yet")

        # Copy base grid
        grid_copy = self.base_grid.copy()

        # Remove objects for hypothetical world
        if removed_objects:
            for obj in removed_objects:
                grid_copy = grid_copy.remove_object(
                    obj, min_height_m=0.1, max_height_m=2.0
                )

        # Create query
        query = NavigationQuery(
            occupancy_mask=grid_copy.get_occupied_mask(),
            start_pose=start_pose,
            goal_pose=goal_pose,
            resolution_m=self.resolution_m,
            origin=grid_copy.origin,
            robot_footprint=self.planner.footprint,
        )

        # Check feasibility
        return self.planner.check_feasibility(query)

    def plan_path(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        removed_objects: list[ObjectKinematicState] | None = None,
    ) -> list[Pose2D] | None:
        """Plan a path from start to goal.

        :param start_pose: Start pose in world frame
        :param goal_pose: Goal pose in world frame
        :param removed_objects: Optional list of objects to remove for hypothetical world
        :return: List of waypoints from start to goal, or None if no path exists
        """
        if self.base_grid is None:
            raise ValueError("No occupancy grid has been built yet")

        # Copy base grid
        grid_copy = self.base_grid.copy()

        # Remove objects for hypothetical world
        if removed_objects:
            for obj in removed_objects:
                grid_copy = grid_copy.remove_object(
                    obj, min_height_m=0.1, max_height_m=2.0
                )

        # Create query
        query = NavigationQuery(
            occupancy_mask=grid_copy.get_occupied_mask(),
            start_pose=start_pose,
            goal_pose=goal_pose,
            resolution_m=self.resolution_m,
            origin=grid_copy.origin,
            robot_footprint=self.planner.footprint,
        )

        # Plan path
        return self.planner.plan(query)

    def get_occupancy_mask(
        self, removed_objects: list[ObjectKinematicState] | None = None
    ) -> tuple[OccupancyGrid2D, np.ndarray]:
        """Get occupancy mask for visualization or analysis.

        :param removed_objects: Optional list of objects to remove
        :return: Tuple of (grid, occupancy_mask)
        """
        import numpy as np

        if self.base_grid is None:
            raise ValueError("No occupancy grid has been built yet")

        # Copy base grid
        grid_copy = self.base_grid.copy()

        # Remove objects for hypothetical world
        if removed_objects:
            for obj in removed_objects:
                grid_copy = grid_copy.remove_object(
                    obj, min_height_m=0.1, max_height_m=2.0
                )

        return grid_copy, grid_copy.get_occupied_mask()
