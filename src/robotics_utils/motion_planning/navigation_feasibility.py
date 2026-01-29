"""Define a high-level interface for checking navigation feasibility in hypothetical worlds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotics_utils.motion_planning.discretization import DiscreteGrid2D
from robotics_utils.motion_planning.navigation_query import NavigationQuery
from robotics_utils.motion_planning.se2_planner import plan_se2_path
from robotics_utils.perception import OccupancyGrid2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D

if TYPE_CHECKING:
    from robotics_utils.collision_models import CollisionModelRasterizer
    from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint
    from robotics_utils.states import ObjectKinematicState


class NavigationFeasibilityChecker:
    """High-level interface for checking navigation feasibility in hypothetical worlds.

    This class maintains a persistent occupancy grid built from LiDAR scans and
    provides navigation feasibility checks that can account for hypothetical
    object removals (e.g., for TAMP planning).
    """

    def __init__(
        self,
        robot_footprint: RectangularFootprint,
        rasterizer: CollisionModelRasterizer,
        resolution_m: float = 0.1,
        grid_size_m: float = 30.0,
        ref_frame: str = DEFAULT_FRAME,
    ) -> None:
        """Initialize the navigation feasibility checker.

        :param robot_footprint: Robot footprint used for collision checking
        :param rasterizer: Collision model rasterizer used to compute object occupancy masks
        :param resolution_m: Grid resolution in meters (default: 0.1 m)
        :param grid_size_m: Grid size in meters (default: 30 m x 30 m)
        :param ref_frame: Name of the global reference frame (default: "map")
        """
        self.robot_footprint = robot_footprint
        self.rasterizer = rasterizer

        # Initialize an occupancy grid based on the other inputs
        size_cells = int(grid_size_m / resolution_m)
        actual_size_m = size_cells * resolution_m
        half_size_m = actual_size_m / 2.0

        origin = Pose2D(x=-half_size_m, y=half_size_m, yaw_rad=0.0, ref_frame=ref_frame)

        discrete_grid = DiscreteGrid2D(origin, resolution_m, size_cells, size_cells)
        self.occupancy_grid = OccupancyGrid2D(discrete_grid)

    def remove_objects(self, objects: list[ObjectKinematicState]) -> OccupancyGrid2D:
        """Create an occupancy grid copy with the given objects' footprints marked as free.

        :param objects: List of objects to be removed from the occupancy grid
        :return: Modified occupancy grid based on the requested removals
        """
        occ_grid = self.occupancy_grid.copy()
        for obj in objects:
            occ_grid.stamp_as_free(obj=obj, rasterizer=self.rasterizer)
        return occ_grid

    def plan_path(
        self,
        start_pose: Pose2D,
        goal_pose: Pose2D,
        removed_objects: list[ObjectKinematicState] | None = None,
    ) -> list[Pose2D] | None:
        """Plan a path from a start pose to a goal pose.

        :param start_pose: Start pose in world frame
        :param goal_pose: Goal pose in world frame
        :param removed_objects: Objects to remove from the map for hypothetical planning
        :return: List of waypoints from start to goal, or None if no path is found
        """
        if removed_objects is None:
            grid = self.occupancy_grid
        else:
            grid = self.remove_objects(removed_objects)

        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=grid,
            robot_footprint=self.robot_footprint,
        )
        return plan_se2_path(query)
