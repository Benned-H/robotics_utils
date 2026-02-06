"""Define a high-level interface for checking navigation feasibility in hypothetical worlds."""

from __future__ import annotations

from typing import TYPE_CHECKING

from robotics_utils.motion_planning.navigation_query import NavigationQuery
from robotics_utils.motion_planning.se2_planner import plan_se2_path

if TYPE_CHECKING:
    from robotics_utils.collision_models import CollisionModelRasterizer
    from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint
    from robotics_utils.perception import OccupancyGrid2D
    from robotics_utils.spatial import Pose2D
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
        occupancy_grid: OccupancyGrid2D,
    ) -> None:
        """Initialize the navigation feasibility checker.

        :param robot_footprint: Robot footprint used for collision checking
        :param rasterizer: Collision model rasterizer used to compute object occupancy masks
        :param occupancy_grid: Grid defining occupied space in the environment
        """
        self.robot_footprint = robot_footprint
        self.rasterizer = rasterizer
        self.occupancy_grid = occupancy_grid

    @property
    def grid_ref_frame(self) -> str:
        """Retrieve the name of the reference frame used by the stored occupancy grid."""
        return self.occupancy_grid.grid.origin.ref_frame

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
        *,
        verbose: bool = True,
    ) -> list[Pose2D] | None:
        """Plan a path from a start pose to a goal pose.

        :param start_pose: Start pose in world frame
        :param goal_pose: Goal pose in world frame
        :param removed_objects: Objects to remove from the map for hypothetical planning
        :param verbose: If True, log planning progress to the console (default: True)
        :return: List of waypoints from start to goal, or None if no path is found
        """
        if removed_objects is None:
            grid = self.occupancy_grid
        else:
            violations = [
                f"pose of '{obj.name}' is in frame '{obj.pose.ref_frame}'"
                for obj in removed_objects
                if obj.pose.ref_frame != self.grid_ref_frame
            ]
            if violations:
                combined = "; ".join(violations)
                raise RuntimeError(
                    f"Occupancy grid uses frame '{self.grid_ref_frame}' but: {combined}.",
                )

            grid = self.remove_objects(removed_objects)

        query = NavigationQuery(
            start_pose=start_pose,
            goal_pose=goal_pose,
            occupancy_grid=grid,
            robot_footprint=self.robot_footprint,
        )
        return plan_se2_path(query, verbose=verbose)
