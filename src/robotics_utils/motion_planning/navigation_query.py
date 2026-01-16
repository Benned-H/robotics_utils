"""Define a class representing a navigation feasibility query."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint
    from robotics_utils.perception import GridCell, OccupancyGrid2D
    from robotics_utils.spatial import Pose2D


@dataclass(frozen=True)
class NavigationQuery:
    """Navigation feasibility query for 2D path planning.

    This data structure encapsulates all information needed to perform a
    navigation feasibility check or path planning query on a 2D occupancy grid.
    """

    start_pose: Pose2D
    goal_pose: Pose2D

    occupancy_grid: OccupancyGrid2D
    """An occupancy grid defining the probability of occupancy for a grid of cells."""

    robot_footprint: RectangularFootprint
    """Robot footprint used for collision checking."""

    @property
    def start_grid_coords(self) -> GridCell:
        """Get the start pose as (row, col) grid coordinates."""
        return self.occupancy_grid.world_to_grid(self.start_pose.x, self.start_pose.y)

    @property
    def goal_grid_coords(self) -> GridCell:
        """Get the goal pose as (row, col) grid coordinates."""
        return self.occupancy_grid.world_to_grid(self.goal_pose.x, self.goal_pose.y)
