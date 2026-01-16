"""Define a class for 2D grid-based path planning using A*."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

from robotics_utils.perception import GridCell
from robotics_utils.spatial import Pose2D

if TYPE_CHECKING:
    from .navigation_query import NavigationQuery


class GridPlanner2D:
    """2D grid-based path planner using A* algorithm with footprint collision checking.

    This planner uses the python-pathfinding library for A* search and incorporates
    robot footprint collision checking at each candidate cell to ensure paths are
    feasible for the robot's actual geometry.
    """

    def plan(self, query: NavigationQuery) -> list[Pose2D] | None:
        """Plan a path from start to goal, checking footprint at each step.

        :param query: Navigation query with start, goal, grid, and footprint
        :return: List of Pose2D waypoints from start to goal, or None if no path exists
        """
        grid = query.occupancy_grid
        start_cell = query.start_grid_coords
        goal_cell = query.goal_grid_coords

        # Check bounds
        if not grid.is_valid_cell(start_cell):
            return None  # Start out of bounds
        if not grid.is_valid_cell(goal_cell):
            return None  # Goal out of bounds

        # Check footprint collision at start and goal
        if self._check_footprint_collision_at_cell(start_cell, query):
            return None  # Start in collision
        if self._check_footprint_collision_at_cell(goal_cell, query):
            return None  # Goal in collision

        # Create cost matrix for pathfinding (0 = traversable, 1 = obstacle)
        occupancy_mask = grid.get_occupied_mask()
        cost_matrix = occupancy_mask.astype(np.uint8)

        # Create Grid for pathfinding library
        # The library uses matrix[x][y] indexing, so we transpose (row, col) -> (col, row)
        pathfinding_grid = Grid(matrix=cost_matrix.T.tolist())

        # Get start and goal nodes using (col, row) order for the library
        start_node = pathfinding_grid.node(start_cell.col, start_cell.row)
        end_node = pathfinding_grid.node(goal_cell.col, goal_cell.row)

        # Find path using A*
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
        path, _ = finder.find_path(start_node, end_node, pathfinding_grid)

        if not path:
            return None  # No path found

        # Convert path to world frame Pose2D list
        # Path returns (col, row) tuples due to the transpose
        waypoints = []
        for col, row in path:
            cell = GridCell(row, col)
            world_point = grid.grid_to_world(cell)
            world_pose = Pose2D(
                x=world_point.x,
                y=world_point.y,
                yaw_rad=grid.origin.yaw_rad,
                ref_frame=grid.origin.ref_frame,
            )
            waypoints.append(world_pose)

        return waypoints

    def check_feasibility(self, query: NavigationQuery) -> bool:
        """Check if navigation query is feasible (path exists).

        :param query: Navigation query to check
        :return: True if path exists, False otherwise
        """
        return self.plan(query) is not None

    def _check_footprint_collision_at_cell(
        self, cell: GridCell, query: NavigationQuery
    ) -> bool:
        """Check if robot footprint at given cell would collide.

        :param cell: Grid cell (row, col) to check
        :param query: Navigation query
        :return: True if collision detected
        """
        grid = query.occupancy_grid
        world_point = grid.grid_to_world(cell)
        robot_pose = Pose2D(
            x=world_point.x,
            y=world_point.y,
            yaw_rad=grid.origin.yaw_rad,
            ref_frame=grid.origin.ref_frame,
        )

        return query.robot_footprint.check_collision(
            robot_pose=robot_pose,
            occupancy_grid=grid,
        )
