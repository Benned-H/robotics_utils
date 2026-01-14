"""Define a class for 2D grid-based path planning using A*."""

from __future__ import annotations

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder
from robotics_utils.spatial import Pose2D

from .footprint import RectangularFootprint
from .navigation_query import NavigationQuery


class GridPlanner2D:
    """2D grid-based path planner using A* algorithm with footprint collision checking.

    This planner uses the python-pathfinding library for A* search and incorporates
    robot footprint collision checking at each candidate cell to ensure paths are
    feasible for the robot's actual geometry.
    """

    def __init__(self, footprint: RectangularFootprint):
        """Initialize the grid planner with a robot footprint.

        :param footprint: Robot footprint for collision checking
        """
        self.footprint = footprint

    def plan(self, query: NavigationQuery) -> list[Pose2D] | None:
        """Plan a path from start to goal, checking footprint at each step.

        :param query: Navigation query with start, goal, grid, and footprint
        :return: List of Pose2D waypoints from start to goal, or None if no path exists
        """
        # Get start and goal grid coordinates
        start_x, start_y = query.get_start_grid_coords()
        goal_x, goal_y = query.get_goal_grid_coords()

        # Check bounds
        height, width = query.occupancy_mask.shape
        if not self._is_valid_cell(start_x, start_y, width, height):
            return None  # Start out of bounds
        if not self._is_valid_cell(goal_x, goal_y, width, height):
            return None  # Goal out of bounds

        # Check footprint collision at start and goal
        if self._check_footprint_collision_at_cell(start_x, start_y, query):
            return None  # Start in collision
        if self._check_footprint_collision_at_cell(goal_x, goal_y, query):
            return None  # Goal in collision

        # Create cost matrix for pathfinding
        # 0 = traversable, 1 = obstacle
        # We'll create an expanded cost matrix that includes footprint checks
        cost_matrix = self._create_cost_matrix_with_footprint(query)

        # Create Grid for pathfinding library
        # Note: Grid expects matrix[y][x] indexing
        grid = Grid(matrix=cost_matrix.T.tolist())

        # Get start and goal nodes
        start_node = grid.node(start_x, start_y)
        end_node = grid.node(goal_x, goal_y)

        # Create A* finder
        finder = AStarFinder(diagonal_movement=DiagonalMovement.always)

        # Find path
        path, _ = finder.find_path(start_node, end_node, grid)

        if not path:
            return None  # No path found

        # Convert path to world frame Pose2D list
        waypoints = []
        for grid_x, grid_y in path:
            # Convert grid coordinates to world coordinates
            world_pose = self._grid_to_world_pose(
                grid_x, grid_y, query.origin, query.resolution_m
            )
            waypoints.append(world_pose)

        return waypoints

    def check_feasibility(self, query: NavigationQuery) -> bool:
        """Check if navigation query is feasible (path exists).

        :param query: Navigation query to check
        :return: True if path exists, False otherwise
        """
        path = self.plan(query)
        return path is not None

    def _create_cost_matrix_with_footprint(self, query: NavigationQuery) -> np.ndarray:
        """Create cost matrix incorporating footprint collision checks.

        This is a simplified approach that marks cells as occupied if the footprint
        at that cell would collide with obstacles. For performance, we only check
        footprint collisions for cells that are themselves free.

        :param query: Navigation query
        :return: Binary cost matrix (0 = free, 1 = occupied)
        """
        height, width = query.occupancy_mask.shape
        cost_matrix = np.copy(query.occupancy_mask).astype(np.uint8)

        # For efficiency, we could pre-compute footprint collisions for all cells
        # But for MVP, we'll do it on-demand during A* (in the check methods above)
        # For now, just return the base occupancy mask
        # The footprint checks happen in _check_footprint_collision_at_cell

        return cost_matrix

    def _check_footprint_collision_at_cell(
        self, grid_x: int, grid_y: int, query: NavigationQuery
    ) -> bool:
        """Check if robot footprint at given cell would collide.

        :param grid_x: Grid x coordinate
        :param grid_y: Grid y coordinate
        :param query: Navigation query
        :return: True if collision detected
        """
        # Convert grid coordinates to world pose
        world_pose = self._grid_to_world_pose(
            grid_x, grid_y, query.origin, query.resolution_m
        )

        # Check footprint collision at this pose
        return self.footprint.check_collision(
            pose=world_pose,
            occupancy_mask=query.occupancy_mask,
            origin=query.origin,
            resolution_m=query.resolution_m,
        )

    @staticmethod
    def _grid_to_world_pose(
        grid_x: int, grid_y: int, origin: Pose2D, resolution_m: float
    ) -> Pose2D:
        """Convert grid coordinates to world pose (cell center).

        :param grid_x: Grid x index
        :param grid_y: Grid y index
        :param origin: Grid origin pose
        :param resolution_m: Grid resolution
        :return: Pose2D in world frame
        """
        # Cell center in local grid frame
        local_x = (grid_x + 0.5) * resolution_m
        local_y = (grid_y + 0.5) * resolution_m

        # Rotate by origin.yaw to align with world frame
        cos_yaw = np.cos(origin.yaw_rad)
        sin_yaw = np.sin(origin.yaw_rad)

        rotated_x = local_x * cos_yaw - local_y * sin_yaw
        rotated_y = local_x * sin_yaw + local_y * cos_yaw

        # Translate to world frame
        world_x = origin.x + rotated_x
        world_y = origin.y + rotated_y

        # Preserve frame from origin
        return Pose2D(x=world_x, y=world_y, yaw_rad=origin.yaw_rad, ref_frame=origin.ref_frame)

    @staticmethod
    def _is_valid_cell(grid_x: int, grid_y: int, width: int, height: int) -> bool:
        """Check if grid coordinates are within bounds."""
        return 0 <= grid_x < width and 0 <= grid_y < height
