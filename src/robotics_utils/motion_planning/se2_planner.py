"""Define an SE(2) A* planner for 2D navigation with robot footprint collision checking."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.motion_planning.discretization import (
    DiscreteAngles,
    DiscreteSE2,
    DiscreteSE2Space,
    GridCell,
)
from robotics_utils.motion_planning.footprint_cell_offsets import FootprintCellOffsets
from robotics_utils.planning import AStarPlanner

if TYPE_CHECKING:
    from robotics_utils.motion_planning.navigation_query import NavigationQuery
    from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint
    from robotics_utils.perception import OccupancyGrid2D
    from robotics_utils.spatial import Pose2D

EIGHT_CONNECTED_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


class SE2AStarPlanner(AStarPlanner[DiscreteSE2]):
    """An A* planner over a discretized SE(2) space of (x, y, heading) indices.

    This planner considers 8-connected grid neighbors with heading changes.
    """

    def __init__(
        self,
        start: DiscreteSE2,
        goal: DiscreteSE2,
        robot_footprint: RectangularFootprint,
        occ_grid: OccupancyGrid2D,
        se2_space: DiscreteSE2Space,
        heading_change_cost: float = 0.5,
    ) -> None:
        """Initialize the SE(2) A* planner with its discrete search space.

        :param start: Initial state during search
        :param goal: Goal state targeted during search
        :param robot_footprint: Rectangular model of the robot base footprint
        :param occ_grid: Occupancy grid of the robot's environment
        :param se2_space: Discrete space over SE(2) defining the search grid and headings
        :param heading_change_cost: Additional cost per discrete heading change (defaults to 0.5)
        """
        self.occupancy_mask = occ_grid.get_occupied_mask()
        self.se2_space = se2_space
        self.heading_change_cost = heading_change_cost

        self._footprint_cells = FootprintCellOffsets(
            se2_space=self.se2_space,
            footprint=robot_footprint,
        )

        super().__init__(start=start, goal=goal)

    def get_neighbors(self, state: DiscreteSE2) -> list[DiscreteSE2]:
        """Return collision-free neighboring states reachable from the given state."""
        neighbors: list[DiscreteSE2] = []

        for dr, dc in EIGHT_CONNECTED_NEIGHBORS:
            new_cell = GridCell(state.cell.row + dr, state.cell.col + dc)

            if not self.se2_space.grid.is_valid_cell(new_cell):
                continue

            # Compute heading index pointing to the neighbor cell
            angle_rad = np.arctan2(dr, dc)
            heading_idx = self.se2_space.headings.nearest_index(angle_rad)

            neighbor = DiscreteSE2(cell=new_cell, heading_idx=heading_idx)
            if self._footprint_cells.is_collision_free(neighbor, self.occupancy_mask):
                neighbors.append(neighbor)

        return neighbors

    def cost(self, pre_state: DiscreteSE2, post_state: DiscreteSE2) -> float:
        """Compute transition cost as Euclidean distance plus a heading change penalty."""
        dr = post_state.cell.row - pre_state.cell.row
        dc = post_state.cell.col - pre_state.cell.col
        distance_m = self.se2_space.grid.resolution_m * np.sqrt(dr * dr + dc * dc)

        heading_diff = abs(post_state.heading_idx - pre_state.heading_idx)
        heading_cost = heading_diff * self.heading_change_cost

        return distance_m + heading_cost

    def heuristic(self, state: DiscreteSE2, goal: DiscreteSE2) -> float:
        """Compute Euclidean distance (meters) as an admissible heuristic for cost-to-go."""
        dr = goal.cell.row - state.cell.row
        dc = goal.cell.col - state.cell.col
        return self.se2_space.grid.resolution_m * np.sqrt(dr * dr + dc * dc)

    def is_goal(self, state: DiscreteSE2, goal: DiscreteSE2) -> bool:
        """Check whether the given state equals the given goal."""
        return state == goal

    def state_key(self, state: DiscreteSE2) -> tuple[int, int, int]:
        """Return a hashable key for the given state."""
        return (state.cell.row, state.cell.col, state.heading_idx)


def plan_se2_path(query: NavigationQuery) -> list[Pose2D] | None:
    """Plan a collision-free path to solve the given navigation query.

    :param query: Navigation query specifying start, goal, occupancy grid, and robot footprint
    :return: List of Pose2D waypoints from start to goal, or None if no path was found
    """
    if not query.robot_footprint.is_collision_free(query.start_pose, query.occupancy_grid):
        return None  # Start pose in collision
    if not query.robot_footprint.is_collision_free(query.goal_pose, query.occupancy_grid):
        return None  # Goal pose in collision

    # Create the discrete SE(2) space used during A* search
    headings = DiscreteAngles(num_angles=8)
    se2_space = DiscreteSE2Space(grid=query.occupancy_grid.grid, headings=headings)

    start = se2_space.discretize(query.start_pose)
    goal = se2_space.discretize(query.goal_pose)

    if not se2_space.grid.is_valid_cell(start.cell):
        return None  # Start cell out of bounds
    if not se2_space.grid.is_valid_cell(goal.cell):
        return None  # Goal cell out of bounds

    planner = SE2AStarPlanner(
        start=start,
        goal=goal,
        robot_footprint=query.robot_footprint,
        occ_grid=query.occupancy_grid,
        se2_space=se2_space,
    )

    # Plan using A* search
    done = False
    log_every_n_steps = 1000

    while not done:
        done = planner.step()

        if not (planner.steps_taken % log_every_n_steps):
            planner.log_info()

    discrete_plan = planner.reconstruct_path()

    if discrete_plan is None:
        return None

    # Convert to Pose2D waypoints (use exact start and goal poses)
    waypoints = [se2_space.convert_discrete_to_pose(indices) for indices in discrete_plan]
    if waypoints:
        waypoints[0] = query.start_pose
        waypoints[-1] = query.goal_pose

    return waypoints
