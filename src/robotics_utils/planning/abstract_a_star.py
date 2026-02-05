"""Define an abstract A* planner over a generic state space.

Reference: Section 3.5.2 (pg. 85-86) of AIMA (4th Ed.) by Russell and Norvig.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

from robotics_utils.io import console

StateT = TypeVar("StateT")


@dataclass(order=True)
class AStarNode(Generic[StateT]):
    """A node in the A* search tree (represents a particular path to a state).

    Nodes are ordered by f-value (estimated total cost) for use in the priority queue.
    """

    f: float
    """Estimated cost of the best path continuing from the node to the goal."""

    g: float = field(compare=False)
    """Path cost from the initial node to this node."""

    state: StateT = field(compare=False)
    parent: AStarNode[StateT] | None = field(default=None, compare=False)


class AStarPlanner(ABC, Generic[StateT]):
    """Abstract A* planner over a generic state space.

    Subclasses implement domain-specific methods for neighbors, cost, heuristic,
    and goal checking. The A* algorithm itself is provided by this base class.
    """

    @abstractmethod
    def get_neighbors(self, state: StateT) -> list[StateT]:
        """Return valid neighboring states reachable from the given state.

        This method should only return states that pass validity checks (e.g., collision-free).
        """

    @abstractmethod
    def cost(self, pre_state: StateT, post_state: StateT) -> float:
        """Compute the cost to transition from one state to another.

        :param pre_state: State beginning the transition
        :param post_state: State after the transition
        :return: Non-negative transition cost
        """

    @abstractmethod
    def heuristic(self, state: StateT, goal: StateT) -> float:
        """Compute an admissible heuristic estimate of cost-to-go.

        Must be optimistic (i.e., never overestimate the true cost) for A* to be cost-optimal.
        """

    @abstractmethod
    def is_goal(self, state: StateT, goal: StateT) -> bool:
        """Check whether the given state satisfies the goal condition."""

    @abstractmethod
    def state_key(self, state: StateT) -> Hashable:
        """Return a hashable key for the given state.

        Used for efficient lookup in the reached set. States with the same key
        are considered equivalent by the A* planner.
        """

    def __init__(self, start: StateT, goal: StateT) -> None:
        """Initialize the A* planner with a start and a goal state.

        :param start: Initial state during search
        :param goal: Goal state to target during search
        """
        self.start = start
        self.goal = goal

        self.frontier: list[AStarNode[StateT]] = []
        """Heap of nodes to be expanded."""

        self.reached: dict[Hashable, float] = {}
        """A map from state keys to the best g-value seen for states with that key."""

        # Initialize the search process using the start state
        start_key = self.state_key(state=self.start)
        self.reached[start_key] = 0.0

        start_h = self.heuristic(self.start, self.goal)
        start_node = AStarNode(f=start_h, g=0.0, state=self.start, parent=None)
        heapq.heappush(self.frontier, start_node)

        self._num_step_calls: int = 0
        """Number of times the `step()` method has been called."""

        self._nodes_expanded: int = 0
        """Number of nodes whose successors have been enumerated and added to the frontier."""

        self._solution_node: AStarNode[StateT] | None = None

    @property
    def steps_taken(self) -> int:
        """Retrieve the number of search steps that the planner has taken."""
        return self._num_step_calls

    def update_frontier(self, state: StateT, parent_node: AStarNode[StateT]) -> None:
        """Update the frontier with a node for the given parent-state pair, if worthwhile.

        :param state: State encountered during A* search
        :param parent_node: Parent node of the given state
        """
        state_key = self.state_key(state=state)
        g = parent_node.g + self.cost(parent_node.state, state)

        # Eager filter: only add nodes with new or better paths
        if state_key not in self.reached or g < self.reached[state_key]:
            self.reached[state_key] = g  # Mark this state as reached via this parent
            h = self.heuristic(state=state, goal=self.goal)
            new_node = AStarNode(f=g + h, g=g, state=state, parent=parent_node)
            heapq.heappush(self.frontier, new_node)

    def step(self) -> bool:
        """Expand the node at the top of the frontier.

        :return: True if search is complete, otherwise False
        """
        self._num_step_calls += 1

        if not self.frontier:
            return True

        current_node = heapq.heappop(self.frontier)
        current_key = self.state_key(current_node.state)

        # Skip stale entries (we found a better path since this was queued)
        if current_node.g > self.reached[current_key]:
            return False

        if self.is_goal(current_node.state, self.goal):
            self._solution_node = current_node
            return True

        for neighbor_state in self.get_neighbors(current_node.state):
            self.update_frontier(state=neighbor_state, parent_node=current_node)

        self._nodes_expanded += 1

        return False

    def reconstruct_path(self) -> list[StateT] | None:
        """Reconstruct the path represented by the stored solution node.

        :return: List of states in the solution plan, or None if no solution was found
        """
        if self._solution_node is None:
            return None

        path: list[StateT] = []
        current: AStarNode[StateT] | None = self._solution_node
        while current is not None:
            path.append(current.state)
            current = current.parent
        path.reverse()
        return path

    def log_info(self) -> None:
        """Log the current state of A* search to the console."""
        console.print(f"Current frontier size: {len(self.frontier)}.")
        console.print(f"Current number of reached nodes: {len(self.reached)}.")
        console.print(f"Search steps taken: {self._num_step_calls}.")
        console.print(f"Nodes expanded: {self._nodes_expanded}.")
