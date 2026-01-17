"""Define an abstract A* planner over a generic state space.

Reference: Section 3.5.2 (pg. 85-86) of AIMA (4th Ed.) by Russell and Norvig.
"""

from __future__ import annotations

import heapq
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Generic, Hashable, TypeVar

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

    def plan(self, start: StateT, goal: StateT) -> list[StateT] | None:
        """Run A* search from a start state to a goal state.

        :param start: Initial state
        :param goal: Target state
        :return: List of states from start to goal, or None if no plan is found
        """
        frontier: list[AStarNode[StateT]] = []  # Kept as a heap
        reached: dict[Hashable, float] = {}  # Maps state keys -> best g-value seen
        reached[self.state_key(start)] = 0.0

        start_h = self.heuristic(start, goal)
        start_node = AStarNode(f=start_h, g=0.0, state=start)
        heapq.heappush(frontier, start_node)

        while frontier:
            current = heapq.heappop(frontier)
            current_key = self.state_key(current.state)

            # Skip stale entries (we found a better path since this was queued)
            if current.g > reached[current_key]:
                continue

            if self.is_goal(current.state, goal):
                return self._reconstruct_path(current)

            for neighbor_state in self.get_neighbors(current.state):
                neighbor_key = self.state_key(neighbor_state)
                g = current.g + self.cost(current.state, neighbor_state)

                # Eager filter: only add nodes with new or better paths
                if neighbor_key not in reached or g < reached[neighbor_key]:
                    reached[neighbor_key] = g
                    h = self.heuristic(neighbor_state, goal)
                    neighbor_node = AStarNode(f=g + h, g=g, state=neighbor_state, parent=current)
                    heapq.heappush(frontier, neighbor_node)

        return None  # Frontier empty -> No path found

    def _reconstruct_path(self, node: AStarNode[StateT]) -> list[StateT]:
        """Reconstruct the path from the starting state to the given node."""
        path: list[StateT] = []
        current: AStarNode[StateT] | None = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        path.reverse()
        return path
