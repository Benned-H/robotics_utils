"""Implement the Efficiently Adaptive State Lattice (EASL) planner.

References:
[1] B. Hedegaard, E. Fahnestock, J. Arkin, A. Menon, and T. M. Howard, “Discrete Optimization
    of Adaptive State Lattices for Iterative Motion Planning on Unmanned Ground Vehicles,” in
    2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), Sep. 2021,
    pp. 5764-5771. doi: 10.1109/IROS51168.2021.9636181.

"""

from __future__ import annotations

from typing import Any

Node = Any  # TODO: Implement Node dataclass
CostMap = Any  # TODO: Implement CostMap class
Path = Any  # TODO: Implement (and maybe rename) Path class
Offset = Any  # TODO: Implement Offset dataclass/named tuple


class EASL:
    """Implements the Efficiently Adaptive State Lattice (EASL) planner."""

    def __init__(self, *, adaptive: bool) -> None:
        """Initialize the state lattice's open and closed lists.

        :param adaptive: If True, the state lattice performs discrete node adaptation
        """
        self._adaptive = adaptive

        self.open_list: list[Node] = []  # TODO: Type?
        self.closed_list: list[Node] = []  # TODO: Type?

        # Initialize empty collections of current and previous discrete node adaptations
        self._current_adaptations: dict[Node, Offset] = {}
        self._prior_adaptations: dict[Node, Offset] = {}

    @property
    def is_adaptive(self) -> bool:
        """Retrieve whether or not this state lattice is adaptive."""
        return self._adaptive

    def search(self, n_start: Node, n_goal: Node, cost_map: CostMap) -> Path | None:
        """Solve the given planning query by running weighted A* search over the state lattice.

        Reference: Algorithm 1 (pg. 5766) of Hedegaard et al., IROS 2021.

        :param n_start: Start node of the search
        :param n_goal: Target node to be reached via search
        :param cost_map: Discrete 2D grid representing environment cost
        :return: Path to the goal, if found, otherwise None
        """
        self.open_list.append(n_start)  # TODO: Priority queue instead of list
        self.closed_list = []

        while self.open_list:
            n_curr = self.open_list.pop()
            self.closed_list.append(n_curr)
            if n_curr == n_goal:
                return backtrack(n_curr)  # TODO: Implement
            self.expand_node(n_curr, n_goal, cost_map)

        return None

    def expand_node(self, n_curr: Node, n_goal: Node, cost_map: CostMap) -> None:
        """Expand the given node and update the planner's open list accordingly.

        Reference: Algorithm 2 (pg. 5767) of Hedegaard et al., IROS 2021.

        :param n_curr: Current node to be expanded
        :param n_goal: Goal node targeted by the search
        :param cost_map: Discrete 2D grid representing environment cost
        """
        for e in self.control_set(n_curr):
            n_new = Node(
                n_curr.x + e.final_offset.x,
                n_curr.y + e.final_offset.y,
                e.n_final.heading_rad,
                e.n_final.v_x_mps,
            )
            if n_new in self.closed_list:
                continue
            if n_new == n_goal:
                n_new.offset = n_goal.offset
            elif self.is_adaptive:
                n_new.offset = self.adapt(n_new, cost_map)

            e_new = self.connect(n_curr, n_new)
            n_new.g = n_curr.g + cost(e_new, cost_map)
            n_new.f = n_new.g + w_h * heuristic(n_new, n_goal)

            n_open = self.open_list.get(n_new.indices)  # TODO: How should this lookup work?
            if n_open is not None and n_new.f < n_open.f:
                self.open_list.replace(n_open, n_new)
            else:
                self.open_list.append(n_new)

    def adapt(self, node: Node, cost_map: CostMap) -> Offset:
        """Identify the best local discrete adaptation of the given node.

        Reference: Algorithm 3 (pg. 5768) of Hedegaard et al., IROS 2021.

        :param node: Discrete node to be adapted in the state lattice
        :param cost_map: Current 2D cost map of the environment
        :return: Best local discrete adaptation for the node
        """
        existing_offset = self._current_adaptations.get(node)
        if existing_offset is not None:
            return existing_offset

        prior_offset = self._prior_adaptations.get(node)
        if prior_offset is not None:
            node.offset = prior_offset

        best_swath_cost = swath_cost(node.swath, node, cost_map)
        best_offset = {"x": 0, "y": 0, "theta": 0, "v_x": 0}
        for perturb_offset in self.perturbations:
            node.offset += perturb_offset
            perturbed_swath_cost = swath_cost(node.swath, node, cost_map)
            if perturbed_swath_cost < best_swath_cost:
                best_swath_cost = perturbed_swath_cost
                best_offset = perturb_offset
            node.offset -= perturb_offset
        node.offset += best_offset

        self._current_adaptations[node] = node.offset
        return node.offset
