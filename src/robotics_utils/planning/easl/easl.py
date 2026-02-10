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
        """Initialize the state lattice."""
        self._adaptive = adaptive

        # TODO

    @property
    def is_adaptive(self) -> bool:
        """Retrieve whether or not this state lattice is adaptive."""
        return self._adaptive

    def search(self, n_start: Node, n_goal: Node, cost_map: CostMap) -> Path | None:
        """Solve the given planning query by running weighted A* search over the state lattice.

        Reference: Algorithm 1 (pg. 5766) of Hedegaard et al., IROS 2021.

        :param n_start: Start node of the search
        :param n_goal: Target node to be reached via search
        :param cost_map: Discrete 2D grid representing environment costs
        :return: Path to the goal, if found, otherwise None.
        """
        open_list = {n_start}
        closed_list = set()

        while open_list:
            n_curr = open_list.pop()
            closed_list.add(n_curr)
            if n_curr == n_goal:
                return backtrack(n_curr)
            open_list = self.expand_node(n_curr, n_goal, open_list, closed_list, cost_map)

        return open_list

    def expand_node(
        self,
        n_curr: Node,
        n_goal: Node,
        open_list: list[Node],
        closed_list: list[Node],
        cost_map: CostMap,
    ) -> list[Node]:
        """Expand the given node and update the open list accordingly.

        Reference: Algorithm 2 (pg. 5767) of Hedegaard et al., IROS 2021.

        TODO: Document parameters

        :return: Updated open list
        """
        for e in self.control_set(n_curr):
            n_new = Node(
                n_curr.x + e.final_offset.x,
                n_curr.y + e.final_offset.y,
                e.n_final.heading_rad,
                e.n_final.v_x_mps,
            )
            if n_new in closed_list:
                continue
            if n_new == n_goal:
                n_new.offset = n_goal.offset
            elif self.is_adaptive:
                n_new.offset = self.adapt(n_new, cost_map)

            e_new = self.connect(n_curr, n_new)
            n_new.g = n_curr.g + cost(e_new, cost_map)
            n_new.f = n_new.g + w_h * heuristic(n_new, n_goal)

            n_open = open_list.get(n_new.indices)  # TODO: How should this lookup actually work?
            if n_open is not None and n_new.f < n_open.f:
                open_list.replace(n_open, n_new)
            else:
                open_list.push(n_new)

        return open_list

    def adapt(self, node: Node, cost_map: CostMap) -> Offset:
        """Identify the best local discrete adaptation of the given node.

        Reference: Algorithm 3 (pg. 5768) of Hedegaard et al., IROS 2021.

        :param node: Discrete node to be adapted in the state lattice
        :param cost_map: Current 2D cost map of the environment
        :return: Best local discrete adaptation for the node
        """
        existing_offset = self.get_offset(node)  # TODO: Implement method
        if existing_offset is not None:
            return existing_offset

        prior_offset = self.get_prior_offset(node)  # TODO: Implement method
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
        self.set_offset(node, node.offset)  # TODO: Implement method
        return node.offset
