"""Define a simulator to generate random transitions in PDDL problems."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.abstractions.learning.transitions import AbstractTransition

if TYPE_CHECKING:
    from robotics_utils.abstractions.pddl.pddl_domain import PDDLDomain
    from robotics_utils.abstractions.pddl.pddl_problem import PDDLProblem
    from robotics_utils.abstractions.symbols.abstract_states import AbstractState
    from robotics_utils.abstractions.symbols.operators import GroundOperator


class PDDLSimulator:
    """A simulator for generating random transitions in a PDDL problem."""

    def __init__(self, domain: PDDLDomain, problem: PDDLProblem) -> None:
        """Initialize the PDDL simulator for the given PDDL problem."""
        self.operators = domain.operators
        self.problem = problem
        self._all_ground_operators = self._compute_ground_operators()

    def _compute_ground_operators(self) -> list[GroundOperator]:
        """Compute all valid groundings for each operator in the domain."""
        all_ground_operators: list[GroundOperator] = []

        for operator in self.operators:
            operator_groundings = operator.compute_all_groundings(self.problem.objects)
            all_ground_operators.extend(operator_groundings)

        return all_ground_operators

    def get_applicable_operators(self, state: AbstractState) -> list[GroundOperator]:
        """Find all ground operators applicable in the given abstract state."""
        return [g_op for g_op in self._all_ground_operators if g_op.applicable_in(state)]

    def generate_random_transition(
        self,
        state: AbstractState,
        rng: np.random.Generator | None = None,
    ) -> AbstractTransition[GroundOperator]:
        """Generate a single random transition (success or fail) from the given abstract state.

        :param state: Current abstract state
        :param rng: Optional random number generator, defaults to None
        :return: Resulting abstract transition (may succeed or fail)
        """
        if rng is None:
            rng = np.random.default_rng()

        num_operators = len(self._all_ground_operators)
        selected_idx = rng.choice(num_operators)
        selected_op = self._all_ground_operators[selected_idx]

        next_state = selected_op.apply(state)

        return AbstractTransition(
            before=state,
            action=selected_op,
            success=(next_state is not None),
            after=next_state,
        )

    def generate_random_transitions(
        self,
        n: int,
        start_state: AbstractState | None = None,
        rng: np.random.Generator | None = None,
    ) -> list[AbstractTransition[GroundOperator]]:
        """Generate N random transitions starting from the given abstract state.

        :param n: Number of transitions to generate
        :param start_state: Initial state (if None, defaults to problem's initial state)
        :param rng: Optional random number generator, defaults to None
        :return: List of generated abstract transitions
        """
        if start_state is None:
            start_state = self.problem.initial_state

        if rng is None:
            rng = np.random.default_rng()

        transitions: list[AbstractTransition[GroundOperator]] = []
        curr_state: AbstractState | None = start_state

        for _ in range(n):
            if curr_state is None:  # Break if the previous operator failed
                break

            next_transition = self.generate_random_transition(curr_state, rng)
            curr_state = next_transition.after
            transitions.append(next_transition)

        return transitions
