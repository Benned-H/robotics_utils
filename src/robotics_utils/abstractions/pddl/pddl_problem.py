"""Define a dataclass to represent PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.abstract_states import AbstractState
    from robotics_utils.abstractions.symbols.ground_atom import GroundAtom
    from robotics_utils.abstractions.symbols.objects import Objects


@dataclass(frozen=True)
class GoalCondition:
    """Specifies the desired abstract state(s) in a planning problem.

    Note: This current definition is simplified from what PDDL typically allows.
        - TODO: Support general first-order logic expressions as goal conditions.

    Reference: https://planning.wiki/ref/pddl/problem
    """

    positive: frozenset[GroundAtom]
    """Conditions that must hold in a goal state."""

    negative: frozenset[GroundAtom]
    """Conditions that must not hold in a goal state."""


@dataclass
class PDDLProblem:
    """A PDDL problem defines an initial environment state and a goal condition."""

    objects: Objects
    """Symbols representing the typed objects in the problem."""

    initial_state: AbstractState
    """Initial abstract state (i.e., set of true ground atoms) in the problem."""

    goal: GoalCondition
    """Conditions for a goal state in the problem."""
