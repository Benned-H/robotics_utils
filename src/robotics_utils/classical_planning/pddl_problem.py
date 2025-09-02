"""Define a dataclass to represent PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.classical_planning.abstract_states import AbstractState
    from robotics_utils.classical_planning.predicates import PredicateInstance
    from robotics_utils.objects import ObjectCentricState


@dataclass(frozen=True)
class GoalCondition:
    """Specifies a desired abstract state of the world for a planning problem.

    TODO: Handle general first-order logic expressions as goal conditions.
    """

    positive: set[PredicateInstance]  # Conditions that must be true
    negative: set[PredicateInstance]  # Conditions that must be false


@dataclass(frozen=True)
class PDDLProblem:
    """A PDDL problem defines an initial environment state and a goal condition."""

    objects: ObjectCentricState
    initial_state: AbstractState
    goal: GoalCondition
