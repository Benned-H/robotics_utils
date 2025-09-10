"""Define a dataclass to represent PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.classical_planning.abstract_states import AbstractState, GoalCondition
    from robotics_utils.classical_planning.objects import Objects


@dataclass(frozen=True)
class PDDLProblem:
    """A PDDL problem defines an initial environment state and a goal condition."""

    objects: Objects
    initial_state: AbstractState
    goal: GoalCondition
