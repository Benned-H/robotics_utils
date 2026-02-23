"""Define a dataclass to represent PDDL problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.planning.task_planning.abstract_states import AbstractState
    from robotics_utils.planning.task_planning.ground_atom import GroundAtom
    from robotics_utils.planning.task_planning.object_symbols import ObjectSymbols


@dataclass(frozen=True)
class SimpleGoalCondition:
    """A simplified means of expressing the desired abstract state in a planning problem."""

    positive: set[GroundAtom]
    """Conditions that must be true to satisfy the goal condition."""

    negative: set[GroundAtom]
    """Conditions that must be false to satsify the goal condition."""


@dataclass(frozen=True)
class PDDLProblem:
    """A PDDL problem defines an initial abstract state and a goal condition."""

    objects: ObjectSymbols
    initial_abstract_state: AbstractState
    goal: SimpleGoalCondition  # TODO: Replace with general first-order logic expressions

    filepath: Path | None = None
    """Optional path to the corresponding PDDL problem file (default: None)."""
