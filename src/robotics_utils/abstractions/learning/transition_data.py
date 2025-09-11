"""Define classes to represent observed state transitions used for abstraction learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic, List

from robotics_utils.abstractions.predicates import StateT

if TYPE_CHECKING:
    from robotics_utils.abstractions.abstract_states import AbstractState
    from robotics_utils.skills import Skill, SkillInstance


@dataclass(frozen=True)
class SkillTransition(Generic[StateT]):
    """An observed state transition resulting from attempting to execute a skill instance."""

    state_before: StateT
    """State from which the skill execution was attempted."""

    skill_instance: SkillInstance
    """Concrete skill that was (possibly) executed."""

    success: bool
    """Was the skill execution successful?"""

    state_after: StateT | None
    """State after the skill execution, if successful (else None)."""

    def __post_init__(self) -> None:
        """Verify that the constructed skill transition is valid.

        :raises ValueError: If the transition was successful but no "after" state is defined
        """
        if self.success and self.state_after is None:
            raise ValueError("A successful skill transition must include an 'after' state.")

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the transition."""
        return self.skill_instance.skill.name


SkillsTrace = List[SkillTransition[StateT]]
"""A sequence of attempted skill executions."""

Dataset = List[SkillsTrace[StateT]]
"""A collection of skill execution traces."""


@dataclass(frozen=True)
class AbstractTransition:
    """A transition representing the change in abstract state due to a skill execution."""

    abstract_before: AbstractState
    """Abstract state before the attempted skill execution."""

    skill_instance: SkillInstance
    """Concrete skill that was (possibly) executed."""

    success: bool
    """Was the skill execution successful?"""

    abstract_after: AbstractState | None
    """Abstract state after the skill execution, if successful (else None)."""

    def __post_init__(self) -> None:
        """Verify that the constructed abstract transition is valid.

        :raises ValueError: If the transition was successful but no "after" state is defined
        """
        if self.success and self.abstract_after is None:
            raise ValueError("A successful abstract transition must include an 'after' state.")

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the abstract transition."""
        return self.skill_instance.skill.name


AbstractTrace = List[AbstractTransition]
"""A sequence of abstracted attempted skill executions."""


@dataclass(frozen=True)
class AbstractDataset:
    """An abstract dataset is a collection of abstracted skill execution traces."""

    abstract_traces: list[AbstractTrace]

    def get_skill_data(self, skill: Skill) -> list[AbstractTransition]:
        """Extract only the abstract transitions involving the given skill.

        :param skill: Skill executed in the extracted abstract transitions
        :return: List of abstract transitions involving the skill
        """
        return [
            transition
            for trace in self.abstract_traces
            for transition in trace
            if transition.skill_instance.skill == skill
        ]
