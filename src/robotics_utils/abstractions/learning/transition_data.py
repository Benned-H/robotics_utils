"""Define classes to represent observed state transitions used for abstraction learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.abstractions.grounding.low_level_state import StateT

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.abstract_states import AbstractState
    from robotics_utils.skills import Skill, SkillInstance


@dataclass(frozen=True)
class SkillTransition(Generic[StateT]):
    """An observed transition resulting from an attempted skill execution."""

    pre: StateT
    """Low-level state from which the skill execution was attempted."""

    skill_instance: SkillInstance
    """Skill instance (possibly) executed to create the transition."""

    success: bool
    """Boolean success indicator of the skill execution."""

    post: StateT | None
    """Low-level state after the skill execution, or None if skill execution failed."""

    def __post_init__(self) -> None:
        """Verify that the constructed transition is valid."""
        if self.success and self.post is None:
            raise ValueError("A successful skill transition must include a post-state.")

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the transition."""
        return self.skill_instance.skill.name


SkillsTrace = list[SkillTransition[StateT]]
"""A sequence of attempted skill executions."""

Dataset = list[SkillsTrace[StateT]]
"""A collection of skill execution traces."""


@dataclass(frozen=True)
class AbstractTransition:
    """A transition representing the change in abstract state due to a skill execution."""

    pre: AbstractState
    """Abstract state before the attempted skill execution."""

    skill_instance: SkillInstance
    """Skill instance (possibly) executed to create the transition."""

    success: bool
    """Boolean success indicator of the skill execution."""

    post: AbstractState | None
    """Abstract state after the skill execution, or None if skill execution failed."""

    def __post_init__(self) -> None:
        """Verify that the constructed abstract transition is valid."""
        if self.success and self.post is None:
            raise ValueError("A successful abstract transition must include a post-state.")

    @property
    def skill_name(self) -> str:
        """Retrieve the name of the skill used in the abstract transition."""
        return self.skill_instance.skill.name


AbstractTrace = list[AbstractTransition]
"""A sequence of abstracted attempted skill executions."""


@dataclass(frozen=True)
class AbstractDataset:
    """An abstract dataset is a collection of abstracted skill execution traces."""

    abstract_traces: list[AbstractTrace]

    def get_data(self, skill: Skill) -> list[AbstractTransition]:
        """Extract all abstract transitions involving the given skill."""
        return [
            transition
            for trace in self.abstract_traces
            for transition in trace
            if transition.skill_instance.skill == skill
        ]  # TODO: Verify that Skill equality works as expected
