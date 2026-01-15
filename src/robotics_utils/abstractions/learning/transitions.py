"""Define classes to represent observed state transitions used for abstraction learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.abstractions.grounding import ActionT, StateT

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols import AbstractState


@dataclass(frozen=True)
class Transition(Generic[StateT, ActionT]):
    """An observed state transition resulting from an attempted action."""

    before: StateT
    """Low-level state from which the action execution was attempted."""

    action: ActionT
    """Concrete parameterized action (possibly) executed to create the transition."""

    success: bool
    """Boolean success indicator of the action execution."""

    after: StateT | None
    """Low-level state after the action execution, or None if the action failed."""

    def __post_init__(self) -> None:
        """Verify that the constructed transition is valid."""
        if self.success and self.after is None:
            raise ValueError("A successful transition must include an 'after' state.")


Trace = list[Transition[StateT, ActionT]]
"""A sequence of attempted actions and resulting state transitions."""

Dataset = list[Trace[StateT, ActionT]]
"""A collection of action execution traces."""


@dataclass(frozen=True)
class AbstractTransition(Generic[ActionT]):
    """A transition between abstract states due to an action execution."""

    before: AbstractState
    """Abstract state before the attempted action execution."""

    action: ActionT
    """Concrete parameterized action (possibly) executed to create the transition."""

    success: bool
    """Boolean success indicator of the action execution."""

    after: AbstractState | None
    """Abstract state after the action execution, or None if the action failed."""

    def __post_init__(self) -> None:
        """Verify that the constructed abstract transition is valid."""
        if self.success and self.after is None:
            raise ValueError("A successful abstract transition must include an 'after' state.")


AbstractTrace = list[AbstractTransition[ActionT]]
"""A sequence of attempted actions and resulting abstract state transitions."""


@dataclass(frozen=True)
class AbstractDataset(Generic[ActionT]):
    """An abstract dataset is a collection of abstract traces."""

    traces: list[AbstractTrace[ActionT]]

    def get_data(self, action_name: str) -> list[AbstractTransition]:
        """Retrieve all abstract transitions involving the named action."""
        return [
            abs_t for trace in self.traces for abs_t in trace if abs_t.action.name == action_name
        ]
