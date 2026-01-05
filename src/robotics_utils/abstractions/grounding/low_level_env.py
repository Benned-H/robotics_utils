"""Define type variables to represent low-level environment states and actions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, TypeVar

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols import DiscreteParameter


class ParameterizedAction(Protocol):
    """A protocol for object-parameterized action instances."""

    @property
    def name(self) -> str:
        """Retrieve the name of the parameterized action."""
        ...

    @property
    def discrete_params(self) -> tuple[DiscreteParameter, ...]:
        """Retrieve the tuple of discrete parameters expected by the action's template."""
        ...

    @property
    def arguments(self) -> tuple[object, ...]:
        """Retrieve the tuple of concrete arguments used to instantiate the action."""
        ...


StateT = TypeVar("StateT")
"""Represents a low-level environment state."""

ActionT = TypeVar("ActionT", bound=ParameterizedAction)
"""Represents a low-level environment action or extended action."""
