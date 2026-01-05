"""Define a class to represent discrete, typed parameters."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_utils.abstractions.symbols.objects import ObjectSymbol


@dataclass(frozen=True)
class DiscreteParameter:
    """A discrete parameter specifying a type constraint.

    Equivalent to a typed variable in PDDL.
    """

    name: str
    """Name of the lifted parameter."""

    type_: str
    """Object type expected by the parameter."""

    semantics: str | None = None
    """Optional natural language description of the parameter's meaning."""

    def __str__(self) -> str:
        """Create a human-readable string representation of the parameter."""
        semantics = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (type {self.type_}){semantics}"

    @property
    def lifted_name(self) -> str:
        """Retrieve the parameter's name as a PDDL variable name."""
        return self.name if self.name.startswith("?") else f"?{self.name}"


Bindings = dict[str, ObjectSymbol]
"""A mapping from parameter names to (symbols representing) bound concrete objects."""
