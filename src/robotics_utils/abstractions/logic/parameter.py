"""Define a class to represent symbolic parameters (e.g., of predicates or operators)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Parameter:
    """A symbolic parameter defining the expected type of a predicate argument."""

    name: str
    type_: str
    """Type of object expected by the parameter."""

    semantics: str | None = None
    """Optional natural language description of the parameter's meaning."""

    def __str__(self) -> str:
        """Create a readable string representation of the parameter."""
        semantics = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (type {self.type_}){semantics}"
