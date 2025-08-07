"""Define classes to represent object-typed discrete parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

ObjectT = TypeVar("ObjectT")
"""Represents a concrete object in the environment."""


@dataclass(frozen=True)
class DiscreteParameter:
    """An object-typed discrete parameter (e.g., of a predicate or operator)."""

    name: str  # Name of the lifted parameter
    object_type: str  # Object type expected by the parameter
    semantics: str | None = None  # Optional NL description of the parameter's meaning

    def __str__(self) -> str:
        """Create a readable string representation of the discrete parameter."""
        semantics_str = f": {self.semantics}" if self.semantics else ""
        return f"{self.name} (Type {self.object_type}){semantics_str}"


Bindings = dict[str, ObjectT]
"""A mapping from parameter names to their bound concrete objects."""
