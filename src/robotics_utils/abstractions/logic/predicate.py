"""Define a class to represent a truth-valued predicate in first-order logic."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_utils.abstractions.logic.parameter import Parameter


@dataclass(frozen=True)
class Predicate:
    """A symbolic predicate with typed parameters."""

    name: str
    parameters: tuple[Parameter, ...]
    semantics: str | None = None
    """Optional natural language description of the predicate's meaning."""

    def __key(self) -> tuple:
        """Define a hash key to uniquely identify the predicate."""
        return (self.name, self.parameters, self.semantics)

    def __hash__(self) -> int:
        """Define a hash key for the predicate."""
        return hash(self.__key())
