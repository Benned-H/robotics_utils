"""Define a class to represent a fully grounded predicate."""

from __future__ import annotations

from dataclasses import dataclass

from robotics_utils.abstractions.logic.object_symbol import ObjectSymbol
from robotics_utils.abstractions.logic.predicate import Predicate


@dataclass(frozen=True)
class GroundAtom:
    """A predicate with all parameters bound to concrete arguments."""

    predicate: Predicate
    arguments: tuple[ObjectSymbol, ...]
    """Symbols bound to the parameters of the predicate."""

    def __key(self) -> tuple:
        """Define a hash key to uniquely identify the ground atom."""
        return (self.predicate, self.arguments)

    def __hash__(self) -> int:
        """Define a hash key for the ground atom."""
        return hash(self.__key())
