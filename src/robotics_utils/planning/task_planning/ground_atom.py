"""Define a class to represent ground atoms (AKA grounded predicates)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.planning.task_planning.object_symbols import ObjectSymbol
    from robotics_utils.planning.task_planning.predicate import Predicate


@dataclass(frozen=True)
class GroundAtom:
    """An atomic formula in which all variables have been bound to concrete arguments."""

    predicate: Predicate
    """The predicate that was grounded to create this ground atom."""

    arguments: tuple[ObjectSymbol, ...]
    """Objects bound to the predicate's parameters (order matches the parameters)."""

    def _key(self) -> tuple:
        """Define a hash key to uniquely identify the ground atom."""
        return (self.predicate._key(), self.arguments)  # noqa: SLF001

    def __eq__(self, other: object) -> bool:
        """Evaluate whether this ground atom is equal to another."""
        if not isinstance(other, GroundAtom):
            return NotImplemented

        return self._key() == other._key()

    def __hash__(self) -> int:
        """Compute a hash value for the ground atom."""
        return hash(self._key())

    def __str__(self) -> str:
        """Retrieve a readable string representation of the ground atom."""
        return f"{self.name}({', '.join(obj.name for obj in self.arguments)})"

    @property
    def name(self) -> str:
        """Retrieve the name of the grounded predicate."""
        return self.predicate.name
