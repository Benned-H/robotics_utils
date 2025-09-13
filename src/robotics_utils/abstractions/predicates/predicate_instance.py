"""Define a class to represent a predicate instance (i.e., a fully grounded predicate)."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Any, Generic

from robotics_utils.abstractions.predicates.dataclass_type import DataclassT
from robotics_utils.abstractions.predicates.low_level_state import StateT

if TYPE_CHECKING:
    from robotics_utils.abstractions.predicates.predicate import Predicate


@dataclass(frozen=True)
class PredicateInstance(Generic[DataclassT, StateT], Hashable):
    """A predicate grounded using particular concrete arguments."""

    predicate: Predicate[DataclassT, StateT]
    arguments: DataclassT

    def __key(self) -> tuple:
        """Define a hash key to uniquely identify the PredicateInstance."""
        return (self.predicate, self.arguments)

    def __hash__(self) -> int:
        """Compute a hash value for the predicate instance."""
        return hash(self.__key())

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        return f"{self.name}({', '.join(map(str, self.args_tuple))})"

    @property
    def name(self) -> str:
        """Retrieve the name of the instantiated predicate."""
        return self.predicate.name

    @property
    def args_tuple(self) -> tuple[Any, ...]:
        """Retrieve the arguments of the predicate instance as a tuple of values."""
        return astuple(self.arguments)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate instance."""
        return f"({self.name} {' '.join(map(str, self.args_tuple))})"

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.relation(self.arguments, state)
