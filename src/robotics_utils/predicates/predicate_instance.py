"""Define a class to represent a predicate instance (i.e., a fully grounded predicate)."""

from __future__ import annotations

from dataclasses import astuple, dataclass
from typing import TYPE_CHECKING, Any, Generic

from robotics_utils.predicates.dataclass_type import DataclassT
from robotics_utils.predicates.low_level_state import StateT

if TYPE_CHECKING:
    from robotics_utils.predicates.predicate import Predicate


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT, DataclassT]):
    """A predicate grounded using particular concrete arguments."""

    predicate: Predicate[StateT, DataclassT]
    arguments: DataclassT

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
        return f"({self.predicate.name} {' '.join(map(str, self.args_tuple))})"

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.arguments)
