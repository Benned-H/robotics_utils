"""Define classes to represent abstract symbolic actions, both lifted and grounded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.classical_planning.abstract_states import AbstractState
from robotics_utils.classical_planning.parameters import Bindings, DiscreteParameter, ObjectT

if TYPE_CHECKING:
    from robotics_utils.classical_planning.predicates import Predicate, PredicateInstance


@dataclass(frozen=True)
class Operator:
    """A lifted abstract action defining an abstract transition model."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    preconditions: Preconditions  # Positive and negative preconditions for applying the operator
    effects: Effects  # Effects added and removed from the abstract state by the operator

    def ground_with(self, bindings: Bindings) -> OperatorInstance:
        """Ground the operator using the given parameter bindings."""
        return OperatorInstance(self, bindings)
