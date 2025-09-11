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


class OperatorInstance(Generic[ObjectT]):
    """An operator grounded by binding concrete objects to its parameters."""

    def __init__(self, operator: Operator, bindings: Bindings[ObjectT]) -> None:
        """Initialize the operator instance using an operator and parameter bindings."""
        self.operator = operator
        self.bindings = bindings

        # Ground the operator instance's preconditions and effects
        self.ground_preconditions = self.operator.preconditions.ground_with(bindings)
        self.ground_effects = self.operator.effects.ground_with(bindings)

    def __str__(self) -> str:
        """Return a readable string representation of the operator instance."""
        ordered_args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.operator.name}({ordered_args})"

    @property
    def arguments(self) -> tuple[ObjectT, ...]:
        """Retrieve the tuple of concrete objects used to ground the operator instance."""
        return tuple(self.bindings[p.name] for p in self.operator.parameters)

    def is_applicable(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the operator instance is applicable in an abstract state."""
        return self.ground_preconditions.satisfied_in(abstract_state)

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the operator instance to transition from the given abstract state."""
        if not self.is_applicable(abstract_state):
            raise ValueError(f"Cannot apply {self} in the abstract state: {abstract_state}")

        return self.ground_effects.apply(abstract_state)
