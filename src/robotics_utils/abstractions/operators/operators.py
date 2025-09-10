"""Define classes to represent abstract symbolic actions, both lifted and grounded."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Generic

from robotics_utils.classical_planning.abstract_states import AbstractState
from robotics_utils.classical_planning.parameters import Bindings, DiscreteParameter, ObjectT

if TYPE_CHECKING:
    from robotics_utils.classical_planning.predicates import Predicate, PredicateInstance


@dataclass(frozen=True)
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    positive: set[Predicate]  # Abstract conditions that must hold true to apply an operator
    negative: set[Predicate]  # Abstract conditions that must be false to apply an operator

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the preconditions."""
        positive_pre = "\n\t".join(sorted(p.to_pddl() for p in self.positive))
        negative_pre = "\n\t".join(sorted(f"(not {p.to_pddl()})" for p in self.negative))

        return f":precondition (and\n\t{positive_pre}\n\t{negative_pre}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedPreconditions:
        """Ground the preconditions using the given parameter bindings."""
        return GroundedPreconditions(
            positive={p.ground_with(bindings) for p in self.positive},
            negative={p.ground_with(bindings) for p in self.negative},
        )


@dataclass(frozen=True)
class GroundedPreconditions:
    """A collection of grounded preconditions to applying an operator."""

    positive: set[PredicateInstance]  # Grounded predicates that must hold to apply an operator
    negative: set[PredicateInstance]  # Grounded predicates that must be false to apply an operator

    def satisfied_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the grounded preconditions are satisfied in an abstract state."""
        if any((pos_pre not in abstract_state) for pos_pre in self.positive):
            return False
        return all((neg_pre not in abstract_state) for neg_pre in self.negative)


@dataclass(frozen=True)
class Effects:
    """A collection of predicates defining add and delete effects of an operator."""

    add: set[Predicate]  # Predicates added to the abstract state by the operator
    delete: set[Predicate]  # Predicates removed from the abstract state by the operator

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the effects."""
        add_eff = "\n\t".join(sorted(p.to_pddl() for p in self.add))
        delete_eff = "\n\t".join(sorted(f"(not {p.to_pddl()})" for p in self.delete))

        return f":effect (and\n\t{add_eff}\n\t{delete_eff}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedEffects:
        """Ground the effects using the given parameter bindings."""
        return GroundedEffects(
            add={p.ground_with(bindings) for p in self.add},
            delete={p.ground_with(bindings) for p in self.delete},
        )


@dataclass(frozen=True)
class GroundedEffects:
    """A collection of grounded effects resulting from applying an operator."""

    add: set[PredicateInstance]  # Grounded predicates added to the abstract state
    delete: set[PredicateInstance]  # Grounded predicates removed from the abstract state

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded effects to the given abstract state."""
        return AbstractState(abstract_state.facts.difference(self.delete).union(self.add))


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
