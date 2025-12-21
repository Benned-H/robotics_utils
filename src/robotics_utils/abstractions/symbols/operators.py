"""Define classes to represent planning operators and their constituent constructs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from robotics_utils.abstractions.symbols.abstract_states import AbstractState

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.discrete_parameter import Bindings, DiscreteParameter
    from robotics_utils.abstractions.symbols.ground_atom import GroundAtom
    from robotics_utils.abstractions.symbols.predicate import Predicate


@dataclass(frozen=True)
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    positive: frozenset[Predicate]
    """Abstract conditions that must hold to apply the relevant operator."""

    negative: frozenset[Predicate]
    """Abstract conditions that must be false to apply the relevant operator."""

    def ground_with(self, bindings: Bindings) -> GroundPreconditions:
        """Ground the preconditions using the given parameter bindings."""
        return GroundPreconditions(
            positive=frozenset(p.fully_bind(bindings) for p in self.positive),
            negative=frozenset(p.fully_bind(bindings) for p in self.negative),
        )


@dataclass(frozen=True)
class GroundPreconditions:
    """A collection of grounded preconditions for applying an operator."""

    positive: frozenset[GroundAtom]
    """Grounded conditions that must hold to apply the relevant operator."""

    negative: frozenset[GroundAtom]
    """Grounded conditions that must be false to apply the relevant operator."""

    def satisfied_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the ground preconditions are satisfied in an abstract state."""
        return all((pos_pre in abstract_state) for pos_pre in self.positive) and all(
            (neg_pre not in abstract_state) for neg_pre in self.negative
        )


@dataclass(frozen=True)
class Effects:
    """A collection of predicates defining add and delete effects of an operator."""

    add: frozenset[Predicate]
    """Predicates whose groundings are added to the abstract state by the operator."""

    delete: frozenset[Predicate]
    """Predicates whose groundings are deleted from the abstract state by the operator."""

    def ground_with(self, bindings: Bindings) -> GroundEffects:
        """Ground the effects using the given parameter bindings."""
        return GroundEffects(
            add=frozenset(p.fully_bind(bindings) for p in self.add),
            delete=frozenset(p.fully_bind(bindings) for p in self.delete),
        )


@dataclass(frozen=True)
class GroundEffects:
    """A collection of grounded effects resulting from applying an operator."""

    add: frozenset[GroundAtom]
    """Grounded predicates added to the abstract state by the operator."""

    delete: frozenset[GroundAtom]
    """Grounded predicates deleted from the abstract state by the operator."""

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded effects to the given abstract state."""
        return AbstractState(abstract_state.facts.difference(self.delete).union(self.add))


@dataclass(frozen=True)
class Operator:
    """A lifted abstract action defining a symbolic transition model.

    Equivalent to a STRIPS-style (`:strips`) PDDL action definition.
    """

    name: str
    parameters: tuple[DiscreteParameter, ...]
    preconditions: Preconditions
    """Positive and negative preconditions for applying the operator."""

    effects: Effects
    """Effects added and removed from the abstract state by the operator."""

    def ground_with(self, bindings: Bindings) -> GroundOperator:
        """Ground the operator using the given parameter bindings."""
        return GroundOperator(self, bindings)


class GroundOperator:
    """A grounded abstract action applied to specific concrete objects."""

    def __init__(self, operator: Operator, bindings: Bindings) -> None:
        """Initialize the ground operator using an operator and parameter bindings."""
        self.operator = operator
        self.bindings = bindings

        self.ground_pre = self.operator.preconditions.ground_with(bindings)
        self.ground_eff = self.operator.effects.ground_with(bindings)

    def applicable_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the ground operator is applicable in an abstract state."""
        return self.ground_pre.satisfied_in(abstract_state)

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the ground operator to transition from the given abstract state."""
        if not self.applicable_in(abstract_state):
            raise ValueError(f"Cannot apply {self} in the abstract state: {abstract_state}")

        return self.ground_eff.apply(abstract_state)


AbstractAction = GroundOperator
