"""Define classes to represent planning operators and their constituent constructs."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

from robotics_utils.abstractions.symbols.abstract_states import AbstractState

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.discrete_parameter import Bindings, DiscreteParameter
    from robotics_utils.abstractions.symbols.ground_atom import GroundAtom
    from robotics_utils.abstractions.symbols.objects import Objects
    from robotics_utils.abstractions.symbols.predicate import Predicate


@dataclass(frozen=True)
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    positive: frozenset[Predicate] = frozenset()
    """Abstract conditions that must hold to apply the relevant operator."""

    negative: frozenset[Predicate] = frozenset()
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

    add: frozenset[Predicate] = frozenset()
    """Predicates whose groundings are added to the abstract state by the operator."""

    delete: frozenset[Predicate] = frozenset()
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

    def compute_all_groundings(self, objects: Objects) -> set[GroundOperator]:
        """Compute all valid groundings of the operator using the given objects.

        :param objects: Collection of object symbols
        :return: Set of all valid groundings of the operator, per its parameter type constraints
        """
        objects_per_param_type = (objects.get_objects_of_type(p.type_) for p in self.parameters)

        # Find all valid tuples of concrete args using a Cartesian product
        all_valid_args = product(*objects_per_param_type)
        all_bindings = (
            {p.name: obj}
            for args in all_valid_args
            for p, obj in zip(self.parameters, args, strict=True)
        )

        return {self.ground_with(bindings) for bindings in all_bindings}


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

    def apply(self, abstract_state: AbstractState) -> AbstractState | None:
        """Apply the ground operator to transition from the given abstract state.

        :param abstract_state: Current abstract state
        :return: Resulting abstract state, or None if the operator is not applicable
        """
        if not self.applicable_in(abstract_state):
            return None

        return self.ground_eff.apply(abstract_state)

    ### Implement properties to satisfy the `ParameterizedAction` protocol ###
    @property
    def name(self) -> str:
        """Retrieve the name of the operator that has been grounded."""
        return self.operator.name

    @property
    def discrete_params(self) -> tuple[DiscreteParameter, ...]:
        """Retrieve the tuple of discrete parameters expected by the operator."""
        return self.operator.parameters

    @property
    def arguments(self) -> tuple[object, ...]:
        """Retrieve the tuple of object symbols used to ground the operator."""
        return tuple(self.bindings[p.name] for p in self.discrete_params)


AbstractAction = GroundOperator
