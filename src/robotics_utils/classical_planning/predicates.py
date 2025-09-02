"""Define classes to represent lifted and (partially) grounded symbolic predicates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from typing import TYPE_CHECKING, Generic, Mapping

from robotics_utils.classical_planning.abstract_states import StateT
from robotics_utils.objects import ObjectT, ObjectTypes

if TYPE_CHECKING:
    from robotics_utils.classical_planning.parameters import DiscreteParameter

# TODO: Delete TypeHierarchy


@dataclass(frozen=True)
class Predicate(ABC, Generic[StateT]):
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None = None  # Optional NL description of the predicate's meaning

    @abstractmethod
    def holds_in(self, state: StateT, bindings: Mapping[str, ObjectT]) -> bool:
        """Evaluate whether the predicate holds in a state under the given parameter bindings.

        :param state: Low-level environment state in which the predicate is evaluated
        :param bindings: Parameter bindings used to ground the predicate
        :return: True if the grounded predicate holds, else False
        """

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        group_by_type: dict[str, list[str]] = defaultdict(list)  # Map type names to parameters
        for p in self.parameters:
            group_by_type[p.object_type].append(p.name)

        groups = [f"{' '.join(params)} - {t_name}" for t_name, params in group_by_type.items()]
        return f"({self.name}{' ' + ' '.join(groups) if groups else ''})"

    def fully_ground(self, bindings: Mapping[str, ObjectT]) -> PredicateInstance:
        """Create a fully grounded predicate instance using the given parameter bindings."""
        return PredicateInstance(self, dict(bindings))

    def as_atom(self, bindings: Mapping[str, ObjectT] | None = None) -> Atom[ObjectT]:
        """Create a (possibly partial) atom using the given parameter bindings.

        :param bindings: Optional parameter bindings (defaults to None)
        :return: Constructed Atom instance
        """
        return Atom(self, dict(bindings) if bindings else {})

    def compute_all_groundings(self, object_types: ObjectTypes) -> set[PredicateInstance]:
        """Compute all valid groundings of the predicate using the given objects.

        :param object_types: Describes the types of objects in the environment
        :return: Set of all valid groundings of the predicate
        """
        parameter_types = (p.object_type for p in self.parameters)
        objs_per_param_type = (object_types.get_objects_of_type(p_t) for p_t in parameter_types)

        # Find all valid tuples of concrete args using a Cartesian product
        all_valid_groundings = product(*objs_per_param_type)
        all_bindings = (
            {p.name: obj}
            for grounding in all_valid_groundings
            for p, obj in zip(self.parameters, grounding, strict=True)
        )

        return {PredicateInstance(self, bindings) for bindings in all_bindings}


@dataclass(frozen=True)
class PredicateInstance(Generic[StateT, ObjectT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate
    bindings: dict[str, ObjectT]

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        return f"{self.predicate.name}({', '.join(map(str, self.arguments))})"

    @property
    def name(self) -> str:
        """Retrieve the name of the instantiated predicate."""
        return self.predicate.name

    @property
    def arguments(self) -> tuple[ObjectT, ...]:
        """Retrieve the tuple of concrete objects used to ground the predicate instance."""
        return tuple(self.bindings[p.name] for p in self.predicate.parameters)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate instance."""
        return f"({self.predicate.name} {', '.join(map(str, self.arguments))})"

    def holds_in(self, state: StateT) -> bool:
        """Evaluate whether the predicate instance holds in the given state."""
        return self.predicate.holds_in(state, self.bindings)


@dataclass(frozen=True)
class Atom(Generic[ObjectT]):
    """An atom (i.e., atomic formula) formed by a predicate symbol with partial bindings.

    Reference: Chapter 8.2.4 ("Atomic sentences"), pg. 260 of AIMA (4th Ed.) by Russell and Norvig.
    """

    predicate: Predicate
    bindings: dict[str, ObjectT]
    """Maps bound parameter names to the corresponding concrete objects."""

    @property
    def name(self) -> str:
        """Retrieve the name of the atomic formula's predicate."""
        return self.predicate.name

    @property
    def is_grounded(self) -> bool:
        """Check whether all parameters of the atom are bound to concrete objects."""
        return len(self.bindings) == len(self.predicate.parameters)

    @property
    def unbound_params(self) -> tuple[DiscreteParameter, ...]:
        """Retrieve all unbound parameters of the atom."""
        bound_param_names = set(self.bindings.keys())
        return tuple(p for p in self.predicate.parameters if p.name not in bound_param_names)

    def ordered_arguments(self, allow_missing: bool = False) -> tuple[ObjectT | None, ...]:
        """Retrieve the atom's arguments, aligned to its predicate parameter order.

        :param allow_missing: Whether to allow unbound parameters (defaults to False)
        :return: Tuple of ordered arguments, with None in place of unbound parameters
        :raises ValueError: If allow_missing=False and any parameter remains unbound
        """
        if not allow_missing and not self.is_grounded:
            unbound = ", ".join(p.name for p in self.unbound_params)
            raise ValueError(f"{self.name} not fully grounded; missing parameters: {unbound}")

        return tuple(self.bindings.get(p.name) for p in self.predicate.parameters)

    def bind(self, **kwargs: ObjectT) -> Atom[ObjectT]:
        """Return a new atom with updated bindings."""
        valid_param_names = {p.name for p in self.predicate.parameters}

        for param in kwargs:
            if param not in valid_param_names:
                raise KeyError(f"Unknown parameter '{param}' for predicate {self.predicate.name}.")

        new_bindings = dict(self.bindings)
        new_bindings.update(kwargs)

        return replace(self, bindings=new_bindings)

    def unbind(self, *param_names: str) -> Atom[ObjectT]:
        """Return a new atom with some parameters made unbound."""
        new_bindings = {p: v for p, v in self.bindings.items() if p not in param_names}
        return replace(self, bindings=new_bindings)

    def as_instance(self) -> PredicateInstance[ObjectT]:
        """Convert the atomic sentence into a fully grounded predicate instance."""
        if not self.is_grounded:
            unbound = ", ".join(p.name for p in self.unbound_params)
            raise ValueError(f"{self.name} not fully grounded; missing parameters: {unbound}")

        return PredicateInstance(self.predicate, self.bindings)
