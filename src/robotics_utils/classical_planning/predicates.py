"""Define classes to represent lifted and grounded symbolic predicates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Generic, Mapping

from robotics_utils.classical_planning.parameters import (
    UNBOUND,
    Bindings,
    DiscreteParameter,
    ObjectT,
    PartialBindings,
    _UnboundType,
    is_unbound,
)


@dataclass(frozen=True)
class Predicate:
    """A symbolic predicate with object-typed parameters."""

    name: str
    parameters: tuple[DiscreteParameter, ...]
    semantics: str | None = None  # Optional NL description of the predicate's meaning

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.object_type}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        types_to_params: dict[str, list[str]] = defaultdict(list)  # Map type names to parameters

        for param in self.parameters:
            types_to_params[param.object_type].append(param.name)

        type_groups: list[str] = []
        for type_name, relevant_params in types_to_params.items():
            pddl_params = " ".join(relevant_params)
            type_groups.append(f"{pddl_params} - {type_name}")

        params_string = (" " + " ".join(type_groups)) if type_groups else ""
        return f"({self.name}{params_string})"

    def ground_with(self, bindings: Bindings) -> PredicateInstance:
        """Ground the predicate using the given parameter bindings."""
        return PredicateInstance(self, bindings)

    def partially_ground(self, bindings: Bindings[ObjectT] | None = None) -> PartialAtom[ObjectT]:
        """Construct an atom (i.e., partially grounded predicate) using the given bindings.

        :param bindings: Optional initial parameter bindings (defaults to None)
        :return: Constructed PartialAtom instance
        """
        return PartialAtom.from_predicate(self, bindings)


@dataclass(frozen=True)
class PredicateInstance(Generic[ObjectT]):
    """A predicate grounded using particular concrete objects."""

    predicate: Predicate
    bindings: Bindings[ObjectT]

    def __str__(self) -> str:
        """Return a readable string representation of the predicate instance."""
        args_string = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.predicate.name}({args_string})"

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
        args_string = " ".join(str(arg) for arg in self.arguments)
        return f"({self.predicate.name} {args_string})"


@dataclass(frozen=True)
class PartialAtom(Generic[ObjectT]):
    """An atom (i.e., atomic formula) is formed by a partially grounded predicate symbol.

    Reference: Chapter 8.2.4 ("Atomic sentences"), pg. 260 of AIMA (4th Ed.) by Russell and Norvig.
    """

    predicate: Predicate
    bindings: PartialBindings[ObjectT]
    """Maps parameter names to bound concrete objects, or the UNBOUND value."""

    @classmethod
    def from_predicate(
        cls,
        predicate: Predicate,
        initial_bindings: Mapping[str, ObjectT] | None = None,
    ) -> PartialAtom[ObjectT]:
        """Construct an atom (i.e., partially grounded predicate) from a lifted predicate schema.

        :param predicate: Lifted predicate schema
        :param initial_bindings: Optional initial bindings (defaults to None)
        :return: Constructed PartialAtom instance
        """
        partial_bindings: PartialBindings[ObjectT] = {p.name: UNBOUND for p in predicate.parameters}

        if initial_bindings:
            for p_name, bound_obj in initial_bindings.items():
                if p_name not in partial_bindings:
                    raise KeyError(f"Unknown parameter '{p_name}' for predicate {predicate.name}.")
                partial_bindings[p_name] = bound_obj

        return PartialAtom(predicate, partial_bindings)

    @property
    def name(self) -> str:
        """Retrieve the name of the partially grounded predicate."""
        return self.predicate.name

    @property
    def arguments(self) -> tuple[ObjectT | _UnboundType, ...]:
        """Retrieve the tuple of parameter-aligned arguments of the atom."""
        return tuple(self.bindings[p.name] for p in self.predicate.parameters)

    @property
    def fully_grounded(self) -> bool:
        """Check whether all parameters of the atom are bound to concrete objects."""
        return all(not is_unbound(arg) for arg in self.bindings.values())

    @property
    def unbound_params(self) -> tuple[DiscreteParameter, ...]:
        """Retrieve all unbound parameters of the atom."""
        return tuple(p for p in self.predicate.parameters if is_unbound(self.bindings[p.name]))

    def bind(self, **kwargs: ObjectT) -> PartialAtom[ObjectT]:
        """Create a new atomic sentence using the given parameter-object bindings."""
        new_bindings = dict(self.bindings)

        for p_name, bound_obj in kwargs.items():
            if p_name not in new_bindings:
                raise KeyError(f"Unknown parameter '{p_name}' for predicate {self.predicate.name}.")

            new_bindings[p_name] = bound_obj

        return replace(self, bindings=new_bindings)

    def as_instance(self) -> PredicateInstance[ObjectT]:
        """Convert the atomic sentence into a fully grounded predicate instance."""
        if not self.fully_grounded:
            raise ValueError(
                f"Predicate {self.predicate.name} is not fully grounded; "
                f"missing parameters {self.unbound_params}.",
            )
        return PredicateInstance(self.predicate, self.bindings)
