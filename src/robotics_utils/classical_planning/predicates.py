"""Define classes to represent lifted and grounded symbolic predicates."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Generic

from robotics_utils.classical_planning.parameters import Bindings, DiscreteParameter, ObjectT


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
    def arguments(self) -> tuple[ObjectT, ...]:
        """Retrieve the tuple of concrete objects used to ground the predicate instance."""
        return tuple(self.bindings[p.name] for p in self.predicate.parameters)

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate instance."""
        args_string = " ".join(str(arg) for arg in self.arguments)
        return f"({self.predicate.name} {args_string})"
