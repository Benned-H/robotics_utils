"""Define a class to represent lifted symbolic predicates."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from itertools import product
from typing import Any, Generic, Mapping

from robotics_utils.predicates.atom import Atom
from robotics_utils.predicates.dataclass_type import DataclassT, DataclassType
from robotics_utils.predicates.low_level_state import StateT
from robotics_utils.predicates.parameter import Parameter
from robotics_utils.predicates.predicate_instance import PredicateInstance


@dataclass(frozen=True)
class Predicate(ABC, Generic[StateT, DataclassT]):
    """A symbolic predicate with typed parameters."""

    name: str
    parameters: tuple[Parameter, ...]
    dataclass_t: DataclassType[DataclassT]
    """Dataclass type defining the structure of the predicate's parameters."""

    semantics: str | None = None
    """Optional natural language description of the predicate's meaning."""

    @abstractmethod
    def holds_in(self, state: StateT, args: DataclassT) -> bool:
        """Evaluate whether the predicate holds in a state for the given arguments.

        :param state: Low-level environment state in which the predicate is evaluated
        :param args: Dataclass storing arguments bound to the predicate's parameters
        :return: True if the grounded predicate holds, else False
        """

    @classmethod
    def from_dataclass(cls, name: str, dataclass_t: type[DataclassT]) -> Predicate:
        """Construct a Predicate instance based on the given dataclass type.

        :param name: Name of the constructed predicate
        :param dataclass_t: Type of a dataclass
        :return: Constructed Predicate instance
        """
        parameters = Parameter.tuple_from_dataclass(dataclass_t)
        return cls(name, parameters, DataclassType(dataclass_t))

    def __str__(self) -> str:
        """Return a readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.type_}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        params_per_type: dict[type, list[str]] = defaultdict(list)
        for p in self.parameters:
            lifted_param_name = p.name if p.name.startswith("?") else f"?{p.name}"
            params_per_type[p.type_].append(lifted_param_name)  # PDDL parameter names begin with ?

        groups = [f"{' '.join(params)} - {t_name}" for t_name, params in params_per_type.items()]
        return f"({self.name}{' ' + ' '.join(groups) if groups else ''})"

    def get_parameter_type(self, param_name: str) -> type:
        """Retrieve the expected type of a predicate parameter.

        :return: Python type expected by the named parameter
        :raises KeyError: If an unknown parameter name is given
        """
        if param_name not in self.dataclass_t.field_types:
            raise KeyError(f"Cannot find type of unknown predicate parameter: '{param_name}'.")
        return self.dataclass_t.field_types[param_name]

    def ground(self, args: DataclassT) -> PredicateInstance:
        """Ground the predicate using the given dataclass of arguments."""
        return PredicateInstance(self, args)

    def fully_ground(self, bindings: Mapping[str, Any]) -> PredicateInstance:
        """Create a fully grounded predicate instance using the given parameter bindings."""
        return self.as_atom(bindings).as_instance()

    def as_atom(self, bindings: Mapping[str, Any] | None = None) -> Atom[StateT, DataclassT]:
        """Create an atomic formula using the given (possibly partial) parameter bindings.

        :param bindings: Optional parameter bindings (defaults to None)
        :return: Constructed Atom instance
        """
        return Atom(self, dict(bindings) if bindings else {})

    def compute_all_groundings(self, objects: set[object]) -> set[PredicateInstance]:
        """Compute all valid groundings of the predicate using the given Python objects.

        :param objects: Collection of Python object instances
        :return: Set of all valid groundings of the predicate
        """
        objects_per_param_type = (
            {obj for obj in objects if isinstance(obj, param.type_)} for param in self.parameters
        )

        # Find all valid tuples of concrete args using a Cartesian product
        all_valid_groundings = product(*objects_per_param_type)
        all_bindings = (
            {p.name: obj}
            for grounding in all_valid_groundings
            for p, obj in zip(self.parameters, grounding, strict=True)
        )

        return {self.fully_ground(bindings) for bindings in all_bindings}
