"""Define a class to represent symbolic predicates representing abstract relations."""

from __future__ import annotations

from collections import defaultdict
from itertools import product
from typing import TYPE_CHECKING, Hashable

from robotics_utils.abstractions.symbols.atom import Atom

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.discrete_parameter import Bindings, DiscreteParameter
    from robotics_utils.abstractions.symbols.ground_atom import GroundAtom
    from robotics_utils.abstractions.symbols.objects import ObjectSymbols


class Predicate(Hashable):
    """A symbol representing an abstract relationship between objects."""

    def __init__(self, name: str, parameters: tuple[DiscreteParameter, ...]) -> None:
        """Initialize the predicate's internal member variables."""
        self.name = name
        self.parameters = parameters
        """Parameters specifying type constraints on expected arguments of the predicate."""

        self._param_to_type = {p.name: p.type_ for p in self.parameters}
        """A mapping from predicate parameter names to expected object types."""

    def _key(self) -> tuple:
        """Define a hash key to uniquely identify the predicate."""
        return (self.name, self.parameters)

    def __eq__(self, other: object) -> bool:
        """Evaluate whether this predicate and another are equal."""
        if not isinstance(other, Predicate):
            return NotImplemented

        return self._key() == other._key()

    def __hash__(self) -> int:
        """Compute a hash value for the predicate."""
        return hash(self._key())

    def __str__(self) -> str:
        """Return a human-readable string representation of the predicate."""
        params = ", ".join(f"{p.name}: {p.type_}" for p in self.parameters)
        return f"{self.name}({params})"

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the predicate."""
        params_per_type: dict[str, list[str]] = defaultdict(list)
        for p in self.parameters:
            params_per_type[p.type_].append(p.lifted_name)

        typed_variables = [f"{' '.join(params)} - {t}" for t, params in params_per_type.items()]
        variables_str = (" " + " ".join(typed_variables)) if typed_variables else ""
        return f"({self.name}{variables_str})"

    def get_parameter_type(self, param_name: str) -> str:
        """Retrieve the type expected by the named parameter.

        :param param_name: Name of a predicate parameter
        :return: Expected object type for objects bound to the parameter
        :raises KeyError: If an unknown parameter name is given
        """
        if param_name not in self._param_to_type:
            raise KeyError(f"Cannot find type of unknown predicate parameter: '{param_name}'.")
        return self._param_to_type[param_name]

    def fully_bind(self, bindings: Bindings) -> GroundAtom:
        """Create a grounded predicate (i.e., ground atom) using the given parameter bindings.

        :param bindings: Mapping from parameter names to bound object symbols
        :return: Resulting GroundAtom instance
        """
        atom = Atom(predicate=self, bindings={})
        return atom.ground(**bindings)

    def compute_all_groundings(self, objects: ObjectSymbols) -> set[GroundAtom]:
        """Compute all valid groundings of the predicate using the given objects.

        :param objects: Collection of object symbols
        :return: Set of all valid groundings of the predicate, per its parameter type constraints
        """
        objects_per_param_type = (objects.get_objects_of_type(p.type_) for p in self.parameters)

        # Find all valid tuples of concrete args using a Cartesian product
        all_valid_args = product(*objects_per_param_type)
        all_bindings = (
            {p.name: obj for p, obj in zip(self.parameters, args, strict=True)}
            for args in all_valid_args
        )

        return {self.fully_bind(bindings) for bindings in all_bindings}
