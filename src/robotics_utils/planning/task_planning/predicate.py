"""Define a class to represent symbolic predicates representing abstract relations."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING, Hashable

from robotics_utils.planning.task_planning.atom import Atom
from robotics_utils.planning.task_planning.ground_atom import GroundAtom
from robotics_utils.planning.task_planning.object_symbols import ObjectSymbols

if TYPE_CHECKING:
    from robotics_utils.planning.task_planning.parameter import Bindings, Parameter


class Predicate(Hashable):
    """A symbol representing an abstract relationship between objects."""

    def __init__(self, name: str, parameters: tuple[Parameter, ...]) -> None:
        """Initialize the predicate's internal member variables."""
        self.name = name
        self.parameters = parameters
        """Parameters specifying type constraints on expected arguments of the predicate."""

        self._param_to_type = {p.name: p.type_ for p in self.parameters}
        """A map from predicate parameter names to expected object types."""

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

    def get_parameter_type(self, param_name: str) -> str:
        """Retrieve the object type expected by the named parameter.

        :param param_name: Name of a predicate parameter
        :return: Expected object type for objects bound to the parameter
        :raises KeyError: If an unknown parameter name is given
        """
        if param_name not in self._param_to_type:
            raise KeyError(f"Cannot find type of unknown predicate parameter: '{param_name}'.")

        return self._param_to_type[param_name]

    def fully_bind(self, bindings: Bindings) -> GroundAtom:
        """Create a grounded predicate (i.e., ground atom) using the given parameter bindings.

        :param bindings: A map from parameter names to bound object symbols
        :return: Resulting GroundAtom instance
        """
        atom = Atom(predicate=self, bindings={})
        return atom.ground(**bindings)

    def compute_all_groundings(self, objects: ObjectSymbols) -> set[GroundAtom]:
        """Compute all valid groundings of the predicate using the given object symbols.

        :param objects: Collection of object symbols
        :return: Set of all valid groundings of the predicate, per its parameter type constraints
        """
        objects_per_param_type = (objects.get_objects_of_type(p.type_) for p in self.parameters)

        # Find all valid tuples of concrete arguments using a Cartesian product
        all_valid_args = product(*objects_per_param_type)
        all_bindings = (
            {p.name: obj for p, obj in zip(self.parameters, args, strict=True)}
            for args in all_valid_args
        )

        return {self.fully_bind(bindings=bindings) for bindings in all_bindings}
