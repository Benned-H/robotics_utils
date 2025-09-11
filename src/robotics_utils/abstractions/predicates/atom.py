"""Define a class to represent a predicate symbol with partial bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic

from robotics_utils.abstractions.predicates.dataclass_type import DataclassT
from robotics_utils.abstractions.predicates.low_level_state import StateT
from robotics_utils.abstractions.predicates.predicate_instance import PredicateInstance

if TYPE_CHECKING:
    from robotics_utils.abstractions.predicates.parameter import Parameter
    from robotics_utils.abstractions.predicates.predicate import Predicate


@dataclass(frozen=True)
class Atom(Generic[DataclassT, StateT]):
    """An atom (i.e., atomic formula) formed by a predicate symbol with partial bindings.

    Reference: Chapter 8.2.4 ("Atomic sentences"), pg. 260 of AIMA (4th Ed.) by Russell and Norvig.
    """

    predicate: Predicate[DataclassT, StateT]
    bindings: dict[str, object]
    """Maps bound parameter names to corresponding concrete arguments."""

    @property
    def name(self) -> str:
        """Retrieve the name of the atomic formula's predicate."""
        return self.predicate.name

    @property
    def is_grounded(self) -> bool:
        """Check whether all parameters of the atom are bound to concrete arguments."""
        return len(self.bindings) == len(self.predicate.parameters)

    @property
    def unbound_params(self) -> tuple[Parameter, ...]:
        """Retrieve a tuple of the unbound parameters of the atom."""
        bound_param_names = set(self.bindings.keys())
        return tuple(p for p in self.predicate.parameters if p.name not in bound_param_names)

    def ordered_arguments(self, allow_missing: bool = False) -> tuple[object | None, ...]:
        """Retrieve the atom's arguments, aligned with its predicate parameter order.

        :param allow_missing: Whether to allow unbound parameters (defaults to False)
        :return: Tuple of ordered arguments, with None in place of unbound parameters
        :raises ValueError: If allow_missing is False and any parameter remains unbound
        """
        if not allow_missing and not self.is_grounded:
            unbound = ", ".join(p.name for p in self.unbound_params)
            raise ValueError(f"{self.name} is not fully grounded; missing parameters: {unbound}.")

        return tuple(self.bindings.get(p.name) for p in self.predicate.parameters)

    def bind(self, **kwargs: object) -> Atom[DataclassT, StateT]:
        """Return a new atom with updated bindings."""
        for param_name, bound_value in kwargs.items():
            param_type = self.predicate.get_parameter_type(param_name)  # Raises on unknown param

            if not isinstance(bound_value, param_type):
                raise TypeError(f"Cannot bind value '{bound_value}' to type '{param_type}'.")

        new_bindings = dict(self.bindings)
        new_bindings.update(kwargs)

        return replace(self, bindings=new_bindings)

    def unbind(self, *param_names: str) -> Atom[DataclassT, StateT]:
        """Return a new atom with some parameters made unbound."""
        new_bindings = {p: v for p, v in self.bindings.items() if p not in param_names}
        return replace(self, bindings=new_bindings)

    def as_instance(self) -> PredicateInstance[DataclassT, StateT]:
        """Convert the atomic formula into a fully grounded predicate instance."""
        if not self.is_grounded:
            unbound = ", ".join(p.name for p in self.unbound_params)
            raise ValueError(f"{self.name} is not fully grounded; missing parameters: {unbound}.")

        args_dataclass = self.predicate.construct_arguments(self.bindings)
        return PredicateInstance(self.predicate, args_dataclass)
