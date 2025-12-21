"""Define a class to represent a predicate symbol with partial bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from robotics_utils.abstractions.symbols.ground_atom import GroundAtom

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.discrete_parameter import DiscreteParameter
    from robotics_utils.abstractions.symbols.objects import ObjectSymbol
    from robotics_utils.abstractions.symbols.predicate import Predicate


@dataclass(frozen=True)
class Atom:
    """An atom (i.e., atomic formula) formed by a predicate symbol with partial bindings.

    Reference: Chapter 8.2.4 ("Atomic sentences"), pg. 260 of AIMA (4th Ed.) by Russell and Norvig.
    """

    predicate: Predicate
    bindings: dict[str, ObjectSymbol]
    """A mapping from bound parameter names to corresponding object arguments."""

    @property
    def name(self) -> str:
        """Retrieve the name of the atomic formula's predicate."""
        return self.predicate.name

    @property
    def is_grounded(self) -> bool:
        """Check whether all parameters of the atom are bound to objects."""
        return not self.unbound_params

    @property
    def unbound_params(self) -> tuple[DiscreteParameter, ...]:
        """Retrieve a tuple of the unbound parameters of the atom."""
        return tuple(p for p in self.predicate.parameters if p.name not in self.bindings)

    def bind(self, **kwargs: ObjectSymbol) -> Atom:
        """Return a new atom with updated bindings."""
        for param_name, obj in kwargs.items():
            param_type = self.predicate.get_parameter_type(param_name)  # KeyError on unknown param

            if obj.type_ != param_type:
                raise ValueError(f"Cannot bind object {obj} to parameter of type '{param_type}'.")

        new_bindings = dict(self.bindings)
        new_bindings.update(kwargs)

        return replace(self, bindings=new_bindings)

    def ground(self, **kwargs: ObjectSymbol) -> GroundAtom:
        """Fully ground the atomic formula using the given bindings, if any.

        :param **kwargs: Bindings assigning object symbols to parameter names (optional)
        :return: Ground atom with all its parameters bound
        """
        updated = self.bind(**kwargs)
        if not updated.is_grounded:
            ubp = ", ".join(p.name for p in updated.unbound_params)
            raise ValueError(f"{updated.name} is not fully grounded; unbound parameters: {ubp}")

        arguments = tuple(self.bindings[p.name] for p in self.predicate.parameters)
        return GroundAtom(self.predicate, arguments)
