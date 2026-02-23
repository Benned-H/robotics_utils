"""Define a class to represent predicate symbols with partial bindings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from robotics_utils.planning.task_planning.ground_atom import GroundAtom

if TYPE_CHECKING:
    from robotics_utils.planning.task_planning.object_symbols import ObjectSymbol
    from robotics_utils.planning.task_planning.parameter import Bindings, Parameter
    from robotics_utils.planning.task_planning.predicate import Predicate


@dataclass(frozen=True)
class Atom:
    """An atom (i.e., atomic formula) is a predicate symbol with partial bindings.

    Reference: Chapter 8.2.4 ("Atomic sentences"), pg. 260 of AIMA (4th Ed.) by Russell and Norvig.
    """

    predicate: Predicate
    bindings: Bindings
    """A mapping from bound parameter names to corresponding object arguments."""

    @property
    def name(self) -> str:
        """Retrieve the name of the atomic formula's predicate."""
        return self.predicate.name

    @property
    def unbound_params(self) -> tuple[Parameter, ...]:
        """Retrieve a tuple of the unbound parameters of the atom."""
        return tuple(p for p in self.predicate.parameters if p.name not in self.bindings)

    @property
    def is_grounded(self) -> bool:
        """Check whether all parameters of the atom are bound to objects."""
        return not self.unbound_params

    def bind(self, **kwargs: ObjectSymbol) -> Atom:
        """Return a new atom with updated bindings.

        :param **kwargs: Bindings assigning object symbols to parameter names
        :return: Updated atom with additional parameters bound
        :raises ValueError: If a given object-parameter binding is invalid due to type constraints
        """
        for param_name, obj_symbol in kwargs.items():
            param_type = self.predicate.get_parameter_type(param_name)  # KeyError on invalid name

            if param_type != obj_symbol.type_:
                raise ValueError(
                    f"Cannot bind object {obj_symbol} to parameter of type '{param_type}'.",
                )

        new_bindings = dict(self.bindings)
        new_bindings.update(kwargs)

        return replace(self, bindings=new_bindings)

    def ground(self, **kwargs: ObjectSymbol) -> GroundAtom:
        """Fully ground the atomic formula using the given bindings.

        :param **kwargs: Bindings assigning object symbols to parameter names (optional)
        :return: Ground atom with all its parameters bound
        :raises ValueError: If the given bindings do not fully ground the atom
        """
        updated_atom = self.bind(**kwargs)
        if not updated_atom.is_grounded:
            ubp = ", ".join(p.name for p in updated_atom.unbound_params)
            raise ValueError(f"{updated_atom.name} is not grounded. Unbound parameters: {ubp}")

        arguments = tuple(updated_atom.bindings[p.name] for p in self.predicate.parameters)
        return GroundAtom(self.predicate, arguments)
