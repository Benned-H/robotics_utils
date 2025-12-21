"""Define a dataclass to represent PDDL domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.abstractions.symbols.operators import Operator
    from robotics_utils.abstractions.symbols.predicate import Predicate


@dataclass
class PDDLDomain:
    """A PDDL domain defining the 'universal' aspects of a class of planning problems.

    Reference: https://planning.wiki/ref/pddl/domain
    """

    name: str

    requirements: set[str]
    """Additional PDDL features required by the domain."""

    types: set[str]
    """Flat collection of object types used in the domain."""

    predicates: set[Predicate]
    """The set of predicates (i.e., lifted relationships between objects) in the domain."""

    operators: set[Operator]
    """The set of operators (i.e., abstract action templates) in the domain."""
