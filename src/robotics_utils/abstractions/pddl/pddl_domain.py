"""Define a dataclass to represent PDDL domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.abstractions.objects import ObjectTypes
    from robotics_utils.abstractions.operators import Operator
    from robotics_utils.abstractions.predicates import Predicate


@dataclass(frozen=True)
class PDDLDomain:
    """A PDDL domain defining the "universal" aspects of a planning problem."""

    name: str
    """Name of the domain."""

    requirements: set[str]
    """Additional PDDL features required by the domain (e.g., `:typing`)."""

    types: ObjectTypes
    """The set of object types used in the domain."""

    predicates: set[Predicate]
    """The set of predicates (i.e., lifted Boolean state classifiers) in the domain."""

    operators: set[Operator]
    """The set of operators (i.e., lifted abstract actions) in the domain."""
