"""Define a dataclass to represent PDDL domains."""

from dataclasses import dataclass

from robotics_utils.classical_planning.operators import Operator
from robotics_utils.classical_planning.predicates import Predicate
from robotics_utils.classical_planning.type_hierarchy import TypeHierarchy


@dataclass(frozen=True)
class PDDLDomain:
    """A PDDL domain defining the 'universal' aspects of a planning problem."""

    name: str
    """Name of the domain."""

    requirements: set[str]
    """Additional PDDL features required by the domain."""

    types: TypeHierarchy
    """A hierarchy of the object types used in the domain."""

    predicates: set[Predicate]
    """The set of predicates (i.e., lifted Boolean state classifiers) in the domain."""

    operators: set[Operator]
    """The set of operators (i.e., lifted abstract actions) in the domain."""
