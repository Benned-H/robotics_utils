"""Define a dataclass to represent PDDL domains."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from robotics_utils.planning.task_planning.object_type_hierarchy import ObjectTypeHierarchy
    from robotics_utils.planning.task_planning.predicate import Predicate

Operator = object  # TODO: Migrate Operator type and handle more than simple PRE/EFF sets


@dataclass(frozen=True)
class PDDLDomain:
    """A PDDL domain defines the "universal" aspects of a planning problem."""

    name: str
    """Name of the domain."""

    requirements: set[str]
    """Additional PDDL features required by the domain (e.g., `:typing`)."""

    types: ObjectTypeHierarchy
    """The hierarchy of object types defined in the domain."""

    predicates: set[Predicate]
    """The set of predicates (i.e., lifted Boolean state classifiers) in the domain."""

    operators: set[Operator]
    """The set of operators (i.e., lifted abstract actions) in the domain."""

    filepath: Path | None = None
    """Optional path to the corresponding PDDL domain file (default: None)."""
