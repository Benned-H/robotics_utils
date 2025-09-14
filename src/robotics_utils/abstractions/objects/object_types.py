"""Define a base class that makes it easy to create Python-typed object types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class BaseObjectType:
    """A base class for simple object types."""

    name: str
    visual_description: str | None = None

    def __str__(self) -> str:
        """Return a readable string representation of the object."""
        return f"{type(self).__name__}({self.name})"


class ObjectTypes:
    """The set of known object types in a PDDL domain."""

    def __init__(self, objects: Iterable[object]) -> None:
        """Initialize the set of known object types using the given objects."""
        self._types: set[type] = {type(obj) for obj in objects}
        """The set of all types in the domain."""
