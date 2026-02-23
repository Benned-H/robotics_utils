"""Define a class to represent a hierarchy of object types."""

from collections import defaultdict


class ObjectTypeHierarchy:
    """A hierarchy of object types (represented as strings) in a planning domain."""

    def __init__(self) -> None:
        """Initialize an empty object type hierarchy."""
        self._child_types: dict[str, set[str]] = defaultdict(set)
        """A map from each parent type to the set of its child types."""

    def add_type(self, type_: str) -> None:
        """Add the given type to the type hierarchy."""
        if type_ not in self._child_types:
            self._child_types[type_] = set()

    def add_child_type(self, parent: str, child: str) -> None:
        """Add a child type of the specified parent type."""
        self._child_types[parent].add(child)
