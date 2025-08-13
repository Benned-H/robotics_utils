"""Define a class to represent typed objects in an environment."""

from __future__ import annotations

from collections import defaultdict
from typing import KeysView


class Objects:
    """A collection of typed objects in an environment."""

    def __init__(self, object_to_types: dict[str, set[str]]) -> None:
        """Initialize the collection of typed objects."""
        self.object_to_types = object_to_types
        """Maps object names to their sets of types."""

        # Construct a map from each object type to the names of all objects of that type
        self._objects_of_type: dict[str, set[str]] = defaultdict(set)
        for obj_name, obj_types in self.object_to_types.items():
            for obj_type in obj_types:
                self._objects_of_type[obj_type].add(obj_name)

    def __contains__(self, object_name: str) -> bool:
        """Evaluate whether the named object is in this collection."""
        return object_name in self.object_to_types

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve the names of all objects in this collection."""
        return self.object_to_types.keys()

    @property
    def all_types(self) -> KeysView[str]:
        """Retrieve all object types used in this collection."""
        return self._objects_of_type.keys()

    def get_types_of(self, object_name: str) -> set[str]:
        """Retrieve the type(s) of the named object."""
        if object_name not in self.object_to_types:
            raise KeyError(f"Cannot retrieve types of unknown object: '{object_name}'.")
        return self.object_to_types[object_name]

    def get_objects_of_type(self, obj_type: str) -> set[str]:
        """Retrieve the names of all objects with the given type."""
        if obj_type not in self._objects_of_type:
            raise KeyError(f"Unknown object type: '{obj_type}'.")

        return self._objects_of_type[obj_type]
