"""Define a class to organize object types."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, KeysView

from robotics_utils.io.yaml_utils import load_yaml_data

if TYPE_CHECKING:
    from pathlib import Path


class ObjectTypes:
    """A data structure for organizing object types."""

    def __init__(self, object_types: dict[str, set[str]]) -> None:
        """Initialize the class using a map from object names to their sets of types."""
        self._object_types = object_types
        """Maps each object's name to its set of types."""

        self._objects_of_type: dict[str, set[str]] = defaultdict(set)
        """Maps each object type to the names of all objects of that type."""

        for obj_name, obj_types in self._object_types.items():
            for obj_type in obj_types:
                self._objects_of_type[obj_type].add(obj_name)

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> ObjectTypes:
        """Construct an ObjectTypes collection from the given YAML file."""
        yaml_data = load_yaml_data(yaml_path, required_keys={"object_types"})
        object_types_data: dict[str, list[str]] = yaml_data["object_types"]
        return ObjectTypes({o_name: set(types) for o_name, types in object_types_data.items()})

    @property
    def all_types(self) -> KeysView[str]:
        """Retrieve all object types in the collection."""
        return self._objects_of_type.keys()

    def get_types_of(self, object_name: str) -> set[str]:
        """Retrieve the type(s) of the named object."""
        if object_name not in self._object_types:
            raise KeyError(f"Cannot retrieve types of unknown object: '{object_name}'.")
        return self._object_types[object_name]

    def get_objects_of_type(self, obj_type: str) -> set[str]:
        """Retrieve the names of all objects with the given type."""
        if obj_type not in self.all_types:
            raise KeyError(f"Cannot retrieve objects of unknown type: '{obj_type}'.")

        return self._objects_of_type[obj_type]
