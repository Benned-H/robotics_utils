"""Define classes to represent symbolic typed objects."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, KeysView


@dataclass(frozen=True)
class ObjectSymbol:
    """A symbol representing a typed object in the environment."""

    name: str
    type_: str


class ObjectSymbols:
    """A collection of symbols representing typed objects."""

    def __init__(self, objects: Iterable[ObjectSymbol]) -> None:
        """Initialize the collection of object symbols."""
        self._objects: dict[str, ObjectSymbol] = {obj.name: obj for obj in objects}

        self._objects_of_type: dict[str, set[ObjectSymbol]] = defaultdict(set)
        """Map from each type name to the set of objects of that type."""

        for obj in self._objects.values():
            self._objects_of_type[obj.type_].add(obj)

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve the names of all objects in this collection."""
        return self._objects.keys()

    @property
    def all_types(self) -> KeysView[str]:
        """Retrieve all object types represented in this collection."""
        return self._objects_of_type.keys()

    def get(self, obj_name: str) -> ObjectSymbol | None:
        """Retrieve the named object symbol.

        :param obj_name: Name of an object
        :return: Object symbol with the name, or None if there isn't one
        """
        return self._objects.get(obj_name)

    def get_objects_of_type(self, obj_type: str) -> set[ObjectSymbol]:
        """Retrieve all stored object symbols with the given type.

        :raises KeyError: If an unknown object type is given
        """
        if obj_type not in self._objects_of_type:
            raise KeyError(f"Unknown object type: '{obj_type}'.")

        return self._objects_of_type[obj_type]
