"""Define a class to represent an object-centric environment state."""

from __future__ import annotations

from typing import TYPE_CHECKING, Generic, KeysView, TypeVar

if TYPE_CHECKING:
    from robotics_utils.objects.object_types import ObjectTypes

ObjectT = TypeVar("ObjectT")
"""Represents a concrete object in the environment."""


class ObjectCentricState(Generic[ObjectT]):
    """An environment state consisting of typed objects."""

    def __init__(self, objects: dict[str, ObjectT], object_types: ObjectTypes) -> None:
        """Initialize the object-centric state with objects and their types.

        :param objects: A map from object names to object instances
        :param object_types: Organizes the types of each object
        """
        self._objects = objects
        """Maps each object's name to the object instance."""

        self.object_types = object_types
        """Organizes the type information of objects in the environment."""

    def __contains__(self, object_name: str) -> bool:
        """Evaluate whether the named object exists in the environment."""
        return object_name in self._objects

    @property
    def object_names(self) -> KeysView[str]:
        """Retrieve the names of all objects in the environment."""
        return self._objects.keys()

    def get_object(self, object_name: str) -> ObjectT:
        """Retrieve the named object from the state.

        :raises KeyError: If an unknown object name is given
        """
        if object_name not in self._objects:
            raise KeyError(f"Cannot retrieve unknown object: '{object_name}'.")
        return self._objects[object_name]
