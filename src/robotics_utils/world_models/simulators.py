"""Define an interface to control the environment state in a simulator."""

from dataclasses import dataclass
from typing import Protocol

from robotics_utils.collision_models import CollisionModel
from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class ObjectModel:
    """A physical object in the environment."""

    name: str
    pose: Pose3D
    collision_model: CollisionModel


class Simulator(Protocol):
    """An interface to control the kinematic state of a simulator."""

    def add_object(self, obj_model: ObjectModel) -> bool:
        """Add an object to the simulator's state.

        :param obj_model: Geometric model of the object to be added
        :return: True if the object was successfully added, else False
        """
        ...

    def remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the simulator state.

        :param obj_name: Name of the object to be removed
        :return: True if the object was successfully removed, else False
        """
        ...

    def set_object_pose(self, obj_name: str, new_pose: Pose3D) -> None:
        """Update the pose of the named object."""
        ...

    def set_collision_model(self, obj_name: str, collision_model: CollisionModel) -> None:
        """Replace the collision geometry of the named object."""
        ...
