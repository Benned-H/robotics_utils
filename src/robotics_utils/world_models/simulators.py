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

    def add_object(self, obj_model: ObjectModel) -> None:
        """Add an object to the simulator's state."""
        ...

    def remove_object(self, obj_name: str) -> None:
        """Remove the named object from the simulator state."""
        ...

    def set_object_pose(self, obj_name: str, new_pose: Pose3D) -> None:
        """Update the pose of the named object."""
        ...

    def set_collision_model(self, obj_name: str, collision_model: CollisionModel) -> None:
        """Replace the collision geometry of the named object."""
        ...
