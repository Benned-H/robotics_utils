"""Define an interface to control the kinematic state of a simulator."""

from typing import Protocol

from robotics_utils.collision_models import CollisionModel
from robotics_utils.spatial import Pose3D
from robotics_utils.states.object_states import ObjectKinematicState


class KinematicSimulator(Protocol):
    """An interface to control the kinematic state of a simulator."""

    def set_object_state(self, obj_state: ObjectKinematicState) -> None:
        """Set the kinematic state of an object in the simulator."""
        ...

    def set_object_pose(self, obj_name: str, pose: Pose3D) -> None:
        """Update the pose of the named object.

        :raises KeyError: If the named object doesn't exist in the simulator
        """
        ...

    def set_collision_model(self, obj_name: str, collision_model: CollisionModel) -> None:
        """Replace the collision geometry of the named object.

        :raises KeyError: If the named object doesn't exist in the simulator
        """

    def remove_object(self, obj_name: str) -> bool:
        """Remove the named object from the simulator state.

        :return: True if the object was successfully removed, else False
        """
        ...
