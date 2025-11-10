"""Define a class to represent the kinematic state of an object."""

from dataclasses import dataclass

from robotics_utils.collision_models import CollisionModel
from robotics_utils.kinematics import Pose3D


@dataclass(frozen=True)
class ObjectKinematicState:
    """The kinematic state of a physical object in the environment."""

    name: str
    pose: Pose3D
    collision_model: CollisionModel
