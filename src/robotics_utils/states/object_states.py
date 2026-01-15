"""Define classes to represent object states."""

from dataclasses import dataclass

from robotics_utils.collision_models import CollisionModel
from robotics_utils.spatial import Pose3D
from robotics_utils.states.visual_states import ObjectVisualState


@dataclass(frozen=True)
class ObjectKinematicState:
    """The kinematic state of a physical object in the environment."""

    name: str
    pose: Pose3D
    collision_model: CollisionModel


@dataclass(frozen=True)
class ObjectState:
    """Full state of an object in the environment, including kinematic and visual data."""

    name: str
    object_type: str
    kinematic_state: ObjectKinematicState
    visual_state: ObjectVisualState
