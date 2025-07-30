"""Define classes to represent the kinematic model of an actuated robot."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from robotics_utils.kinematics.collision_models import CollisionModel
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.poses import Pose3D


class JointType(Enum):
    """An enumeration of robot joint types."""

    REVOLUTE = 0
    PRISMATIC = 1


@dataclass
class Link:
    """A rigid body attached to other rigid bodies."""

    name: str
    geometry: CollisionModel


@dataclass
class Joint:
    """An actuated joint of a robot."""

    name: str
    joint_type: JointType

    parent_link: Link
    child_link: Link

    origin: Pose3D
    """Transform from the parent link to the child link."""

    axis: Point3D
    """Joint axis, specified in the joint frame."""

    state: float
    """Current position (rad or m) of the joint."""

    lower_limit: float
    """Minimum position (rad or m) of the joint."""

    upper_limit: float
    """Maximum position (rad or m) of the joint."""


@dataclass
class RevoluteJoint(Joint):
    """An actuated revolute joint of a robot."""

    @property
    def angle_rad(self) -> float:
        """Retrieve the angle (in radians) of the joint."""
        return self.state


@dataclass
class PrismaticJoint(Joint):
    """An actuated prismatic joint of a robot."""

    @property
    def position_m(self) -> float:
        """Retrieve the position (in meters) of the joint."""
        return self.state


@dataclass(frozen=True)
class RobotModel:
    """A kinematic model of an actuated robot."""

    joints: set[Joint]
    links: set[Link]
