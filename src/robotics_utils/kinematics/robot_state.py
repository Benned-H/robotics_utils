"""Define a dataclass to represent the state of a mobile manipulator robot."""

from dataclasses import dataclass

from robotics_utils.kinematics.kinematics_core import Configuration
from robotics_utils.kinematics.poses import Pose3D


@dataclass
class RobotState:
    """The kinematic state of a mobile manipulator robot."""

    base_pose: Pose3D
    configuration: Configuration
