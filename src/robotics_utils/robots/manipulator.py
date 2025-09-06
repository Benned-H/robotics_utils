"""Define a general-purpose interface for a robot manipulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration, Pose3D
    from robotics_utils.motion_planning import Trajectory
    from robotics_utils.robots.angular_gripper import AngularGripper


class Manipulator(ABC):
    """An interface for a robot manipulator."""

    def __init__(self, name: str, base_frame: str, gripper: AngularGripper) -> None:
        """Initialize the manipulator with its base frame and gripper."""
        self.name = name
        self.base_frame = base_frame
        self.gripper = gripper

    @property
    @abstractmethod
    def end_effector_link_name(self) -> str:
        """Retrieve the name of the manipulator's end-effector link."""
        ...

    @property
    @abstractmethod
    def configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        ...

    @property
    def joint_values(self) -> tuple[float, ...]:
        """Retrieve the current joint values of the manipulator in their canonical order."""
        return tuple(self.configuration.values())

    @abstractmethod
    def convert_to_base_frame(self, pose: Pose3D) -> Pose3D:
        """Convert the given pose into the manipulator's base frame."""
        ...

    @abstractmethod
    def execute_motion_plan(self, trajectory: Trajectory) -> None:
        """Execute the given trajectory using the manipulator."""
        ...

    @abstractmethod
    def compute_ik(self, ee_target: Pose3D) -> Configuration | None:
        """Compute an inverse kinematics solution to place the end-effector at the given pose.

        :param ee_target: Target pose of the end-effector
        :return: Manipulator configuration solving the IK problem (else None)
        """
        ...
