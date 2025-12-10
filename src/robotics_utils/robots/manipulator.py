"""Define a general-purpose interface for a robot manipulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration, Pose3D
    from robotics_utils.motion_planning import Trajectory
    from robotics_utils.robots.angular_gripper import AngularGripper
    from robotics_utils.skills import Outcome


class Manipulator(ABC):
    """An interface for a robot manipulator."""

    def __init__(self, name: str, gripper: AngularGripper | None) -> None:
        """Initialize the manipulator with an optional gripper."""
        self.name = name
        self.gripper = gripper

    @property
    @abstractmethod
    def ee_link_name(self) -> str:
        """Retrieve the name of the manipulator's end-effector link."""
        ...

    @property
    @abstractmethod
    def joint_names(self) -> tuple[str, ...]:
        """Retrieve the names of the joints in the manipulator in their canonical order."""
        ...

    @property
    @abstractmethod
    def configuration(self) -> Configuration:
        """Retrieve the manipulator's current configuration."""
        ...

    @property
    def joint_values(self) -> tuple[float, ...]:
        """Retrieve the current joint values of the manipulator in their canonical order."""
        return tuple(self.configuration[joint_name] for joint_name in self.joint_names)

    @abstractmethod
    def execute_motion_plan(self, trajectory: Trajectory) -> bool:
        """Execute the given trajectory using the manipulator.

        :return: True if execution succeeded, False otherwise
        """
        ...

    @abstractmethod
    def compute_ik(self, ee_target: Pose3D) -> Configuration | None:
        """Compute an inverse kinematics solution to place the end-effector at the given pose.

        :param ee_target: Target pose of the end-effector
        :return: Manipulator configuration solving the IK problem (else None)
        """
        ...

    @abstractmethod
    def grasp(self, object_name: str) -> Outcome:
        """Grasp the named object using the manipulator's gripper.

        :return: Boolean success of the grasp and an outcome message
        """
        ...

    @abstractmethod
    def release(self, object_name: str) -> Outcome:
        """Release the named object using the manipulator's gripper.

        :return: Boolean success of the release and an outcome message
        """
        ...
