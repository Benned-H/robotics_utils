"""Define a general-purpose interface for a robot manipulator."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from robotics_utils.kinematics import Configuration
    from robotics_utils.motion_planning import MotionPlanningQuery
    from robotics_utils.robots.angular_gripper import AngularGripper
    from robotics_utils.skills import Outcome
    from robotics_utils.spatial import Pose3D

TrajectoryT = TypeVar("TrajectoryT")
"""Type of the motion plans computed and/or executed by a manipulator."""


class Manipulator(ABC, Generic[TrajectoryT]):
    """An interface for a robot manipulator."""

    def __init__(self, name: str, gripper: AngularGripper) -> None:
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
    def compute_motion_plan(self, query: MotionPlanningQuery) -> TrajectoryT | None:
        """Compute a motion plan for the given planning query.

        :param query: Specifies an end-effector target and (optionally) objects to ignore
        :return: Computed motion plan trajectory, or None if no plan was found
        """
        ...

    @abstractmethod
    def execute_motion_plan(self, trajectory: TrajectoryT) -> bool:
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
    def grasp(self, object_name: str) -> Outcome[Pose3D]:
        """Grasp the named object using the manipulator's gripper.

        This method should close the robot's gripper and update the kinematic state appropriately.

        :return: Boolean success, outcome message, and end-effector relative pose of the object
        """
        ...

    @abstractmethod
    def release(self, object_name: str, placed_frame: str) -> Outcome[Pose3D]:
        """Release the named object using the manipulator's gripper.

        This method should open the robot's gripper and update the kinematic state appropriately.

        :param object_name: Name of the held object to be released
        :param placed_frame: Parent frame of the object after it has been released

        :return: Boolean success, outcome message, and resulting relative pose of the object
        """
        ...
