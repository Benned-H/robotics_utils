"""Define a general-purpose interface for a mobile robot base."""

from abc import ABC, abstractmethod

from robotics_utils.kinematics import Pose2D
from robotics_utils.motion_planning.navigation_goal import NavigationGoal


class MobileRobot(ABC):
    """An abstract interface for a mobile robot base."""

    @property
    @abstractmethod
    def current_base_pose(self) -> Pose2D:
        """Retrieve the robot's current base pose."""
        ...

    def goal_reached(self, goal: NavigationGoal, *, change_frames: bool) -> bool:
        """Check whether the mobile robot has reached the given navigation goal.

        :param goal: Goal base pose during navigation
        :param change_frames: Whether to change frames to resolve a frame mismatch
        :return: True if the mobile robot has reached the goal, else False
        """
        return goal.check_reached_by(self.current_base_pose, change_frames=change_frames)
