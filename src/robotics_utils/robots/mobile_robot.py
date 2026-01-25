"""Define a general-purpose interface for a mobile robot base."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from robotics_utils.motion_planning import NavigationGoal
    from robotics_utils.skills import Outcome
    from robotics_utils.spatial import Pose2D


class MobileRobot(ABC):
    """An abstract interface for a mobile robot base."""

    @property
    @abstractmethod
    def current_base_pose(self) -> Pose2D:
        """Retrieve the robot's current base pose."""
        ...

    @abstractmethod
    def compute_navigation_plan(self, initial: Pose2D, goal: Pose2D) -> list[Pose2D] | None:
        """Compute a navigation plan between the two given robot base poses.

        :param initial: Robot base pose from which the plan begins
        :param goal: Target base pose to be reached by the navigation plan
        :return: Navigation plan (list of base pose waypoints), or None if no plan is found
        """
        ...

    @abstractmethod
    def execute_navigation_plan(self, nav_plan: list[Pose2D], timeout_s: float = 60.0) -> Outcome:
        """Execute the given navigation plan on the mobile robot.

        :param nav_plan: Navigation plan of 2D base pose waypoints
        :param timeout_s: Duration (seconds) after which the plan times out (default: 60 seconds)
        :return: Boolean success indicator and explanatory message
        """
        ...

    def goal_reached(self, goal: NavigationGoal, *, change_frames: bool) -> bool:
        """Check whether the mobile robot has reached the given navigation goal.

        :param goal: Goal base pose during navigation
        :param change_frames: Whether to change frames to resolve a frame mismatch
        :return: True if the mobile robot has reached the goal, else False
        """
        return goal.check_reached_by(self.current_base_pose, change_frames=change_frames)
