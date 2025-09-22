"""Define a class representing the interface for a mobile robot."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

from robotics_utils.kinematics import Pose2D
from robotics_utils.math.polyline import Polyline2D
from robotics_utils.motion_planning.navigation import NavigationGoal
from robotics_utils.skills.skill import SkillResult


@dataclass(frozen=True)
class FollowPathCommand:
    """Parameters for a command to a path-following mobile robot."""

    path: Polyline2D
    goal: NavigationGoal
    lookahead_m: float = 2.0

    local_timeout_s: float = 0.25
    """Duration (seconds) after which each local-control command will time out."""

    total_timeout_s: float = 30.0
    """Duration (seconds) after which the overall path-following command will time out."""

    min_progress_m: float = 0.05
    """Minimum progress distance (meters) required between iterations to continue."""


class MobileRobot(ABC):
    """A mobile robot capable of self-locomotion."""

    @property
    @abstractmethod
    def current_base_pose(self) -> Pose2D:
        """Retrieve the robot's current base pose."""
        ...

    @abstractmethod
    def go_to_pose(self, base_pose: Pose2D, timeout_s: float) -> SkillResult:
        """Move directly to the specified base pose.

        :param base_pose: Target base pose for the robot
        :param timeout_s: Timeout (seconds) for the movement command
        :return: Tuple containing Boolean success and an outcome message
        """
        ...

    def has_reached(self, goal: NavigationGoal) -> bool:
        """Check whether the mobile robot has reached the given navigation goal.

        :param goal: Goal base pose during navigation
        :return: True if the mobile robot has reached the goal, else False
        """
        return goal.check_reached_by(self.current_base_pose, change_frames=True)

    def follow_path(self, command: FollowPathCommand) -> bool:
        """Follow the given path of target base poses.

        :param command: Parameters specifying path-following behavior (see the dataclass)
        :return: True if the final base pose is reached, otherwise False
        """
        end_time = time.time() + command.total_timeout_s

        total_arclength_m = command.path.total_arclength_m
        prev_s = -float("inf")  # Arclength progress along the path

        while time.time() < end_time:
            s_here, _, _ = command.path.project_point(self.current_base_pose.position)

            if s_here - prev_s < command.min_progress_m:
                break  # Little or no progress; break to avoid getting stuck

            s_target = min(s_here + command.lookahead_m, total_arclength_m)
            target_point, target_seg = command.path.interpolate_at_s(s_target)
            target_yaw_rad = command.path.tangent_yaw_at_segment(target_seg)
            target_pose = Pose2D(target_point.x, target_point.y, target_yaw_rad)

            # Command the robot toward the new target pose
            _ = self.go_to_pose(target_pose, command.local_timeout_s)

            prev_s = s_here

            if s_target >= total_arclength_m - 1e-6:
                break  # Reached (or essentially reached) the end of the path

        self.go_to_pose(command.goal.pose, timeout_s=command.local_timeout_s)

        return self.has_reached(command.goal)
