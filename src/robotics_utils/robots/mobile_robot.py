"""Define a class representing the interface for a mobile robot."""

from abc import ABC, abstractmethod

from robotics_utils.kinematics import Point2D, Pose2D
from robotics_utils.math.polyline import Polyline2D
from robotics_utils.motion_planning.navigation import GoalBasePose


class MobileRobot(ABC):
    """A mobile robot capable of self-locomotion."""

    @property
    @abstractmethod
    def current_base_pose(self) -> Pose2D:
        """Retrieve the robot's current base pose."""
        ...

    @abstractmethod
    def go_to_pose(self, base_pose: Pose2D, timeout_s: float) -> bool:
        """Navigate to the specified base pose.

        :param base_pose: Target base pose for the robot
        :param timeout_s: Timeout (seconds) for the navigation command
        :return: True if the robot reached the base pose, else False
        """
        ...

    def follow_path(
        self,
        path: Polyline2D,
        goal: GoalBasePose,
        lookahead_m: float,
        timeout_s: float,
        min_progress_m: float,
    ) -> bool:
        """Follow the given path of target base poses.

        :param path: Sequence of 2D navigation waypoints
        :param goal: Goal base pose with thresholds specifying when it's considered 'reached'
        :param lookahead_m: Lookahead distance (meters of arclength) along the path
        :param timeout_s: Timeout (seconds) per "GoToPose" command
        :param min_progress_m: Minimum progress (meters) between iterations required to continue
        :return: True if the final base pose is reached, otherwise False
        """
        total_arclength_m = path.total_arclength_m
        prev_s = -float("inf")  # Arclength progress along the path

        while True:
            s_here, closest_point, seg_idx = path.project_point(self.current_base_pose.position)

            if s_here - prev_s < min_progress_m:
                break  # Little or no progress; break to avoid getting stuck

            s_target = min(s_here + lookahead_m, total_arclength_m)
            target_point, target_seg = path.interpolate_at_s(s_target)
            target_yaw_rad = path.tangent_yaw_at_segment(target_seg)
            target_pose = Pose2D(target_point.x, target_point.y, target_yaw_rad)

            # Command the robot toward the new target pose
            _ = self.go_to_pose(target_pose, timeout_s=timeout_s)

            prev_s = s_here

            if s_target >= total_arclength_m - 1e-6:
                break  # Reached (or essentially reached) the end of the path

        self.go_to_pose(goal.goal_pose, timeout_s=timeout_s)

        return goal.check_reached(self, change_frames=True)
