"""Define a class to represent a target base pose during navigation."""

from dataclasses import dataclass

from robotics_utils.io import log_info
from robotics_utils.math import angle_difference_rad
from robotics_utils.spatial import Pose2D, euclidean_distance_2d_m


@dataclass(frozen=True)
class NavigationGoal:
    """A goal base pose with thresholds determining when a robot has 'reached' it."""

    pose: Pose2D

    goal_reached_m: float = 0.2
    """Distance (meters) within which a base pose is considered to have 'reached' the goal."""

    goal_yaw_tolerance_rad: float = 0.3
    """Absolute angle (radians) within which the robot's yaw is 'close enough' to the goal yaw."""

    def check_reached_by(self, base_pose: Pose2D, *, change_frames: bool) -> bool:
        """Check whether the given base pose has 'reached' the goal base pose.

        :param base_pose: Base pose evaluated for proximity to the goal base pose
        :param change_frames: Whether to change frames to fix a reference frame mismatch
        :return: True if the base pose is sufficiently close to the goal pose, else False
        """
        if not change_frames and base_pose.ref_frame != self.pose.ref_frame:
            rf = base_pose.ref_frame
            gf = self.pose.ref_frame
            raise ValueError(f"Base pose frame '{rf}' differs from goal pose frame '{gf}'.")

        distance_2d_m = euclidean_distance_2d_m(base_pose, self.pose, change_frames=change_frames)
        angle_error_rad = angle_difference_rad(base_pose.yaw_rad, self.pose.yaw_rad)

        distance_reached = distance_2d_m < self.goal_reached_m
        angle_reached = angle_error_rad < self.goal_yaw_tolerance_rad
        result = distance_reached and angle_reached

        log_info(f"Current Euclidean distance to goal pose: {distance_2d_m} m")
        log_info(f"Current absolute angular error from goal pose: {angle_error_rad} rad")
        log_info(f"Goal reached? {result}")

        return result
