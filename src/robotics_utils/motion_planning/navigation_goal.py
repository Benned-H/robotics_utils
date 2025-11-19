"""Define a class to represent a target base pose during navigation."""

from dataclasses import dataclass

from robotics_utils.io import log_info
from robotics_utils.kinematics import Pose2D
from robotics_utils.math.distances import angle_difference_rad, euclidean_distance_2d_m


@dataclass(frozen=True)
class NavigationGoal:
    """A goal base pose with thresholds determining when a robot has 'reached' it."""

    pose: Pose2D

    reached_distance_m: float
    """Distance (meters) within which a base pose is considered to have 'reached' the goal."""

    reached_abs_angle_rad: float
    """Absolute angle (radians) from this goal's yaw for it to be considered 'reached'."""

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

        distance_2d_m = euclidean_distance_2d_m(base_pose, self.pose, change_frames)
        angle_error_rad = angle_difference_rad(base_pose.yaw_rad, self.pose.yaw_rad)

        distance_reached = distance_2d_m < self.reached_distance_m
        angle_reached = angle_error_rad < self.reached_abs_angle_rad
        result = distance_reached and angle_reached

        log_info(f"Current Euclidean distance to goal pose: {distance_2d_m} m")
        log_info(f"Current absolute angular error from goal pose: {angle_error_rad} rad")
        log_info(f"Goal reached? {result}")

        return result
