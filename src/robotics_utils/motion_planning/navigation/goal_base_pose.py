"""Define a class to represent goal base poses for navigation."""

from dataclasses import dataclass

from robotics_utils.io.logging import log_info
from robotics_utils.kinematics import Pose2D
from robotics_utils.math.distances import angle_difference_rad, euclidean_distance_2d_m
from robotics_utils.robots.mobile_robot import MobileRobot


@dataclass(frozen=True)
class GoalBasePose:
    """A goal base pose with thresholds determining when a robot has 'reached' it."""

    goal_pose: Pose2D

    reached_distance_m: float
    """Distance (meters) from the goal base pose within which it is considered 'reached'."""

    reached_abs_angle_rad: float
    """Absolute angle (radians) from goal base pose yaw within which it is considered 'reached'."""

    def check_reached(self, robot: MobileRobot, change_frames: bool) -> bool:
        """Check whether the given robot has reached the goal base pose.

        :param robot: Mobile robot with the base pose to be checked
        :param change_frames: Whether to change frames to fix a goal-robot pose mismatch
        :return: True if the robot is sufficiently close to the target pose, else False
        """
        curr_pose = robot.current_base_pose

        if not change_frames and curr_pose.ref_frame != self.goal_pose.ref_frame:
            rf = curr_pose.ref_frame
            gf = self.goal_pose.ref_frame
            raise ValueError(f"Robot base pose frame '{rf}' differs from goal pose frame '{gf}'.")

        distance_2d_m = euclidean_distance_2d_m(curr_pose, self.goal_pose, change_frames)
        angle_error_rad = angle_difference_rad(curr_pose.yaw_rad, self.goal_pose.yaw_rad)

        distance_reached = distance_2d_m < self.reached_distance_m
        angle_reached = angle_error_rad < self.reached_abs_angle_rad
        result = distance_reached and angle_reached

        log_info(f"Current Euclidean distance to goal pose: {distance_2d_m} m")
        log_info(f"Current absolute angular error from goal pose: {angle_error_rad} rad")
        log_info(f"Ending navigation? {result}")

        return result
