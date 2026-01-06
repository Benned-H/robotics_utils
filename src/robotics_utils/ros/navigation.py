"""Define ROS-dependent utilities for managing robot navigation."""

import time
from dataclasses import dataclass

import rospy

from robotics_utils.math import angle_difference_rad
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.spatial import Pose2D, euclidean_distance_2d_m


@dataclass(frozen=True)
class GoalReachedThresholds:
    """Thresholds specifying when a robot is considered to have reached a goal pose."""

    distance_m: float  # Distance (meters) from the goal base pose
    abs_angle_rad: float  # Absolute angle (radians) from the yaw of the base pose


def check_reached_goal(
    target_pose_2d: Pose2D,
    thresholds: GoalReachedThresholds,
    pose_lookup_timeout_s: float = 5.0,
) -> bool:
    """Check whether the robot is considered to have reached a goal pose.

    :param target_pose_2d: Target base pose for a mobile robot
    :param thresholds: Thresholds specifying when the robot is considered to have reached a goal pose
    :param pose_lookup_timeout_s: Duration (sec) after which pose lookup times out (defaults to 5)
    :return: True if the robot is sufficiently close to the target pose, else False
    """
    pose_lookup_end_time = time.time() + pose_lookup_timeout_s
    target_frame = target_pose_2d.ref_frame

    curr_pose = None
    while curr_pose is None and time.time() < pose_lookup_end_time:
        curr_pose = TransformManager.lookup_transform("body", target_frame, timeout_s=0.1)

    if curr_pose is None:
        rospy.logfatal(f"Could not look up body pose in frame '{target_pose_2d.ref_frame}'.")
        return False

    distance_2d_m = euclidean_distance_2d_m(target_pose_2d, curr_pose.to_2d(), change_frames=True)
    angle_error_rad = angle_difference_rad(target_pose_2d.yaw_rad, curr_pose.yaw_rad)

    distance_reached = distance_2d_m < thresholds.distance_m
    angle_reached = angle_error_rad < thresholds.abs_angle_rad
    result = distance_reached and angle_reached

    rospy.loginfo(f"Current Euclidean distance to target pose: {distance_2d_m} m")
    rospy.loginfo(f"Current absolute angular error from target pose: {angle_error_rad} rad")
    rospy.loginfo(f"Ending navigation? {result}")

    return result
