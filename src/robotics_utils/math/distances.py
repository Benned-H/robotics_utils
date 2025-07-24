"""Define utility functions to compute various distance metrics."""

from __future__ import annotations

import numpy as np

from robotics_utils.kinematics.poses import Pose2D
from robotics_utils.math.angles import normalize_angle
from robotics_utils.ros.transform_manager import TransformManager


def euclidean_distance_2d_m(pose_a: Pose2D, pose_b: Pose2D) -> float:
    """Compute the Euclidean distance (meters) between two poses on the 2D plane.

    :param pose_a: First 2D pose used to compute the distance
    :param pose_b: Second 2D pose used to compute the distance
    :return: Straight-line distance (meters) between the two 2D poses
    """
    if pose_a.ref_frame != pose_b.ref_frame:  # Ensure that the poses are in the same frame
        pose_a = TransformManager.convert_to_frame(pose_a.to_3d(), pose_b.ref_frame).to_2d()

    return np.linalg.norm(np.array([pose_a.x, pose_a.y]), np.array([pose_b.x, pose_b.y]))


def angle_difference_rad(a_rad: float, b_rad: float) -> float:
    """Compute the absolute difference (in normalized radians) between two angles.

    :param a_rad: First angle (radians) in the difference
    :param b_rad: Second angle (radians) in the difference
    :return: Absolute angle difference (radians, between 0 and pi)
    """
    difference_rad = normalize_angle(a_rad - b_rad)
    return abs(difference_rad)
