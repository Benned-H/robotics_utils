"""Define utility functions to compute various distance metrics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from robotics_utils.spatial.poses import Pose2D, Pose3D
    from robotics_utils.spatial.rotations import Quaternion


def euclidean_distance_2d_m(pose_a: Pose2D, pose_b: Pose2D, *, change_frames: bool) -> float:
    """Compute the Euclidean distance (meters) between two poses on the 2D plane.

    :param pose_a: First 2D pose used to compute the distance
    :param pose_b: Second 2D pose used to compute the distance
    :param change_frames: Whether or not to change both poses into the same frame (requires ROS)
    :return: Straight-line distance (meters) between the two 2D poses
    """
    if change_frames and pose_a.ref_frame != pose_b.ref_frame:
        from robotics_utils.ros.transform_manager import TransformManager  # noqa: PLC0415

        pose_a = TransformManager.convert_to_frame(pose_a.to_3d(), pose_b.ref_frame).to_2d()

    return float(np.linalg.norm(np.array([pose_a.x - pose_b.x, pose_a.y - pose_b.y])))


def euclidean_distance_3d_m(pose_a: Pose3D, pose_b: Pose3D, *, change_frames: bool) -> float:
    """Compute the Euclidean distance (meters) between two 3D poses.

    :param pose_a: First 3D pose used to compute the distance
    :param pose_b: Second 3D pose used to compute the distance
    :param change_frames: Whether or not to change both poses into the same frame (requires ROS)
    :return: Straight-line distance (meters) in 3D space between the two poses
    """
    if change_frames and pose_a.ref_frame != pose_b.ref_frame:
        from robotics_utils.ros.transform_manager import TransformManager  # noqa: PLC0415

        pose_a = TransformManager.convert_to_frame(pose_a, pose_b.ref_frame)

    return float(np.linalg.norm(pose_a.position.to_array() - pose_b.position.to_array()))


def angle_between_quaternions_deg(q1: Quaternion, q2: Quaternion) -> float:
    """Compute the angle (degrees) between two unit quaternions representing 3D rotations.

    Reference: https://math.stackexchange.com/a/167828
    """
    product = q1 * q2.conjugate()
    angle_rad = 2.0 * np.arccos(product.w)
    return np.rad2deg(angle_rad)
