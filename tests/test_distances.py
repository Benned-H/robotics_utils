"""Unit tests for distance utility functions defined in distances.py."""

import numpy as np
from hypothesis import given

from robotics_utils.kinematics import Pose2D
from robotics_utils.math.distances import angle_difference_rad, euclidean_distance_2d_m

from .kinematics_strategies import angles_rad, poses_2d


@given(poses_2d(), poses_2d())
def test_euclidean_distance_2d_m(pose_a: Pose2D, pose_b: Pose2D) -> None:
    """Verify that the Euclidean distance (meters) between any two 2D poses is non-negative."""
    # Arrange/Act - Given two 2D poses, compute the Euclidean distance between them
    distance_2d_m = euclidean_distance_2d_m(pose_a, pose_b, change_frames=False)

    # Assert - Euclidean distances should be non-negative, and zero if the poses are equal in (x,y)
    assert distance_2d_m >= 0.0
    if pose_a.x == pose_b.x and pose_a.y == pose_b.y:
        assert distance_2d_m == 0.0


@given(angles_rad(), angles_rad())
def test_angle_difference_rad(a_rad: float, b_rad: float) -> None:
    """Verify that any two angles' absolute difference in radians is within [0, pi]."""
    # Arrange/Act - Given two angles in radians, compute the absolute difference between them
    abs_difference_rad = angle_difference_rad(a_rad, b_rad)

    # Assert - The difference should be within [0, pi], and should be zero if the angles are equal
    assert 0.0 <= abs_difference_rad <= np.pi
    if a_rad == b_rad:
        assert abs_difference_rad == 0.0
