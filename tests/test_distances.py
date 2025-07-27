"""Unit tests for distance utility functions defined in distances.py."""

import pytest
from hypothesis import given

from robotics_utils.kinematics.poses import Pose2D
from robotics_utils.math.distances import euclidean_distance_2d_m

from .hypothesis_strategies import poses_2d


@given(poses_2d(), poses_2d())
def test_euclidean_distance_2d_m(pose_a: Pose2D, pose_b: Pose2D) -> None:
    """Verify that the Euclidean distance (meters) between any two 2D poses is non-negative."""
    # Arrange/Act - Given two 2D poses, compute the distance between them
    distance_2d_m = euclidean_distance_2d_m(pose_a, pose_b, change_frames=False)

    # Assert - Euclidean distances should be non-negative, and zero if the poses are equal in (x,y)
    assert distance_2d_m >= 0.0
    if pose_a.x == pose_b.x and pose_a.y == pose_b.y:
        assert distance_2d_m == 0.0
