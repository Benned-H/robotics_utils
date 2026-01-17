"""Unit tests for the Pose2D class."""

from hypothesis import given

from robotics_utils.spatial import Pose2D

from .strategies.spatial_strategies import poses_2d


@given(poses_2d())
def test_pose2d_to_homogeneous_matrix_and_back(pose: Pose2D) -> None:
    """Verify that any Pose2D is unchanged after converting to and from a homogeneous matrix."""
    # Arrange/Act - Given a 2D pose, convert into a homogeneous matrix, then back to a Pose2D
    matrix = pose.to_homogeneous_matrix()
    result_pose = Pose2D.from_homogeneous_matrix(matrix=matrix, ref_frame=pose.ref_frame)

    # Assert - Expect that the matrix is 3x3 and the resulting pose approx. equals the original
    assert matrix.shape == (3, 3)
    assert pose.approx_equal(result_pose)
