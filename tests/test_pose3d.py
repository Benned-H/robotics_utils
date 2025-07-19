"""Unit tests for Pose3D, a class representing poses in 3D space."""

from hypothesis import given

from robotics_utils.kinematics.pose3d import Pose3D

from .hypothesis_strategies import poses


@given(poses())
def test_pose3d_to_homogeneous_matrix_and_back(pose: Pose3D) -> None:
    """Verify that any Pose3D is unchanged after converting to and from a homogeneous matrix."""
    # Arrange/Act - Given a 3D pose, convert to and from a homogeneous transformation matrix
    matrix = pose.to_homogeneous_matrix()
    result_pose = Pose3D.from_homogeneous_matrix(matrix, ref_frame=pose.ref_frame)

    # Assert - Expect that the matrix is 4x4 and the resulting Pose3D equals the original
    assert matrix.shape == (4, 4)
    assert pose.approx_equal(result_pose)


@given(poses())
def test_pose3d_to_list_and_back(pose: Pose3D) -> None:
    """Verify that any Pose3D is unchanged after converting to and from an equivalent list."""
    # Arrange/Act - Given a 3D pose, convert to and from a list of [x, y, z, roll, pitch, yaw]
    pose_list = pose.to_list()
    result_pose = Pose3D.from_list(pose_list, ref_frame=pose.ref_frame)

    # Assert - Expect that the list has length six and the resulting Pose3D equals the original
    assert len(pose_list) == 6, "Expected pose list of the form [x, y, z, roll, pitch, yaw]"
    assert pose.approx_equal(result_pose)
