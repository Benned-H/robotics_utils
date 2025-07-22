"""Unit tests for Pose3D, a class representing poses in 3D space."""

import hypothesis.strategies as st
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
    assert pose.approx_equal(result_pose, atol=1e-07)


@given(poses())
def test_pose3d_to_yaml_and_back(pose: Pose3D) -> None:
    """Verify that any Pose3D is unchanged after converting to and from YAML data."""
    # Arrange/Act - Given a 3D pose, convert to and from a dictionary for export to YAML
    pose_dict = pose.to_yaml_dict()
    result_pose = Pose3D.from_yaml_dict(pose_dict)

    # Assert - Expect that the resulting Pose3D equals the original
    assert pose.approx_equal(result_pose)


@given(poses())
def test_pose3d_identity_multiplication(pose: Pose3D) -> None:
    """Verify that any Pose3D is unchanged after multiplication by the identity pose."""
    # Arrange - Create a pose representing the identity transformation
    identity_pose = Pose3D.identity()

    # Act - Compute left-side and right-side multiplications by the identity pose
    left_result = identity_pose @ pose
    right_result = pose @ identity_pose

    # Replace the frame of the left-side result; it won't match the original pose otherwise
    left_result.ref_frame = pose.ref_frame

    # Assert - Expect that both multiplication results equal the original pose
    assert pose.approx_equal(left_result)
    assert pose.approx_equal(right_result)


@given(poses(), st.text())
def test_pose3d_inverse_multiplication(pose: Pose3D, pose_frame: str) -> None:
    """Verify that multiplying any Pose3D by its inverse gives the identity transform."""
    # Arrange/Act - Given a 3D pose, find its inverse and the result of multiplying the two
    inverse_pose = pose.inverse(pose_frame)
    left_product = inverse_pose @ pose
    right_product = pose @ inverse_pose

    # Assert - Expect that both multiplication results equal the identity pose
    identity_wrt_pose_frame = Pose3D.identity(pose_frame)
    identity_wrt_ref_frame = Pose3D.identity(pose.ref_frame)

    # Expect that left-multiplying results in the pose's frame as the reference frame
    assert identity_wrt_pose_frame.approx_equal(left_product, atol=1e-06)

    # Expect that right-multiplying results in the same reference frame as the pose
    assert identity_wrt_ref_frame.approx_equal(right_product, atol=1e-06)
