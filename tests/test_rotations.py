"""Unit tests for classes representing 3D rotations and orientations."""

import numpy as np
import pytest
from hypothesis import given

from robotics_utils.kinematics.rotations import Quaternion, normalize_angle

from .hypothesis_strategies import angles_rad, quaternions


@given(angles_rad())
def test_normalize_angle(angle_rad: float) -> None:
    """Verify that any angle is correctly normalized into the interval [-pi, pi]."""
    # Arrange/Act - Given any angle (in radians), compute the corresponding normalized angle
    normalized_rad = normalize_angle(angle_rad)

    # Assert - Normalized angle should be in the interval [-pi, pi]
    assert -np.pi <= normalized_rad <= np.pi

    # Normalization should not have changed the cosine or sine of the angle
    assert np.cos(angle_rad) == pytest.approx(np.cos(normalized_rad), abs=1e-6)
    assert np.sin(angle_rad) == pytest.approx(np.sin(normalized_rad), abs=1e-6)


@given(quaternions())
def test_quaternion_to_euler_rpy_and_back(quat: Quaternion) -> None:
    """Verify that any Quaternion is unchanged after converting to and from Euler angles."""
    # Arrange/Act - Given a unit quaternion, convert to and from Euler RPY angles
    euler_rpy = quat.to_euler_rpy()
    result_quat = euler_rpy.to_quaternion()

    # Assert - Expect that the resulting quaternion equals the original (modulo negation)
    assert quat.approx_equal(result_quat)


@given(quaternions())
def test_quaternion_to_rotation_matrix_and_back(quat: Quaternion) -> None:
    """Verify that any Quaternion is unchanged after converting to and from a rotation matrix."""
    # Arrange/Act - Given a unit quaternion, convert to and from a rotation matrix
    r_matrix = quat.to_rotation_matrix()
    result_quat = Quaternion.from_rotation_matrix(r_matrix)

    # Assert - Expect that the resulting quaternion equals the original (modulo negation)
    assert quat.approx_equal(result_quat)


def test_zero_quaternion_raises_error() -> None:
    """Verify that attempting to construct an all-zero Quaternion raises a ValueError."""
    # Arrange/Act/Assert - Expect that constructing an all-zero Quaternion will raise an error
    with pytest.raises(ValueError, match="zero"):
        _ = Quaternion(0.0, 0.0, 0.0, 0.0)
