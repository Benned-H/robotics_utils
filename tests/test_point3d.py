"""Unit tests for Point3D, a class representing 3D positions."""

from hypothesis import given

from robotics_utils.kinematics.point3d import Point3D

from .hypothesis_strategies import positions


@given(positions())
def test_point3d_to_array_and_back(point: Point3D) -> None:
    """Verify that Point3Ds correctly convert to and from NumPy arrays."""
    # Arrange/Act - Given a Point3D, convert into a NumPy array and then back
    point_arr = point.to_array()
    result_point = Point3D.from_array(point_arr)

    # Assert - Expect that the resulting point exactly equals the original
    assert point == result_point
