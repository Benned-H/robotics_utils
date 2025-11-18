"""Unit tests for Point3D, a class representing 3D positions."""

from pathlib import Path

import pytest
from hypothesis import given

from robotics_utils.kinematics import Point3D

from .strategies.kinematics_strategies import positions


@given(positions())
def test_point3d_to_array_and_back(point: Point3D) -> None:
    """Verify that Point3Ds correctly convert to and from NumPy arrays."""
    # Arrange/Act - Given a Point3D, convert into a NumPy array and then back
    point_arr = point.to_array()
    result_point = Point3D.from_array(point_arr)

    # Assert - Expect that the resulting point exactly equals the original
    assert point == result_point


@pytest.fixture
def points_yaml() -> Path:
    """Return a path to a YAML file specifying an example sequence of 3D points."""
    yaml_path = Path(__file__).parent / "test_data/yaml/example_points.yaml"
    assert yaml_path.exists(), f"Expected to find file: {yaml_path}"
    return yaml_path


def test_load_points_from_yaml(points_yaml: Path) -> None:
    """Verify that a sequence of 3D points can be loaded from YAML."""
    # Act/Assert - Attempt to load the points and ensure that there are as many as expected
    points = Point3D.load_points_from_yaml(points_yaml, collection_name="points")

    assert len(points) == 4
    for p in points:
        assert isinstance(p, Point3D)
