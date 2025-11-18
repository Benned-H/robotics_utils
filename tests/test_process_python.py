"""Unit tests for utility functions that process Python files."""

from __future__ import annotations

from robotics_utils.kinematics import Pose3D
from robotics_utils.meta import get_default_values, load_class_from_module


def test_load_class_from_module() -> None:
    """Verify that an example class can be loaded from the appropriate module."""
    # Arrange - Define which class (`Pose3D`) should be imported from which module (`poses`)
    class_name = "Pose3D"
    module_name = "robotics_utils.kinematics.poses"

    # Act - Attempt to load the class from the module
    loaded_class = load_class_from_module(class_name, module_name)

    # Assert - Verify that the `Pose3D` class was successfully loaded
    assert loaded_class == Pose3D


def test_get_default_values() -> None:
    """Verify that default function arguments can be identified."""
    # Arrange - Create example functions with some arguments that have default values

    def function_one(a: int, b: float = 10.0, c: str = "default value") -> bool:
        """Define an example function with default argument values."""
        return True

    def function_two(x: float | None = None, y: int | None = 5, z: type | None = str) -> None:
        """Define another example function with default argument values."""

    # Act - Get the default argument values
    one_defaults = get_default_values(function=function_one)
    two_defaults = get_default_values(function=function_two)

    # Assert - Expect that the correct default values were found
    assert one_defaults == {"b": 10.0, "c": "default value"}
    assert two_defaults == {"x": None, "y": 5, "z": str}
