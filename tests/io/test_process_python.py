"""Unit tests for utility functions that process Python files."""

from robotics_utils.io.process_python import load_class_from_module
from robotics_utils.kinematics import Pose3D


def test_load_class_from_module() -> None:
    """Verify that an example class can be loaded from the appropriate module."""
    # Arrange - Define which class (`Pose3D`) should be imported from which module (`poses`)
    class_name = "Pose3D"
    module_name = "robotics_utils.kinematics.poses"

    # Act - Attempt to load the class from the module
    loaded_class = load_class_from_module(class_name, module_name)

    # Assert - Verify that the `Pose3D` class was successfully loaded
    assert loaded_class == Pose3D
