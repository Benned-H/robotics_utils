"""Unit tests to validate magic path constants."""

from robotics_utils.meta import ROBOTICS_UTILS_ROOT


def test_robotics_utils_root() -> None:
    """Verify that `ROBOTICS_UTILS_ROOT` points to the root of the repository."""
    # Act - Construct paths that should point to the project's pyproject.toml and src folder
    pyproject_toml = ROBOTICS_UTILS_ROOT / "pyproject.toml"
    src_dir = ROBOTICS_UTILS_ROOT / "src"

    # Assert - Expect to find the root directory
    assert ROBOTICS_UTILS_ROOT.is_dir()
    assert pyproject_toml.is_file()
    assert src_dir.is_dir()
