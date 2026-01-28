"""Unit tests to validate viewpoints."""

from pathlib import Path

import pytest

from robotics_utils.states import ViewpointTemplates


@pytest.fixture
def viewpoints_yaml() -> Path:
    """Return a path to a YAML file specifying camera viewpoints."""
    yaml_path = Path(__file__).parent / "test_data/yaml/example_viewpoints.yaml"
    assert yaml_path.exists(), f"Expected to find file: {yaml_path}"
    return yaml_path
