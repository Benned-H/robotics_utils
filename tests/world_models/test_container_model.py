"""Unit tests for the ContainerModel class."""

from pathlib import Path

import pytest

from robotics_utils.kinematics.kinematic_tree import KinematicTree


@pytest.fixture
def container_env_yaml() -> Path:
    """Specify a path to an example YAML file representing an environment with a container."""
    env_yaml = Path(__file__).parent.parent / "test_data/container_env.yaml"
    assert env_yaml.exists(), f"Expected to find file: {env_yaml}"
    return env_yaml


def test_environment_with_containers_from_yaml(container_env_yaml: Path) -> None:
    """Verify that a KinematicTree including containers can be loaded from YAML."""
    tree = KinematicTree.from_yaml(container_env_yaml)

    # Assert - Expect that the tree contains only two objects (i.e., the two containers)
    assert len(tree.object_names) == 2
    assert len(tree.container_models) == 2  # Expect two containers: one empty and one closed

    closed_cabinet = tree.container_models.get("closed_cabinet")
    assert closed_cabinet is not None
    assert closed_cabinet.is_closed

    open_cabinet = tree.container_models.get("open_cabinet")
    assert open_cabinet is not None
    assert open_cabinet.is_open

    # Expect that `cup1` isn't in the kinematic state (it's in the closed cabinet)
    assert "cup1" not in tree.object_names
    assert "cup1" not in tree.frames
    assert "cup1" not in tree.collision_models
