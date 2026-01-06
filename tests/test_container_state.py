"""Unit tests for the ContainerState class."""

from pathlib import Path

import pytest

from robotics_utils.kinematics import KinematicTree


@pytest.fixture
def container_env_yaml() -> Path:
    """Return a path to a YAML file specifying an environment with containers."""
    env_yaml = Path(__file__).parent / "test_data/yaml/filing_cabinets_env.yaml"
    assert env_yaml.exists(), f"Expected to find file: {env_yaml}"
    return env_yaml


def test_environment_with_containers_from_yaml(container_env_yaml: Path) -> None:
    """Verify that a KinematicTree including containers can be loaded from YAML."""
    tree = KinematicTree.from_yaml(container_env_yaml)

    assert len(tree.object_names) == 4  # Expect 4 objects (the fifth is inside a closed cabinet)
    assert len(tree.containers) == 3  # Expect three containers: two open and one closed

    closed_cabinet = tree.containers.get("closed_cabinet")
    assert closed_cabinet is not None
    assert closed_cabinet.is_closed
    assert "cup1" in closed_cabinet.contained_objects

    eraser_open_cabinet = tree.containers.get("open_cabinet1")
    assert eraser_open_cabinet is not None
    assert eraser_open_cabinet.is_open
    assert "eraser1" in eraser_open_cabinet.contained_objects

    empty_open_cabinet = tree.containers.get("open_cabinet2")
    assert empty_open_cabinet is not None
    assert empty_open_cabinet.is_open
    assert not empty_open_cabinet.contained_objects

    # Expect that `cup1` isn't in the kinematic state (it's in the closed cabinet)
    assert "cup1" not in tree.object_names
    assert "cup1" not in tree.frames
    assert "cup1" not in tree.collision_models
    assert "cup1" not in tree.children


def test_container_state_open_and_close(container_env_yaml: Path) -> None:
    """Verify that opening and closing containers appropriately affects the state."""
    tree = KinematicTree.from_yaml(container_env_yaml)

    # Act/Assert - Expect that objects in closed containers are removed from the kinematic state

    # Initial state:
    #   - Closed cabinet contains the cup
    #   - Open cabinet 1 contains the eraser
    #   - Open cabinet 2 is empty
    assert "cup1" not in tree.object_names
    assert "eraser1" in tree.object_names

    # Open the closed cabinet
    tree.open_container("closed_cabinet")
    assert "cup1" in tree.object_names
    assert "eraser1" in tree.object_names
    assert tree.containers["closed_cabinet"].is_open

    # Close the first open cabinet
    tree.close_container("open_cabinet1")
    assert "cup1" in tree.object_names
    assert "eraser1" not in tree.object_names
    assert tree.containers["open_cabinet1"].is_closed

    # Close the second open cabinet (no object states should change)
    tree.close_container("open_cabinet2")
    assert "cup1" in tree.object_names
    assert "eraser1" not in tree.object_names
    assert tree.containers["open_cabinet2"].is_closed

    # Re-close the originally closed cabinet
    tree.close_container("closed_cabinet")
    assert "cup1" not in tree.object_names
    assert "eraser1" not in tree.object_names
    assert tree.containers["closed_cabinet"].is_closed
