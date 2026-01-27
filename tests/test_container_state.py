"""Unit tests for the ContainerState class."""

from pathlib import Path

import pytest

from robotics_utils.states import ObjectCentricState


@pytest.fixture
def container_env_yaml() -> Path:
    """Return a path to a YAML file specifying an environment with containers."""
    env_yaml = Path(__file__).parent / "test_data/yaml/filing_cabinets_env.yaml"
    assert env_yaml.exists(), f"Expected to find file: {env_yaml}"
    return env_yaml


def test_environment_with_containers_from_yaml(container_env_yaml: Path) -> None:
    """Verify that a ObjectCentricState including containers can be loaded from YAML."""
    state = ObjectCentricState.from_yaml(container_env_yaml)

    assert len(state.object_names) == 5  # Expect 5 total objects in the scene
    assert len(state.containers) == 3  # Expect three containers: two open and one closed

    closed_cabinet = state.containers.get("closed_cabinet")
    assert closed_cabinet is not None
    assert closed_cabinet.is_closed
    assert closed_cabinet.contains("cup1")

    eraser_open_cabinet = state.containers.get("open_cabinet1")
    assert eraser_open_cabinet is not None
    assert eraser_open_cabinet.is_open
    assert eraser_open_cabinet.contains("eraser1")

    empty_open_cabinet = state.containers.get("open_cabinet2")
    assert empty_open_cabinet is not None
    assert empty_open_cabinet.is_open

    # Expect that the names and poses of both contained objects are specified in the state
    assert "cup1" in state.object_names
    assert "cup1" in state.object_poses
    assert "eraser1" in state.object_names
    assert "eraser1" in state.object_poses


def test_container_state_open_and_close(container_env_yaml: Path) -> None:
    """Verify that opening and closing containers appropriately affects the state."""
    state = ObjectCentricState.from_yaml(container_env_yaml)

    # Act/Assert - Expect that contained objects' poses are always known

    # Initial state:
    #   - Closed cabinet contains the cup
    #   - Open cabinet 1 contains the eraser
    #   - Open cabinet 2 is empty
    cup1_closed = state.get_object_pose("cup1")
    assert cup1_closed is not None

    eraser1_open = state.get_object_pose("eraser1")
    assert eraser1_open is not None

    # Open the closed cabinet, which is expected to change the pose of contained objects (cup1)
    state.open_container("closed_cabinet")
    cup1_open = state.get_object_pose("cup1")
    assert cup1_open is not None
    assert not cup1_open.approx_equal(cup1_closed)
    assert state.containers["closed_cabinet"].is_open

    # Close the first open cabinet, which is expected to change the pose of eraser1
    state.close_container("open_cabinet1")
    eraser1_closed = state.get_object_pose("eraser1")
    assert eraser1_closed is not None
    assert not eraser1_closed.approx_equal(eraser1_open)
    assert state.containers["open_cabinet1"].is_closed

    # Close the second open cabinet (no object states should change)
    state.close_container("open_cabinet2")
    cup1_still_open = state.get_object_pose("cup1")
    assert cup1_still_open == cup1_open
    eraser1_still_closed = state.get_object_pose("eraser1")
    assert eraser1_still_closed == eraser1_closed
    assert state.containers["open_cabinet2"].is_closed

    # Re-close the originally closed cabinet (its contained objects' poses should revert back)
    state.close_container("closed_cabinet")
    cup1_closed_final = state.get_object_pose("cup1")
    assert cup1_closed_final is not None
    assert cup1_closed_final == cup1_closed
    assert state.containers["closed_cabinet"].is_closed
