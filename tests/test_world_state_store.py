"""Unit tests for runtime world-state checkpoint persistence."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from robotics_utils.io.yaml_utils import export_yaml_data
from robotics_utils.spatial import Pose3D
from robotics_utils.states import GraspAttachment, ObjectCentricState, WorldStateStore


@pytest.fixture
def container_env_yaml() -> Path:
    """Return a YAML path for an environment containing openable containers."""
    yaml_path = Path(__file__).parent / "test_data/yaml/filing_cabinets_env.yaml"
    assert yaml_path.exists(), f"Expected test YAML to exist: {yaml_path}"
    return yaml_path


@pytest.fixture
def example_env_yaml() -> Path:
    """Return a second environment YAML for baseline mismatch tests."""
    yaml_path = Path(__file__).parent / "test_data/yaml/example_environment.yaml"
    assert yaml_path.exists(), f"Expected test YAML to exist: {yaml_path}"
    return yaml_path


def _make_state_with_runtime_changes(container_env_yaml: Path) -> ObjectCentricState:
    """Create a state with grasp/hidden/container runtime edits."""
    state = ObjectCentricState.from_yaml(container_env_yaml)

    state.add_end_effector(robot_name="spot", ee_link_name="gripper_link")
    grasp = GraspAttachment(
        obj_name="eraser1",
        robot_name="spot",
        ee_link_name="gripper_link",
        pose_ee_o=Pose3D.from_xyz_rpy(x=0.1, y=0.0, z=0.3, ref_frame="gripper_link"),
        touching_link_names={"gripper_link"},
    )
    state.attach_grasp(grasp)

    state.open_container("closed_cabinet")
    state.hide_object("cup1")

    return state


def test_world_state_store_round_trip(tmp_path: Path, container_env_yaml: Path) -> None:
    """Verify that persisted checkpoints round-trip all phase-1 runtime fields."""
    state = _make_state_with_runtime_changes(container_env_yaml)

    store = WorldStateStore(overlay_dir=tmp_path, stale_after_s=300.0)
    checkpoint_path = store.save_checkpoint(
        state=state,
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    assert checkpoint_path.exists()

    load_outcome = store.load_checkpoint(
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    assert load_outcome.checkpoint is not None

    checkpoint = load_outcome.checkpoint
    assert checkpoint.container_statuses["closed_cabinet"] == "open"
    assert "cup1" in checkpoint.hidden_objects
    assert "eraser1" in checkpoint.known_object_poses
    assert len(checkpoint.active_grasps) == 1
    assert checkpoint.active_grasps[0].obj_name == "eraser1"


def test_world_state_store_skips_stale_checkpoint(tmp_path: Path, container_env_yaml: Path) -> None:
    """Verify stale checkpoints are skipped based on configured TTL."""
    state = _make_state_with_runtime_changes(container_env_yaml)

    store = WorldStateStore(overlay_dir=tmp_path, stale_after_s=5.0)
    store.save_checkpoint(
        state=state,
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
        saved_at_unix_s=time.time() - 30.0,
    )

    load_outcome = store.load_checkpoint(
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    assert load_outcome.checkpoint is None
    assert "stale" in load_outcome.message.lower()


def test_world_state_store_skips_baseline_path_mismatch(
    tmp_path: Path,
    container_env_yaml: Path,
    example_env_yaml: Path,
) -> None:
    """Verify checkpoints are skipped if metadata baseline path mismatches requested baseline."""
    state = _make_state_with_runtime_changes(container_env_yaml)

    store = WorldStateStore(overlay_dir=tmp_path, stale_after_s=300.0)
    checkpoint = store.build_checkpoint(
        state=state,
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    tampered_checkpoint = checkpoint.model_copy(update={"baseline_env_yaml": example_env_yaml.resolve()})

    checkpoint_path = store.checkpoint_path(
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    export_yaml_data(
        data=tampered_checkpoint.model_dump(mode="json"),
        filepath=checkpoint_path,
    )

    load_outcome = store.load_checkpoint(
        robot_key="spot-host",
        baseline_env_yaml=container_env_yaml,
    )
    assert load_outcome.checkpoint is None
    assert "baseline path mismatch" in load_outcome.message.lower()
