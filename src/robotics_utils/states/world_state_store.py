"""Define persistence helpers for runtime world-state overlays."""

from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from robotics_utils.io.world_state_checkpoint_schema import (
    GraspAttachmentSchema,
    WorldStateCheckpointSchema,
)
from robotics_utils.io.yaml_utils import export_yaml_data

if TYPE_CHECKING:
    from robotics_utils.states import ObjectCentricState


@dataclass(frozen=True)
class LoadCheckpointOutcome:
    """Result of attempting to load a world-state checkpoint."""

    checkpoint: WorldStateCheckpointSchema | None
    message: str


class WorldStateStore:
    """Store and retrieve runtime world-state checkpoints on disk."""

    def __init__(self, overlay_dir: Path, stale_after_s: float = 300.0) -> None:
        """Initialize the store with a base directory and staleness threshold."""
        self.overlay_dir = overlay_dir
        self.stale_after_s = stale_after_s

    @staticmethod
    def _sanitize_robot_key(robot_key: str) -> str:
        """Sanitize a robot key for use in filenames."""
        sanitized = re.sub(r"[^A-Za-z0-9_.-]+", "_", robot_key.strip())
        return sanitized.strip("_") or "robot"

    @staticmethod
    def _environment_hash(env_yaml: Path) -> str:
        """Compute a stable hash for an environment file path."""
        normalized = str(env_yaml.resolve()).encode("utf-8")
        return hashlib.sha1(normalized).hexdigest()[:10]

    def checkpoint_path(self, *, robot_key: str, baseline_env_yaml: Path) -> Path:
        """Return the checkpoint filepath for a robot/environment pair."""
        safe_robot_key = self._sanitize_robot_key(robot_key)
        env_hash = self._environment_hash(baseline_env_yaml)
        filename = f"{safe_robot_key}_{env_hash}_state_overlay.yaml"
        return self.overlay_dir / filename

    def clear_checkpoint(self, *, robot_key: str, baseline_env_yaml: Path) -> bool:
        """Delete an existing checkpoint file, if present."""
        checkpoint_path = self.checkpoint_path(
            robot_key=robot_key,
            baseline_env_yaml=baseline_env_yaml,
        )
        if not checkpoint_path.exists():
            return False

        checkpoint_path.unlink()
        return True

    def build_checkpoint(
        self,
        *,
        state: ObjectCentricState,
        robot_key: str,
        baseline_env_yaml: Path,
        saved_at_unix_s: float | None = None,
    ) -> WorldStateCheckpointSchema:
        """Build a checkpoint schema from the given runtime state."""
        baseline_env_yaml = baseline_env_yaml.resolve()
        baseline_env_mtime_s = baseline_env_yaml.stat().st_mtime
        saved_at_unix_s = time.time() if saved_at_unix_s is None else saved_at_unix_s

        default_frame = state.kinematic_tree.root_frame
        known_object_poses = {
            obj_name: pose.to_schema(default_frame=default_frame)
            for obj_name, pose in state.known_object_poses.items()
        }

        container_statuses = {
            container_name: container_state.status
            for container_name, container_state in state.containers.items()
        }

        sorted_grasps = sorted(
            state.grasp_attachments,
            key=lambda grasp: (grasp.robot_name, grasp.ee_link_name, grasp.obj_name),
        )
        active_grasps = [
            GraspAttachmentSchema(
                obj_name=grasp.obj_name,
                robot_name=grasp.robot_name,
                ee_link_name=grasp.ee_link_name,
                pose_ee_o=grasp.pose_ee_o.to_schema(default_frame=default_frame),
                touching_link_names=set(grasp.touching_link_names),
            )
            for grasp in sorted_grasps
        ]

        return WorldStateCheckpointSchema(
            schema_version=1,
            saved_at_unix_s=saved_at_unix_s,
            robot_key=robot_key,
            baseline_env_yaml=baseline_env_yaml,
            baseline_env_mtime_s=baseline_env_mtime_s,
            known_object_poses=known_object_poses,
            hidden_objects=sorted(state.hidden_object_names),
            container_statuses=container_statuses,
            active_grasps=active_grasps,
        )

    def save_checkpoint(
        self,
        *,
        state: ObjectCentricState,
        robot_key: str,
        baseline_env_yaml: Path,
        saved_at_unix_s: float | None = None,
    ) -> Path:
        """Persist a runtime state checkpoint and return its filepath."""
        checkpoint = self.build_checkpoint(
            state=state,
            robot_key=robot_key,
            baseline_env_yaml=baseline_env_yaml,
            saved_at_unix_s=saved_at_unix_s,
        )

        checkpoint_path = self.checkpoint_path(
            robot_key=robot_key,
            baseline_env_yaml=baseline_env_yaml,
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint_data = checkpoint.model_dump(mode="json")
        export_yaml_data(data=checkpoint_data, filepath=checkpoint_path)

        return checkpoint_path

    def load_checkpoint(
        self,
        *,
        robot_key: str,
        baseline_env_yaml: Path,
    ) -> LoadCheckpointOutcome:
        """Load a checkpoint for the robot/environment pair if it passes guardrails."""
        baseline_env_yaml = baseline_env_yaml.resolve()
        checkpoint_path = self.checkpoint_path(
            robot_key=robot_key,
            baseline_env_yaml=baseline_env_yaml,
        )

        if not checkpoint_path.exists():
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=f"No checkpoint found at '{checkpoint_path}'.",
            )

        try:
            checkpoint = WorldStateCheckpointSchema.validate_yaml(checkpoint_path)
        except Exception as err:  # noqa: BLE001
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=f"Unable to parse checkpoint '{checkpoint_path}': {err}",
            )

        if checkpoint.robot_key != robot_key:
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=(
                    f"Checkpoint robot key mismatch ('{checkpoint.robot_key}' vs expected "
                    f"'{robot_key}')."
                ),
            )

        if checkpoint.baseline_env_yaml.resolve() != baseline_env_yaml:
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=(
                    f"Checkpoint baseline path mismatch ('{checkpoint.baseline_env_yaml}' "
                    f"vs expected '{baseline_env_yaml}')."
                ),
            )

        current_baseline_mtime_s = baseline_env_yaml.stat().st_mtime
        if current_baseline_mtime_s > checkpoint.baseline_env_mtime_s:
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=(
                    f"Baseline environment '{baseline_env_yaml}' is newer than the checkpoint "
                    "metadata."
                ),
            )

        if self.stale_after_s >= 0 and (time.time() - checkpoint.saved_at_unix_s) > self.stale_after_s:
            return LoadCheckpointOutcome(
                checkpoint=None,
                message=(
                    f"Checkpoint at '{checkpoint_path}' is stale (older than "
                    f"{self.stale_after_s} seconds)."
                ),
            )

        return LoadCheckpointOutcome(
            checkpoint=checkpoint,
            message=f"Loaded checkpoint from '{checkpoint_path}'.",
        )
