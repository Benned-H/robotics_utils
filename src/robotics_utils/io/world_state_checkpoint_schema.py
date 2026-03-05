"""Define Pydantic schemata for persisted world-state checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Set

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from robotics_utils.io.pydantic_schemata import Pose3DSchema
from robotics_utils.io.yaml_utils import load_yaml_data


class GraspAttachmentSchema(BaseModel):
    """Schema for a persisted object-to-end-effector grasp attachment."""

    obj_name: str
    robot_name: str
    ee_link_name: str
    pose_ee_o: Pose3DSchema
    touching_link_names: Set[str] = Field(default_factory=set)

    model_config = ConfigDict(extra="forbid")


class WorldStateCheckpointSchema(BaseModel):
    """Schema for persisted runtime world state overlaid on a baseline environment YAML."""

    schema_version: Literal[1]
    saved_at_unix_s: float
    robot_key: str
    baseline_env_yaml: Path
    baseline_env_mtime_s: float

    known_object_poses: Dict[str, Pose3DSchema] = Field(default_factory=dict)
    hidden_objects: List[str] = Field(default_factory=list)
    container_statuses: Dict[str, Literal["open", "closed"]] = Field(default_factory=dict)
    active_grasps: List[GraspAttachmentSchema] = Field(default_factory=list)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def validate_yaml(cls, yaml_path: Path) -> WorldStateCheckpointSchema:
        """Validate a checkpoint YAML file and return a parsed schema instance."""
        yaml_data = load_yaml_data(yaml_path)

        try:
            return WorldStateCheckpointSchema.model_validate(yaml_data)
        except ValidationError as v_err:
            raise RuntimeError(f"Validation error in {yaml_path}: {v_err}") from v_err
