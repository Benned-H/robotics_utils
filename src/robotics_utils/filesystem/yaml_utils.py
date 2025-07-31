"""Define utility functions for importing from and exporting to YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.collision_models import Mesh
from robotics_utils.kinematics.poses import Pose3D


def export_yaml_data(data: dict[str, Any] | list[Any], filepath: Path) -> None:
    """Output the given YAML data to the given file."""
    yaml_string = yaml.dump(data, sort_keys=True, default_flow_style=True)

    with filepath.open("w") as file:
        file.write(yaml_string)
        file.close()

    if not filepath.exists():
        raise FileNotFoundError(f"Exported to YAML file '{filepath}' yet it doesn't exist")


def load_yaml_data(yaml_path: Path, required_keys: set[str] | None = None) -> Any:
    """Load data from a YAML file into Python data structures.

    :param yaml_path: Path to the YAML file to be imported
    :param required_keys: Set of keys required to exist in the loaded data (if None, ignored)
    :return: Dictionary mapping strings to values, or a list of dictionaries, etc.
    :raises KeyError: If a required key is missing in the loaded data
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot load data from nonexistent YAML file: {yaml_path}")

    try:
        with yaml_path.open() as yaml_file:
            yaml_data: dict | list = yaml.safe_load(yaml_file)
    except yaml.YAMLError as error:
        raise RuntimeError(f"Failed to load from YAML file: {yaml_path}") from error

    if required_keys is not None:
        for key in required_keys:
            if key not in yaml_data:
                raise KeyError(f"Required key '{key}' was missing in data loaded from {yaml_path}")

    return yaml_data


def load_named_poses(yaml_path: Path, collection_name: str) -> dict[str, Pose3D]:
    """Load a collection of named poses from the given YAML file.

    :param yaml_path: Path to a YAML file containing pose data
    :param collection_name: Name of the collection of poses to be imported (e.g., "object_poses")
    :return: Dictionary mapping pose-frame names to their imported 3D poses
    """
    yaml_data = load_yaml_data(yaml_path, required_keys={collection_name})
    default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)
    poses_data: dict[str, Any] = yaml_data[collection_name]

    return {
        pose_name: Pose3D.from_yaml_data(pose_data, default_frame)
        for pose_name, pose_data in poses_data.items()
    }


def load_named_mesh(mesh_key: str, yaml_path: Path) -> Mesh:
    """Load the specified mesh from the given YAML file.

    :param mesh_key: YAML key used to access the imported mesh
    :param yaml_path: Path to a YAML file specifying mesh data
    :return: Constructed Mesh instance
    """
    yaml_data = load_yaml_data(yaml_path, required_keys={"meshes"})
    mesh_data = yaml_data["meshes"].get(mesh_key)
    if mesh_data is None:
        raise KeyError(f"Could not find mesh named '{mesh_key}' in YAML file {yaml_path}")

    return Mesh.from_yaml_data(mesh_data)
