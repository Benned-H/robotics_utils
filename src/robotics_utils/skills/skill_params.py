"""Define a class representing skill-specific parameters for objects in a domain."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from robotics_utils.io.yaml_utils import load_yaml_data

if TYPE_CHECKING:
    from pathlib import Path


class SkillParams:
    """Per-skill object-specific parameters loaded from a domain YAML file.

    Structure in YAML:

        skill_params:
          pick:
            eraser1:
              grasp_pose: [x, y, z, roll, pitch, yaw]   # in object frame by default
              pre_grasp_x_m: 0.15
              lift_z_m: 0.15
          estimate-pose:
            eraser1:
              viewpoint_pose: [x, y, z, roll, pitch, yaw]   # in object frame by default

    Access skill parameters via `skill_params["pick"]["eraser1"]`, giving a dictionary.
    """

    def __init__(self, data: dict[str, dict[str, dict[str, Any]]]) -> None:
        """Initialize from a nested dict: skill -> object -> params.

        :param data: Raw dictionary containing skill parameters loaded from YAML
        """
        self._data = data

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> SkillParams:
        """Load a SkillParams instance from a TAMP environment YAML file.

        Returns an empty SkillParams if the file contains no `skill_params` key.

        :param yaml_path: Path to an environment YAML file
        :return: Constructed SkillParams instance
        """
        yaml_data: dict[str, Any] = load_yaml_data(yaml_path)
        data: dict[str, dict[str, dict[str, Any]]] = yaml_data.get("skill_params", {})
        return cls(data)

    def __getitem__(self, skill_name: str) -> dict[str, dict[str, Any]]:
        """Return the per-object parameter dictionary for the given skill name."""
        return self._data[skill_name]

    def get(self, skill_name: str, object_name: str) -> dict[str, Any] | None:
        """Return the parameter dictionary for the given skill and object, else None."""
        return self._data.get(skill_name, {}).get(object_name)

    def __contains__(self, skill_name: str) -> bool:
        """Return True if the skill name has any parameters."""
        return skill_name in self._data

    def __repr__(self) -> str:
        """Retrieve a human-readable representation of the skill parameters."""
        skills = list(self._data.keys())
        return f"SkillParams(skills={skills})"
