"""Define utility functions for importing from and exporting to YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def export_yaml_data(data: dict[str, Any] | list[Any], filepath: Path) -> None:
    """Output the given dictionary of YAML data to the given file."""
    yaml_string = yaml.dump(data, sort_keys=True, default_flow_style=True)

    with filepath.open("w") as file:
        file.write(yaml_string)
        file.close()

    if not filepath.exists():
        raise FileNotFoundError(f"Exported to YAML file '{filepath}' yet it doesn't exist")


def load_yaml_into_dict(yaml_path: Path) -> Any:
    """Load data from a YAML file into a Python dictionary.

    :param yaml_path: Path to the YAML file to be imported
    :return: Dictionary mapping strings to values, or a list of dictionaries, etc.
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"Cannot load data from nonexistent YAML file: {yaml_path}")

    try:
        with yaml_path.open() as yaml_file:
            yaml_data: dict[str, Any] = yaml.safe_load(yaml_file)
    except yaml.YAMLError as error:
        raise RuntimeError(f"Failed to load YAML file: {yaml_path}") from error

    return yaml_data
