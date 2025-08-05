"""Define utility functions for importing and exporting to/from YAML files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


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
