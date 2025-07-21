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
