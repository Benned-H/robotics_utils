"""Visualize collision models loaded from YAML files."""

from pathlib import Path

import click

from robotics_utils.filesystem.yaml_utils import load_collision_model
from robotics_utils.kinematics.collision_models import NullSimplifier


@click.command()
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_name")
def visualize(yaml_path: Path, model_name: str) -> None:
    """Visualize the named collision model imported from the given YAML file.

    :param yaml_path: YAML file from which the collision model is imported
    :param model_name: Name of the collision model to be imported
    """
    collision_model = load_collision_model(model_name, yaml_path, simplifier=NullSimplifier())
    collision_model.visualize()


if __name__ == "__main__":
    visualize()
