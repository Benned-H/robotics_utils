"""Visualize collision models loaded from YAML files."""

from pathlib import Path

import click
from rich.console import Console

from robotics_utils.filesystem.yaml_utils import load_collision_model
from robotics_utils.kinematics.collision_models import NullSimplifier
from robotics_utils.visualization.viz_utils import visualize_collision_model


@click.command()
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_name")
def visualize(yaml_path: Path, model_name: str) -> None:
    """Visualize the named collision model imported from the given YAML file.

    :param yaml_path: YAML file from which the collision model is imported
    :param model_name: Name of the collision model to be imported
    """
    console = Console()
    console.print(f"[yellow]Loading collision model {model_name} from {yaml_path}[/yellow]...")
    collision_model = load_collision_model(model_name, yaml_path, simplifier=NullSimplifier())
    console.print(
        f"[green]Successfully loaded collision model with {len(collision_model.meshes)} "
        f"meshes and {len(collision_model.primitives)} primitive shapes.[/green]",
    )

    visualize_collision_model(collision_model)


if __name__ == "__main__":
    visualize()
