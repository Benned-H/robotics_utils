"""Visualize collision models loaded from YAML files."""

from pathlib import Path

import click
from rich.console import Console

from robotics_utils.filesystem.mesh_utils import load_mesh
from robotics_utils.filesystem.yaml_utils import load_collision_model
from robotics_utils.kinematics.collision_models import MeshData
from robotics_utils.visualization.viz_utils import visualize_collision_model


@click.group()
@click.pass_context
def visualize_cli(ctx: click.Context) -> None:
    """Visualize collision models or meshes through a command-line interface."""
    ctx.ensure_object(dict)  # Create ctx.obj if it doesn't exist
    ctx.obj["console"] = Console()


@visualize_cli.command()
@click.argument("yaml_path", type=click.Path(exists=True, path_type=Path))
@click.argument("model_name")
@click.pass_context
def collision_model(ctx: click.Context, yaml_path: Path, model_name: str) -> None:
    """Visualize a collision model loaded from a YAML file.

    :param ctx: Context object providing access to a CLI console
    :param yaml_path: YAML file from which the collision model is imported
    :param model_name: Name of the collision model to be imported
    """
    console: Console = ctx.obj["console"]
    console.print(f"[yellow]Loading collision model {model_name} from {yaml_path}...[/yellow]")

    collision_model = load_collision_model(model_name, yaml_path, simplifier=None)

    console.print(
        f"[green]Successfully loaded collision model with {len(collision_model.meshes)} "
        f"meshes and {len(collision_model.primitives)} primitive shapes[/green]",
    )

    visualize_collision_model(collision_model)


@visualize_cli.command()
@click.argument("mesh_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def mesh(ctx: click.Context, mesh_path: Path) -> None:
    """Visualize a mesh loaded from file.

    :param ctx: Context object providing access to a CLI console
    :param mesh_path: Path to a mesh file
    """
    console: Console = ctx.obj["console"]
    console.print(f"[yellow]Loading mesh from {mesh_path}...[/yellow]")

    mesh = load_mesh(mesh_path)
    mesh.show()


if __name__ == "__main__":
    visualize_cli()
