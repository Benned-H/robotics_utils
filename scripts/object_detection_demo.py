"""Example usage of the ObjectDetector class."""

from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path

import click
import cv2
import numpy as np
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robotics_utils.vision.images import RGBImage
from robotics_utils.vision.object_detector import ObjectDetector, TextQueries


@click.group()
@click.option("--model-name", "model_name", help="Name of the OWL-ViT model to load")
@click.pass_context
def object_detection_cli(ctx: click.Context, model_name: str | None) -> None:
    """Run object detection through a command-line interface."""
    ctx.ensure_object(dict)  # Create ctx.obj if it doesn't exist
    ctx.obj["detector"] = ObjectDetector() if model_name is None else ObjectDetector(model_name)
    ctx.obj["console"] = Console()


@object_detection_cli.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def interactive(ctx: click.Context, image_path: Path) -> None:
    """Run object detection in an interactive loop."""
    detector: ObjectDetector = ctx.obj["detector"]
    console: Console = ctx.obj["console"]

    image = RGBImage.from_file(image_path)

    menu_table = Table(title="Menu Options", border_style="cyan", title_style="bold cyan")
    menu_table.add_column("Option", style="bold", width=8)
    menu_table.add_column("Description", style="white")

    menu_items = [
        ("1", "Add a text query"),
        ("2", "Remove a text query"),
        ("3", "Call object detector"),
        ("4", "Quit"),
    ]

    for option, description in menu_items:
        menu_table.add_row(option, description)

    current_queries = TextQueries()

    while True:
        console.print()
        console.print(menu_table)

        choice = click.prompt("\nSelect option", type=click.IntRange(1, 4))

        try:
            if choice == 1:
                query: str = click.prompt("Enter text query (or multiple separated by commas)")
                current_queries.add(query)
                console.print(f"[green]✓[/green] Current pending queries:\n{current_queries}")

            elif choice == 2:
                console.print(f"Current pending queries:\n{current_queries}")

                remove_query: str = click.prompt("Enter text query to be removed").strip()
                if remove_query and current_queries.remove(remove_query):
                    console.print(f"[green]Query '{remove_query}' was removed[/green]")
                else:
                    console.print(f"[red]Could not remove query '{remove_query}'[/red]")

            elif choice == 3:
                if not current_queries:
                    console.print("[red]Cannot call the object detector without a query![/red]")
                    continue

                console.print("[yellow]Calling object detector...[/yellow]")
                detections = detector.detect(image, current_queries)

                rng = np.random.default_rng()
                query_colors: dict[str, tuple] = {}
                for q in current_queries:
                    query_colors[q] = tuple(int(n) for n in rng.integers(0, 255, size=3))

                vis_image = deepcopy(image)
                for detection in detections:
                    color = query_colors[detection.query]
                    detection.draw(vis_image, color)

                vis_image.visualize("Detections (press any key to exit)")

                for i, detection in enumerate(detections):
                    cropped = detection.bounding_box.crop(image, scale_box=1.2)
                    cropped.visualize(f"Detection {i}/{len(detections)}: '{detection.query}'")

                if click.confirm("Clear the current text queries?"):
                    current_queries.clear()

                console.print("[green]✓[/green] Object detection completed")

            elif choice == 4:
                goodbye_panel = Panel(
                    Text("👋 Goodbye!", style="bold green", justify="center"),
                    border_style="green",
                )
                console.print(goodbye_panel)
                break

        except click.Abort:
            console.print("[yellow]⚠️ Operation canceled[/yellow]")

        except Exception:
            exc_text = traceback.format_exc()
            error_panel = Panel(Text(f"❌ Error: {exc_text}", style="bold red"), border_style="red")
            console.print(error_panel)


if __name__ == "__main__":
    object_detection_cli()
