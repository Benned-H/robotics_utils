"""Example usage of the ObjectDetector class."""

from __future__ import annotations

import traceback
from pathlib import Path

import click
import cv2
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robotics_utils.vision.object_detector import ObjectDetector, visualize_detections
from robotics_utils.vision.vision_utils import load_image


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

    image = load_image(image_path)

    menu_table = Table(title="Menu Options", border_style="cyan", title_style="bold cyan")
    menu_table.add_column("Option", style="bold", width=4)
    menu_table.add_column("Description", style="white")

    menu_items = [
        ("1", "Add a text query"),
        ("2", "Call object detector"),
        ("3", "Quit"),
    ]

    for option, description in menu_items:
        menu_table.add_row(option, description)

    current_queries = []

    while True:
        console.print()
        console.print(menu_table)

        choice = click.prompt("\nSelect option", type=click.IntRange(1, 3))

        try:
            if choice == 1:
                query = click.prompt("Enter text query")
                current_queries.append(query)

                queries_str = "\n\t".join(current_queries)
                console.print(f"[green]✓[/green] Current pending queries:\n\t{queries_str}")

            elif choice == 2:
                console.print("[yellow]Calling object detector...[/yellow]")
                detections = detector.detect(image, current_queries)
                vis_image = visualize_detections(image, detections)

                # Convert back to BGR for OpenCV display
                vis_image_bgr = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)
                cv2.imshow("Detections (press any key to exit)", vis_image_bgr)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

                current_queries = []
                console.print("[green]✓[/green] Object detection completed")

            elif choice == 3:
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
