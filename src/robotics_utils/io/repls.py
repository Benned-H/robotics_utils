"""Define classes implementing common REPL utilities for the package."""

import traceback
from pathlib import Path
from typing import Callable, Generic, TypeVar

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robotics_utils.perception.vision import RGBImage
from robotics_utils.perception.vision.vlms import TextQueries
from robotics_utils.visualization import Displayable

ResultT = TypeVar("ResultT", bound=Displayable)
"""Type resulting from an object detection call."""

DetectFunction = Callable[[RGBImage, list[str], Console | None], ResultT]
DisplayFunction = Callable[[Console, ResultT], None]


class ObjectDetectionREPL(Generic[ResultT]):
    """A read-eval-print loop to process user queries for an object detector."""

    def __init__(
        self,
        console: Console,
        image_path: Path,
        detect_func: DetectFunction,
        display_func: DisplayFunction,
    ) -> None:
        """Initialize the object detection REPL and enter the loop.

        :param console: CLI console used to interact with the user
        :param image_path: Path to the image used for object detection
        :param detect_func: Function that calls the object detector and returns the result
        :param display_func: Function that (optionally) displays the result using the console
        """
        self.console = console
        self.image = RGBImage.from_file(image_path)
        self.detect_func = detect_func
        self.display_func = display_func

        self.menu_table = Table(title="Menu Options", border_style="cyan", title_style="bold cyan")
        self.menu_table.add_column("Option", style="bold", width=8)
        self.menu_table.add_column("Description", style="white")

        menu_items = [
            ("1", "Add a text query"),
            ("2", "Remove a text query"),
            ("3", "Call object detector"),
            ("4", "Quit"),
        ]

        for option, description in menu_items:
            self.menu_table.add_row(option, description)

        self.queries = TextQueries()

    def loop(self) -> None:
        """Enter the read-eval-print loop."""
        while True:
            self.console.print()
            self.console.print(self.menu_table)

            choice = click.prompt("\nSelect option", type=click.IntRange(1, 4))

            try:
                if choice == 1:
                    query: str = click.prompt("Enter text query (or multiple separated by commas)")
                    self.queries.add(query)
                    self.console.print(f"[green]✓[/green] Current pending queries:\n{self.queries}")

                elif choice == 2:
                    self.console.print(f"Current pending queries:\n{self.queries}")

                    remove_query: str = click.prompt("Enter text query to be removed").strip()
                    if remove_query and self.queries.remove(remove_query):
                        self.console.print(f"[green]Query '{remove_query}' was removed[/green]")
                    else:
                        self.console.print(f"[red]Could not remove query '{remove_query}'[/red]")

                elif choice == 3:
                    if not self.queries:
                        self.console.print(
                            "[red]Cannot call the object detector without a query![/red]",
                        )
                        continue

                    self.console.print("[yellow]Calling object detector...[/yellow]")
                    detected = self.detect_func(self.image, list(self.queries), self.console)

                    self.display_func(self.console, detected)

                    if click.confirm("Clear the current text queries?"):
                        self.queries.clear()

                    self.console.print("[green]✓[/green] Object detection completed")

                elif choice == 4:
                    goodbye_panel = Panel(
                        Text("👋 Goodbye!", style="bold green", justify="center"),
                        border_style="green",
                    )
                    self.console.print(goodbye_panel)
                    break

            except click.Abort:
                self.console.print("[yellow]⚠️ Operation canceled[/yellow]")

            except Exception:
                exc_text = traceback.format_exc()
                error_panel = Panel(
                    Text(f"❌ Error: {exc_text}", style="bold red"),
                    border_style="red",
                )
                self.console.print(error_panel)
