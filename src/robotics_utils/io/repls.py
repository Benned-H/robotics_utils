"""Define classes implementing common REPL utilities for the package."""

import traceback
from pathlib import Path
from typing import Callable, Generic, TypeVar

import click
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from robotics_utils.io.logging import console
from robotics_utils.vision import RGBImage
from robotics_utils.vision.vlms import TextQueries
from robotics_utils.visualization import Displayable

ResultT = TypeVar("ResultT", bound=Displayable)
"""Type resulting from a call to some type of computer vision model."""

ProcessQueryFunction = Callable[[RGBImage, list[str]], ResultT]
DisplayFunction = Callable[[ResultT], None]


class OpenVocabVisionREPL(Generic[ResultT]):
    """A read-eval-print loop to process open-vocabulary queries to a computer vision model.

    :param ResultT: Output type from the computer vision model when run on an image
    """

    def __init__(
        self,
        image_path: Path,
        process_func: ProcessQueryFunction,
        display_func: DisplayFunction,
        model_type: str,
    ) -> None:
        """Initialize the open-vocabulary computer vision REPL.

        :param image_path: Path to the image used for the computer vision task
        :param process_func: Function that processes a user query for the image
        :param display_func: Function that (optionally) displays the result
        :param model_type: Description of the computer vision model (e.g., "object detector")
        """
        self.image = RGBImage.from_file(image_path)
        self.process_func = process_func
        self.display_func = display_func
        self.model_type = model_type

        self.menu_table = Table(title="Menu Options", border_style="cyan", title_style="bold cyan")
        self.menu_table.add_column("Option", style="bold", width=8)
        self.menu_table.add_column("Description", style="white")

        menu_items = [
            ("1", "Add a text query"),
            ("2", "Remove a text query"),
            ("3", f"Call {self.model_type}"),
            ("4", "Quit"),
        ]

        for option, description in menu_items:
            self.menu_table.add_row(option, description)

        self.queries = TextQueries()

    def loop(self) -> None:
        """Enter the read-eval-print loop."""
        while True:
            console.print()
            console.print(self.menu_table)

            choice = click.prompt("\nSelect option", type=click.IntRange(1, 4))

            try:
                if choice == 1:
                    query: str = click.prompt("Enter text query (or multiple separated by commas)")
                    self.queries.add(query)
                    console.print(f"[green]✓[/green] Current pending queries:\n{self.queries}")

                elif choice == 2:
                    console.print(f"Current pending queries:\n{self.queries}")

                    remove_query: str = click.prompt("Enter text query to be removed").strip()
                    if remove_query and self.queries.remove(remove_query):
                        console.print(f"[green]Query '{remove_query}' was removed[/green]")
                    else:
                        console.print(f"[red]Could not remove query '{remove_query}'[/red]")

                elif choice == 3:
                    if not self.queries:
                        console.print(
                            f"[red]Cannot call the {self.model_type} without a query![/red]",
                        )
                        continue

                    console.print(f"[yellow]Calling {self.model_type}...[/yellow]")
                    result = self.process_func(self.image, list(self.queries))

                    self.display_func(result)

                    if click.confirm("Clear the current text queries?"):
                        self.queries.clear()

                    console.print(f"[green]✓[/green] Call to {self.model_type} completed")

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
                error_panel = Panel(
                    Text(f"❌ Error: {exc_text}", style="bold red"),
                    border_style="red",
                )
                console.print(error_panel)
