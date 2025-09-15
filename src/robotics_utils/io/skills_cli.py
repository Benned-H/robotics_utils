"""Define a command-line interface for executing skills from a skills inventory."""

from __future__ import annotations

import click
from rich.console import Console
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from robotics_utils.io.cli_handlers import ParamUI, SkillsUI
from robotics_utils.skills import Skill, SkillsInventory, SkillsProtocol
from robotics_utils.skills.protocols.spot_skills import SkillResult


def _render_inventory_table(inv: SkillsInventory) -> Table:
    """Render a numbered skills table for skill selection."""
    table = Table(title=f"Skills Inventory: {inv.name}", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Skill", style="bold")
    table.add_column("Parameters", style="magenta")

    for idx, skill in enumerate(inv, start=1):
        params = ", ".join(f"{p.name}: {p.type_name}" for p in skill.parameters)
        table.add_row(str(idx), skill.name, params or "—")
    return table


def _prompt_for_bindings(console: Console, skill: Skill, skills_ui: SkillsUI) -> dict[str, object]:
    """Prompt the user for each parameter of the given skill.

    :return: Map from skill parameter names to bound objects
    :raises KeyError: If a skill parameter type is missing an input handler
    """
    console.print(Panel.fit(f"[bold]{skill.name}[/bold]", border_style="green"))
    bindings: dict[str, object] = {}

    for p in skill.parameters:
        handler = skills_ui.handlers.get(p.type_)
        if handler is None:
            raise KeyError(
                f"No input handler registered for type {p.type_name} (parameter '{p.name}').",
            )

        p_label = p.name if p.semantics is None else f"{p.name}: {p.semantics}"
        ui = skills_ui.param_overrides.get((skill.name, p.name), ParamUI(label=p_label))
        bindings[p.name] = handler(ui, console)

    return bindings


def build_cli(protocol: SkillsProtocol[SkillResult], skills_ui: SkillsUI) -> click.Command:
    """Create a Click command that exposes a Rich-driven interactive CLI for a skills inventory.

    :param protocol: Class with methods defining the structure of available skills
    :param skills_ui: User interface used to create the CLI for the skills
    :return: A Click command that can be used as an entry point or subcommand
    """
    inventory = SkillsInventory.from_protocol(protocol)
    idx_to_skill: dict[int, Skill] = dict(enumerate(inventory, start=1))

    console = Console()

    # We don't fail here; only error once we actually try to run a skill missing a type handler
    unhandled_types = inventory.all_argument_types - set(skills_ui.handlers.keys())
    unhandled_type_names = ", ".join(t.__name__ for t in unhandled_types)
    if unhandled_type_names:
        console.print(f"[yellow]Types missing input handlers: {unhandled_type_names}.[/yellow]")

    @click.command(context_settings={"help_option_names": ["-h", "--help"]})
    @click.option("--yes", is_flag=True, help="Skip confirmation prompts.")
    def cli(yes: bool) -> None:
        """Create an interactive CLI for invoking skills with typed parameters."""
        while True:
            console.print(_render_inventory_table(inventory))
            choice = Prompt.ask("Select a skill number (or 'q' to quit)").strip().lower()
            if choice in {"q", "quit", "exit"}:
                console.print("[dim]Bye.[/dim]")
                break

            if not choice.isdigit():
                console.print("[red]Please enter a number or 'q'.[/red]")
                continue

            idx = int(choice)
            if idx < 1 or idx > len(inventory.skills):
                console.print("[red]Out of range.[/red]")
                continue

            skill = idx_to_skill[idx]
            try:
                bindings = _prompt_for_bindings(console, skill, skills_ui)
            except KeyError as err:
                console.print(f"[red]{err}[/red]")
                continue

            console.print(Padding(f"Bindings: {bindings}", (0, 1)))
            if yes or Confirm.ask(f"[bold]Execute {skill.name}?[/]"):
                result: SkillResult = skill.execute(protocol, bindings)
                success, message = result
                if success:
                    console.print(f"[green]{message}[/green]")
                else:
                    console.print(f"[red]{message}[/red]")

            if not Confirm.ask("Invoke another skill?", default=True):
                console.print("[dim]Bye.[/dim]")
                break

    return cli
