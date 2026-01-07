"""Define a command-line interface for executing skills."""

from __future__ import annotations

import click
from prompt_toolkit import prompt as pt_prompt
from prompt_toolkit.completion import WordCompleter
from rich.padding import Padding
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from robotics_utils.io import console
from robotics_utils.io.cli_handlers import INPUT_HANDLERS, ParamUI, SkillsUI
from robotics_utils.skills import Skill, SkillsInventory, SkillsProtocol, find_default_param_values


def _idx_to_skill(inventory: SkillsInventory) -> dict[int, Skill]:
    """Assign indices (starting from 1) to the skills in the given inventory.

    :param inventory: Collection of skills assigned integer indices
    :return: Map from integer indices to Skill objects
    """
    sorted_names = sorted(skill.name for skill in inventory)

    return {idx + 1: inventory.skills[skill_name] for idx, skill_name in enumerate(sorted_names)}


def _render_selection_table(inv: SkillsInventory) -> Table:
    """Render a numbered table for CLI skill selection."""
    table = Table(title=f"Skills Inventory: {inv.name}", show_lines=False)
    table.add_column("#", justify="right", style="cyan", no_wrap=True)
    table.add_column("Skill", style="bold")
    table.add_column("Parameters", style="magenta")

    for idx, skill in _idx_to_skill(inv).items():
        params = ", ".join(f"{p.name}: {p.type_name}" for p in skill.parameters)
        table.add_row(str(idx), skill.name, params or "-")
    return table


def _prompt_for_bindings(skill: Skill, skills_ui: SkillsUI) -> dict[str, object]:
    """Prompt the user for parameter values for the given skill.

    :return: Map from skill parameter names to bound values
    :raises KeyError: If a skill parameter type is missing an input handler
    """
    console.print(Panel.fit(f"[bold]{skill.name}[/]", border_style="green"))
    bindings: dict[str, object] = {}

    # Fail early if a skill parameter's type doesn't have a handler
    for p in skill.parameters:
        if p.type_ not in skills_ui.handlers:
            raise KeyError(f"No input handler for type {p.type_name} (parameter '{p.name}').")

    for p in skill.parameters:
        handler = skills_ui.handlers[p.type_]
        prompt = p.name if p.semantics is None else f"{p.name}: {p.semantics}"
        param_ui = ParamUI(prompt, default=skills_ui.default_values.get((skill.name, p.name)))
        bindings[p.name] = handler(param_ui)

    return bindings


def build_cli(protocol_instance: SkillsProtocol) -> click.Command:
    """Create a Click command that exposes an interactive CLI for a skills inventory.

    :param protocol_instance: Instance of a protocol with methods defining available skills
    :return: A Click command that can be used as an entry point or subcommand
    """
    inventory = SkillsInventory.from_protocol(protocol_instance.__class__)
    idx_to_skill = _idx_to_skill(inventory)

    exit_options = {"q", "quit", "exit"}

    # Create a tab-completer for the skill names
    skill_names = sorted(skill.name for skill in inventory)
    skill_completer = WordCompleter([*skill_names, *exit_options], ignore_case=True)

    default_param_values = find_default_param_values(protocol_instance.__class__)
    skills_ui = SkillsUI(INPUT_HANDLERS, default_param_values)

    unhandled_types = inventory.all_argument_types - set(skills_ui.handlers.keys())
    unhandled_type_names = ", ".join(t.__name__ for t in unhandled_types)
    if unhandled_type_names:
        console.print(f"[yellow]Types missing input handlers: {unhandled_type_names}.[/]")

    @click.command()
    def cli() -> None:
        """Create an interactive CLI for invoking skills with Python-typed parameters."""
        while True:
            console.print(_render_selection_table(inventory))
            choice: str = pt_prompt(
                "Select a skill by name or number (or 'q' to quit): ",
                completer=skill_completer,
            ).strip()

            if choice.lower() in exit_options:
                console.print("[dim]Bye.[/]")
                break

            # Try to match by name first (case-insensitive)
            skill = inventory.skills.get(choice)

            # Fall back to number matching
            if skill is None and choice.isdigit():
                idx = int(choice)
                if idx < 1 or idx > len(inventory.skills):
                    console.print("[red]Out of range.[/]")
                    continue
                skill = idx_to_skill[idx]

            if skill is None:
                console.print("[red]Please enter a skill name, an integer, or 'q'.[/]")
                continue

            try:
                bindings = _prompt_for_bindings(skill, skills_ui)
            except KeyError as err:
                console.print(f"\n[red]{err}[/]")
                continue
            except KeyboardInterrupt:
                console.print("\n[yellow]Canceled.[/]")
                continue

            console.print(Padding(f"Bindings: {bindings}", (1, 4)))
            if Confirm.ask(f"[bold]Execute {skill.name}?[/]"):
                try:
                    outcome = skill.execute(protocol_instance, bindings)
                    color = "green" if outcome.success else "red"
                    console.print(f"[{color}]{outcome.message}[/]")
                except KeyboardInterrupt:
                    console.print("\n[yellow]Execution interrupted.[/]")

    return cli
