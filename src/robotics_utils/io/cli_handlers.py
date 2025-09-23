"""Define functions creating CLI handlers for different input types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Generic, Mapping, Optional, Tuple, TypeVar

import click
from rich.console import Console
from rich.prompt import FloatPrompt, IntPrompt, Prompt

from robotics_utils.kinematics import Pose3D
from robotics_utils.skills.skill_templates import OpenDrawerTemplate, PickTemplate, PlaceTemplate

InputT = TypeVar("InputT")
"""An input type being requested via CLI."""

Validator = Callable[[InputT], Optional[str]]
"""An input validator where None = OK or string = error message."""


@dataclass(frozen=True)
class ParamUI(Generic[InputT]):
    """Per-parameter validation overlay passed to an input handler."""

    label: str
    default: InputT | None = None
    validators: list[Validator[InputT]] | None = None


InputHandler = Callable[[ParamUI[InputT], Console], InputT]
"""A handler function used to prompt a user for input data of some type."""

ParamKey = Tuple[str, str]
"""A tuple identifying a skill name and parameter name."""


@dataclass(frozen=True)
class SkillsUI:
    """A user interface to support executing a collection of skills."""

    handlers: Mapping[type, InputHandler]
    """Maps Python types to functions that prompt a user for an object of that type."""

    param_overrides: Mapping[ParamKey, ParamUI]
    """Maps (skill name, param name) tuples to override UIs with parameter-specific configs."""


def validate(
    value: InputT | None,
    validators: list[Validator[InputT]] | None,
    console: Console,
) -> bool:
    """Validate the given value using the given validators.

    :param value: Value to be validated (or None)
    :param validators: Optional list of validators specifying conditions for the value
    :param console: Command-line interface
    :return: True if the value is valid, else False
    """
    if value is None:
        return False

    if validators is None:
        return True

    valid = True
    for v in validators:
        error_message = v(value)
        if error_message is not None:
            console.print(f"[red]{error_message}[/]: {value}")
            valid = False

    return valid


def handle_bool(ui: ParamUI[bool], console: Console) -> bool:
    """Prompt the user for a Boolean value using the CLI."""
    return click.confirm(text=ui.label, default=ui.default)


def handle_string(ui: ParamUI[str], console: Console) -> str:
    """Prompt the user for a string using the CLI."""
    if ui.default is None:
        return Prompt.ask(ui.label)

    return Prompt.ask(ui.label, default=ui.default)


def handle_float(ui: ParamUI[float], console: Console) -> float:
    """Prompt the user for a float using the CLI."""
    if ui.default is None:
        return FloatPrompt.ask(ui.label)

    return FloatPrompt.ask(ui.label, default=ui.default)


def handle_int(ui: ParamUI[int], console: Console) -> int:
    """Prompt the user for an integer using the CLI."""
    while True:
        if ui.default is None:
            value = IntPrompt.ask(ui.label)
        else:
            value = IntPrompt.ask(ui.label, default=ui.default)

        if validate(value, ui.validators, console):
            return value


def handle_filepath(ui: ParamUI[Path], console: Console) -> Path:
    """Prompt the user for a filepath using the CLI."""
    p = None if ui.default is None else Path(ui.default).expanduser().resolve()

    while p is None or not validate(p, ui.validators, console):
        raw = Prompt.ask(f"{ui.label} (absolute or relative)")
        p = Path(raw).expanduser().resolve()

    console.print(f"[cyan]Using filepath:[/] {p}")
    return p


def handle_pose(ui: ParamUI[Pose3D], console: Console) -> Pose3D:
    """Prompt the user for a 3D pose using the CLI."""
    console.print(f"[cyan]{ui.label}[/]")

    x = handle_float(ParamUI("x (m)", 0 if ui.default is None else ui.default.position.x), console)
    y = handle_float(ParamUI("y (m)", 0 if ui.default is None else ui.default.position.y), console)
    z = handle_float(ParamUI("z (m)", 0 if ui.default is None else ui.default.position.z), console)

    rpy = None if ui.default is None else ui.default.orientation.to_euler_rpy()
    r = handle_float(ParamUI("roll (rad)", 0 if rpy is None else rpy.roll_rad), console)
    p = handle_float(ParamUI("pitch (rad)", 0 if rpy is None else rpy.pitch_rad), console)
    yaw = handle_float(ParamUI("yaw (rad)", 0 if rpy is None else rpy.yaw_rad), console)

    ref_frame_ui = ParamUI("ref_frame", None if ui.default is None else ui.default.ref_frame)
    ref_frame = handle_string(ref_frame_ui, console)

    return Pose3D.from_list([x, y, z, r, p, yaw], ref_frame=ref_frame)


def handle_pick_template(ui: ParamUI[PickTemplate], console: Console) -> PickTemplate:
    """Prompt the user for a template for a 'Pick' skill using the CLI."""
    console.print(f"[cyan]{ui.label}[/cyan]")

    obj_name = handle_string(
        ParamUI(
            label="Name of the object to be picked.",
            default=None if ui.default is None else ui.default.object_name,
        ),
        console,
    )
    open_rad = handle_float(
        ParamUI(
            label="Gripper angle (radians) before picking the object.",
            default=None if ui.default is None else ui.default.open_gripper_angle_rad,
        ),
        console,
    )
    pre_grasp_x_m = handle_float(
        ParamUI(
            "Offset (absolute meters) of the pre-grasp pose 'back' (-x) from the grasp pose.",
            default=None if ui.default is None else ui.default.pre_grasp_x_m,
        ),
        console,
    )
    pose_o_g = handle_pose(
        ParamUI(
            "Object-relative pose of the end-effector when the object is grasped.",
            default=None if ui.default is None else ui.default.pose_o_g,
        ),
        console,
    )
    post_grasp_lift_m = handle_float(
        ParamUI(
            "Offset (m) of the post-grasp pose 'up' (+z) from the grasp pose in the world frame.",
            default=None if ui.default is None else ui.default.post_grasp_lift_m,
        ),
        console,
    )
    stow_after = handle_bool(
        ParamUI(
            "If True, stow the arm after picking the object.",
            default=None if ui.default is None else ui.default.stow_after,
        ),
        console,
    )

    return PickTemplate(obj_name, open_rad, pre_grasp_x_m, pose_o_g, post_grasp_lift_m, stow_after)


def handle_place_template(ui: ParamUI[PlaceTemplate], console: Console) -> PlaceTemplate:
    """Prompt the user for a template for a 'Place' skill using the CLI."""
    console.print(f"[cyan]{ui.label}[/cyan]")

    ee_link = handle_string(
        ParamUI(
            label="Name of the end-effector link used to place an object.",
            default=None if ui.default is None else ui.default.ee_link_name,
        ),
        console,
    )
    obj_name = handle_string(
        ParamUI(
            label="Name of the held object to be placed.",
            default=None if ui.default is None else ui.default.object_name,
        ),
        console,
    )
    surface_name = handle_string(
        ParamUI(
            label="Name of the surface the object is placed onto.",
            default=None if ui.default is None else ui.default.surface_name,
        ),
        console,
    )
    pre_place_lift_m = handle_float(
        ParamUI(
            label="Offset (m) of the pre-place pose 'up' (+z world frame) from the place pose.",
            default=None if ui.default is None else ui.default.pre_place_lift_m,
        ),
        console,
    )
    place_pose_s_o = handle_pose(
        ParamUI(
            label="Surface-relative placement pose of the placed object.",
            default=None if ui.default is None else ui.default.place_pose_s_o,
        ),
        console,
    )
    post_place_x_m = handle_float(
        ParamUI(
            label="Offset (abs. meters) of the post-place pose 'back' (-x) from the place pose.",
            default=None if ui.default is None else ui.default.post_place_x_m,
        ),
        console,
    )

    return PlaceTemplate(
        ee_link,
        obj_name,
        surface_name,
        pre_place_lift_m,
        place_pose_s_o,
        post_place_x_m,
    )


def handle_open_drawer_template(
    ui: ParamUI[OpenDrawerTemplate],
    console: Console,
) -> OpenDrawerTemplate:
    """Prompt the user for a template for an 'OpenDrawer' skill using the CLI."""
    console.print(f"[cyan]{ui.label}[/cyan]")

    pregrasp_pose = handle_pose(
        ParamUI(
            label="Target end-effector pose before the drawer-grasping pose.",
            default=None if ui.default is None else ui.default.pregrasp_pose_ee,
        ),
        console,
    )
    grasp_pose = handle_pose(
        ParamUI(
            label="End-effector pose used to grasp the drawer handle.",
            default=None if ui.default is None else ui.default.grasp_drawer_pose_ee,
        ),
        console,
    )
    pull_pose = handle_pose(
        ParamUI(
            label="Target end-effector pose after initially pulling the drawer open.",
            default=None if ui.default is None else ui.default.pull_drawer_pose_ee,
        ),
        console,
    )
    open_traj_path = handle_filepath(
        ParamUI(
            label="Path to a YAML file containing the trajectory used to finish opening the drawer.",
            default=None if ui.default is None else ui.default.open_traj_path,
        ),
        console,
    )

    return OpenDrawerTemplate(pregrasp_pose, grasp_pose, pull_pose, open_traj_path)


INPUT_HANDLERS = {
    bool: handle_bool,
    str: handle_string,
    int: handle_int,
    float: handle_float,
    Path: handle_filepath,
    Pose3D: handle_pose,
    PickTemplate: handle_pick_template,
    PlaceTemplate: handle_place_template,
    OpenDrawerTemplate: handle_open_drawer_template,
}
