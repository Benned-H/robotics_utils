"""Define functions providing CLI handlers for various input types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

from click import confirm
from rich.prompt import FloatPrompt, IntPrompt, Prompt

from robotics_utils.io import console
from robotics_utils.kinematics import Point3D, Pose3D

if TYPE_CHECKING:
    from robotics_utils.skills import SkillParamKey


InputT = TypeVar("InputT")
"""The type of an input being requested via CLI."""


class InvalidParameterValueError(Exception):
    """An error raised when a user has provided an invalid parameter value."""


Validator = Callable[[InputT], None]
"""Raises an `InvalidParameterValueError` on invalid inputs."""


@dataclass(frozen=True)
class ParamUI(Generic[InputT]):
    """Skill parameter validation overlay passed to the input handler for the parameter's type."""

    prompt: str
    """String used to prompt the user for a parameter value."""

    default: InputT | None = None
    """Default value for the parameter (defaults to None = no default value)."""

    validators: list[Validator[InputT]] | None = None
    """Optional list of input validators (defaults to None)."""


def validate(value: InputT | None, validators: list[Validator[InputT]] | None) -> bool:
    """Evaluate whether the given value is valid under the given validators.

    :param value: User input to be validated (or None)
    :param validators: Optional list of validators specifying constraints on valid values
    :return: True if the value is valid, else False
    """
    if value is None:
        return False

    if validators is None:
        return True

    try:
        for v in validators:
            v(value)
    except InvalidParameterValueError as inv:
        console.print(f"[red]{inv}[/]: {value}")
        return False

    return True


def handle_bool(ui: ParamUI[bool]) -> bool:
    """Prompt the user for a Boolean value using the CLI."""
    while True:
        value = confirm(text=ui.prompt, default=ui.default)

        if validate(value, ui.validators):
            return value


def handle_string(ui: ParamUI[str]) -> str:
    """Prompt the user for a string using the CLI."""
    while True:
        value = (
            Prompt.ask(ui.prompt)
            if ui.default is None
            else Prompt.ask(ui.prompt, default=ui.default)
        )

        if validate(value, ui.validators):
            return value


def handle_float(ui: ParamUI[float]) -> float:
    """Prompt the user for a float using the CLI."""
    while True:
        value = (
            FloatPrompt.ask(ui.prompt)
            if ui.default is None
            else FloatPrompt.ask(ui.prompt, default=ui.default)
        )

        if validate(value, ui.validators):
            return value


def handle_int(ui: ParamUI[int]) -> int:
    """Prompt the user for an integer using the CLI."""
    while True:
        value = (
            IntPrompt.ask(ui.prompt)
            if ui.default is None
            else IntPrompt.ask(ui.prompt, default=ui.default)
        )

        if validate(value, ui.validators):
            return value


def handle_filepath(ui: ParamUI[Path]) -> Path:
    """Prompt the user for a filepath using the CLI."""
    full_prompt = f"{ui.prompt} (absolute or relative)"

    while True:
        raw = (
            Prompt.ask(full_prompt)
            if ui.default is None
            else Prompt.ask(full_prompt, default=str(ui.default))
        )
        path = Path(raw).expanduser().resolve()

        if validate(path, ui.validators):
            console.print(f"[cyan]Using filepath:[/] {path}")
            return path


def handle_point_3d(ui: ParamUI[Point3D]) -> Point3D:
    """Prompt the user for a 3D point using the CLI."""
    while True:
        console.print(f"[cyan]{ui.prompt}[/]")

        x = handle_float(ParamUI("x (m)", 0.0 if ui.default is None else ui.default.x))
        y = handle_float(ParamUI("y (m)", 0.0 if ui.default is None else ui.default.y))
        z = handle_float(ParamUI("z (m)", 0.0 if ui.default is None else ui.default.z))

        point = Point3D(x, y, z)
        if validate(point, ui.validators):
            return point


def handle_pose_3d(ui: ParamUI[Pose3D]) -> Pose3D:
    """Prompt the user for a 3D pose using the CLI."""
    while True:
        console.print(f"[cyan]{ui.prompt}[/]")

        x = handle_float(ParamUI("x", 0.0 if ui.default is None else ui.default.position.x))
        y = handle_float(ParamUI("y", 0.0 if ui.default is None else ui.default.position.y))
        z = handle_float(ParamUI("z", 0.0 if ui.default is None else ui.default.position.z))

        rpy = None if ui.default is None else ui.default.orientation.to_euler_rpy()
        roll = handle_float(ParamUI("roll (rad)", 0.0 if rpy is None else rpy.roll_rad))
        pitch = handle_float(ParamUI("pitch (rad)", 0.0 if rpy is None else rpy.pitch_rad))
        yaw = handle_float(ParamUI("yaw (rad)", 0.0 if rpy is None else rpy.yaw_rad))

        ref_frame_ui = ParamUI("ref_frame", None if ui.default is None else ui.default.ref_frame)
        ref_frame = handle_string(ref_frame_ui)

        pose = Pose3D.from_sequence([x, y, z, roll, pitch, yaw], ref_frame=ref_frame)
        if validate(pose, ui.validators):
            return pose


INPUT_HANDLERS = {
    bool: handle_bool,
    str: handle_string,
    float: handle_float,
    int: handle_int,
    Path: handle_filepath,
    Point3D: handle_point_3d,
    Pose3D: handle_pose_3d,
}

InputHandler = Callable[[ParamUI[InputT]], InputT]
"""A handler function used to prompt a user for input data of some type."""


@dataclass
class SkillsUI:
    """A user interface to supporting executing a collection of skills."""

    handlers: dict[type, InputHandler]
    """A map from Python types to functions that prompt a user for a value of that type."""

    default_values: dict[SkillParamKey, Any]
    """A map from (skill name, param name) tuples to the default value for that skill parameter."""
