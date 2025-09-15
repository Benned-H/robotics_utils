"""Run skills through a command-line interface."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from robotics_utils.io.cli_handlers import (
    ParamUI,
    SkillsUI,
    handle_filepath,
    handle_pose,
    handle_string,
)
from robotics_utils.io.skills_cli import build_cli
from robotics_utils.kinematics import Pose3D
from robotics_utils.ros import TransformManager
from robotics_utils.skills.protocols.spot_skills import SpotSkillsProtocol


def yaml_filepath_exists(path: Path) -> str | None:
    """Validate that the given path points to a YAML file that exists."""
    if not path.exists():
        return "Path does not exist"
    if path.suffix not in {".yaml", ".yml"}:
        return "Path is not a YAML file"
    return None


def main() -> None:
    """Enter a CLI loop where the user can select skills to execute."""
    TransformManager.init_node("skills_cli")

    input_handlers = {
        str: handle_string,
        Path: handle_filepath,
        Pose3D: handle_pose,
    }

    param_overrides = {
        ("PlaybackTrajectory", "yaml_path"): ParamUI(
            label="YAML file specifying a relative end-effector trajectory",
            validators=[yaml_filepath_exists],
        ),
        ("OpenDrawer", "grasp_pose"): ParamUI(
            label="End-effector pose used to grasp the dresser drawer handle",
            default=Pose3D.from_xyz_rpy(
                x=0.471,
                z=0.476,
                yaw_rad=3.14159,
                ref_frame="black_dresser",
            ),
        ),
        ("OpenDrawer", "pull_pose"): ParamUI(
            label="End-effector pose at the end of pulling the drawer open",
            default=Pose3D.from_xyz_rpy(x=0.65, z=0.51, yaw_rad=3.14159, ref_frame="black_dresser"),
        ),
    }

    spot_skills_ui = SkillsUI(input_handlers, param_overrides)

    env_yaml = Path(
        "/docker/spot_skills/src/spot_skills/domains/environments/OpenBlackDresserDrawer.yaml",
    )
    if not env_yaml.exists():
        raise FileNotFoundError(f"YAML file does not exist: {env_yaml}")

    spot_skills_executor = SpotSkillsProtocol(env_yaml, Console(), take_control=True)

    cli = build_cli(spot_skills_executor, spot_skills_ui)
    cli()


if __name__ == "__main__":
    main()
