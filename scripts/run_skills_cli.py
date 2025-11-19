"""Run skills through a command-line interface."""

from pathlib import Path

from robotics_utils.io.skills_cli import build_cli
from robotics_utils.ros import TransformManager
from robotics_utils.skills.protocols.spot_skills import SpotSkillsConfig, SpotSkillsProtocol


def main() -> None:
    """Enter a CLI loop where a user selects skills to execute."""
    TransformManager.init_node("skills_cli")

    spot_skills_catkin_pkg = Path("/docker/spot_skills/src/spot_skills")
    env_yaml = spot_skills_catkin_pkg / "config/env.yaml"
    markers_yaml = spot_skills_catkin_pkg / "config/markers.yaml"

    config = SpotSkillsConfig(env_yaml, markers_yaml)
    skills_protocol = SpotSkillsProtocol(config)

    cli = build_cli(skills_protocol)
    cli()


if __name__ == "__main__":
    main()
