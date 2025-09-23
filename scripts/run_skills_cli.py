"""Run skills through a command-line interface."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console

from robotics_utils.io.cli_handlers import INPUT_HANDLERS, ParamUI, SkillsUI
from robotics_utils.io.skills_cli import build_cli
from robotics_utils.kinematics import Pose3D
from robotics_utils.ros import TransformManager
from robotics_utils.skills.protocols.spot_skills import (
    SPOT_GRIPPER_HALF_OPEN_RAD,
    SpotSkillsConfig,
    SpotSkillsProtocol,
)
from robotics_utils.skills.skill_templates import OpenDrawerTemplate, PickTemplate, PlaceTemplate


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

    spot_skills_catkin_pkg = Path("/docker/spot_skills/src/spot_skills")
    env_yaml = spot_skills_catkin_pkg / "config/icra_video/env.yaml"

    param_overrides = {
        ("PlaybackTrajectory", "yaml_path"): ParamUI(
            label="YAML file specifying a relative end-effector trajectory",
            validators=[yaml_filepath_exists],
        ),
        ("OpenDrawer", "template"): ParamUI(
            label="Template for an 'OpenDrawer' skill",
            default=OpenDrawerTemplate(
                pregrasp_pose_ee=Pose3D.from_xyz_rpy(
                    x=0.68,
                    z=0.56,
                    yaw_rad=3.1416,
                    ref_frame="black_dresser",
                ),
                grasp_drawer_pose_ee=Pose3D.from_xyz_rpy(
                    x=0.47,
                    z=0.56,
                    yaw_rad=3.1416,
                    ref_frame="black_dresser",
                ),
                pull_drawer_pose_ee=Pose3D.from_xyz_rpy(
                    x=0.68,
                    z=0.63,
                    yaw_rad=3.1416,
                    ref_frame="black_dresser",
                ),
                open_traj_path=Path("/docker/spot_skills/drawer.yaml"),
            ),
        ),
        ("LookForObject", "object_name"): ParamUI(
            label="Name of the object looked for",
            default="eraser1",
        ),
        # ("LookForObject", "ee_pose"): ParamUI(
        #     label="Pose of the end-effector when looking",
        #     default=Pose3D.from_xyz_rpy(
        #         x=0.77,
        #         y=0.2,
        #         z=0.65,
        #         pitch_rad=0.78,
        #         yaw_rad=0.5,
        #         ref_frame="body",
        #     ),
        # ),
        ("LookForObject", "duration_s"): ParamUI(
            label="Duration (seconds) to wait during pose estimation",
            default=5.0,
        ),
        ("EraseBoard", "whiteboard_x_m"): ParamUI(
            label="x-coordinate of the whiteboard in Spot's body frame",
            default=1.2,
        ),
        ("RecordTrajectory", "outpath"): ParamUI(
            label="Output path where the recorded trajectory is saved",
            default=Path("/docker/spot_skills/recorded_traj.yaml"),
        ),
        ("RecordTrajectory", "ref_frame"): ParamUI(
            label="Reference frame used for the initial relative pose",
            default="black_dresser",
        ),
        ("RecordTrajectory", "tracked_frame"): ParamUI(
            label="Frame tracked during the recording",
            default="arm_link_wr1",
        ),
        ("Pick", "template"): ParamUI(
            label="Template for a 'Pick' skill",
            default=PickTemplate(
                object_name="eraser1",
                open_gripper_angle_rad=SPOT_GRIPPER_HALF_OPEN_RAD,
                pre_grasp_x_m=0.05,
                pose_o_g=Pose3D.from_xyz_rpy(
                    z=0.36,
                    pitch_rad=1.5708,
                    yaw_rad=-1.5708,
                    ref_frame="eraser1",
                ),
                post_grasp_lift_m=0.03,
                stow_after=True,
            ),
        ),
        ("Place", "template"): ParamUI(
            label="Template for a 'Place' skill",
            default=PlaceTemplate(
                ee_link_name="arm_link_wr1",
                object_name="eraser1",
                surface_name="filing_cabinet",
                pre_place_lift_m=0.05,
                place_pose_s_o=Pose3D.from_xyz_rpy(
                    x=0.1,
                    y=0.1,
                    z=0.7,
                    yaw_rad=0.523599,
                    ref_frame="filing_cabinet",
                ),
                post_place_x_m=0.08,
            ),
        ),
        ("LoadPlanningScene", "env_yaml"): ParamUI(
            label="Path to a YAML file specifying an environment state",
            default=env_yaml,
        ),
        ("MoveEeToPose", "ee_target"): ParamUI(
            label="End-effector target pose",
            default=Pose3D.from_xyz_rpy(z=0.4, x=0.4, ref_frame="body"),  # TODO: Check OK?
        ),
    }

    spot_skills_ui = SkillsUI(INPUT_HANDLERS, param_overrides)

    spot_skills_catkin_pkg = Path("/docker/spot_skills/src/spot_skills")

    config = SpotSkillsConfig(
        env_yaml=env_yaml,
        console=Console(),
        markers_yaml=spot_skills_catkin_pkg / "config/icra_video/markers.yaml",
        marker_topic_prefix="/ar_pose_marker",
    )

    spot_skills_executor = SpotSkillsProtocol(config)

    cli = build_cli(spot_skills_executor, spot_skills_ui)
    cli()


if __name__ == "__main__":
    main()
