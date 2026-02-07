"""Run skills through a command-line interface."""

from spot_skills_py.spot.spot_conversion import SPOT_GRIPPER_CLOSED_RAD, SPOT_GRIPPER_OPEN_RAD

from robotics_utils.io.skills_cli import build_cli
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import TransformManager
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.skills.protocols.spot_skills import SpotSkillsProtocol


def main() -> None:
    """Enter a CLI loop where a user selects skills to execute."""
    TransformManager.init_node("skills_cli")

    gripper = ROSAngularGripper(
        limits=GripperAngleLimits(
            open_rad=SPOT_GRIPPER_OPEN_RAD,
            closed_rad=SPOT_GRIPPER_CLOSED_RAD,
        ),
        grasping_group="gripper",
        action_name="gripper_controller/gripper_action",
    )
    manipulator = MoveItManipulator(
        name="arm",
        robot_name="Spot",
        base_frame="body",
        planning_frame="map",
        gripper=gripper,
    )

    skills_protocol = SpotSkillsProtocol(manipulator)

    cli = build_cli(skills_protocol)
    cli()


if __name__ == "__main__":
    main()
