"""Define a protocol for skills on the Boston Dynamics Spot mobile manipulator."""

import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import rospy
from spot_skills.srv import (
    NameService,
    NameServiceRequest,
    NameServiceResponse,
    OpenDoor,
    OpenDoorRequest,
    OpenDoorResponse,
    PlaybackTrajectory,
    PlaybackTrajectoryRequest,
    PlaybackTrajectoryResponse,
    PoseLookup,
    PoseLookupRequest,
    PoseLookupResponse,
)

from robotics_utils.io.logging import console
from robotics_utils.kinematics import Pose3D, Waypoints
from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import (
    MoveItMotionPlanner,
    PlanningSceneManager,
    PoseBroadcastThread,
    ServiceCaller,
    trigger_service,
)
from robotics_utils.ros.msg_conversion import pose_from_msg
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.skills import SkillOutcome, SkillsProtocol, skill_method

SPOT_GRIPPER_OPEN_RAD = -1.5707
SPOT_GRIPPER_CLOSED_RAD = 0.0
SPOT_GRIPPER_HALF_OPEN_RAD = (SPOT_GRIPPER_OPEN_RAD + SPOT_GRIPPER_CLOSED_RAD) / 2.0


@dataclass(frozen=True)
class SpotSkillsConfig:
    """Configuration parameters for the Spot skills protocol."""

    env_yaml: Path
    """Path to a YAML file representing the environment."""

    markers_yaml: Path
    """Path to a YAML file specifying a fiducial marker system."""

    pose_estimate_window_size: int = 15
    """Number of poses used in the rolling average estimate for each frame."""

    take_control: bool = False
    """Whether or not to immediately take control of Spot."""

    def __post_init__(self) -> None:
        """Verify that the constructed configuration is valid."""
        if not self.env_yaml.exists():
            raise FileNotFoundError(f"YAML file does not exist: {self.env_yaml}")

        if not self.markers_yaml.exists():
            raise FileNotFoundError(f"YAML file does not exist: {self.markers_yaml}")


class SpotSkillsProtocol(SkillsProtocol):
    """Define a Python interface for executing Spot skills."""

    def __init__(self, config: SpotSkillsConfig) -> None:
        """Initialize the Spot skills executor."""
        self.robot_name = "Spot"

        self._waypoints = Waypoints.from_yaml(config.env_yaml)

        self._nav_to_waypoint_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "/spot/navigation/to_waypoint",
            NameService,
        )
        self._traj_playback_caller = ServiceCaller[
            PlaybackTrajectoryRequest,
            PlaybackTrajectoryResponse,
        ]("spot/playback_trajectory", PlaybackTrajectory)
        self._open_door_caller = ServiceCaller[OpenDoorRequest, OpenDoorResponse](
            "spot/open_door",
            OpenDoor,
        )
        self._pose_lookup_caller = ServiceCaller[PoseLookupRequest, PoseLookupResponse](
            "pose_lookup",
            PoseLookup,
        )

        self._gripper = ROSAngularGripper(
            limits=GripperAngleLimits(
                open_rad=SPOT_GRIPPER_OPEN_RAD,
                closed_rad=SPOT_GRIPPER_CLOSED_RAD,
            ),
            grasping_group="gripper",
            action_name="gripper_controller/gripper_action",
        )

        self._arm = MoveItManipulator(name="arm", base_frame="body", gripper=self._gripper)
        self._planning_scene = PlanningSceneManager(move_group_name=self._arm.name)
        self._motion_planner = MoveItMotionPlanner(self._arm, self._planning_scene)

        self._pose_broadcaster = PoseBroadcastThread()

        self._kinematic_tree = KinematicTree.from_yaml(config.env_yaml)

    def spin_once(self, duration_s: float = 0.1) -> None:
        """Sleep for the given duration to allow background processing."""
        rospy.sleep(duration_s)

    @skill_method
    def navigate_to_waypoint(self, waypoint: str) -> SkillOutcome:
        """Navigate to the named waypoint using global path planning.

        :param waypoint: Name of a navigation waypoint
        :return: Boolean success indicator and an outcome message
        """
        if waypoint not in self._waypoints:
            return SkillOutcome(
                success=False,
                message=(
                    f"Cannot navigate to unknown waypoint: '{waypoint}'. "
                    f"Available waypoints: {list(self._waypoints.keys())}."
                ),
            )

        request = NameServiceRequest(name=waypoint)
        response = self._nav_to_waypoint_caller(request)

        if response is None:
            return SkillOutcome(False, "NavigateToWaypoint service response was None.")

        return SkillOutcome(response.success, response.message)

    @skill_method
    def stow_arm(self) -> SkillOutcome:
        """Stow Spot's arm."""
        success = trigger_service("spot/stow_arm")
        if success:
            self._gripper.close()
        message = "Spot's arm was stowed." if success else "Could not stow Spot's arm."
        return SkillOutcome(success=success, message=message)

    @skill_method
    def erase_board(self) -> SkillOutcome:
        """Erase a whiteboard using a force-controlled trajectory.

        :return: Boolean success indicator and an outcome message
        """
        success = trigger_service("spot/erase_board")
        message = "Erased the board." if success else "Unable to erase the board."
        return SkillOutcome(success, message)

    @skill_method
    def open_drawer(
        self,
        pregrasp_pose_ee: Pose3D,
        grasp_pose_ee: Pose3D,
        pull_pose_ee: Pose3D,
        drawer_object_name: str,
    ) -> SkillOutcome:
        """Open a drawer using Spot's end-effector.

        :param pregrasp_pose_ee: Intermediate end-effector pose before Spot grasps the drawer
        :param grasp_pose_ee: Target end-effector pose when Spot grasps the drawer handle
        :param pull_pose_ee: Target end-effector pose after Spot pulls the drawer open
        :param drawer_object_name: Object name of the container that has the drawer
        :return: Boolean success indicator and an outcome message
        """
        nav_outcome = self.navigate_to_waypoint("open_drawer")
        if not nav_outcome.success:
            return SkillOutcome(success=False, message=nav_outcome.message)

        self._gripper.open()  # Open Spot's gripper before approaching the dresser

        self._pose_broadcaster.poses["pregrasp_drawer"] = pregrasp_pose_ee
        pre_outcome = self._move_ee_to_pose(pregrasp_pose_ee)
        if not pre_outcome.success:
            return pre_outcome

        self._pose_broadcaster.poses["grasp_drawer"] = grasp_pose_ee
        grasp_outcome = self._move_ee_to_pose(grasp_pose_ee)
        if not grasp_outcome.success:
            return grasp_outcome

        self._gripper.close()
        time.sleep(3)  # Wait a few seconds for the gripper to settle

        self._pose_broadcaster.poses["pull_drawer"] = pull_pose_ee
        pull_outcome = self._move_ee_to_pose(pull_pose_ee)
        if not pull_outcome.success:
            return pull_outcome

        self._gripper.open()

        # TODO: What frame is the pull pose specified in? And how does that frame work?
        # After letting go, pull the gripper farther back and then stow the arm
        post_pull_pose = deepcopy(pull_pose_ee)
        post_pull_pose.position.x += 0.1

        self._pose_broadcaster.poses["postpull_drawer"] = post_pull_pose
        post_outcome = self._move_ee_to_pose(post_pull_pose)
        if not post_outcome.success:
            return post_outcome

        stow_outcome = self.stow_arm()
        if not stow_outcome.success:
            return stow_outcome

        # Update the mesh of the opened drawer's container in MoveIt
        self._kinematic_tree.open_container(drawer_object_name)

        return SkillOutcome(success=True, message="Successfully opened the drawer.")

    @skill_method
    def _take_control(self) -> SkillOutcome:
        """Take control of the Spot and unlock its arm, if necessary."""
        if not trigger_service("spot/take_control"):
            return SkillOutcome(False, "Unable to take control of Spot.")
        if not trigger_service("spot/unlock_arm"):
            return SkillOutcome(False, "Unable to unlock Spot's arm.")
        return SkillOutcome(True, "Successfully took control of Spot and unlocked Spot's arm.")

    @skill_method
    def open_gripper(self) -> SkillOutcome:
        """Open Spot's gripper."""
        self._gripper.open()
        return SkillOutcome(success=True, message="Opened Spot's gripper.")

    @skill_method
    def close_gripper(self) -> SkillOutcome:
        """Close Spot's gripper."""
        self._gripper.close()
        return SkillOutcome(success=True, message="Closed Spot's gripper.")

    @skill_method
    def _move_ee_to_pose(self, ee_target: Pose3D) -> SkillOutcome:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :return: Boolean success indicator and an outcome message
        """
        query = MotionPlanningQuery(ee_target)

        plan = self._motion_planner.compute_motion_plan(query)
        if plan is None:
            return SkillOutcome(success=False, message="No motion plan found.")

        with console.status("Executing trajectory..."):
            self._arm.execute_motion_plan(plan)

        return SkillOutcome(success=True, message="Motion plan has been executed.")

    @skill_method
    def _lookup_pose(self, frame: str, ref_frame: str) -> SkillOutcome:
        """Look up the pose of a frame w.r.t. a reference frame.

        :param frame: Name of the frame whose pose is found
        :param ref_frame: Reference frame used for the lookup
        :return: Boolean success indicator and an outcome message
        """
        request = PoseLookupRequest()
        request.source_frame = frame
        request.target_frame = ref_frame

        response = self._pose_lookup_caller(request)
        if response is None:
            return SkillOutcome(False, "Pose lookup service response was None.")

        if response.success:
            pose = pose_from_msg(response.relative_pose)
            console.print(f"[cyan]Pose of {frame} w.r.t. {ref_frame}: {pose}.[/]")

        return SkillOutcome(response.success, response.message)

    @skill_method
    def _playback_trajectory(self, yaml_path: Path) -> SkillOutcome:
        """Play back a relative end-effector trajectory loaded from file.

        :param yaml_path: YAML file specifying a relative end-effector trajectory
        :return: Boolean success indicator and an outcome message
        """
        request = PlaybackTrajectoryRequest(str(yaml_path))
        response = self._traj_playback_caller(request)

        if response is None:
            return SkillOutcome(False, "Trajectory playback service response was None.")

        return SkillOutcome(response.success, response.message)

    @skill_method
    def open_door(self, body_pitch_rad: float, is_pull: bool, hinge_on_left: bool) -> SkillOutcome:
        """Open a door using the Spot SDK.

        :param body_pitch_rad: Pitch (radians) of Spot's body when taking the door handle image
        :param is_pull: Whether the door opens by pulling toward Spot
        :param hinge_on_left: Whether the door's hinge is on the left, from Spot's perspective
        :return: Boolean success indicator and an outcome message
        """
        request = OpenDoorRequest(body_pitch_rad, is_pull, hinge_on_left)
        response = self._open_door_caller(request)
        if response is None:
            return SkillOutcome(False, "OpenDoor service response was None.")
        return SkillOutcome(response.success, response.message)
