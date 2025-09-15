"""Define a skills protocol for the Spot mobile manipulator."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

from rich.console import Console
from rich.panel import Panel
from spot_skills.srv import (
    NameService,
    NameServiceRequest,
    NameServiceResponse,
    PlaybackTrajectory,
    PlaybackTrajectoryRequest,
    PlaybackTrajectoryResponse,
    PoseLookup,
    PoseLookupRequest,
    PoseLookupResponse,
)

from robotics_utils.kinematics import Pose3D, Waypoints
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import MoveItMotionPlanner, PlanningSceneManager, TransformManager
from robotics_utils.ros.msg_conversion import pose_from_msg
from robotics_utils.ros.pose_broadcast_thread import PoseBroadcastThread
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.ros.services import ServiceCaller, trigger_service
from robotics_utils.skills import SkillsProtocol, skill_method
from robotics_utils.skills.skill import SkillResult


class SpotSkillsProtocol(SkillsProtocol):
    """Define the structure of skills for Spot."""

    def __init__(self, env_yaml: Path, console: Console, take_control: bool) -> None:
        """Initialize the Spot skills executor.

        :param env_yaml: Path to a YAML file representing the environment
        :param console: Console used to output CLI messages
        :param take_control: Whether or not to immediately take control of Spot
        """
        self._waypoints = Waypoints.from_yaml(env_yaml)
        self._console = console

        self._nav_to_waypoint_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "/spot/navigation/to_waypoint",
            NameService,
        )
        self._pose_lookup_caller = ServiceCaller[PoseLookupRequest, PoseLookupResponse](
            "pose_lookup",
            PoseLookup,
        )
        self._traj_playback_caller = ServiceCaller[
            PlaybackTrajectoryRequest,
            PlaybackTrajectoryResponse,
        ]("spot/playback_trajectory", PlaybackTrajectory)
        self._open_door_srv_name = "spot/open_door"
        self._take_control_srv_name = "spot/take_control"
        self._unlock_arm_srv_name = "spot/unlock_arm"
        self._erase_board_srv_name = "spot/erase_board"

        self._gripper = ROSAngularGripper(
            limits=GripperAngleLimits(open_rad=-1.5707, closed_rad=0.0),
            grasping_group="gripper",
            action_name="gripper_controller/gripper_action",
        )
        self._arm = MoveItManipulator(name="arm", base_frame="body", gripper=self._gripper)
        self._scene = PlanningSceneManager(body_frame=self._arm.base_frame)
        self._motion_planner = MoveItMotionPlanner(self._arm, self._scene)

        if take_control:
            trigger_service(self._take_control_srv_name)
            trigger_service(self._unlock_arm_srv_name)

        self._pose_broadcaster = PoseBroadcastThread()

    @skill_method
    def go_to(self, waypoint: str) -> SkillResult:
        """Navigate to the named waypoint.

        :param waypoint: Name of a navigation waypoint
        :return: Tuple containing a Boolean skill success and outcome message
        """
        if waypoint not in self._waypoints:
            return False, (
                f"Cannot navigate to unknown waypoint: '{waypoint}'. "
                f"Available waypoints: {list(self._waypoints.keys())}."
            )

        request = NameServiceRequest(name=waypoint)
        response = self._nav_to_waypoint_caller(request)

        if response is None:
            return False, "Navigation service response was None."

        return response.success, response.message

    @skill_method
    def open_door(self) -> SkillResult:  # TODO: Should take arguments!
        """Open a door using Spot's built-in skill."""
        success = trigger_service(self._open_door_srv_name)
        message = "Door has been opened!" if success else "Could not open the door."
        return success, message

    @skill_method
    def erase_board(self) -> SkillResult:  # TODO: Should take args
        """Erase a whiteboard using a force-controlled trajectory."""
        success = trigger_service(self._erase_board_srv_name)
        message = "Erased the board." if success else "Could not erase the board."
        return success, message

    @skill_method  # TODO: Args to specify which drawer
    def open_drawer(self, grasp_pose: Pose3D, pull_pose: Pose3D) -> SkillResult:
        """Open a drawer using Spot's gripper.

        :param grasp_pose: End-effector pose used to grasp the dresser drawer handle
        :param pull_pose: End-effector pose after initially pulling the drawer open
        """
        nav_success, nav_outcome = self.go_to("open_black_dresser")
        if not nav_success:
            return False, nav_outcome

        self._gripper.open()  # Open Spot's gripper before approaching the dresser

        grasp_success, grasp_outcome = self._move_ee_to_pose(grasp_pose)
        if not grasp_success:
            return False, grasp_outcome

        self._gripper.close()
        time.sleep(3)  # Wait 3 seconds for the gripper to settle

        pull_pose = Pose3D.from_xyz_rpy(x=0.65, z=0.51, yaw_rad=3.14159, ref_frame="black_dresser")
        pull_success, pull_outcome = self._move_ee_to_pose(pull_pose)
        if not pull_success:
            return False, pull_outcome

        self._gripper.open()

        # TODO: Finish the skill!

        return True, "Successfully opened the drawer."

    @skill_method
    def _lookup_pose(self, frame: str, ref_frame: str) -> SkillResult:
        """Look up the pose of a frame w.r.t. a reference frame.

        :param frame: Name of the frame whose pose is found
        :param ref_frame: Reference frame for the pose lookup
        :return: Tuple containing a Boolean skill success and outcome message
        """
        request = PoseLookupRequest()
        request.source_frame = frame
        request.target_frame = ref_frame

        response = self._pose_lookup_caller(request)
        if response is None:
            return False, "Pose lookup service response was None."

        if response.success:
            pose = pose_from_msg(response.relative_pose)
            self._console.print(f"[blue]Pose of {frame} w.r.t. {ref_frame}: {pose}.[/blue]")

        return response.success, response.message

    @skill_method
    def _playback_trajectory(self, yaml_path: Path) -> SkillResult:
        """Play back a relative end-effector trajectory loaded from file.

        :param yaml_path: YAML file specifying a relative end-effector trajectory
        :return: Tuple containing a Boolean skill success and outcome message
        """
        request = PlaybackTrajectoryRequest(yaml_path)
        response = self._traj_playback_caller(request)
        if response is None:
            return False, "Trajectory playback service response was None."

        return response.success, response.message

    @skill_method
    def _move_ee_to_pose(self, ee_target: Pose3D) -> SkillResult:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :return: Tuple containing a Boolean skill success and outcome message
        """
        self._pose_broadcaster.poses["ee_target"] = ee_target

        query = MotionPlanningQuery(ee_target)

        traj = self._motion_planner.compute_motion_plan(query)
        if traj is None:
            return False, "❌ No plan found."

        with self._console.status("Executing trajectory...", spinner="dots"):
            self._arm.execute_motion_plan(traj)

        return True, "✅ Reached target pose."  # TODO: Doesn't actually check to verify
