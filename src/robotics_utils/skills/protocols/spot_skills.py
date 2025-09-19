"""Define a skills protocol for the Spot mobile manipulator."""

from __future__ import annotations

import time
from contextlib import suppress
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import rospy
from rich.console import Console
from spot_skills.srv import (
    Float64Service,
    Float64ServiceRequest,
    Float64ServiceResponse,
    NameService,
    NameServiceRequest,
    NameServiceResponse,
    NavigateToPose,
    NavigateToPoseRequest,
    NavigateToPoseResponse,
    PlaybackTrajectory,
    PlaybackTrajectoryRequest,
    PlaybackTrajectoryResponse,
    PoseLookup,
    PoseLookupRequest,
    PoseLookupResponse,
)

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D, Waypoints
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.perception.pose_estimation import FiducialSystem
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import (
    FiducialTracker,
    MoveItMotionPlanner,
    PlanningSceneManager,
    TransformManager,
)
from robotics_utils.ros.msg_conversion import pose_from_msg, pose_to_stamped_msg
from robotics_utils.ros.pose_broadcast_thread import PoseBroadcastThread
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.ros.services import ServiceCaller, trigger_service
from robotics_utils.ros.transform_recorder import TransformRecorder
from robotics_utils.skills import SkillsProtocol, skill_method
from robotics_utils.skills.skill import SkillResult
from robotics_utils.skills.skill_templates import PickTemplate, PlaceTemplate


@dataclass(frozen=True)
class SpotSkillsConfig:
    """All configuration needed for the Spot skills protocol."""

    env_yaml: Path
    """Path to a YAML file representing the environment."""

    console: Console
    """Console used to output CLI messages."""

    markers_yaml: Path
    """Path to a YAML file specifying a fiducial marker system."""

    marker_topic_prefix: str

    pose_estimate_window_size: int = 10
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
    """Define the structure of skills for Spot."""

    def __init__(self, config: SpotSkillsConfig) -> None:
        """Initialize the Spot skills executor.

        :param config: Configuration for the Spot skills protocol
        """
        self._waypoints = Waypoints.from_yaml(config.env_yaml)
        self._console = config.console

        # Construct a fiducial tracker used to update object pose estimates
        env_data = load_yaml_data(config.env_yaml)

        known_poses = None
        if "object_poses" in env_data:
            known_poses = Pose3D.load_named_poses(config.env_yaml, "object_poses")

        self._fiducial_tracker = FiducialTracker(
            FiducialSystem.from_yaml(config.markers_yaml),
            config.marker_topic_prefix,
            config.pose_estimate_window_size,
            known_poses,
        )
        self._console.print(f"Fiducial markers loaded from YAML: {self._fiducial_tracker.system}")

        self._nav_to_waypoint_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "/spot/navigation/to_waypoint",
            NameService,
        )
        self._nav_to_pose_caller = ServiceCaller[NavigateToPoseRequest, NavigateToPoseResponse](
            "/spot/navigation/to_pose",
            NavigateToPose,
        )
        self._pose_lookup_caller = ServiceCaller[PoseLookupRequest, PoseLookupResponse](
            "pose_lookup",
            PoseLookup,
        )
        self._traj_playback_caller = ServiceCaller[
            PlaybackTrajectoryRequest,
            PlaybackTrajectoryResponse,
        ]("spot/playback_trajectory", PlaybackTrajectory)
        self._erase_board_caller = ServiceCaller[Float64ServiceRequest, Float64ServiceResponse](
            "spot/erase_board",
            Float64Service,
        )

        self._open_door_srv_name = "spot/open_door"
        self._take_control_srv_name = "spot/take_control"
        self._unlock_arm_srv_name = "spot/unlock_arm"
        self._stow_arm_srv_name = "spot/stow_arm"
        self._dock_srv_name = "spot/dock"

        self._gripper = ROSAngularGripper(
            limits=GripperAngleLimits(open_rad=-1.5707, closed_rad=0.0),
            grasping_group="gripper",
            action_name="gripper_controller/gripper_action",
        )
        self._arm = MoveItManipulator(name="arm", base_frame="body", gripper=self._gripper)
        self._scene = PlanningSceneManager(body_frame=self._arm.base_frame)
        self._motion_planner = MoveItMotionPlanner(self._arm, self._scene)

        if config.take_control:
            self._take_control()

        self._pose_broadcaster = PoseBroadcastThread()

    def spin_once(self, duration_s: float = 0.1) -> None:
        """Sleep for the given duration to allow background processing."""
        rospy.sleep(duration_s)

    @skill_method
    def go_to_waypoint(self, waypoint: str) -> SkillResult:
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
            return False, "NavigateToWaypoint service response was None."

        return response.success, response.message

    @skill_method
    def go_to_pose(self, base_pose: Pose3D) -> SkillResult:
        """Navigate to the given base pose.

        :param base_pose: Target base pose for the robot
        :return: Tuple containing a Boolean skill success and outcome message
        """
        request = NavigateToPoseRequest(target_base_pose=pose_to_stamped_msg(base_pose))
        response = self._nav_to_pose_caller(request)

        if response is None:
            return False, "NavigateToPose service response was None."

        return response.success, response.message

    @skill_method
    def open_door(self) -> SkillResult:  # TODO: Should take arguments!
        """Open a door using Spot's built-in skill."""
        success = trigger_service(self._open_door_srv_name)
        message = "Door has been opened!" if success else "Could not open the door."
        return success, message

    @skill_method
    def erase_board(self, whiteboard_x_m: float) -> SkillResult:  # TODO: Should take args
        """Erase a whiteboard using a force-controlled trajectory.

        :param whiteboard_x_m: x-coordinate of the whiteboard in Spot's body frame
        :return: Tuple containing a Boolean skill success and outcome message
        """
        request = Float64ServiceRequest(value=whiteboard_x_m)
        response = self._erase_board_caller(request)
        if response is None:
            return False, "EraseBoard service response was None."

        return response.success, response.message

    @skill_method
    def stow_arm(self) -> SkillResult:
        """Stow Spot's arm."""
        success = trigger_service(self._stow_arm_srv_name)
        if success:
            self._gripper.close()
        message = "Spot's arm was stowed." if success else "Could not stow Spot's arm."
        return success, message

    @skill_method
    def dock(self, dock_id: int) -> SkillResult:
        """Command Spot to dock at the specified dock.

        :param dock_id: ID of the dock Spot should dock at
        """
        success = trigger_service(self._dock_srv_name)  # TODO: Use the dock ID in the service call
        message = "Spot successfully docked." if success else "Spot failed to dock."
        return success, message

    @skill_method  # TODO: Args to specify which drawer
    def open_drawer(
        self,
        pre_pose: Pose3D,
        grasp_pose: Pose3D,
        pull_pose: Pose3D,
        traj_yaml: Path,
    ) -> SkillResult:
        """Open a drawer using Spot's gripper.

        :param pre_pose: Intermediate end-effector pose target before the grasp pose
        :param grasp_pose: End-effector pose used to grasp the dresser drawer handle
        :param pull_pose: End-effector pose after initially pulling the drawer open
        :param traj_yaml: Path to a YAML file containing the skill's final trajectory
        """
        nav_success, nav_outcome = self.go_to_waypoint("open_drawer")
        if not nav_success:
            return False, nav_outcome

        self._gripper.open()  # Open Spot's gripper before approaching the dresser

        pre_success, pre_outcome = self._move_ee_to_pose(pre_pose, "pre_grasp_drawer")
        if not pre_success:
            return False, pre_outcome

        grasp_success, grasp_outcome = self._move_ee_to_pose(grasp_pose, "grasp_drawer")
        if not grasp_success:
            return False, grasp_outcome

        self._gripper.close()
        time.sleep(3)  # Wait 3 seconds for the gripper to settle

        pull_success, pull_outcome = self._move_ee_to_pose(pull_pose, "pull_drawer")
        if not pull_success:
            return False, pull_outcome

        self._gripper.open()

        # After letting go, pull the gripper back (+10 cm x) and then stow the arm
        post_pull_pose = deepcopy(pull_pose)
        post_pull_pose.position.x += 0.1

        post_success, post_outcome = self._move_ee_to_pose(post_pull_pose, "post_pull_drawer")
        if not post_success:
            return False, post_outcome

        stow_success, stow_msg = self.stow_arm()
        if not stow_success:
            return False, stow_msg

        # Next, play the recorded trajectory that finishes opening the drawer
        traj_success, traj_msg = self._playback_trajectory(traj_yaml)
        if not traj_success:
            return False, traj_msg

        stow_success, stow_msg = self.stow_arm()
        if not stow_success:
            return False, stow_msg

        return True, "Successfully opened the drawer."

    @skill_method
    def look_for_object(
        self,
        ee_pose: Pose3D,
        object_name: str,
        duration_s: float,
        stow_after: bool,
    ) -> SkillResult:
        """Look for the named object using the gripper camera, then stow Spot's arm.

        :param ee_pose: Pose of the end-effector when looking
        :param object_name: Name of the object looked for
        :param duration_s: Duration (seconds) to wait during pose estimation
        :param stow_after: If True, stow Spot's arm at the end of the skill
        :return: Tuple containing a Boolean skill success and outcome message
        """
        move_ee_ok, move_ee_msg = self._move_ee_to_pose(ee_pose, f"look_for_{object_name}")
        if not move_ee_ok:
            return False, move_ee_msg

        self._gripper.open()

        object_pose_estimate = self._fiducial_tracker.reestimate(object_name, duration_s)
        if object_pose_estimate is None:
            return False, f"Could not find an updated pose estimate for object '{object_name}'."

        self._gripper.close()

        if stow_after:
            stow_ok, stow_msg = self.stow_arm()
            if not stow_ok:
                return False, stow_msg

        return True, f"Updated object pose estimate: {object_pose_estimate}."

    @skill_method
    def pick(self, object_name: str, template: PickTemplate) -> SkillResult:
        """Pick an object based on the given skill template.

        :param object_name: Name of the object to be picked
        :param template: Template for a 'Pick' skill
        :return: Tuple containing a Boolean skill success and outcome message
        """
        if object_name != template.pose_o_g.ref_frame:
            gpose_frame = template.pose_o_g.ref_frame
            self._console.print(f"[yellow]Warning: Grasp pose given in frame {gpose_frame}.[/]")
            fixed_pose_o_g = TransformManager.convert_to_frame(template.pose_o_g, object_name)
            template = replace(template, pose_o_g=fixed_pose_o_g)

        # Compute the pre-grasp pose
        pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-template.pre_grasp_x_m)
        pose_o_pregrasp = template.pose_o_g @ pose_g_pregrasp

        # Compute the post-grasp pose
        pose_w_g = TransformManager.convert_to_frame(template.pose_o_g, target_frame=DEFAULT_FRAME)
        pose_w_postg = deepcopy(pose_w_g)
        pose_w_postg.position.z += template.post_grasp_lift_m

        self._gripper.open()

        pre_ok, pre_msg = self._move_ee_to_pose(pose_o_pregrasp, f"pre_grasp_{object_name}")
        if not pre_ok:
            return False, pre_msg

        grasp_ok, grasp_msg = self._move_ee_to_pose(template.pose_o_g, f"grasp_{object_name}")
        if not grasp_ok:
            return False, grasp_msg

        self._gripper.close()

        post_ok, post_msg = self._move_ee_to_pose(pose_w_postg, f"post_grasp_{object_name}")
        if not post_ok:
            return False, post_msg

        carry_ok, carry_msg = self._move_ee_to_pose(template.pose_b_carry, f"carry_{object_name}")
        if not carry_ok:
            return False, carry_msg

        return True, f"Successfully picked object '{object_name}'."

    @skill_method
    def place(self, surface_name: str, template: PlaceTemplate) -> SkillResult:
        """Place a held object onto a surface based on the given template.

        :param surface_name: Name of the surface to place the object on
        :param template: Template for a 'Place' skill
        :return: Tuple containing a Boolean skill success and outcome message
        """
        if surface_name != template.pose_s_o.ref_frame:
            place_frame = template.pose_s_o.ref_frame
            self._console.print(f"[yellow]Warning: Place pose given in frame {place_frame}.[/]")
            fixed_pose_s_o = TransformManager.convert_to_frame(template.pose_s_o, surface_name)
            template = replace(template, pose_s_o=fixed_pose_s_o)

        # TODO

    @skill_method
    def _take_control(self) -> SkillResult:
        """Take control of the Spot robot (and unlock its arm, if necessary)."""
        if not trigger_service(self._take_control_srv_name):
            return False, "Unable to take control of Spot."
        if not trigger_service(self._unlock_arm_srv_name):
            return False, "Unable to unlock Spot's arm."
        return True, "Successfully took control of Spot and unlocked Spot's arm."

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
    def _record_trajectory(
        self,
        outpath: Path,
        overwrite: bool,
        ref_frame: str,
        tracked_frame: str,
    ) -> SkillResult:
        """Record an end-effector relative trajectory and save it to YAML.

        :param outpath: Output path where the YAML file is created
        :param overwrite: Whether to allow overwriting the output path
        :param ref_frame: Reference frame used for the initial relative pose
        :param tracked_frame: Frame tracked during the recording
        :return: Tuple containing a Boolean skill success and outcome message
        """
        if not overwrite and outpath.exists():
            return False, f"Cannot overwrite existing output path: {outpath}"

        config_before = self._arm.configuration

        recorder = TransformRecorder(ref_frame, tracked_frame)

        rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
        try:
            while not rospy.is_shutdown():
                recorder.update()
                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            return False, f"{ros_exc}"
        finally:
            config_after = self._arm.configuration
            recorder.save_to_file(outpath, config_before, config_after)

        return True, f"Saved to YAML file: {outpath}."

    @skill_method
    def _playback_trajectory(self, yaml_path: Path) -> SkillResult:
        """Play back a relative end-effector trajectory loaded from file.

        :param yaml_path: YAML file specifying a relative end-effector trajectory
        :return: Tuple containing a Boolean skill success and outcome message
        """
        request = PlaybackTrajectoryRequest(str(yaml_path))
        response = self._traj_playback_caller(request)
        if response is None:
            return False, "Trajectory playback service response was None."

        return response.success, response.message

    @skill_method
    def _move_ee_to_pose(self, ee_target: Pose3D, label: str) -> SkillResult:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :param label: Label of the pose used in RViz
        :return: Tuple containing a Boolean skill success and outcome message
        """
        self._pose_broadcaster.poses[label] = ee_target

        query = MotionPlanningQuery(ee_target)

        traj = self._motion_planner.compute_motion_plan(query)
        if traj is None:
            return False, "❌ No plan found."

        with self._console.status("Executing trajectory...", spinner="dots"):
            self._arm.execute_motion_plan(traj)

        return True, "✅ Reached target pose."  # TODO: Doesn't actually check to verify
