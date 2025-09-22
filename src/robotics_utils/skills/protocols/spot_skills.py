"""Define a skills protocol for the Spot mobile manipulator."""

from __future__ import annotations

import time
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import rospy
from rich.console import Console
from spot_skills.srv import (
    Float64Service,
    Float64ServiceRequest,
    Float64ServiceResponse,
    GetRGBImages,
    GetRGBImagesRequest,
    GetRGBImagesResponse,
    NameService,
    NameServiceRequest,
    NameServiceResponse,
    NavigateToPose,
    NavigateToPoseRequest,
    NavigateToPoseResponse,
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

from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D, Waypoints
from robotics_utils.kinematics.kinematic_tree import KinematicTree
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
from robotics_utils.skills.skill_templates import PickTemplate, PlacementSurface, PlaceTemplate


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
        self._rgb_images_caller = ServiceCaller[GetRGBImagesRequest, GetRGBImagesResponse](
            "spot/get_rgb_images",
            GetRGBImages,
        )
        self._open_door_caller = ServiceCaller[OpenDoorRequest, OpenDoorResponse](
            "spot/open_door",
            OpenDoor,
        )

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
        self._scene = PlanningSceneManager(move_group_name=self._arm.name)
        self._motion_planner = MoveItMotionPlanner(self._arm, self._scene)

        if config.take_control:
            self._take_control()

        self._pose_broadcaster = PoseBroadcastThread()

        self._kinematic_tree = KinematicTree.from_yaml(config.env_yaml)

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
    def open_door(self, is_pull: bool, hinge_on_left: bool) -> SkillResult:
        """Open a door using Spot's built-in skill.

        :param is_pull: Whether the door opens by pulling toward Spot
        :param hinge_on_left: Whether the door's hinge is on the left, from Spot's perspective
        """
        request = OpenDoorRequest(is_pull=is_pull, hinge_on_left=hinge_on_left)
        response = self._open_door_caller(request)
        if response is None:
            return False, "OpenDoor service response was None."

        return response.success, response.message

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

        self._pose_broadcaster.poses["pre_grasp_drawer"] = pre_pose
        pre_success, pre_outcome = self._move_ee_to_pose(pre_pose)
        if not pre_success:
            return False, pre_outcome

        self._pose_broadcaster.poses["grasp_drawer"] = grasp_pose
        grasp_success, grasp_outcome = self._move_ee_to_pose(grasp_pose)
        if not grasp_success:
            return False, grasp_outcome

        self._gripper.close()
        time.sleep(3)  # Wait 3 seconds for the gripper to settle

        self._pose_broadcaster.poses["pull_drawer"] = pull_pose
        pull_success, pull_outcome = self._move_ee_to_pose(pull_pose)
        if not pull_success:
            return False, pull_outcome

        self._gripper.open()

        # After letting go, pull the gripper back (+10 cm x) and then stow the arm
        post_pull_pose = deepcopy(pull_pose)
        post_pull_pose.position.x += 0.1

        self._pose_broadcaster.poses["post_pull_drawer"] = post_pull_pose
        post_success, post_outcome = self._move_ee_to_pose(post_pull_pose)
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
    def look_for_object(self, object_name: str, duration_s: float) -> SkillResult:
        """Look for the named object using Spot's gripper camera.

        :param object_name: Name of the object looked for
        :param duration_s: Duration (seconds) to wait during pose estimation
        :return: Tuple containing a Boolean skill success and outcome message
        """
        self._gripper.open()

        object_pose_estimate = self._fiducial_tracker.reestimate(object_name, duration_s)
        if object_pose_estimate is None:
            return False, f"Could not find an updated pose estimate for object '{object_name}'."

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

        # Before trying to pick, see if we can update our pose estimate for the object
        self.look_for_object(object_name, duration_s=5.0)

        # Check if we should "flip" the grasp pose to account for which side Spot is on
        pose_o_spot = TransformManager.lookup_transform("body", object_name)
        need_to_flip = pose_o_spot.position.y < 0
        if need_to_flip:
            self._console.print("Flipping grasp pose due to Spot's body location...")

            flipped_pose_o_g = template.pose_o_g @ Pose3D.from_xyz_rpy(roll_rad=3.14159)
            self._pose_broadcaster.poses["flipped_pose_o_g"] = flipped_pose_o_g
            self._pose_broadcaster.poses["pose_o_g"] = template.pose_o_g

            template = replace(template, pose_o_g=flipped_pose_o_g)

        # Compute the pre-grasp pose
        pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-template.pre_grasp_x_m)
        pose_o_pregrasp = template.pose_o_g @ pose_g_pregrasp

        # Compute the post-grasp pose
        pose_w_g = TransformManager.convert_to_frame(template.pose_o_g, target_frame=DEFAULT_FRAME)
        pose_w_postg = deepcopy(pose_w_g)
        pose_w_postg.position.z += template.post_grasp_lift_m

        self._pose_broadcaster.poses[f"pre_grasp_{object_name}"] = pose_o_pregrasp
        self._pose_broadcaster.poses[f"grasp_{object_name}"] = pose_w_g
        self._pose_broadcaster.poses[f"post_grasp_{object_name}"] = pose_w_postg

        self._gripper.open()

        pre_ok, pre_msg = self._move_ee_to_pose(pose_o_pregrasp)
        if not pre_ok:
            return False, pre_msg

        grasp_ok, grasp_msg = self._move_ee_to_pose(template.pose_o_g)
        if not grasp_ok:
            return False, grasp_msg

        self._gripper.close()

        post_ok, post_msg = self._move_ee_to_pose(pose_w_postg)
        if not post_ok:
            return False, post_msg

        if template.stow_carry:
            stow_ok, stow_msg = self.stow_arm()
            if not stow_ok:
                return False, stow_msg

        else:
            self._pose_broadcaster.poses[f"carry_{object_name}"] = template.pose_b_carry
            carry_ok, carry_msg = self._move_ee_to_pose(template.pose_b_carry)
            if not carry_ok:
                return False, carry_msg

        return True, f"Successfully picked object '{object_name}'."

    @skill_method
    def place(self, held_obj_name: str, template: PlaceTemplate) -> SkillResult:
        """Place a held object onto a surface based on the given template.

        :param held_obj_name: Name of the object currently held by the robot
        :param template: Template for a 'Place' skill
        :return: Tuple containing a Boolean skill success and outcome message
        """
        # First, navigate to the base pose used for the skill
        nav_success, nav_msg = self.go_to_pose(template.pose_s_base)
        if not nav_success:
            return False, nav_msg

        # Second, compute and visualize the pre-place, place, and post-place poses
        ee_name = self._arm.ee_link_name
        pose_o_ee = TransformManager.lookup_transform(ee_name, held_obj_name)
        if pose_o_ee is None:
            return False, f"Could not look up transform of {ee_name} w.r.t. {held_obj_name}."
        place_pose_s_ee = template.place_pose_s_o @ pose_o_ee

        place_pose_w_ee = TransformManager.convert_to_frame(place_pose_s_ee, DEFAULT_FRAME)
        pre_place_w_ee = deepcopy(place_pose_w_ee)
        pre_place_w_ee.position.z += template.pre_place_lift_m

        pose_p_postp = Pose3D.from_xyz_rpy(x=-template.post_place_x_m)  # Post-place w.r.t. place
        postplace_pose_s_ee = place_pose_s_ee @ pose_p_postp

        # Name the poses for being broadcast as /tf frames
        self._pose_broadcaster.poses[f"pre_place_{held_obj_name}"] = pre_place_w_ee
        self._pose_broadcaster.poses[f"place_{held_obj_name}"] = place_pose_s_ee
        self._pose_broadcaster.poses[f"post_place_{held_obj_name}"] = postplace_pose_s_ee

        # Finally, execute the rest of the skill as planned
        preplace_ok, preplace_msg = self._move_ee_to_pose(pre_place_w_ee)
        if not preplace_ok:
            return False, preplace_msg

        place_ok, place_msg = self._move_ee_to_pose(place_pose_s_ee)
        if not place_ok:
            return False, place_msg

        self.open_gripper()

        postplace_ok, postplace_msg = self._move_ee_to_pose(postplace_pose_s_ee)
        if not postplace_ok:
            return False, postplace_msg

        stow_ok, stow_msg = self.stow_arm()
        if not stow_ok:
            return False, stow_msg

        return True, f"Successfully placed object '{held_obj_name}'."

    @skill_method
    def _take_control(self) -> SkillResult:
        """Take control of the Spot robot (and unlock its arm, if necessary)."""
        if not trigger_service(self._take_control_srv_name):
            return False, "Unable to take control of Spot."
        if not trigger_service(self._unlock_arm_srv_name):
            return False, "Unable to unlock Spot's arm."
        return True, "Successfully took control of Spot and unlocked Spot's arm."

    @skill_method
    def _get_rgb_image(self, camera_name: str, output_dir: Path) -> SkillResult:
        """Take an RGB image using the named camera on Spot and save it to the given path.

        :param camera_name: Name of a robot camera
        :param output_dir: Directory into which the captured image is saved
        :return: Tuple containing a Boolean skill success and outcome message
        """

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
    def open_gripper(self) -> SkillResult:
        """Open Spot's gripper."""
        self._gripper.open()
        return True, "Successfully opened Spot's gripper."

    @skill_method
    def close_gripper(self) -> SkillResult:
        """Close Spot's gripper."""
        self._gripper.close()
        return True, "Successfully closed Spot's gripper."

    @skill_method
    def _move_ee_to_pose(self, ee_target: Pose3D) -> SkillResult:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :return: Tuple containing a Boolean skill success and outcome message
        """
        query = MotionPlanningQuery(ee_target)

        traj = self._motion_planner.compute_motion_plan(query)
        if traj is None:
            return False, "❌ No plan found."

        with self._console.status("Executing trajectory...", spinner="dots"):
            self._arm.execute_motion_plan(traj)

        return True, "✅ Reached target pose."  # TODO: Doesn't actually check to verify

    @skill_method
    def _load_planning_scene(self, env_yaml: Path) -> SkillResult:
        """Update the MoveIt planning scene based on a YAML file.

        :param env_yaml: Path to a YAML file specifying an environment state
        :return: Tuple containing a Boolean skill success and outcome message
        """
        tree = KinematicTree.from_yaml(env_yaml)

        # 0) Update the fiducial tracker's "known poses" to the updated state
        for object_name, object_pose in tree.object_poses.items():
            self._fiducial_tracker.known_poses[object_name] = object_pose

        # 1) Ensure that all object poses are updated in /tf
        for object_name, object_pose in tree.object_poses.items():
            TransformManager.broadcast_transform(object_name, object_pose)

        # 2) Synchronize the planning scene with the kinematic tree
        success = self._scene.synchronize_state(tree)
        message = (
            "Planning scene has been updated."
            if success
            else "Unable to update the planning scene."
        )
        return success, message
