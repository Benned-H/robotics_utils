"""Define a protocol for skills on the Boston Dynamics Spot mobile manipulator."""

import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
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

from robotics_utils.io import console
from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D, Waypoints
from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import (
    PoseBroadcastThread,
    ServiceCaller,
    TransformManager,
    TransformRecorder,
    trigger_service,
)
from robotics_utils.ros.msg_conversion import pose_from_msg
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.skills import Outcome, SkillsProtocol, skill_method

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

    robot_name: str = "Spot"

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
        self.robot_name = config.robot_name

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

        self._arm = MoveItManipulator(
            name="arm",
            robot_name=self.robot_name,
            base_frame="body",
            gripper=self._gripper,
        )

        self._pose_broadcaster = PoseBroadcastThread()

        self._kinematic_tree = KinematicTree.from_yaml(config.env_yaml)

    def spin_once(self, duration_s: float = 0.1) -> None:
        """Sleep for the given duration (in seconds) to allow background processing."""
        rospy.sleep(duration_s)  # TODO: When should this be called?

    @skill_method
    def navigate_to_waypoint(self, waypoint: str) -> Outcome:
        """Navigate to the named waypoint using global path planning.

        :param waypoint: Name of a navigation waypoint
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Navigating to waypoint '{waypoint}'...")
        if waypoint not in self._waypoints:
            return Outcome(
                success=False,
                message=(
                    f"Cannot navigate to unknown waypoint: '{waypoint}'. "
                    f"Available waypoints: {self._waypoints.waypoint_names}."
                ),
            )

        request = NameServiceRequest(name=waypoint)
        response = self._nav_to_waypoint_caller(request)

        if response is None:
            return Outcome(False, "NavigateToWaypoint service response was None.")

        return Outcome(response.success, response.message)

    @skill_method
    def undock(self) -> Outcome:
        """Undock Spot from its charging dock."""
        console.print("Undocking Spot...")
        success = trigger_service("spot/undock")
        message = "Successfully undocked Spot." if success else "Unable to undock Spot."
        return Outcome(success, message)

    @skill_method
    def dock(self) -> Outcome:
        """Dock Spot onto its charging dock."""
        console.print("Docking Spot...")
        nav_outcome = self.navigate_to_waypoint("dock")
        if not nav_outcome.success:
            return nav_outcome

        success = trigger_service("spot/dock")
        message = "Successfully docked Spot." if success else "Unable to dock Spot."
        return Outcome(success, message)

    @skill_method
    def stow_arm(self) -> Outcome:
        """Stow Spot's arm."""
        console.print("Stowing Spot's arm...")
        success = trigger_service("spot/stow_arm")
        if success:
            close_outcome = self.close_gripper()
            if not close_outcome.success:
                return close_outcome

        message = "Spot's arm was stowed." if success else "Could not stow Spot's arm."
        return Outcome(success=success, message=message)

    @skill_method
    def erase_board(self) -> Outcome:
        """Erase a whiteboard using a force-controlled trajectory.

        :return: Boolean success indicator and an outcome message
        """
        console.print("Erasing the board...")
        success = trigger_service("spot/erase_board")
        message = "Erased the board." if success else "Unable to erase the board."
        return Outcome(success, message)

    @skill_method
    def open_drawer(
        self,
        pregrasp_pose_ee: Pose3D = Pose3D.from_xyz_rpy(
            x=0.68,
            z=0.56,
            yaw_rad=3.1416,
            ref_frame="black_dresser",
        ),
        grasp_pose_ee: Pose3D = Pose3D.from_xyz_rpy(
            x=0.47,
            z=0.56,
            yaw_rad=3.1416,
            ref_frame="black_dresser",
        ),
        pull_pose_ee: Pose3D = Pose3D.from_xyz_rpy(
            x=0.68,
            z=0.63,
            yaw_rad=3.1416,
            ref_frame="black_dresser",
        ),
        container_name: str = "black_dresser",
    ) -> Outcome:
        """Open a drawer using Spot's end-effector.

        :param pregrasp_pose_ee: Intermediate end-effector pose before Spot grasps the drawer
        :param grasp_pose_ee: Target end-effector pose when Spot grasps the drawer handle
        :param pull_pose_ee: Target end-effector pose after Spot pulls the drawer open
        :param container_name: Name of the container that has the drawer
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Preparing to open the drawer of '{container_name}'...")
        stow_outcome = self.stow_arm()
        if not stow_outcome.success:
            return stow_outcome

        nav_outcome = self.navigate_to_waypoint("open_drawer")
        if not nav_outcome.success:
            return nav_outcome

        open_outcome = self.open_gripper()  # Open Spot's gripper before it nears the dresser
        if not open_outcome.success:
            return open_outcome

        self._pose_broadcaster.poses["pregrasp_drawer"] = pregrasp_pose_ee
        pre_outcome = self._move_ee_to_pose(pregrasp_pose_ee)
        if not pre_outcome.success:
            return pre_outcome

        self._pose_broadcaster.poses["grasp_drawer"] = grasp_pose_ee
        grasp_outcome = self._move_ee_to_pose(grasp_pose_ee)
        if not grasp_outcome.success:
            return grasp_outcome

        close_outcome = self.close_gripper()
        if not close_outcome.success:
            return close_outcome
        time.sleep(3)  # Wait a few seconds for the gripper to settle

        self._pose_broadcaster.poses["pull_drawer"] = pull_pose_ee
        pull_outcome = self._move_ee_to_pose(pull_pose_ee)
        if not pull_outcome.success:
            return pull_outcome

        open_outcome = self.open_gripper()
        if not open_outcome.success:
            return open_outcome

        # After letting go, pull the gripper farther back and then stow the arm
        # Compute the post-pull end-effector pose based on its pull pose
        post_pull_x = pull_pose_ee.position.x + 0.1
        post_pull_position = replace(pull_pose_ee.position, x=post_pull_x)
        post_pull_pose = replace(pull_pose_ee, position=post_pull_position)

        self._pose_broadcaster.poses["postpull_drawer"] = post_pull_pose
        post_outcome = self._move_ee_to_pose(post_pull_pose)
        if not post_outcome.success:
            return post_outcome

        stow_outcome = self.stow_arm()
        if not stow_outcome.success:
            return stow_outcome

        # Update the mesh of the opened drawer's container in MoveIt
        self._kinematic_tree.open_container(container_name)

        return Outcome(success=True, message="Successfully opened the drawer.")

    @skill_method
    def _take_control(self) -> Outcome:
        """Take control of the Spot and unlock its arm, if necessary."""
        console.print("Taking control of Spot...")
        if not trigger_service("spot/take_control"):
            return Outcome(False, "Unable to take control of Spot.")
        if not trigger_service("spot/unlock_arm"):
            return Outcome(False, "Unable to unlock Spot's arm.")
        return Outcome(True, "Successfully took control of Spot and unlocked Spot's arm.")

    @skill_method
    def open_gripper(self) -> Outcome:
        """Open Spot's gripper."""
        console.print("Opening Spot's gripper...")
        success = self._gripper.open()
        message = "Opened Spot's gripper." if success else "Could not open Spot's gripper."
        return Outcome(success, message)

    @skill_method
    def close_gripper(self) -> Outcome:
        """Close Spot's gripper."""
        console.print("Closing Spot's gripper...")
        success = self._gripper.close()
        message = "Closed Spot's gripper." if success else "Could not close Spot's gripper."
        return Outcome(success, message)

    @skill_method
    def _move_ee_to_pose(self, ee_target: Pose3D) -> Outcome:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Moving Spot's end-effector to pose: {ee_target.to_xyz_rpy()}")
        query = MotionPlanningQuery(ee_target)

        plan_msg = self._arm.motion_planner.compute_motion_plan(query, self._arm.planning_scene)
        if plan_msg is None:
            return Outcome(success=False, message="No motion plan found.")

        with console.status("Executing trajectory..."):
            success = self._arm.execute_trajectory_msg(plan_msg)

        return Outcome(success=success, message="Motion plan has been executed.")

    @skill_method
    def _lookup_pose(self, child_frame: str, parent_frame: str) -> Outcome:
        """Look up the pose of a frame w.r.t. a reference frame.

        :param child_frame: Name of the frame whose pose is found
        :param parent_frame: Reference frame used for the lookup
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Finding pose of frame '{child_frame}' w.r.t. frame '{parent_frame}'...")
        request = PoseLookupRequest()
        request.child_frame = child_frame
        request.parent_frame = parent_frame

        response = self._pose_lookup_caller(request)
        if response is None:
            return Outcome(False, "Pose lookup service response was None.")

        output_pose = None
        if response.success:
            output_pose = pose_from_msg(response.relative_pose)
            console.print(f"[cyan]Pose of {child_frame} w.r.t. {parent_frame}: {output_pose}.[/]")

        return Outcome(response.success, response.message, output=output_pose)

    @skill_method
    def _playback_trajectory(self, yaml_path: Path) -> Outcome:
        """Play back a relative end-effector trajectory loaded from file.

        :param yaml_path: YAML file specifying a relative end-effector trajectory
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Playing back trajectory from file: {yaml_path}...")
        request = PlaybackTrajectoryRequest(str(yaml_path))
        response = self._traj_playback_caller(request)

        if response is None:
            return Outcome(False, "Trajectory playback service response was None.")

        return Outcome(response.success, response.message)

    @skill_method
    def open_door(
        self,
        *,
        is_pull: bool = False,
        hinge_on_left: bool = True,
        body_pitch_rad: float = -0.1,
        door_offset_m: float = 0.9,
        ray_search_dist_m: float = 0.15,
    ) -> Outcome:
        """Open a door using the Spot SDK.

        :param is_pull: Whether the door opens by pulling toward Spot
        :param hinge_on_left: Whether the door's hinge is on the left, from Spot's perspective
        :param body_pitch_rad: Pitch (radians) of Spot's body when taking the door handle image
        :param door_offset_m: Distance (m) Spot stands from the door when searching for the handle
        :param ray_search_dist_m: Distance (m) searched along the ray to the door handle
        :return: Boolean success indicator and an outcome message
        """
        console.print("Commanding Spot to open the door...")
        request = OpenDoorRequest(
            body_pitch_rad,
            is_pull,
            hinge_on_left,
            door_offset_m,
            ray_search_dist_m,
        )
        response = self._open_door_caller(request)
        if response is None:
            return Outcome(False, "OpenDoor service response was None.")
        return Outcome(response.success, response.message)

    @skill_method
    def _record_trajectory(
        self,
        output_path: Path,
        fixed_frame: str,
        tracked_frame: str,
        *,
        overwrite: bool = False,
    ) -> Outcome:
        """Record an end-effector relative trajectory and save it to YAML.

        :param output_path: Path to the output YAML file
        :param fixed_frame: Fixed reference frame in which the relative trajectory is expressed
        :param tracked_frame: Frame tracked during the recording
        :param overwrite: Whether to allow overwriting the output path, defaults to False
        :return: Boolean success indicator and an outcome message
        """
        if not overwrite and output_path.exists():
            return Outcome(False, f"Cannot overwrite existing output path: {output_path}")

        recorder = TransformRecorder(reference_frame=fixed_frame, tracked_frame=tracked_frame)

        rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
        try:
            while not rospy.is_shutdown_requested():
                recorder.update()
                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            return Outcome(success=False, message=f"Failed to record trajectory: {ros_exc}")
        finally:
            recorder.save_to_file(output_path=output_path)  # TODO: Add before/after configurations

        return Outcome(success=True, message=f"Saved to YAML file: {output_path}")

    @skill_method
    def _grasp_object(self, object_name: str) -> Outcome:
        """Grasp the named object by closing Spot's gripper.

        :param object_name: Name of the object to be grasped
        :return: Boolean success indicator and an outcome message
        """
        return self._arm.grasp(object_name=object_name)

    @skill_method
    def _release_object(self, object_name: str) -> Outcome:
        """Release the named object by opening Spot's gripper.

        :param object_name: Name of the object to be released
        :return: Boolean success indicator and an outcome message
        """
        return self._arm.release(object_name=object_name)

    @skill_method
    def pick(
        self,
        object_name: str,
        open_gripper_rad: float,
        pre_grasp_x_m: float,
        pose_o_g: Pose3D,
        lift_z_m: float,
        *,
        stow_after: bool,
    ) -> Outcome:
        """Pick the named object using Spot's gripper.

        Steps in the skill:
          1. Open the gripper to the specified angle (`open_gripper_rad`).
          2. Move the end-effector to the pre-grasp pose (computed using `pre_grasp_x_m`).
             The pre-grasp pose is "back" (-x w.r.t. the end-effector frame) from the grasp pose.
          3. Move the end-effector to the grasp pose (`pose_o_g` in the object frame).
          4. Close the gripper, thus grasping the object.
          5. Move the end-effector to the post-grasp pose (computed using `post_grasp_z_m`).
          6. Stow Spot's arm, if requested (`stow_after`).

        :param object_name: Name of the object to be picked
        :param open_gripper_rad: Angle (radians) to open the gripper for approaching the object
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :param pose_o_g: Object-relative end-effector pose used to grasp the object
        :param lift_z_m: Offset (m) of the post-grasp pose "up" (+z) w.r.t. the world frame
        :param stow_after: If True, stow the arm after picking the object
        :return: Boolean success indicator and an outcome message
        """
        if object_name != pose_o_g.ref_frame:
            g_frame = pose_o_g.ref_frame
            console.print(f"[yellow]Warning: Grasp pose given in frame '{g_frame}'.[/]")
            pose_o_g = TransformManager.convert_to_frame(pose_o_g, target_frame=object_name)

        # TODO: Possibly attempt to re-pose-estimate the object before or during the skill

        self._pose_broadcaster.poses["pose_o_g"] = pose_o_g  # Grasp pose w.r.t. object frame

        # Check if we should "flip" the grasp pose to account for which side Spot is on
        pose_o_spot = TransformManager.lookup_transform("body", object_name)
        if pose_o_spot is None:
            return Outcome(
                success=False,
                message=f"Unable to look up pose of 'body' frame w.r.t. '{object_name}'.",
            )
        need_to_flip = pose_o_spot.position.y < 0.0
        if need_to_flip:
            console.print(f"Flipping grasp pose due to Spot's location w.r.t. '{object_name}'...")
            flipped_pose_o_g = pose_o_g @ Pose3D.from_xyz_rpy(roll_rad=np.pi)
            self._pose_broadcaster.poses["flipped_pose_o_g"] = flipped_pose_o_g
            pose_o_g = flipped_pose_o_g

        # Compute the pre-grasp and post-grasp poses using the updated grasp pose
        pose_g_pregrasp = Pose3D.from_xyz_rpy(x=-abs(pre_grasp_x_m))  # pre-grasp w.r.t. grasp
        pose_o_pregrasp = pose_o_g @ pose_g_pregrasp  # pre-grasp w.r.t. object

        pose_w_g = TransformManager.convert_to_frame(pose_o_g, target_frame=DEFAULT_FRAME)
        postgrasp_z = pose_w_g.position.z + lift_z_m  # z-coordinate w.r.t. world
        postgrasp_xyz = replace(pose_w_g.position, z=postgrasp_z)
        pose_w_postgrasp = replace(pose_w_g, position=postgrasp_xyz)  # post-grasp w.r.t. world

        self._pose_broadcaster.poses[f"pre_grasp_{object_name}"] = pose_o_pregrasp
        self._pose_broadcaster.poses[f"grasp_{object_name}"] = pose_o_g
        self._pose_broadcaster.poses[f"post_grasp_{object_name}"] = pose_w_postgrasp

        if not self._gripper.move_to_angle_rad(target_rad=open_gripper_rad):
            return Outcome(False, f"Unable to pick '{object_name}' because gripper didn't open.")

        pre_outcome = self._move_ee_to_pose(ee_target=pose_o_pregrasp)
        if not pre_outcome.success:
            return pre_outcome

        move_to_grasp_outcome = self._move_ee_to_pose(ee_target=pose_o_g)
        if not move_to_grasp_outcome.success:
            return move_to_grasp_outcome

        grasp_outcome = self._grasp_object(object_name)
        if not grasp_outcome.success:
            return grasp_outcome

        post_outcome = self._move_ee_to_pose(ee_target=pose_w_postgrasp)
        if not post_outcome.success:
            return post_outcome

        if stow_after:
            stow_outcome = self.stow_arm()
            if not stow_outcome.success:
                return stow_outcome

        return Outcome(success=True, message=f"Successfully picked object '{object_name}'.")

    @skill_method
    def place(
        self,
        object_name: str,
        surface_name: str,
        pre_place_lift_m: float,
        place_pose_s_o: Pose3D,
        post_place_x_m: float,
        *,
        stow_after: bool,
    ) -> Outcome:
        """Place the named object onto the named surface.

        :param object_name: Name of the held object to be placed
        :param surface_name: Name of the surface onto which the object is placed
        :param pre_place_lift_m: Offset (+z meters) from the placement pose to the pre-place pose
        :param place_pose_s_o: Placement pose of the object w.r.t. the surface frame
        :param post_place_x_m: Offset (abs. m) of the post-place pose "back" (-x) from place pose
        :param stow_after: If True, stow Spot's arm after placing the object
        :return: Boolean success indicator and outcome message
        """
        grasped_pose_o_ee = TransformManager.lookup_transform(self._arm.ee_link_name, object_name)
        if grasped_pose_o_ee is None:
            return Outcome(False, f"Unable to place '{object_name}' due to pose lookup failure.")
        place_pose_s_ee = place_pose_s_o @ grasped_pose_o_ee  # end-effector w.r.t. surface

        place_pose_w_ee = TransformManager.convert_to_frame(place_pose_s_ee, DEFAULT_FRAME)
        preplace_z = place_pose_w_ee.position.z + pre_place_lift_m
        preplace_position = replace(place_pose_w_ee.position, z=preplace_z)
        preplace_pose_w_ee = replace(place_pose_w_ee, position=preplace_position)

        postplace_wrt_place = Pose3D.from_xyz_rpy(x=-abs(post_place_x_m))
        postplace_pose_s_ee = place_pose_s_ee @ postplace_wrt_place

        self._pose_broadcaster.poses[f"preplace_{object_name}"] = preplace_pose_w_ee
        self._pose_broadcaster.poses[f"place_{object_name}"] = place_pose_s_ee
        self._pose_broadcaster.poses[f"postplace_{object_name}"] = postplace_pose_s_ee

        preplace_outcome = self._move_ee_to_pose(preplace_pose_w_ee)
        if not preplace_outcome.success:
            return preplace_outcome

        move_to_place_outcome = self._move_ee_to_pose(place_pose_s_ee)
        if not move_to_place_outcome.success:
            return move_to_place_outcome

        # Look up the pose of the end-effector w.r.t. the surface before releasing
        curr_pose_s_ee = TransformManager.lookup_transform(self._arm.ee_link_name, surface_name)
        if curr_pose_s_ee is None:
            return Outcome(False, f"Unable to place '{object_name}' due to pose lookup failure.")
        pose_ee_o = grasped_pose_o_ee.inverse(pose_frame=self._arm.ee_link_name)
        curr_pose_s_o = curr_pose_s_ee @ pose_ee_o

        # Release the object (i.e., open the gripper, then update the kinematic state)
        release_outcome = self._release_object(object_name)
        if not release_outcome.success:
            return release_outcome

        TransformManager.broadcast_transform(object_name, curr_pose_s_o)

        postplace_outcome = self._move_ee_to_pose(postplace_pose_s_ee)
        if not postplace_outcome.success:
            return postplace_outcome

        if stow_after:
            stow_outcome = self.stow_arm()
            if not stow_outcome.success:
                return stow_outcome

        return Outcome(success=True, message=f"Placed '{object_name}' on '{surface_name}'.")
