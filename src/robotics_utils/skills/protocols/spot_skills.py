"""Define a protocol for skills on the Boston Dynamics Spot mobile manipulator."""

import time
from dataclasses import dataclass, replace
from pathlib import Path

import rospy
from rich.prompt import Prompt
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
    ProbeSurface,
    ProbeSurfaceRequest,
    ProbeSurfaceResponse,
)

from robotics_utils.geometry import Point3D
from robotics_utils.io import console
from robotics_utils.kinematics import KinematicTree, Waypoints
from robotics_utils.motion_planning import MotionPlanningQuery
from robotics_utils.motion_planning.grasping import PickPoses
from robotics_utils.robots import GripperAngleLimits
from robotics_utils.ros import (
    PlanningSceneManager,
    PoseBroadcastThread,
    ServiceCaller,
    TransformManager,
    TransformRecorder,
    trigger_service,
)
from robotics_utils.ros.msg_conversion import point_to_vector3_msg, pose_from_msg
from robotics_utils.ros.robots import MoveItManipulator, ROSAngularGripper
from robotics_utils.skills import Outcome, SkillsProtocol, skill_method
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D

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

    planning_frame: str = "map"
    """Reference frame used for MoveIt motion planning."""

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

        self._pause_est_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/pose_estimation/pause",
            NameService,
        )
        self._resume_est_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/pose_estimation/resume",
            NameService,
        )  # TODO: When should this be called?

        self._probe_caller = ServiceCaller[ProbeSurfaceRequest, ProbeSurfaceResponse](
            "spot/probe_surface",
            ProbeSurface,
        )

        self._gripper = ROSAngularGripper(
            limits=GripperAngleLimits(
                open_rad=SPOT_GRIPPER_OPEN_RAD,
                closed_rad=SPOT_GRIPPER_CLOSED_RAD,
            ),
            grasping_group="gripper",
            action_name="gripper_controller/gripper_action",
        )

        self.planning_frame = config.planning_frame
        self._arm = MoveItManipulator(
            name="arm",
            robot_name=config.robot_name,
            base_frame="body",
            planning_frame=self.planning_frame,
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
            return Outcome(success=False, message="NavigateToWaypoint service response was None.")

        return Outcome(response.success, response.message)

    @skill_method
    def undock(self) -> Outcome:
        """Undock Spot from its charging dock."""
        console.print("Undocking Spot...")
        return trigger_service("spot/undock")

    @skill_method
    def dock(self) -> Outcome:
        """Dock Spot onto its charging dock."""
        console.print("Docking Spot...")
        nav_outcome = self.navigate_to_waypoint("dock")
        if not nav_outcome.success:
            return nav_outcome

        return trigger_service("spot/dock")

    @skill_method
    def stow_arm(self, *, close_gripper: bool = False) -> Outcome:
        """Stow Spot's arm and optionally close Spot's gripper.

        :param close_gripper: If True, close Spot's gripper after stowing (defaults to False)
        :return: Boolean success indicator and an outcome message
        """
        console.print("Stowing Spot's arm...")
        stow_outcome = trigger_service("spot/stow_arm")
        if stow_outcome.success and close_gripper:
            close_outcome = self.close_gripper()
            if not close_outcome.success:
                return close_outcome

        return stow_outcome

    @skill_method
    def erase_board(self) -> Outcome:
        """Erase a whiteboard using a force-controlled trajectory.

        :return: Boolean success indicator and an outcome message
        """
        console.print("Erasing the board...")
        return trigger_service("spot/erase_board")

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

        pre_outcome = self._move_ee_to_pose(pregrasp_pose_ee, "pregrasp_drawer")
        if not pre_outcome.success:
            return pre_outcome

        grasp_outcome = self._move_ee_to_pose(grasp_pose_ee, "grasp_drawer")
        if not grasp_outcome.success:
            return grasp_outcome

        close_outcome = self.close_gripper()
        if not close_outcome.success:
            return close_outcome
        time.sleep(3)  # Wait a few seconds for the gripper to settle

        pull_outcome = self._move_ee_to_pose(pull_pose_ee, "pull_drawer")
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

        post_outcome = self._move_ee_to_pose(post_pull_pose, "postpull_drawer")
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
        control_outcome = trigger_service("spot/take_control")
        if not control_outcome.success:
            return control_outcome

        unlock_outcome = trigger_service("spot/unlock_arm")
        if not unlock_outcome.success:
            return unlock_outcome

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
    def _move_ee_to_pose(
        self,
        ee_target: Pose3D,
        target_name: str = "ee_target",
        ignored_objects: str = "",
    ) -> Outcome:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :param target_name: Name describing the end-effector target pose
        :param ignored_objects: Comma-separated list of object names to ignore (defaults to "")
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Moving Spot's end-effector to '{target_name}': {ee_target}")
        self._pose_broadcaster.poses[target_name] = ee_target

        ignored_objects_set = set()
        if ignored_objects.strip():
            ignored_objects_set = set(ignored_objects.strip().split(","))
        query = MotionPlanningQuery(ee_target, ignored_objects=ignored_objects_set)

        plan_msg = self._arm.planner.compute_motion_plan(query)
        if plan_msg is None:
            return Outcome(success=False, message="No motion plan found.")

        with console.status("Executing trajectory..."):
            success = self._arm.execute_trajectory_msg(plan_msg)

        message = "Motion plan has been executed." if success else "Could not execute motion plan."

        return Outcome(success=success, message=message)

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
        console.print(f"Grasping object '{object_name}'...")
        return self._arm.grasp(object_name=object_name)

    @skill_method
    def _release_object(self, object_name: str) -> Outcome:
        """Release the named object by opening Spot's gripper.

        :param object_name: Name of the object to be released
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Releasing object '{object_name}'...")
        return self._arm.release(object_name=object_name)

    @skill_method
    def pick(
        self,
        object_name: str = "eraser1",
        view_pose_o_ee: Pose3D = Pose3D.from_xyz_rpy(
            x=-0.05,
            z=0.5,
            pitch_rad=1.5708,
            ref_frame="eraser1",
        ),
        pre_grasp_rad: float = -0.9,
        pre_grasp_x_m: float = 0.1,
        pose_o_g: Pose3D = Pose3D.from_xyz_rpy(
            x=-0.02,
            z=0.25,
            pitch_rad=1.5708,
            ref_frame="eraser1",
        ),
        lift_z_m: float = 0.1,
        *,
        pauses: bool = False,
        yaw_symmetric: bool = True,
        stow_after: bool = True,
    ) -> Outcome:
        """Pick the named object using Spot's gripper.

        :param object_name: Name of the object to be picked
        :param view_pose_o_ee: End-effector pose (w.r.t. the object) used to view the object
        :param pre_grasp_rad: Angle (radians) to open the gripper before grasping
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :param pose_o_g: Object-relative end-effector pose used to grasp the object
        :param lift_z_m: Offset (m) of the post-grasp pose "up" (+z) w.r.t. the world frame
        :param pauses: If True, the skill will pause until user input between each major motion
        :param yaw_symmetric: Indicates that the object is rotationally symmetric about its z-axis
        :param stow_after: If True, stow the arm after picking the object
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Picking object '{object_name}'...")

        # 1. Fully open the gripper, move to the view pose, and re-pose-estimate the object
        open_outcome = self.open_gripper()
        if not open_outcome.success:
            return open_outcome

        # Reflect the given view pose if it doesn't have an IK solution
        if self._arm.compute_ik(view_pose_o_ee) is None and yaw_symmetric:
            rotate_object = Pose3D.from_xyz_rpy(yaw_rad=3.14159, ref_frame=object_name)
            view_pose_o_ee = rotate_object @ view_pose_o_ee

        view_outcome = self._move_ee_to_pose(view_pose_o_ee, f"view_{object_name}")
        if not view_outcome.success:
            return view_outcome

        estimate_outcome = self.estimate_pose(object_name, duration_s=5.0)
        if not estimate_outcome.success:
            return estimate_outcome

        # 2. Identify which candidate grasp pose to use, if the object is symmetric
        if object_name != pose_o_g.ref_frame:
            console.print(f"[yellow]Warning: Grasp pose given in frame '{pose_o_g.ref_frame}'.[/]")
            pose_o_g = TransformManager.convert_to_frame(pose_o_g, target_frame=object_name)
        self._pose_broadcaster.poses["pose_o_g"] = pose_o_g  # Grasp pose w.r.t. object frame

        candidates = [pose_o_g]
        if yaw_symmetric:
            rotate_object = Pose3D.from_xyz_rpy(yaw_rad=3.14159, ref_frame=object_name)
            alternative_pose_o_g = rotate_object @ pose_o_g
            self._pose_broadcaster.poses["alternative_pose_o_g"] = alternative_pose_o_g
            candidates.append(alternative_pose_o_g)

        # Select a grasp pose with IK solutions for its pre-grasp and post-grasp poses
        valid_grasp = PickPoses.select_grasp_pose(candidates, self._arm, pre_grasp_x_m, lift_z_m)

        if valid_grasp is None:
            return Outcome(
                success=False,
                message=f"Cannot pick '{object_name}' because no grasp poses were valid.",
            )

        valid_poses = PickPoses.from_grasp_pose(valid_grasp, pre_grasp_x_m, lift_z_m)
        self._pose_broadcaster.poses[f"pre_grasp_{object_name}"] = valid_poses.pregrasp_pose
        self._pose_broadcaster.poses[f"grasp_{object_name}"] = valid_poses.grasp_pose
        self._pose_broadcaster.poses[f"post_grasp_{object_name}"] = valid_poses.postgrasp_pose

        # 3. Open the gripper to prepare for picking
        if not self._gripper.move_to_angle_rad(pre_grasp_rad):
            return Outcome(
                success=False,
                message=f"Unable to pick '{object_name}' because the gripper didn't open.",
            )

        # 4. Move the end-effector to the pre-grasp pose ("back" from the grasp pose)
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the pre-grasp pose")

        pre_outcome = self._move_ee_to_pose(valid_poses.pregrasp_pose)
        if not pre_outcome.success:
            return pre_outcome

        # 5. Move the end-effector to the grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the grasp pose")

        move_to_grasp_outcome = self._move_ee_to_pose(valid_poses.grasp_pose)
        if not move_to_grasp_outcome.success:
            return move_to_grasp_outcome

        # 6. Grasp the object by closing the gripper
        if pauses:
            Prompt.ask(f"Press [bold]Enter[/] to grasp [cyan]'{object_name}'[/]")

        grasp_outcome = self._grasp_object(object_name)
        if not grasp_outcome.success:
            return grasp_outcome

        # 7. Move the end-effector to the post-grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the post-grasp pose")

        post_outcome = self._move_ee_to_pose(valid_poses.postgrasp_pose)
        if not post_outcome.success:
            return post_outcome

        # 8. Stow Spot's arm, if requested
        if stow_after:
            if pauses:
                Prompt.ask("Press [bold]Enter[/] to stow Spot's arm")

            time.sleep(1.5)  # Allow arm to settle before stowing
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
        console.print(f"Placing object '{object_name}' on surface '{surface_name}'...")

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

        preplace_outcome = self._move_ee_to_pose(preplace_pose_w_ee, f"preplace_{object_name}")
        if not preplace_outcome.success:
            return preplace_outcome

        move_to_place_outcome = self._move_ee_to_pose(place_pose_s_ee, f"place_{object_name}")
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

        postplace_outcome = self._move_ee_to_pose(postplace_pose_s_ee, f"postplace_{object_name}")
        if not postplace_outcome.success:
            return postplace_outcome

        if stow_after:
            stow_outcome = self.stow_arm()
            if not stow_outcome.success:
                return stow_outcome

        return Outcome(success=True, message=f"Placed '{object_name}' on '{surface_name}'.")

    @skill_method
    def _reset_planning_scene(
        self,
        yaml_path: Path = Path("/docker/spot_skills/src/spot_skills/config/env.yaml"),
    ) -> Outcome:
        """Reset the MoveIt planning scene to the state specified by a YAML file.

        :param yaml_path: Filepath to a YAML file specifying environment geometry
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Resetting the MoveIt planning scene based on the YAML file: {yaml_path}")
        return PlanningSceneManager.reset_per_yaml(
            yaml_path=yaml_path,
            planning_frame=self.planning_frame,
        )

    @skill_method
    def estimate_pose(self, object_name: str, duration_s: float) -> Outcome:
        """Estimate the pose of the named object using AprilTag markers.

        :param object_name: Name of the object to pose estimate
        :param duration_s: Duration (seconds) to pause for pose estimation to update
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Updating the pose estimate for object '{object_name}'...")

        # Ensure the object's pose estimation is active
        resume_outcome = self._resume_object_pose_estimation(object_name)
        if not resume_outcome.success:
            return resume_outcome

        time.sleep(duration_s)

        pause_outcome = self._pause_object_pose_estimation(object_name)
        if not pause_outcome.success:
            return pause_outcome

        return Outcome(success=True, message=f"Updated pose estimate of '{object_name}'.")

    @skill_method
    def _pause_object_pose_estimation(self, object_name: str) -> Outcome:
        """Pause pose estimation for the named object.

        :param object_name: Name of an object
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Pausing pose estimation for object '{object_name}'...")

        request = NameServiceRequest(name=object_name)
        response = self._pause_est_caller(request)

        if response is None:
            return Outcome(success=False, message="Pose estimation pause service returned None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def _resume_object_pose_estimation(self, object_name: str) -> Outcome:
        """Resume pose estimation for the named object.

        :param object_name: Name of an object
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Resuming pose estimation for object '{object_name}'...")

        request = NameServiceRequest(name=object_name)
        response = self._resume_est_caller(request)

        if response is None:
            return Outcome(success=False, message="Pose estimation resume service returned None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def _broadcast_pose(
        self,
        pose: Pose3D = Pose3D.from_xyz_rpy(x=0.21, z=0.015, ref_frame="arm_link_wr1"),
        frame_name: str = "fingertip",
    ) -> Outcome:
        """Broadcast the given pose to /tf until the user presses Enter.

        :param pose: Pose to be broadcast as a /tf frame
        :param frame_name: Name of the published frame
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Broadcasting a transform for reference frame '{frame_name}'...")
        if frame_name in self._pose_broadcaster.poses:
            return Outcome(
                success=False,
                message=f"Cannot broadcast frame '{frame_name}' because it already exists.",
            )

        self._pose_broadcaster.poses[frame_name] = pose
        Prompt.ask(f"[green]Press Enter to stop broadcasting frame '{frame_name}'[/]")

        self._pose_broadcaster.poses.pop(frame_name, None)

        return Outcome(success=True, message=f"Done broadcasting frame '{frame_name}'.")

    @skill_method
    def probe(
        self,
        direction: Point3D = Point3D(1.0, 0.0, 0.0),
        max_distance_m: float = 0.2,
        velocity_mps: float = 0.02,
        force_threshold_n: float = 5.0,
        force_check_hz: float = 50.0,
        num_probes: int = 3,
        probe_interval_s: float = 1.0,
    ) -> Outcome:
        """Use Spot's gripper to probe for a nearby surface.

        :param direction: Direction vector in end-effector frame
        :param max_distance_m: Maximum distance to probe forward (meters)
        :param velocity_mps: Probing velocity (meters per second)
        :param force_threshold_n: Force threshold to detect contact (Newtons)
        :param force_check_hz: Frequency (Hz) to check force sensor
        :param num_probes: Number of probes to perform for averaging
        :param probe_interval_s: Time to wait between probes (seconds)
        :return: Boolean success indicator and outcome message
        """
        console.print("Probing for a surface with Spot's end-effector...")

        request = ProbeSurfaceRequest(
            direction=point_to_vector3_msg(direction),
            max_distance_m=max_distance_m,
            velocity_mps=velocity_mps,
            force_threshold_n=force_threshold_n,
            force_check_hz=force_check_hz,
            num_probes=num_probes,
            probe_interval_s=probe_interval_s,
        )

        response = self._probe_caller(request)
        if response is None:
            return Outcome(success=False, message="Probe service response was None.")

        return Outcome(success=response.success, message=response.message)
