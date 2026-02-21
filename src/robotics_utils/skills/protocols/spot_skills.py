"""Define a protocol for skills on the Boston Dynamics Spot mobile manipulator."""

import time
from dataclasses import replace
from pathlib import Path

import rospy
from moveit_msgs.msg import RobotTrajectory
from rich.prompt import Prompt
from spot_skills.srv import (
    CaptureImageObservation,
    CaptureImageObservationRequest,
    CaptureImageObservationResponse,
    ComputeMotionPlan,
    ComputeMotionPlanRequest,
    ComputeMotionPlanResponse,
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
from robotics_utils.motion_planning.ros import PickPoses, PlanningSceneManager
from robotics_utils.ros import (
    PoseBroadcastThread,
    ServiceCaller,
    TransformManager,
    TransformRecorder,
    trigger_service,
)
from robotics_utils.ros.msg_conversion import (
    point_to_vector3_msg,
    pose_from_msg,
    pose_to_stamped_msg,
)
from robotics_utils.ros.robots import MoveItManipulator
from robotics_utils.skills import Outcome, SkillsProtocol, skill_method
from robotics_utils.spatial import DEFAULT_FRAME, Pose3D
from robotics_utils.states import GraspAttachment

SPOT_GRIPPER_OPEN_RAD = -1.5707
SPOT_GRIPPER_CLOSED_RAD = 0.0
SPOT_GRIPPER_HALF_OPEN_RAD = (SPOT_GRIPPER_OPEN_RAD + SPOT_GRIPPER_CLOSED_RAD) / 2.0


class SpotSkillsProtocol(SkillsProtocol):
    """Define a Python interface for executing Spot skills."""

    def __init__(self, manipulator: MoveItManipulator) -> None:
        """Initialize the Spot skills executor."""
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
        )

        self._probe_caller = ServiceCaller[ProbeSurfaceRequest, ProbeSurfaceResponse](
            "spot/probe_surface",
            ProbeSurface,
        )

        self._grasp_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/grasp_object",
            NameService,
        )
        self._release_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/release_object",
            NameService,
        )
        self._reset_state_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/reset_state",
            NameService,
        )
        self._set_open_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/set_container_open",
            NameService,
        )
        self._image_obs_caller = ServiceCaller[
            CaptureImageObservationRequest,
            CaptureImageObservationResponse,
        ]("spot/capture_image_observation", CaptureImageObservation)

        self._hide_object_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/moveit/hide_object",
            NameService,
        )
        self._unhide_object_caller = ServiceCaller[NameServiceRequest, NameServiceResponse](
            "spot/moveit/unhide_object",
            NameService,
        )

        self._motion_plan_caller = ServiceCaller[
            ComputeMotionPlanRequest,
            ComputeMotionPlanResponse,
        ]("spot/compute_motion_plan", ComputeMotionPlan)

        self._arm = manipulator
        self._gripper = manipulator.gripper

        self._pose_broadcaster = PoseBroadcastThread()

        self._EE_POSES_FOR_POSE_ESTIMATION: dict[str, Pose3D] = {
            "black_dresser": Pose3D.from_xyz_rpy(
                x=0.666,
                y=-0.075,
                z=1.251,
                roll_rad=-0.125,
                pitch_rad=1.139,
                yaw_rad=-2.952,
                ref_frame="black_dresser",
            ),
        }

    @property
    def planning_scene(self) -> PlanningSceneManager:
        """Access the active interface to the MoveIt planning scene."""
        return self._arm.planning_scene

    @skill_method
    def capture_image_observation(
        self,
        camera: str = "hand",
        ref_frame: str = DEFAULT_FRAME,
        image_path: Path = Path("data/spot-images/rgb.jpg"),
    ) -> Outcome:
        """Capture an image observation (image + camera pose) and save it to file.

        :param camera: Camera used to capture the image observation
        :param ref_frame: Reference frame used for the exported camera pose
        :param image_path: Preferred filepath for the exported image
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Capturing an image with camera '{camera}'...")

        request = CaptureImageObservationRequest(camera, ref_frame, str(image_path.resolve()))
        response = self._image_obs_caller(request)

        if response is None:
            return Outcome(success=False, message="CaptureImageObservation response was None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def navigate_to_waypoint(self, waypoint: str = "pick_from_drawer") -> Outcome:
        """Navigate to the named waypoint using global path planning.

        :param waypoint: Name of a navigation waypoint
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Navigating to waypoint '{waypoint}'...")

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
            x=0.45,
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
        # stow_outcome = self.stow_arm()
        # if not stow_outcome.success:
        #     return stow_outcome

        # nav_outcome = self.navigate_to_waypoint("open_drawer")
        # if not nav_outcome.success:
        #     return nav_outcome

        open_outcome = self.open_gripper()  # Open Spot's gripper before it nears the dresser
        if not open_outcome.success:
            return open_outcome

        # Move Spot's end-effector to approximate viewing location for the drawer
        ee_during_pose_est = self._EE_POSES_FOR_POSE_ESTIMATION[container_name]
        pre_estimation_outcome = self._move_ee_to_pose(ee_during_pose_est)
        if not pre_estimation_outcome.success:
            return pre_estimation_outcome

        estimate_outcome = self.estimate_pose(container_name, duration_s=5.0)
        if not estimate_outcome.success:
            return estimate_outcome

        pre_outcome = self._move_ee_to_pose(
            pregrasp_pose_ee,
            # "pregrasp_drawer",
            ignored_objects="black_dresser",
        )
        if not pre_outcome.success:
            return pre_outcome

        grasp_outcome = self._move_ee_to_pose(
            grasp_pose_ee,
            # "grasp_drawer",
            ignored_objects="black_dresser",
        )
        if not grasp_outcome.success:
            return grasp_outcome

        close_outcome = self.close_gripper()
        if not close_outcome.success:
            return close_outcome
        time.sleep(3)  # Wait a few seconds for the gripper to settle

        pull_outcome = self._move_ee_to_pose(
            pull_pose_ee,
            # "pull_drawer",
            ignored_objects="black_dresser",
        )
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

        post_outcome = self._move_ee_to_pose(post_pull_pose)  # , "postpull_drawer")
        if not post_outcome.success:
            return post_outcome

        stow_outcome = self.stow_arm()
        if not stow_outcome.success:
            return stow_outcome

        # Request that the mesh of the opened drawer be updated in MoveIt
        request = NameServiceRequest(name=container_name)
        response = self._set_open_caller(request)
        if response is None:
            return Outcome(success=False, message="Set container open service returned None.")

        return Outcome(success=response.success, message=response.message)

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
        ee_target: Pose3D = Pose3D.from_xyz_rpy(
            x=0.47,
            z=0.56,
            yaw_rad=3.1416,
            ref_frame="black_dresser",
        ),
        target_name: str = "",
        ignored_objects: str = "black_dresser",
        *,
        display_and_pause: bool = False,
    ) -> Outcome:
        """Move Spot's end-effector to the specified pose.

        :param ee_target: End-effector target pose
        :param target_name: Name describing the end-effector target pose
        :param ignored_objects: Comma-separated list of object names to ignore (defaults to "")
        :param display_and_pause: If True, display the trajectory in RViz and pause for user input
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Moving Spot's end-effector to '{target_name}': {ee_target}")
        if target_name:
            self._pose_broadcaster.poses[target_name] = ee_target

        default_pose_outcome = trigger_service("spot/default_body_pose")
        if not default_pose_outcome.success:
            return default_pose_outcome

        # Parse ignored objects into a list
        ignored_objects_list = []
        if ignored_objects.strip():
            ignored_objects_list = [s.strip() for s in ignored_objects.strip().split(",")]

        # Use the centralized motion planning service (SpotROS1Wrapper owns the planning scene)
        request = ComputeMotionPlanRequest()
        request.target_pose = pose_to_stamped_msg(ee_target)
        request.ignored_objects = ignored_objects_list
        request.ignore_all_collisions = False

        console.print(f"Calling compute_motion_plan service for target: {ee_target}")

        response = self._motion_plan_caller(request)

        if response is None:
            return Outcome(success=False, message="Motion planning service returned None.")

        if not response.success:
            return Outcome(success=False, message=response.message)

        # Wrap the JointTrajectory in a RobotTrajectory for execution
        plan_msg = RobotTrajectory()
        plan_msg.joint_trajectory = response.trajectory

        if display_and_pause:
            self._arm.planner.visualize_plan(plan_msg)
            Prompt.ask("Press [bold]Enter[/] to execute the visualized trajectory...")

        with console.status("Executing trajectory..."):
            success = self._arm.execute_trajectory_msg(plan_msg)

        message = "Motion plan has been executed." if success else "Could not execute motion plan."

        return Outcome(success=success, message=message)

    @skill_method
    def _lookup_pose(self, child_frame: str, parent_frame: str = "map") -> Outcome:
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
            return Outcome(success=False, message="Pose lookup service response was None.")

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
            return Outcome(success=False, message="Trajectory playback service response was None.")

        return Outcome(response.success, response.message)

    @skill_method
    def open_door(
        self,
        *,
        is_pull: bool = False,
        hinge_on_left: bool = True,
        navigate_first: bool = True,
        body_pitch_rad: float = -0.1,
        door_offset_m: float = 1.25,
        ray_search_dist_m: float = 0.25,
    ) -> Outcome:
        """Open a door using the Spot SDK.

        :param is_pull: Whether the door opens by pulling toward Spot
        :param hinge_on_left: Whether the door's hinge is on the left, from Spot's perspective
        :param navigate_first: If True, the skill will first navigate to the `open_door` waypoint
        :param body_pitch_rad: Pitch (radians) of Spot's body when taking the door handle image
        :param door_offset_m: Distance (m) Spot stands from the door when searching for the handle
        :param ray_search_dist_m: Distance (m) searched along the ray to the door handle
        :return: Boolean success indicator and an outcome message
        """
        console.print("Commanding Spot to open the door...")

        # Begin by navigating to the "open_door" waypoint
        if navigate_first:
            nav_outcome = self.navigate_to_waypoint("open_door")
            if not nav_outcome.success:
                return Outcome(
                    success=False,
                    message=f"Failed to navigate to 'open_door' waypoint: {nav_outcome}",
                )

        request = OpenDoorRequest(
            body_pitch_rad,
            is_pull,
            hinge_on_left,
            door_offset_m,
            ray_search_dist_m,
        )
        response = self._open_door_caller(request)
        if response is None:
            return Outcome(success=False, message="OpenDoor service response was None.")
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
        response = self._grasp_caller(NameServiceRequest(name=object_name))
        if response is None:
            return Outcome(success=False, message="Grasp object service returned None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def _release_object(self, object_name: str) -> Outcome:
        """Release the named object by opening Spot's gripper.

        :param object_name: Name of the object to be released
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Releasing object '{object_name}'...")
        response = self._release_caller(NameServiceRequest(name=object_name))
        if response is None:
            return Outcome(success=False, message="Release object service returned None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def pick(
        self,
        object_name: str = "eraser1",
        pre_grasp_rad: float = -0.9,
        pre_grasp_x_m: float = 0.15,
        pose_o_g: Pose3D = Pose3D.from_xyz_rpy(
            x=-0.02,
            z=0.24,
            pitch_rad=1.5708,
            ref_frame="eraser1",
        ),
        lift_z_m: float = 0.15,
        *,
        pauses: bool = False,
        yaw_symmetric: bool = True,
        stow_after: bool = True,
    ) -> Outcome:
        """Pick the named object using Spot's gripper.

        :param object_name: Name of the object to be picked
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

        # 1. Fully open the gripper and re-pose-estimate the object
        open_outcome = self.open_gripper()
        if not open_outcome.success:
            return open_outcome

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

        pre_outcome = self._move_ee_to_pose(valid_poses.pregrasp_pose, display_and_pause=pauses)
        if not pre_outcome.success:
            return pre_outcome

        # 5. Move the end-effector to the grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the grasp pose")

        to_grasp_outcome = self._move_ee_to_pose(valid_poses.grasp_pose, display_and_pause=pauses)
        if not to_grasp_outcome.success:
            return to_grasp_outcome

        # 6. Grasp the object by closing the gripper
        if pauses:
            Prompt.ask(f"Press [bold]Enter[/] to grasp [cyan]'{object_name}'[/]")

        grasp_outcome = self._grasp_object(object_name)
        if not grasp_outcome.success:
            return grasp_outcome

        # 7. Move the end-effector to the post-grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the post-grasp pose")

        post_outcome = self._move_ee_to_pose(valid_poses.postgrasp_pose, display_and_pause=pauses)
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
    def pick_from_drawer(
        self,
        object_name: str = "eraser1",
        drawer_name: str = "black_dresser",
        pre_grasp_rad: float = -0.9,
        pre_grasp_x_m: float = 0.15,
        pose_o_g: Pose3D = Pose3D.from_xyz_rpy(
            x=-0.02,
            z=0.255,
            pitch_rad=1.5708,
            ref_frame="eraser1",
        ),
        lift_z_m: float = 0.25,
        *,
        pauses: bool = False,
        yaw_symmetric: bool = True,
        stow_after: bool = True,
    ) -> Outcome:
        """Pick an object from an open drawer using Spot's gripper.

        :param object_name: Name of the object to be picked
        :param drawer_name: Name of the drawer the object is picked from
        :param pre_grasp_rad: Angle (radians) to open the gripper before grasping
        :param pre_grasp_x_m: Offset (abs. m) of the pre-grasp pose "back" (-x) from the grasp pose
        :param pose_o_g: Object-relative end-effector pose used to grasp the object
        :param lift_z_m: Offset (m) of the post-grasp pose "up" (+z) w.r.t. the world frame
        :param pauses: If True, the skill will pause until user input between each major motion
        :param yaw_symmetric: Indicates that the object is rotationally symmetric about its z-axis
        :param stow_after: If True, stow the arm after picking the object
        :return: Boolean success indicator and an outcome message
        """
        console.print(f"Picking object '{object_name}' from drawer of '{drawer_name}'...")

        # Navigate to the "pick_from_drawer" navigation waypoint
        nav_outcome = self.navigate_to_waypoint(waypoint="pick_from_drawer")
        if not nav_outcome.success:
            return nav_outcome

        # Move Spot's end-effector to approximate viewing location for the object
        ee_during_pose_est = self._EE_POSES_FOR_POSE_ESTIMATION[drawer_name]
        pre_estimation_outcome = self._move_ee_to_pose(ee_during_pose_est)
        if not pre_estimation_outcome.success:
            return pre_estimation_outcome

        # Fully open the gripper and re-pose-estimate the object
        open_outcome = self.open_gripper()
        if not open_outcome.success:
            return open_outcome

        obj_estimate_outcome = self.estimate_pose(object_name, duration_s=5.0)
        if not obj_estimate_outcome.success:
            return obj_estimate_outcome

        drawer_est_outcome = self.estimate_pose(drawer_name, duration_s=5.0)
        if not drawer_est_outcome.success:
            return drawer_est_outcome

        # Identify which candidate grasp pose to use, if the object is symmetric
        if object_name != pose_o_g.ref_frame:
            console.print(f"[yellow]Warning: Grasp pose given in frame '{pose_o_g.ref_frame}'.[/]")
            pose_o_g = TransformManager.convert_to_frame(pose_o_g, target_frame=object_name)
        # self._pose_broadcaster.poses["pose_o_g"] = pose_o_g  # Grasp pose w.r.t. object frame

        candidates = [pose_o_g]
        if yaw_symmetric:
            rotate_object = Pose3D.from_xyz_rpy(yaw_rad=3.14159, ref_frame=object_name)
            alternative_pose_o_g = rotate_object @ pose_o_g
            # self._pose_broadcaster.poses["alternative_pose_o_g"] = alternative_pose_o_g
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

        # Open the gripper to prepare for picking
        if not self._gripper.move_to_angle_rad(pre_grasp_rad):
            return Outcome(
                success=False,
                message=f"Unable to pick '{object_name}' because the gripper didn't open.",
            )

        # Move the end-effector to the pre-grasp pose ("back" from the grasp pose)
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the pre-grasp pose")

        pre_outcome = self._move_ee_to_pose(valid_poses.pregrasp_pose, display_and_pause=pauses)
        if not pre_outcome.success:
            return pre_outcome

        # Move the end-effector to the grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the grasp pose")

        to_grasp_outcome = self._move_ee_to_pose(
            valid_poses.grasp_pose,
            # ignored_objects=drawer_name,
            display_and_pause=pauses,
        )
        if not to_grasp_outcome.success:
            return to_grasp_outcome

        # Grasp the object by closing the gripper
        if pauses:
            Prompt.ask(f"Press [bold]Enter[/] to grasp [cyan]'{object_name}'[/]")

        grasp_outcome = self._grasp_object(object_name)
        if not grasp_outcome.success:
            return grasp_outcome

        # Move the end-effector to the post-grasp pose
        if pauses:
            Prompt.ask("Press [bold]Enter[/] to move to the post-grasp pose")

        post_outcome = self._move_ee_to_pose(
            valid_poses.postgrasp_pose,
            ignored_objects=object_name,
            # ignored_objects=f"{drawer_name},{object_name}",
            display_and_pause=pauses,
        )
        if not post_outcome.success:
            return post_outcome

        # Stow Spot's arm, if requested
        if stow_after:
            if pauses:
                Prompt.ask("Press [bold]Enter[/] to stow Spot's arm")

            time.sleep(1.5)  # Allow arm to settle before stowing
            stow_outcome = self.stow_arm()
            if not stow_outcome.success:
                return stow_outcome

        return Outcome(
            success=True,
            message=f"Successfully picked object '{object_name}' from drawer of '{object_name}'.",
        )

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

        release_outcome = self._release_object(object_name)
        if not release_outcome.success:
            return release_outcome

        postplace_outcome = self._move_ee_to_pose(postplace_pose_s_ee, f"postplace_{object_name}")
        if not postplace_outcome.success:
            return postplace_outcome

        if stow_after:
            stow_outcome = self.stow_arm()
            if not stow_outcome.success:
                return stow_outcome

        return Outcome(success=True, message=f"Placed '{object_name}' on '{surface_name}'.")

    @skill_method
    def _reset_state(
        self,
        yaml_path: Path = Path("/docker/spot_skills/src/spot_skills/config/env.yaml"),
    ) -> Outcome:
        """Reset the environment state as specified by a YAML file.

        :param yaml_path: Filepath to a YAML file specifying environment geometry
        :return: Boolean success indicator and outcome message
        """
        console.print(f"Resetting the environment state based on the YAML file: {yaml_path}")

        request = NameServiceRequest(name=str(yaml_path))
        response = self._reset_state_caller(request)
        if response is None:
            return Outcome(success=False, message="Reset state service returned None.")

        return Outcome(success=response.success, message=response.message)

    @skill_method
    def estimate_pose(
        self,
        object_name: str = "filing_cabinet",
        duration_s: float = 5.0,
    ) -> Outcome:
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
