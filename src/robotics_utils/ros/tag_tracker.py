"""Define a class to track the estimated poses of visual fiducials."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import rospy
import yaml
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from robotics_utils.kinematics import DEFAULT_FRAME, XYZ_RPY, Pose3D
from robotics_utils.kinematics.kinematic_tree import KinematicTree
from robotics_utils.ros.call_loop_thread import CallLoopThread
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.state_estimation import PoseEstimateAverager
from robotics_utils.vision.fiducials import AprilTagDetector, FiducialMarker, FiducialSystem

if TYPE_CHECKING:
    from robotics_utils.states import KinematicSimulator
    from robotics_utils.vision import RGBCamera


class TagTracker:
    """Manages pose estimation for visual fiducials and dependent objects."""

    def __init__(
        self,
        system: FiducialSystem,
        cameras: Sequence[RGBCamera],
        window_size: int = 10,
    ) -> None:
        """Initialize the TagTracker for the given system of fiducials and cameras.

        :param system: System of known visual fiducials and the names of cameras to detect them
        :param cameras: List of RGB cameras on the robot
        :param window_size: Size of the sliding window of poses used to compute pose averages
        """
        self.system = system
        self.rgb_cameras = cameras
        self.pose_averager = PoseEstimateAverager(window_size)

        self.simulators: list[KinematicSimulator] = []
        """Kinematic simulators to be synchronized with the tag tracker state."""

        self.frame_to_parent: dict[str, int] = {}
        """A map from the name of each tag-dependent frame to the frame's parent marker's ID."""

        for marker in self.system.markers.values():
            for relative_frame in marker.relative_frames:
                self.frame_to_parent[relative_frame] = marker.id

        # Check if there's a YAML file specifying known object poses
        known_poses_param = get_ros_param("/tag_tracker/known_poses_yaml_path", str, "")
        if known_poses_param:
            rospy.loginfo(f"Loading known object poses from file: {known_poses_param}")
        self.kinematic_state = (
            KinematicTree.from_yaml(Path(known_poses_param))
            if known_poses_param
            else KinematicTree(root_frame=DEFAULT_FRAME)
        )

        self.detector = AprilTagDetector(self.system)

        self.output_srv = rospy.Service("~output_to_yaml", Trigger, self.handle_output_to_yaml)

        self.update_loop_thread = CallLoopThread(self.update)
        self.sync_loop_thread = CallLoopThread(self.sync_simulators, loop_hz=5.0)

    @property
    def all_object_poses(self) -> dict[str, Pose3D | None]:
        """Construct a map from object names to their world-frame poses (or None if unknown)."""
        all_poses = {}

        known_object_poses = self.kinematic_state.known_object_poses

        for obj_name in self.kinematic_state.object_names:
            obj_pose = (
                known_object_poses[obj_name]
                if obj_name in known_object_poses
                else self.get_estimated_pose(obj_name)
            )

            all_poses[obj_name] = obj_pose

        return all_poses

    def get_object_pose(self, obj_name: str) -> Pose3D | None:
        """Retrieve the pose of the named object.

        :param obj_name: Name of an object
        :return: Pose estimate for the object, or None if no estimate exists
        """
        known_object_poses = self.kinematic_state.known_object_poses

        if obj_name in known_object_poses:
            return known_object_poses[obj_name]

        return self.get_estimated_pose(obj_name)

    def get_estimated_pose(self, obj_name: str) -> Pose3D | None:
        """Retrieve the world-frame estimated pose of an object (or None if unknown).

        :param obj_name: Name of an object
        :return: Estimated pose of the object, or None if no estimate exists
        """
        if obj_name not in self.frame_to_parent:
            return None

        marker_id = self.frame_to_parent[obj_name]
        marker = self.system.markers[marker_id]
        pose_m_o = marker.relative_frames[obj_name]  # object w.r.t. marker
        pose_w_m = self.pose_averager.get(marker.frame_name)  # marker w.r.t. world frame

        if pose_w_m is None:
            return None

        return pose_w_m @ pose_m_o  # Return object w.r.t. world frame

    def handle_output_to_yaml(self, _: TriggerRequest) -> TriggerResponse:
        """Dump the current pose estimates to YAML.

        :return: ROS message conveying whether the export succeeded
        """
        yaml_path = get_ros_param("/tag_tracker/output_yaml_path", Path)
        if yaml_path.suffix not in {".yaml", ".yml"}:
            return TriggerResponse(success=False, message=f"Invalid YAML file suffix: {yaml_path}")

        poses_data: dict[str, XYZ_RPY] = {}

        for marker in self.system.markers.values():
            pose_w_m = self.pose_averager.get(marker.frame_name)
            if pose_w_m is None:
                continue

            poses_data[marker.frame_name] = pose_w_m.to_xyz_rpy()

            for obj_name, pose_m_o in marker.relative_frames.items():
                pose_w_o = pose_w_m @ pose_m_o
                poses_data[obj_name] = pose_w_o.to_xyz_rpy()

        yaml_data = {"poses": poses_data, "default_frame": DEFAULT_FRAME}

        yaml_string = yaml.dump(yaml_data, sort_keys=True, default_flow_style=True)
        with yaml_path.open("w") as yaml_file:
            yaml_file.write(yaml_string)
        ok = yaml_path.exists()

        message = f"Wrote YAML to {yaml_path}." if ok else f"Failed to write to {yaml_path}."
        return TriggerResponse(success=ok, message=message)

    def _update_estimates(self) -> None:
        """Update the pose estimates using one new image per tag-detecting camera."""
        for rgb_camera in self.rgb_cameras:
            detections_wrt_camera = self.detector.detect_from_camera(rgb_camera)
            for detection in detections_wrt_camera:
                frame_name = FiducialMarker.id_to_frame_name(detection.id)
                tag_wrt_world = TransformManager.convert_to_frame(detection.pose, DEFAULT_FRAME)
                self.pose_averager.update(frame_name, tag_wrt_world)

    def _broadcast_frames(self) -> None:
        """Broadcast the current averaged marker and object frames to /tf."""
        curr_pose_averages = self.pose_averager.compute_all_averages()
        known_object_poses = self.kinematic_state.known_object_poses

        # Publish all pose estimates available from the pose averager (skip known poses)
        for frame_name, pose_avg in curr_pose_averages.items():
            if pose_avg is None or frame_name in known_object_poses:
                continue

            TransformManager.broadcast_transform(frame_name, pose_avg)

        # Publish all marker-relative poses in the fiducial system (skip known poses)
        for marker in self.system.markers.values():
            for obj_name, rel_pose in marker.relative_frames.items():
                if obj_name in known_object_poses:
                    continue

                TransformManager.broadcast_transform(obj_name, rel_pose)

        # Publish all known poses as currently stored
        for frame_name, known_pose in known_object_poses.items():
            TransformManager.broadcast_transform(frame_name, known_pose)

    def update(self) -> None:
        """Collect new images, update pose estimates, and broadcast current estimates."""
        self._update_estimates()
        self._broadcast_frames()

    def sync_simulators(self) -> None:
        """Synchronize stored kinematic simulators with updated world-frame pose estimates."""
        if not self.simulators:
            return

        for obj_name, obj_pose in self.all_object_poses.items():
            if obj_pose is None:
                continue

            for sim in self.simulators:
                sim.set_object_pose(obj_name, obj_pose)
