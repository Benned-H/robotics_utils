"""Define a class to track the estimated poses of visual fiducials."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import rospy
import yaml
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from robotics_utils.ros.call_loop_thread import CallLoopThread
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.spatial import DEFAULT_FRAME, XYZ_RPY, Pose3D
from robotics_utils.state_estimation import PoseEstimateAverager
from robotics_utils.vision.fiducials import AprilTagDetector, FiducialMarker, FiducialSystem

if TYPE_CHECKING:
    from robotics_utils.parallelism import ResourceManager
    from robotics_utils.vision.cameras import RGBCamera


class TagTracker:
    """Manages pose estimation for visual fiducials and dependent objects."""

    def __init__(
        self,
        system: FiducialSystem,
        cameras: Sequence[RGBCamera],
        window_size: int = 10,
        resource_manager: ResourceManager | None = None,
    ) -> None:
        """Initialize the TagTracker for the given system of fiducials and cameras.

        :param system: System of known visual fiducials and the names of cameras to detect them
        :param cameras: List of RGB cameras on the robot
        :param window_size: Size of the sliding window of poses used to compute pose averages
        :param resource_manager: Optional resource manager for shared robot resources
        """
        self.system = system
        self.rgb_cameras = cameras
        self.pose_averager = PoseEstimateAverager(window_size)
        self.detector = AprilTagDetector(self.system)

        self.frame_to_parent: dict[str, int] = {}
        """A map from the name of each tag-dependent frame to the frame's parent marker's ID."""

        for marker in self.system.markers.values():
            for relative_frame in marker.relative_frames:
                self.frame_to_parent[relative_frame] = marker.id

        self.output_srv = rospy.Service("~output_to_yaml", Trigger, self.handle_output_to_yaml)

        self.update_thread = CallLoopThread(
            self._update_estimates,
            loop_hz=20.0,
            name="TagTracker",
            resource_manager=resource_manager,
        )

    @property
    def all_estimated_poses(self) -> dict[str, Pose3D]:
        """Retrieve a map from reference frame names to their estimated poses (if available)."""
        estimated_poses: dict[str, Pose3D] = {
            frame_name: pose_avg
            for frame_name, pose_avg in self.pose_averager.compute_all_averages().items()
            if pose_avg is not None
        }

        for frame_name in self.frame_to_parent:
            if frame_name not in estimated_poses:
                pose_estimate = self.get_estimated_pose(frame_name)
                if pose_estimate is not None:
                    estimated_poses[frame_name] = pose_estimate

        return estimated_poses

    def get_estimated_pose(self, frame_name: str) -> Pose3D | None:
        """Retrieve the world-frame estimated pose of a reference frame (or None if unavailable).

        :param frame_name: Name of a reference frame
        :return: Estimated pose of the frame, or None if no estimate exists
        """
        if frame_name not in self.frame_to_parent:
            return None

        marker_id = self.frame_to_parent[frame_name]
        marker = self.system.markers[marker_id]
        pose_m_f = marker.relative_frames[frame_name]  # requested frame (f) w.r.t. marker (m)
        pose_w_m = self.pose_averager.get(marker.frame_name)  # marker w.r.t. world frame (w)

        if pose_w_m is None:
            return None

        return pose_w_m @ pose_m_f  # Result: requested frame w.r.t. world frame

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

            for child_frame, pose_m_c in marker.relative_frames.items():
                pose_w_c = pose_w_m @ pose_m_c  # child frame w.r.t. world frame
                poses_data[child_frame] = pose_w_c.to_xyz_rpy()

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
                world_t_marker = TransformManager.convert_to_frame(detection.pose, DEFAULT_FRAME)
                marker_frame = FiducialMarker.id_to_frame_name(detection.id)
                self.pose_averager.update(marker_frame, world_t_marker)
