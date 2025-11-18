"""Define a class to track the estimated poses of visual fiducials."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import rospy
import yaml
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.ros.call_loop_thread import CallLoopThread
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.state_estimation import PoseEstimateAverager
from robotics_utils.vision.fiducials import AprilTagDetector, FiducialMarker, FiducialSystem

if TYPE_CHECKING:
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

        self.detector = AprilTagDetector(self.system)

        self.output_srv = rospy.Service("~output_to_yaml", Trigger, self.handle_output_to_yaml)

        self.loop_thread = CallLoopThread(self.update)

    def handle_output_to_yaml(self, _: TriggerRequest) -> TriggerResponse:
        """Dump the current pose estimates to YAML.

        :return: ROS message conveying whether the export succeeded
        """
        yaml_path = get_ros_param("/tag_tracker/output_yaml_path", Path)
        if yaml_path.suffix not in {".yaml", ".yml"}:
            return TriggerResponse(success=False, message=f"Invalid YAML file suffix: {yaml_path}")

        poses_data: dict[str, list[float]] = {}

        for marker in self.system.markers.values():
            pose_w_m = self.pose_averager.get(marker.frame_name)
            if pose_w_m is None:
                continue

            poses_data[marker.frame_name] = pose_w_m.to_list()

            for obj_name, pose_m_o in marker.relative_frames.items():
                pose_w_o = pose_w_m @ pose_m_o
                poses_data[obj_name] = pose_w_o.to_list()

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

        # Publish all pose estimates available from the pose averager
        for frame_name, pose_avg in curr_pose_averages.items():
            if pose_avg is None:
                continue
            TransformManager.broadcast_transform(frame_name, pose_avg)

            # Publish all marker-relative poses in the fiducial system
            for marker in self.system.markers.values():
                for obj_name, rel_pose in marker.relative_frames.items():
                    TransformManager.broadcast_transform(obj_name, rel_pose)

    def update(self) -> None:
        """Collect new images, update pose estimates, and broadcast current estimates."""
        self._update_estimates()
        self._broadcast_frames()
