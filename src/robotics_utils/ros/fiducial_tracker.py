"""Define a class to manage visual fiducial detections and dependent object poses."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import rospy
import yaml
from ar_track_alvar_msgs.msg import AlvarMarkers
from pose_estimation_msgs.msg import PoseEstimate
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from robotics_utils.kinematics import DEFAULT_FRAME, Pose3D
from robotics_utils.perception.pose_estimation import FiducialSystem, PoseEstimateAverager
from robotics_utils.ros.msg_conversion import pose_from_msg
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager


@dataclass(frozen=True)
class MarkersCallbackArgs:
    """Organizes arguments to the FiducialTracker.markers_callback() method."""

    camera_name: str  # Name of the camera source of a detection


class FiducialTracker:
    """Track fiducial detections, aggregate their pose estimates, and republish averages."""

    def __init__(
        self,
        system: FiducialSystem,
        prefix: str,
        window_size: int = 10,
        known_poses: dict[str, Pose3D] | None = None,
    ) -> None:
        """Initialize the FiducialTracker for the given system of fiducials and cameras.

        :param system: System of known visual fiducials and cameras to detect them
        :param prefix: Prefix used for the tracker's ROS subscribers for marker data
        :param window_size: Size of the sliding window of poses used to compute pose averages
        :param known_poses: Optional collection of known, fixed poses (defaults to None)
        """
        self.system = system
        self.pose_averager = PoseEstimateAverager(window_size)

        self.known_poses = known_poses or {}

        self.marker_subs: list[rospy.Subscriber] = []
        for camera_name in self.system.cameras:
            self.marker_subs.append(
                rospy.Subscriber(
                    f"{prefix}/{camera_name}",
                    AlvarMarkers,
                    callback=self.markers_callback,
                    callback_args=MarkersCallbackArgs(camera_name),
                    queue_size=5,
                ),
            )

        self.pose_pub = rospy.Publisher("~object_pose_estimates", PoseEstimate, queue_size=10)

        self.output_srv = rospy.Service("~output_to_yaml", Trigger, self.handle_output_to_yaml)

        # Create a thread to publish TF frames (daemon = thread exits when main process does)
        self._tf_pub_thread = threading.Thread(target=self._publish_frames_loop, daemon=True)
        self._tf_pub_thread.start()

    def markers_callback(self, markers_msg: AlvarMarkers, args: MarkersCallbackArgs) -> None:
        """Update pose estimates based on new fiducial marker detections."""
        camera_detects = self.system.camera_detects.get(args.camera_name)
        if camera_detects is None:
            rospy.logwarn(f"Unrecognized camera name: '{args.camera_name}'.")
            return

        if markers_msg.markers is None:
            rospy.logwarn("Markers message had 'None' in place of markers messages.")
            return

        for marker_msg in markers_msg.markers:
            if marker_msg.id not in self.system.markers:
                rospy.logwarn(f"Unrecognized marker ID: {marker_msg.id}.")
                continue

            if marker_msg.id not in camera_detects:
                continue  # This camera doesn't detect this marker; move to the next detection

            raw_pose = pose_from_msg(marker_msg.pose)
            raw_pose.ref_frame = marker_msg.header.frame_id
            marker_pose = TransformManager.convert_to_frame(raw_pose, DEFAULT_FRAME)

            marker = self.system.markers[marker_msg.id]
            self.pose_averager.update(marker.frame_name, marker_pose)

    def reestimate(self, frame_name: str, duration_s: float = 10.0) -> Pose3D | None:
        """Re-estimate the pose of the named frame, then store its updated estimate as 'known'.

        :param frame_name: Name of the frame (i.e., marker or object) to be pose-estimated
        :param duration_s: Duration (seconds) to wait for new pose estimates, defaults to 10.0
        :return: Updated pose estimate, if one was found, else None
        """
        # If the frame refers to an object, update its parent marker's pose estimate
        if frame_name in self.system.object_names:
            rospy.loginfo(f"Processing frame '{frame_name}' as an object frame...")

            parent_marker = self.system.parent_marker.get(frame_name)
            if parent_marker is None:
                return None

            pose_w_m = self.reestimate(parent_marker.frame_name, duration_s)
            if pose_w_m is None:
                rospy.logwarn(f"Unable to re-estimate the pose of marker {parent_marker.id}.")
                return None

            pose_m_o = parent_marker.relative_frames.get(frame_name)
            if pose_m_o is None:
                rospy.logwarn(
                    f"Couldn't find pose for '{frame_name}' w.r.t. {parent_marker.frame_name}.",
                )
                return None

            self.known_poses[frame_name] = pose_w_m @ pose_m_o

        else:
            rospy.loginfo(f"Processing frame '{frame_name}' as a marker frame...")

            prev_average = self.pose_averager.reset_frame(frame_name)

            rospy.sleep(duration_s)

            new_average = self.pose_averager.get(frame_name)
            if new_average is not None:
                self.known_poses[frame_name] = new_average
            elif prev_average is not None:  # Restore previous average if we didn't get a new one
                self.pose_averager.update(frame_name, prev_average)

        return self.known_poses.get(frame_name)

    def handle_output_to_yaml(self, _: TriggerRequest) -> TriggerResponse:
        """Dump the current pose estimates to YAML.

        :return: ROS message conveying whether the export succeeded
        """
        yaml_path = get_ros_param("~output_yaml_path", Path)
        if yaml_path.suffix not in {".yaml", ".yml"}:
            return TriggerResponse(success=False, message=f"Invalid YAML file suffix: {yaml_path}")

        marker_poses_data: dict[int, list[float]] = {}
        object_poses_data: dict[str, list[float]] = {}

        for marker in self.system.markers.values():
            pose_w_m = self.pose_averager.get(marker.frame_name)
            if pose_w_m is None:
                continue

            marker_poses_data[marker.id] = pose_w_m.to_list()

            for obj_name, pose_m_o in marker.relative_frames.items():
                pose_w_o = pose_w_m @ pose_m_o
                object_poses_data[obj_name] = pose_w_o.to_list()

        yaml_data = {
            "marker_poses": marker_poses_data,
            "object_poses": object_poses_data,
        }

        yaml_string = yaml.dump(yaml_data, sort_keys=True, default_flow_style=True)
        with yaml_path.open("w") as yaml_file:
            yaml_file.write(yaml_string)
        ok = yaml_path.exists()

        message = f"Wrote YAML to {yaml_path}." if ok else f"Failed to write to {yaml_path}."
        return TriggerResponse(success=ok, message=message)

    def _publish_frames_loop(self) -> None:
        """Continually broadcast averaged marker and object frames to /tf."""
        rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
        try:
            while not rospy.is_shutdown():
                # Publish all pose estimates available from the pose averager
                curr_pose_averages = self.pose_averager.compute_all_averages()
                for frame_name, pose_avg in curr_pose_averages.items():
                    if pose_avg is None or frame_name in self.known_poses:
                        continue
                    TransformManager.broadcast_transform(frame_name, pose_avg)

                # Publish all marker-relative poses in the visual fiducial system
                for marker in self.system.markers.values():
                    for obj_name, rel_pose in marker.relative_frames.items():
                        if obj_name in self.known_poses:
                            continue
                        TransformManager.broadcast_transform(obj_name, rel_pose)

                # Publish all known poses as stored
                for frame_name, known_pose in self.known_poses.items():
                    TransformManager.broadcast_transform(frame_name, known_pose)

                rate_hz.sleep()

        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[FiducialTracker] TF publish loop interrupted: {ros_exc}")
