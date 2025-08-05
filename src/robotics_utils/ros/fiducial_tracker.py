"""Define a class to manage visual fiducial detections and dependent object poses."""

import threading
from dataclasses import dataclass
from pathlib import Path

import rospy
import yaml
from ar_track_alvar_msgs.msg import AlvarMarkers
from pose_estimation_msgs.msg import PoseEstimate
from std_srvs.srv import Trigger, TriggerRequest, TriggerResponse

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.perception.pose_estimation import PoseEstimateAverager
from robotics_utils.ros.msg_conversion import pose_from_msg, pose_to_stamped_msg
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.sensors.visual_fiducials import VisualFiducialSystem


@dataclass(frozen=True)
class MarkersCallbackArgs:
    """Organizes arguments to the FiducialTracker.markers_callback() method."""

    camera_name: str  # Name of the camera source of a detection


class FiducialTracker:
    """Track fiducial detections, aggregate their pose estimates, and republish averages."""

    def __init__(self, system: VisualFiducialSystem, prefix: str, max_estimates: int = 10) -> None:
        """Initialize the FiducialTracker for the given system of fiducials and detectors.

        :param system: System of known visual fiducials and cameras to detect them
        :param prefix: Prefix used for the tracker's ROS suscribers for marker data
        :param max_estimates: Maximum number of recent pose estimates to retain per frame
        """
        self.system = system
        self.pose_averager = PoseEstimateAverager(max_estimates)

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

        self.pose_pub = rospy.Publisher("/object_pose_estimates", PoseEstimate, queue_size=10)

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

        for marker in markers_msg.markers:
            if marker.id not in camera_detects:
                continue  # This camera doesn't detect this marker; move to the next detection

            raw_pose = pose_from_msg(marker.pose)
            raw_pose.ref_frame = marker.header.frame_id
            marker_pose = TransformManager.convert_to_frame(raw_pose, DEFAULT_FRAME)

            frame_name = str(marker.id)
            self.pose_averager.update(frame_name, marker_pose)

    def handle_output_to_yaml(self, _: TriggerRequest) -> TriggerResponse:
        """Dump the current pose estimates to YAML.

        :return: ROS message conveying whether the export succeeded
        """
        yaml_path = get_ros_param("~yaml_output_path", Path)
        if yaml_path.suffix not in {".yaml", ".yml"}:
            return TriggerResponse(success=False, message=f"Invalid YAML file suffix: {yaml_path}")

        poses_data: dict[str, list[float]] = {}

        for fiducial in self.system.markers.values():
            pose_w_f = self.pose_averager.get(fiducial.frame_name)
            if pose_w_f is None:
                continue

            for obj_name, pose_f_o in fiducial.relative_frames.items():
                pose_w_o = pose_w_f @ pose_f_o
                poses_data[obj_name] = pose_w_o.to_list()

        yaml_data = {"object_poses": poses_data, "default_frame": DEFAULT_FRAME}

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
            pose_estimate_msg = PoseEstimate()
            while not rospy.is_shutdown():
                for fiducial in self.system.markers.values():
                    fiducial_pose = self.pose_averager.get(fiducial.frame_name)
                    if fiducial_pose is not None:
                        TransformManager.broadcast_transform(fiducial.frame_name, fiducial_pose)

                        for obj_name, rel_pose in fiducial.relative_frames.items():
                            TransformManager.broadcast_transform(obj_name, rel_pose)

                            pose_estimate_msg.object_name = obj_name
                            pose_estimate_msg.pose = pose_to_stamped_msg(rel_pose)
                            pose_estimate_msg.confidence = 0.0

                rate_hz.sleep()

        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[FiducialTracker] TF publish loop interrupted: {ros_exc}")
