"""Launch a ROS node to track visual fiducial detections."""

from pathlib import Path

import rospy

from robotics_utils.kinematics import Pose3D
from robotics_utils.perception.pose_estimation import FiducialSystem
from robotics_utils.ros.fiducial_tracker import FiducialTracker
from robotics_utils.ros.params import get_ros_param
from robotics_utils.ros.transform_manager import TransformManager


def main() -> None:
    """Track visual fiducial detections and forward dependent object poses to /tf."""
    TransformManager.init_node("fiducial_tracker_node")

    rospy.sleep(10)  # Allow ROS parameters to populate

    markers_yaml_path = get_ros_param("~markers_yaml_path", Path)
    fiducial_system = FiducialSystem.from_yaml(markers_yaml_path)

    topic_prefix = get_ros_param("~marker_topic_prefix", str)

    # Check for a YAML file specifying objects/frames with known, fixed poses
    known_poses_yaml_path = get_ros_param("~known_poses_yaml_path", Path)
    known_poses = Pose3D.load_named_poses(known_poses_yaml_path, collection_name="known_poses")

    _ = FiducialTracker(fiducial_system, topic_prefix, 10, known_poses)
    rospy.spin()


if __name__ == "__main__":
    main()
