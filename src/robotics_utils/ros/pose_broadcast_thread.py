"""Define a class that creates a thread to continually broadcast stored poses."""

import threading
from copy import deepcopy
from typing import TYPE_CHECKING

import rospy

from robotics_utils.ros import TransformManager

if TYPE_CHECKING:
    from robotics_utils.kinematics import Pose3D


class PoseBroadcastThread:
    """A class that continually broadcasts a collection of named poses."""

    def __init__(self) -> None:
        """Initialize a thread to broadcast any stored poses."""
        self.poses: dict[str, Pose3D] = {}
        """Named poses that are continually republished to /tf."""

        self._thread = threading.Thread(target=self._broadcast_poses, daemon=True)
        self._thread.start()

    def _broadcast_poses(self) -> None:
        """Continually republish the stored poses to /tf."""
        try:
            rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
            while not rospy.is_shutdown():
                if self.poses:
                    curr_poses = deepcopy(self.poses)
                    for frame_name, pose in curr_poses.items():
                        TransformManager.broadcast_transform(frame_name, pose)

                rate_hz.sleep()
        except rospy.ROSInterruptException as ros_exc:
            rospy.logwarn(f"[PoseBroadcastThread] {ros_exc}")
