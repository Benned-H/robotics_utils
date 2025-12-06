"""Define a class to manage setting and reading transforms from the /tf tree."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import rospy
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from robotics_utils.kinematics import Pose2D, Pose3D
from robotics_utils.ros.msg_conversion import pose_from_tf_stamped_msg, pose_to_tf_stamped_msg

if TYPE_CHECKING:
    from geometry_msgs.msg import TransformStamped


class TransformManager:
    """A static class used to manage and read from the /tf tree."""

    LOOP_HZ = 10.0  # Frequency (Hz) of any transform request/send loops

    # Delay initialization of TF2 objects until ROS is available
    _tf_broadcaster: TransformBroadcaster | None = None
    _tf_buffer: Buffer | None = None
    _tf_listener: TransformListener | None = None

    @staticmethod
    def init_node(node_name: str = "transform_manager") -> None:
        """Initialize a ROS node if this process does not yet have a ROS node.

        This method allows the TransformManager to initialize its transform
            listener as soon as it knows that ROS is available.

        :param node_name: Name of the node, defaults to "transform_manager"
        """
        if rospy.get_name() in ["", "/unnamed"]:
            rospy.init_node(node_name)
            rospy.loginfo(f"Initialized node with name '{rospy.get_name()}'")

        TransformManager.tf_listener()  # Ensure transform listener is initialized

    @staticmethod
    def tf_broadcaster() -> TransformBroadcaster:
        """Retrieve the transform broadcaster, initializing it if necessary.

        :return: Transform broadcaster used to send transforms to the TF2 server
        """
        if TransformManager._tf_broadcaster is None:
            TransformManager._tf_broadcaster = TransformBroadcaster()
        return TransformManager._tf_broadcaster

    @staticmethod
    def tf_buffer() -> Buffer:
        """Retrieve the transform buffer, initializing it if necessary.

        :return: Transform buffer used to store known transforms
        """
        if TransformManager._tf_buffer is None:
            TransformManager._tf_buffer = Buffer()
        return TransformManager._tf_buffer

    @staticmethod
    def tf_listener() -> TransformListener:
        """Retrieve the transform listener, initializing it if necessary.

        :return: Transform listener used to receive transforms from the TF2 server
        """
        if TransformManager._tf_listener is None:
            tf_buffer = TransformManager.tf_buffer()
            TransformManager._tf_listener = TransformListener(tf_buffer)
        return TransformManager._tf_listener

    @staticmethod
    def broadcast_transform(frame_name: str, relative_pose: Pose3D) -> None:
        """Broadcast the given transform for the named frame into /tf.

        :param frame_name: Name of the reference frame to be updated
        :param relative_pose: Transform of the frame relative to its parent frame
        """
        tf_stamped_msg = pose_to_tf_stamped_msg(relative_pose, frame_name)
        tf_stamped_msg.header.stamp = rospy.Time.now()

        TransformManager.tf_broadcaster().sendTransform(tf_stamped_msg)

    # TODO: Source = child
    # TODO: Target = Parent

    @staticmethod
    def lookup_transform(
        child_frame: str,
        parent_frame: str,
        when: rospy.Time | None = None,
        timeout_s: float = 5.0,
    ) -> Pose3D | None:
        """Look up the transform to convert from one frame to another using /tf.

        Frame notation: Child frame (c) and parent frame (p).

        Say our data is originally expressed in the child frame (data_wrt_c).
        This function outputs transform_p_c ("child relative to parent"), which lets us compute:

            transform_p_c @ data_wrt_c = data_wrt_p (i.e., "data expressed in the parent frame")

        Reference: https://docs.ros.org/en/noetic/api/tf2_ros/html/c++/classtf2__ros_1_1Buffer.html#ada7f9d7d8d12655d7ce5c5f303303f5f

        :param child_frame: Frame whose relative pose we want to find
        :param parent_frame: Frame relative to which the transform is found
        :param when: Timestamp for which the relative transform is found (if None, use latest data)
        :param timeout_s: Duration (seconds) after which to abandon the lookup (defaults to 5)
        :return: Pose3D representing transform (i.e., transform_p_c) or None (if lookup failed)
        """
        if when is None:
            when = rospy.Time(0)

        rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
        rate_hz.sleep()

        start_time_s = rospy.get_time()  # Start time in seconds (float)
        timeout_time_s = start_time_s + timeout_s

        tf_stamped_msg: TransformStamped | None = None
        while (rospy.get_time() < timeout_time_s) and (not rospy.is_shutdown_requested()):
            try:
                tf_stamped_msg = TransformManager.tf_buffer().lookup_transform(
                    target_frame=parent_frame,
                    source_frame=child_frame,
                    time=when,
                    timeout=rate_hz.sleep_dur,
                )
                break
            except TransformException as t_exc:
                rospy.logwarn(
                    f"[TransformManager.lookup_transform] Lookup of '{child_frame}' w.r.t. "
                    f"'{parent_frame}' at time {when.to_time():.2f} gave exception: {t_exc}",
                )
                rate_hz.sleep()

        if tf_stamped_msg is None:
            rospy.logerr(
                f"[TransformManager.lookup_transform] Could not look up transform of "
                f"'{child_frame}' w.r.t. '{parent_frame}' within {timeout_s} seconds.",
            )
            return None

        pose_p_c = pose_from_tf_stamped_msg(tf_stamped_msg)

        if parent_frame != pose_p_c.ref_frame:
            raise RuntimeError(
                f"Expected result in '{parent_frame}' but instead found '{pose_p_c.ref_frame}'",
            )

        return pose_p_c

    @staticmethod
    def convert_to_frame(
        pose_c_p: Pose3D | Pose2D,
        target_frame: str,
        timeout_s: float = 5.0,
    ) -> Pose3D:
        """Convert the given pose into the target reference frame.

        Frames: Frame implied by the pose (p), current ref. frame (c), target ref. frame (t)

        :param pose_c_p: Pose (frame p) w.r.t. its current reference frame (frame c)
        :param target_frame: Target reference frame (frame t) of the output pose
        :param timeout_s: Duration (seconds) after which the conversion times out (default: 5 sec)
        :return: Pose3D relative to the target reference frame (i.e., pose_t_p)
        """
        if isinstance(pose_c_p, Pose2D):
            pose_c_p = pose_c_p.to_3d()

        current_frame = pose_c_p.ref_frame
        if current_frame == target_frame:
            return pose_c_p

        pose_t_c = None
        end_time = time.time() + timeout_s
        while pose_t_c is None and time.time() < end_time:
            pose_t_c = TransformManager.lookup_transform(current_frame, target_frame, timeout_s=1)

        if pose_t_c is None:
            raise RuntimeError(f"Lookup from frame '{current_frame}' to '{target_frame}' failed")

        return pose_t_c @ pose_c_p
