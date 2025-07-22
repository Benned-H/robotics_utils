"""Define a class to manage setting and reading transforms from the /tf tree."""

from __future__ import annotations

import rospy
from geometry_msgs.msg import TransformStamped
from tf2_ros import Buffer, TransformBroadcaster, TransformException, TransformListener

from robotics_utils.kinematics.pose3d import Pose3D
from robotics_utils.ros.msg_conversion import pose_from_tf_stamped_msg, pose_to_tf_stamped_msg


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

    @staticmethod
    def lookup_transform(
        source_frame: str,
        target_frame: str,
        when: rospy.Time | None = None,
        timeout_s: float = 5.0,
    ) -> Pose3D | None:
        """Look up the transform to convert from one frame to another using /tf.

        Frame notation: Relative pose of some data (d), source frame (s), target frame (t).

        Say our input data originates in the source frame: pose_s_d ("data w.r.t. source frame").
        This function outputs transform_t_s ("source relative to target"), which lets us compute:

            transform_t_s @ pose_s_d = pose_t_d (i.e., "data expressed in the target frame")

        :param source_frame: Frame where the data originated
        :param target_frame: Frame to which the data will be transformed
        :param when: Timestamp for which the relative transform is found (if None, uses 'now')
        :param timeout_s: Duration (seconds) after which to abandon the lookup (defaults to 5)
        :return: Pose3D representing transform (i.e., transform_t_s) or None (if lookup failed)
        """
        if when is None:
            when = rospy.Time.now()

        rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
        rate_hz.sleep()

        start_time_s = rospy.get_time()  # Start time in seconds (float)
        timeout_time_s = start_time_s + timeout_s

        tf_stamped_msg: TransformStamped | None = None
        while (rospy.get_time() < timeout_time_s) and (not rospy.is_shutdown_requested()):
            try:
                tf_stamped_msg = TransformManager.tf_buffer().lookup_transform(
                    target_frame=target_frame,
                    source_frame=source_frame,
                    time=when,
                    timeout=rate_hz.sleep_dur,
                )
                break
            except TransformException as t_exc:
                rospy.logwarn(
                    f"[TransformManager.lookup_transform] Lookup for '{source_frame}' to "
                    f"'{target_frame}' at time {when.to_time():.2f} gave exception: {t_exc}",
                )
                rate_hz.sleep()

        if tf_stamped_msg is None:
            rospy.logerr(
                f"[TransformManager.lookup_transform] Could not look up transform from "
                f"'{source_frame}' to '{target_frame}' within {timeout_s} seconds.",
            )
            return None

        pose_t_s = pose_from_tf_stamped_msg(tf_stamped_msg)

        if target_frame != pose_t_s.ref_frame:
            raise RuntimeError(
                f"Expected result in '{target_frame}' but instead found '{pose_t_s.ref_frame}'",
            )

        return pose_t_s

    @staticmethod
    def convert_to_frame(pose_c_p: Pose3D, target_frame: str) -> Pose3D:
        """Convert the given pose into the target reference frame.

        Frames: Frame implied by the pose (p), current ref. frame (c), target ref. frame (t)

        :param pose_c_p: Pose (frame p) w.r.t. its current reference frame (frame c)
        :param target_frame: Target reference frame (frame t) of the output pose
        :return: Pose3D relative to the target reference frame (i.e., pose_t_p)
        """
        current_frame = pose_c_p.ref_frame
        if current_frame == target_frame:
            return pose_c_p

        pose_t_c = TransformManager.lookup_transform(current_frame, target_frame, None, 10.0)
        if pose_t_c is None:
            raise RuntimeError(f"Lookup from frame '{current_frame}' to '{target_frame}' failed")

        return pose_t_c @ pose_c_p
