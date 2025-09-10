"""Record a relative trajectory from /tf and save it to file."""

import argparse
from pathlib import Path

import rospy

from robotics_utils.ros.robots.manipulator import Manipulator
from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.ros.transform_recorder import TransformRecorder


def record(output_path: Path, overwrite: bool, reference_frame: str, tracked_frame: str) -> None:
    """Record transforms from /tf until rospy is shut down.

    :raises FileExistsError: If the output path already exists but the overwrite flag isn't set
    """
    if not overwrite and output_path.exists():
        raise FileExistsError(f"Cannot overwrite existing output path: {output_path}")

    TransformManager.init_node("tf_transform_recorder")

    arm = Manipulator(move_group="arm", base_link="body", gripper=None)
    config_before = arm.get_configuration()

    recorder = TransformRecorder(reference_frame, tracked_frame)

    rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
    try:
        while not rospy.is_shutdown():
            recorder.update()
            rate_hz.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        config_after = arm.get_configuration()
        recorder.save_to_file(output_path, config_before, config_after)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("record_transforms")
    parser.add_argument("output_path", type=Path)
    parser.add_argument(
        "--overwrite",
        type=bool,
        default=False,
        help="Whether to allow overwriting the output path",
    )
    parser.add_argument(
        "--reference-frame",
        type=str,
        default="body",
        help="Reference frame for the initial tracked pose",
    )
    parser.add_argument(
        "--tracked-frame",
        type=str,
        default="arm_link_wr1",
        help="Frame for which relative motion is tracked",
    )

    args = parser.parse_args()
    output_path: Path = args.output_path
    overwrite: bool = args.overwrite
    reference_frame: str = args.reference_frame
    tracked_frame: str = args.tracked_frame

    record(output_path, overwrite, reference_frame, tracked_frame)
