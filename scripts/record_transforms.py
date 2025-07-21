"""Define a command-line interface to record transforms from /tf."""

from pathlib import Path

import click
import rospy

from robotics_utils.ros.transform_manager import TransformManager
from robotics_utils.ros.transform_recorder import TransformRecorder


@click.command
@click.argument("output_path", type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Whether to allow overwriting the output path")
@click.option("--reference-frame", default="body", help="Reference frame for the recording")
@click.option(
    "--tracked-frame",
    default="arm_link_wr1",
    help="Frame for which relative motion is tracked",
)
def record(output_path: Path, overwrite: bool, reference_frame: str, tracked_frame: str) -> None:
    """Record transforms from /tf until rospy is shut down."""
    if not overwrite and output_path.exists():
        raise FileExistsError(f"Cannot overwrite existing output path: {output_path}")

    TransformManager.init_node("tf_transform_recorder")

    recorder = TransformRecorder(reference_frame, tracked_frame)

    rate_hz = rospy.Rate(TransformManager.LOOP_HZ)
    try:
        while not rospy.is_shutdown():
            recorder.update()
            rate_hz.sleep()
    except rospy.ROSInterruptException:
        pass
    finally:
        recorder.save_to_file()


if __name__ == "__main__":
    record()
