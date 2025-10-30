"""Demo script playing with an Intel RealSense camera."""

from robotics_utils.io.logging import log_info
from robotics_utils.perception.vision import Pointcloud
from robotics_utils.perception.vision.realsense import D455_SPEC, RealSense
from robotics_utils.visualization import PointcloudVisualizer, display_in_window


def main() -> None:
    """Demo the Intel RealSense."""
    with RealSense(depth_spec=D455_SPEC) as sensor, PointcloudVisualizer() as vis:
        while True:
            rgbd = sensor.get_rgbd(timeout_ms=5000)
            if not display_in_window(rgbd, "RGB-D Image ('q' to exit)", wait_for_input=False):
                break

            log_info(f"Min depth: {rgbd.depth.min_depth_m} Max depth: {rgbd.depth.max_depth_m}")

            # Compute and visualize a pointcloud based on the current depth image
            pointcloud = Pointcloud.from_depth_image(rgbd.depth, sensor.depth_intrinsics)
            vis.visualize(pointcloud)


if __name__ == "__main__":
    main()
