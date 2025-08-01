"""Demo script playing with an Intel RealSense camera."""

from robotics_utils.vision.pointcloud import Pointcloud, PointcloudVisualizer
from robotics_utils.vision.realsense import D415_SPEC, RealSense


def main() -> None:
    """Demo the Intel RealSense."""
    with RealSense(camera_spec=D415_SPEC) as sensor, PointcloudVisualizer() as vis:
        while True:
            rgbd = sensor.get_rgbd(timeout_ms=5000)
            if not rgbd.visualize("RGB-D Image (press 'q' to exit)", wait_for_input=False):
                break

            rgbd_intrinsics = sensor.get_intrinsics()

            # Compute and visualize a pointcloud based on the current depth image
            pointcloud = Pointcloud.from_depth_image(rgbd.depth, rgbd_intrinsics.depth)
            vis.visualize(pointcloud)


if __name__ == "__main__":
    main()
