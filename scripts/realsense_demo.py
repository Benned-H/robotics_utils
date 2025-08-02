"""Demo script playing with an Intel RealSense camera."""

from robotics_utils.vision.pointcloud import Pointcloud, PointcloudVisualizer
from robotics_utils.vision.realsense import D415_SPEC, RealSense
from robotics_utils.visualization.image_display import ImageDisplay


def main() -> None:
    """Demo the Intel RealSense."""
    displayer = ImageDisplay()

    with RealSense(camera_spec=D415_SPEC) as sensor, PointcloudVisualizer() as vis:
        while True:
            rgbd = sensor.get_rgbd(timeout_ms=5000)
            if not displayer.show(rgbd, "RGB-D Image (press 'q' to exit)", wait_for_input=False):
                break

            print(f"Min depth: {rgbd.depth.min_depth_m} Max depth: {rgbd.depth.max_depth_m}")

            # Compute and visualize a pointcloud based on the current depth image
            pointcloud = Pointcloud.from_depth_image(rgbd.depth, sensor.depth_intrinsics)
            vis.visualize(pointcloud)


if __name__ == "__main__":
    main()
