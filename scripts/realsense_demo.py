"""Demo script playing with an Intel RealSense camera."""

import click

from robotics_utils.io.logging import log_info
from robotics_utils.reconstruction import PointCloud
from robotics_utils.vision.cameras import D455_SPEC, RealSense
from robotics_utils.visualization import display_in_window
from robotics_utils.visualization.open3d_visualizer import Open3DVisualizer


@click.command()
@click.option("--pointcloud-coloring", type=click.Choice(["rgb", "depth"]), default="depth")
def main(pointcloud_coloring: str) -> None:
    """Demo the Intel RealSense."""
    with RealSense(depth_spec=D455_SPEC) as sensor, Open3DVisualizer("RealSense Data") as vis:
        while True:
            rgbd = sensor.get_rgbd(timeout_ms=5000)
            if not display_in_window(rgbd, "RGB-D Image ('q' to exit)", wait=False):
                break

            log_info(f"Min depth: {rgbd.depth.min_depth_m} Max depth: {rgbd.depth.max_depth_m}")

            # Compute and visualize a pointcloud based on the current RGB-D or depth image
            if pointcloud_coloring == "rgb":
                pointcloud = PointCloud.from_rgbd_image(rgbd, sensor.depth_intrinsics)
            elif pointcloud_coloring == "depth":
                pointcloud = PointCloud.from_depth_image(rgbd.depth, sensor.depth_intrinsics)
            else:
                raise ValueError(f"Invalid pointcloud coloring choice: '{pointcloud_coloring}'.")

            vis.add_pointcloud("current", pointcloud=pointcloud)


if __name__ == "__main__":
    main()
