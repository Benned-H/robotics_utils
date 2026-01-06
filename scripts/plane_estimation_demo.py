"""Demo script that estimates a plane based on an object segmentation from Intel RealSense data.

To profile this demo using Austin, use the command:

    uv run austin -o .austin/plane_estimation.austin python scripts/plane_estimation_demo.py \
        --query "six of diamonds playing card"

"""

from __future__ import annotations

import click

from robotics_utils.io import console
from robotics_utils.reconstruction import PlaneEstimate, Pointcloud
from robotics_utils.vision import RGBImage
from robotics_utils.vision.cameras import D415_SPEC, D455_SPEC, RealSense
from robotics_utils.vision.vlms import SAM3
from robotics_utils.visualization import display_in_window
from robotics_utils.visualization.open3d_visualizer import Open3DVisualizer


@click.command()
@click.option("--realsense", type=click.Choice(["D415", "D455"]), default="D415")
@click.option("--query", type=str, default=None, help="Object query used for segmentation")
def main(realsense: str, query: str | None) -> None:
    """Run the plane estimation demo using data from an Intel RealSense camera."""
    main_window_title = "Plane Estimation Demo (press 'q' to exit)"
    projected_window_title = "2D Projection of Inliers onto Estimated Plane (press 'q' to exit)"

    depth_spec = {"D415": D415_SPEC, "D455": D455_SPEC}.get(realsense)
    if depth_spec is None:
        raise ValueError(f"Unrecognized RealSense type: {realsense}")

    segmenter = SAM3()

    if query is None:
        query = click.prompt("\nEnter object query", type=str)
    assert query is not None, "Query must be provided."

    with RealSense(depth_spec) as sensor, Open3DVisualizer("Plane Estimation") as vis:
        console.print("RealSense camera initialized, starting loop...")

        while True:
            rgbd = sensor.get_rgbd()
            if not rgbd.same_dimensions:
                console.print(
                    "[red]Cannot process RGB-D data with differing dimensions.[/]\n\tRGB shape: "
                    f"{rgbd.rgb.resolution}\n\tDepth shape: {rgbd.depth.resolution}",
                )

            vis_image = RGBImage(rgbd.rgb.data.copy())  # We'll add overlays as they're computed

            console.print(f"Segmenting '{query}' in the image...")
            segmentations = segmenter.segment(rgbd.rgb, queries=[query])
            if not segmentations.segmentations:
                console.print(f"No object instances found for query '{query}'.")
                if not display_in_window(vis_image, main_window_title, wait=False):
                    break
                continue

            for segmented_instance in segmentations:
                segmented_instance.draw(vis_image, color=(0, 255, 0))

            if not display_in_window(vis_image, main_window_title, wait=False):
                break

            # Select the largest segmented instance by mask area
            biggest = max(segmentations, key=lambda s: s.mask.sum())

            # Compute a pointcloud using the selected segmentation
            pointcloud = Pointcloud.from_segmented_rgbd(rgbd, sensor.depth_intrinsics, biggest)
            console.print(f"Pointcloud from largest segmentation has {len(pointcloud)} points.")

            estimate = PlaneEstimate.fit_plane_ransac(
                pointcloud,
                inlier_threshold_m=0.01,
                ransac_n=5,
                iterations=1000,
            )
            if estimate is None:
                console.print("No plane estimate. Continuing...")
                continue

            console.print(f"Plane center: {estimate.plane.point}")
            console.print(f"Plane normal: {estimate.plane.normal}")
            console.print(f"Plane equation: {estimate.plane.equation_string}")
            console.print(estimate.get_inlier_text())

            # Visualize a 2D projection of inliers onto the estimated plane
            if not display_in_window(estimate, projected_window_title, wait=False):
                break

            # Update the live 3D visualization (non-blocking)
            vis.add_plane_estimate("current", estimate, plane_size_m=0.15, axes_size_m=0.1)


if __name__ == "__main__":
    main()
