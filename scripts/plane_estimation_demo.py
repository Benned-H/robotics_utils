"""Demo script that estimates a plane around a keypoint detection from Intel RealSense data."""

from __future__ import annotations

import click

from robotics_utils.io import console
from robotics_utils.vision import (
    BoundingBox,
    CameraIntrinsics,
    ImagePatch,
    Pointcloud,
    RGBDImage,
    RGBImage,
)
from robotics_utils.vision.realsense import D415_SPEC, D455_SPEC, RealSense
from robotics_utils.vision.vlms import OwlViTBoundingBoxDetector
from robotics_utils.vision.vlms.gemini import GeminiRoboticsER
from robotics_utils.visualization import display_in_window
from robotics_utils.visualization.open3d_visualizer import Open3DVisualizer


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["owl-vit", "gemini"], case_sensitive=False),
    default="owl-vit",
)
@click.option("--realsense", type=click.Choice(["D415", "D455"]), default="D415")
@click.option(
    "--patch-size",
    type=int,
    default=100,
    help="Size of cropped patch around the detected object(s) (length in pixels)",
)
@click.option("--api-key")
def main(backend: str, realsense: str, patch_size: int, api_key: str | None) -> None:
    """Run the plane estimation demo using data from an Intel RealSense camera."""
    if backend == "owl-vit":
        detector = OwlViTBoundingBoxDetector()
    elif backend == "gemini":
        assert api_key is not None, "Cannot use Gemini Robotics-ER 1.5 without a Google API key."
        detector = GeminiRoboticsER(api_key=api_key)
    else:
        raise ValueError(f"Unrecognized backend: {backend}")
    window_title = "Plane Estimation Demo (press 'q' to exit)"

    depth_spec = {"D415": D415_SPEC, "D455": D455_SPEC}.get(realsense)
    if depth_spec is None:
        raise ValueError(f"Unrecognized RealSense type: {realsense}")

    query: str = click.prompt("\nEnter object query", type=str)

    with RealSense(depth_spec) as sensor, Open3DVisualizer("Plane Estimation") as vis:
        console.print("RealSense camera initialized, starting detection loop...")
        console.print("Press 'q' to quit the RGB-D display window.")

        while True:  # TODO: Quit RGB-D window when q entered
            rgbd = sensor.get_rgbd()
            if rgbd.rgb.width != rgbd.depth.width or rgbd.rgb.height != rgbd.depth.height:
                console.print(
                    "[red]Cannot process RGB-D data with differing dimensions.[/]\n\tRGB shape: "
                    f"{rgbd.rgb.resolution}\n\tDepth shape: {rgbd.depth.resolution}",
                )

            vis_image = RGBImage(rgbd.rgb.data.copy())  # We'll add overlays as they're computed

            console.print(f"Detecting '{query}' in the image...")
            detected = detector.detect_bounding_boxes(rgbd.rgb, queries=[query])
            if not detected.detections:
                console.print(f"No detections found for query '{query}'.")
                if not display_in_window(vis_image, window_title, wait=False):
                    break
                continue

            for detection in detected.detections:
                detection.bounding_box.draw(vis_image, color=(0, 255, 0))

            # Select the largest detected bounding box
            biggest = max(detected.detections, key=lambda d: d.bounding_box.area_square_px)
            center_pixel = biggest.bounding_box.center_pixel

            patch_box = BoundingBox.from_center(center_pixel, width=patch_size, height=patch_size)
            depth_patch = ImagePatch.crop(rgbd.depth, patch_box)
            rgb_patch = ImagePatch.crop(rgbd.rgb, patch_box)  # Used to color inlier points
            patch_rgbd = RGBDImage(rgb_patch.patch, depth_patch.patch)

            depth_patch.bounding_box.draw(vis_image, color=(255, 0, 0), thickness=2)
            if not display_in_window(vis_image, window_title, wait=False):
                break

            # Adjust camera intrinsics to the cropped patch (i.e., offset the principal point)
            patch_min_x = depth_patch.bounding_box.top_left.x
            patch_min_y = depth_patch.bounding_box.top_left.y
            patch_intrinsics = CameraIntrinsics(
                fx=sensor.depth_intrinsics.fx,
                fy=sensor.depth_intrinsics.fy,
                x0=sensor.depth_intrinsics.x0 - patch_min_x,
                y0=sensor.depth_intrinsics.y0 - patch_min_y,
            )

            pointcloud = Pointcloud.from_rgbd_image(patch_rgbd, patch_intrinsics)
            console.print(f"Pointcloud from largest bounding box has {len(pointcloud)} points.")

            estimate = pointcloud.fit_plane_ransac(
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

            n_inliers = len(estimate.inlier_indices)
            total_points = len(estimate.pointcloud)
            inlier_ratio = 100 * n_inliers / total_points
            console.print(f"Inliers: {n_inliers}/{total_points} ({inlier_ratio:.1f}%)")

            # Update the live 3D visualization (non-blocking)
            vis.add_plane_estimate("current", estimate, plane_size_m=0.15, axes_size_m=0.1)


if __name__ == "__main__":
    main()
