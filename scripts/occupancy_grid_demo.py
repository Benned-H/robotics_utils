"""Create an occupancy grid using simulated laser scan data.

To run this script, use the commands:

    uv venv --clear && uv sync --extra vision
    uv run scripts/occupancy_grid_demo.py

"""

import click
import cv2
import numpy as np

from robotics_utils.io import console
from robotics_utils.io.cli_handlers import ParamUI, handle_float, handle_int, handle_pose_2d
from robotics_utils.motion_planning import DiscreteGrid2D
from robotics_utils.perception import LaserScan2D, OccupancyGrid2D
from robotics_utils.spatial import Pose2D
from robotics_utils.vision import RGBImage
from robotics_utils.visualization import display_in_window

MAX_LOG_ODDS = 10
"""Maximum value for log-odds occupancies; greater values are filled in as this value."""

MIN_LOG_ODDS = -10
"""Minimum value for log-odds occupancies; lesser values are filled in as this value."""


def main() -> None:
    """Generate simulated laser scan data and use it to create an occupancy grid."""
    rng = np.random.default_rng()

    lidar_pose_ui = ParamUI[Pose2D](prompt="Provide a LiDAR sensor pose:")
    num_scans_ui = ParamUI[int](prompt="Number of LiDAR scans to simulate:", default=1)
    num_beams_ui = ParamUI[int](prompt="Number of LiDAR beams to use:", default=10000)
    std_dev_ui = ParamUI[float](
        prompt="Standard deviation (m) of LiDAR range measurements:",
        default=0.1,
    )

    while True:
        console.print()
        lidar_pose = handle_pose_2d(lidar_pose_ui)
        num_beams = handle_int(num_beams_ui)
        num_scans = handle_int(num_scans_ui)
        std_dev_m = handle_float(std_dev_ui)

        # Initialize an empty occupancy grid
        discrete_grid = DiscreteGrid2D(
            origin=Pose2D(-5, 5, 0),
            resolution_m=0.05,
            width_cells=200,
            height_cells=200,
        )
        occ_grid = OccupancyGrid2D(discrete_grid)

        # Generate noiseless beam data (reused across all simulated scans)
        beam_angles_rad = np.linspace(start=0.0, stop=2 * np.pi, num=num_beams)
        beam_ranges_m = np.abs(10.0 * np.sin(beam_angles_rad) * np.cos(beam_angles_rad))
        console.print(f"Shape of beam angles array: {beam_angles_rad.shape}")
        console.print(f"Shape of beam ranges array: {beam_ranges_m.shape}")

        for _ in range(num_scans):
            beam_noise_m = rng.normal(scale=std_dev_m, size=beam_ranges_m.shape)
            noisy_ranges_m = beam_ranges_m + beam_noise_m

            beam_data = np.stack([noisy_ranges_m, beam_angles_rad], axis=-1, dtype=np.float32)
            console.print(f"Combined beam data shape: {beam_data.shape}")

            # Construct a 2D laser scan using the generated range and angle values
            laser_scan = LaserScan2D(lidar_pose, beam_data, range_min_m=0.01, range_max_m=5.0)

            # Update the occupancy grid using the simulated laser scan
            occ_grid.update(scan=laser_scan)

        # Extract the occupancy map and its thresholded mask
        raw_log_odds = occ_grid.log_odds.copy()
        raw_log_odds[raw_log_odds > MAX_LOG_ODDS] = MAX_LOG_ODDS
        raw_log_odds[raw_log_odds < MIN_LOG_ODDS] = MIN_LOG_ODDS

        normalized_log_odds = cv2.normalize(raw_log_odds, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        uint8_log_odds = np.asarray(normalized_log_odds, dtype=np.uint8)
        console.print(f"Shape of normalized log-odds data: {uint8_log_odds.shape}")

        occupied_mask = occ_grid.get_occupied_mask().astype(np.uint8)
        normalized_mask = cv2.normalize(occupied_mask, None, 0, 255, norm_type=cv2.NORM_MINMAX)
        uint8_mask = np.asarray(normalized_mask, dtype=np.uint8)
        console.print(f"Shape of normalized occupancy data: {uint8_mask.shape}")

        stacked_grid_data = np.vstack([uint8_log_odds, uint8_mask], dtype=np.uint8)
        console.print(f"Shape of stacked grid data: {stacked_grid_data.shape}")

        stacked_image_data = cv2.applyColorMap(stacked_grid_data, colormap=cv2.COLORMAP_BONE)
        rgb_image = RGBImage(data=np.asarray(stacked_image_data))

        display_in_window(rgb_image, title="Log-Odds Occupancy and Occupancy Mask")

        if click.confirm("Exit program?"):
            console.print("[green]Bye![/]")
            break


if __name__ == "__main__":
    main()
