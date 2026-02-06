"""Define a class to precompute occupancy masks of a robot footprint at discretized poses."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.spatial import Pose2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.motion_planning.discretization import DiscreteSE2, DiscreteSE2Space
    from robotics_utils.motion_planning.rectangular_footprint import RectangularFootprint


class FootprintCellOffsets:
    """Precomputed masks of grid cell offsets for a robot footprint at a set of discrete headings.

    For each possible discrete heading, this class stores the list of (row, col) offsets
    from the robot's center cell that the footprint covers. Collision checking then
    simply iterates over these offsets and checks the occupancy grid.
    """

    def __init__(self, se2_space: DiscreteSE2Space, footprint: RectangularFootprint) -> None:
        """Precompute robot footprint cell offsets for each discrete heading.

        :param se2_space: Discrete space of possible robot base poses
        :param footprint: Rectangular model of a robot's base footprint
        """
        self.se2_space = se2_space
        self.footprint = footprint

        # Precompute footprint cell offsets for each heading
        self._offsets_by_heading: dict[int, list[tuple[int, int]]] = defaultdict(list)
        self._precompute_footprint_offsets()

    def _precompute_footprint_offsets(self) -> None:
        """Precompute the grid cell offsets covered by the robot footprint at each heading."""
        resolution_m = self.se2_space.grid.resolution_m

        max_extent_m = max(
            abs(self.footprint.max_x_m),
            abs(self.footprint.min_x_m),
            self.footprint.half_length_y_m,
        )
        search_radius_cells = int(np.ceil(max_extent_m / resolution_m)) + 1

        for heading_idx in range(self.se2_space.headings.num_angles):
            # Construct the pose of the grid in the robot frame (only differs in angle)
            angle_rad = self.se2_space.headings.index_to_angle_rad(heading_idx)
            pose_r_g = Pose2D(x=0.0, y=0.0, yaw_rad=-angle_rad)
            transform_r_g = pose_r_g.to_homogeneous_matrix()

            # Check all cells potentially within the bounding radius
            for dr in range(-search_radius_cells, search_radius_cells + 1):
                for dc in range(-search_radius_cells, search_radius_cells + 1):
                    # Find the cell center (frame c) in the grid frame (frame g)
                    position_g_c = Point2D(x=dc * resolution_m, y=dr * resolution_m)
                    h_coord_g_c = position_g_c.to_homogeneous_coordinate()

                    h_coord_r_c = transform_r_g @ h_coord_g_c
                    body_x = h_coord_r_c[0]
                    body_y = h_coord_r_c[1]

                    # Check if the cell center is inside the robot footprint
                    if (
                        self.footprint.min_x_m <= body_x <= self.footprint.max_x_m
                        and abs(body_y) <= self.footprint.half_length_y_m
                    ):
                        self._offsets_by_heading[heading_idx].append((dr, dc))

    def is_collision_free(self, state: DiscreteSE2, occupied_mask: NDArray[np.bool_]) -> bool:
        """Check whether the robot collides with obstacles at the given discrete state.

        :param state: Discretized SE(2) pose (grid cell + heading index)
        :param occupied_mask: Boolean mask where True indicates occupied cells
        :return: True if the state is collision-free, else False
        """
        offsets = np.array(self._offsets_by_heading[state.heading_idx])  # Shape (N, 2)
        rows = state.cell.row + offsets[:, 0]  # Shape (N,) of cell row indices
        cols = state.cell.col + offsets[:, 1]  # Shape (N,) of cell column indices

        valid_cells = (
            (rows >= 0)
            & (rows < occupied_mask.shape[0])
            & (cols >= 0)
            & (cols < occupied_mask.shape[1])
        )
        if not np.all(valid_cells):  # Treat out-of-bounds cells as a collision
            return False

        return not np.any(occupied_mask[rows, cols])
