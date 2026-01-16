"""Define classes to represent discretized search spaces."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.spatial import Pose2D


class GridCell(NamedTuple):
    """A pair of (row, column) indices to a cell in a discrete grid."""

    row: int
    col: int


class DiscreteGrid2D:
    """A discrete 2D grid of cells on the global x-y plane."""

    def __init__(
        self,
        origin: Pose2D,
        resolution_m: float,
        width_cells: int,
        height_cells: int,
        frame_name: str = "grid",
    ) -> None:
        """Initialize the 2D grid.

        :param origin: Origin pose of the grid in a global frame (upper-left corner)
        :param resolution_m: Size of cells in the grid (meters)
        :param width_cells: Grid width in cells (along x-axis)
        :param height_cells: Grid height in cells (along y-axis)
        :param frame_name: Optional name of the grid's local reference frame (defaults to "grid")
        """
        self.origin = origin  # pose_w_g (grid in world frame)
        self.resolution_m = resolution_m
        self.width_cells = width_cells
        self.height_cells = height_cells
        self.frame_name = frame_name

        # Cache transformation matrices for efficient coordinate conversion
        self._transform_w_g = origin.to_homogeneous_matrix()  # Grid frame -> World frame
        self._transform_g_w = np.linalg.inv(self._transform_w_g)  # World frame -> Grid frame

    def to_grid_frame(self, pose_w: Pose2D) -> Pose2D:
        """Convert a world-frame 2D pose into the reference frame of the grid."""
        transform_g = self._transform_g_w @ pose_w.to_homogeneous_matrix()
        return Pose2D.from_homogeneous_matrix(transform_g, ref_frame=self.frame_name)

    def to_world_frame(self, pose_g: Pose2D) -> Pose2D:
        """Convert a grid-frame 2D pose into the world reference frame."""
        transform_w = self._transform_w_g @ pose_g.to_homogeneous_matrix()
        return Pose2D.from_homogeneous_matrix(transform_w, ref_frame=self.origin.ref_frame)

    def local_x_to_col(self, x: float) -> int:
        """Convert a local-frame x-coordinate to the corresponding column index."""
        return int(np.floor(x / self.resolution_m))

    def local_y_to_row(self, y: float) -> int:
        """Convert a local-frame y-coordinate to the corresponding row index."""
        return int(np.floor(-y / self.resolution_m))

    def world_to_cell(self, point_w: Point2D) -> GridCell:
        """Convert a world-frame (x, y) point to the corresponding grid indices."""
        homogeneous_coord_g = self._transform_g_w @ np.array([point_w.x, point_w.y, 1.0])
        col = self.local_x_to_col(homogeneous_coord_g[0])
        row = self.local_y_to_row(homogeneous_coord_g[1])
        return GridCell(row, col)

    def is_valid_cell(self, cell: GridCell) -> bool:
        """Check whether the given cell coordinate is within the grid."""
        return 0 <= cell.row < self.height_cells and 0 <= cell.col < self.width_cells


class DiscreteAngles:
    """A discrete space of evenly-spaced angles (radians)."""

    def __init__(self, num_angles: int = 8) -> None:
        """Initialize the discrete space with the number of angles it should contain."""
        self.num_angles = num_angles
        self.angles_space = np.linspace(0, 2 * np.pi, num=num_angles, endpoint=False)

    def index_to_angle_rad(self, index: int) -> float:
        """Retrieve the angle (radians) with the given index."""
        if index < 0 or index >= self.num_angles:
            raise ValueError(f"Invalid angle index: {index} (num_angles={self.num_angles})")
        return float(self.angles_space[index])

    def nearest_index(self, angle_rad: float) -> int:
        """Find the index of the nearest discretized angle to the given angle (in radians)."""
        diff_rad = self.angles_space - angle_rad
        normalized_rad = np.abs(np.arctan2(np.sin(diff_rad), np.cos(diff_rad)))
        return int(np.argmin(normalized_rad))


@dataclass(frozen=True)
class DiscretePoseSpace:
    """A discrete space of 2D poses consisting of (x, y, yaw) values."""

    grid: DiscreteGrid2D
    angles: DiscreteAngles
