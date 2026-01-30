"""Define classes to represent discretized search spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple, overload

import numpy as np

from robotics_utils.geometry import Point2D
from robotics_utils.spatial import DEFAULT_FRAME, Pose2D

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.spatial.poses import Multiply2D


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

        :param origin: Origin pose of the grid in a global frame (bottom-left corner)
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

    @classmethod
    def from_bounds(
        cls,
        resolution_m: float,
        ref_frame: str = DEFAULT_FRAME,
        *,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> DiscreteGrid2D:
        """Construct a 2D grid of cells covering the specified area of the x-y plane."""
        width_m = x_max - x_min
        height_m = y_max - y_min

        return DiscreteGrid2D(
            origin=Pose2D(x=x_min, y=y_min, yaw_rad=0.0, ref_frame=ref_frame),
            resolution_m=resolution_m,
            width_cells=int(width_m / resolution_m),
            height_cells=int(height_m / resolution_m),
        )

    def to_grid_frame(self, value_w: Multiply2D) -> Multiply2D:
        """Convert a world-frame 2D pose or point into the reference frame of the grid."""
        if isinstance(value_w, Pose2D):
            transform_g = self._transform_g_w @ value_w.to_homogeneous_matrix()
            return Pose2D.from_homogeneous_matrix(transform_g, ref_frame=self.frame_name)
        if isinstance(value_w, Point2D):
            h_coord_g = self._transform_g_w @ value_w.to_homogeneous_coordinate()
            return Point2D.from_homogeneous_coordinate(h_coord_g)

        raise NotImplementedError(f"Cannot transform type: {type(value_w)}.")

    def to_world_frame(self, value_g: Multiply2D) -> Multiply2D:
        """Convert a grid-frame 2D pose or point into the world reference frame."""
        if isinstance(value_g, Pose2D):
            transform_w = self._transform_w_g @ value_g.to_homogeneous_matrix()
            return Pose2D.from_homogeneous_matrix(transform_w, ref_frame=self.origin.ref_frame)
        if isinstance(value_g, Point2D):
            h_coord_w = self._transform_w_g @ value_g.to_homogeneous_coordinate()
            return Point2D.from_homogeneous_coordinate(h_coord_w)

        raise NotImplementedError(f"Cannot transform type: {type(value_g)}.")

    def local_x_to_col(self, x: float) -> int:
        """Convert a local-frame x-coordinate to the corresponding column index."""
        return int(np.floor(x / self.resolution_m))

    def local_y_to_row(self, y: float) -> int:
        """Convert a local-frame y-coordinate to the corresponding row index."""
        return int(np.floor(y / self.resolution_m))

    @overload
    def col_to_local_x(self, col: int) -> float: ...
    @overload
    def col_to_local_x(self, col: NDArray[np.intp]) -> NDArray[np.floating]: ...
    def col_to_local_x(self, col: int | NDArray[np.intp]) -> float | NDArray[np.floating]:
        """Convert column index/indices into local-frame x-coordinate(s) at cell center(s)."""
        return (col + 0.5) * self.resolution_m

    @overload
    def row_to_local_y(self, row: int) -> float: ...
    @overload
    def row_to_local_y(self, row: NDArray[np.intp]) -> NDArray[np.floating]: ...
    def row_to_local_y(self, row: int | NDArray[np.intp]) -> float | NDArray[np.floating]:
        """Convert row index/indices into local-frame y-coordinate(s) at cell center(s)."""
        return (row + 0.5) * self.resolution_m

    def world_to_cell(self, point_w: Point2D) -> GridCell:
        """Convert a world-frame (x, y) point to the corresponding grid indices."""
        homogeneous_coord_g = self._transform_g_w @ np.array([point_w.x, point_w.y, 1.0])
        col = self.local_x_to_col(homogeneous_coord_g[0])
        row = self.local_y_to_row(homogeneous_coord_g[1])
        return GridCell(row, col)

    def cell_to_world(self, cell: GridCell) -> Point2D:
        """Convert grid cell indices to a world-frame position at the cell center.

        :param cell: Grid cell (row, col) indices
        :return: Point in world frame at cell center
        """
        local_x = self.col_to_local_x(cell.col)
        local_y = self.row_to_local_y(cell.row)

        homogeneous_world = self._transform_w_g @ np.array([local_x, local_y, 1.0])
        return Point2D(homogeneous_world[0], homogeneous_world[1])

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
class DiscreteSE2:
    """A discrete node of indices corresponding to a 2D pose."""

    cell: GridCell
    heading_idx: int


@dataclass(frozen=True)
class DiscreteSE2Space:
    """A discrete space of 2D poses consisting of (x, y, yaw) values."""

    grid: DiscreteGrid2D
    headings: DiscreteAngles

    def discretize(self, pose: Pose2D) -> DiscreteSE2:
        """Compute the indices within the discrete SE(2) space nearest to the given pose."""
        if pose.ref_frame != self.grid.origin.ref_frame:
            raise ValueError(
                f"Cannot discretize a pose in frame '{pose.ref_frame}' using a "
                f"discrete grid with an origin in frame '{self.grid.origin.ref_frame}'.",
            )

        cell = self.grid.world_to_cell(Point2D(pose.x, pose.y))
        heading_idx = self.headings.nearest_index(pose.yaw_rad)
        return DiscreteSE2(cell=cell, heading_idx=heading_idx)

    def convert_discrete_to_pose(self, indices: DiscreteSE2) -> Pose2D:
        """Convert discrete indices in the space into the corresponding 2D pose."""
        point = self.grid.cell_to_world(indices.cell)
        yaw_rad = self.headings.index_to_angle_rad(indices.heading_idx)
        return Pose2D(x=point.x, y=point.y, yaw_rad=yaw_rad, ref_frame=self.grid.origin.ref_frame)
