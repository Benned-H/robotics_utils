"""Define classes to represent poses in 2D and 3D space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Tuple, TypeVar

import numpy as np

from robotics_utils.geometry import Point2D, Point3D
from robotics_utils.io.yaml_utils import load_yaml_data
from robotics_utils.spatial.frames import DEFAULT_FRAME
from robotics_utils.spatial.rotations import EulerRPY, Quaternion

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence
    from pathlib import Path

MultiplyT = TypeVar("MultiplyT", "Pose3D", Point3D)

XYZ_RPY = Tuple[float, float, float, float, float, float]
"""A 6-tuple of (x, y, z, roll, pitch, yaw) values."""


@dataclass(frozen=True)
class Pose2D:
    """A position and orientation on the 2D plane."""

    x: float
    y: float
    yaw_rad: float
    ref_frame: str = DEFAULT_FRAME  # Reference frame of the pose

    def __iter__(self) -> Iterator[float]:
        """Provide an iterator over the (x,y,yaw) values of the 2D pose."""
        yield from self.to_tuple()

    @classmethod
    def from_sequence(cls, pose_seq: Sequence[float], ref_frame: str = DEFAULT_FRAME) -> Pose2D:
        """Construct a Pose2D from a sequence of values.

        :param pose_seq: Sequence of [x, y, yaw] values
        :param ref_frame: Reference frame of the constructed pose (defaults to `DEFAULT_FRAME`)
        :return: Constructed Pose2D instance
        """
        if len(pose_seq) != 3:
            raise ValueError(f"Pose2D expects 3 values, got {len(pose_seq)}")

        return Pose2D(pose_seq[0], pose_seq[1], pose_seq[2], ref_frame)

    def to_tuple(self) -> tuple[float, float, float]:
        """Convert the 2D pose into an (x,y,yaw) tuple."""
        return (float(self.x), float(self.y), float(self.yaw_rad))

    @property
    def position(self) -> Point2D:
        """Retrieve the (x,y) position of the 2D pose."""
        return Point2D(self.x, self.y)

    def to_yaml_data(self, default_frame: str | None) -> dict[str, Any] | list[float]:
        """Convert the Pose2D into a form suitable for export to YAML.

        :param default_frame: Default frame assumed in the parent YAML file (ignored if None)
        :return: Dictionary with the pose's data and frame, or a list if its frame is the default
        """
        if default_frame is not None and self.ref_frame == default_frame:
            return list(self)

        return {"x_y_yaw": list(self), "frame": self.ref_frame}

    def to_3d(self) -> Pose3D:
        """Convert the 2D pose into an equivalent Pose3D."""
        return Pose3D.from_xyz_rpy(
            x=self.x,
            y=self.y,
            yaw_rad=self.yaw_rad,
            ref_frame=self.ref_frame,
        )

    def to_homogeneous_matrix(self) -> np.ndarray:
        """Convert the 2D pose into a 3x3 homogeneous transformation matrix.

        Reference: https://lavalle.pl/planning/node108.html
        """
        cos_yaw = np.cos(self.yaw_rad)
        sin_yaw = np.sin(self.yaw_rad)
        return np.array([[cos_yaw, -sin_yaw, self.x], [sin_yaw, cos_yaw, self.y], [0, 0, 1]])


@dataclass(frozen=True)
class Pose3D:
    """A position and orientation in 3D space."""

    position: Point3D
    orientation: Quaternion
    ref_frame: str = DEFAULT_FRAME

    def __matmul__(self, other: MultiplyT) -> MultiplyT:
        """Compose the homogeneous transformation matrix of this pose with another object.

        :param other: 3D pose or 3D point right-multiplied with this pose
        :return: Result from the matrix multiplication
        """
        if isinstance(other, Pose3D):
            return self._matrix_multiply_with_pose(other)
        if isinstance(other, Point3D):
            return self._matrix_multiply_with_point(other)

        raise NotImplementedError(f"Cannot matrix-multiply Pose3D with: {other}")

    def _matrix_multiply_with_pose(self, other: Pose3D) -> Pose3D:
        """Multiply the homogeneous transformation matrix of this pose with another pose.

        Consider: pose_A_B @ pose_B_C = pose_A_C, meaning the pose of 'C' relative to frame A.
            Therefore, we see that the resulting pose takes the "left-side" reference frame.

        :param other: Pose defining the right-side matrix in the multiplication
        :return: Pose3D resulting from the matrix multiplication
        """
        left_m = self.to_homogeneous_matrix()
        right_m = other.to_homogeneous_matrix()
        result_ref_frame = self.ref_frame  # Result takes the "leftmost" reference frame
        return Pose3D.from_homogeneous_matrix(left_m @ right_m, result_ref_frame)

    def _matrix_multiply_with_point(self, other: Point3D) -> Point3D:
        """Multiply the homogeneous transformation matrix of this pose with a 3D point.

        :param other: 3D point treated as a homogeneous coordinate in the multiplication
        :return: Point3D resulting from the matrix multiplication
        """
        result = self.to_homogeneous_matrix() @ other.to_homogeneous_coordinate()
        return Point3D.from_homogeneous_coordinate(result)

    def __str__(self) -> str:
        """Return a human-readable string representation of the Pose3D."""
        xyz_rpy_strings = [f"{value:.3f}" for value in self.to_xyz_rpy()]
        xyz_rpy = ", ".join(xyz_rpy_strings)
        return f'Pose3D([{xyz_rpy}], ref_frame="{self.ref_frame}")'

    @property
    def yaw_rad(self) -> float:
        """Retrieve the yaw (radians) from the orientation of the pose."""
        return self.orientation.to_euler_rpy().yaw_rad

    @classmethod
    def identity(cls, ref_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D corresponding to the identity transformation."""
        return Pose3D(Point3D.identity(), Quaternion.identity(), ref_frame)

    @classmethod
    def from_xyz_rpy(
        cls,
        x: float = 0.0,
        y: float = 0.0,
        z: float = 0.0,
        roll_rad: float = 0.0,
        pitch_rad: float = 0.0,
        yaw_rad: float = 0.0,
        ref_frame: str = DEFAULT_FRAME,
    ) -> Pose3D:
        """Construct a Pose3D from the given XYZ coordinates and Euler RPY angles.

        :param x: Translation along the x-axis
        :param y: Translation along the y-axis
        :param z: Translation along the z-axis
        :param roll_rad: Fixed-frame roll angle (radians) about the x-axis
        :param pitch_rad: Fixed-frame pitch angle (radians) about the y-axis
        :param yaw_rad: Fixed-frame yaw angle (radians) about the z-axis
        :param ref_frame: Reference frame of the constructed pose
        :return: Constructed Pose3D instance
        """
        position = Point3D(x, y, z)
        orientation = EulerRPY(roll_rad, pitch_rad, yaw_rad).to_quaternion()

        return Pose3D(position, orientation, ref_frame)

    def to_xyz_rpy(self) -> XYZ_RPY:
        """Convert the pose into a tuple of its (x, y, z, roll, pitch, yaw) values.

        :return: 6-tuple of (x, y, z, roll, pitch, yaw) values with angles in radians
        """
        return (*self.position.to_tuple(), *self.orientation.to_euler_rpy().to_tuple())

    @classmethod
    def from_sequence(cls, data: XYZ_RPY | list[float], ref_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D from the given sequence of XYZ-RPY data.

        :param data: Sequence of six floats specifying (x, y, z, roll, pitch, yaw)
        :param ref_frame: Reference frame of the constructed Pose3D
        :return: Constructed Pose3D instance
        """
        if len(data) != 6:
            raise ValueError(f"Cannot construct Pose3D from sequence of length {len(data)}.")
        x, y, z, roll, pitch, yaw = data
        return Pose3D.from_xyz_rpy(x, y, z, roll, pitch, yaw, ref_frame)

    @classmethod
    def from_homogeneous_matrix(cls, matrix: np.ndarray, ref_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D from a 4x4 homogeneous transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 matrix but received shape {matrix.shape}")

        position = Point3D(float(matrix[0, 3]), float(matrix[1, 3]), float(matrix[2, 3]))
        orientation = Quaternion.from_homogeneous_matrix(matrix)
        return Pose3D(position, orientation, ref_frame)

    def to_homogeneous_matrix(self) -> np.ndarray:
        """Convert the Pose3D into a 4x4 homogeneous transformation matrix."""
        matrix = self.orientation.to_homogeneous_matrix()
        matrix[:3, 3] = self.position.to_array()
        return matrix

    @classmethod
    def from_yaml_data(cls, pose_data: dict | list, default_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D instance from data imported from YAML.

        :param pose_data: Dictionary or list of YAML data representing a 3D pose
        :param default_frame: Default frame used for the pose, if the YAML doesn't provide one
        :return: Constructed Pose3D instance
        :raises TypeError: If the given YAML data has an unsupported type
        """
        if isinstance(pose_data, dict):
            xyz_rpy = pose_data["xyz_rpy"]
            ref_frame = pose_data["frame"]
        elif isinstance(pose_data, list):
            xyz_rpy = pose_data
            ref_frame = default_frame
        else:
            raise TypeError(f"Cannot load Pose3D from YAML data of type {type(pose_data)}")

        return Pose3D.from_sequence(xyz_rpy, ref_frame)

    def to_yaml_data(self) -> dict[str, Any]:
        """Convert the pose into a dictionary suitable for export to YAML."""
        return {"xyz_rpy": self.to_xyz_rpy(), "frame": self.ref_frame}

    @classmethod
    def load_named_poses(cls, yaml_path: Path, collection_name: str) -> dict[str, Pose3D]:
        """Load a collection of named poses from the given YAML file.

        :param yaml_path: Path to a YAML file containing pose data
        :param collection_name: Name of collection of poses to be imported (e.g., "object_poses")
        :return: Dictionary mapping pose-frame names to their imported 3D poses
        """
        yaml_data = load_yaml_data(yaml_path, required_keys={collection_name})
        default_frame = yaml_data.get("default_frame", DEFAULT_FRAME)
        poses_data: dict[str, Any] = yaml_data[collection_name]

        return {
            pose_name: Pose3D.from_yaml_data(pose_data, default_frame)
            for pose_name, pose_data in poses_data.items()
        }

    def to_2d(self) -> Pose2D:
        """Convert the 3D pose into a 2D pose by discarding its z-coordinate, roll, and pitch."""
        (x, y, z, roll, pitch, yaw) = self.to_xyz_rpy()
        return Pose2D(x=x, y=y, yaw_rad=yaw, ref_frame=self.ref_frame)

    def inverse(self, pose_frame: str) -> Pose3D:
        """Return a pose representing the inverse transformation of this pose.

        :param pose_frame: Name of the reference frame represented by this pose
        """
        inverse_matrix = np.linalg.inv(self.to_homogeneous_matrix())
        return Pose3D.from_homogeneous_matrix(inverse_matrix, pose_frame)

    def approx_equal(self, other: Pose3D, rtol: float = 1e-05, atol: float = 1e-08) -> bool:
        """Evaluate whether another Pose3D is approximately equal to this one."""
        return (
            self.ref_frame == other.ref_frame
            and self.position.approx_equal(other.position, rtol=rtol, atol=atol)
            and self.orientation.approx_equal(other.orientation, rtol=rtol, atol=atol)
        )
