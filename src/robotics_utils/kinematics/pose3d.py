"""Define a class to represent poses in 3D space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.point3d import Point3D
from robotics_utils.kinematics.rotations import EulerRPY, Quaternion


@dataclass
class Pose3D:
    """A position and orientation in 3D space."""

    position: Point3D
    orientation: Quaternion
    ref_frame: str = DEFAULT_FRAME

    def __matmul__(self, other: Pose3D) -> Pose3D:
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

    @classmethod
    def identity(cls) -> Pose3D:
        """Construct a Pose3D corresponding to the identity transformation."""
        return Pose3D(Point3D.identity(), Quaternion.identity())

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

    def to_xyz_rpy(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Convert the pose into the corresponding (x, y, z) and (roll, pitch, yaw) tuples.

        :return: Pair of tuples (x, y, z) and (roll, pitch, yaw) with angles in radians
        """
        return (self.position.to_tuple(), self.orientation.to_euler_rpy().to_tuple())

    @classmethod
    def from_list(cls, xyz_rpy: list[float], ref_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D from the given list of XYZ-RPY data.

        :param xyz_rpy: List of six floats specifying (x, y, z, roll, pitch, yaw)
        :param ref_frame: Reference frame of the constructed Pose3D
        :return: Constructed Pose3D instance
        """
        if len(xyz_rpy) != 6:
            raise ValueError(f"Cannot construct Pose3D from list of length {len(xyz_rpy)}.")
        x, y, z, roll, pitch, yaw = xyz_rpy
        return Pose3D.from_xyz_rpy(x, y, z, roll, pitch, yaw, ref_frame)

    def to_list(self) -> list[float]:
        """Convert the Pose3D into a list of the form [x, y, z, roll (radians), pitch, yaw]."""
        (x, y, z), (roll_rad, pitch_rad, yaw_rad) = self.to_xyz_rpy()
        return [x, y, z, roll_rad, pitch_rad, yaw_rad]

    @classmethod
    def from_homogeneous_matrix(cls, matrix: np.ndarray, ref_frame: str = DEFAULT_FRAME) -> Pose3D:
        """Construct a Pose3D from a 4x4 homogeneous transformation matrix."""
        if matrix.shape != (4, 4):
            raise ValueError(f"Expected a 4x4 matrix but received shape {matrix.shape}")

        position = Point3D(matrix[0, 3], matrix[1, 3], matrix[2, 3])
        orientation = Quaternion.from_rotation_matrix(matrix[:3, :3])
        return Pose3D(position, orientation, ref_frame)

    def to_homogeneous_matrix(self) -> np.ndarray:
        """Convert the Pose3D into a 4x4 homogeneous transformation matrix."""
        matrix = np.eye(4)
        matrix[:3, 3] = self.position.to_array()
        matrix[:3, :3] = self.orientation.to_rotation_matrix()
        return matrix

    def approx_equal(self, other: Pose3D) -> bool:
        """Evaluate whether another Pose3D is approximately equal to this one."""
        return (
            self.ref_frame == other.ref_frame
            and self.position.approx_equal(other.position)
            and self.orientation.approx_equal(other.orientation)
        )
