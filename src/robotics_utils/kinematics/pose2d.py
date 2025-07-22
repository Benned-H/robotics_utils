"""Define a class to represent poses in 2D space."""

from dataclasses import dataclass

from robotics_utils.kinematics import DEFAULT_FRAME
from robotics_utils.kinematics.pose3d import Pose3D


@dataclass
class Pose2D:
    """A position and orientation on the 2D plane."""

    x: float
    y: float
    yaw_rad: float
    ref_frame: str = DEFAULT_FRAME  # Reference frame of the pose

    def to_3d(self) -> Pose3D:
        """Convert the 2D pose into an equivalent Pose3D."""
        return Pose3D.from_xyz_rpy(
            x=self.x,
            y=self.y,
            yaw_rad=self.yaw_rad,
            ref_frame=self.ref_frame,
        )
