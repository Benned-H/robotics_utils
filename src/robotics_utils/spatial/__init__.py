"""Import classes and definitions representing 3D coordinate frames, poses, and rotations."""

from .averaging import average_poses as average_poses
from .averaging import average_positions as average_positions
from .averaging import average_quaternions as average_quaternions
from .distances import angle_between_quaternions_deg as angle_between_quaternions_deg
from .distances import euclidean_distance_2d_m as euclidean_distance_2d_m
from .distances import euclidean_distance_3d_m as euclidean_distance_3d_m
from .frames import DEFAULT_FRAME as DEFAULT_FRAME
from .poses import XYZ_RPY as XYZ_RPY
from .poses import Pose2D as Pose2D
from .poses import Pose3D as Pose3D
from .rotations import EulerRPY as EulerRPY
from .rotations import Quaternion as Quaternion
