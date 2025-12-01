"""Define a class to detect AprilTags from images."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass, replace
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pupil_apriltags

from robotics_utils.kinematics import EulerRPY, Point3D, Pose3D, Quaternion
from robotics_utils.vision import CameraIntrinsics, PixelXY, RGBCamera, RGBImage
from robotics_utils.visualization import Displayable

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from robotics_utils.vision.fiducials.visual_fiducials import FiducialSystem


@dataclass
class TagDetection(Displayable):
    """An AprilTag detected in an image."""

    id: int
    family: str
    pose: Pose3D
    """Estimated pose of the tag (x = right, y = down, z = through the tag)."""

    hamming: int  # How many error bits were corrected?
    corners: list[PixelXY]  # Pixel coordinates of detection corners (wraps counter-clockwise)
    center: PixelXY  # Pixel coordinate of the detection's center
    visualized: RGBImage  # Image in which the AprilTag detection has been visualized

    @classmethod
    def from_pupil_apriltags(cls, d: pupil_apriltags.Detection, image: RGBImage) -> TagDetection:
        """Construct a TagDetection from a pupil_apriltags.Detection and an RGB image.

        :param d: AprilTag detection in the pupil_apriltags format
        :param image: Image in which the detection occurred
        :return: Tag detection in this dataclass format
        """
        translation = Point3D.from_array(np.asarray(d.pose_t))

        # Apply 180° rotation about the z-axis to correct for pupil_apriltags convention
        rotation_correction = EulerRPY(0, 0, np.pi).to_quaternion().to_rotation_matrix()
        corrected_matrix = np.asarray(d.pose_R) @ rotation_correction
        corrected_rotation = Quaternion.from_rotation_matrix(corrected_matrix)

        corners = np.asarray(d.corners, dtype=np.float32).reshape((4, 1, 2))
        corner_pixels = [PixelXY(corners[row, :, :]) for row in range(4)]
        center_pixel = PixelXY(np.asarray(d.center))

        # Draw the detection on the provided image using its corners
        visualized_data = deepcopy(image.data)
        ids = np.asarray([[d.tag_id]], dtype=np.int32)
        cv2.aruco.drawDetectedMarkers(visualized_data, [corners], ids)

        return TagDetection(
            id=d.tag_id,
            family=d.tag_family,
            pose=Pose3D(translation, corrected_rotation),
            hamming=d.hamming,
            corners=corner_pixels,
            center=center_pixel,
            visualized=RGBImage(visualized_data),
        )

    def convert_for_visualization(self) -> NDArray[np.uint8]:
        """Return an image visualizing the AprilTag detection."""
        return self.visualized.convert_for_visualization()


class AprilTagDetector:
    """A wrapper class around pupil_apriltags.Detector."""

    def __init__(
        self,
        fiducial_system: FiducialSystem,
        quad_decimate: float = 1.0,
        quad_sigma: float = 0.8,
        nthreads: int = 1,
        refine_edges: int = 1,
        decode_sharpening: float = 0.25,
    ) -> None:
        """Initialize the detector with defaults appropriate for noisy cameras.

        Default values were selected based on pupil_apriltags documentation.

        Reference: https://pupil-apriltags.readthedocs.io/en/stable/api.html
        """
        self._fiducial_system = fiducial_system
        self._tag_sizes_cm: list[float] = []
        self._tag_ids_per_size_cm: defaultdict[int, set[int]] = defaultdict(set)
        """A map from each index in `self._tag_sizes_cm` to the set of IDs of tags that size."""

        for marker in fiducial_system.markers.values():
            marker_added = False
            for idx, size_cm in enumerate(self._tag_sizes_cm):
                if np.allclose(marker.size_cm, size_cm):
                    self._tag_ids_per_size_cm[idx].add(marker.id)
                    marker_added = True
                    break

            if not marker_added:
                new_idx = len(self._tag_sizes_cm)
                self._tag_sizes_cm.append(marker.size_cm)
                self._tag_ids_per_size_cm[new_idx].add(marker.id)

        self._detector = pupil_apriltags.Detector(
            families="tag36h11",
            nthreads=nthreads,
            quad_decimate=quad_decimate,  # >1 speeds up, slight accuracy hit
            quad_sigma=quad_sigma,  # mild blur helps noisy edges
            refine_edges=refine_edges,
            decode_sharpening=decode_sharpening,
        )

    def detect_in_image(self, image: RGBImage, intrinsics: CameraIntrinsics) -> list[TagDetection]:
        """Detect AprilTags in an image taken with the given camera intrinsics.

        :param image: RGB image to detect AprilTags in
        :param intrinsics: Intrinsics of the camera that took the image
        :return: List of tag detections, represented as TagDetection objects (may be empty)
        """
        intrinsics_tuple = astuple(intrinsics)
        gray_image = np.asarray(cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY), dtype=np.uint8)

        output: list[TagDetection] = []

        for idx, size_cm in enumerate(self._tag_sizes_cm):
            size_m = size_cm / 100.0

            raw_detections = self._detector.detect(
                gray_image,
                estimate_tag_pose=True,
                camera_params=intrinsics_tuple,
                tag_size=size_m,
            )

            tags_of_size_cm = self._tag_ids_per_size_cm[idx]
            for raw_det in raw_detections:
                det = TagDetection.from_pupil_apriltags(raw_det, image)
                if det.id in tags_of_size_cm:
                    output.append(det)

        return output

    def detect_from_camera(self, camera: RGBCamera) -> list[TagDetection]:
        """Detect AprilTags in a new image captured using the given camera.

        :return: List of AprilTag detections, with their estimated poses w.r.t. the camera
        """
        if camera.frame_name is None:
            raise ValueError(f"Cannot estimate AprilTag poses w/o frame for camera: {camera.name}")

        rgb_image = camera.get_image()
        detections = self.detect_in_image(rgb_image, camera.intrinsics)
        for d in detections:
            d.pose = replace(d.pose, ref_frame=camera.frame_name)
        return detections
