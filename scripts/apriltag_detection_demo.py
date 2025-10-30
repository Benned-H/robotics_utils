"""Demo script that detects AprilTags in an image loaded from file.

An example image is available here:
    https://blog.fixermark.com/posts/2022/april-tags-python-recognizer/
"""

from copy import deepcopy
from pathlib import Path

from robotics_utils.perception.pose_estimation import (
    AprilTagDetector,
    FiducialMarker,
    FiducialSystem,
)
from robotics_utils.perception.sensors.cameras import CameraIntrinsics
from robotics_utils.perception.vision import RGBImage
from robotics_utils.visualization import display_in_window

IMAGE_PATH = Path("images/apriltags.png")
TAG_SIZE_CM = 17.78
MARKERS = [
    FiducialMarker(1, TAG_SIZE_CM, {}),
    FiducialMarker(42, TAG_SIZE_CM, {}),
    FiducialMarker(422, TAG_SIZE_CM, {}),
]
CAMERAS = ["camera"]
INTRINSICS = CameraIntrinsics(fx=1607.36, fy=1607.36, x0=1024.0, y0=768.0)


def main() -> None:
    """Attempt to detect AprilTags in an image loaded from a hard-coded filepath."""
    system = FiducialSystem({m.id: m for m in MARKERS}, CAMERAS)
    detector = AprilTagDetector(system)

    image = RGBImage.from_file(IMAGE_PATH)
    vis_image = deepcopy(image)

    detections = detector.detect(image, INTRINSICS)
    for d in detections:
        d.draw_on_image(vis_image)

    display_in_window(vis_image, window_title="AprilTag Detections")


if __name__ == "__main__":
    main()
