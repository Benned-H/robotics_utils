"""Demo script that uses Gemini Robotics-ER 1.5 to detect keypoints in an image."""

from __future__ import annotations

from pathlib import Path

import click

from robotics_utils.io.repls import ObjectDetectionREPL
from robotics_utils.vision.vlms import DetectedKeypoints
from robotics_utils.vision.vlms.gemini_robotics_er import GeminiRoboticsER
from robotics_utils.visualization import display_in_window


def display_detected_keypoints(keypoints: DetectedKeypoints) -> None:
    """Display the given keypoint detections."""
    display_in_window(keypoints, "Object Keypoints")


@click.command()
@click.argument("api_key")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def object_detection(api_key: str, image_path: Path) -> None:
    """Run object detection in an interactive loop."""
    detector = GeminiRoboticsER(api_key)

    repl: ObjectDetectionREPL[DetectedKeypoints] = ObjectDetectionREPL(
        image_path,
        detect_func=detector.detect_keypoints,
        display_func=display_detected_keypoints,
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
