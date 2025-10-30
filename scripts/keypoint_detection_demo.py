"""Demo script that uses Gemini Robotics-ER 1.5 to detect keypoints in an image."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from robotics_utils.io.repls import ObjectDetectionREPL
from robotics_utils.perception.vision.vlms import ObjectKeypoints
from robotics_utils.perception.vision.vlms.gemini_robotics_er import GeminiRoboticsER
from robotics_utils.visualization import display_in_window


def display_detected_keypoints(console: Console, keypoints: ObjectKeypoints) -> None:
    """Display the given bounding box detections using the given console."""
    display_in_window(keypoints, "Object Keypoints")


@click.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.argument("api_key")
def object_detection(image_path: Path, api_key: str) -> None:
    """Run object detection in an interactive loop."""
    console = Console()
    detector = GeminiRoboticsER(api_key)

    repl: ObjectDetectionREPL[ObjectKeypoints] = ObjectDetectionREPL(
        console,
        image_path,
        detect_func=detector.detect_keypoints,
        display_func=display_detected_keypoints,
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
