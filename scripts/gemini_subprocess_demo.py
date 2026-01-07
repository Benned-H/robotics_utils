"""Demo script (Python 3.8) that calls Gemini Robotics-ER 1.5 in a Python 3.10+ subprocess."""

from pathlib import Path

import click

from robotics_utils.vision import RGBImage
from robotics_utils.vision.vlms.gemini import GeminiRoboticsBridge
from robotics_utils.visualization import display_in_window


@click.command()
@click.argument("api_key")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.argument("queries")
@click.option("--detection_type", type=click.Choice(["keypoints"]), default="keypoints")
def call_gemini(api_key: str, image_path: Path, queries: str, detection_type: str) -> None:
    """Call Gemini Robotics-ER 1.5 in a subprocess and display the resulting detections."""
    queries_list = [q.strip() for q in queries.split(",")]
    image = RGBImage.from_file(image_path)

    if detection_type == "keypoints":
        detections = GeminiRoboticsBridge.detect_keypoints(api_key, image, queries=queries_list)
        display_in_window(detections, "Object Keypoints")
    else:
        raise ValueError(f"Unrecognized or unhandled type of detection: {detection_type}")


if __name__ == "__main__":
    call_gemini()
