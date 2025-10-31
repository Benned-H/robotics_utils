"""Demo script that detects object bounding boxes in an image."""

from __future__ import annotations

from pathlib import Path

import click

from robotics_utils.io.repls import ObjectDetectionREPL
from robotics_utils.perception.vision import RGBImage
from robotics_utils.perception.vision.vlms import (
    ObjectBoundingBoxes,
    OwlViTBoundingBoxDetector,
)
from robotics_utils.perception.vision.vlms.gemini_robotics_er import GeminiRoboticsER
from robotics_utils.visualization import display_in_window


def display_detected_bounding_boxes(boxes: ObjectBoundingBoxes) -> None:
    """Display the given bounding box detections."""
    display_in_window(boxes, "Object Detections")

    if click.confirm("Display cropped images for each detection?"):
        for i, d in enumerate(boxes.detections):
            cropped = d.bounding_box.crop(boxes.image, scale_ratio=1.2)
            display_in_window(cropped, f"Detection {i}/{len(boxes.detections)}: '{d.query}'")

            RGBImage(cropped.data).to_file(f"{d.query}{i}.png")


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["owl-vit", "gemini"], case_sensitive=False),
    default="owl-vit",
    help="Specifies which bounding box detection model to use",
)
@click.option("--api-key")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def object_detection(image_path: Path, backend: str, api_key: str | None) -> None:
    """Run object detection in an interactive loop."""
    if backend == "owl-vit":
        detector = OwlViTBoundingBoxDetector()
    elif backend == "gemini":
        assert api_key is not None, "Cannot use Gemini Robotics ER 1.5 without a Google API key."
        detector = GeminiRoboticsER(api_key)
    else:
        raise ValueError(f"Unrecognized backend: {backend}")

    repl: ObjectDetectionREPL[ObjectBoundingBoxes] = ObjectDetectionREPL(
        image_path,
        detect_func=detector.detect_bounding_boxes,
        display_func=display_detected_bounding_boxes,
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
