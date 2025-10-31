"""Demo script that detects object bounding boxes in an image."""

from __future__ import annotations

from pathlib import Path

import click

from robotics_utils.io.repls import ObjectDetectionREPL
from robotics_utils.vision import RGBImage
from robotics_utils.vision.vlms import (
    DetectedBoundingBoxes,
    OwlViTBoundingBoxDetector,
)
from robotics_utils.vision.vlms.gemini_robotics_er import GeminiRoboticsER
from robotics_utils.visualization import display_in_window


def display_detected_bounding_boxes(boxes: DetectedBoundingBoxes) -> None:
    """Display the given bounding box detections."""
    display_in_window(boxes, "Object Detections")

    if click.confirm("Display cropped images for each detection?"):
        for i, d in enumerate(boxes.detections):
            cropped = d.bounding_box.crop(boxes.image, scale_ratio=1.2)
            display_in_window(cropped, f"Detection {i}/{len(boxes.detections)}: '{d.query}'")

    if click.confirm("Output cropped images to file?"):
        output_dir = click.prompt(
            "Enter output directory for cropped images",
            type=click.Path(writable=True, file_okay=False, path_type=Path),
        )

        for i, d in enumerate(boxes.detections):
            cropped = d.bounding_box.crop(boxes.image, scale_ratio=1.2)
            RGBImage(cropped.data).to_file(output_dir / f"{d.query}_{i}.png")


@click.command()
@click.option(
    "--backend",
    type=click.Choice(["owl-vit", "gemini"], case_sensitive=False),
    default="owl-vit",
    help="Specifies which bounding box detection model to use",
)
@click.option("--api-key")
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def object_detection(backend: str, api_key: str | None, image_path: Path) -> None:
    """Run object detection in an interactive loop."""
    if backend == "owl-vit":
        detector = OwlViTBoundingBoxDetector()
    elif backend == "gemini":
        assert api_key is not None, "Cannot use Gemini Robotics ER 1.5 without a Google API key."
        detector = GeminiRoboticsER(api_key)
    else:
        raise ValueError(f"Unrecognized backend: {backend}")

    repl: ObjectDetectionREPL[DetectedBoundingBoxes] = ObjectDetectionREPL(
        image_path,
        detect_func=detector.detect_bounding_boxes,
        display_func=display_detected_bounding_boxes,
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
