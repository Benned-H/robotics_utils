"""Example usage of the BoundingBoxDetector class."""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from robotics_utils.io.repls import ObjectDetectionREPL
from robotics_utils.perception.vision.vlms import (
    ObjectBoundingBoxes,
    OwlViTBoundingBoxDetector,
)
from robotics_utils.visualization import display_in_window


def display_detected_bounding_boxes(console: Console, boxes: ObjectBoundingBoxes) -> None:
    """Display the given bounding box detections using the given console."""
    display_in_window(boxes, "Object Detections")

    if click.confirm("Display cropped images for each detection?"):
        for i, d in enumerate(boxes.detections):
            cropped = d.bounding_box.crop(boxes.image, scale_ratio=1.2)
            display_in_window(cropped, f"Detection {i}/{len(boxes.detections)}: '{d.query}'")


@click.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
@click.option("--model-name", "model_name", help="Name of the OWL-ViT model to load")
def object_detection(image_path: Path, model_name: str | None) -> None:
    """Run object detection in an interactive loop."""
    console = Console()
    detector = (
        OwlViTBoundingBoxDetector() if model_name is None else OwlViTBoundingBoxDetector(model_name)
    )

    repl: ObjectDetectionREPL[ObjectBoundingBoxes] = ObjectDetectionREPL(
        console,
        image_path,
        detect_func=detector.detect,
        display_func=display_detected_bounding_boxes,
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
