"""Demo script that detects object bounding boxes in an image."""

from __future__ import annotations

from pathlib import Path

import click

from robotics_utils.io.repls import OpenVocabVisionREPL
from robotics_utils.vision import ImagePatch, RGBImage
from robotics_utils.vision.vlms import DetectedBoundingBoxes, OwlViTBoundingBoxDetector
from robotics_utils.vision.vlms.gemini import GeminiRoboticsER
from robotics_utils.visualization import display_in_window


def display_detected_bounding_boxes(boxes: DetectedBoundingBoxes) -> None:
    """Display the given bounding box detections."""
    display_in_window(boxes, "Object Detections")

    num_detections = len(boxes.detections)
    if click.confirm("Display cropped images for each detection?"):
        for i, d in enumerate(boxes.detections):
            cropped = ImagePatch.scaled_crop(boxes.image, d.bounding_box, scaling=1.2)
            display_in_window(cropped.patch, f"Detection {i + 1}/{num_detections}: '{d.query}'")

    if click.confirm("Output cropped images to file?"):
        output_dir = click.prompt(
            "Enter output directory for cropped images",
            type=click.Path(writable=True, file_okay=False, path_type=Path),
        )

        for i, d in enumerate(boxes.detections):
            cropped = ImagePatch[RGBImage].scaled_crop(boxes.image, d.bounding_box, scaling=1.2)
            cropped.patch.to_file(output_dir / f"{d.query}_{i}.png")


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

    repl: OpenVocabVisionREPL[DetectedBoundingBoxes] = OpenVocabVisionREPL(
        image_path,
        process_func=detector.detect_bounding_boxes,
        display_func=display_detected_bounding_boxes,
        model_type="bounding box detector",
    )
    repl.loop()


if __name__ == "__main__":
    object_detection()
