"""Demo script that uses SAM 3 to segment objects in an image."""

from pathlib import Path

import click

from robotics_utils.io.repls import OpenVocabVisionREPL
from robotics_utils.vision.vlms import SAM3, ObjectSegmentations
from robotics_utils.visualization import display_in_window


def display_segmentations(segmentations: ObjectSegmentations) -> None:
    """Display the given segmentation result."""
    display_in_window(segmentations, "Object Instance Segmentations")


@click.command()
@click.argument("image_path", type=click.Path(exists=True, path_type=Path))
def segmentation(image_path: Path) -> None:
    """Run object segmentation in an interactive loop."""
    segmenter = SAM3()

    repl: OpenVocabVisionREPL[ObjectSegmentations] = OpenVocabVisionREPL(
        image_path,
        process_func=segmenter.segment,
        display_func=display_segmentations,
        model_type="object segmenter",
    )
    repl.loop()


if __name__ == "__main__":
    segmentation()
