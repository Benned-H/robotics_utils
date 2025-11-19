#!/usr/bin/env python3

# /// script
# dependencies = [
#   "google-genai>=0.2.0",
#   "numpy>=1.24.0",
#   "pillow>=10.0.0",
# ]
# requires-python = ">=3.10"
# ///
"""Standalone service for Gemini keypoint detection.

This script runs in a modern Python environment (3.12+) with google-genai installed.
It's designed to be called from older Python environments (like ROS 1 Noetic's Python 3.8)
via subprocess, communicating through JSON I/O.

The dependencies are specified inline using PEP 723 script metadata, so uv will
automatically handle the virtual environment and package installation.

Usage:
    uv run --python 3.12 gemini_keypoint_service.py <image_path> <query1> [query2] [...]

Environment Variables:
    GEMINI_API_KEY or GOOGLE_API_KEY: Required API key for Gemini access
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from robotics_utils.vision import RGBImage
from robotics_utils.vision.vlms.gemini_robotics_er import GeminiRoboticsER


def propagate_error(message: str) -> None:
    """Propagate the error message to stderr so that caller processes can access it, then exit."""
    print(json.dumps({"error": message}), file=sys.stderr)
    sys.exit(1)


def main() -> None:
    """Run Gemini keypoint detection and output results as JSON."""
    if len(sys.argv) < 3:
        propagate_error("Usage: gemini_keypoint_service.py <image_path> <query1> [query2] ...")

    image_path = Path(sys.argv[1])
    queries = sys.argv[2:]  # All remaining args are queries

    if not image_path.exists():
        propagate_error(f"Image file not found: {image_path}")

    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        propagate_error("GEMINI_API_KEY or GOOGLE_API_KEY environment variable not set.")

    try:
        image = RGBImage.from_file(image_path)
        detector = GeminiRoboticsER(api_key=api_key, timeout_s=30.0)
        result = detector.detect_keypoints(image, queries)

        # Convert result to JSON-serializable format
        output = {
            "success": True,
            "detections": [
                {
                    "query": det.query,
                    "keypoint": {"x": det.keypoint.x, "y": det.keypoint.y},
                }
                for det in result.detections
            ],
        }

        # Output JSON to stdout
        print(json.dumps(output))

    except Exception as e:  # On error, output error JSON to stderr
        propagate_error(f"{type(e).__name__}: {e!s}")


if __name__ == "__main__":
    main()
