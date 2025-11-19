#!/usr/bin/env python3

# /// script
# dependencies = [
#   "robotics_utils[gemini] @ file:///${ROBOTICS_UTILS_ROOT}",
# ]
# requires-python = ">=3.10" # Needed to support the latest version of google-genai
# ///

"""Standalone service for Gemini keypoint detection.

This script runs in a modern Python environment (3.10+) with google-genai installed.
It's designed to be called from older Python environments (like ROS 1 Noetic's Python 3.8)
via subprocess, communicating through JSON I/O.

The dependencies are specified inline using PEP 723 script metadata, so uv will
automatically handle the virtual environment and package installation.

Usage:
    uv run gemini_keypoint_service.py <image_path> <query1> [query2] [...]

Environment Variables:
    GEMINI_API_KEY or GOOGLE_API_KEY: Required API key for Gemini access
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from robotics_utils.io import error_console
from robotics_utils.vision import RGBImage
from robotics_utils.vision.vlms.gemini import GeminiRoboticsER


def propagate_error(message: str) -> None:
    """Propagate the error message as JSON to the parent process, then exit."""
    error_output = {"success": False, "error": message}
    error_console.print(json.dumps(error_output))
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
        detections = detector.detect_keypoints(image, queries)

        json_data = detections.to_json()
        json_output = {"success": True, "output": json_data}

        # Ensure that the JSON is the last line of stdout (for parent process to parse)
        dumped_json = json.dumps(json_output).replace("\n", "")
        print(dumped_json)  # noqa: T201

    except Exception as e:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        if exc_tb is None:
            file_line = ""
        else:
            exc_file = Path(exc_tb.tb_frame.f_code.co_filename).name
            exc_line = exc_tb.tb_lineno
            file_line = f" ({exc_file}, line {exc_line})"

        propagate_error(f"{type(e).__name__}: {e!s}{file_line}")


if __name__ == "__main__":
    main()
