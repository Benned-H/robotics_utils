"""Define utility functions for working with file paths."""

from __future__ import annotations

import re
from pathlib import Path


def make_unique_path(filepath: Path) -> Path:
    """Return a unique filepath by incrementing a numeric suffix if the path exists.

    If the given filepath does not exist, return it unchanged. Otherwise, extract the
    numeric suffix from the stem (e.g., "image12" -> 12), increment it, and check
    again until a non-existent path is found.

    Examples:
        - "image12.jpg" exists -> return "image13.jpg" (if it doesn't exist)
        - "photo.png" exists -> return "photo1.png" (if it doesn't exist)
        - "data5.csv" doesn't exist -> return "data5.csv"

    :param filepath: Requested filepath to make unique
    :return: A unique filepath that does not currently exist

    """
    if not filepath.exists():
        return filepath

    stem = filepath.stem
    suffix = filepath.suffix
    parent = filepath.parent

    # Extract trailing digits from stem (e.g., "image12" -> ("image", 12))
    match = re.match(r"^(.*?)(\d+)$", stem)
    if match:
        stem_prefix = match.group(1)
        number = int(match.group(2))
    else:
        stem_prefix = stem
        number = 0

    while True:  # Increment until we find a non-existent path
        number += 1
        new_path = parent / f"{stem_prefix}{number}{suffix}"
        if not new_path.exists():
            return new_path
