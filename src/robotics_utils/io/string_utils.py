"""Define utility functions for processing strings."""

import re


def camel_to_snake(string: str) -> str:
    """Convert a CamelCase string to snake_case."""
    # Insert an underscore before any uppercase letter that follows a lowercase letter or digit
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string).lower()


def snake_to_camel(string: str) -> str:
    """Convert a snake_case string to CamelCase."""
    chunks = string.split("_")
    return "".join(word.capitalize() for word in chunks)


def is_snakecase(string: str) -> bool:
    """Check whether the given string is snake_case."""
    chunks = string.split()
    return len(chunks) == 1 and chunks[0] == string.lower()


def is_camelcase(string: str) -> bool:
    """Check whether the given string is CamelCase."""
    return string.lower() != string and ("_" not in string)
