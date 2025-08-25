"""Define utility functions for processing strings.

Definitions:

    PascalCase - A string is Pascal case if it's fully alphanumeric, begins with a letter, and
        all of its consecutive sequences of letters begin with an uppercase letter.

    snake_case - A string is snake case if it contains at least one letter, is lowercase, and
        contains only letters, numbers, and underscores.

    No string can be both PascalCase and snake_case.
"""

import re


def pascal_to_snake(string: str) -> str:
    """Convert a PascalCase string to snake_case."""
    # Insert an underscore before any uppercase letter that follows a lowercase letter or digit
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", string).lower()


def snake_to_pascal(string: str) -> str:
    """Convert a snake_case string to PascalCase."""
    chunks = string.split("_")
    return "".join(word.capitalize() for word in chunks)


def is_snake_case(string: str) -> bool:
    """Check whether the given string is snake_case."""
    if string != string.lower():
        return False

    if not all((c.isalnum() or c == "_") for c in string):
        return False

    return bool(string) and not string[0].isnumeric()


def is_pascal_case(string: str) -> bool:
    """Check whether the given string is PascalCase."""
    if not string or not string.isalnum():
        return False

    if string[0].lower() == string[0]:
        return False  # Any PascalCase string must begin with an uppercase letter

    for c_prev, c_next in zip(string[:-1], string[1:], strict=True):
        # Ensure that a lowercase letter never follows a digit
        if c_prev.isnumeric() and c_next.isalpha() and c_next.lower() == c_next:
            return False

    return True
