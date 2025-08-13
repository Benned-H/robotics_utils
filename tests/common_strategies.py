"""Define strategies for generating common representations for property-based testing."""

from __future__ import annotations

import re
from pathlib import Path

import hypothesis.strategies as st


@st.composite
def real_ranges(draw: st.DrawFn, finite: bool) -> tuple[float, float]:
    """Generate random [a,b] ranges of real numbers (i.e., floats)."""
    a = draw(st.floats(allow_infinity=(not finite), allow_nan=False))
    b = draw(st.floats(allow_infinity=(not finite), allow_nan=False))
    return (min(a, b), max(a, b))


@st.composite
def integer_ranges(draw: st.DrawFn, min_value: int, max_value: int) -> tuple[int, int]:
    """Generate random [a,b] ranges of integers."""
    a = draw(st.integers(min_value=min_value, max_value=max_value))
    b = draw(st.integers(min_value=min_value, max_value=max_value))
    return (min(a, b), max(a, b))


def test_data_path() -> Path:
    """Retrieve the path to the `test_data` folder."""
    path = Path(__file__).parent / "test_data"
    assert path.exists()
    return path


lowercase_ascii_regex = re.compile(r"\A[a-z]\Z", flags=re.ASCII)
uppercase_ascii_regex = re.compile(r"\A[A-Z]\Z", flags=re.ASCII)
digits_regex = re.compile(r"\A\d\Z", flags=re.ASCII)


@st.composite
def lowercase_letters(draw: st.DrawFn) -> str:
    """Generate random lowercase letters."""
    return draw(st.from_regex(lowercase_ascii_regex, fullmatch=True))


@st.composite
def uppercase_letters(draw: st.DrawFn) -> str:
    """Generate random uppercase letters."""
    return draw(st.from_regex(uppercase_ascii_regex, fullmatch=True))


@st.composite
def letters(draw: st.DrawFn) -> str:
    """Generate random letters (uppercase or lowercase)."""
    return draw(st.one_of(lowercase_letters(), uppercase_letters()))


@st.composite
def digits(draw: st.DrawFn) -> str:
    """Generate random digit characters."""
    return draw(st.from_regex(digits_regex, fullmatch=True))


@st.composite
def snake_case_strings(draw: st.DrawFn) -> str:
    """Generate random snake_case strings."""
    lowercase_or_underscore = st.one_of(st.just("_"), lowercase_letters())
    first_char = draw(lowercase_or_underscore)

    if first_char == "_":
        first_char += draw(lowercase_letters())  # Ensure there's at least one letter

    snake_case_chars = st.one_of(digits(), lowercase_or_underscore)
    rest_of_string = draw(st.text(alphabet=snake_case_chars))

    return first_char + rest_of_string


@st.composite
def camel_case_strings(draw: st.DrawFn) -> str:
    """Generate random CamelCase strings."""
    first_char = draw(uppercase_letters())
    rest_of_string = draw(st.text(st.one_of(digits(), letters())))

    output_string = first_char
    for c in rest_of_string:
        if c.upper() == c:
            output_string += c
            continue

        # Otherwise, we have a lowercase letter
        prev_char = output_string[-1]
        if prev_char.isnumeric():
            output_string += c.upper()  # CamelCase: Letters after digits should be uppercase
        else:
            output_string += c

    return output_string
