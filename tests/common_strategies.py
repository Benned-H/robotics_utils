"""Define strategies for generating common representations for property-based testing."""

from __future__ import annotations

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
