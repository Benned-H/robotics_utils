"""Unit tests for utility functions that process strings."""

from __future__ import annotations

import pytest
from hypothesis import given

from robotics_utils.io.string_utils import is_camel_case, is_snake_case

from ..common_strategies import camel_case_strings, snake_case_strings


@given(camel_case_strings(), snake_case_strings())
def test_is_camel_case(camel_case_str: str, snake_case_str: str) -> None:
    """Verify that is_camel_case() correctly classifies CamelCase and snake_case strings."""
    # Arrange/Act - Given a CamelCase and a snake_case string, evaluate if they're camel-case
    camel_is_camel = is_camel_case(camel_case_str)
    snake_is_camel = is_camel_case(snake_case_str)

    # Assert - Expect that CamelCase and snake_case strings are correctly classified
    assert camel_is_camel
    assert not snake_is_camel


@given(camel_case_strings(), snake_case_strings())
def test_is_snake_case(camel_case_str: str, snake_case_str: str) -> None:
    """Verify that is_snake_case() correctly classifies CamelCase and snake_case strings."""
    # Arrange/Act - Given a CamelCase and a snake_case string, evaluate if they're snake-case
    camel_is_snake = is_snake_case(camel_case_str)
    snake_is_snake = is_snake_case(snake_case_str)

    # Assert - Expect that CamelCase and snake_case strings are correctly classified
    assert not camel_is_snake
    assert snake_is_snake


@pytest.fixture
def snake_case_examples() -> list[str]:
    """Create a list of example snake_case strings."""
    return ["str", "snake_case_strings", "numbers1_2_3", "apple1"]


@pytest.fixture
def camel_case_examples() -> list[str]:
    """Create a list of example CamelCase strings."""
    return ["CamelCaseStrings", "ChatGPT", "Apple1", "PredicateInstance3A"]


@pytest.fixture
def neither_examples() -> list[str]:
    """Create a list of example strings that are neither snake_case nor CamelCase."""
    return ["Hello World", "lower case", "aBCD", "", "super-duper", "=", "Tricky1case", "1"]


def test_is_camel_case_on_examples(
    snake_case_examples: list[str],
    camel_case_examples: list[str],
) -> None:
    """Verify that is_camel_case() correctly classifies a selection of example strings."""
    snake_results = [is_camel_case(s) for s in snake_case_examples]
    camel_results = [is_camel_case(s) for s in camel_case_examples]

    for was_camel_case, snake_str in zip(snake_results, snake_case_examples, strict=True):
        assert not was_camel_case, f"String '{snake_str}' is not CamelCase."

    for was_camel_case, camel_str in zip(camel_results, camel_case_examples, strict=True):
        assert was_camel_case, f"String '{camel_str}' is CamelCase."


def test_is_snake_case_on_examples(
    snake_case_examples: list[str],
    camel_case_examples: list[str],
) -> None:
    """Verify that is_snake_case() correctly classifies a selection of example strings."""
    snake_results = [is_snake_case(s) for s in snake_case_examples]
    camel_results = [is_snake_case(s) for s in camel_case_examples]

    for was_snake_case, snake_str in zip(snake_results, snake_case_examples, strict=True):
        assert was_snake_case, f"String '{snake_str}' is snake_case."

    for was_snake_case, camel_str in zip(camel_results, camel_case_examples, strict=True):
        assert not was_snake_case, f"String '{camel_str}' is not snake_case."


def test_neither_snake_nor_camel_case_examples(neither_examples: list[str]) -> None:
    """Verify that example strings that aren't snake_case or CamelCase are correctly classified."""
    were_snake_case = [is_snake_case(s) for s in neither_examples]
    were_camel_case = [is_camel_case(s) for s in neither_examples]

    for was_snake_case, string in zip(were_snake_case, neither_examples, strict=True):
        assert not was_snake_case, f"String '{string}' is not snake_case."

    for was_camel_case, string in zip(were_camel_case, neither_examples, strict=True):
        assert not was_camel_case, f"String '{string}' is not CamelCase."
