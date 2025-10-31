"""Unit tests for utility functions that process strings."""

from __future__ import annotations

import pytest
from hypothesis import given

from robotics_utils.io.string_utils import is_pascal_case, is_snake_case

from .strategies.common_strategies import pascal_case_strings, snake_case_strings


@given(pascal_case_strings(), snake_case_strings())
def test_is_pascal_case(pascal_case_str: str, snake_case_str: str) -> None:
    """Verify that is_pascal_case() correctly classifies PascalCase and snake_case strings."""
    # Arrange/Act - Given a PascalCase and a snake_case string, evaluate if they're Pascal case
    pascal_is_pascal = is_pascal_case(pascal_case_str)
    snake_is_pascal = is_pascal_case(snake_case_str)

    # Assert - Expect that PascalCase and snake_case strings are correctly classified
    assert pascal_is_pascal
    assert not snake_is_pascal


@given(pascal_case_strings(), snake_case_strings())
def test_is_snake_case(pascal_case_str: str, snake_case_str: str) -> None:
    """Verify that is_snake_case() correctly classifies PascalCase and snake_case strings."""
    # Arrange/Act - Given a PascalCase and a snake_case string, evaluate if they're snake case
    pascal_is_snake = is_snake_case(pascal_case_str)
    snake_is_snake = is_snake_case(snake_case_str)

    # Assert - Expect that PascalCase and snake_case strings are correctly classified
    assert not pascal_is_snake
    assert snake_is_snake


@pytest.fixture
def snake_case_examples() -> list[str]:
    """Create a list of example snake_case strings."""
    return ["str", "snake_case_strings", "numbers1_2_3", "apple1"]


@pytest.fixture
def pascal_case_examples() -> list[str]:
    """Create a list of example PascalCase strings."""
    return ["PascalCaseStrings", "ChatGPT", "Apple1", "PredicateInstance3A"]


@pytest.fixture
def neither_examples() -> list[str]:
    """Create a list of example strings that are neither snake_case nor PascalCase."""
    return ["Hello World", "lower case", "aBCD", "", "super-duper", "=", "Tricky1case", "1"]


def test_is_pascal_case_on_examples(
    snake_case_examples: list[str],
    pascal_case_examples: list[str],
) -> None:
    """Verify that is_pascal_case() correctly classifies a selection of example strings."""
    snake_results = [is_pascal_case(s) for s in snake_case_examples]
    pascal_results = [is_pascal_case(s) for s in pascal_case_examples]

    for was_pascal_case, snake_str in zip(snake_results, snake_case_examples, strict=True):
        assert not was_pascal_case, f"String '{snake_str}' is not PascalCase."

    for was_pascal_case, pascal_str in zip(pascal_results, pascal_case_examples, strict=True):
        assert was_pascal_case, f"String '{pascal_str}' is PascalCase."


def test_is_snake_case_on_examples(
    snake_case_examples: list[str],
    pascal_case_examples: list[str],
) -> None:
    """Verify that is_snake_case() correctly classifies a selection of example strings."""
    snake_results = [is_snake_case(s) for s in snake_case_examples]
    pascal_results = [is_snake_case(s) for s in pascal_case_examples]

    for was_snake_case, snake_str in zip(snake_results, snake_case_examples, strict=True):
        assert was_snake_case, f"String '{snake_str}' is snake_case."

    for was_snake_case, pascal_str in zip(pascal_results, pascal_case_examples, strict=True):
        assert not was_snake_case, f"String '{pascal_str}' is not snake_case."


def test_neither_snake_nor_pascal_case_examples(neither_examples: list[str]) -> None:
    """Verify that example strings that aren't snake_case or PascalCase are correctly classified."""
    were_snake_case = [is_snake_case(s) for s in neither_examples]
    were_pascal_case = [is_pascal_case(s) for s in neither_examples]

    for was_snake_case, string in zip(were_snake_case, neither_examples, strict=True):
        assert not was_snake_case, f"String '{string}' is not snake_case."

    for was_pascal_case, string in zip(were_pascal_case, neither_examples, strict=True):
        assert not was_pascal_case, f"String '{string}' is not PascalCase."
