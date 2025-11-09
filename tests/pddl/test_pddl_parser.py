"""Unit tests for the PDDLParser class."""

from robotics_utils.abstractions.pddl import PDDLParser, PDDLTokenType


def test_parse_typed_list_of_names(typed_list_of_names: str) -> None:
    """Verify that the PDDLParser class can parse an example typed list."""
    # Arrange - The PDDL example is provided by the test fixture
    parser = PDDLParser(typed_list_of_names)

    # Act - Match a typed list of names
    result = parser.typed_list(token_type=PDDLTokenType.NAME)

    # Assert - Verify that the types were parsed correctly
    result_token_names = [t.value for t in result.tokens]

    assert result_token_names == ["integer", "float", "physob"]
    assert result.pddl_types == ["number", "number", "object"]
