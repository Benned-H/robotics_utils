"""Define a class to parse PDDL domain and problem files."""

from typing import Any

from robotics_utils.task_planning.pddl_scanner import PDDLScanner, PDDLTokenType


class PDDLParser:
    """A parser for a subset of the Planning Domain Definition Language (PDDL)."""

    def __init__(self, pddl_string: str) -> None:
        """Initialize the PDDL parser for the given string of PDDL.

        :param pddl_string: String of PDDL to be parsed
        """
        self.scanner = PDDLScanner()
        self.remaining_tokens = self.scanner.tokenize(pddl_string)
        self.input_token = next(self.remaining_tokens)

    def match(self, expected_type: PDDLTokenType, value: str | None = None) -> None:
        """Consume a token of the given type from the scanner.

        :param expected_type: Expected type of the next PDDL token
        :param value: Exact expected value of the next token (optional; defaults to None)
        """
        if value is not None and value != self.input_token:
            raise RuntimeError(f"Expected '{value}' as next token but found '{self.input_token}'")

        if self.input_token.type is expected_type:  # Consume the input token
            self.input_token = next(self.remaining_tokens)
        else:
            raise RuntimeError(
                f"Expected PDDL token of type {expected_type.name} but found {self.input_token}",
            )
