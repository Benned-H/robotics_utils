"""Define a class to parse PDDL domain and problem files."""

from typing import Any

from robotics_utils.task_planning.pddl_scanner import PDDLScanner, PDDLToken, PDDLTokenType
from robotics_utils.task_planning.predicates import Predicate

PDDLDomain = Any  # TODO: Create real types
Types = Any
Predicates = Any


class PDDLParser:
    """A parser for a subset of the Planning Domain Definition Language (PDDL)."""

    def __init__(self, pddl_string: str) -> None:
        """Initialize the PDDL parser for the given string of PDDL.

        :param pddl_string: String of PDDL to be parsed
        """
        self.scanner = PDDLScanner()
        self.remaining_tokens = self.scanner.tokenize(pddl_string)
        self.input_token = next(self.remaining_tokens)

    def match(self, expected_type: PDDLTokenType, value: str | None = None) -> PDDLToken:
        """Consume a token of the given type from the scanner.

        :param expected_type: Expected type of the next PDDL token
        :param value: Exact expected value of the next token (optional; defaults to None)
        :return: Matched PDDL token
        """
        if value is not None and value != self.input_token:
            raise RuntimeError(f"Expected '{value}' as next token but found '{self.input_token}'")

        if self.input_token.type is expected_type:  # Consume the input token
            matched_token = self.input_token
            self.input_token = next(self.remaining_tokens)
            return matched_token

        raise RuntimeError(
            f"Expected PDDL token of type {expected_type.name} but found {self.input_token}",
        )

    def domain(self) -> PDDLDomain:
        """Parse a PDDL domain from the input tokens.

        :return: Constructed PDDLDomain instance
        """
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, value="domain")
        domain_name = self.match(PDDLTokenType.NAME)
        self.match(PDDLTokenType.CLOSE_PAREN)

        # Permit :requirements, :types, :predicates, and :action in any order
        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            self.match(PDDLTokenType.OPEN_PAREN)
            keyword = self.match(PDDLTokenType.KEYWORD)

            match keyword:
                case ":types":
                    self.types()

        self.match(PDDLTokenType.CLOSE_PAREN)

    def types(self) -> Types:
        """Parse PDDL types until a close parenthesis is matched.

        :return: Collection of parsed PDDL types
        """
        # TODO

    def predicates(self) -> set[Predicate]:
        """Parse PDDL predicates until a close parenthesis is matched.

        :return: Set of parsed predicates
        """
        return set()  # TODO
