"""Define classes to tokenize PDDL domain and problem files."""

import re
from collections.abc import Generator
from dataclasses import dataclass
from enum import StrEnum

PDDL_NAME_REGEX = r"[a-zA-Z]{1}[a-zA-Z0-9\-_]*"


class PDDLTokenType(StrEnum):
    """Enumeration of token types for parsing PDDL."""

    NAME = PDDL_NAME_REGEX
    """Name of a PDDL domain, type, predicate, operator, etc."""

    VARIABLE = r"\?" + PDDL_NAME_REGEX
    """Name of a PDDL variable (e.g., '?x')."""

    KEYWORD = r":" + PDDL_NAME_REGEX
    """A PDDL keyword marked by an initial colon (e.g., ':types')."""

    MINUS = r"-"
    """Separates PDDL variable names from their types."""

    OPEN_PAREN = r"\("
    """An open parenthesis."""

    CLOSE_PAREN = r"\)"
    """A close parenthesis."""

    COMMENT = r";[^\n]*"
    """A PDDL comment lasts from a semicolon to the end of the line."""

    NEWLINE = r"\n"
    """A newline marks the end of a line."""

    SKIP = r"[ \t]+"
    """Whitespace to be ignored."""

    MISMATCH = r"."
    """Any other character is a mismatch."""

    @property
    def named_group_regex(self) -> str:
        """Retrieve the named group regular expression for the token type."""
        return f"(?P<{self.name}>{self.value})"


@dataclass(frozen=True)
class PDDLToken:
    """A token scanned from a string of PDDL.

    Reference: https://docs.python.org/3/library/re.html#writing-a-tokenizer
    """

    type: PDDLTokenType
    value: str
    line: int
    column: int


class PDDLScanner:
    """A scanner for a subset of the Planning Domain Definition Language (PDDL).

    References:
        Chapter 2.2 (pg. 46) of "Programming Language Pragmatics" by Michael L. Scott.

        "Writing a Tokenizer" (https://docs.python.org/3/library/re.html#writing-a-tokenizer).

    """

    def __init__(self) -> None:
        """Initialize regular expressions for scanning tokens of PDDL."""
        self.keywords = {
            ":requirements",
            ":types",
            ":predicates",
            ":action",
            ":parameters",
            ":precondition",
            ":effect",
        }

        self.token_regex = "|".join(tt.named_group_regex for tt in PDDLTokenType)

    def tokenize(self, pddl_string: str) -> Generator[PDDLToken]:
        """Tokenize a string of PDDL into an iterator over tokens.

        :param pddl_string: String of PDDL to be tokenized
        :yield: Iterator over PDDL tokens in the string
        """
        line_num = 1
        line_start = 0
        for match in re.finditer(self.token_regex, pddl_string):
            if match.lastgroup is None:
                raise RuntimeError(f"Failed to parse PDDL string:\n{pddl_string}")

            token_type: PDDLTokenType = getattr(PDDLTokenType, match.lastgroup)
            value = match.group()
            column = match.start() - line_start

            match token_type:
                case PDDLTokenType.KEYWORD:
                    if value not in self.keywords:
                        raise RuntimeError(f"Unknown PDDL keyword: {value}")

                case PDDLTokenType.COMMENT | PDDLTokenType.SKIP:
                    continue  # Skip comments and whitespace

                case PDDLTokenType.NEWLINE:
                    line_start = match.end()
                    line_num += 1
                    continue

                case PDDLTokenType.MISMATCH:
                    raise RuntimeError(f"{value!r} unexpected on line {line_num}.")

                case _:
                    pass

            yield PDDLToken(token_type, value, line_num, column)
