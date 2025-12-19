"""Implement a scanner for the Planning Domain Definition Language (PDDL).

Reference: PDDL - The Planning Domain Definition Language (Version 1.2) (Ghallab et al., 1998)
"""

import sys

if sys.version_info < (3, 10):
    raise RuntimeError("This module requires Python 3.10 or higher.")

import re
from dataclasses import dataclass
from enum import StrEnum
from typing import Generator

PDDL_NAME_REGEX = r"[a-zA-Z]{1}[a-zA-Z0-9\-_]*"
"""Names in PDDL begin with a letter and contain only letters, digits, hyphens, and underscores."""


class PDDLTokenType(StrEnum):
    """Enumeration of token types when parsing PDDL."""

    NAME = PDDL_NAME_REGEX
    """Name of a PDDL domain, type, predicate, operator, etc."""

    VARIABLE = r"\?" + PDDL_NAME_REGEX
    """Name of a PDDL variable."""

    KEYWORD = r":" + PDDL_NAME_REGEX
    """A PDDL keyword starts with a colon."""

    MINUS = r"-"
    """Separates PDDL entities from their types in typed lists."""

    OPEN_PAREN = r"\("
    """An open parenthesis."""

    CLOSE_PAREN = r"\)"
    """A close parenthesis."""

    COMMENT = r";[^\n]*"
    """Comments in PDDL begin with a semicolon and end with the next newline."""

    NEWLINE = r"\n"

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

    type_: PDDLTokenType
    value: str
    line: int
    column: int


PDDL_REQ_FLAGS = {
    ":strips": "Basic STRIPS-style adds and deletes",
    ":typing": "Allow type names in declarations of variables",
    ":disjunctive-preconditions": "Allow `or` in goal descriptions",
    ":equality": "Support `=` as built-in predicate",
    ":existential-preconditions": "Allow `exists` in goal descriptions",
    ":universal-preconditions": "Allow `forall` in goal descriptions",
    ":quantified-preconditions": "Allow existential and universal preconditions",
    ":conditional-effects": "Allow `when` in action effects",
    ":adl": (
        "Support :strips + :typing + :disjunctive-preconditions + "
        ":equality + :quantified-preconditions + :conditional-effects"
    ),
}
"""Definitions for supported PDDL requirements flags.

Reference: Section 15 ("Current Requirement Flags") of Ghallab et al. (1998).
"""  # TODO: Support these!


class PDDLScanner:
    """A scanner for a subset of the Planning Domain Definition Language (PDDL)."""

    def __init__(self) -> None:
        """Initialize regular expressions for scanning tokens of PDDL.

        Reference: https://docs.python.org/3/library/re.html#writing-a-tokenizer
        """
        self.token_regex = "|".join(tt.named_group_regex for tt in PDDLTokenType)

        self.keywords = {
            ":requirements",
            ":types",
            ":predicates",
            ":action",
            ":precondition",
            ":effect",
        }

    def tokenize(self, string: str) -> Generator[PDDLToken]:
        """Tokenize a string of PDDL into an iterator over tokens.

        :param string: String containing PDDL to be tokenized
        :yield: Iterator over PDDL tokens in the string
        """
        line_num = 1
        line_start = 0
        for mo in re.finditer(self.token_regex, string):
            if mo.lastgroup is None:
                raise RuntimeError(f"Failed to tokenize string into PDDL:\n{string}")

            token_type: PDDLTokenType = getattr(PDDLTokenType, mo.lastgroup)
            value = mo.group()
            column = mo.start() - line_start

            match token_type:
                case PDDLTokenType.KEYWORD:
                    if value not in self.keywords:
                        raise RuntimeError(f"Unknown PDDL keyword: '{value}'.")

                case PDDLTokenType.MISMATCH:
                    raise RuntimeError(f"Cannot tokenize '{value}' on line {line_num}.")

                case PDDLTokenType.COMMENT | PDDLTokenType.SKIP:
                    continue  # Skip comments and whitespace

                case PDDLTokenType.NEWLINE:
                    line_start = mo.end()
                    line_num += 1
                    continue

                case _:
                    pass

            yield PDDLToken(token_type, value, line_num, column)
