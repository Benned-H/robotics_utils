"""Implement a scanner for the Planning Domain Definition Language (PDDL).

Reference: PDDL - The Planning Domain Definition Language (Version 1.2) (Ghallab et al., 1998)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterator

PDDL_NAME_REGEX = r"[a-zA-Z]{1}[a-zA-Z0-9\-_]*"
"""Names in PDDL begin with a letter and contain only letters, digits, hyphens, and underscores."""


class PDDLTokenType(Enum):
    """Enumeration of token types when parsing PDDL."""

    NAME = auto()
    """Name of a PDDL domain, type, predicate, operator, etc."""

    VARIABLE = auto()
    """Name of a PDDL variable."""

    KEYWORD = auto()
    """A PDDL keyword starts with a colon."""

    MINUS = auto()
    """Separates PDDL entities from their types in typed lists."""

    EQUALS = auto()
    """Some PDDL domains include `=` as a built-in predicate."""

    OPEN_PAREN = auto()
    """An open parenthesis."""

    CLOSE_PAREN = auto()
    """A close parenthesis."""

    COMMENT = auto()
    """Comments in PDDL begin with a semicolon and end with the next newline."""

    NEWLINE = auto()

    SKIP = auto()
    """Whitespace to be ignored."""

    MISMATCH = auto()
    """Any other character is a mismatch."""

    NONE = auto()
    """Represents when an input stream of tokens has been exhausted."""


PDDL_TOKEN_TYPE_TO_REGEX = {
    PDDLTokenType.NAME: PDDL_NAME_REGEX,
    PDDLTokenType.VARIABLE: r"\?" + PDDL_NAME_REGEX,
    PDDLTokenType.KEYWORD: r":" + PDDL_NAME_REGEX,
    PDDLTokenType.MINUS: r"-",
    PDDLTokenType.EQUALS: r"=",
    PDDLTokenType.OPEN_PAREN: r"\(",
    PDDLTokenType.CLOSE_PAREN: r"\)",
    PDDLTokenType.COMMENT: r";[^\n]*",
    PDDLTokenType.NEWLINE: r"\n",
    PDDLTokenType.SKIP: r"[ \t]+",
    PDDLTokenType.MISMATCH: r".",
}


def named_group_regex(token_type: PDDLTokenType) -> str:
    """Construct a named group regular expression for a PDDL token type."""
    return f"(?P<{token_type.name}>{PDDL_TOKEN_TYPE_TO_REGEX[token_type]})"


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

PDDL_KEYWORDS = {
    ":requirements",
    ":types",
    ":constants",
    ":predicates",
    ":action",
    ":parameters",
    ":precondition",
    ":effect",
    ":domain",
    ":objects",
    ":init",
    ":goal",
}.union(PDDL_REQ_FLAGS.keys())


class PDDLScanner:
    """A scanner for a subset of the Planning Domain Definition Language (PDDL)."""

    def __init__(self, known_keywords: set[str]) -> None:
        """Initialize regular expressions for scanning tokens of PDDL.

        Reference: https://docs.python.org/3/library/re.html#writing-a-tokenizer

        :param known_keywords: Set of PDDL requirement flags recognized by the scanner
        """
        self.token_regex = "|".join(named_group_regex(tt) for tt in PDDLTokenType)
        self.known_keywords = known_keywords

    def tokenize(self, string: str) -> Iterator[PDDLToken]:
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

            if token_type == PDDLTokenType.KEYWORD:
                if value not in self.known_keywords:
                    raise RuntimeError(f"Unknown PDDL keyword: '{value}'.")

            elif token_type == PDDLTokenType.MISMATCH:
                raise RuntimeError(f"Cannot tokenize '{value}' on line {line_num}.")

            elif token_type in {PDDLTokenType.COMMENT, PDDLTokenType.SKIP}:
                continue  # Skip comments and whitespace

            elif token_type == PDDLTokenType.NEWLINE:
                line_start = mo.end()
                line_num += 1
                continue

            yield PDDLToken(token_type, value, line_num, column)
