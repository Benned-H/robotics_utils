"""Implement a parser for the Planning Domain Definition Language (PDDL).

Reference: PDDL - The Planning Domain Definition Language (Version 1.2) (Ghallab et al., 1998)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any  # TODO: Remove use of Any

from robotics_utils.abstractions.pddl.pddl_scanner import PDDLScanner, PDDLToken, PDDLTokenType

AtomicFormulaT = Any  # TODO: Replace with real type
LiteralT = Any  # TODO: Replace with real type
GoalDescriptionT = Any  # TODO: Replace with real type
PreconditionsT = Any  # TODO: Replace with real type
EffectsT = Any  # TODO: Replace with real type
OperatorT = Any  # TODO: Replace with real type
AtomicFormulaSkeletonT = Any  # TODO: Replace with real type
PDDLDomainT = Any  # TODO: Replace with real type
PDDLProblemT = Any  # TODO: Replace with real type


@dataclass(frozen=True)
class TypedTokens:
    """A list of parsed PDDL tokens corresponding to typed PDDL entities."""

    tokens: list[PDDLToken]
    pddl_types: list[str]

    def __post_init__(self) -> None:
        """Verify that there are an equal number of tokens and PDDL types."""
        if len(self.tokens) != len(self.pddl_types):
            raise RuntimeError(
                f"Found {len(self.tokens)} tokens but {len(self.pddl_types)} PDDL types.",
            )


class PDDLParser:
    """A parser for a subset of the Planning Domain Definition Language (PDDL)."""

    def __init__(self, string: str) -> None:
        """Initialize the PDDL parser for the given string."""
        self.scanner = PDDLScanner()
        self.remaining_tokens = self.scanner.tokenize(string)
        self.input_token: PDDLToken = next(self.remaining_tokens)
        """Once the input token type is `PDDLTokenType.NONE`, all tokens have been consumed."""

    def match(self, token_type: PDDLTokenType, value: str | None = None) -> PDDLToken:
        """Consume a token of the given type from the scanner.

        :param token_type: Expected type of the next PDDL token
        :param value: Expected string value of the next token (optional; defaults to None)
        :return: PDDL token consumed from the scanner
        """
        if self.input_token.type_ == PDDLTokenType.NONE:
            raise RuntimeError("Cannot match another token; input stream has been exhausted.")

        if value is not None and value != self.input_token.value:
            raise RuntimeError(
                f"Expected '{value}' as next token but found '{self.input_token.value}'.",
            )

        if self.input_token.type_ == token_type:  # Consume the input token
            matched_token = self.input_token
            try:
                self.input_token = next(self.remaining_tokens)
            except StopIteration:
                self.input_token = PDDLToken(PDDLTokenType.NONE, value="", line=-1, column=-1)

            return matched_token

        raise RuntimeError(
            f"Expected PDDL token type {token_type.name} but found {self.input_token.type_.name}.",
        )

    def atomic_formula(
        self,
        term_type: PDDLTokenType | None = None,
        match_open_paren: bool = False,
    ) -> AtomicFormulaT:
        """Parse a PDDL atomic formula from the input stream of tokens.

        Default behavior: Parse the formula's predicate name through its closing parenthesis.

        :param term_type: Type of PDDL token expected as terms (default None = accept all types)
        :param match_open_paren: Whether to match the formula's open parenthesis (default: False)
        :return: Parsed PDDL atomic formula
        """
        if match_open_paren:
            self.match(PDDLTokenType.OPEN_PAREN)

        name = self.match(PDDLTokenType.NAME)

        terms: list[PDDLToken] = []
        while self.input_token.type_ not in {PDDLTokenType.CLOSE_PAREN, PDDLTokenType.NONE}:
            if term_type is not None and term_type != self.input_token.type_:
                raise RuntimeError(f"Unexpected token type: {self.input_token}")

            next_term = self.match(self.input_token.type_)
            terms.append(next_term)

        self.match(PDDLTokenType.CLOSE_PAREN)
        return AtomicFormulaT(name, terms)

    def literal(
        self,
        term_type: PDDLTokenType | None = None,
        match_open_paren: bool = False,
    ) -> LiteralT:
        """Parse a PDDL literal from the input stream of tokens.

        :param term_type: Type of PDDL token expected as terms (default None = accept all types)
        :param match_open_paren: Whether to match the literal's open parenthesis (default: False)
        :return: Parsed PDDL literal
        """
        if match_open_paren:
            self.match(PDDLTokenType.OPEN_PAREN)

        if self.input_token.type_ != PDDLTokenType.NAME:
            raise RuntimeError(f"Unexpected token type: {self.input_token}")

        if self.input_token.value == "not":  # Negated literal
            self.match(PDDLTokenType.NAME, value="not")
            formula = self.atomic_formula(term_type=term_type, match_open_paren=True)
            self.match(PDDLTokenType.CLOSE_PAREN)  # Close the negation

            return LiteralT(formula=formula, negated=True)

        formula = self.atomic_formula(term_type=term_type)  # Match from predicate name onward
        return LiteralT(formula=formula, negated=False)

    def typed_list(self, token_type: PDDLTokenType) -> TypedTokens:
        """Parse a PDDL-typed list of the given token type.

        This method does not match a following closing parenthesis, if present.

        :param token_type: Type of PDDL token (e.g., `VARIABLE`) being assigned PDDL types
        :return: Collection of parsed tokens and their corresponding PDDL types
        """
        tokens: list[PDDLToken] = []
        types: list[str] = []
        tokens_awaiting_types = 0

        while self.input_token.type_ not in {PDDLTokenType.CLOSE_PAREN, PDDLTokenType.NONE}:
            if self.input_token.type_ == token_type:
                token = self.match(token_type)
                tokens.append(token)
                tokens_awaiting_types += 1
                continue

            if self.input_token.type_ == PDDLTokenType.MINUS:  # Match "-" and the following type
                if not tokens_awaiting_types:
                    raise RuntimeError(f"Unexpected minus in a typed list: {self.input_token}")

                self.match(PDDLTokenType.MINUS)
                parent_type = self.match(PDDLTokenType.NAME).value
                types.extend([parent_type] * tokens_awaiting_types)
                tokens_awaiting_types = 0
                continue

            raise RuntimeError(f"Unexpected token: {self.input_token}")

        if tokens_awaiting_types:
            types.extend(["object"] * tokens_awaiting_types)  # Default parent type in PDDL

        return TypedTokens(tokens, types)

    def atomic_formula_skeleton(self) -> AtomicFormulaSkeletonT:
        """Parse a PDDL atomic formula skeleton from the stream of input tokens."""
        self.match(PDDLTokenType.OPEN_PAREN)
        predicate_name = self.match(PDDLTokenType.NAME)
        variables = self.typed_list(PDDLTokenType.VARIABLE)
        self.match(PDDLTokenType.CLOSE_PAREN)
        return AtomicFormulaSkeletonT(predicate_name, variables)

    def goal_description(self) -> GoalDescriptionT:
        """Parse a PDDL goal description from the input stream of tokens.

        Reference: Section 6 (pg. 8-9) of Ghallab et al., 1998.

        :return: Parsed PDDL goal description
        """
        self.match(PDDLTokenType.OPEN_PAREN)

        if self.input_token.type_ != PDDLTokenType.NAME:
            raise RuntimeError(f"Unexpected token type: {self.input_token}")

        lookahead = self.input_token.value

        if lookahead == "and":
            self.match(PDDLTokenType.NAME, value="and")
            and_goals: list[GoalDescriptionT] = []
            while self.input_token.type_ == PDDLTokenType.OPEN_PAREN:
                and_goals.append(self.goal_description())
            self.match(PDDLTokenType.CLOSE_PAREN)
            return GoalDescriptionT("and", and_goals)

        if lookahead == "or":  # For the :disjunctive-preconditions requirement flag
            self.match(PDDLTokenType.NAME, value="or")
            or_goals: list[GoalDescriptionT] = []
            while self.input_token.type_ == PDDLTokenType.OPEN_PAREN:
                or_goals.append(self.goal_description())
            self.match(PDDLTokenType.CLOSE_PAREN)
            return GoalDescriptionT("or", or_goals)

        if lookahead == "not":
            self.match(PDDLTokenType.NAME, value="not")
            nested_goal = self.goal_description()
            return GoalDescriptionT("not", nested_goal)

        if lookahead == "exists":  # For the :existential-preconditions requirement flag
            self.match(PDDLTokenType.NAME, value="exists")
            self.match(PDDLTokenType.OPEN_PAREN)
            variables = self.typed_list(PDDLTokenType.VARIABLE)
            self.match(PDDLTokenType.CLOSE_PAREN)
            existential_goal = self.goal_description()
            self.match(PDDLTokenType.CLOSE_PAREN)
            return GoalDescriptionT("exists", variables, existential_goal)

        if lookahead == "forall":  # For the :universal-preconditions requirement flag
            self.match(PDDLTokenType.NAME, value="forall")
            self.match(PDDLTokenType.OPEN_PAREN)
            variables = self.typed_list(PDDLTokenType.VARIABLE)
            self.match(PDDLTokenType.CLOSE_PAREN)
            universal_goal = self.goal_description()
            self.match(PDDLTokenType.CLOSE_PAREN)
            return GoalDescriptionT("forall", variables, universal_goal)

        # Otherwise, just match a literal containing terms (i.e., names or variables)
        return self.literal()

    def effects(self, match_keyword: bool = False) -> EffectsT:
        """Parse PDDL action effects from the input stream of tokens.

        :param match_keyword: Whether to match the preceding `:effect` keyword (default: False)
        :return: Parsed PDDL action effects
        """
        if match_keyword:
            self.match(PDDLTokenType.KEYWORD, ":effect")
        self.match(PDDLTokenType.OPEN_PAREN)

        if self.input_token.type_ != PDDLTokenType.NAME:
            raise RuntimeError(f"Unexpected token type: {self.input_token}")

        lookahead = self.input_token.value
        if lookahead == "and":
            self.match(PDDLTokenType.NAME, value="and")
            parsed_eff: list[EffectsT] = []
            while self.input_token.type_ == PDDLTokenType.OPEN_PAREN:
                parsed_eff.append(self.effects())
            self.match(PDDLTokenType.CLOSE_PAREN)
            return EffectsT("and", parsed_eff)

        if lookahead == "not":
            self.match(PDDLTokenType.NAME, value="not")
            formula = self.atomic_formula(match_open_paren=True)
            self.match(PDDLTokenType.CLOSE_PAREN)
            return EffectsT("not", formula)

        if lookahead == "forall":  # For the :conditional-effects requirement flag
            self.match(PDDLTokenType.NAME, value="forall")
            self.match(PDDLTokenType.OPEN_PAREN)
            variables: list[PDDLToken] = []
            while self.input_token.type_ == PDDLTokenType.VARIABLE:
                variables.append(self.match(PDDLTokenType.VARIABLE))
            self.match(PDDLTokenType.CLOSE_PAREN)
            quantified_eff = self.effects()
            self.match(PDDLTokenType.CLOSE_PAREN)
            return EffectsT("forall", variables, quantified_eff)

        if lookahead == "when":  # For the :conditional-effects requirement flag
            self.match(PDDLTokenType.NAME, value="when")
            condition = self.goal_description()
            conditional_effects = self.effects()
            self.match(PDDLTokenType.CLOSE_PAREN)
            return EffectsT("when", condition, conditional_effects)

        return self.atomic_formula()

    def action(self, match_open_paren: bool = False) -> OperatorT:
        """Parse a PDDL action definition from the stream of input tokens.

        :param match_open_paren: Whether to match the action's open parenthesis (default: False)
        :return: Parsed PDDL action
        """
        if match_open_paren:
            self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.KEYWORD, value=":action")

        name = self.match(PDDLTokenType.NAME).value

        self.match(PDDLTokenType.KEYWORD, value=":parameters")
        self.match(PDDLTokenType.OPEN_PAREN)
        parsed_params = self.typed_list(PDDLTokenType.VARIABLE)
        self.match(PDDLTokenType.CLOSE_PAREN)

        # Parse :vars for the :existential-preconditions and :conditional-effects requirement flags
        variables = None
        preconditions = None
        effects = None

        while self.input_token.type_ == PDDLTokenType.KEYWORD:
            if self.input_token.value == ":vars":
                self.match(PDDLTokenType.KEYWORD, value=":vars")
                self.match(PDDLTokenType.OPEN_PAREN)
                variables = self.typed_list(PDDLTokenType.VARIABLE)
                self.match(PDDLTokenType.CLOSE_PAREN)

            if self.input_token.value == ":precondition":
                self.match(PDDLTokenType.KEYWORD, value=":precondition")
                preconditions = self.goal_description()

            if self.input_token.value == ":effect":
                effects = self.effects(match_keyword=True)

        self.match(PDDLTokenType.CLOSE_PAREN)
        return OperatorT(name, parsed_params, variables, preconditions, effects)

    def require_def(self, match_open_paren: bool = False) -> set[str]:
        """Parse PDDL domain requirements from the stream of input tokens.

        :param match_open_paren: Whether to match the opening parenthesis (defaults to False)
        :return: Set of parsed PDDL requirement keys
        """
        if match_open_paren:
            self.match(PDDLTokenType.OPEN_PAREN)

        self.match(PDDLTokenType.KEYWORD, value=":requirements")

        reqs = set()
        while self.input_token.type_ == PDDLTokenType.KEYWORD:
            requirement = self.match(PDDLTokenType.KEYWORD).value
            reqs.add(requirement)  # TODO: Validate it's supported by this parser!
        self.match(PDDLTokenType.CLOSE_PAREN)
        return reqs

    def domain(self) -> PDDLDomainT:
        """Parse a PDDL domain from the stream of input tokens."""
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, value="define")
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, value="domain")
        domain_name = self.match(PDDLTokenType.NAME)
        self.match(PDDLTokenType.CLOSE_PAREN)

        reqs = None
        types = None
        constants = None
        predicates = None
        actions = None

        # Permit :requirements, :types, :constants, :predicates, and :action in any order
        while self.input_token.type_ != PDDLTokenType.CLOSE_PAREN:
            self.match(PDDLTokenType.OPEN_PAREN)
            if self.input_token.type_ != PDDLTokenType.KEYWORD:
                raise RuntimeError(f"Expected a keyword token, but found: {self.input_token}")

            lookahead = self.input_token.value
            if lookahead == ":requirements":
                reqs = self.require_def()

            elif lookahead == ":types":
                self.match(PDDLTokenType.KEYWORD, value=":types")
                types = self.typed_list(PDDLTokenType.NAME)
                self.match(PDDLTokenType.CLOSE_PAREN)

            elif lookahead == ":constants":
                self.match(PDDLTokenType.KEYWORD, value=":constants")
                constants = self.typed_list(PDDLTokenType.NAME)
                self.match(PDDLTokenType.CLOSE_PAREN)

            elif lookahead == ":predicates":
                self.match(PDDLTokenType.KEYWORD, value=":predicates")
                predicates = []
                while self.input_token.type_ != PDDLTokenType.CLOSE_PAREN:
                    predicates.append(self.atomic_formula_skeleton())
                self.match(PDDLTokenType.CLOSE_PAREN)

            elif lookahead == ":action":
                if actions is None:
                    actions = []
                actions.append(self.action())

        self.match(PDDLTokenType.CLOSE_PAREN)
        return PDDLDomainT(domain_name, reqs, types, constants, predicates, actions)

    def problem(self) -> PDDLProblemT:
        """Parse a PDDL problem from the stream of input tokens.

        Reference: Section 13 (pg. 18) of Ghallab et al., 1998.
        """
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, value="define")
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, value="problem")
        problem_name = self.match(PDDLTokenType.NAME)
        self.match(PDDLTokenType.CLOSE_PAREN)

        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.KEYWORD, value=":domain")
        domain_name = self.match(PDDLTokenType.NAME)
        self.match(PDDLTokenType.CLOSE_PAREN)

        reqs = None  # The :requirements field is optional in a PDDL problem
        objects = None  # The :objects field is optional in a PDDL problem
        initial_state: list[LiteralT] = []  # Treat the :init field as required for PDDL problems
        goals: list[GoalDescriptionT] = []  # A PDDL problem must define at least one :goal

        while self.input_token == PDDLTokenType.OPEN_PAREN:
            self.match(PDDLTokenType.OPEN_PAREN)
            if self.input_token.type_ != PDDLTokenType.KEYWORD:
                raise RuntimeError(f"Unexpected token type: {self.input_token}")
            lookahead = self.input_token.value

            if lookahead == ":requirements":
                reqs = self.require_def()

            elif lookahead == ":objects":
                self.match(PDDLTokenType.KEYWORD, value=":objects")
                objects = self.typed_list(PDDLTokenType.NAME)
                self.match(PDDLTokenType.CLOSE_PAREN)

            elif lookahead == ":init":
                self.match(PDDLTokenType.KEYWORD, value=":init")
                while self.input_token != PDDLTokenType.CLOSE_PAREN:
                    literal = self.literal(term_type=PDDLTokenType.NAME, match_open_paren=True)
                    initial_state.append(literal)
                self.match(PDDLTokenType.CLOSE_PAREN)

            elif lookahead == ":goal":
                self.match(PDDLTokenType.KEYWORD, value=":goal")
                goals.append(self.goal_description())
                self.match(PDDLTokenType.CLOSE_PAREN)

        return PDDLProblemT(problem_name, domain_name, reqs, objects, initial_state, goals)
