"""Define a class to parse PDDL domain and problem files."""

from dataclasses import dataclass
from typing import Any

from robotics_utils.task_planning.parameters import DiscreteParameter
from robotics_utils.task_planning.pddl_scanner import PDDLScanner, PDDLToken, PDDLTokenType
from robotics_utils.task_planning.predicates import Predicate


@dataclass
class TypedList:
    """A list of typed entities parsed from PDDL."""

    type_map: dict[str, set[PDDLToken]]
    """A map from type names to the set of entities (e.g., subtypes or variables) of the type."""

    entity_order: list[PDDLToken]
    """A list tracking the order of the parsed entities."""

    def to_parameters(self) -> tuple[DiscreteParameter, ...]:
        """Convert the TypedList into a tuple of object-typed parameters."""
        param_types: dict[str, str] = {}  # Map each parameter name to its PDDL type
        for p_type, param_tokens in self.type_map.items():
            for p_token in param_tokens:
                if p_token.type != PDDLTokenType.VARIABLE:
                    raise RuntimeError(f"Cannot convert token {p_token} into a DiscreteParameter.")
                param_types[p_token.value] = p_type

        param_names = [p_token.value for p_token in self.entity_order]
        return tuple(DiscreteParameter(param, param_types[param]) for param in param_names)


@dataclass
class Literal:
    """A predicate or its negation."""

    predicate: Predicate
    negated: bool


PDDLDomain = Any  # TODO: Create real types
Operator = Any


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
        operators: set[Operator] = {}
        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            self.match(PDDLTokenType.OPEN_PAREN)
            keyword = self.match(PDDLTokenType.KEYWORD)

            match keyword.value:
                case ":requirements":
                    reqs = self.requirements()

                case ":types":
                    types_list = self.typed_list(PDDLTokenType.NAME)

                case ":predicates":
                    predicates = self.predicate_defs()

                case ":action":
                    operator = self.action()
                    operators.add(operator)

        self.match(PDDLTokenType.CLOSE_PAREN)

        return PDDLDomain(domain_name, reqs, types_list, predicates, operators)  # TODO: Real type!

    def requirements(self) -> set[str]:
        """Parse requirements from a PDDL domain until a closing parenthesis is matched.

        :return: Set of strings specifying requirements of the domain
        """
        parsed_reqs = self.untyped_list(PDDLTokenType.KEYWORD)
        return {req_token.value for req_token in parsed_reqs}

    def untyped_list(self, token_type: PDDLTokenType) -> list[PDDLToken]:
        """Parse a list of the given token type until a closing parenthesis is matched."""
        tokens: list[PDDLToken] = []
        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            tokens.append(self.match(token_type))
        self.match(PDDLTokenType.CLOSE_PAREN)
        return tokens

    def typed_list(self, token_type: PDDLTokenType) -> TypedList:
        """Parse a typed list of the given token type until a closing parenthesis is matched.

        :param token_type: Type of PDDL token (e.g., VARIABLE) being assigned types
        :return: Map from type names to the set of entities of that type
        """
        type_map: dict[str, set[PDDLToken]] = {}  # Map each type to the set of entities of the type
        entity_order: list[PDDLToken] = []  # Track the order of parsed entities
        awaiting_type: set[PDDLToken] = set()  # Entities awaiting "-" followed by a parent type

        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            if self.input_token.type == PDDLTokenType.MINUS:  # Match "-" plus the following type
                self.match(PDDLTokenType.MINUS)
                parent_type = self.match(PDDLTokenType.NAME).value

                if parent_type not in type_map:
                    type_map[parent_type] = set()

                type_map[parent_type].update(awaiting_type)
                awaiting_type.clear()  # Reset the set of entities awaiting a parent
                continue

            if self.input_token.type == token_type:  # New to-be-typed entity => Await its type
                token = self.match(token_type)
                entity_order.append(token)
                awaiting_type.add(token)
                continue

            raise RuntimeError(f"Unexpected token while parsing typed list: {self.input_token}")

        self.match(PDDLTokenType.CLOSE_PAREN)
        return TypedList(type_map, entity_order)

    def predicate_defs(self) -> set[Predicate]:
        """Parse PDDL predicate definitions until a closing parenthesis is matched.

        :return: Set of parsed predicate definitions
        """
        parsed_predicates: set[Predicate] = set()
        while self.input_token.type == PDDLTokenType.OPEN_PAREN:  # Each predicate begins with "("
            parsed_predicates.add(self.predicate_def())

        self.match(PDDLTokenType.CLOSE_PAREN)  # Match the closing parenthesis for :predicates
        return parsed_predicates

    def predicate_def(self) -> Predicate:
        """Parse a PDDL predicate definition from the input tokens."""
        self.match(PDDLTokenType.OPEN_PAREN)
        predicate_name = self.match(PDDLTokenType.NAME).value
        parsed_params = self.typed_list(PDDLTokenType.VARIABLE)
        parameters = parsed_params.to_parameters()
        return Predicate(predicate_name, parameters)

    def operator_predicate(self) -> Predicate:
        """Parse a PDDL predicate as written within an operator definition."""
        self.match(PDDLTokenType.OPEN_PAREN)
        predicate_name = self.match(PDDLTokenType.NAME).value

        parsed_params = self.untyped_list(PDDLTokenType.VARIABLE)
        param_names = [p_token.value for p_token in parsed_params]

        # TODO: Construct Predicate w/in operator, probably using code from SkillWrapper
        return Predicate(predicate_name, None)  # TODO: Implement

    def operator_literal(self) -> Literal:
        """Parse a PDDL literal as written within an operator definition."""
        self.match(PDDLTokenType.OPEN_PAREN)
        initial_token = self.match(PDDLTokenType.NAME)
        negated = initial_token.value == "not"

        if negated:
            predicate = self.operator_predicate()
            self.match(PDDLTokenType.CLOSE_PAREN)  # Close the negation
        else:  # Finish parsing the rest of the predicate
            predicate_name = initial_token.value
            parsed_params = self.untyped_list(PDDLTokenType.VARIABLE)  # Consumes the closing ")"
            param_names = [p_token.value for p_token in parsed_params]

            # TODO: Construct a Predicate given the list of parameter names and type map
            predicate = Predicate(predicate_name, None)  # TODO: Implement

        self.match(PDDLTokenType.CLOSE_PAREN)  # Close the negation or predicate
        return Literal(predicate, negated)

    def action(self) -> Operator:
        """Parse a PDDL operator (i.e., lifted action) until its closing parenthesis.

        :return: Parsed Operator instance
        """
        operator_name = self.match(PDDLTokenType.NAME).value

        self.match(PDDLTokenType.KEYWORD, value=":parameters")
        self.match(PDDLTokenType.OPEN_PAREN)
        parsed_params = self.typed_list(PDDLTokenType.VARIABLE)
        parameters = parsed_params.to_parameters()

        preconditions = self.preconditions()
        effects = self.effects()

        self.match(PDDLTokenType.CLOSE_PAREN)
        return Operator(operator_name, parameters, preconditions, effects)  # TODO: Actual type!

    def preconditions(self) -> Any:  # TODO: Type?
        """Parse preconditions from a PDDL operator definition.

        :return: TODO
        """
        self.match(PDDLTokenType.KEYWORD, ":precondition")
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, "and")

        positive_pre: list[Predicate] = []
        negative_pre: list[Predicate] = []
        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            literal = self.operator_literal()
            if literal.negated:
                negative_pre.append(literal.predicate)
            else:
                positive_pre.append(literal.predicate)
        self.match(PDDLTokenType.CLOSE_PAREN)

        return (positive_pre, negative_pre)  # TODO: ACTUAL TYPE

    def effects(self) -> Any:  # TODO: Type?
        """Parse effects from a PDDL operator definition.

        :return: TODO
        """
        self.match(PDDLTokenType.KEYWORD, ":effect")
        self.match(PDDLTokenType.OPEN_PAREN)
        self.match(PDDLTokenType.NAME, "and")

        add_effects: list[Predicate] = []
        delete_effects: list[Predicate] = []
        while self.input_token.type != PDDLTokenType.CLOSE_PAREN:
            literal = self.operator_literal()
            if literal.negated:
                delete_effects.append(literal.predicate)
            else:
                add_effects.append(literal.predicate)
        self.match(PDDLTokenType.CLOSE_PAREN)

        return (add_effects, delete_effects)  # TODO: ACTUAL TYPE
