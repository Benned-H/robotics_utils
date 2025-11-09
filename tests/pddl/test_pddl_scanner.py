"""Unit tests for the PDDLScanner class."""

from robotics_utils.abstractions.pddl import PDDLScanner, PDDLTokenType


def test_scan_briefcase_world_domain(briefcase_world_domain: str) -> None:
    """Verify that the PDDLScanner class can scan the `briefcase-world` PDDL domain."""
    # Arrange - PDDL text is provided by the test fixture
    scanner = PDDLScanner()

    # Act/Assert - Scan the PDDL domain and expect that no token is a mismatch
    for token in scanner.tokenize(briefcase_world_domain):
        assert token.type_ != PDDLTokenType.MISMATCH, f"Mismatched token: {token}."


def test_scan_mov_b_action(mov_b_action: str) -> None:
    """Verify that the PDDLScanner class can scan the `mov_b_action` PDDL action."""
    # Arrange - PDDL text is provided by the test fixture
    scanner = PDDLScanner()

    # Act/Assert - Scan the PDDL action and expect that no token is a mismatch
    for token in scanner.tokenize(mov_b_action):
        assert token.type_ != PDDLTokenType.MISMATCH, f"Mismatched token: {token}."


def test_scan_put_in_action(put_in_action: str) -> None:
    """Verify that the PDDLScanner class can scan the `put_in_action` PDDL action."""
    # Arrange - PDDL text is provided by the test fixture
    scanner = PDDLScanner()

    # Act/Assert - Scan the PDDL action and expect that no token is a mismatch
    for token in scanner.tokenize(put_in_action):
        assert token.type_ != PDDLTokenType.MISMATCH, f"Mismatched token: {token}."


def test_scan_take_out_action(take_out_action: str) -> None:
    """Verify that the PDDLScanner class can scan the `take_out_action` PDDL action."""
    # Arrange - PDDL text is provided by the test fixture
    scanner = PDDLScanner()

    # Act/Assert - Scan the PDDL action and expect that no token is a mismatch
    for token in scanner.tokenize(take_out_action):
        assert token.type_ != PDDLTokenType.MISMATCH, f"Mismatched token: {token}."


def test_scan_get_paid_problem(get_paid_problem: str) -> None:
    """Verify that the PDDLScanner class can scan the `get_paid_problem` PDDL problem."""
    # Arrange - PDDL text is provided by the test fixture
    scanner = PDDLScanner()

    # Act/Assert - Scan the PDDL problem and expect that no token is a mismatch
    for token in scanner.tokenize(get_paid_problem):
        assert token.type_ != PDDLTokenType.MISMATCH, f"Mismatched token: {token}."
