@dataclass(frozen=True)
class Preconditions:
    """A collection of predicates defining positive and negative preconditions."""

    positive: set[Predicate]  # Abstract conditions that must hold true to apply an operator
    negative: set[Predicate]  # Abstract conditions that must be false to apply an operator

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the preconditions."""
        positive_pre = "\n\t".join(sorted(p.to_pddl() for p in self.positive))
        negative_pre = "\n\t".join(sorted(f"(not {p.to_pddl()})" for p in self.negative))

        return f":precondition (and\n\t{positive_pre}\n\t{negative_pre}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedPreconditions:
        """Ground the preconditions using the given parameter bindings."""
        return GroundedPreconditions(
            positive={p.ground_with(bindings) for p in self.positive},
            negative={p.ground_with(bindings) for p in self.negative},
        )


@dataclass(frozen=True)
class GroundedPreconditions:
    """A collection of grounded preconditions to applying an operator."""

    positive: set[PredicateInstance]  # Grounded predicates that must hold to apply an operator
    negative: set[PredicateInstance]  # Grounded predicates that must be false to apply an operator

    def satisfied_in(self, abstract_state: AbstractState) -> bool:
        """Evaluate whether the grounded preconditions are satisfied in an abstract state."""
        if any((pos_pre not in abstract_state) for pos_pre in self.positive):
            return False
        return all((neg_pre not in abstract_state) for neg_pre in self.negative)
