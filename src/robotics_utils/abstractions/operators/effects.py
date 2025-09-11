@dataclass(frozen=True)
class Effects:
    """A collection of predicates defining add and delete effects of an operator."""

    add: set[Predicate]  # Predicates added to the abstract state by the operator
    delete: set[Predicate]  # Predicates removed from the abstract state by the operator

    def to_pddl(self) -> str:
        """Return a PDDL string representation of the effects."""
        add_eff = "\n\t".join(sorted(p.to_pddl() for p in self.add))
        delete_eff = "\n\t".join(sorted(f"(not {p.to_pddl()})" for p in self.delete))

        return f":effect (and\n\t{add_eff}\n\t{delete_eff}\n)"

    def ground_with(self, bindings: Bindings) -> GroundedEffects:
        """Ground the effects using the given parameter bindings."""
        return GroundedEffects(
            add={p.ground_with(bindings) for p in self.add},
            delete={p.ground_with(bindings) for p in self.delete},
        )


@dataclass(frozen=True)
class GroundedEffects:
    """A collection of grounded effects resulting from applying an operator."""

    add: set[PredicateInstance]  # Grounded predicates added to the abstract state
    delete: set[PredicateInstance]  # Grounded predicates removed from the abstract state

    def apply(self, abstract_state: AbstractState) -> AbstractState:
        """Apply the grounded effects to the given abstract state."""
        return AbstractState(abstract_state.facts.difference(self.delete).union(self.add))
