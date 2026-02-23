"""Define a class to represent symbolic predicates representing abstract relations."""

# def to_pddl(self) -> str:
#     """Return a PDDL string representation of the predicate."""
#     params_per_type: dict[str, list[str]] = defaultdict(list)
#     for p in self.parameters:
#         params_per_type[p.type_].append(p.lifted_name)

#     typed_variables = [f"{' '.join(params)} - {t}" for t, params in params_per_type.items()]
#     variables_str = (" " + " ".join(typed_variables)) if typed_variables else ""
#     return f"({self.name}{variables_str})"
