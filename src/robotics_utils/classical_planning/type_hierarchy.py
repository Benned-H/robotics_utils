"""Define a class to represent hierarchies of object types."""


class TypeHierarchy:
    """A hierarchy of object types, permitting types to have subtypes."""

    def __init__(self) -> None:
        """Initialize an empty type hierarchy."""
        self._to_children: dict[str, set[str]] = {}
        """A map from each type name to the set of names of their child types."""

        self._to_parent: dict[str, str | None]
        """A map from each type to its parent type, if it has one (else None)."""

    def validate(self) -> None:
        """Verify that the current contents of the type hierarchy are consistent."""
        for parent_type, children in self._to_children.items():
            for child_type in children:
                if self._to_parent[child_type] != parent_type:
                    raise ValueError(
                        f"Parent type {parent_type} has child {child_type} but the "
                        f"parent of {child_type} is {self._to_parent[child_type]}",
                    )

        for child_type, parent_type in self._to_parent.items():
            if parent_type is not None and (child_type not in self._to_children[parent_type]):
                raise ValueError(
                    f"Child type {child_type} has parent {parent_type} but the children of "
                    f"{parent_type} don't contain {child_type}: {self._to_children[parent_type]}",
                )
