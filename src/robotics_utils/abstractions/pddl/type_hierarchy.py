"""Define a class to represent hierarchies of object types."""

from collections import defaultdict


class TypeHierarchy:
    """A hierarchy of object types, permitting types to have subtypes."""

    def __init__(self) -> None:
        """Initialize an empty type hierarchy."""
        self._to_children: dict[str, set[str]] = defaultdict(set)
        """A map from the name of each type to the names of the type's subtypes."""

        self._to_parent: dict[str, str]
        """A map from each subtype to its parent type (undefined for parent-less types)."""

    def _validate(self) -> None:
        """Verify that the type hierarchy, as currently defined, is consistent.

        :raises ValueError: If the hierarchy contains contradictory parent-child relationships
        """
        error_msgs = [
            str(
                f"The parent of type '{child}' is defined as '{parent}' but the children "
                f"of '{parent}' don't include '{child}': {self._to_children[parent]}.",
            )
            for child, parent in self._to_parent.items()
            if child not in self._to_children[parent]
        ]

        for parent, children in self._to_children.items():
            for child in children:
                parent_of_the_child = self._to_parent.get(child)
                if parent_of_the_child != parent:
                    error_msgs.append(
                        str(
                            f"Parent type '{parent}' has child '{child}' but the "
                            f"parent of '{child}' is defined as '{parent_of_the_child}'.",
                        ),
                    )

        if error_msgs:
            raise ValueError("Invalid TypeHierarchy:\n\t" + "\n\t".join(error_msgs))
