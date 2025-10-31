"""Define classes to represent queries for VLMs."""

from typing import Iterator


class TextQueries:
    """A set of text queries for an object detection model."""

    def __init__(self) -> None:
        """Initialize an empty list (acting as a set) of text queries."""
        self._queries: list[str] = []

    def __bool__(self) -> bool:
        """Return a Boolean indicating if the set of text queries is empty (empty = False)."""
        return bool(self._queries)

    def __iter__(self) -> Iterator[str]:
        """Return an iterator over the set of text queries."""
        return iter(self._queries)

    def __str__(self) -> str:
        """Return a readable string representation of the text queries."""
        return "\t" + "\n\t".join(self._queries)

    def add(self, query: str) -> None:
        """Add the given text query, or multiple comma-separated queries, to the set."""
        new_queries = [q.strip() for q in query.split(",")]
        for q in new_queries:
            if q and q not in self._queries:
                self._queries.append(q)
        self._queries = sorted(self._queries)

    def remove(self, query: str) -> bool:
        """Remove the given text query from the set.

        :param query: Text query to be removed
        :return: Boolean value indicating if the given query was removed
        """
        try:
            self._queries.remove(query.strip())
        except ValueError as _:
            return False
        return True

    def clear(self) -> None:
        """Clear the set of text queries."""
        self._queries.clear()
