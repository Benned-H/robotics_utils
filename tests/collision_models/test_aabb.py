"""Unit tests for axis-aligned bounding boxes (i.e., AABBs)."""

from __future__ import annotations

from hypothesis import given
from hypothesis.strategies import lists

from robotics_utils.collision_models import AxisAlignedBoundingBox

from .collision_model_strategies import aabbs


@given(aabbs())
def test_aabb_vertices(aabb: AxisAlignedBoundingBox) -> None:
    """Verify that any AABB has eight vertices, each at an intersection of box edges."""
    # Arrange/Act - Compute the list of vertices of the given AABB
    vertices_list = list(aabb.vertices)

    # Assert - Verify that there are eight vertices, they're ~distinct, and along the box edges
    expected_total = 8
    assert len(vertices_list) == expected_total

    expected_distinct = 8
    if aabb.min_xyz.x == aabb.max_xyz.x:
        expected_distinct /= 2
    if aabb.min_xyz.y == aabb.max_xyz.y:
        expected_distinct /= 2
    if aabb.min_xyz.z == aabb.max_xyz.z:
        expected_distinct /= 2

    distinct_vertices = {tuple(v) for v in vertices_list}
    assert len(distinct_vertices) == expected_distinct

    for vertex in vertices_list:
        assert vertex.x in (aabb.min_xyz.x, aabb.max_xyz.x)
        assert vertex.y in (aabb.min_xyz.y, aabb.max_xyz.y)
        assert vertex.z in (aabb.min_xyz.z, aabb.max_xyz.z)


@given(lists(elements=aabbs(), max_size=10000))
def test_aabb_union(aabb_list: list[AxisAlignedBoundingBox]) -> None:
    """Verify that unions of axis-aligned bounding boxes (AABBs) are computed correctly."""
    # Arrange/Act - Given a list of AABBs, compute their union
    union_aabb = AxisAlignedBoundingBox.union(aabb_list)

    # Assert - Expect that the union AABB contains all AABBs in the list
    for aabb in aabb_list:
        assert union_aabb.contains(aabb)
