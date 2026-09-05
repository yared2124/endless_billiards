
from __future__ import annotations

import math
from collections import defaultdict
from typing import Protocol, TypeAlias, runtime_checkable

from endless_billiards.utils.math2d import BoundingBox, Vector2

CellCoord: TypeAlias = tuple[int, int]


@runtime_checkable
class SpatialEntity(Protocol):
    """Interface required for entities managed by the spatial grid."""

    @property
    def id(self) -> int:
        """Unique numerical identifier of the entity."""
        ...

    @property
    def position(self) -> Vector2:
        """Center position of the entity in continuous space."""
        ...

    @property
    def radius(self) -> float:
        """Bounding radius of the entity."""
        ...

    def get_aabb(self) -> BoundingBox:
        """Calculate the Axis-Aligned Bounding Box enclosing the entity."""
        ...


class SpatialHashGrid:
    """A uniform 2D grid that reduces O(n^2) collisions to near O(n).

    Entities overlapping multiple cells are tracked across each cell, while
    queries ensure each candidate pair is deduplicated.
    """

    __slots__ = ("_cell_size", "_inv_cell_size", "_grid", "_entity_cells")

    def __init__(self, cell_size: float = 64.0) -> None:
        """Initialize the grid.

        Args:
            cell_size: Width and height of each spatial cell. Optimal value
                is typically 2x to 4x the maximum entity diameter.
        """
        if cell_size <= 0.0:
            raise ValueError("Cell size must be strictly positive.")

        self._cell_size: float = float(cell_size)
        self._inv_cell_size: float = 1.0 / self._cell_size
        self._grid: dict[CellCoord, list[SpatialEntity]] = defaultdict(list)
        self._entity_cells: dict[int, list[CellCoord]] = defaultdict(list)

    def _hash_point(self, x: float, y: float) -> CellCoord:
        """Map world coordinates to integer grid coordinates."""
        return (
            math.floor(x * self._inv_cell_size),
            math.floor(y * self._inv_cell_size),
        )

    def _get_overlapping_cells(self, aabb: BoundingBox) -> list[CellCoord]:
        """Compute all cell coordinates that intersect the bounding box."""
        min_cell_x = math.floor(aabb.min_x * self._inv_cell_size)
        max_cell_x = math.floor(aabb.max_x * self._inv_cell_size)
        min_cell_y = math.floor(aabb.min_y * self._inv_cell_size)
        max_cell_y = math.floor(aabb.max_y * self._inv_cell_size)

        cells: list[CellCoord] = []
        for cx in range(min_cell_x, max_cell_x + 1):
            for cy in range(min_cell_y, max_cell_y + 1):
                cells.append((cx, cy))
        return cells

    def insert(self, entity: SpatialEntity) -> None:
        """Insert an entity into all intersecting grid cells.

        If the entity already exists in the grid, its old positions are cleared.

        Args:
            entity: An object fulfilling the SpatialEntity protocol.
        """
        if entity.id in self._entity_cells:
            self.remove(entity)

        cells = self._get_overlapping_cells(entity.get_aabb())
        for cell in cells:
            self._grid[cell].append(entity)

        self._entity_cells[entity.id] = cells

    def remove(self, entity: SpatialEntity) -> None:
        """Remove an entity from all occupied cells.

        Args:
            entity: The entity to remove.
        """
        cells = self._entity_cells.pop(entity.id, None)
        if not cells:
            return

        for cell in cells:
            bucket = self._grid.get(cell)
            if bucket:
                self._grid[cell] = [e for e in bucket if e.id != entity.id]
                if not self._grid[cell]:
                    del self._grid[cell]

    def query(self, entity: SpatialEntity) -> list[SpatialEntity]:
        """Find potential collision candidates intersecting the entity's AABB.

        Deduplicates candidate entities spanning the same cells as the query.

        Args:
            entity: Target entity initiating the query.

        Returns:
            List of distinct neighboring entities. Excludes the query entity.
        """
        cells = self._get_overlapping_cells(entity.get_aabb())
        candidates: dict[int, SpatialEntity] = {}

        for cell in cells:
            bucket = self._grid.get(cell)
            if not bucket:
                continue
            for neighbor in bucket:
                if neighbor.id != entity.id:
                    candidates[neighbor.id] = neighbor

        return list(candidates.values())

    def clear(self) -> None:
        """Reset the internal structure and clear all buckets."""
        self._grid.clear()
        self._entity_cells.clear()