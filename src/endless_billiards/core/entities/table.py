
from __future__ import annotations

from typing import NamedTuple

from endless_billiards.config.constants import (
    POCKET_RADIUS_SQUARED,
    TABLE_MAX_X,
    TABLE_MAX_Y,
    TABLE_MIN_X,
    TABLE_MIN_Y,
)
from endless_billiards.utils.math2d import Vector2


class Segment(NamedTuple):
    """Directed line segment representing a physical cushion border."""

    p1: Vector2
    p2: Vector2


class Table:
    """Headless playing table geometry and pocket sensor model."""

    __slots__ = ("cushions", "pockets")

    def __init__(self) -> None:
        """Initialize standard cushion segments and 6 pocket points."""
        tl = Vector2(TABLE_MIN_X, TABLE_MIN_Y)
        tr = Vector2(TABLE_MAX_X, TABLE_MIN_Y)
        br = Vector2(TABLE_MAX_X, TABLE_MAX_Y)
        bl = Vector2(TABLE_MIN_X, TABLE_MAX_Y)

        # 4 perimeter line segments (clockwise orientation)
        self.cushions: tuple[Segment, ...] = (
            Segment(tl, tr),  # Top Rail
            Segment(tr, br),  # Right Rail
            Segment(br, bl),  # Bottom Rail
            Segment(bl, tl),  # Left Rail
        )

        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5

        # Pockets: 4 corners + 2 center side pockets
        self.pockets: tuple[Vector2, ...] = (
            tl,                                 # Top-Left
            Vector2(center_x, TABLE_MIN_Y),    # Top-Center
            tr,                                 # Top-Right
            br,                                 # Bottom-Right
            Vector2(center_x, TABLE_MAX_Y),    # Bottom-Center
            bl,                                 # Bottom-Left
        )

    def is_in_pocket(self, pos: Vector2) -> bool:
        """Check if a position falls within any table pocket radius.

        Args:
            pos: Point coordinate to query.

        Returns:
            True if position overlaps a pocket, otherwise False.
        """
        for pocket_pos in self.pockets:
            if pos.distance_squared_to(pocket_pos) <= POCKET_RADIUS_SQUARED:
                return True
        return False