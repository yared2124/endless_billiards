
from __future__ import annotations

import random
from typing import Sequence

from endless_billiards.config.constants import (
    BALL_RADIUS,
    POCKET_RADIUS,
    TABLE_MAX_X,
    TABLE_MAX_Y,
    TABLE_MIN_X,
    TABLE_MIN_Y,
)
from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.table import Table
from endless_billiards.utils.math2d import Vector2


class BallSpawner:
    """Generates non-overlapping target balls within valid table boundaries."""

    __slots__ = ("_table", "_min_spawn_dist_sq", "_pocket_margin_sq")

    def __init__(self, table: Table) -> None:
        """Initialize spawner geometry references.

        Args:
            table: Core table instance for bounds and pocket validation.
        """
        self._table: Table = table
        # 2x diameter clearance to avoid overlap on spawn
        self._min_spawn_dist_sq = (BALL_RADIUS * 4.0) ** 2
        # Clearance from pocket funnels
        self._pocket_margin_sq = (POCKET_RADIUS * 2.5) ** 2

    def _is_position_valid(self, pos: Vector2, existing_balls: Sequence[Ball]) -> bool:
        """Check if candidate point clears rails, pockets, and other balls."""
        # 1. Clear of pocket funnels
        for pocket in self._table.pockets:
            if pos.distance_squared_to(pocket) < self._pocket_margin_sq:
                return False

        # 2. Clear of other active balls
        for ball in existing_balls:
            if ball.state != BallState.POCKETED:
                if pos.distance_squared_to(ball.pos) < self._min_spawn_dist_sq:
                    return False

        return True

    def spawn_ball(self, existing_balls: Sequence[Ball], max_attempts: int = 50) -> Ball | None:
        """Generate a single target ball at a valid, non-overlapping location.

        Args:
            existing_balls: Active balls to check clearance against.
            max_attempts: Rejection sampling threshold before aborting.

        Returns:
            A new stationary target Ball entity, or None if the table is overcrowded.
        """
        margin = BALL_RADIUS * 3.0
        min_x = TABLE_MIN_X + margin
        max_x = TABLE_MAX_X - margin
        min_y = TABLE_MIN_Y + margin
        max_y = TABLE_MAX_Y - margin

        for _ in range(max_attempts):
            candidate_pos = Vector2(
                random.uniform(min_x, max_x),
                random.uniform(min_y, max_y),
            )
            if self._is_position_valid(candidate_pos, existing_balls):
                return Ball(pos=candidate_pos, radius=BALL_RADIUS, state=BallState.STATIONARY)

        return None

    def replenish_cluster(
        self,
        existing_balls: list[Ball],
        desired_target_count: int,
    ) -> list[Ball]:
        """Ensure the table maintains a target number of object balls.

        Args:
            existing_balls: Current ball list on the table (including cue ball).
            desired_target_count: Target number of non-cue balls.

        Returns:
            List of newly spawned balls added to the table.
        """
        # Count non-pocketed target balls (excluding ball at index 0 which is cue ball)
        active_targets = [
            b for b in existing_balls[1:] if b.state != BallState.POCKETED
        ]
        deficit = desired_target_count - len(active_targets)

        newly_spawned: list[Ball] = []
        for _ in range(deficit):
            new_ball = self.spawn_ball(existing_balls + newly_spawned)
            if new_ball is not None:
                newly_spawned.append(new_ball)

        return newly_spawned