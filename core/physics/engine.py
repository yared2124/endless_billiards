from __future__ import annotations

from typing import Sequence

from endless_billiards.config.constants import (
    BALL_RADIUS,
    CUSHION_RESTITUTION,
    FIXED_TIMESTEP,
)
from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.table import Table
from endless_billiards.core.physics.collisions import (
    circle_circle,
    circle_line,
    resolve_collision,
)
from endless_billiards.utils.spatial_grid import SpatialHashGrid


class PhysicsEngine:
    """Fixed-timestep, broad-to-narrow phase 2D physics engine."""

    __slots__ = ("_grid", "_table", "_dt")

    def __init__(self, table: Table, dt: float = FIXED_TIMESTEP) -> None:
        """Initialize engine with table boundaries and spatial grid partition.

        Args:
            table: Static boundaries and pocket geometry.
            dt: Fixed timestep integration delta in seconds.
        """
        self._table: Table = table
        self._dt: float = dt
        # Cell size set to ~3x maximum ball radius for spatial efficiency
        self._grid: SpatialHashGrid = SpatialHashGrid(cell_size=BALL_RADIUS * 3.0)

    @property
    def table(self) -> Table:
        """Get table reference."""
        return self._table

    def step(self, balls: Sequence[Ball]) -> None:
        """Execute a single atomic physics frame.

        Sequence:
            1. Integrate velocity and positions.
            2. Pocket detection and containment.
            3. Cushion and rail reflections.
            4. Broad-phase candidate discovery (Spatial Hash Grid).
            5. Narrow-phase impulse collision resolution.

        Args:
            balls: Collection of active ball entities to advance.
        """
        # 1. Integrate Ball Dynamics
        for ball in balls:
            ball.update(self._dt)

        # 2. Check Pocket Ingestion
        for ball in balls:
            if ball.state != BallState.POCKETED and self._table.is_in_pocket(ball.pos):
                ball.state = BallState.POCKETED
                ball.vel = ball.vel * 0.0

        # 3. Rail/Cushion Segment Bouncing
        for ball in balls:
            if ball.state == BallState.POCKETED:
                continue

            for segment in self._table.cushions:
                normal = circle_line(ball, segment.p1, segment.p2)
                if normal is not None:
                    # Reflect velocity: v' = v - 2(v . n)n
                    dot = ball.vel.dot(normal)
                    if dot < 0.0:  # Only bounce if moving into the rail
                        ball.vel = (ball.vel - (normal * (2.0 * dot))) * CUSHION_RESTITUTION
                        ball.state = BallState.MOVING

        # 4. Broad-Phase Insertion & Candidate Generation
        self._grid.clear()
        active_balls = [b for b in balls if b.state != BallState.POCKETED]

        for ball in active_balls:
            self._grid.insert(ball)

        # 5. Narrow-Phase Collision Evaluation
        checked_pairs: set[tuple[int, int]] = set()

        for b1 in active_balls:
            candidates = self._grid.query(b1)
            for candidate in candidates:
                # Upcast protocol to Ball entity
                b2 = candidate
                if not isinstance(b2, Ball):
                    continue

                pair_key = (min(b1.id, b2.id), max(b1.id, b2.id))
                if pair_key in checked_pairs:
                    continue

                checked_pairs.add(pair_key)

                if circle_circle(b1, b2):
                    resolve_collision(b1, b2)