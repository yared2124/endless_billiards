
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import itertools

from endless_billiards.config.constants import (
    BALL_MASS,
    BALL_RADIUS,
    FRICTION_DECAY,
    MAX_BALL_SPEED,
    SLEEP_VELOCITY_THRESHOLD,
)
from endless_billiards.utils.math2d import BoundingBox, Vector2


class BallState(Enum):
    """Execution states for simulation entities."""

    STATIONARY = auto()
    MOVING = auto()
    POCKETED = auto()


_ball_id_generator = itertools.count(1)


@dataclass(slots=True)
class Ball:
    """Stateful, headless billiard ball physics entity.

    Attributes:
        pos: World position vector.
        vel: Linear velocity vector in units per second.
        radius: Collision circle radius.
        mass: Inelastic and elastic mass in kg.
        state: Operational lifecycle status.
        id: Unique identifier for broad-phase deduplication.
    """

    pos: Vector2
    vel: Vector2 = Vector2.zero()
    radius: float = BALL_RADIUS
    mass: float = BALL_MASS
    state: BallState = BallState.STATIONARY
    id: int = -1

    def __post_init__(self) -> None:
        """Assign unique ID if not provided explicitly."""
        if self.id == -1:
            self.id = next(_ball_id_generator)

    @property
    def position(self) -> Vector2:
        """Spatial entity protocol requirement."""
        return self.pos

    def get_aabb(self) -> BoundingBox:
        """Calculate Axis-Aligned Bounding Box for spatial indexing.

        Returns:
            Bounding box enclosing current ball circle.
        """
        return BoundingBox(
            min_x=self.pos.x - self.radius,
            min_y=self.pos.y - self.radius,
            max_x=self.pos.x + self.radius,
            max_y=self.pos.y + self.radius,
        )

    def update(self, dt: float) -> None:
        """Advance integration and resolve friction damping.

        Args:
            dt: Delta time increment in seconds.
        """
        if self.state == BallState.POCKETED:
            self.vel = Vector2.zero()
            return

        speed_sq = self.vel.magnitude_squared()
        if speed_sq < (SLEEP_VELOCITY_THRESHOLD * SLEEP_VELOCITY_THRESHOLD):
            self.vel = Vector2.zero()
            self.state = BallState.STATIONARY
            return

        self.state = BallState.MOVING

        # Apply velocity clamp to avoid tunneling
        self.vel = self.vel.clamp_magnitude(MAX_BALL_SPEED)

        # Euler position integration
        self.pos += self.vel * dt

        # Friction decay normalized across variable frame steps
        # Decays proportionally using friction constant powered by step factor
        decay = FRICTION_DECAY ** (dt * 120.0)
        self.vel *= decay

        # Check sleep threshold post-integration
        if self.vel.magnitude_squared() < (SLEEP_VELOCITY_THRESHOLD * SLEEP_VELOCITY_THRESHOLD):
            self.vel = Vector2.zero()
            self.state = BallState.STATIONARY