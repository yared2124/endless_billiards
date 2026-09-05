
from __future__ import annotations

import math
from typing import Optional

from endless_billiards.config.constants import RESTITUTION
from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.utils.math2d import Vector2


def circle_circle(b1: Ball, b2: Ball) -> bool:
    """Test circle-circle intersection between two balls.

    Args:
        b1: First ball instance.
        b2: Second ball instance.

    Returns:
        True if bounding radii overlap, otherwise False.
    """
    if b1.state == BallState.POCKETED or b2.state == BallState.POCKETED:
        return False

    min_dist = b1.radius + b2.radius
    return b1.pos.distance_squared_to(b2.pos) < (min_dist * min_dist)


def resolve_collision(b1: Ball, b2: Ball) -> None:
    """Execute 2D impulse resolution and position correction between two balls.

    Args:
        b1: First colliding ball.
        b2: Second colliding ball.
    """
    if b1.state == BallState.POCKETED or b2.state == BallState.POCKETED:
        return

    delta = b1.pos - b2.pos
    distance_sq = delta.magnitude_squared()
    min_distance = b1.radius + b2.radius

    if distance_sq >= (min_distance * min_distance) or distance_sq == 0.0:
        return

    distance = math.sqrt(distance_sq)
    normal = delta / distance  # Normal pointing from b2 to b1

    # Positional depenetration (Anti-clumping)
    overlap = min_distance - distance
    total_mass = b1.mass + b2.mass
    b1_ratio = b2.mass / total_mass
    b2_ratio = b1.mass / total_mass

    b1.pos += normal * (overlap * b1_ratio)
    b2.pos -= normal * (overlap * b2_ratio)

    # Relative velocity
    relative_vel = b1.vel - b2.vel
    vel_along_normal = relative_vel.dot(normal)

    # Do not resolve if velocities are separating
    if vel_along_normal > 0.0:
        return

    # Impulse scalar
    impulse_magnitude = -(1.0 + RESTITUTION) * vel_along_normal
    impulse_magnitude /= (1.0 / b1.mass) + (1.0 / b2.mass)

    # Apply impulse along normal vector
    impulse = normal * impulse_magnitude
    b1.vel += impulse / b1.mass
    b2.vel -= impulse / b2.mass

    b1.state = BallState.MOVING
    b2.state = BallState.MOVING


def circle_line(ball: Ball, p1: Vector2, p2: Vector2) -> Optional[Vector2]:
    """Test and detect collision between a ball and a directed line segment.

    Args:
        ball: The target ball.
        p1: Segment starting endpoint.
        p2: Segment ending endpoint.

    Returns:
        The normalized collision vector (surface normal pointing toward the ball)
        if collision occurs, otherwise None.
    """
    if ball.state == BallState.POCKETED:
        return None

    seg_vec = p2 - p1
    seg_len_sq = seg_vec.magnitude_squared()

    if seg_len_sq == 0.0:
        # Degenerate line segment into point
        diff = ball.pos - p1
        if diff.magnitude_squared() <= ball.radius * ball.radius:
            return diff.normalize()
        return None

    # Project ball position onto segment vector (t clamped to [0, 1])
    ball_vec = ball.pos - p1
    t = max(0.0, min(1.0, ball_vec.dot(seg_vec) / seg_len_sq))

    closest_point = p1 + (seg_vec * t)
    distance_vec = ball.pos - closest_point
    distance_sq = distance_vec.magnitude_squared()

    if distance_sq <= ball.radius * ball.radius:
        distance = math.sqrt(distance_sq)
        if distance > 1e-6:
            normal = distance_vec / distance
            # Correct penetration
            overlap = ball.radius - distance
            ball.pos += normal * overlap
            return normal
        else:
            # Ball center sits exactly on the line: calculate perpendicular normal
            seg_norm = seg_vec.normalize()
            return Vector2(-seg_norm.y, seg_norm.x)

    return None