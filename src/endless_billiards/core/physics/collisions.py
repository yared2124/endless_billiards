"""Collision primitives and impulse-based narrow-phase resolution functions."""

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
    normal = delta / distance

    # Positional depenetration
    overlap = min_distance - distance
    total_mass = b1.mass + b2.mass
    b1_ratio = b2.mass / total_mass
    b2_ratio = b1.mass / total_mass

    b1.pos += normal * (overlap * b1_ratio)
    b2.pos -= normal * (overlap * b2_ratio)

    relative_vel = b1.vel - b2.vel
    vel_along_normal = relative_vel.dot(normal)

    if vel_along_normal > 0.0:
        return

    impulse_magnitude = -(1.0 + RESTITUTION) * vel_along_normal
    impulse_magnitude /= (1.0 / b1.mass) + (1.0 / b2.mass)

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
        The normalized collision vector if collision occurs, otherwise None.
    """
    if ball.state == BallState.POCKETED:
        return None

    seg_vec = p2 - p1
    seg_len_sq = seg_vec.magnitude_squared()

    if seg_len_sq == 0.0:
        diff = ball.pos - p1
        if diff.magnitude_squared() <= ball.radius * ball.radius:
            return diff.normalize()
        return None

    ball_vec = ball.pos - p1
    t = max(0.0, min(1.0, ball_vec.dot(seg_vec) / seg_len_sq))

    closest_point = p1 + (seg_vec * t)
    distance_vec = ball.pos - closest_point
    distance_sq = distance_vec.magnitude_squared()

    if distance_sq <= ball.radius * ball.radius:
        distance = math.sqrt(distance_sq)
        if distance > 1e-6:
            normal = distance_vec / distance
            overlap = ball.radius - distance
            ball.pos += normal * overlap
            return normal
        else:
            seg_norm = seg_vec.normalize()
            return Vector2(-seg_norm.y, seg_norm.x)

    return None


def swept_circle_line(
    ball: Ball,
    p1: Vector2,
    p2: Vector2,
    dt: float,
) -> Optional[tuple[Vector2, float]]:
    """Continuous collision detection (CCD) for a moving circle vs. static segment."""
    if ball.state == BallState.POCKETED:
        return None

    displacement = ball.vel * dt
    disp_len_sq = displacement.magnitude_squared()

    if disp_len_sq < 1e-6:
        normal = circle_line(ball, p1, p2)
        return (normal, 0.0) if normal is not None else None

    seg = p2 - p1
    seg_len_sq = seg.magnitude_squared()
    if seg_len_sq < 1e-6:
        return None

    seg_len = math.sqrt(seg_len_sq)
    cushion_normal = Vector2(-seg.y / seg_len, seg.x / seg_len)

    vel_dot_normal = displacement.dot(cushion_normal)
    if vel_dot_normal >= 0.0:
        return None

    dist_to_line = (ball.pos - p1).dot(cushion_normal)
    t_hit = (ball.radius - dist_to_line) / vel_dot_normal

    if 0.0 <= t_hit <= 1.0:
        impact_pos = ball.pos + (displacement * t_hit)
        projection = (impact_pos - p1).dot(seg) / seg_len_sq
        if 0.0 <= projection <= 1.0:
            return cushion_normal, t_hit

    for endpoint in (p1, p2):
        m = ball.pos - endpoint
        b = m.dot(displacement)
        c = m.magnitude_squared() - (ball.radius * ball.radius)

        if c < 0.0:
            return m.normalize(), 0.0

        discriminant = b * b - disp_len_sq * c
        if discriminant >= 0.0:
            t = (-b - math.sqrt(discriminant)) / disp_len_sq
            if 0.0 <= t <= 1.0:
                hit_pos = ball.pos + (displacement * t)
                normal = (hit_pos - endpoint).normalize()
                return normal, t

    return None
