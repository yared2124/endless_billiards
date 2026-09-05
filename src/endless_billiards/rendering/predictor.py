
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import pygame

from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.table import Table
from endless_billiards.core.physics.collisions import circle_line
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.utils.math2d import Vector2


@dataclass(frozen=True, slots=True)
class RaycastHit:
    """Intersection telemetry along a trajectory path."""

    hit_point: Vector2
    normal: Vector2
    distance: float
    target_ball: Ball | None


class TrajectoryPredictor:
    """Calculates ghost ball impacts and tangent reflection angles before shooting."""

    __slots__ = ("_display", "_table")

    def __init__(self, display: DisplayManager, table: Table) -> None:
        self._display: DisplayManager = display
        self._table: Table = table

    def _cast_ray(
        self,
        origin: Vector2,
        direction: Vector2,
        radius: float,
        other_balls: Sequence[Ball],
        max_distance: float = 1200.0,
    ) -> RaycastHit | None:
        """Find the earliest forward intersection against other balls or rails."""
        closest_dist = max_distance
        best_hit: RaycastHit | None = None

        # 1. Circle-Circle Raycast vs Object Balls
        for ball in other_balls:
            if ball.state == BallState.POCKETED:
                continue

            to_ball = ball.pos - origin
            projection = to_ball.dot(direction)

            if projection <= 0.0 or projection >= closest_dist:
                continue

            # Perpendicular distance to the ray
            perp_dist_sq = to_ball.magnitude_squared() - (projection * projection)
            combined_r = radius + ball.radius

            if perp_dist_sq < (combined_r * combined_r):
                # Distance along the ray to the surface intersection point
                half_chord = math.sqrt(max(0.0, (combined_r * combined_r) - perp_dist_sq))
                hit_distance = projection - half_chord

                if 0.0 < hit_distance < closest_dist:
                    closest_dist = hit_distance
                    hit_point = origin + (direction * hit_distance)
                    normal = (hit_point - ball.pos).normalize()
                    best_hit = RaycastHit(
                        hit_point=hit_point,
                        normal=normal,
                        distance=hit_distance,
                        target_ball=ball,
                    )

        # 2. Raycast vs Cushion Segments
        for seg in self._table.cushions:
            seg_vec = seg.p2 - seg.p1
            seg_len = seg_vec.magnitude()
            if seg_len == 0.0:
                continue

            cushion_normal = Vector2(-seg_vec.y / seg_len, seg_vec.x / seg_len)
            denominator = direction.dot(cushion_normal)

            if denominator >= -1e-6:
                continue  # Moving parallel or away from the rail

            # Plane offset by ball radius
            dist_to_plane = (seg.p1 - origin).dot(cushion_normal) + radius
            t = dist_to_plane / denominator

            if 0.0 < t < closest_dist:
                intersect = origin + (direction * t)
                seg_proj = (intersect - seg.p1).dot(seg_vec) / (seg_len * seg_len)

                if 0.0 <= seg_proj <= 1.0:
                    closest_dist = t
                    best_hit = RaycastHit(
                        hit_point=intersect,
                        normal=cushion_normal,
                        distance=t,
                        target_ball=None,
                    )

        return best_hit

    def draw(
        self,
        cue_ball: Ball,
        aim_angle: float,
        other_balls: Sequence[Ball],
    ) -> None:
        """Render the primary shot ray, the ghost cue ball, and the target's cut line."""
        if cue_ball.state != BallState.STATIONARY:
            return

        direction = Vector2(-math.cos(aim_angle), -math.sin(aim_angle))
        hit = self._cast_ray(cue_ball.pos, direction, cue_ball.radius, other_balls)
        if hit is None:
            return

        surface = self._display.screen
        start_px = self._display.world_to_screen(cue_ball.pos)
        hit_px = self._display.world_to_screen(hit.hit_point)

        # 1. Primary trajectory path
        pygame.draw.line(surface, (255, 255, 255), start_px, hit_px, 1)

        # 2. Ghost Cue Ball at impact location
        ghost_r = self._display.scale_scalar(cue_ball.radius)
        pygame.draw.circle(surface, (255, 255, 255), hit_px, ghost_r, 1)

        # 3. Deflection / Cut Lines
        line_len = self._display.scale_scalar(60.0)

        if hit.target_ball is not None:
            # Target ball travels along impact line of centers (inverted collision normal)
            target_dir = -hit.normal
            target_screen = self._display.world_to_screen(hit.target_ball.pos)
            target_end = (
                int(target_screen[0] + target_dir.x * line_len),
                int(target_screen[1] + target_dir.y * line_len),
            )
            pygame.draw.line(surface, (255, 180, 40), target_screen, target_end, 2)

            # Cue ball deflects tangentially (perpendicular to impact normal)
            tangent = Vector2(-target_dir.y, target_dir.x)
            if direction.dot(tangent) < 0.0:
                tangent = -tangent
            cue_end = (
                int(hit_px[0] + tangent.x * (line_len * 0.7)),
                int(hit_px[1] + tangent.y * (line_len * 0.7)),
            )
            pygame.draw.line(surface, (150, 200, 255), hit_px, cue_end, 1)
        else:
            # Cushion bounce reflection line
            refl = direction - (hit.normal * (2.0 * direction.dot(hit.normal)))
            refl_end = (
                int(hit_px[0] + refl.x * line_len),
                int(hit_px[1] + refl.y * line_len),
            )
            pygame.draw.line(surface, (150, 200, 255), hit_px, refl_end, 1)