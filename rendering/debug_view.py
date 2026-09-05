
from __future__ import annotations

from typing import Sequence

import pygame

from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.utils.math2d import Vector2


class DebugView:
    """Renders debug vectors and entity colliders over the active frame."""

    __slots__ = ("_display",)

    def __init__(self, display: DisplayManager) -> None:
        self._display: DisplayManager = display

    def draw(self, balls: Sequence[Ball]) -> None:
        """Render velocity arrows and AABB wireframes."""
        surface = self._display.screen

        for ball in balls:
            if ball.state == BallState.POCKETED:
                continue

            # 1. AABB wireframe
            aabb = ball.get_aabb()
            screen_tl = self._display.world_to_screen(Vector2(aabb.min_x, aabb.min_y))
            screen_br = self._display.world_to_screen(Vector2(aabb.max_x, aabb.max_y))
            rect = pygame.Rect(
                screen_tl[0],
                screen_tl[1],
                screen_br[0] - screen_tl[0],
                screen_br[1] - screen_tl[1],
            )
            pygame.draw.rect(surface, (70, 120, 220), rect, width=1)

            # 2. Velocity vector arrow
            if ball.is_moving:
                start_pt = self._display.world_to_screen(ball.pos)
                # Scale velocity visually for readability (0.1s forward horizon)
                vector_end = ball.pos + (ball.vel * 0.1)
                end_pt = self._display.world_to_screen(vector_end)
                pygame.draw.line(surface, (255, 60, 60), start_pt, end_pt, 2)