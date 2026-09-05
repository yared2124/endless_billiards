
from __future__ import annotations

import math
from typing import Sequence

import pygame

from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.table import Table
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.utils.math2d import Vector2


class TableSprite:
    """Renders table felt, outer cushion rails, and pocket apertures."""

    __slots__ = ("_display",)

    def __init__(self, display: DisplayManager) -> None:
        """Initialize sprite with screen projection adapter.

        Args:
            display: Active display subsystem.
        """
        self._display: DisplayManager = display

    def draw(self, table: Table) -> None:
        """Render table cushions, pockets, and surface borders.

        Args:
            table: Headless table model.
        """
        surface = self._display.screen

        # Draw cushion borders
        for segment in table.cushions:
            p1 = self._display.world_to_screen(segment.p1)
            p2 = self._display.world_to_screen(segment.p2)
            cushion_thickness = self._display.scale_scalar(14.0)
            pygame.draw.line(surface, (25, 75, 40), p1, p2, cushion_thickness)

        # Draw pockets
        pocket_radius = self._display.scale_scalar(22.0)
        for pocket in table.pockets:
            screen_pos = self._display.world_to_screen(pocket)
            # Outer brass rim
            pygame.draw.circle(surface, (180, 150, 60), screen_pos, pocket_radius + 3)
            # Pocket drop
            pygame.draw.circle(surface, (8, 8, 10), screen_pos, pocket_radius)


class BallSprite:
    """Renders a single Ball entity with pseudo-3D shading."""

    __slots__ = ("_display",)

    def __init__(self, display: DisplayManager) -> None:
        """Initialize ball sprite renderer.

        Args:
            display: Active display subsystem.
        """
        self._display: DisplayManager = display

    def draw(self, ball: Ball, primary_color: tuple[int, int, int]) -> None:
        """Draw a ball entity to screen if not pocketed.

        Args:
            ball: Target headless ball entity.
            primary_color: RGB tint of the ball.
        """
        if ball.state == BallState.POCKETED:
            return

        screen_pos = self._display.world_to_screen(ball.pos)
        radius = self._display.scale_scalar(ball.radius)
        surface = self._display.screen

        # Base sphere
        pygame.draw.circle(surface, primary_color, screen_pos, radius)

        # Specular highlight (gives the ball depth)
        specular_offset = max(1, radius // 3)
        specular_radius = max(1, radius // 4)
        highlight_pos = (screen_pos[0] - specular_offset, screen_pos[1] - specular_offset)
        pygame.draw.circle(surface, (255, 255, 255), highlight_pos, specular_radius)

        # Darkened rim
        pygame.draw.circle(surface, (20, 20, 20), screen_pos, radius, 1)


class CueSprite:
    """Renders aiming guides and the billiard cue stick during shot prep."""

    __slots__ = ("_display",)

    def __init__(self, display: DisplayManager) -> None:
        """Initialize cue sprite renderer.

        Args:
            display: Active display subsystem.
        """
        self._display: DisplayManager = display

    def draw(
        self,
        cue_ball: Ball,
        aim_angle: float,
        power_ratio: float,
        is_charging: bool,
    ) -> None:
        """Draw aiming trajectory guide and retracted wooden stick.

        Args:
            cue_ball: The target cue ball entity.
            aim_angle: Current aiming heading in radians.
            power_ratio: Normalized shot charge [0.0, 1.0].
            is_charging: Whether the player is currently charging a shot.
        """
        if cue_ball.state != BallState.STATIONARY:
            return

        surface = self._display.screen
        ball_screen = self._display.world_to_screen(cue_ball.pos)

        # 1. Aiming trajectory guideline (pointing forward)
        guide_length = self._display.scale_scalar(300.0)
        aim_dir_x = -math.cos(aim_angle)
        aim_dir_y = -math.sin(aim_angle)

        guide_end = (
            int(ball_screen[0] + aim_dir_x * guide_length),
            int(ball_screen[1] + aim_dir_y * guide_length),
        )
        pygame.draw.line(surface, (255, 255, 120), ball_screen, guide_end, 1)

        # 2. Retracted Cue Stick (drawn behind the ball)
        pullback_distance = self._display.scale_scalar(20.0 + (power_ratio * 70.0))
        stick_length = self._display.scale_scalar(240.0)
        thickness = self._display.scale_scalar(6.0)

        # Vector pointing away from shot heading
        back_dir_x = math.cos(aim_angle)
        back_dir_y = math.sin(aim_angle)

        tip_x = ball_screen[0] + back_dir_x * pullback_distance
        tip_y = ball_screen[1] + back_dir_y * pullback_distance
        butt_x = tip_x + back_dir_x * stick_length
        butt_y = tip_y + back_dir_y * stick_length

        # Main shaft (Wood)
        pygame.draw.line(surface, (170, 110, 60), (tip_x, tip_y), (butt_x, butt_y), thickness)
        # Cue tip (White chalk end)
        cue_tip_length = self._display.scale_scalar(10.0)
        tip_end_x = tip_x + back_dir_x * cue_tip_length
        tip_end_y = tip_y + back_dir_y * cue_tip_length
        pygame.draw.line(surface, (230, 240, 255), (tip_x, tip_y), (tip_end_x, tip_end_y), thickness)