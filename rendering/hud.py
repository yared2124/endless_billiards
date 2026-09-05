
from __future__ import annotations

import pygame

from endless_billiards.rendering.display import DisplayManager


class HUD:
    """Renders textual indicators, charge meters, and camera gesture debugging info."""

    __slots__ = ("_display", "_font_small", "_font_large")

    def __init__(self, display: DisplayManager) -> None:
        """Initialize HUD fonts and layout metrics.

        Args:
            display: Active display subsystem.
        """
        self._display: DisplayManager = display
        pygame.font.init()
        self._font_small = pygame.font.SysFont("monospace", 16, bold=True)
        self._font_large = pygame.font.SysFont("sans-serif", 32, bold=True)

    def draw(
        self,
        score: int,
        streak: int,
        power: float,
        is_charging: bool,
        current_gesture: str,
        fps: float,
    ) -> None:
        """Render HUD elements onto the display surface.

        Args:
            score: Current game score.
            streak: Consecutive trick-shot multiplier count.
            power: Normalized charge value in [0.0, 1.0].
            is_charging: Whether the shot power is accumulating.
            current_gesture: Identifier string of the detected hand pose.
            fps: Current measured runtime framerate.
        """
        surface = self._display.screen

        # 1. Score and Streak Top Bar
        score_txt = self._font_large.render(f"SCORE: {score:,}", True, (240, 240, 240))
        streak_txt = self._font_small.render(f"STREAK: x{streak}", True, (255, 180, 40))
        surface.blit(score_txt, (30, 20))
        surface.blit(streak_txt, (32, 60))

        # 2. Performance / Gesture State Overlay
        gesture_color = (80, 220, 100) if current_gesture != "UNKNOWN" else (180, 80, 80)
        status_txt = self._font_small.render(f"GESTURE: {current_gesture}", True, gesture_color)
        fps_txt = self._font_small.render(f"FPS: {fps:.0f}", True, (140, 140, 140))
        surface.blit(status_txt, (30, 85))
        surface.blit(fps_txt, (30, 110))

        # 3. Dynamic Power Gauge (Bottom Center)
        bar_w = self._display.scale_scalar(300.0)
        bar_h = self._display.scale_scalar(16.0)
        bar_x = (surface.get_width() - bar_w) // 2
        bar_y = surface.get_height() - bar_h - 25

        # Background slot
        bg_rect = pygame.Rect(bar_x, bar_y, bar_w, bar_h)
        pygame.draw.rect(surface, (30, 30, 35), bg_rect, border_radius=4)

        # Active charge fill
        fill_width = max(0, min(bar_w, int(bar_w * power)))
        if fill_width > 0:
            fill_color = (255, 200, 30) if is_charging else (100, 220, 255)
            fill_rect = pygame.Rect(bar_x, bar_y, fill_width, bar_h)
            pygame.draw.rect(surface, fill_color, fill_rect, border_radius=4)

        # Outer frame
        pygame.draw.rect(surface, (120, 120, 130), bg_rect, width=2, border_radius=4)

        # Meter label
        label_txt = self._font_small.render("POWER", True, (200, 200, 200))
        surface.blit(
            label_txt,
            (bar_x + (bar_w - label_txt.get_width()) // 2, bar_y - 20),
        )