"""Heads-Up Display: renders power gauge, trick-shot stats, gesture badges, and FPS."""

from __future__ import annotations

import pygame

from endless_billiards.rendering.display import DisplayManager


class HUD:
    """Overlays game telemetry and input recognition feedback onto the active surface."""

    __slots__ = ("_display", "_font_small", "_font_large")

    def __init__(self, display: DisplayManager) -> None:
        self._display: DisplayManager = display
        pygame.font.init()
        self._font_small = pygame.font.SysFont("Courier", 15, bold=True)
        self._font_large = pygame.font.SysFont("Arial", 30, bold=True)

    def draw(
        self,
        score: int,
        streak: int,
        power: float,
        is_charging: bool,
        gesture_name: str,
        fps: float,
    ) -> None:
        """Render HUD metrics and dynamic power charge bar."""
        surface = self._display.screen

        # 1. Top Header: Score & Streak
        score_surface = self._font_large.render(f"SCORE: {score:,}", True, (245, 245, 245))
        streak_surface = self._font_small.render(f"STREAK: x{streak}", True, (255, 175, 50))
        surface.blit(score_surface, (28, 22))
        surface.blit(streak_surface, (30, 60))

        # 2. Gesture and Engine Telemetry Badge
        badge_bg = pygame.Rect(26, 88, 210, 62)
        pygame.draw.rect(surface, (20, 22, 28), badge_bg, border_radius=6)
        pygame.draw.rect(surface, (50, 55, 65), badge_bg, width=1, border_radius=6)

        color_map = {
            "OPEN": (120, 200, 255),
            "PINCHING": (255, 200, 40),
            "FLICK": (255, 80, 80),
            "STEADY": (80, 240, 140),
            "SEARCHING": (160, 160, 160),
        }
        badge_color = color_map.get(gesture_name, (180, 180, 180))

        gesture_lbl = self._font_small.render(f"HAND: {gesture_name}", True, badge_color)
        fps_lbl = self._font_small.render(f"FPS : {fps:04.1f}", True, (160, 170, 180))
        surface.blit(gesture_lbl, (36, 96))
        surface.blit(fps_lbl, (36, 122))

        # 3. Dynamic Shot Power Gauge (Bottom Screen Center)
        gauge_width = self._display.scale_scalar(360.0)
        gauge_height = self._display.scale_scalar(18.0)
        gauge_x = (surface.get_width() - gauge_width) // 2
        gauge_y = surface.get_height() - gauge_height - 28

        bg_bar = pygame.Rect(gauge_x, gauge_y, gauge_width, gauge_height)
        pygame.draw.rect(surface, (25, 27, 32), bg_bar, border_radius=5)

        fill_pixels = max(0, min(gauge_width, int(gauge_width * power)))
        if fill_pixels > 0:
            fill_color = (255, 75, 45) if power > 0.85 else (255, 195, 35)
            fill_bar = pygame.Rect(gauge_x, gauge_y, fill_pixels, gauge_height)
            pygame.draw.rect(surface, fill_color, fill_bar, border_radius=5)

        # Border outline
        pygame.draw.rect(surface, (110, 115, 130), bg_bar, width=2, border_radius=5)

        # Power text label
        pct_label = self._font_small.render(
            f"POWER {int(power * 100)}%",
            True,
            (230, 230, 230) if is_charging else (140, 140, 140),
        )
        surface.blit(pct_label, (gauge_x + (gauge_width - pct_label.get_width()) // 2, gauge_y - 22))