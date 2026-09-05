"""Pygame display manager handling window sizing, aspect scaling, and camera projection."""

from __future__ import annotations

import pygame

from endless_billiards.config.constants import (
    LOGICAL_HEIGHT,
    LOGICAL_WIDTH,
    TABLE_MAX_X,
    TABLE_MAX_Y,
    TABLE_MIN_X,
    TABLE_MIN_Y,
)
from endless_billiards.utils.math2d import Vector2


class DisplayManager:
    """Manages the OS window surface and transforms 1920x1080 logical coordinates to pixels."""

    __slots__ = (
        "_screen",
        "_clock",
        "_window_width",
        "_window_height",
        "_target_fps",
        "_scale",
        "_offset_x",
        "_offset_y",
    )

    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 720,
        target_fps: int = 60,
    ) -> None:
        """Initialize the Pygame video surface with letterbox projection.

        Args:
            window_width: Initial window pixel width.
            window_height: Initial window pixel height.
            target_fps: Frame limiter ceiling.
        """
        pygame.init()
        pygame.display.set_caption("Endless Billiards — Hand-Gesture Controller")

        self._target_fps: int = target_fps
        self._window_width: int = window_width
        self._window_height: int = window_height
        self._screen: pygame.Surface = pygame.display.set_mode(
            (window_width, window_height), pygame.RESIZABLE | pygame.DOUBLEBUF
        )
        self._clock: pygame.time.Clock = pygame.time.Clock()

        self._scale: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._recalculate_aspect_ratio()

    @property
    def screen(self) -> pygame.Surface:
        """Active frame surface."""
        return self._screen

    @property
    def scale(self) -> float:
        """Current scalar multiplier from logical space to screen pixels."""
        return self._scale

    def _recalculate_aspect_ratio(self) -> None:
        """Compute pillarbox/letterbox scaling for the 16:9 target logical space."""
        aspect_target = LOGICAL_WIDTH / LOGICAL_HEIGHT
        aspect_current = self._window_width / max(1, self._window_height)

        if aspect_current > aspect_target:
            self._scale = self._window_height / LOGICAL_HEIGHT
            self._offset_x = (self._window_width - (LOGICAL_WIDTH * self._scale)) * 0.5
            self._offset_y = 0.0
        else:
            self._scale = self._window_width / LOGICAL_WIDTH
            self._offset_x = 0.0
            self._offset_y = (self._window_height - (LOGICAL_HEIGHT * self._scale)) * 0.5

    def handle_resize(self, width: int, height: int) -> None:
        """Recalculate projection offsets on window resize events."""
        self._window_width = max(640, width)
        self._window_height = max(360, height)
        self._screen = pygame.display.set_mode(
            (self._window_width, self._window_height),
            pygame.RESIZABLE | pygame.DOUBLEBUF,
        )
        self._recalculate_aspect_ratio()

    def world_to_screen(self, pos: Vector2) -> tuple[int, int]:
        """Project continuous logical units to discrete screen pixel coordinates."""
        return (
            int(pos.x * self._scale + self._offset_x),
            int(pos.y * self._scale + self._offset_y),
        )

    def screen_to_world(self, screen_pos: tuple[int, int]) -> Vector2:
        """Inverse-project screen pixel coordinates back into 1920x1080 logical units."""
        return Vector2(
            (screen_pos[0] - self._offset_x) / self._scale,
            (screen_pos[1] - self._offset_y) / self._scale,
        )

    def scale_scalar(self, distance: float) -> int:
        """Project a scalar distance (radius, line thickness) to pixel dimensions."""
        return max(1, int(distance * self._scale))

    def get_table_screen_rect(self) -> pygame.Rect:
        """Get the screen-space bounding rectangle for the inner felt surface."""
        tl = self.world_to_screen(Vector2(TABLE_MIN_X, TABLE_MIN_Y))
        br = self.world_to_screen(Vector2(TABLE_MAX_X, TABLE_MAX_Y))
        return pygame.Rect(tl[0], tl[1], br[0] - tl[0], br[1] - tl[1])

    def clear(self) -> None:
        """Fill letterbox boundaries with background tone."""
        self._screen.fill((12, 13, 15))

    def present(self) -> float:
        """Flip double buffers and throttle frame time. Returns delta time in seconds."""
        pygame.display.flip()
        return self._clock.tick(self._target_fps) / 1000.0

    def cleanup(self) -> None:
        """Cleanly exit video subsystems."""
        pygame.quit()