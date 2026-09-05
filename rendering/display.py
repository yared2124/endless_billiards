
from __future__ import annotations

import pygame

from endless_billiards.config.constants import LOGICAL_HEIGHT, LOGICAL_WIDTH
from endless_billiards.utils.math2d import Vector2


class DisplayManager:
    """Manages window lifecycle, coordinate transformations, and display refreshes."""

    __slots__ = (
        "_target_fps",
        "_screen",
        "_clock",
        "_window_width",
        "_window_height",
        "_scale_x",
        "_scale_y",
        "_offset_x",
        "_offset_y",
    )

    def __init__(
        self,
        window_width: int = 1280,
        window_height: int = 720,
        target_fps: int = 60,
    ) -> None:
        """Initialize the Pygame display context with letterboxed aspect ratio support.

        Args:
            window_width: Physical window width in pixels.
            window_height: Physical window height in pixels.
            target_fps: Frame rate limiter frequency.
        """
        pygame.init()
        pygame.display.set_caption("Endless Billiards - Hand Tracking Arcade")

        self._target_fps: int = target_fps
        self._window_width: int = window_width
        self._window_height: int = window_height
        self._screen: pygame.Surface = pygame.display.set_mode(
            (window_width, window_height), pygame.RESIZABLE | pygame.DOUBLEBUF
        )
        self._clock: pygame.time.Clock = pygame.time.Clock()

        self._scale_x: float = 1.0
        self._scale_y: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._update_aspect_scaling()

    @property
    def screen(self) -> pygame.Surface:
        """Get the active display surface."""
        return self._screen

    def _update_aspect_scaling(self) -> None:
        """Calculate letterbox offset and uniform scaling factor."""
        aspect_logical = LOGICAL_WIDTH / LOGICAL_HEIGHT
        aspect_window = self._window_width / max(1, self._window_height)

        if aspect_window > aspect_logical:
            # Window is wider than the target aspect ratio: pillarbox
            scale = self._window_height / LOGICAL_HEIGHT
            self._offset_x = (self._window_width - (LOGICAL_WIDTH * scale)) * 0.5
            self._offset_y = 0.0
        else:
            # Window is taller than the target aspect ratio: letterbox
            scale = self._window_width / LOGICAL_WIDTH
            self._offset_x = 0.0
            self._offset_y = (self._window_height - (LOGICAL_HEIGHT * scale)) * 0.5

        self._scale_x = scale
        self._scale_y = scale

    def handle_resize(self, width: int, height: int) -> None:
        """Update window metrics upon receiving a VIDEORESIZE event.

        Args:
            width: New window width.
            height: New window height.
        """
        self._window_width = max(320, width)
        self._window_height = max(180, height)
        self._screen = pygame.display.set_mode(
            (self._window_width, self._window_height),
            pygame.RESIZABLE | pygame.DOUBLEBUF,
        )
        self._update_aspect_scaling()

    def world_to_screen(self, pos: Vector2) -> tuple[int, int]:
        """Transform simulation space coordinates to window pixel coordinates.

        Args:
            pos: Coordinate in continuous logical units (1920x1080 space).

        Returns:
            Tuple of screen-space pixel coordinates (x, y).
        """
        sx = int(pos.x * self._scale_x + self._offset_x)
        sy = int(pos.y * self._scale_y + self._offset_y)
        return sx, sy

    def scale_scalar(self, distance: float) -> int:
        """Scale a world-space scalar (e.g., radius) to pixel dimensions.

        Args:
            distance: Magnitude in logical units.

        Returns:
            Screen-scaled pixel magnitude.
        """
        return max(1, int(distance * self._scale_x))

    def begin_frame(self) -> None:
        """Clear the background buffer for frame composition."""
        self._screen.fill((15, 15, 18))

    def end_frame(self) -> float:
        """Flip buffers and throttle execution to the target framerate.

        Returns:
            Delta time in seconds since the last frame.
        """
        pygame.display.flip()
        return self._clock.tick(self._target_fps) / 1000.0

    def cleanup(self) -> None:
        """Deinitialize Pygame resources."""
        pygame.quit()