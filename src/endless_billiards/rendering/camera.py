"""Screen-shake camera system for heavy impacts and break shots."""

from __future__ import annotations

import random
from endless_billiards.rendering.display import DisplayManager


class CameraController:
    """Manages 2D screen trauma and decay to produce responsive physical rumble."""

    __slots__ = ("_display", "_trauma", "_base_offset_x", "_base_offset_y")

    def __init__(self, display: DisplayManager) -> None:
        self._display: DisplayManager = display
        self._trauma: float = 0.0
        self._base_offset_x: float = display._offset_x
        self._base_offset_y: float = display._offset_y

    def add_trauma(self, amount: float) -> None:
        """Inject trauma, clamping max accumulation to 1.0."""
        self._trauma = min(1.0, self._trauma + amount)

    def update(self, dt: float) -> None:
        """Decay trauma non-linearly and displace display projections."""
        # Baseline centering
        self._base_offset_x = self._display._offset_x
        self._base_offset_y = self._display._offset_y

        if self._trauma <= 0.0:
            return

        # Trauma squared creates punchier falloff
        shake_intensity = self._trauma * self._trauma
        max_jitter = self._display.scale_scalar(14.0)

        offset_x = max_jitter * shake_intensity * random.uniform(-1.0, 1.0)
        offset_y = max_jitter * shake_intensity * random.uniform(-1.0, 1.0)

        self._display._offset_x += offset_x
        self._display._offset_y += offset_y

        # Decay trauma linearly
        self._trauma = max(0.0, self._trauma - (1.8 * dt))

    def restore_offsets(self) -> None:
        """Reset offsets before next frame projection recalculations."""
        self._display._offset_x = self._base_offset_x
        self._display._offset_y = self._base_offset_y