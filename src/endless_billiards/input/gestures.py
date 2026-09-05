
from __future__ import annotations

import collections
import math
from enum import Enum, auto
from typing import Deque

from endless_billiards.config.settings import Settings
from endless_billiards.vision.landmarks import GestureNormalizer
from endless_billiards.vision.tracker import HandTrackingFrame


class Gesture(Enum):
    """Enumeration of recognized user hand poses."""

    OPEN = auto()
    PINCHING = auto()
    FLICK = auto()
    STEADY = auto()


class GestureClassifier:
    """Classifies temporal hand tracking landmark sequences into discrete gestures."""

    __slots__ = (
        "_settings",
        "_normalizer",
        "_pinch_history",
        "_wrist_history",
        "_is_pinching",
    )

    def __init__(self, settings: Settings, normalizer: GestureNormalizer) -> None:
        """Initialize the gesture classifier.

        Args:
            settings: Runtime user and threshold configurations.
            normalizer: Landmark converter used to determine pinch metrics.
        """
        self._settings: Settings = settings
        self._normalizer: GestureNormalizer = normalizer
        # Maintain short history (approx. 5 frames) for velocity/flick evaluation
        self._pinch_history: Deque[float] = collections.deque(maxlen=5)
        self._wrist_history: Deque[tuple[float, float]] = collections.deque(maxlen=5)
        self._is_pinching: bool = False

    def classify(self, frame: HandTrackingFrame) -> Gesture:
        """Evaluate landmark configurations and return the dominant gesture.

        Classification Pipeline:
            1. Detect if a pinch release transition (FLICK) occurred.
            2. Detect active pinching state (PINCHING).
            3. Differentiate between stationary orientation (STEADY) and broad hand movement (OPEN).

        Args:
            frame: Real-time hand tracking landmark payload.

        Returns:
            The determined Gesture enum variant.
        """
        pinch_dist = self._normalizer.get_pinch_distance(frame)
        self._pinch_history.append(pinch_dist)
        self._wrist_history.append((frame.wrist.x, frame.wrist.y))

        # Check if pinching (within normalized threshold bounds)
        is_currently_pinched = pinch_dist <= self._settings.pinch_threshold

        # 1. Evaluate Flick/Shot execution
        # Occurs when moving abruptly from a pinched state to a released state
        if self._is_pinching and not is_currently_pinched:
            # Check expansion velocity: delta distance across buffer
            if len(self._pinch_history) >= 2:
                expansion_speed = pinch_dist - self._pinch_history[0]
                if expansion_speed > 0.15:  # Rapid release threshold
                    self._is_pinching = False
                    return Gesture.FLICK

        self._is_pinching = is_currently_pinched

        # 2. Evaluate Active Pinch
        if self._is_pinching:
            return Gesture.PINCHING

        # 3. Evaluate Wrist Jitter / Steady state
        if len(self._wrist_history) == self._wrist_history.maxlen:
            dx = self._wrist_history[-1][0] - self._wrist_history[0][0]
            dy = self._wrist_history[-1][1] - self._wrist_history[0][1]
            drift = math.hypot(dx, dy)

            # Hand is relatively fixed in space, allowing fine-tuned aiming
            if drift < self._settings.deadzone:
                return Gesture.STEADY

        return Gesture.OPEN

    def reset(self) -> None:
        """Reset historical temporal buffers."""
        self._pinch_history.clear()
        self._wrist_history.clear()
        self._is_pinching = False