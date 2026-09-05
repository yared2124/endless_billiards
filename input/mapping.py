from __future__ import annotations

import math
from typing import Optional

from endless_billiards.config.settings import Settings
from endless_billiards.input.events import (
    AimPayload,
    EventBus,
    InputEvent,
    PowerPayload,
    ShotFiredPayload,
)
from endless_billiards.input.gestures import Gesture, GestureClassifier
from endless_billiards.vision.landmarks import GestureNormalizer
from endless_billiards.vision.tracker import HandTrackingFrame


class AngleEMAFilter:
    """5-frame Exponential Moving Average filter handling periodic circular data [-pi, pi]."""

    __slots__ = ("_alpha", "_sin_ema", "_cos_ema", "_initialized")

    def __init__(self, window_size: int = 5) -> None:
        """Initialize filter with weighting constant derived from window size.

        Args:
            window_size: Number of frames representing the smoothing horizon.
        """
        # Standard EMA smoothing factor: 2 / (N + 1)
        self._alpha: float = 2.0 / (window_size + 1.0)
        self._sin_ema: float = 0.0
        self._cos_ema: float = 0.0
        self._initialized: bool = False

    def update(self, angle: float) -> float:
        """Apply EMA smoothing over vector components to safely handle radians wrap-around.

        Args:
            angle: Raw angle in radians [-pi, pi].

        Returns:
            Jitter-free smoothed angle in radians [-pi, pi].
        """
        sin_val = math.sin(angle)
        cos_val = math.cos(angle)

        if not self._initialized:
            self._sin_ema = sin_val
            self._cos_ema = cos_val
            self._initialized = True
        else:
            self._sin_ema = self._alpha * sin_val + (1.0 - self._alpha) * self._sin_ema
            self._cos_ema = self._alpha * cos_val + (1.0 - self._alpha) * self._cos_ema

        return math.atan2(self._sin_ema, self._cos_ema)

    def reset(self) -> None:
        """Reset internal accumulator registers."""
        self._sin_ema = 0.0
        self._cos_ema = 0.0
        self._initialized = False


class ControlMapper:
    """Translates normalized tracking data and gestures into actionable game commands."""

    __slots__ = (
        "_event_bus",
        "_settings",
        "_normalizer",
        "_classifier",
        "_angle_filter",
        "_current_angle",
        "_current_power",
        "_is_charging",
    )

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        normalizer: GestureNormalizer,
        classifier: GestureClassifier,
    ) -> None:
        """Initialize mapper state machines and filters.

        Args:
            event_bus: Central event bus publisher target.
            settings: Runtime settings for scale thresholds.
            normalizer: Geometry coordinate normalizer.
            classifier: Hand gesture pose classifier.
        """
        self._event_bus: EventBus = event_bus
        self._settings: Settings = settings
        self._normalizer: GestureNormalizer = normalizer
        self._classifier: GestureClassifier = classifier

        self._angle_filter: AngleEMAFilter = AngleEMAFilter(window_size=5)
        self._current_angle: float = 0.0
        self._current_power: float = 0.0
        self._is_charging: bool = False

    def process_frame(self, frame: Optional[HandTrackingFrame]) -> None:
        """Ingest tracking frames, apply filters, and dispatch events.

        Args:
            frame: Hand landmarks container, or None if tracking is lost.
        """
        if frame is None:
            if self._is_charging:
                # Cancel charging if tracking is abruptly dropped
                self._is_charging = False
                self._current_power = 0.0
                self._event_bus.post(InputEvent.POWER_CHANGED, PowerPayload(power=0.0))
            return

        gesture = self._classifier.classify(frame)

        # 1. Process Aim Angle (Active on STEADY or OPEN gestures)
        if gesture in (Gesture.STEADY, Gesture.OPEN):
            raw_angle = self._normalizer.get_cue_angle(frame)
            smoothed_angle = self._angle_filter.update(raw_angle)
            self._current_angle = smoothed_angle
            self._event_bus.post(InputEvent.AIM_CHANGED, AimPayload(angle=smoothed_angle))

        # 2. Process Power Charging (PINCHING gesture)
        if gesture == Gesture.PINCHING:
            self._is_charging = True
            # Distance: 0.0 (fingers tight) to 1.0 (fingers spread)
            pinch_dist = self._normalizer.get_pinch_distance(frame)
            # Power scales inversely: tight pinch -> maximum pull-back charge
            power_factor = 1.0 - pinch_dist
            self._current_power = max(0.0, min(1.0, power_factor))
            self._event_bus.post(
                InputEvent.POWER_CHANGED, PowerPayload(power=self._current_power)
            )

        # 3. Fire Cue Shot (FLICK gesture)
        elif gesture == Gesture.FLICK and self._is_charging:
            # Only trigger shot if non-zero power was charged
            if self._current_power > 0.05:
                self._event_bus.post(
                    InputEvent.SHOT_FIRED,
                    ShotFiredPayload(
                        angle=self._current_angle,
                        power=self._current_power,
                    ),
                )
            # Reset charging status post shot
            self._is_charging = False
            self._current_power = 0.0
            self._event_bus.post(InputEvent.POWER_CHANGED, PowerPayload(power=0.0))

        # 4. Handle Passive Pinch Cancel (if relaxed without flick velocity)
        elif gesture == Gesture.OPEN and self._is_charging:
            self._is_charging = False
            self._current_power = 0.0
            self._event_bus.post(InputEvent.POWER_CHANGED, PowerPayload(power=0.0))

    def reset(self) -> None:
        """Reset state tracking components and historical filters."""
        self._classifier.reset()
        self._normalizer.reset()
        self._angle_filter.reset()
        self._is_charging = False
        self._current_power = 0.0