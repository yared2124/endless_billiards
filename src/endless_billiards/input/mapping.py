"""Control mapping layer translating classified hand gestures to EventBus events."""

from __future__ import annotations

import json
import math
from pathlib import Path
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
    """Translates gestures to game events with persistent zero-angle calibration."""

    __slots__ = (
        "_event_bus",
        "_settings",
        "_normalizer",
        "_classifier",
        "_angle_filter",
        "_current_angle",
        "_current_power",
        "_is_charging",
        "_calibration_mode",
        "_calibration_start_time",
        "_calibration_samples",
        "_aim_angle_offset",
        "_profile_path",
    )

    def __init__(
        self,
        event_bus: EventBus,
        settings: Settings,
        normalizer: GestureNormalizer,
        classifier: GestureClassifier,
        profile_path: Path | str = "src/endless_billiards/config/profiles/default.json",
    ) -> None:
        self._event_bus: EventBus = event_bus
        self._settings: Settings = settings
        self._normalizer: GestureNormalizer = normalizer
        self._classifier: GestureClassifier = classifier
        self._profile_path: Path = Path(profile_path)

        self._angle_filter: AngleEMAFilter = AngleEMAFilter(window_size=5)
        self._current_angle: float = 0.0
        self._current_power: float = 0.0
        self._is_charging: bool = False

        # Calibration state
        self._calibration_mode: bool = False
        self._calibration_start_time: float = 0.0
        self._calibration_samples: list[float] = []
        self._aim_angle_offset: float = self._load_profile_offset()

    def _load_profile_offset(self) -> float:
        """Load persistent aiming offset calibration from disk."""
        if not self._profile_path.exists():
            return 0.0
        try:
            with self._profile_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return float(data.get("aim_angle_offset", 0.0))
        except (json.JSONDecodeError, ValueError, KeyError):
            return 0.0

    def start_calibration(self) -> None:
        """Trigger interactive 2-second calibration mode."""
        self._calibration_mode = True
        self._calibration_start_time = time.monotonic() if 'time' in globals() else __import__('time').monotonic()
        self._calibration_samples.clear()

    @property
    def is_calibrating(self) -> bool:
        """Check if calibration mode is actively gathering samples."""
        return self._calibration_mode

    def _save_profile_offset(self, offset: float) -> None:
        """Persist calibrated angle offset to disk profile."""
        self._profile_path.parent.mkdir(parents=True, exist_ok=True)
        import time
        payload = {"aim_angle_offset": offset, "updated_at": time.time()}
        with self._profile_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4)

    def process_frame(self, frame: Optional[HandTrackingFrame]) -> None:
        """Process tracking frames with calibration interception and offset application."""
        if frame is None:
            if self._is_charging:
                self._is_charging = False
                self._current_power = 0.0
                self._event_bus.post(InputEvent.POWER_CHANGED, PowerPayload(power=0.0))
            return

        # 1. Intercept for Calibration Routine
        if self._calibration_mode:
            import time
            raw_angle = self._normalizer.get_cue_angle(frame)
            self._calibration_samples.append(raw_angle)

            elapsed = time.monotonic() - self._calibration_start_time
            if elapsed >= 2.0:
                if self._calibration_samples:
                    sin_sum = sum(math.sin(a) for a in self._calibration_samples)
                    cos_sum = sum(math.cos(a) for a in self._calibration_samples)
                    self._aim_angle_offset = math.atan2(sin_sum, cos_sum)
                    self._save_profile_offset(self._aim_angle_offset)

                self._calibration_mode = False
                self._calibration_samples.clear()
            return

        gesture = self._classifier.classify(frame)

        # 2. Aim Evaluation
        if gesture in (Gesture.STEADY, Gesture.OPEN):
            raw_angle = self._normalizer.get_cue_angle(frame)
            calibrated_angle = (
                raw_angle - self._aim_angle_offset + math.pi
            ) % (2.0 * math.pi) - math.pi

            smoothed_angle = self._angle_filter.update(calibrated_angle)
            self._current_angle = smoothed_angle
            self._event_bus.post(InputEvent.AIM_CHANGED, AimPayload(angle=smoothed_angle))

        # 3. Power Charging
        if gesture == Gesture.PINCHING:
            self._is_charging = True
            pinch_dist = self._normalizer.get_pinch_distance(frame)
            power_factor = 1.0 - pinch_dist
            self._current_power = max(0.0, min(1.0, power_factor))
            self._event_bus.post(
                InputEvent.POWER_CHANGED,
                PowerPayload(power=self._current_power),
            )

        # 4. Shot Release
        elif gesture == Gesture.FLICK and self._is_charging:
            if self._current_power > 0.05:
                self._event_bus.post(
                    InputEvent.SHOT_FIRED,
                    ShotFiredPayload(angle=self._current_angle, power=self._current_power),
                )
            self._is_charging = False
            self._current_power = 0.0
            self._event_bus.post(InputEvent.POWER_CHANGED, PowerPayload(power=0.0))

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