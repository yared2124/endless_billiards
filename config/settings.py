from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class Settings:
    """Mutable runtime configuration parameters.

    Attributes:
        gesture_sensitivity: Multiplier for hand-tracking aiming rotation speed.
        deadzone: Normalized threshold [0.0, 1.0] under which landmark jitter is ignored.
        audio_volume: Master sound volume clamped between 0.0 and 1.0.
        pinch_threshold: Normalized hand landmark distance determining a shot charge action.
        show_debug_overlays: Flag enabling collision hitboxes and vector rendering.
    """

    gesture_sensitivity: float = 1.25
    deadzone: float = 0.03
    audio_volume: float = 0.80
    pinch_threshold: float = 0.06
    show_debug_overlays: bool = False

    def validate(self) -> None:
        """Validate bounded attributes to prevent unstable simulation states.

        Raises:
            ValueError: If a parameter is out of its safe numeric range.
        """
        if not (0.1 <= self.gesture_sensitivity <= 5.0):
            raise ValueError(f"gesture_sensitivity must be in [0.1, 5.0], got {self.gesture_sensitivity}")
        if not (0.0 <= self.deadzone <= 0.5):
            raise ValueError(f"deadzone must be in [0.0, 0.5], got {self.deadzone}")
        if not (0.0 <= self.audio_volume <= 1.0):
            raise ValueError(f"audio_volume must be in [0.0, 1.0], got {self.audio_volume}")
        if not (0.01 <= self.pinch_threshold <= 0.2):
            raise ValueError(f"pinch_threshold must be in [0.01, 0.2], got {self.pinch_threshold}")

    @classmethod
    def from_json(cls, file_path: Path | str) -> Self:
        """Load and deserialize settings from a JSON file path.

        Falls back to default instance values if the file is missing or corrupt.

        Args:
            file_path: Filesystem path to configuration file.

        Returns:
            A validated Settings instance.
        """
        path = Path(file_path)
        if not path.exists():
            logger.warning("Configuration file %s does not exist. Using defaults.", path)
            return cls()

        try:
            with path.open("r", encoding="utf-8") as stream:
                data = json.load(stream)

            if not isinstance(data, dict):
                logger.error("Invalid JSON format in %s. Expected key-value object.", path)
                return cls()

            instance = cls(
                gesture_sensitivity=float(data.get("gesture_sensitivity", 1.25)),
                deadzone=float(data.get("deadzone", 0.03)),
                audio_volume=float(data.get("audio_volume", 0.80)),
                pinch_threshold=float(data.get("pinch_threshold", 0.06)),
                show_debug_overlays=bool(data.get("show_debug_overlays", False)),
            )
            instance.validate()
            return instance

        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            logger.error("Failed to parse settings file %s: %s. Using defaults.", path, exc)
            return cls()

    def to_json(self, file_path: Path | str) -> None:
        """Serialize current settings to disk in JSON format.

        Args:
            file_path: Output target destination.
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.validate()

        with path.open("w", encoding="utf-8") as stream:
            json.dump(asdict(self), stream, indent=4)