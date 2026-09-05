"""Session logging recorder and playback diagnostic harness for trick-shot replays."""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class ReplayFrame:
    """Timestamped snapshot of player input telemetry."""

    time_offset: float
    aim_angle: float
    power: float
    gesture: str
    is_shot: bool


class ReplaySystem:
    """Records gameplay session inputs to disk for deterministic diagnostics."""

    __slots__ = ("_records", "_start_time", "_recording")

    def __init__(self) -> None:
        self._records: list[ReplayFrame] = []
        self._start_time: float = 0.0
        self._recording: bool = False

    def start_recording(self) -> None:
        """Begin capturing input events."""
        self._records.clear()
        self._start_time = time.monotonic()
        self._recording = True

    def record_frame(
        self,
        aim_angle: float,
        power: float,
        gesture: str,
        is_shot: bool = False,
    ) -> None:
        """Append an input snapshot."""
        if not self._recording:
            return

        self._records.append(
            ReplayFrame(
                time_offset=time.monotonic() - self._start_time,
                aim_angle=aim_angle,
                power=power,
                gesture=gesture,
                is_shot=is_shot,
            )
        )

    def save_to_disk(self, output_path: str = "assets/replays/last_session.json") -> None:
        """Serialize logged frames into a JSON payload."""
        self._recording = False
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = [asdict(f) for f in self._records]
        with path.open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)

    @classmethod
    def load_replay(cls, input_path: str) -> list[ReplayFrame]:
        """Read and deserialize recorded session frames."""
        path = Path(input_path)
        if not path.exists():
            return []

        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)

        return [ReplayFrame(**item) for item in data]