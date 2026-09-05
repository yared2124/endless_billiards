"""Background threading loop capturing OpenCV frames and running MediaPipe hand tracking."""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError:
    mp = None

logger = logging.getLogger("endless_billiards.vision.tracker")


@dataclass(slots=True)
class HandTrackingFrame:
    """Encapsulates raw frame data and detected hand landmarks."""

    landmarks: Optional[list[Any]]
    image: Optional[np.ndarray]
    timestamp: float


class HandTracker:
    """Asynchronous background worker capturing camera frames and extracting hand landmarks."""

    __slots__ = (
        "_camera_index",
        "_width",
        "_height",
        "_output_queue",
        "_running",
        "_thread",
        "_cap",
    )

    def __init__(
        self,
        camera_index: int = 0,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self._camera_index: int = camera_index
        self._width: int = width
        self._height: int = height
        self._output_queue: queue.Queue[HandTrackingFrame] = queue.Queue(maxsize=2)
        self._running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._cap: Optional[cv2.VideoCapture] = None

    @property
    def output_queue(self) -> queue.Queue[HandTrackingFrame]:
        """Thread-safe queue providing latest hand tracking payloads."""
        return self._output_queue

    def start(self) -> None:
        """Spawn background capture thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, name="MediaPipeWorkerThread", daemon=True)
        self._thread.start()
        logger.info("HandTracker background thread initialized.")

    def _worker_loop(self) -> None:
        """Continuous capture and inference loop running in background thread."""
        self._cap = cv2.VideoCapture(self._camera_index)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)

        if not self._cap.isOpened():
            logger.error(f"Failed to open video capture device index {self._camera_index}.")
            self._running = False
            return

        hands_detector = None
        if mp is not None and hasattr(mp, "solutions") and hasattr(mp.solutions, "hands"):
            mp_hands = mp.solutions.hands
            hands_detector = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.65,
                min_tracking_confidence=0.65,
            )
        else:
            logger.warning("MediaPipe solutions API not found. Hand tracking will run in dummy mode.")

        import time

        try:
            while self._running:
                success, frame = self._cap.read()
                if not success:
                    time.sleep(0.01)
                    continue

                # Flip frame horizontally for natural mirror feel
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                landmarks = None
                if hands_detector is not None:
                    results = hands_detector.process(rgb_frame)
                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0].landmark

                tracking_frame = HandTrackingFrame(
                    landmarks=landmarks,
                    image=frame,
                    timestamp=time.time(),
                )

                if self._output_queue.full():
                    try:
                        self._output_queue.get_nowait()
                    except queue.Empty:
                        pass
                self._output_queue.put(tracking_frame)

        except Exception as e:
            logger.error(f"Error in hand tracker worker loop: {e}")
        finally:
            if hands_detector is not None:
                hands_detector.close()
            if self._cap is not None:
                self._cap.release()
            logger.info("HandTracker stopped.")

    def stop(self) -> None:
        """Signal background thread termination and join execution."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.5)
            self._thread = None