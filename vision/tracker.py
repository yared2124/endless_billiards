
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import NamedTuple, Optional

import cv2
import mediapipe as mp

logger = logging.getLogger(__name__)


class NormalizedPoint(NamedTuple):
    """Normalized point representing screen-space coordinates in [0.0, 1.0]."""

    x: float
    y: float
    z: float


@dataclass(frozen=True, slots=True)
class HandTrackingFrame:
    """Snapshot containing extracted landmark keys for interaction processing.

    Attributes:
        timestamp_ms: System capture monotonic time in milliseconds.
        wrist: Normalized wrist position (Landmark 0).
        thumb_tip: Normalized thumb tip position (Landmark 4).
        index_tip: Normalized index finger tip position (Landmark 8).
        middle_mcp: Normalized middle finger MCP joint position (Landmark 9).
    """

    timestamp_ms: int
    wrist: NormalizedPoint
    thumb_tip: NormalizedPoint
    index_tip: NormalizedPoint
    middle_mcp: NormalizedPoint


class HandTracker:
    """Asynchronous video capture worker executing MediaPipe inference on a daemon thread.

    Thread safe: Emits frame packages to a thread-safe output queue while decoupling
    inference latency from the main game loop.
    """

    def __init__(
        self,
        camera_index: int = 0,
        output_queue_size: int = 2,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize tracker resources and setup execution thread.

        Args:
            camera_index: OpenCV VideoCapture device index.
            output_queue_size: Maximum buffer depth for outgoing tracking data.
            min_detection_confidence: Minimum hand detection threshold.
            min_tracking_confidence: Minimum landmark tracking threshold.
        """
        self._camera_index: int = camera_index
        self._min_detection_confidence: float = min_detection_confidence
        self._min_tracking_confidence: float = min_tracking_confidence

        self._queue: queue.Queue[Optional[HandTrackingFrame]] = queue.Queue(
            maxsize=output_queue_size
        )
        self._stop_event: threading.Event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    @property
    def output_queue(self) -> queue.Queue[Optional[HandTrackingFrame]]:
        """Access the thread-safe output queue."""
        return self._queue

    def start(self) -> None:
        """Start the background processing thread."""
        if self._thread is not None and self._thread.is_alive():
            logger.warning("HandTracker thread is already active.")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="MediaPipeWorkerThread",
            daemon=True,
        )
        self._thread.start()
        logger.info("HandTracker background thread initialized.")

    def stop(self) -> None:
        """Signal worker termination and join the capture thread."""
        if self._thread is None:
            return

        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        logger.info("HandTracker stopped.")

    def _worker_loop(self) -> None:
        """Core execution loop for capture and landmark estimation."""
        cap = cv2.VideoCapture(self._camera_index)
        if not cap.isOpened():
            logger.error("Unable to access system webcam index %d.", self._camera_index)
            return

        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=self._min_detection_confidence,
            min_tracking_confidence=self._min_tracking_confidence,
        )

        try:
            while not self._stop_event.is_set():
                success, frame = cap.read()
                if not success:
                    time.sleep(0.005)
                    continue

                # Mirror frame along horizontal axis for natural user interaction
                frame = cv2.flip(frame, 1)

                # MediaPipe requires RGB color format
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                timestamp_now = time.monotonic_ns() // 1_000_000

                if results.multi_hand_landmarks:
                    # Select primary hand
                    landmarks = results.multi_hand_landmarks[0].landmark

                    wrist_lm = landmarks[0]
                    thumb_lm = landmarks[4]
                    index_lm = landmarks[8]
                    middle_mcp_lm = landmarks[9]

                    data = HandTrackingFrame(
                        timestamp_ms=timestamp_now,
                        wrist=NormalizedPoint(wrist_lm.x, wrist_lm.y, wrist_lm.z),
                        thumb_tip=NormalizedPoint(thumb_lm.x, thumb_lm.y, thumb_lm.z),
                        index_tip=NormalizedPoint(index_lm.x, index_lm.y, index_lm.z),
                        middle_mcp=NormalizedPoint(
                            middle_mcp_lm.x, middle_mcp_lm.y, middle_mcp_lm.z
                        ),
                    )
                    self._enqueue_latest(data)
                else:
                    self._enqueue_latest(None)

        finally:
            hands.close()
            cap.release()

    def _enqueue_latest(self, item: Optional[HandTrackingFrame]) -> None:
        """Enqueue tracking frame, discarding stale items if queue is saturated."""
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                _ = self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(item)