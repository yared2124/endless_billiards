
from __future__ import annotations

import queue
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
from typing import Callable, Generic, TypeAlias, TypeVar


class InputEvent(Enum):
    """Enumeration of all discrete and continuous input event triggers."""

    AIM_CHANGED = auto()
    POWER_CHANGED = auto()
    SHOT_FIRED = auto()
    RESET_REQUESTED = auto()


@dataclass(frozen=True, slots=True)
class AimPayload:
    """Payload carrying directional aim information.

    Attributes:
        angle: The targeted cue angle in radians.
    """

    angle: float


@dataclass(frozen=True, slots=True)
class PowerPayload:
    """Payload carrying shot impulse power metrics.

    Attributes:
        power: Normalized shot charge factor clamped between [0.0, 1.0].
    """

    power: float


@dataclass(frozen=True, slots=True)
class ShotFiredPayload:
    """Payload emitted when a shot execution sequence is finalized.

    Attributes:
        angle: Launch heading in radians.
        power: Normalized power multiplier clamped between [0.0, 1.0].
    """

    angle: float
    power: float


@dataclass(frozen=True, slots=True)
class ResetPayload:
    """Payload emitted when the game/table state requires an explicit reset.

    Attributes:
        timestamp_ms: Timestamp representing when the reset request occurred.
    """

    timestamp_ms: int


# Type boundary for allowed event payload schemas
PayloadT = TypeVar(
    "PayloadT",
    AimPayload,
    PowerPayload,
    ShotFiredPayload,
    ResetPayload,
)

EventCallback: TypeAlias = Callable[[PayloadT], None]


@dataclass(frozen=True, slots=True)
class _QueuedEvent(Generic[PayloadT]):
    """Internal message envelope passed across thread boundaries."""

    event_type: InputEvent
    payload: PayloadT


class EventBus:
    """Thread-safe, decoupled publisher/subscriber event dispatch pipeline.

    Asynchronous threads (e.g., OpenCV/MediaPipe capture pipelines) can invoke
    `post()` safely. Handlers are invoked synchronously on the consumer thread
    during `process_queue()` calls, avoiding game-loop concurrency bugs.
    """

    __slots__ = ("_queue", "_subscribers")

    def __init__(self) -> None:
        """Initialize event ingestion queue and subscriber dispatch map."""
        self._queue: queue.Queue[_QueuedEvent[PayloadT]] = queue.Queue()
        self._subscribers: dict[InputEvent, list[Callable[[PayloadT], None]]] = (
            defaultdict(list)
        )

    def subscribe(
        self,
        event_type: InputEvent,
        callback: Callable[[PayloadT], None],
    ) -> None:
        """Register a callback for an event type.

        Subscriptions should be performed on the main thread during game setup.

        Args:
            event_type: The enum variant to subscribe to.
            callback: Function invoked with the corresponding typed payload.
        """
        self._subscribers[event_type].append(callback)

    def unsubscribe(
        self,
        event_type: InputEvent,
        callback: Callable[[PayloadT], None],
    ) -> None:
        """Remove a registered callback from an event type.

        Args:
            event_type: The enum variant to detach from.
            callback: The target callback to remove.
        """
        handlers = self._subscribers.get(event_type)
        if handlers and callback in handlers:
            handlers.remove(callback)

    def post(self, event_type: InputEvent, payload: PayloadT) -> None:
        """Enqueue an event from any thread (producer).

        This operation is non-blocking and thread-safe.

        Args:
            event_type: The category of the input event being emitted.
            payload: Concrete typed data container matching the event definition.
        """
        self._queue.put_nowait(_QueuedEvent(event_type=event_type, payload=payload))

    def process_queue(self) -> int:
        """Drain the queue and dispatch events to subscribers (consumer).

        This method must be called once per simulation or render cycle
        from the main execution thread.

        Returns:
            The total count of events processed in this frame cycle.
        """
        processed_count = 0

        while True:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break

            handlers = self._subscribers.get(item.event_type, [])
            for handler in handlers:
                handler(item.payload)

            self._queue.task_done()
            processed_count += 1

        return processed_count

    def clear(self) -> None:
        """Flush any pending messages from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except queue.Empty:
                break