"""."""

from __future__ import annotations

import logging
import math
import queue
import sys
from typing import Optional

import pygame

from endless_billiards.config.constants import (
    FIXED_TIMESTEP,
    LOGICAL_HEIGHT,
    LOGICAL_WIDTH,
    MAX_POWER,
    TABLE_MAX_X,
    TABLE_MAX_Y,
    TABLE_MIN_X,
    TABLE_MIN_Y,
)
from endless_billiards.config.settings import Settings
from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.table import Table
from endless_billiards.core.physics.engine import PhysicsEngine
from endless_billiards.input.events import (
    AimPayload,
    EventBus,
    InputEvent,
    PowerPayload,
    ShotFiredPayload,
)
from endless_billiards.input.gestures import GestureClassifier
from endless_billiards.input.mapping import ControlMapper
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.rendering.hud import HUD
from endless_billiards.rendering.sprites import BallSprite, CueSprite, TableSprite
from endless_billiards.utils.math2d import Vector2
from endless_billiards.vision.landmarks import GestureNormalizer
from endless_billiards.vision.tracker import HandTracker, HandTrackingFrame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("endless_billiards")


class GameOrchestrator:
    """Manages system lifecycles, event routing, and the fixed-step update loop."""

    __slots__ = (
        "_settings",
        "_event_bus",
        "_display",
        "_table",
        "_physics",
        "_balls",
        "_cue_ball",
        "_tracker",
        "_normalizer",
        "_classifier",
        "_mapper",
        "_table_sprite",
        "_ball_sprite",
        "_cue_sprite",
        "_hud",
        "_current_angle",
        "_current_power",
        "_is_charging",
        "_score",
        "_streak",
        "_running",
        "_detected_gesture",
    )

    def __init__(self) -> None:
        """Initialize all subsystems, models, pipelines, and bindings."""
        # 1. Configuration & Core Models
        self._settings = Settings()
        self._event_bus = EventBus()
        self._display = DisplayManager(window_width=1280, window_height=720, target_fps=60)

        self._table = Table()
        self._physics = PhysicsEngine(table=self._table, dt=FIXED_TIMESTEP)

        # 2. Ball Entities (Cue Ball + Endless Trick-Shot Targets)
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        self._cue_ball = Ball(pos=Vector2(center_x - 300.0, center_y))
        self._balls: list[Ball] = [
            self._cue_ball,
            Ball(pos=Vector2(center_x + 200.0, center_y)),
            Ball(pos=Vector2(center_x + 240.0, center_y - 25.0)),
            Ball(pos=Vector2(center_x + 240.0, center_y + 25.0)),
        ]

        # 3. Vision Pipeline & Gesture Decoupling
        self._tracker = HandTracker(camera_index=0)
        self._normalizer = GestureNormalizer(self._settings)
        self._classifier = GestureClassifier(self._settings, self._normalizer)
        self._mapper = ControlMapper(
            self._event_bus,
            self._settings,
            self._normalizer,
            self._classifier,
        )

        # 4. Rendering Layer
        self._table_sprite = TableSprite(self._display)
        self._ball_sprite = BallSprite(self._display)
        self._cue_sprite = CueSprite(self._display)
        self._hud = HUD(self._display)

        # 5. Controller State Registers
        self._current_angle: float = 0.0
        self._current_power: float = 0.0
        self._is_charging: bool = False
        self._score: int = 0
        self._streak: int = 0
        self._running: bool = True
        self._detected_gesture: str = "SEARCHING"

        # 6. Bind Event Bus Subscribers
        self._bind_events()

    def _bind_events(self) -> None:
        """Subscribe local state machine callbacks to InputEvent notifications."""
        self._event_bus.subscribe(InputEvent.AIM_CHANGED, self._on_aim_changed)
        self._event_bus.subscribe(InputEvent.POWER_CHANGED, self._on_power_changed)
        self._event_bus.subscribe(InputEvent.SHOT_FIRED, self._on_shot_fired)

    def _on_aim_changed(self, payload: AimPayload) -> None:
        """Handle normalized aim updates from the event bus."""
        self._current_angle = payload.angle

    def _on_power_changed(self, payload: PowerPayload) -> None:
        """Handle shot charge level changes from the event bus."""
        self._current_power = payload.power
        self._is_charging = payload.power > 0.01

    def _on_shot_fired(self, payload: ShotFiredPayload) -> None:
        """Apply kinetic impulse to cue ball when input fires a shot."""
        if self._cue_ball.state != BallState.STATIONARY:
            return

        # Power scalar converted to physical impulse
        impulse_magnitude = payload.power * MAX_POWER * 60.0
        shot_vector = Vector2(
            -math.cos(payload.angle) * impulse_magnitude,
            -math.sin(payload.angle) * impulse_magnitude,
        )
        self._cue_ball.vel = shot_vector
        self._cue_ball.state = BallState.MOVING

        # Reset charge status post-shot
        self._is_charging = False
        self._current_power = 0.0

    def _respawn_pocketed_balls(self) -> None:
        """Maintain the endless game loop by recycling pocketed targets."""
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        # Cue ball scratch check
        if self._cue_ball.state == BallState.POCKETED:
            self._cue_ball.pos = Vector2(center_x - 300.0, center_y)
            self._cue_ball.vel = Vector2.zero()
            self._cue_ball.state = BallState.STATIONARY
            self._streak = 0

        # Object balls respawn
        for ball in self._balls:
            if ball is not self._cue_ball and ball.state == BallState.POCKETED:
                self._score += 100 * max(1, self._streak)
                self._streak += 1
                ball.pos = Vector2(
                    center_x + float(100 + (self._score % 300)),
                    center_y + float(-100 + ((self._score * 37) % 200)),
                )
                ball.vel = Vector2.zero()
                ball.state = BallState.STATIONARY

    def run(self) -> None:
        """Execute the primary game loop with accumulator-based fixed physics steps."""
        self._tracker.start()
        accumulator = 0.0

        try:
            while self._running:
                # 1. Process OS and Window Events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self._display.handle_resize(event.w, event.h)
                    elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        self._running = False

                # 2. Ingest Latest Tracking Frame from Vision Worker
                latest_frame: Optional[HandTrackingFrame] = None
                while True:
                    try:
                        latest_frame = self._tracker.output_queue.get_nowait()
                    except queue.Empty:
                        break

                if latest_frame is not None:
                    self._detected_gesture = "TRACKING"
                    self._mapper.process_frame(latest_frame)
                elif self._tracker.output_queue.empty() and latest_frame is None:
                    # No new frame arrived this iteration
                    pass

                # 3. Drain Event Bus (Synchronize Thread-Safe Inputs to Game State)
                self._event_bus.process_queue()

                # 4. Integrate Physics Simulation (Fixed-Timestep Accumulator)
                # Cap delta time to prevent spiraling physics updates during stalls
                dt = min(self._display.end_frame(), 0.1)
                accumulator += dt

                while accumulator >= FIXED_TIMESTEP:
                    self._physics.step(self._balls)
                    self._respawn_pocketed_balls()
                    accumulator -= FIXED_TIMESTEP

                # 5. Render Scene Composition
                self._display.begin_frame()

                # Entities
                self._table_sprite.draw(self._table)
                for ball in self._balls:
                    color = (245, 245, 245) if ball is self._cue_ball else (210, 40, 40)
                    self._ball_sprite.draw(ball, color)

                # Cue stick overlay
                self._cue_sprite.draw(
                    cue_ball=self._cue_ball,
                    aim_angle=self._current_angle,
                    power_ratio=self._current_power,
                    is_charging=self._is_charging,
                )

                # Heads-Up Display
                fps = 1.0 / max(dt, 1e-5)
                self._hud.draw(
                    score=self._score,
                    streak=self._streak,
                    power=self._current_power,
                    is_charging=self._is_charging,
                    current_gesture=self._detected_gesture,
                    fps=fps,
                )

        finally:
            self._cleanup()

    def _cleanup(self) -> None:
        """Release threads, camera capture, and Pygame contexts."""
        logger.info("Executing teardown sequence...")
        self._tracker.stop()
        self._display.cleanup()
        logger.info("Shutdown complete.")


def main() -> None:
    """Bootstrap entry point."""
    game = GameOrchestrator()
    game.run()


if __name__ == "__main__":
    main()