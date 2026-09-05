
from __future__ import annotations

import logging
import math
import queue
import sys
from typing import Optional

import pygame

from endless_billiards.config.constants import (
    FIXED_TIMESTEP,
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
from endless_billiards.input.gestures import Gesture, GestureClassifier
from endless_billiards.input.mapping import ControlMapper
from endless_billiards.rendering.debug_view import DebugView
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.rendering.hud import HUD
from endless_billiards.rendering.sprites import BallSprite, CueSprite, TableSprite
from endless_billiards.utils.math2d import Vector2
from endless_billiards.vision.landmarks import GestureNormalizer
from endless_billiards.vision.tracker import HandTracker, HandTrackingFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("endless_billiards")


class Game:
    """Orchestrates threading, input events, fixed-step simulation, and presentation."""

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
        "_debug_view",
        "_current_angle",
        "_current_power",
        "_is_charging",
        "_score",
        "_streak",
        "_running",
        "_current_gesture_str",
    )

    def __init__(self) -> None:
        # 1. Configuration & Core State
        self._settings = Settings.from_json("assets/config/default_settings.json")
        self._event_bus = EventBus()
        self._display = DisplayManager(window_width=1280, window_height=720, target_fps=60)

        self._table = Table()
        self._physics = PhysicsEngine(table=self._table, dt=FIXED_TIMESTEP)

        # 2. Spawn Cue Ball & Initial Trick-Shot Targets
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        self._cue_ball = Ball(pos=Vector2(center_x - 360.0, center_y))
        self._balls: list[Ball] = [
            self._cue_ball,
            Ball(pos=Vector2(center_x + 200.0, center_y)),
            Ball(pos=Vector2(center_x + 245.0, center_y - 28.0)),
            Ball(pos=Vector2(center_x + 245.0, center_y + 28.0)),
        ]

        # 3. Vision & Input Decoupling Bridge
        self._tracker = HandTracker(camera_index=0)
        self._normalizer = GestureNormalizer(self._settings)
        self._classifier = GestureClassifier(self._settings, self._normalizer)
        self._mapper = ControlMapper(
            self._event_bus,
            self._settings,
            self._normalizer,
            self._classifier,
        )

        # 4. Rendering Subsystems
        self._table_sprite = TableSprite(self._display)
        self._ball_sprite = BallSprite(self._display)
        self._cue_sprite = CueSprite(self._display)
        self._hud = HUD(self._display)
        self._debug_view = DebugView(self._display)

        # 5. Local State
        self._current_angle: float = 0.0
        self._current_power: float = 0.0
        self._is_charging: bool = False
        self._score: int = 0
        self._streak: int = 0
        self._running: bool = True
        self._current_gesture_str: str = "SEARCHING"

        self._register_subscribers()

    def _register_subscribers(self) -> None:
        """Subscribe game mechanics handlers to event bus topics."""
        self._event_bus.subscribe(InputEvent.AIM_CHANGED, self._on_aim)
        self._event_bus.subscribe(InputEvent.POWER_CHANGED, self._on_power)
        self._event_bus.subscribe(InputEvent.SHOT_FIRED, self._on_shot)

    def _on_aim(self, payload: AimPayload) -> None:
        self._current_angle = payload.angle

    def _on_power(self, payload: PowerPayload) -> None:
        self._current_power = payload.power
        self._is_charging = payload.power > 0.01

    def _on_shot(self, payload: ShotFiredPayload) -> None:
        if self._cue_ball.state != BallState.STATIONARY:
            return

        # Map normalized power [0.0, 1.0] to physics launch impulse
        impulse_speed = payload.power * MAX_POWER * 70.0
        self._cue_ball.vel = Vector2(
            -math.cos(payload.angle) * impulse_speed,
            -math.sin(payload.angle) * impulse_speed,
        )
        self._cue_ball.state = BallState.MOVING

        self._is_charging = False
        self._current_power = 0.0

    def _resolve_endless_rules(self) -> None:
        """Handle scoring, resets on scratches, and endless ball recycling."""
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        # Cue ball scratch
        if self._cue_ball.state == BallState.POCKETED:
            self._cue_ball.pos = Vector2(center_x - 360.0, center_y)
            self._cue_ball.vel = Vector2.zero()
            self._cue_ball.state = BallState.STATIONARY
            self._streak = 0

        # Object ball sink
        for ball in self._balls:
            if ball is not self._cue_ball and ball.state == BallState.POCKETED:
                self._streak += 1
                self._score += 150 * self._streak
                # Respawn in varied table trick zones
                offset_x = 100.0 + ((self._score * 23) % 400)
                offset_y = -150.0 + ((self._score * 67) % 300)
                ball.pos = Vector2(center_x + offset_x, center_y + offset_y)
                ball.vel = Vector2.zero()
                ball.state = BallState.STATIONARY

    def run(self) -> None:
        """Main game loop combining OS polling, CV queue, physics, and frame presentation."""
        logger.info("Spinning up background MediaPipe tracker thread...")
        self._tracker.start()

        physics_accumulator: float = 0.0

        try:
            while self._running:
                # ---------------------------------------------------------
                # 1. OS Event Loop (Window & Keyboard overrides)
                # ---------------------------------------------------------
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._running = False
                    elif event.type == pygame.VIDEORESIZE:
                        self._display.handle_resize(event.w, event.h)
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self._running = False
                        elif event.key == pygame.K_F1:
                            self._settings.show_debug_overlays = not self._settings.show_debug_overlays
                        elif event.key == pygame.K_SPACE:
                            self._mapper.start_calibration()

                # ---------------------------------------------------------
                # 2. Ingest Asynchronous Vision Frame
                # ---------------------------------------------------------
                latest_frame: Optional[HandTrackingFrame] = None
                while True:
                    try:
                        latest_frame = self._tracker.output_queue.get_nowait()
                    except queue.Empty:
                        break

                if latest_frame is not None:
                    self._mapper.process_frame(latest_frame)
                    current_gesture = self._classifier.classify(latest_frame)
                    self._current_gesture_str = current_gesture.name
                elif self._tracker.output_queue.empty() and latest_frame is None:
                    self._current_gesture_str = "SEARCHING"

                # ---------------------------------------------------------
                # 3. Synchronize Bus Events to Simulation State
                # ---------------------------------------------------------
                self._event_bus.process_queue()

                # ---------------------------------------------------------
                # 4. Step Physics with Fixed Timestep Accumulator
                # ---------------------------------------------------------
                dt = min(self._display.present(), 0.1)
                physics_accumulator += dt

                while physics_accumulator >= FIXED_TIMESTEP:
                    self._physics.step(self._balls)
                    self._resolve_endless_rules()
                    physics_accumulator -= FIXED_TIMESTEP

                # ---------------------------------------------------------
                # 5. Render Scene Composition
                # ---------------------------------------------------------
                self._display.clear()

                # Table felt & geometry
                self._table_sprite.draw(self._table)

                # Active billiard balls
                for ball in self._balls:
                    color = (250, 250, 250) if ball is self._cue_ball else (225, 45, 45)
                    self._ball_sprite.draw(ball, color)

                # Aim trajectory line and cue stick
                self._cue_sprite.draw(
                    cue_ball=self._cue_ball,
                    aim_angle=self._current_angle,
                    power_ratio=self._current_power,
                    is_charging=self._is_charging,
                )

                # Optional debug hitboxes/velocity arrows (Toggle with F1)
                if self._settings.show_debug_overlays:
                    self._debug_view.draw(self._balls)

                # Telemetry HUD
                current_fps = 1.0 / max(dt, 1e-4)
                self._hud.draw(
                    score=self._score,
                    streak=self._streak,
                    power=self._current_power,
                    is_charging=self._is_charging,
                    gesture_name=self._current_gesture_str,
                    fps=current_fps,
                )

        finally:
            self._teardown()

    def _teardown(self) -> None:
        """Release background capture threads and graphical context cleanly."""
        logger.info("Initiating teardown sequence...")
        self._tracker.stop()
        self._display.cleanup()
        logger.info("Teardown completed cleanly.")


def main() -> None:
    """Launch execution."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()