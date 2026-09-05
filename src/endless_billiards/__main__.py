"""Production game loop integrating MediaPipe vision, physics, polish, and rendering."""

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
    POCKET_RADIUS_SQUARED,
    TABLE_MAX_X,
    TABLE_MAX_Y,
    TABLE_MIN_X,
    TABLE_MIN_Y,
)
from endless_billiards.config.settings import Settings
from endless_billiards.core.entities.ball import Ball, BallState
from endless_billiards.core.entities.spawner import BallSpawner
from endless_billiards.core.entities.table import Table
from endless_billiards.core.physics.engine import PhysicsEngine
from endless_billiards.core.rules.scoring import ScoreKeeper
from endless_billiards.core.rules.trick_shots import ChallengeManager
from endless_billiards.input.events import (
    AimPayload,
    EventBus,
    InputEvent,
    PowerPayload,
    ShotFiredPayload,
)
from endless_billiards.input.gestures import GestureClassifier
from endless_billiards.input.mapping import ControlMapper
from endless_billiards.rendering.audio import SoundManager
from endless_billiards.rendering.camera import CameraController
from endless_billiards.rendering.debug_view import DebugView
from endless_billiards.rendering.display import DisplayManager
from endless_billiards.rendering.hud import HUD
from endless_billiards.rendering.particles import ParticleSystem
from endless_billiards.rendering.predictor import TrajectoryPredictor
from endless_billiards.rendering.sprites import BallSprite, CueSprite, TableSprite
from endless_billiards.tools.replay import ReplaySystem
from endless_billiards.utils.math2d import Vector2
from endless_billiards.vision.landmarks import GestureNormalizer
from endless_billiards.vision.tracker import HandTracker, HandTrackingFrame

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("endless_billiards")


class Game:
    """Orchestrates threading, simulation, input mapping, polish, and scene presentation."""

    __slots__ = (
        "_settings",
        "_event_bus",
        "_display",
        "_table",
        "_physics",
        "_score_keeper",
        "_spawner",
        "_challenges",
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
        "_sound",
        "_particles",
        "_camera",
        "_predictor",
        "_replay",
        "_current_angle",
        "_current_power",
        "_is_charging",
        "_running",
        "_current_gesture_str",
    )

    def __init__(self) -> None:
        # 1. Configuration & Core Systems
        self._settings = Settings.from_json("assets/config/default_settings.json")
        self._event_bus = EventBus()
        self._display = DisplayManager(window_width=1280, window_height=720, target_fps=60)

        self._table = Table()
        self._physics = PhysicsEngine(table=self._table, dt=FIXED_TIMESTEP)

        # 2. Game Mechanics & State
        self._score_keeper = ScoreKeeper(initial_time_seconds=60.0)
        self._spawner = BallSpawner(self._table)
        self._challenges = ChallengeManager()
        self._challenges.generate_challenge()

        # 3. Ball Entities Setup
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        self._cue_ball = Ball(pos=Vector2(center_x - 360.0, center_y))
        self._balls: list[Ball] = [self._cue_ball]

        initial_targets = self._spawner.replenish_cluster(self._balls, desired_target_count=4)
        self._balls.extend(initial_targets)

        # 4. Computer Vision & Input Mapping Bridge
        self._tracker = HandTracker(camera_index=0)
        self._normalizer = GestureNormalizer(self._settings)
        self._classifier = GestureClassifier(self._settings, self._normalizer)
        self._mapper = ControlMapper(
            self._event_bus,
            self._settings,
            self._normalizer,
            self._classifier,
        )

        # 5. Core Rendering & Visual Sprites
        self._table_sprite = TableSprite(self._display)
        self._ball_sprite = BallSprite(self._display)
        self._cue_sprite = CueSprite(self._display)
        self._hud = HUD(self._display)
        self._debug_view = DebugView(self._display)

        # 6. Polish Subsystems
        self._sound = SoundManager(master_volume=self._settings.audio_volume)
        self._particles = ParticleSystem(self._display)
        self._camera = CameraController(self._display)
        self._predictor = TrajectoryPredictor(self._display, self._table)
        self._replay = ReplaySystem()
        self._replay.start_recording()

        # 7. Dynamic Control State Registers
        self._current_angle: float = 0.0
        self._current_power: float = 0.0
        self._is_charging: bool = False
        self._running: bool = True
        self._current_gesture_str: str = "SEARCHING"

        self._register_subscribers()

    def _register_subscribers(self) -> None:
        """Subscribe game logic callbacks to decoupled EventBus topics."""
        self._event_bus.subscribe(InputEvent.AIM_CHANGED, self._on_aim)
        self._event_bus.subscribe(InputEvent.POWER_CHANGED, self._on_power)
        self._event_bus.subscribe(InputEvent.SHOT_FIRED, self._on_shot)

    def _on_aim(self, payload: AimPayload) -> None:
        self._current_angle = payload.angle

    def _on_power(self, payload: PowerPayload) -> None:
        self._current_power = payload.power
        self._is_charging = payload.power > 0.01

    def _on_shot(self, payload: ShotFiredPayload) -> None:
        if self._cue_ball.state != BallState.STATIONARY or self._score_keeper.is_game_over:
            return

        self._challenges.reset_shot_telemetry()

        # 1. Physical Impulse
        impulse_speed = payload.power * MAX_POWER * 70.0
        shot_direction = Vector2(-math.cos(payload.angle), -math.sin(payload.angle))
        self._cue_ball.vel = shot_direction * impulse_speed
        self._cue_ball.state = BallState.MOVING

        # 2. Polish: Sound, Particles, Screen Shake, and Replay Recording
        self._sound.play_cue_hit(power_factor=payload.power)
        self._particles.spawn_chalk_burst(self._cue_ball.pos, shot_direction)

        if payload.power > 0.65:
            self._camera.add_trauma(payload.power * 0.45)

        self._replay.record_frame(
            aim_angle=payload.angle,
            power=payload.power,
            gesture=self._current_gesture_str,
            is_shot=True,
        )

        self._is_charging = False
        self._current_power = 0.0

    def _get_nearest_pocket_index(self, pos: Vector2) -> int:
        """Find the 0-5 pocket array index matching a coordinate."""
        for idx, pocket_pos in enumerate(self._table.pockets):
            if pos.distance_squared_to(pocket_pos) <= POCKET_RADIUS_SQUARED:
                return idx
        return 0

    def _resolve_pocketed_balls(self) -> None:
        """Process score updates, pocket particle bursts, audio cues, and spawner logic."""
        center_x = (TABLE_MIN_X + TABLE_MAX_X) * 0.5
        center_y = (TABLE_MIN_Y + TABLE_MAX_Y) * 0.5

        # 1. Cue Ball Scratch Handling
        if self._cue_ball.state == BallState.POCKETED:
            self._sound.play_pocket_sink()
            self._particles.spawn_pocket_splash(self._cue_ball.pos)
            self._camera.add_trauma(0.25)

            self._score_keeper.register_scratch()
            self._cue_ball.pos = Vector2(center_x - 360.0, center_y)
            self._cue_ball.vel = Vector2.zero()
            self._cue_ball.state = BallState.STATIONARY

        # 2. Target Balls Sinking & Evaluation
        for ball in self._balls:
            if ball is not self._cue_ball and ball.state == BallState.POCKETED:
                # Polish: Audio, Splash Particles, and Micro-trauma
                self._sound.play_pocket_sink()
                self._particles.spawn_pocket_splash(ball.pos)
                self._camera.add_trauma(0.18)

                pocket_idx = self._get_nearest_pocket_index(ball.pos)
                is_trick, _ = self._challenges.evaluate_pocket(pocket_idx)

                self._score_keeper.register_pocketed_ball(
                    base_value=100,
                    rail_hits=self._challenges._current_rail_bounces,
                    is_trick_shot=is_trick,
                )

        # 3. Clean Table and Replenish Cluster to Maintain Endless Targets
        retained_balls = [self._cue_ball] + [
            b for b in self._balls[1:] if b.state != BallState.POCKETED
        ]
        newly_spawned = self._spawner.replenish_cluster(retained_balls, desired_target_count=4)
        self._balls = retained_balls + newly_spawned

    def run(self) -> None:
        """Primary execution loop running CV ingestion, fixed physics, polish, and draw passes."""
        logger.info("Starting background MediaPipe tracker thread...")
        self._tracker.start()

        physics_accumulator: float = 0.0

        try:
            while self._running:
                # ---------------------------------------------------------
                # 1. OS Event Loop (Window Management & Hotkeys)
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
                            self._sound.play_ui_blip()
                        elif event.key == pygame.K_SPACE:
                            self._mapper.start_calibration()
                            self._sound.play_ui_blip()
                        elif event.key == pygame.K_r and self._score_keeper.is_game_over:
                            self._score_keeper.reset(initial_time_seconds=60.0)
                            self._sound.play_ui_blip()

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

                # Record regular tracking telemetry
                self._replay.record_frame(
                    aim_angle=self._current_angle,
                    power=self._current_power,
                    gesture=self._current_gesture_str,
                    is_shot=False,
                )

                # ---------------------------------------------------------
                # 4. Step Simulation (Fixed-Timestep Accumulator)
                # ---------------------------------------------------------
                # Measure frame time while isolating render throttle
                dt = min(self._display.present(), 0.1)
                physics_accumulator += dt

                # Update camera shake and kinematic particle systems
                self._camera.update(dt)
                self._particles.update(dt)

                while physics_accumulator >= FIXED_TIMESTEP:
                    if not self._score_keeper.is_game_over:
                        self._score_keeper.update_time(FIXED_TIMESTEP)
                        self._challenges.update(FIXED_TIMESTEP)

                    self._physics.step(self._balls)
                    self._resolve_pocketed_balls()
                    physics_accumulator -= FIXED_TIMESTEP

                # ---------------------------------------------------------
                # 5. Scene Rendering & Composition
                # ---------------------------------------------------------
                self._display.clear()

                # Table Felt and Rail Cushions
                self._table_sprite.draw(self._table)

                # Trajectory Prediction (Ghost Ball & Cut Angle Line)
                if not self._score_keeper.is_game_over:
                    self._predictor.draw(
                        cue_ball=self._cue_ball,
                        aim_angle=self._current_angle,
                        other_balls=self._balls[1:],
                    )

                # Billiard Spheres
                for ball in self._balls:
                    color = (250, 250, 250) if ball is self._cue_ball else (225, 45, 45)
                    self._ball_sprite.draw(ball, color)

                # Cue Stick
                if not self._score_keeper.is_game_over:
                    self._cue_sprite.draw(
                        cue_ball=self._cue_ball,
                        aim_angle=self._current_angle,
                        power_ratio=self._current_power,
                        is_charging=self._is_charging,
                    )

                # Particles Overlay (Chalk and Pocket Splashes)
                self._particles.draw()

                # Hitboxes & Velocity Vectors (F1 Toggle)
                if self._settings.show_debug_overlays:
                    self._debug_view.draw(self._balls)

                # HUD Overlay
                current_fps = 1.0 / max(dt, 1e-4)
                self._hud.draw(
                    score=self._score_keeper.score,
                    streak=self._score_keeper.streak,
                    power=self._current_power,
                    is_charging=self._is_charging,
                    gesture_name=self._current_gesture_str,
                    fps=current_fps,
                )

                # Restore un-jittered display offsets before the next frame loop
                self._camera.restore_offsets()

        finally:
            self._teardown()

    def _teardown(self) -> None:
        """Gracefully terminate background workers, flush replays, and clean display context."""
        logger.info("Initiating teardown sequence...")
        self._tracker.stop()
        self._replay.save_to_disk("assets/replays/last_session.json")
        self._display.cleanup()
        logger.info("Teardown completed cleanly.")


def main() -> None:
    """Bootstrap application execution."""
    game = Game()
    game.run()


if __name__ == "__main__":
    main()