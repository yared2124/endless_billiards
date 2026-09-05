"""Particle simulation system rendering chalk dust and pocket impact effects."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

import pygame

from endless_billiards.rendering.display import DisplayManager
from endless_billiards.utils.math2d import Vector2


@dataclass(slots=True)
class Particle:
    """Kinematic particle data."""

    pos: Vector2
    vel: Vector2
    color: tuple[int, int, int]
    radius: float
    alpha: float
    decay: float


class ParticleSystem:
    """Spawns, integrates, and draws visual particle effects on screen."""

    __slots__ = ("_display", "_particles")

    def __init__(self, display: DisplayManager) -> None:
        self._display: DisplayManager = display
        self._particles: list[Particle] = []

    def spawn_chalk_burst(self, pos: Vector2, shoot_direction: Vector2, count: int = 16) -> None:
        """Spawn chalk particles when the cue tip impacts the cue ball."""
        base_angle = math.atan2(shoot_direction.y, shoot_direction.x)

        for _ in range(count):
            angle = base_angle + random.uniform(-0.6, 0.6)
            speed = random.uniform(80.0, 320.0)
            vel = Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            color = random.choice([
                (220, 240, 255),
                (180, 215, 245),
                (245, 245, 255),
            ])
            self._particles.append(
                Particle(
                    pos=Vector2(pos.x, pos.y),
                    vel=vel,
                    color=color,
                    radius=random.uniform(2.0, 4.5),
                    alpha=255.0,
                    decay=random.uniform(280.0, 480.0),
                )
            )

    def spawn_pocket_splash(self, pos: Vector2, count: int = 24) -> None:
        """Spawn radial bursts when a ball falls into a pocket."""
        for _ in range(count):
            angle = random.uniform(0.0, 2.0 * math.pi)
            speed = random.uniform(40.0, 180.0)
            vel = Vector2(math.cos(angle) * speed, math.sin(angle) * speed)
            color = random.choice([
                (255, 200, 60),
                (255, 140, 30),
                (240, 240, 240),
            ])
            self._particles.append(
                Particle(
                    pos=Vector2(pos.x, pos.y),
                    vel=vel,
                    color=color,
                    radius=random.uniform(2.5, 5.0),
                    alpha=255.0,
                    decay=random.uniform(200.0, 350.0),
                )
            )

    def update(self, dt: float) -> None:
        """Integrate particles and clear fully faded instances."""
        alive: list[Particle] = []
        for p in self._particles:
            p.pos += p.vel * dt
            p.vel *= 0.94  # Aerodynamic drag
            p.alpha -= p.decay * dt

            if p.alpha > 0.0:
                alive.append(p)
        self._particles = alive

    def draw(self) -> None:
        """Render active particles with alpha transparency onto the surface."""
        surface = self._display.screen

        for p in self._particles:
            screen_pos = self._display.world_to_screen(p.pos)
            radius = max(1, self._display.scale_scalar(p.radius * (p.alpha / 255.0)))

            circle_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
            color_with_alpha = (*p.color, int(p.alpha))
            pygame.draw.circle(circle_surf, color_with_alpha, (radius, radius), radius)
            surface.blit(circle_surf, (screen_pos[0] - radius, screen_pos[1] - radius))