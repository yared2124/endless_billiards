from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
import random


class ChallengeType(Enum):
    """Categories of trick-shot conditions."""

    MULTI_RAIL = auto()       # Ball must bounce off cushions N times before sinking
    SPEED_SINK = auto()       # Pocket any ball within N seconds
    BANK_SHOT = auto()        # Cue ball must hit rail before contacting target ball
    POCKET_SPECIFIC = auto()  # Sink a ball in a designated pocket index


@dataclass(slots=True)
class Challenge:
    """Specification of an active trick-shot objective."""

    challenge_type: ChallengeType
    description: str
    target_value: int          # E.g., rail bounce count or pocket index
    time_limit: float          # Remaining window in seconds to fulfill challenge
    bonus_points: int
    is_active: bool = True

    def update(self, dt: float) -> bool:
        """Tick down challenge timer. Returns True if expired."""
        if self.time_limit > 0.0:
            self.time_limit -= dt
            if self.time_limit <= 0.0:
                self.is_active = False
                return True
        return False


class ChallengeManager:
    """Generates, tracks, and validates procedural trick-shot challenges."""

    __slots__ = (
        "_active_challenge",
        "_current_rail_bounces",
        "_cue_hit_rail_first",
        "_cue_hit_target",
    )

    def __init__(self) -> None:
        self._active_challenge: Challenge | None = None
        self._current_rail_bounces: int = 0
        self._cue_hit_rail_first: bool = False
        self._cue_hit_target: bool = False

    @property
    def active_challenge(self) -> Challenge | None:
        """Currently active objective."""
        return self._active_challenge

    def record_rail_bounce(self) -> None:
        """Increment rail hit count during the current shot trajectory."""
        self._current_rail_bounces += 1
        if not self._cue_hit_target:
            self._cue_hit_rail_first = True

    def record_ball_collision(self) -> None:
        """Register impact between cue ball and an object ball."""
        self._cue_hit_target = True

    def reset_shot_telemetry(self) -> None:
        """Clear shot trajectory sensors for the next cue strike."""
        self._current_rail_bounces = 0
        self._cue_hit_rail_first = False
        self._cue_hit_target = False

    def generate_challenge(self) -> Challenge:
        """Create a procedural challenge from the pool."""
        challenge_type = random.choice(list(ChallengeType))

        if challenge_type == ChallengeType.MULTI_RAIL:
            rails = random.choice([1, 2])
            challenge = Challenge(
                challenge_type=challenge_type,
                description=f"Bank Shot: Hit {rails} cushion{'s' if rails > 1 else ''} & sink",
                target_value=rails,
                time_limit=20.0,
                bonus_points=300 * rails,
            )
        elif challenge_type == ChallengeType.SPEED_SINK:
            challenge = Challenge(
                challenge_type=challenge_type,
                description="Speed Sink: Pocket any ball in 8s!",
                target_value=1,
                time_limit=8.0,
                bonus_points=400,
            )
        elif challenge_type == ChallengeType.BANK_SHOT:
            challenge = Challenge(
                challenge_type=challenge_type,
                description="Rail First: Hit a cushion before target ball",
                target_value=1,
                time_limit=25.0,
                bonus_points=350,
            )
        else:  # POCKET_SPECIFIC
            pocket_idx = random.randint(0, 5)
            pocket_names = ["Top-Left", "Top-Mid", "Top-Right", "Bot-Right", "Bot-Mid", "Bot-Left"]
            challenge = Challenge(
                challenge_type=challenge_type,
                description=f"Call Pocket: Sink in {pocket_names[pocket_idx]}",
                target_value=pocket_idx,
                time_limit=30.0,
                bonus_points=500,
            )

        self._active_challenge = challenge
        return challenge

    def evaluate_pocket(self, pocket_index: int) -> tuple[bool, int]:
        """Verify whether the pocketed ball satisfies the active challenge.

        Args:
            pocket_index: Table pocket array index (0-5) where ball dropped.

        Returns:
            Tuple of (is_successful, bonus_points_awarded).
        """
        if not self._active_challenge or not self._active_challenge.is_active:
            return False, 0

        ch = self._active_challenge
        success = False

        if ch.challenge_type == ChallengeType.MULTI_RAIL:
            if self._current_rail_bounces >= ch.target_value:
                success = True
        elif ch.challenge_type == ChallengeType.SPEED_SINK:
            success = True
        elif ch.challenge_type == ChallengeType.BANK_SHOT:
            if self._cue_hit_rail_first:
                success = True
        elif ch.challenge_type == ChallengeType.POCKET_SPECIFIC:
            if pocket_index == ch.target_value:
                success = True

        if success:
            bonus = ch.bonus_points
            self.generate_challenge()  # Immediately rotate to next challenge
            return True, bonus

        return False, 0

    def update(self, dt: float) -> None:
        """Step challenge timer; generate replacement on expiration."""
        if not self._active_challenge:
            self.generate_challenge()
            return

        expired = self._active_challenge.update(dt)
        if expired:
            self.generate_challenge()