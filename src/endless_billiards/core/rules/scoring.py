
from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ScoreBreakdown:
    """Detailed metadata for a scored shot."""

    base_points: int
    multiplier: int
    streak_bonus: int
    time_bonus: int
    total_added: int


class ScoreKeeper:
    """Manages the player's score, streak ladders, and trick bonuses."""

    __slots__ = (
        "_score",
        "_streak",
        "_highest_streak",
        "_combo_multiplier",
        "_time_remaining",
        "_is_game_over",
    )

    def __init__(self, initial_time_seconds: float = 60.0) -> None:
        """Initialize scoreboard with baseline time limit.

        Args:
            initial_time_seconds: Starting countdown timer for endless play.
        """
        self._score: int = 0
        self._streak: int = 0
        self._highest_streak: int = 0
        self._combo_multiplier: int = 1
        self._time_remaining: float = initial_time_seconds
        self._is_game_over: bool = False

    @property
    def score(self) -> int:
        """Current total points."""
        return self._score

    @property
    def streak(self) -> int:
        """Current unbroken streak count."""
        return self._streak

    @property
    def combo_multiplier(self) -> int:
        """Active point multiplier."""
        return self._combo_multiplier

    @property
    def time_remaining(self) -> float:
        """Countdown timer in seconds."""
        return max(0.0, self._time_remaining)

    @property
    def is_game_over(self) -> bool:
        """Check if time expired."""
        return self._is_game_over

    def update_time(self, dt: float) -> None:
        """Tick down remaining countdown clock.

        Args:
            dt: Delta time elapsed in seconds.
        """
        if self._is_game_over:
            return

        self._time_remaining -= dt
        if self._time_remaining <= 0.0:
            self._time_remaining = 0.0
            self._is_game_over = True

    def add_time_bonus(self, seconds: float) -> None:
        """Reward extra time for trick-shot execution."""
        if not self._is_game_over:
            self._time_remaining += seconds

    def register_pocketed_ball(
        self,
        base_value: int = 100,
        rail_hits: int = 0,
        is_trick_shot: bool = False,
    ) -> ScoreBreakdown:
        """Calculate points for pocketing an object ball and step combo multiplier.

        Args:
            base_value: Baseline point value for object ball.
            rail_hits: Cushion bounces that occurred during the shot.
            is_trick_shot: Whether this shot satisfied an active challenge.

        Returns:
            Calculated score breakdown for HUD display.
        """
        self._streak += 1
        if self._streak > self._highest_streak:
            self._highest_streak = self._streak

        # Multiplier grows every 3 consecutive successful pockets
        self._combo_multiplier = 1 + (self._streak // 3)

        streak_bonus = (self._streak - 1) * 25
        rail_bonus = rail_hits * 50
        trick_bonus = 200 if is_trick_shot else 0
        time_award = 4.0 + (1.0 * min(self._combo_multiplier, 5))

        total_points = (
            (base_value + rail_bonus + trick_bonus) * self._combo_multiplier
        ) + streak_bonus

        self._score += total_points
        self.add_time_bonus(time_award)

        return ScoreBreakdown(
            base_points=base_value,
            multiplier=self._combo_multiplier,
            streak_bonus=streak_bonus,
            time_bonus=int(time_award),
            total_added=total_points,
        )

    def register_scratch(self) -> None:
        """Handle cue ball scratch penalties."""
        self._streak = 0
        self._combo_multiplier = 1
        # Time penalty on scratch
        self._time_remaining = max(0.0, self._time_remaining - 5.0)
        if self._time_remaining == 0.0:
            self._is_game_over = True

    def reset(self, initial_time_seconds: float = 60.0) -> None:
        """Reset scoreboard to fresh state."""
        self._score = 0
        self._streak = 0
        self._highest_streak = 0
        self._combo_multiplier = 1
        self._time_remaining = initial_time_seconds
        self._is_game_over = False