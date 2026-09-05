
from __future__ import annotations

import math
from typing import Optional

from endless_billiards.config.settings import Settings
from endless_billiards.utils.math2d import Vector2
from endless_billiards.vision.tracker import HandTrackingFrame


class GestureNormalizer:
    """Normalizes raw hand landmark coordinates into simulation control signals.

    Features deadzone rejection, angle smoothing integration, and sensitivity adjustments.
    """

    __slots__ = ("_settings", "_previous_angle")

    def __init__(self, settings: Settings) -> None:
        """Initialize the normalizer with dynamic game parameters.

        Args:
            settings: Runtime settings instance containing deadzone and sensitivity factors.
        """
        self._settings: Settings = settings
        self._previous_angle: Optional[float] = None

    def get_cue_angle(self, frame: HandTrackingFrame) -> float:
        """Calculate cue aiming angle based on the vector from Wrist to Middle MCP.

        Applies deadzone filtering relative to previous frames and scales with
        gesture sensitivity.

        Args:
            frame: Hand landmark tracking data.

        Returns:
            Calculated cue aiming angle in radians [-pi, pi].
        """
        # Aim vector: from wrist (base) pointing toward middle metacarpophalangeal joint
        dir_x = frame.middle_mcp.x - frame.wrist.x
        dir_y = frame.middle_mcp.y - frame.wrist.y

        raw_vector = Vector2(dir_x, dir_y)
        vector_length = raw_vector.magnitude()

        # Reject negligible movement below deadzone threshold
        if vector_length < self._settings.deadzone:
            if self._previous_angle is not None:
                return self._previous_angle
            return 0.0

        # Base angle relative to the 2D plane
        raw_angle = math.atan2(raw_vector.y, raw_vector.x)

        if self._previous_angle is None:
            self._previous_angle = raw_angle
            return raw_angle

        # Calculate smallest angular difference (handling wrapping at [-pi, pi])
        delta_angle = (raw_angle - self._previous_angle + math.pi) % (2.0 * math.pi) - math.pi

        # Apply deadzone threshold directly on angular fluctuations
        if abs(delta_angle) < (self._settings.deadzone * 0.5):
            return self._previous_angle

        # Scale movement with gesture sensitivity
        scaled_delta = delta_angle * self._settings.gesture_sensitivity
        new_angle = self._previous_angle + scaled_delta

        # Normalize back within range [-pi, pi]
        normalized_angle = (new_angle + math.pi) % (2.0 * math.pi) - math.pi
        self._previous_angle = normalized_angle

        return normalized_angle

    def get_pinch_distance(self, frame: HandTrackingFrame) -> float:
        """Compute normalized distance between index finger tip and thumb tip.

        Accounts for hand size by scaling the Euclidean distance between fingertips
        by the reference hand scale (distance from wrist to middle MCP).

        Args:
            frame: Hand landmark tracking data.

        Returns:
            Normalized pinch metric clamped between [0.0, 1.0].
            0.0 corresponds to full pinch; 1.0 corresponds to wide open fingers.
        """
        # Distance between Thumb Tip (4) and Index Tip (8)
        dx_fingers = frame.thumb_tip.x - frame.index_tip.x
        dy_fingers = frame.thumb_tip.y - frame.index_tip.y
        fingertip_dist = math.hypot(dx_fingers, dy_fingers)

        # Baseline hand scale: Wrist (0) to Middle MCP (9)
        dx_scale = frame.middle_mcp.x - frame.wrist.x
        dy_scale = frame.middle_mcp.y - frame.wrist.y
        hand_scale = math.hypot(dx_scale, dy_scale)

        # Guard against zero division if hand landmarks collapse
        if hand_scale < 1e-4:
            return 1.0

        # Normalize distance invariant to camera distance
        # Typical pinch closed is ~0.1 to 0.2 of hand length; fully open is ~0.8 to 1.2
        relative_distance = fingertip_dist / hand_scale

        # Deadzone: collapse any distance below configured threshold to zero
        if relative_distance <= self._settings.pinch_threshold:
            return 0.0

        # Map distance range [pinch_threshold, 1.0] to [0.0, 1.0]
        span = max(1e-4, 1.0 - self._settings.pinch_threshold)
        normalized_pinch = (relative_distance - self._settings.pinch_threshold) / span

        return max(0.0, min(1.0, normalized_pinch))

    def reset(self) -> None:
        """Reset historical smoothing and angle tracking states."""
        self._previous_angle = None