from __future__ import annotations

import math
from typing import NamedTuple, Self


class Vector2:
    """A high-performance, two-dimensional Cartesian vector.

    Attributes:
        x: Coordinate on the horizontal axis.
        y: Coordinate on the vertical axis.
    """

    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        """Initialize a 2D Vector.

        Args:
            x: Horizontal coordinate.
            y: Vertical coordinate.
        """
        self.x: float = float(x)
        self.y: float = float(y)

    def __repr__(self) -> str:
        """Return developer-readable representation."""
        return f"Vector2(x={self.x:.4f}, y={self.y:.4f})"

    def __iter__(self):
        """Enable unpacking (e.g., x, y = vec)."""
        yield self.x
        yield self.y

    def __eq__(self, other: object) -> bool:
        """Evaluate equality with floating-point tolerance."""
        if not isinstance(other, Vector2):
            return False
        return math.isclose(self.x, other.x, abs_tol=1e-7) and math.isclose(
            self.y, other.y, abs_tol=1e-7
        )

    def __add__(self, other: Vector2) -> Vector2:
        """Add two vectors."""
        return Vector2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector2) -> Vector2:
        """Subtract another vector from this vector."""
        return Vector2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vector2:
        """Multiply vector by a scalar value."""
        return Vector2(self.x * scalar, self.y * scalar)

    def __rmul__(self, scalar: float) -> Vector2:
        """Right-multiply vector by a scalar value."""
        return self.__mul__(scalar)

    def __truediv__(self, scalar: float) -> Vector2:
        """Divide vector by a scalar value."""
        if scalar == 0.0:
            raise ZeroDivisionError("Cannot divide Vector2 by zero scalar.")
        inv = 1.0 / scalar
        return Vector2(self.x * inv, self.y * inv)

    def __neg__(self) -> Vector2:
        """Negate the vector components."""
        return Vector2(-self.x, -self.y)

    def dot(self, other: Vector2) -> float:
        """Calculate the scalar dot product between two vectors.

        Args:
            other: Target vector.

        Returns:
            Scalar result of the dot product.
        """
        return self.x * other.x + self.y * other.y

    def cross(self, other: Vector2) -> float:
        """Compute the 2D cross product analog (perpendicular dot product).

        Args:
            other: Target vector.

        Returns:
            The signed magnitude of the cross-product z-component.
        """
        return self.x * other.y - self.y * other.x

    def magnitude_squared(self) -> float:
        """Calculate the squared magnitude of the vector.

        Returns:
            Squared Euclidean norm.
        """
        return self.x * self.x + self.y * self.y

    def magnitude(self) -> float:
        """Calculate the Euclidean length of the vector.

        Returns:
            Length of the vector.
        """
        return math.hypot(self.x, self.y)

    def normalize(self) -> Vector2:
        """Return a unit vector pointing in the same direction.

        Returns:
            A normalized Vector2, or a zero vector if original length is 0.
        """
        mag = self.magnitude()
        if math.isclose(mag, 0.0, abs_tol=1e-12):
            return Vector2(0.0, 0.0)
        inv = 1.0 / mag
        return Vector2(self.x * inv, self.y * inv)

    def distance_to(self, other: Vector2) -> float:
        """Compute Euclidean distance to another point/vector.

        Args:
            other: Destination point.

        Returns:
            Straight-line distance between both positions.
        """
        return math.hypot(self.x - other.x, self.y - other.y)

    def distance_squared_to(self, other: Vector2) -> float:
        """Compute squared distance to avoid expensive square root calls.

        Args:
            other: Destination point.

        Returns:
            Squared distance between both positions.
        """
        dx = self.x - other.x
        dy = self.y - other.y
        return dx * dx + dy * dy

    def reflect(self, normal: Vector2) -> Vector2:
        """Reflect this vector across a surface normal vector.

        Formula: v' = v - 2 * (v . n) * n

        Args:
            normal: A normalized surface normal vector.

        Returns:
            Reflected velocity or directional vector.
        """
        dot_product = self.dot(normal)
        return self - (normal * (2.0 * dot_product))

    def lerp(self, target: Vector2, alpha: float) -> Vector2:
        """Linearly interpolate toward another vector.

        Args:
            target: Destination vector.
            alpha: Interpolation factor bounded roughly between [0.0, 1.0].

        Returns:
            Interpolated Vector2.
        """
        return Vector2(
            self.x + (target.x - self.x) * alpha,
            self.y + (target.y - self.y) * alpha,
        )

    def clamp_magnitude(self, max_length: float) -> Vector2:
        """Limit vector magnitude to an upper bound without changing direction.

        Args:
            max_length: Maximum allowed magnitude.

        Returns:
            Vector2 with magnitude <= max_length.
        """
        if max_length < 0.0:
            raise ValueError("max_length cannot be negative.")

        mag_sq = self.magnitude_squared()
        if mag_sq > max_length * max_length:
            mag = math.sqrt(mag_sq)
            factor = max_length / mag
            return Vector2(self.x * factor, self.y * factor)
        return Vector2(self.x, self.y)

    @classmethod
    def zero(cls) -> Vector2:
        """Create a Vector2 at (0.0, 0.0)."""
        return cls(0.0, 0.0)


class BoundingBox(NamedTuple):
    """Immutable Axis-Aligned Bounding Box (AABB) representation."""

    min_x: float
    min_y: float
    max_x: float
    max_y: float