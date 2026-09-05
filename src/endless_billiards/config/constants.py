"""Physics constants, dimensions, and game simulation configuration parameters."""

from typing import Final

# --- Screen & Table Dimensions (Logical Coordinate System) ---
LOGICAL_WIDTH: Final[float] = 1920.0
LOGICAL_HEIGHT: Final[float] = 1080.0

# Table playing surface area (inset inside borders)
TABLE_MARGIN_X: Final[float] = 160.0
TABLE_MARGIN_Y: Final[float] = 90.0

TABLE_MIN_X: Final[float] = TABLE_MARGIN_X
TABLE_MAX_X: Final[float] = LOGICAL_WIDTH - TABLE_MARGIN_X
TABLE_MIN_Y: Final[float] = TABLE_MARGIN_Y
TABLE_MAX_Y: Final[float] = LOGICAL_HEIGHT - TABLE_MARGIN_Y

TABLE_PLAYABLE_WIDTH: Final[float] = TABLE_MAX_X - TABLE_MIN_X
TABLE_PLAYABLE_HEIGHT: Final[float] = TABLE_MAX_Y - TABLE_MIN_Y

# --- Billiard Ball Metrics ---
BALL_RADIUS: Final[float] = 15.0
BALL_DIAMETER: Final[float] = BALL_RADIUS * 2.0
BALL_MASS: Final[float] = 0.17  # Standard billiard ball mass in kg

# --- Pocket Dimensions ---
POCKET_RADIUS: Final[float] = 22.0
POCKET_RADIUS_SQUARED: Final[float] = POCKET_RADIUS * POCKET_RADIUS

# --- Simulation Dynamics ---
PHYSICS_HZ: Final[int] = 120
FIXED_TIMESTEP: Final[float] = 1.0 / float(PHYSICS_HZ)

FRICTION_DECAY: Final[float] = 0.985  
RESTITUTION: Final[float] = 0.92       
CUSHION_RESTITUTION: Final[float] = 0.85  

SLEEP_VELOCITY_THRESHOLD: Final[float] = 0.8  
MAX_BALL_SPEED: Final[float] = 3000.0         
MAX_POWER: Final[float] = 2500.0              