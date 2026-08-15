"""Action/observation space constants, mirrored from OpenFrontIO/ofai/env/spec.ts.

Keep in sync with the TypeScript side; EnvServer validates nothing, so a
mismatch here silently corrupts training.
"""

SPATIAL_CHANNELS = 10
SPATIAL_SIZE = 64
NUM_PLAYER_SLOTS = 16  # slot 0 is always the agent
PLAYER_FEATURES = 14
GLOBAL_FEATURES = 10

REGION_GRID = 32
NUM_REGIONS = REGION_GRID * REGION_GRID

NUM_ACTION_TYPES = 9
NUM_QUANTITIES = 5
NUM_UNIT_TYPES = 10

ACTION_NOOP = 0
ACTION_SPAWN = 1
ACTION_ATTACK = 2
ACTION_RETREAT_ALL = 3
ACTION_BUILD = 4
ACTION_BOAT = 5
ACTION_ALLY = 6
ACTION_BREAK_ALLY = 7
ACTION_EMBARGO = 8

ACTION_NAMES = [
    "noop",
    "spawn",
    "attack",
    "retreat_all",
    "build",
    "boat",
    "ally",
    "break_ally",
    "embargo",
]

# Sizes of each factorized head, in action-vector order.
ACTION_HEADS = [NUM_ACTION_TYPES, NUM_PLAYER_SLOTS, NUM_REGIONS, NUM_QUANTITIES, NUM_UNIT_TYPES]

UNIT_NAMES = [
    "City",
    "DefensePost",
    "SAMLauncher",
    "MissileSilo",
    "Port",
    "Factory",
    "AtomBomb",
    "HydrogenBomb",
    "MIRV",
    "Warship",
]
