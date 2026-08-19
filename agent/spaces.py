"""Action/observation space constants, mirrored from OpenFrontIO/ofai/env/spec.ts.

Keep in sync with the TypeScript side; EnvServer validates nothing, so a
mismatch here silently corrupts training.
"""

from __future__ import annotations

SPATIAL_CHANNELS = 10
SPATIAL_SIZE = 64
NUM_PLAYER_SLOTS = 16  # slot 0 is always the agent
# Overflow (>15 opponents): slots 1-14 stay targetable individuals;
# slot 15 is an untargetable aggregate (Nation+Bot flags). Same shape.
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

# Land combat plus boat (peninsula maps). Build/diplomacy unlock later.
# EARLY_ACTIONS aliases the duel/small-ffa set.
LAND_ACTIONS = ["noop", "spawn", "attack", "retreat_all"]
# Halkidiki / other peninsula maps need boats to reach a land opponent.
LAND_BOAT_ACTIONS = LAND_ACTIONS + ["boat"]
ISLAND_ACTIONS = LAND_ACTIONS + ["build", "boat"]
EARLY_ACTIONS = list(LAND_BOAT_ACTIONS)
FULL_ACTIONS = list(ACTION_NAMES)

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

TERMINAL_CAUSES = (
    "none",
    "win",
    "curriculum_success",
    "death",
    "loss_alive",
    "timeout",
    "no_spawn",
)

# Flat target-mask layout: [NUM_ACTION_TYPES, NUM_PLAYER_SLOTS] row-major.
TARGET_MASKS_SIZE = NUM_ACTION_TYPES * NUM_PLAYER_SLOTS

# Which parameter heads are semantically active for each action type.
# Action-type head is always active; these are the conditional heads only.
_ACTIVE_COND: dict[int, frozenset[str]] = {
    ACTION_NOOP: frozenset(),
    ACTION_SPAWN: frozenset({"region"}),
    ACTION_ATTACK: frozenset({"target", "quantity"}),
    ACTION_RETREAT_ALL: frozenset(),
    ACTION_BUILD: frozenset({"unit", "region"}),
    ACTION_BOAT: frozenset({"region", "quantity"}),
    ACTION_ALLY: frozenset({"target"}),
    ACTION_BREAK_ALLY: frozenset({"target"}),
    ACTION_EMBARGO: frozenset({"target"}),
}

HEAD_INDEX = {
    "action": 0,
    "target": 1,
    "region": 2,
    "quantity": 3,
    "unit": 4,
}


def active_conditional_heads(action_type: int) -> frozenset[str]:
    return _ACTIVE_COND.get(int(action_type), frozenset())


def uses_target(action_type: int) -> bool:
    return "target" in active_conditional_heads(action_type)


def uses_region(action_type: int) -> bool:
    return "region" in active_conditional_heads(action_type)


def uses_quantity(action_type: int) -> bool:
    return "quantity" in active_conditional_heads(action_type)


def uses_unit(action_type: int) -> bool:
    return "unit" in active_conditional_heads(action_type)


def encode_allowed_actions(actions: list[str] | list[int]) -> list[int]:
    """Map curriculum/eval action names to EnvConfig.allowedActions ids."""
    out: list[int] = []
    for a in actions:
        if isinstance(a, int):
            if not 0 <= a < NUM_ACTION_TYPES:
                raise ValueError(f"invalid action id {a}")
            out.append(int(a))
        else:
            if a not in ACTION_NAMES:
                raise ValueError(f"unknown action name {a!r}")
            out.append(ACTION_NAMES.index(a))
    return out
