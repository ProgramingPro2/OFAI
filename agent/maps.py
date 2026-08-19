"""OpenFront map keys and staged train / holdout / OOD sets.

Wire keys are GameMapType enum names (e.g. FourIslands), not display values
("Four Islands"). Invalid keys fail init in AgentEnv. Tensor / action shapes
do not depend on the map.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_MAP_ENUM_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9]*)\s*=", re.MULTILINE)

# Curated keys actually scheduled by the curriculum. Never include arcade,
# tournament, all-land (TheBox), Giant World, or cosmic maps.
PENINSULA_COMPACT = ("Halkidiki",)
ISLAND_NORMAL = ("FourIslands", "Hawaii", "Caribbean", "FaroeIslands")
PENINSULA_NORMAL = ("Halkidiki", "Balkans", "Britannia")
CONTINENT_NORMAL = ("World", "Europe", "NorthAmerica")
OOD_PENINSULA = ("Balkans",)
# When Balkans is already in the train set, hold out another peninsula.
OOD_PENINSULA_HELD = ("Britannia",)
OOD_ISLAND = ("ArchipelagoSea",)
OOD_CONTINENT = ("SouthAmerica",)

CURATED_MAP_KEYS: frozenset[str] = frozenset(
    (
        *PENINSULA_COMPACT,
        *ISLAND_NORMAL,
        *PENINSULA_NORMAL,
        *CONTINENT_NORMAL,
        *OOD_PENINSULA,
        *OOD_PENINSULA_HELD,
        *OOD_ISLAND,
        *OOD_CONTINENT,
    )
)


def parse_game_map_keys(source: str) -> list[str]:
    """Extract GameMapType enum keys from OpenFront TypeScript."""
    start = source.find("enum GameMapType")
    if start < 0:
        return []
    end = source.find("}", start)
    if end < 0:
        return []
    return _MAP_ENUM_RE.findall(source[start:end])


def installed_game_map_keys(openfront_root: Path | None = None) -> frozenset[str]:
    """Parse installed OpenFront Maps.gen.ts when the fork is present."""
    root = Path(openfront_root) if openfront_root is not None else ROOT / "OpenFrontIO"
    for rel in ("src/core/game/Maps.gen.ts", "src/core/game/Game.ts"):
        path = root / rel
        if path.is_file():
            return frozenset(parse_game_map_keys(path.read_text(encoding="utf-8")))
    return frozenset()


def assert_map_key(name: str, installed: frozenset[str] | None = None) -> str:
    """Reject keys that are not real OpenFront GameMapType names."""
    key = str(name)
    known = installed if installed is not None else installed_game_map_keys()
    if known:
        if key not in known:
            raise ValueError(f"unknown OpenFront map key {key!r}")
        return key
    if key not in CURATED_MAP_KEYS:
        raise ValueError(f"unknown OpenFront map key {key!r} (Maps.gen.ts missing)")
    return key


def validate_map_list(names: list[str] | tuple[str, ...]) -> list[str]:
    installed = installed_game_map_keys()
    return [assert_map_key(n, installed) for n in names]
