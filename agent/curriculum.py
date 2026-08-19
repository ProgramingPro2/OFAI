"""Curriculum manager: competency stages from duel -> FFA -> nations.

Advance requires `min_episodes` plus one frozen holdout mastery decision.
Sampled true-win rate must clear the stage threshold and a Wilson lower
bound. Deterministic argmax is logged as a diagnostic and never gates.
Training-time sampled rates and tile milestones never gate advancement.
`win` is a core FFA win only.

Resume is keyed by stage name. v2 checkpoints that only stored `idx` are
migrated through V2_STAGE_NAMES so inserted stages cannot silently remap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import maps as mapcat
from . import spaces
from .seeds import holdout_seed, ood_seed, pick_from_set

# Bump when the meaning of a mastery pass changes so older checkpoints
# do not mix det-only / consecutive lottery passes with the holdout gate.
MASTERY_VERSION = 3

# Frozen names of schema-v2 stages, in their original index order.
# Used only to migrate idx-only checkpoints after stages are inserted.
V2_STAGE_NAMES: tuple[str, ...] = (
    "duel",
    "small-ffa",
    "island-ffa",
    "tribes20",
    "nations-easy-small",
    "nations-easy-mid",
    "nations-medium-mid",
    "nations-hard-mid",
    "nations-impossible-world",
)

DEFAULT_HOLDOUT_EPISODES = 32
DEFAULT_DET_EVAL_EPISODES = 8
DEFAULT_OOD_EVAL_EPISODES = 8


class CurriculumRestoreError(ValueError):
    """Checkpoint stage identity cannot be mapped onto the current list."""


def wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    """Wilson score interval lower bound for a binomial win rate."""
    if n <= 0:
        return 0.0
    wins = max(0, min(int(wins), int(n)))
    n = int(n)
    p = wins / n
    z2 = z * z
    den = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / den
    half = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)) ** 0.5) / den
    return max(0.0, center - half)


@dataclass
class Stage:
    name: str
    map: str
    map_size: str = "Normal"
    nations: int | str = "disabled"
    difficulty: str = "Easy"
    bots: int = 20
    max_ticks: int = 12000
    decision_interval: int = 10
    shaping: float = 0.0
    allowed_actions: list[str] = field(default_factory=lambda: list(spaces.EARLY_ACTIONS))
    # Deterministic competency (logged only; not a gate).
    win_threshold: float = 0.55
    window: int = 50
    min_episodes: int = 100
    # Retained for checkpoint JSON compat; advancement is one holdout decision.
    eval_passes_required: int = 1
    # Sampled true-win mastery (the deployed PPO policy). Untrained random
    # on the duel is ~20-32% and must not pass. n=32 makes 0.80 a real 26/32
    # bar instead of a 7/8 lottery.
    sampled_win_threshold: float = 0.8
    sampled_wilson_floor: float = 0.50
    sampled_wilson_z: float = 1.96
    min_eval_episodes: int = DEFAULT_HOLDOUT_EPISODES
    # Training competency flag (core FFA win remains 80%). 0 disables.
    # Crossing this sets stageSuccess; it does not end the episode.
    win_tiles_frac: float = 0.0
    # Canonical map is `map`. Extra train maps unlock after map_burn_in
    # stage episodes. Holdout always uses holdout_map (defaults to canonical).
    train_maps: list[str] = field(default_factory=list)
    holdout_map: str = ""
    holdout_map_size: str = ""
    ood_maps: list[str] = field(default_factory=list)
    ood_map_size: str = ""
    # -1 → min_episodes // 4 canonical-only burn-in.
    map_burn_in: int = -1

    def __post_init__(self) -> None:
        if self.map_burn_in < 0:
            self.map_burn_in = max(0, int(self.min_episodes) // 4)
        if not self.holdout_map:
            self.holdout_map = self.map
        if not self.holdout_map_size:
            self.holdout_map_size = self.map_size
        if not self.ood_map_size:
            self.ood_map_size = self.map_size
        self.train_maps = list(self.train_maps)
        self.ood_maps = list(self.ood_maps)
        for key in (self.map, self.holdout_map, *self.train_maps, *self.ood_maps):
            mapcat.assert_map_key(key)
        overlap = set(self.ood_maps) & set(self.train_map_set)
        if overlap:
            raise ValueError(
                f"stage {self.name} OOD maps overlap the train set: {sorted(overlap)}"
            )

    @property
    def train_map_set(self) -> list[str]:
        seen: list[str] = []
        for key in (self.map, *self.train_maps):
            if key not in seen:
                seen.append(key)
        return seen


# Competency order: contiguous expand/duel -> small FFA -> island/boat ->
# tribes -> intro nations -> 8-nation FFA -> peninsula nations -> mid
# difficulty -> world easy -> world impossible.
_LAND_BOAT = list(spaces.LAND_BOAT_ACTIONS)
_ISLAND = list(spaces.ISLAND_ACTIONS)
_FULL = list(spaces.FULL_ACTIONS)

STAGES: list[Stage] = [
    Stage(
        name="duel",
        map="Halkidiki",
        map_size="Compact",
        nations="disabled",
        difficulty="Easy",
        bots=1,
        max_ticks=8000,
        shaping=1.0,
        allowed_actions=_LAND_BOAT,
        win_threshold=0.65,
        window=40,
        min_episodes=80,
        win_tiles_frac=0.30,
        ood_maps=list(mapcat.OOD_PENINSULA),
    ),
    Stage(
        name="small-ffa",
        map="Halkidiki",
        map_size="Compact",
        nations="disabled",
        difficulty="Easy",
        bots=5,
        max_ticks=10000,
        shaping=0.6,
        allowed_actions=_LAND_BOAT,
        win_threshold=0.55,
        window=50,
        min_episodes=120,
        ood_maps=list(mapcat.OOD_PENINSULA),
    ),
    Stage(
        name="island-ffa",
        map="FourIslands",
        map_size="Normal",
        nations="disabled",
        difficulty="Easy",
        bots=10,
        max_ticks=12000,
        shaping=0.4,
        allowed_actions=_ISLAND,
        win_threshold=0.5,
        window=60,
        min_episodes=150,
        train_maps=list(mapcat.ISLAND_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_ISLAND),
    ),
    Stage(
        name="tribes20",
        map="FourIslands",
        map_size="Normal",
        nations="disabled",
        difficulty="Easy",
        bots=20,
        max_ticks=16000,
        shaping=0.25,
        allowed_actions=_FULL,
        win_threshold=0.45,
        window=80,
        min_episodes=200,
        train_maps=list(mapcat.ISLAND_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_ISLAND),
    ),
    Stage(
        name="nations-easy-intro",
        map="FourIslands",
        map_size="Normal",
        nations=3,
        difficulty="Easy",
        bots=8,
        max_ticks=12000,
        shaping=0.2,
        allowed_actions=_FULL,
        win_threshold=0.45,
        window=80,
        min_episodes=180,
        train_maps=list(mapcat.ISLAND_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_ISLAND),
    ),
    Stage(
        name="nations-easy-small",
        map="FourIslands",
        map_size="Normal",
        nations=8,
        difficulty="Easy",
        bots=20,
        max_ticks=12000,
        shaping=0.15,
        allowed_actions=_FULL,
        win_threshold=0.4,
        window=100,
        min_episodes=250,
        train_maps=list(mapcat.ISLAND_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_ISLAND),
    ),
    Stage(
        name="nations-easy-peninsula",
        map="Halkidiki",
        map_size="Compact",
        nations=8,
        difficulty="Easy",
        bots=16,
        max_ticks=12000,
        shaping=0.12,
        allowed_actions=_FULL,
        win_threshold=0.38,
        window=100,
        min_episodes=220,
        train_maps=["Balkans"],
        ood_maps=list(mapcat.OOD_PENINSULA_HELD),
    ),
    Stage(
        name="nations-easy-mid",
        map="Halkidiki",
        map_size="Normal",
        nations=12,
        difficulty="Easy",
        bots=40,
        max_ticks=14000,
        shaping=0.1,
        allowed_actions=_FULL,
        win_threshold=0.35,
        window=100,
        min_episodes=300,
        train_maps=list(mapcat.PENINSULA_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_CONTINENT),
        ood_map_size="Normal",
    ),
    Stage(
        name="nations-medium-mid",
        map="Halkidiki",
        map_size="Normal",
        nations=12,
        difficulty="Medium",
        bots=40,
        max_ticks=14000,
        shaping=0.05,
        allowed_actions=_FULL,
        win_threshold=0.3,
        window=120,
        min_episodes=350,
        train_maps=list(mapcat.PENINSULA_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_CONTINENT),
        ood_map_size="Normal",
    ),
    Stage(
        name="nations-hard-mid",
        map="Halkidiki",
        map_size="Normal",
        nations=12,
        difficulty="Hard",
        bots=40,
        max_ticks=16000,
        shaping=0.0,
        allowed_actions=_FULL,
        win_threshold=0.25,
        window=120,
        min_episodes=400,
        train_maps=list(mapcat.PENINSULA_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_CONTINENT),
        ood_map_size="Normal",
    ),
    Stage(
        name="nations-world-easy",
        map="World",
        map_size="Normal",
        nations=12,
        difficulty="Easy",
        bots=20,
        max_ticks=18000,
        shaping=0.0,
        allowed_actions=_FULL,
        win_threshold=0.22,
        window=140,
        min_episodes=400,
        train_maps=list(mapcat.CONTINENT_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_CONTINENT),
        ood_map_size="Normal",
    ),
    Stage(
        name="nations-impossible-world",
        map="World",
        map_size="Normal",
        nations="default",
        difficulty="Impossible",
        bots=100,
        max_ticks=20000,
        shaping=0.0,
        allowed_actions=_FULL,
        win_threshold=0.2,
        window=150,
        min_episodes=500,
        train_maps=list(mapcat.CONTINENT_NORMAL[1:]),
        ood_maps=list(mapcat.OOD_CONTINENT),
        ood_map_size="Normal",
    ),
]


def all_curriculum_maps(stages: list[Stage] | None = None) -> set[str]:
    """Canonical + train + holdout + OOD maps referenced by the curriculum."""
    out: set[str] = set()
    for stage in stages or STAGES:
        out.add(stage.map)
        out.add(stage.holdout_map)
        out.update(stage.train_maps)
        out.update(stage.ood_maps)
    return out


class Curriculum:
    def __init__(self, stages: list[Stage] | None = None, start: int = 0):
        self.stages = stages or STAGES
        self.idx = max(0, min(start, len(self.stages) - 1))
        self.consecutive_eval_passes = 0
        self.last_eval_passed = False
        self.stage_episodes = 0
        # Stage-isolated competency window (logged; not an advance gate)
        self._stage_successes: list[float] = []

    @property
    def stage(self) -> Stage:
        return self.stages[min(self.idx, len(self.stages) - 1)]

    def _index_by_name(self, name: str) -> int | None:
        for i, stage in enumerate(self.stages):
            if stage.name == name:
                return i
        return None

    def _resolve_stage_index(self, d: dict) -> int:
        name = d.get("stage") or d.get("stage_name")
        if name:
            idx = self._index_by_name(str(name))
            if idx is None:
                known = [s.name for s in self.stages]
                raise CurriculumRestoreError(
                    f"checkpoint stage {name!r} is not in the current curriculum "
                    f"{known}"
                )
            return idx
        raw_idx = int(d.get("idx", self.idx))
        if raw_idx < 0 or raw_idx >= len(V2_STAGE_NAMES):
            raise CurriculumRestoreError(
                f"checkpoint idx {raw_idx} is outside the v2 stage table "
                f"(0..{len(V2_STAGE_NAMES) - 1})"
            )
        migrated = V2_STAGE_NAMES[raw_idx]
        idx = self._index_by_name(migrated)
        if idx is None:
            raise CurriculumRestoreError(
                f"v2 stage {migrated!r} (idx {raw_idx}) is missing from the "
                "current curriculum"
            )
        return idx

    def _choose_map(
        self,
        seed: str,
        *,
        suite: str,
        completed_episodes: int | None = None,
    ) -> tuple[str, str]:
        s = self.stage
        if suite == "holdout":
            return s.holdout_map, s.holdout_map_size
        if suite == "ood":
            keys = s.ood_maps or [s.map]
            return pick_from_set(keys, seed), s.ood_map_size
        seen = (
            self.stage_episodes
            if completed_episodes is None
            else int(completed_episodes)
        )
        if seen < s.map_burn_in or len(s.train_map_set) <= 1:
            return s.map, s.map_size
        return pick_from_set(s.train_map_set, seed), s.map_size

    def make_env_config(
        self,
        seed: str,
        *,
        objective: bool = False,
        suite: str = "train",
        completed_episodes: int | None = None,
    ) -> dict:
        """Build an EnvConfig. `objective=True` omits the tile milestone flag
        so infos only report core outcomes; eval still infers stage success
        from peak tiles.

        `suite` is train (staged map set), holdout (frozen canonical), or ood.
        Map choice is a pure function of (stage, suite, seed,
        completed_episodes or stage_episodes). Pass completed_episodes
        when building the *next* auto-reset config so burn-in unlocks
        on the episode after the threshold, not one later.
        """
        s = self.stage
        mmap, msize = self._choose_map(
            seed, suite=suite, completed_episodes=completed_episodes
        )
        cfg = {
            "map": mmap,
            "mapSize": msize,
            "nations": s.nations,
            "difficulty": s.difficulty,
            "bots": s.bots,
            "seed": seed,
            "maxTicks": s.max_ticks,
            "decisionInterval": s.decision_interval,
            "shaping": s.shaping,
            "allowedActions": spaces.encode_allowed_actions(s.allowed_actions),
        }
        if not objective and s.win_tiles_frac > 0:
            cfg["winTilesFrac"] = s.win_tiles_frac
        return cfg

    def holdout_seeds(self, n: int) -> list[str]:
        return [holdout_seed(self.stage.name, i) for i in range(n)]

    def holdout_configs(self, n: int, *, objective: bool = True) -> list[dict]:
        return [
            {
                **self.make_env_config(seed, objective=objective, suite="holdout"),
                "shaping": 0.0,
            }
            for seed in self.holdout_seeds(n)
        ]

    def ood_configs(self, n: int) -> list[dict]:
        s = self.stage
        if not s.ood_maps:
            return []
        return [
            {
                **self.make_env_config(
                    ood_seed(s.name, i), objective=True, suite="ood"
                ),
                "shaping": 0.0,
            }
            for i in range(n)
        ]

    def suite_meta(self) -> dict:
        s = self.stage
        return {
            "stage": s.name,
            "canonical_map": s.map,
            "canonical_map_size": s.map_size,
            "train_maps": s.train_map_set,
            "map_burn_in": s.map_burn_in,
            "holdout_map": s.holdout_map,
            "holdout_map_size": s.holdout_map_size,
            "ood_maps": list(s.ood_maps),
            "ood_map_size": s.ood_map_size,
            "min_eval_episodes": s.min_eval_episodes,
            "sampled_win_threshold": s.sampled_win_threshold,
            "sampled_wilson_floor": s.sampled_wilson_floor,
            "gate": "sampled_true_win_wilson",
        }

    def record_episode(self, stage_success: bool) -> None:
        """Track stage-local competency; honor Stage.window bound."""
        self.stage_episodes += 1
        self._stage_successes.append(1.0 if stage_success else 0.0)
        w = self.stage.window
        if len(self._stage_successes) > w:
            self._stage_successes = self._stage_successes[-w:]

    @property
    def stage_success_rate(self) -> float:
        if not self._stage_successes:
            return 0.0
        return float(sum(self._stage_successes) / len(self._stage_successes))

    @property
    def stage_win_rate(self) -> float:
        """Deprecated alias; this is stage success, not core win rate."""
        return self.stage_success_rate

    def note_eval(self, passed: bool) -> None:
        self.last_eval_passed = bool(passed)
        self.consecutive_eval_passes = 1 if passed else 0

    def judge_eval(self, ev: dict) -> dict:
        """Decide whether a frozen holdout eval is a mastery pass.

        Only sampled *true-win* mastery with a Wilson lower bound above
        the untrained-random floor can pass. Deterministic competency is
        reported and never gates. Sampled stageSuccess / milestone rate
        never passes.
        """
        s = self.stage
        det = ev.get("deterministic") or {}
        samp = ev.get("sampled") or {}
        det_n = int(det.get("episodes") or 0)
        samp_n = int(samp.get("episodes") or 0)
        det_wr = float(det.get("win_rate") or 0.0)
        det_ss = float(det.get("stage_success_rate") or 0.0)
        samp_wr = float(samp.get("win_rate") or 0.0)
        samp_ss = float(samp.get("stage_success_rate") or 0.0)
        samp_wins = int(round(samp_wr * max(samp_n, 0)))
        samp_lb = wilson_lower_bound(samp_wins, samp_n, s.sampled_wilson_z)

        samp_enough = samp_n >= s.min_eval_episodes
        samp_mastery = (
            samp_enough
            and samp_wr >= s.sampled_win_threshold
            and samp_lb >= s.sampled_wilson_floor
        )
        reason = "sampled_true_win" if samp_mastery else "none"
        return {
            "passed": bool(samp_mastery),
            "reason": reason,
            "eval_modes": ["deterministic", "sampled"],
            "gate": "sampled_true_win_wilson",
            "min_eval_episodes": s.min_eval_episodes,
            "sampled_episodes": samp_n,
            "sampled_win_rate": round(samp_wr, 4),
            "sampled_stage_success_rate": round(samp_ss, 4),
            "sampled_wilson_lower": round(samp_lb, 4),
            "sampled_win_threshold": s.sampled_win_threshold,
            "sampled_wilson_floor": s.sampled_wilson_floor,
            "sampled_wilson_z": s.sampled_wilson_z,
            "det_episodes": det_n,
            "det_win_rate": round(det_wr, 4),
            "det_stage_success_rate": round(det_ss, 4),
            "det_threshold": s.win_threshold,
            "det_diagnostic_only": True,
            "consecutive_eval_passes": self.consecutive_eval_passes,
            "eval_passes_required": 1,
            "last_eval_passed": self.last_eval_passed,
            "stage": s.name,
            "holdout_map": s.holdout_map,
            "holdout_map_size": s.holdout_map_size,
        }

    def maybe_advance(self) -> bool:
        """Advance on min_episodes + one frozen holdout mastery pass.

        Training-time sampled stage_success_rate is not a gate.
        """
        s = self.stage
        if self.idx >= len(self.stages) - 1:
            return False
        if self.stage_episodes < s.min_episodes:
            return False
        if not self.last_eval_passed:
            return False
        self.idx += 1
        self._reset_stage_stats()
        return True

    def _reset_stage_stats(self) -> None:
        self.stage_episodes = 0
        self._stage_successes = []
        self.consecutive_eval_passes = 0
        self.last_eval_passed = False

    def state_dict(self) -> dict:
        successes = list(self._stage_successes)
        return {
            "idx": self.idx,
            "stage": self.stage.name,
            "consecutive_eval_passes": self.consecutive_eval_passes,
            "last_eval_passed": self.last_eval_passed,
            "stage_episodes": self.stage_episodes,
            "stage_successes": successes,
            # Older checkpoints/tests used this name for the same window.
            "stage_wins": successes,
            "mastery_version": MASTERY_VERSION,
        }

    def load_state_dict(self, d: dict) -> None:
        self.idx = self._resolve_stage_index(d)
        self.stage_episodes = int(d.get("stage_episodes", 0))
        raw = d.get("stage_successes", d.get("stage_wins", []))
        self._stage_successes = list(raw)
        w = self.stage.window
        if len(self._stage_successes) > w:
            self._stage_successes = self._stage_successes[-w:]
        version = int(d.get("mastery_version", 1))
        # Pre-v3 consecutive / det-only passes must not count.
        if version < MASTERY_VERSION:
            self.last_eval_passed = False
            self.consecutive_eval_passes = 0
            return
        if "last_eval_passed" in d:
            self.last_eval_passed = bool(d.get("last_eval_passed"))
        else:
            self.last_eval_passed = int(d.get("consecutive_eval_passes", 0)) > 0
        self.consecutive_eval_passes = 1 if self.last_eval_passed else 0
