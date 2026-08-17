"""Curriculum manager: competency stages from duel -> FFA -> nations.

Advance requires `min_episodes` plus two consecutive fixed-seed eval
passes. A pass is either deterministic competency or sampled true-win
mastery (Wilson-guarded). Training-time sampled rates and tile
milestones never gate advancement. `win` is a core FFA win only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from . import spaces

# Bump when the meaning of consecutive_eval_passes changes so older
# checkpoints do not mix det-only passes with the dual mastery gate.
MASTERY_VERSION = 2


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
    # Deterministic competency: true win_rate, or stage_success_rate when
    # win_tiles_frac > 0. Greedy/argmax is a robustness diagnostic.
    win_threshold: float = 0.55
    window: int = 50
    min_episodes: int = 100
    eval_passes_required: int = 2
    # Sampled true-win mastery (the deployed PPO policy). Untrained random
    # on the duel is ~20-32% and must not pass.
    sampled_win_threshold: float = 0.8
    sampled_wilson_floor: float = 0.50
    sampled_wilson_z: float = 1.96
    min_eval_episodes: int = 8
    # Training competency flag (core FFA win remains 80%). 0 disables.
    # Crossing this sets stageSuccess; it does not end the episode.
    win_tiles_frac: float = 0.0


# Competency order: contiguous expand/duel -> small FFA -> island/boat -> full.
_LAND = list(spaces.LAND_ACTIONS)
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
        # Flag only (no termination). 0.15 ended every smoke episode and
        # collapsed true-objective sampled wins (untrained 32% -> final 2%).
        # 0.30 is a deterministic expand competency (untrained det 0%).
        win_tiles_frac=0.30,
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
    ),
]


class Curriculum:
    def __init__(self, stages: list[Stage] | None = None, start: int = 0):
        self.stages = stages or STAGES
        self.idx = max(0, min(start, len(self.stages) - 1))
        self.consecutive_eval_passes = 0
        self.stage_episodes = 0
        # Stage-isolated competency window (logged; not an advance gate)
        self._stage_successes: list[float] = []

    @property
    def stage(self) -> Stage:
        return self.stages[min(self.idx, len(self.stages) - 1)]

    def make_env_config(self, seed: str, *, objective: bool = False) -> dict:
        """Build an EnvConfig. `objective=True` omits the tile milestone flag
        so infos only report core outcomes; eval still infers stage success
        from peak tiles.
        """
        s = self.stage
        cfg = {
            "map": s.map,
            "mapSize": s.map_size,
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
        if passed:
            self.consecutive_eval_passes += 1
        else:
            self.consecutive_eval_passes = 0

    def judge_eval(self, ev: dict) -> dict:
        """Decide whether a fixed-seed eval is a mastery pass.

        Path A: deterministic competency (true win, or det stage success
        on a milestone stage). Path B: sampled *true-win* mastery with a
        Wilson lower bound above the untrained-random floor. Sampled
        stageSuccess / milestone rate never passes.
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

        det_enough = det_n >= s.min_eval_episodes
        samp_enough = samp_n >= s.min_eval_episodes
        det_win_pass = det_enough and det_wr >= s.win_threshold
        det_comp_pass = det_enough and (
            det_ss >= s.win_threshold if s.win_tiles_frac > 0 else det_wr >= s.win_threshold
        )
        samp_mastery = (
            samp_enough
            and samp_wr >= s.sampled_win_threshold
            and samp_lb >= s.sampled_wilson_floor
        )
        if samp_mastery:
            reason = "sampled_true_win"
        elif det_win_pass:
            reason = "deterministic_true_win"
        elif det_comp_pass:
            reason = "deterministic_competency"
        else:
            reason = "none"
        passed = reason != "none"
        return {
            "passed": passed,
            "reason": reason,
            "eval_modes": ["deterministic", "sampled"],
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
            "consecutive_eval_passes": self.consecutive_eval_passes,
            "eval_passes_required": s.eval_passes_required,
            "stage": s.name,
        }

    def maybe_advance(self) -> bool:
        """Advance on min_episodes + consecutive dual-mastery eval passes.

        Training-time sampled stage_success_rate is not a gate.
        """
        s = self.stage
        if self.idx >= len(self.stages) - 1:
            return False
        if self.stage_episodes < s.min_episodes:
            return False
        if self.consecutive_eval_passes < s.eval_passes_required:
            return False
        self.idx += 1
        self._reset_stage_stats()
        return True

    def _reset_stage_stats(self) -> None:
        self.stage_episodes = 0
        self._stage_successes = []
        self.consecutive_eval_passes = 0

    def state_dict(self) -> dict:
        successes = list(self._stage_successes)
        return {
            "idx": self.idx,
            "consecutive_eval_passes": self.consecutive_eval_passes,
            "stage_episodes": self.stage_episodes,
            "stage_successes": successes,
            # Older checkpoints/tests used this name for the same window.
            "stage_wins": successes,
            "mastery_version": MASTERY_VERSION,
        }

    def load_state_dict(self, d: dict) -> None:
        self.idx = int(d.get("idx", self.idx))
        self.consecutive_eval_passes = int(d.get("consecutive_eval_passes", 0))
        self.stage_episodes = int(d.get("stage_episodes", 0))
        raw = d.get("stage_successes", d.get("stage_wins", []))
        self._stage_successes = list(raw)
        w = self.stage.window
        if len(self._stage_successes) > w:
            self._stage_successes = self._stage_successes[-w:]
        # Old det-only passes must not count toward the dual mastery gate.
        if int(d.get("mastery_version", 1)) < MASTERY_VERSION:
            self.consecutive_eval_passes = 0
