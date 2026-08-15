"""Curriculum manager: advances through stages when the rolling win rate
clears a threshold. Stage definitions follow docs/PLAN.md Phase 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage:
    name: str
    map: str
    map_size: str = "Normal"
    nations: int | str = "default"
    difficulty: str = "Easy"
    bots: int = 20
    max_ticks: int = 12000
    decision_interval: int = 10
    shaping: float = 0.0
    # Advance when rolling win rate (last `window` episodes) >= win_threshold
    win_threshold: float = 0.6
    window: int = 100
    min_episodes: int = 200  # never advance before this many episodes


STAGES: list[Stage] = [
    Stage(
        name="tribes",
        map="FourIslands",
        nations="disabled",
        difficulty="Easy",
        bots=20,
        max_ticks=6000,
        win_threshold=0.7,
        min_episodes=100,
    ),
    Stage(
        name="nations-easy-small",
        map="FourIslands",
        nations=8,
        difficulty="Easy",
        bots=20,
        max_ticks=9000,
        win_threshold=0.6,
        min_episodes=200,
    ),
    Stage(
        name="nations-easy-mid",
        map="Halkidiki",
        nations=12,
        difficulty="Easy",
        bots=40,
        max_ticks=12000,
        win_threshold=0.6,
        min_episodes=300,
    ),
    Stage(
        name="nations-medium-mid",
        map="Halkidiki",
        nations=12,
        difficulty="Medium",
        bots=40,
        max_ticks=12000,
        win_threshold=0.55,
        min_episodes=300,
    ),
    Stage(
        name="nations-hard-mid",
        map="Halkidiki",
        nations=12,
        difficulty="Hard",
        bots=40,
        max_ticks=12000,
        win_threshold=0.5,
        min_episodes=400,
    ),
    Stage(
        name="nations-impossible-world",
        map="World",
        nations="default",
        difficulty="Impossible",
        bots=100,
        max_ticks=16000,
        win_threshold=0.4,
        min_episodes=500,
    ),
]


class Curriculum:
    def __init__(self, stages: list[Stage] | None = None, start: int = 0):
        self.stages = stages or STAGES
        self.idx = start

    @property
    def stage(self) -> Stage:
        return self.stages[min(self.idx, len(self.stages) - 1)]

    def make_env_config(self, seed: str) -> dict:
        s = self.stage
        return {
            "map": s.map,
            "mapSize": s.map_size,
            "nations": s.nations,
            "difficulty": s.difficulty,
            "bots": s.bots,
            "seed": seed,
            "maxTicks": s.max_ticks,
            "decisionInterval": s.decision_interval,
            "shaping": s.shaping,
        }

    def maybe_advance(self, win_rate: float, total_episodes: int, episodes_in_stage: int) -> bool:
        s = self.stage
        if self.idx >= len(self.stages) - 1:
            return False
        if episodes_in_stage < s.min_episodes:
            return False
        if win_rate >= s.win_threshold:
            self.idx += 1
            return True
        return False
