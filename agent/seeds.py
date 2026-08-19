"""Deterministic mixed seed stream for train / holdout / OOD.

Python owns episode identity. TypeScript auto-reset may apply a provided
next config, or fall back to `{seed}-r{n}` when the trainer is absent
(watch / ad-hoc eval). Counters persist across checkpoint resume.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


def mix_seed(*parts: object) -> str:
    """Independent hex seed from structured parts (not sequential djb2)."""
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(str(part).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def holdout_seed(stage_name: str, index: int) -> str:
    """Frozen per-stage holdout identity. Stable across evals and retries."""
    return mix_seed("holdout", stage_name, int(index))


def ood_seed(stage_name: str, index: int) -> str:
    return mix_seed("ood", stage_name, int(index))


def pick_from_set(items: list[str] | tuple[str, ...], seed: str) -> str:
    """Deterministic uniform choice from a non-empty set."""
    if not items:
        raise ValueError("pick_from_set requires at least one item")
    if len(items) == 1:
        return items[0]
    n = int.from_bytes(
        hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest(),
        "little",
    )
    return items[n % len(items)]


@dataclass
class SeedStream:
    """Per-env episode counters and mixed train seeds.

    `episode_index[i]` is the in-progress (or next-to-start) episode on
    env i. It advances only when that env reports `done`.
    """

    run_name: str
    master_seed: int
    n_envs: int
    episode_index: list[int] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.master_seed = int(self.master_seed)
        self.n_envs = int(self.n_envs)
        if not self.episode_index:
            self.episode_index = [0] * self.n_envs
        elif len(self.episode_index) != self.n_envs:
            # Shard count changed on resume: keep overlap, pad/truncate.
            prev = [int(x) for x in self.episode_index]
            self.episode_index = [
                prev[i] if i < len(prev) else 0 for i in range(self.n_envs)
            ]

    def env_seed(self, env_i: int) -> str:
        return mix_seed(
            "train",
            self.run_name,
            self.master_seed,
            int(env_i),
            int(self.episode_index[env_i]),
        )

    def next_env_seed(self, env_i: int) -> str:
        return mix_seed(
            "train",
            self.run_name,
            self.master_seed,
            int(env_i),
            int(self.episode_index[env_i]) + 1,
        )

    def note_done(self, env_i: int) -> None:
        self.episode_index[env_i] = int(self.episode_index[env_i]) + 1

    def all_seeds(self) -> list[str]:
        return [self.env_seed(i) for i in range(self.n_envs)]

    def state_dict(self) -> dict:
        return {
            "run_name": self.run_name,
            "master_seed": self.master_seed,
            "n_envs": self.n_envs,
            "episode_index": list(self.episode_index),
        }

    def load_state_dict(self, d: dict) -> None:
        self.run_name = str(d.get("run_name", self.run_name))
        self.master_seed = int(d.get("master_seed", self.master_seed))
        stored_n = int(d.get("n_envs", self.n_envs))
        prev = [int(x) for x in d.get("episode_index", [])]
        if stored_n != self.n_envs:
            self.episode_index = [
                prev[i] if i < len(prev) else 0 for i in range(self.n_envs)
            ]
        elif prev:
            if len(prev) != self.n_envs:
                self.episode_index = [
                    prev[i] if i < len(prev) else 0 for i in range(self.n_envs)
                ]
            else:
                self.episode_index = prev
        else:
            self.episode_index = [0] * self.n_envs

    def summary(self) -> dict:
        idxs = self.episode_index
        return {
            "n_envs": self.n_envs,
            "master_seed": self.master_seed,
            "min_episode_index": min(idxs) if idxs else 0,
            "max_episode_index": max(idxs) if idxs else 0,
            "sum_episode_index": int(sum(idxs)),
        }
