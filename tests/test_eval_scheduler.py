"""Pending-seed eval scheduler: recovery, no duplicates, gate-only reject.

Self-contained / mocked — does not require a live EnvServer.
Run: .venv/bin/python -m pytest tests/test_eval_scheduler.py -q
"""

from __future__ import annotations

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.curriculum import Curriculum, cannot_pass_mastery_gate, wilson_lower_bound
from agent.env_client import RolloutInterrupted, ShardedVecEnv, StepBatch
from agent.model import Policy
from agent.train import (
    EVAL_PURPOSE_BEST,
    EVAL_PURPOSE_GATE,
    EVAL_PURPOSE_MASTERY,
    EVAL_PURPOSE_SCORE,
    FIXED_SEED_EVAL_RETRIES,
    PendingSeedScheduler,
    eval_allows_early_reject,
    run_fixed_seed_eval,
)
from tests.test_rl_pipeline import _EvalInterruptEnv, _fake_info, _numpy_obs


class _SeedOutcomeEnv:
    """Finishes every claimed seed on the first step. Outcome is seed-id."""

    def __init__(self, k: int, *, lose_all: bool = False, win_all: bool = False):
        self.k = k
        self.lose_all = lose_all
        self.win_all = win_all
        self.init_calls: list[list[str]] = []
        self.step_calls = 0
        self.step_widths: list[int] = []
        self._cfgs: list[dict] = []

    def init(self, cfgs):
        self._cfgs = list(cfgs)
        self.init_calls.append([c["seed"] for c in cfgs])
        return _numpy_obs(len(cfgs))

    def init_active(self, cfgs):
        return self.init(cfgs)

    def step(self, actions):
        return self._finish(actions)

    def step_active(self, actions, live=None):
        return self._finish(actions)

    def _win_for(self, seed: str) -> bool:
        if self.lose_all:
            return False
        if self.win_all:
            return True
        try:
            return int(str(seed).rsplit("-", 1)[-1]) % 2 == 0
        except ValueError:
            return False

    def _finish(self, actions):
        self.step_calls += 1
        n = int(actions.shape[0])
        self.step_widths.append(n)
        infos = []
        for i in range(n):
            seed = str(self._cfgs[i]["seed"]) if i < len(self._cfgs) else str(i)
            win = self._win_for(seed)
            infos.append(
                _fake_info(
                    terminalCause="win" if win else "timeout",
                    win=win,
                    tick=4 + i,
                    peakTilesFrac=0.4 if win else 0.05,
                    tilesFrac=0.4 if win else 0.05,
                )
            )
        return StepBatch(
            obs=_numpy_obs(n),
            rewards=np.ones(n, np.float32),
            dones=np.ones(n, dtype=bool),
            infos=infos,
        )


class _InterruptThenWinEnv:
    """Complete the first lane, then interrupt once; later inits finish."""

    def __init__(self, k: int):
        self.k = k
        self.init_calls: list[list[str]] = []
        self._steps_since_init = 0
        self._interrupted_once = False
        self.seen_seeds: list[str] = []

    def init(self, cfgs):
        seeds = [c["seed"] for c in cfgs]
        self.init_calls.append(seeds)
        self.seen_seeds.extend(seeds)
        self._steps_since_init = 0
        return _numpy_obs(len(cfgs))

    def init_active(self, cfgs):
        return self.init(cfgs)

    def step(self, actions):
        return self._step(actions)

    def step_active(self, actions, live=None):
        return self._step(actions)

    def _step(self, actions):
        self._steps_since_init += 1
        n = int(actions.shape[0])
        if not self._interrupted_once and self._steps_since_init == 2:
            self._interrupted_once = True
            raise RolloutInterrupted([0], failed_lanes=[0])
        win_info = _fake_info(
            terminalCause="win", win=True, tick=10, peakTilesFrac=0.5, tilesFrac=0.5
        )
        if not self._interrupted_once and self._steps_since_init == 1:
            dones = np.zeros(n, dtype=bool)
            dones[0] = True
            infos = [win_info]
            infos.extend(
                [_fake_info(terminalCause="none", win=False, tick=5)] * (n - 1)
            )
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.ones(n, np.float32),
                dones=dones,
                infos=infos,
            )
        return StepBatch(
            obs=_numpy_obs(n),
            rewards=np.ones(n, np.float32),
            dones=np.ones(n, dtype=bool),
            infos=[win_info] * n,
        )


def test_eval_allows_early_reject_only_for_gate():
    assert eval_allows_early_reject(EVAL_PURPOSE_GATE) is True
    assert eval_allows_early_reject("gate-only") is True
    assert eval_allows_early_reject(EVAL_PURPOSE_SCORE) is False
    assert eval_allows_early_reject(EVAL_PURPOSE_MASTERY) is False
    assert eval_allows_early_reject(EVAL_PURPOSE_BEST) is False
    assert eval_allows_early_reject("best") is False
    with pytest.raises(ValueError, match="unknown eval purpose"):
        eval_allows_early_reject("warmup")


def test_cannot_pass_mastery_gate_is_irreversible():
    assert cannot_pass_mastery_gate(
        0, 7, 32, min_episodes=32, win_threshold=0.8, wilson_floor=0.50
    )
    assert not cannot_pass_mastery_gate(
        0, 6, 32, min_episodes=32, win_threshold=0.8, wilson_floor=0.50
    )
    assert cannot_pass_mastery_gate(
        25, 32, 32, min_episodes=32, win_threshold=0.8, wilson_floor=0.50
    )
    assert not cannot_pass_mastery_gate(
        26, 32, 32, min_episodes=32, win_threshold=0.8, wilson_floor=0.50
    )
    assert cannot_pass_mastery_gate(
        0, 0, 8, min_episodes=32, win_threshold=0.8, wilson_floor=0.50
    )
    assert wilson_lower_bound(26, 32) >= 0.50


def test_pending_scheduler_keeps_completed_and_rejects_duplicates():
    sched = PendingSeedScheduler(
        [{"seed": "a", "map": "Halkidiki"}, {"seed": "b", "map": "Halkidiki"}]
    )
    first = sched.claim(2)
    assert [c["seed"] for c in first] == ["a", "b"]
    sched.complete(
        {
            "seed": "a",
            "map": "Halkidiki",
            "won": True,
            "ret": 1.0,
            "peak": 0.4,
            "final": 0.4,
            "ticks": 3,
        }
    )
    sched.complete(
        {
            "seed": "a",
            "map": "Halkidiki",
            "won": False,
            "ret": 9.0,
            "peak": 0.9,
            "final": 0.9,
            "ticks": 9,
        }
    )
    sched.release_incomplete()
    assert sched.completed_count == 1
    assert sched.completed["a"]["won"] is True
    assert sched.pending_count == 1
    retry = sched.claim(2)
    assert [c["seed"] for c in retry] == ["b"]
    sched.complete(
        {
            "seed": "b",
            "map": "Halkidiki",
            "won": False,
            "ret": 0.0,
            "peak": 0.1,
            "final": 0.1,
            "ticks": 2,
        }
    )
    summary = sched.summarize(stage="duel", suite="holdout")
    assert summary["seeds"] == ["a", "b"]
    assert summary["episodes"] == 2
    assert summary["win_rate"] == pytest.approx(0.5)


def test_interrupt_retries_only_pending_seeds():
    env = _InterruptThenWinEnv(k=4)
    ev = run_fixed_seed_eval(
        Policy(),
        env,
        Curriculum(start=0),
        torch.device("cpu"),
        n_episodes=4,
        n_det=0,
        n_sampled=4,
        seed_prefix="recov",
    )
    samp = [f"recov-sampled-{i}" for i in range(4)]
    assert env.init_calls[0] == samp
    assert env.init_calls[1] == samp[1:]
    assert samp[0] not in env.init_calls[1]
    assert ev["sampled"]["episodes"] == 4
    assert ev["sampled"]["seeds"] == samp
    assert ev["sampled"]["win_rate"] == pytest.approx(1.0)
    assert ev["sampled"]["terminal_causes"] == {"win": 4}
    assert ev["early_rejected"] is False


def test_results_invariant_to_lane_count():
    cur = Curriculum(start=0)
    evs = []
    for k in (2, 5, 8):
        env = _SeedOutcomeEnv(k=k)
        evs.append(
            run_fixed_seed_eval(
                Policy(),
                env,
                cur,
                torch.device("cpu"),
                n_episodes=8,
                n_det=0,
                n_sampled=8,
                seed_prefix="lanes",
            )
        )
    seeds = [f"lanes-sampled-{i}" for i in range(8)]
    for ev in evs:
        assert ev["sampled"]["seeds"] == seeds
        assert ev["sampled"]["episodes"] == 8
        assert ev["sampled"]["win_rate"] == evs[0]["sampled"]["win_rate"]
        assert ev["sampled"]["terminal_causes"] == evs[0]["sampled"]["terminal_causes"]


def test_results_invariant_to_interrupt_failures():
    cur = Curriculum(start=0)
    clean = run_fixed_seed_eval(
        Policy(),
        _SeedOutcomeEnv(k=4, win_all=True),
        cur,
        torch.device("cpu"),
        n_episodes=4,
        n_det=0,
        n_sampled=4,
        seed_prefix="stab",
    )
    recovered = run_fixed_seed_eval(
        Policy(),
        _InterruptThenWinEnv(k=4),
        cur,
        torch.device("cpu"),
        n_episodes=4,
        n_det=0,
        n_sampled=4,
        seed_prefix="stab",
    )
    assert clean["sampled"]["seeds"] == recovered["sampled"]["seeds"]
    assert recovered["sampled"]["episodes"] == 4
    assert len(set(recovered["sampled"]["seeds"])) == 4
    assert clean["sampled"]["win_rate"] == recovered["sampled"]["win_rate"]
    assert clean["sampled"]["terminal_causes"] == recovered["sampled"]["terminal_causes"]


def test_gate_only_early_rejects_when_impossible():
    evs = []
    for k in (2, 8, 32):
        env = _SeedOutcomeEnv(k=k, lose_all=True)
        evs.append(
            run_fixed_seed_eval(
                Policy(),
                env,
                Curriculum(start=0),
                torch.device("cpu"),
                n_episodes=32,
                n_det=8,
                n_sampled=32,
                purpose=EVAL_PURPOSE_GATE,
            )
        )
    for ev in evs:
        assert ev["early_rejected"] is True
        assert ev["purpose"] == EVAL_PURPOSE_GATE
        assert ev["sampled"]["episodes"] == 7
        assert ev["sampled"]["win_rate"] == pytest.approx(0.0)
        assert ev["deterministic"]["episodes"] == 0
        assert ev["sampled"]["early_rejected"] is True


@pytest.mark.parametrize(
    "purpose",
    [EVAL_PURPOSE_SCORE, EVAL_PURPOSE_MASTERY, EVAL_PURPOSE_BEST],
)
def test_scoring_suites_always_complete(purpose):
    env = _SeedOutcomeEnv(k=8, lose_all=True)
    ev = run_fixed_seed_eval(
        Policy(),
        env,
        Curriculum(start=0),
        torch.device("cpu"),
        n_episodes=32,
        n_det=8,
        n_sampled=32,
        purpose=purpose,
    )
    assert ev["early_rejected"] is False
    assert ev["sampled"]["episodes"] == 32
    assert ev["deterministic"]["episodes"] == 8
    assert ev["sampled"]["win_rate"] == pytest.approx(0.0)
    assert ev["purpose"] == purpose


def test_legacy_interrupt_env_does_not_double_count():
    env = _EvalInterruptEnv(k=2)
    ev = run_fixed_seed_eval(
        Policy(),
        env,
        Curriculum(start=0),
        torch.device("cpu"),
        n_episodes=2,
        seed_prefix="legacy",
    )
    assert ev["deterministic"]["episodes"] == 2
    assert ev["sampled"]["episodes"] == 2
    assert ev["deterministic"]["seeds"] == [
        "legacy-deterministic-0",
        "legacy-deterministic-1",
    ]


def test_step_active_skips_finished_shards():
    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 1
    env.k = 4
    env.ports = [1, 2, 3, 4]
    env.connect = True
    env.servers = []
    env._configs = [{"seed": f"train-{i}"} for i in range(4)]
    env.pool = ThreadPoolExecutor(max_workers=4)
    env._obs_buf = None
    env._next_config_fn = None
    env._on_episode_reset = None
    env._active_assignment = []
    env._active_shards = []
    env._active_n = 0
    stepped: list[int] = []

    class FakeClient:
        def __init__(self, idx: int):
            self.idx = idx

        def init(self, cfgs):
            return _numpy_obs(len(cfgs))

        def step(self, actions, next_configs=None):
            stepped.append(self.idx)
            n = actions.shape[0]
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.ones(n, np.float32),
                dones=np.ones(n, dtype=bool),
                infos=[{"shard": self.idx} for _ in range(n)],
            )

    env.clients = [FakeClient(i) for i in range(4)]
    env.init_active([{"seed": f"e-{i}"} for i in range(3)])
    live = np.array([False, True, True])
    batch = env.step_active(np.zeros((3, 5), np.int32), live=live)
    assert 0 not in stepped
    assert sorted(stepped) == [1, 2]
    assert batch.dones[0]
    assert list(batch.dones[1:]) == [True, True]
    env.pool.shutdown(wait=False)


def test_rollout_interrupted_reports_failed_lanes():
    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 1
    env.k = 2
    env.ports = [9001, 9002]
    env.connect = False
    env._configs = [{}, {}]
    env.servers = []
    env._obs_buf = None
    env._next_config_fn = None
    env._on_episode_reset = None
    env._active_assignment = []
    env._active_shards = []
    env._active_n = 0
    env.pool = ThreadPoolExecutor(max_workers=2)

    class FakeClient:
        def __init__(self, idx: int):
            self.idx = idx

        def init(self, cfgs):
            if self.idx == 1:
                raise ConnectionError("shard down")
            return _numpy_obs(len(cfgs))

        def close(self):
            pass

    env.clients = [FakeClient(0), FakeClient(1)]
    env.servers = [type("S", (), {"alive": lambda self: True})() for _ in range(2)]
    env._restart_shard = lambda i: None
    with pytest.raises(RolloutInterrupted) as ei:
        env.init_active([{"seed": "a"}, {"seed": "b"}])
    assert 1 in ei.value.failed_shards
    assert 1 in ei.value.failed_lanes
    env.pool.shutdown(wait=False)


def test_always_interrupt_still_exhausts_retries():
    class Boom:
        def __init__(self):
            self.k = 2
            self.init_calls = 0

        def init(self, cfgs):
            self.init_calls += 1
            return _numpy_obs(self.k)

        def step(self, actions):
            raise RolloutInterrupted([0])

    env = Boom()
    with pytest.raises(RuntimeError, match="failed after 3 shard-restart"):
        run_fixed_seed_eval(
            Policy(),
            env,
            Curriculum(start=0),
            torch.device("cpu"),
            n_episodes=2,
            seed_prefix="boom",
        )
    assert env.init_calls == FIXED_SEED_EVAL_RETRIES
