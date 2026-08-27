"""Official-core eval harness: frozen seeds, stage config, Wilson, JSON.

Mocked — does not start training. Run:
  .venv/bin/python -m pytest tests/test_eval_harness.py -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import spaces
from agent.curriculum import (
    DEFAULT_DET_EVAL_EPISODES,
    DEFAULT_HOLDOUT_EPISODES,
    STAGES,
    wilson_interval,
)
from agent.env_client import StepBatch
from agent.eval import (
    GOAL_HOLDOUT_DET,
    GOAL_HOLDOUT_SAMPLED,
    GOAL_OOD_SAMPLED,
    GOAL_STAGE,
    WILSON_Z,
    curriculum_for_stage,
    frozen_suite_seeds,
    parse_args,
    resolve_eval_plan,
    run_eval,
    spawn_eval_server,
    wilson_report,
)
from agent.seeds import holdout_seed, ood_seed
from tests.test_rl_pipeline import _fake_info, _numpy_obs


def _plan(argv: list[str]):
    return resolve_eval_plan(parse_args(argv))


class _RecordingEnv:
    """Finishes every claimed seed on the first step. Outcome is seed-id."""

    def __init__(self):
        self.init_calls: list[list[str]] = []
        self.step_widths: list[int] = []
        self._cfgs: list[dict] = []

    def init(self, cfgs):
        self._cfgs = list(cfgs)
        self.init_calls.append([str(c["seed"]) for c in cfgs])
        return _numpy_obs(len(cfgs))

    def step(self, actions):
        n = int(actions.shape[0])
        self.step_widths.append(n)
        infos = []
        for i in range(n):
            seed = str(self._cfgs[i]["seed"])
            win = (int(seed[-2:], 16) % 2 == 0) if len(seed) >= 2 else False
            infos.append(
                _fake_info(
                    terminalCause="win" if win else "timeout",
                    win=win,
                    tick=10 + i,
                    peakTilesFrac=0.4 if win else 0.05,
                    tilesFrac=0.4 if win else 0.05,
                    actionAccepted=True,
                )
            )
        return StepBatch(
            obs=_numpy_obs(n),
            rewards=np.ones(n, np.float32),
            dones=np.ones(n, dtype=bool),
            infos=infos,
        )

    def close(self):
        pass


def test_goal_holdout_seed_identities_match_curriculum():
    plan = _plan(
        [
            "--stage",
            GOAL_STAGE,
            "--suite",
            "holdout",
            "--sampled-episodes",
            str(GOAL_HOLDOUT_SAMPLED),
            "--det-episodes",
            str(GOAL_HOLDOUT_DET),
        ]
    )
    expected = [holdout_seed(GOAL_STAGE, i) for i in range(GOAL_HOLDOUT_SAMPLED)]
    assert [c["seed"] for c in plan.sampled_configs] == expected
    assert [c["seed"] for c in plan.det_configs] == expected[:GOAL_HOLDOUT_DET]
    assert frozen_suite_seeds(GOAL_STAGE, "holdout", 64) == expected
    cur = curriculum_for_stage(GOAL_STAGE)
    assert cur.holdout_seeds(64) == expected
    assert plan.sampled_n == 64
    assert plan.det_n == 8


def test_goal_ood_uses_south_america_and_ood_seeds():
    plan = _plan(
        [
            "--stage",
            GOAL_STAGE,
            "--suite",
            "ood",
            "--sampled-episodes",
            str(GOAL_OOD_SAMPLED),
            "--det-episodes",
            "0",
        ]
    )
    expected = [ood_seed(GOAL_STAGE, i) for i in range(GOAL_OOD_SAMPLED)]
    assert [c["seed"] for c in plan.sampled_configs] == expected
    assert plan.det_n == 0
    assert {c["map"] for c in plan.sampled_configs} == {"SouthAmerica"}
    assert all(c["mapSize"] == "Normal" for c in plan.sampled_configs)
    assert all(c["difficulty"] == "Impossible" for c in plan.sampled_configs)


def test_goal_stage_defaults_are_64_8_and_16_ood():
    hold = _plan(["--stage", GOAL_STAGE])
    assert hold.suite == "holdout"
    assert hold.sampled_n == GOAL_HOLDOUT_SAMPLED
    assert hold.det_n == GOAL_HOLDOUT_DET
    ood = _plan(["--stage", GOAL_STAGE, "--suite", "ood"])
    assert ood.sampled_n == GOAL_OOD_SAMPLED
    assert ood.det_n == 0


def test_lane_count_does_not_change_seeds_or_win_rate():
    argv = [
        "--stage",
        GOAL_STAGE,
        "--sampled-episodes",
        "8",
        "--det-episodes",
        "0",
        "--scripted",
        "expander",
    ]
    seed_lists = []
    reports = []
    for envs in (2, 5, 8):
        plan = _plan([*argv, "--envs", str(envs)])
        env = _RecordingEnv()
        report = run_eval(None, env, plan, torch.device("cpu"))
        flat = [s for call in env.init_calls for s in call]
        seed_lists.append(flat)
        reports.append(report)
        assert flat == [c["seed"] for c in plan.sampled_configs]
    assert seed_lists[0] == seed_lists[1] == seed_lists[2]
    assert reports[0]["sampled"]["win_rate"] == reports[1]["sampled"]["win_rate"]
    assert reports[0]["sampled"]["win_rate"] == reports[2]["sampled"]["win_rate"]
    assert (
        reports[0]["sampled"]["terminal_causes"]
        == reports[1]["sampled"]["terminal_causes"]
    )


def test_impossible_stage_uses_full_actions_not_early_default():
    plan = _plan(
        ["--stage", GOAL_STAGE, "--sampled-episodes", "2", "--det-episodes", "0"]
    )
    assert plan.allowed_actions == list(spaces.FULL_ACTIONS)
    assert plan.allowed_actions != list(spaces.EARLY_ACTIONS)
    cfg = plan.sampled_configs[0]
    assert cfg["map"] == "World"
    assert cfg["mapSize"] == "Normal"
    assert cfg["nations"] == "default"
    assert cfg["difficulty"] == "Impossible"
    assert cfg["bots"] == 100
    assert cfg["maxTicks"] == 20000
    assert cfg["allowedActions"] == spaces.encode_allowed_actions(spaces.FULL_ACTIONS)
    assert cfg["shaping"] == 0.0
    manual = _plan(["--episodes", "2", "--mode", "sampled"])
    assert manual.allowed_actions == list(spaces.EARLY_ACTIONS)
    assert manual.sampled_configs[0]["map"] == "Halkidiki"


def test_duel_stage_keeps_land_boat_actions():
    plan = _plan(["--stage", "duel", "--sampled-episodes", "1", "--det-episodes", "0"])
    duel = next(s for s in STAGES if s.name == "duel")
    assert plan.allowed_actions == list(duel.allowed_actions)
    assert plan.map == "Halkidiki"
    assert plan.map_size == "Compact"
    assert plan.bots == 1
    assert plan.sampled_n == 1


def test_wilson_report_matches_curriculum_interval():
    for wins, n in ((0, 32), (26, 32), (52, 64), (0, 0), (8, 8)):
        wr = wilson_report(wins, n)
        lo, hi = wilson_interval(wins, n, z=WILSON_Z)
        assert wr["wilson_low"] == pytest.approx(lo)
        assert wr["wilson_high"] == pytest.approx(hi)
        assert wr["wins"] == (0 if n <= 0 else min(wins, n))
        if n:
            assert wr["win_rate"] == pytest.approx(wins / n)


def test_workers_propagated_to_env_server(monkeypatch, tmp_path):
    spawned: dict = {}

    class FakeServer:
        def __init__(
            self, port, cwd=None, timeout_s=120.0, workers=0, backend="official"
        ):
            spawned["port"] = int(port)
            spawned["workers"] = int(workers)

        def stop(self):
            spawned["stopped"] = True

    class FakeEnv(_RecordingEnv):
        def __init__(self, host, port, timeout_s=120.0):
            super().__init__()
            spawned["vec_port"] = int(port)

    monkeypatch.setattr("agent.eval.EnvServerProcess", FakeServer)
    monkeypatch.setattr("agent.eval.VecEnv", FakeEnv)
    from agent import eval as eval_mod

    out = tmp_path / "eval.json"
    eval_mod.main(
        [
            "--scripted",
            "expander",
            "--stage",
            "duel",
            "--sampled-episodes",
            "2",
            "--det-episodes",
            "0",
            "--workers",
            "3",
            "--port",
            "9899",
            "--envs",
            "2",
            "--json-out",
            str(out),
        ]
    )
    assert spawned["workers"] == 3
    assert spawned["port"] == 9899
    assert spawned["stopped"] is True
    payload = json.loads(out.read_text())
    assert payload["workers"] == 3
    spawn_eval_server(1, 4)
    assert spawned["workers"] == 4


def test_sampled_and_det_counts_are_independent():
    plan = _plan(
        [
            "--stage",
            GOAL_STAGE,
            "--sampled-episodes",
            "5",
            "--det-episodes",
            "2",
            "--scripted",
            "expander",
        ]
    )
    assert plan.sampled_n == 5
    assert plan.det_n == 2
    assert len(plan.sampled_configs) == 5
    assert len(plan.det_configs) == 2
    env = _RecordingEnv()
    report = run_eval(None, env, plan, torch.device("cpu"))
    assert report["sampled"]["episodes"] == 5
    assert report["deterministic"]["episodes"] == 2
    assert report["sampled"]["seeds"] == [holdout_seed(GOAL_STAGE, i) for i in range(5)]
    assert report["deterministic"]["seeds"] == [
        holdout_seed(GOAL_STAGE, i) for i in range(2)
    ]
    sampled_only = _plan(
        ["--stage", GOAL_STAGE, "--mode", "sampled", "--sampled-episodes", "3"]
    )
    assert sampled_only.sampled_n == 3
    assert sampled_only.det_n == 0
    det_only = _plan(
        ["--stage", GOAL_STAGE, "--mode", "deterministic", "--det-episodes", "4"]
    )
    assert det_only.sampled_n == 0
    assert det_only.det_n == 4
    manual = _plan(["--episodes", "6", "--mode", "sampled", "--envs", "4"])
    assert manual.sampled_n == 6
    assert manual.det_n == 0
    assert [c["seed"] for c in manual.sampled_configs] == [
        "eval-samp-0-0",
        "eval-samp-0-1",
        "eval-samp-0-2",
        "eval-samp-0-3",
        "eval-samp-4-0",
        "eval-samp-4-1",
    ]


def test_json_report_fields_and_hygiene():
    plan = _plan(
        [
            "--stage",
            GOAL_STAGE,
            "--sampled-episodes",
            "4",
            "--det-episodes",
            "1",
            "--scripted",
            "expander",
            "--workers",
            "2",
        ]
    )
    env = _RecordingEnv()
    report = run_eval(None, env, plan, torch.device("cpu"))
    for key in (
        "stage",
        "suite",
        "workers",
        "allowed_actions",
        "config",
        "sampled",
        "deterministic",
    ):
        assert key in report
    assert report["stage"] == GOAL_STAGE
    assert report["suite"] == "holdout"
    assert report["workers"] == 2
    samp = report["sampled"]
    for key in (
        "episodes",
        "wins",
        "win_rate",
        "wilson_low",
        "wilson_high",
        "wilson_z",
        "terminal_causes",
        "action_hist",
        "accept_rate",
        "hygiene",
        "seeds",
        "maps",
    ):
        assert key in samp
    assert samp["episodes"] == 4
    assert "noop_frac" in samp["hygiene"]
    assert "no_spawn_rate" in samp["hygiene"]
    assert report["config"]["difficulty"] == "Impossible"
    assert report["config"]["map"] == "World"
    assert samp["wilson_z"] == pytest.approx(WILSON_Z)
    assert report["deterministic"]["episodes"] == 1


def test_invalid_flag_combinations():
    with pytest.raises(ValueError, match="--suite requires --stage"):
        _plan(["--suite", "ood"])
    with pytest.raises(ValueError, match="unknown stage"):
        _plan(["--stage", "not-a-stage"])
    with pytest.raises(ValueError, match="--workers"):
        _plan(["--stage", "duel", "--workers", "-1"])
    with pytest.raises(ValueError, match="--sampled-episodes"):
        _plan(["--stage", "duel", "--sampled-episodes", "-2"])
    with pytest.raises(ValueError, match="--det-episodes"):
        _plan(["--stage", "duel", "--det-episodes", "-1"])
    with pytest.raises(ValueError, match="sampled cannot be combined"):
        _plan(
            [
                "--stage",
                GOAL_STAGE,
                "--mode",
                "sampled",
                "--det-episodes",
                "3",
            ]
        )
    with pytest.raises(ValueError, match="deterministic cannot be combined"):
        _plan(
            [
                "--stage",
                GOAL_STAGE,
                "--mode",
                "deterministic",
                "--sampled-episodes",
                "3",
            ]
        )
    with pytest.raises(SystemExit):
        parse_args(["--suite", "train"])


def test_manual_cli_defaults_are_unchanged():
    args = parse_args([])
    assert args.episodes == 20
    assert args.envs == 4
    assert args.port == 8770
    assert args.map == "Halkidiki"
    assert args.map_size == "Compact"
    assert args.bots == 1
    assert args.mode == "both"
    assert args.stage == ""
    plan = resolve_eval_plan(args)
    assert plan.suite == "manual"
    assert plan.sampled_n == 20
    assert plan.det_n == 20
    assert plan.workers == 0
    assert plan.allowed_actions == list(spaces.EARLY_ACTIONS)
    other = _plan(
        ["--stage", "small-ffa", "--sampled-episodes", "3", "--det-episodes", "1"]
    )
    assert other.sampled_n == 3
    assert other.det_n == 1
    assert other.sampled_n != DEFAULT_HOLDOUT_EPISODES
    assert DEFAULT_DET_EVAL_EPISODES == 8
