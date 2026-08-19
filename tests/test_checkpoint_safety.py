"""Regression tests for best.pt safety, metric windows, resume, and collapse guards.

Does not start training or modify repository checkpoints.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.checkpoint import (
    BEST_SCORE_SENTINEL,
    CollapsedCheckpointError,
    _eval_score,
    collapse_resume_reason,
    format_resume_plan,
    guard_collapsed_resume,
    is_sentinel_score,
    normalize_best_score,
    on_stage_advance_scores,
    plan_resume,
    resume_event_record,
    should_write_global_best,
    should_write_stage_best,
    stage_best_filename,
    training_budget_warning,
)
from agent.curriculum import STAGES, Curriculum
from agent.model import Policy
from agent.ppo import PPOConfig, PPOTrainer, TrainStats, atomic_torch_save
from agent.seeds import SeedStream
from agent.train import parse_args
from tests.test_rl_pipeline import _ep, _mode_stats

IT800 = ROOT / "checkpoints" / "v2run1" / "it800.pt"
_META_SKIP = frozenset({"model", "optimizer", "rng"})


def _competent_score() -> tuple[float, float, float, float]:
    return (1.0, 0.0, 0.8113855201678821, 0.0130677228863047)


def _zero_win_boat_eval() -> dict:
    """Confirmed post-advance collapsed best.pt eval (v2run1)."""
    return {
        "sampled": {
            "win_rate": 0.0,
            "mean_peak_tiles": 0.15649404982592968,
            "episodes": 32,
            "terminal_causes": {"death": 4, "loss_alive": 3, "timeout": 25},
            "action_hist": {
                "spawn": 32,
                "ally": 5170,
                "attack": 5658,
                "noop": 6344,
                "boat": 8758,
                "embargo": 5299,
                "retreat_all": 4014,
                "build": 7419,
                "break_ally": 272,
            },
        },
        "deterministic": {
            "win_rate": 0.0,
            "mean_peak_tiles": 0.027521995211981728,
            "episodes": 8,
            "terminal_causes": {"timeout": 8},
            "action_hist": {"boat": 800, "noop": 200},
        },
    }


def _passive_noop_eval() -> dict:
    return {
        "sampled": {
            "win_rate": 0.0,
            "mean_peak_tiles": 0.0002,
            "episodes": 32,
            "terminal_causes": {"timeout": 32},
            "action_hist": {"spawn": 32, "noop": 18172},
        },
        "deterministic": _mode_stats(0.0, peak=0.0),
    }


def _it800_like_meta(*, with_eval: dict | None = None) -> dict:
    meta = {
        "schema_version": 2,
        "curriculum": {
            "idx": 4,
            "stage_episodes": 9170,
            "mastery_version": 2,
            "consecutive_eval_passes": 1,
            "stage_successes": [1.0, 0.0],
        },
        "best_score": list(_competent_score()),
        "stage_baseline": {
            "win_rate": 1.0,
            "mean_return": 3.35,
            "env_steps": 2637824,
        },
        "stats": {
            "iterations": 800,
            "env_steps": 13_107_200,
            "episodes": 14131,
            "wins": 7785,
        },
        "metrics": {
            "win_20": 0.45,
            "win_100": 0.47,
            "win_500": 0.466,
            "return_100": 1.0908,
        },
        "seed_stream": {
            "run_name": "v2run1",
            "master_seed": 1,
            "n_envs": 8,
            "episode_index": [12, 11, 10, 9, 8, 7, 6, 5],
        },
    }
    if with_eval is not None:
        meta["eval"] = with_eval
        meta["last_eval"] = with_eval
    return meta


def _copy_it800_meta() -> dict:
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    ckpt = torch.load(IT800, map_location="cpu", weights_only=False)
    return copy.deepcopy({k: v for k, v in ckpt.items() if k not in _META_SKIP})


def test_stage_advance_zero_win_does_not_replace_global_best():
    """Exact bug: sentinel reset made a 0-win first eval replace best.pt."""
    competent = _competent_score()
    ev = _zero_win_boat_eval()
    score = _eval_score(ev)
    assert score > BEST_SCORE_SENTINEL
    kept, stage_bar = on_stage_advance_scores(competent)
    assert kept == competent
    assert is_sentinel_score(stage_bar)
    assert not should_write_global_best(score, kept, ev)
    assert not should_write_global_best(score, BEST_SCORE_SENTINEL, ev)


def test_zero_win_first_eval_does_not_write_best_pt_file(tmp_path):
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"competent-best")
    ev = _zero_win_boat_eval()
    score = _eval_score(ev)
    if should_write_global_best(score, BEST_SCORE_SENTINEL, ev):
        dest.write_bytes(b"overwritten")
    assert dest.read_bytes() == b"competent-best"


def test_collapsed_eval_cannot_replace_competent_best_file(tmp_path):
    dest = tmp_path / "best.pt"
    dest.write_bytes(b"competent-best")
    ev = _passive_noop_eval()
    score = _eval_score(ev)
    if should_write_global_best(score, _competent_score(), ev):
        dest.write_bytes(b"overwritten")
    assert dest.read_bytes() == b"competent-best"


def test_non_zero_win_can_write_new_run_best_pt():
    ev = {
        "sampled": _mode_stats(0.3, episodes=32, peak=0.4, cause="win"),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.2),
    }
    score = _eval_score(ev)
    assert should_write_global_best(score, BEST_SCORE_SENTINEL, ev)
    assert not should_write_global_best(score, _competent_score(), ev)


def test_stage_best_allows_zero_win_peak_but_not_collapse():
    peak_only = {
        "sampled": _mode_stats(0.0, episodes=32, peak=0.40),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.20),
    }
    score = _eval_score(peak_only)
    assert should_write_stage_best(score, BEST_SCORE_SENTINEL, peak_only)
    assert not should_write_stage_best(
        _eval_score(_passive_noop_eval()),
        BEST_SCORE_SENTINEL,
        _passive_noop_eval(),
    )
    assert stage_best_filename("nations-easy-peninsula") == (
        "best-nations-easy-peninsula.pt"
    )


def test_normalize_best_score_pads_legacy_triples():
    assert normalize_best_score([0.4, 0.1, 100.0]) == (0.4, 0.1, 100.0, 0.0)
    assert normalize_best_score(None) == BEST_SCORE_SENTINEL


def test_rolling_windows_reset_on_stage_transition_not_reinit():
    stats = TrainStats()
    for _ in range(25):
        stats.record_episode(_ep(win=True, terminal_cause="win"))
    assert stats.episodes == 25
    assert stats.wins == 25
    assert stats.win_rate_20 == pytest.approx(1.0)
    env_steps = 12_000
    stats.env_steps = env_steps
    stats.iterations = 10
    snap = stats.rolling_snapshot("nations-easy-small")
    assert snap["win_20"] == pytest.approx(1.0)
    assert snap["window_stage"] == "nations-easy-small"

    stats.reset_rolling_windows()
    assert stats.episodes == 25
    assert stats.wins == 25
    assert stats.env_steps == env_steps
    assert stats.iterations == 10
    assert stats.win_rate_20 == 0.0
    assert stats.win_rate == 0.0
    assert list(stats.wins_20) == []
    assert stats.rolling_snapshot("nations-easy-peninsula")["win_20"] == 0.0

    trainer = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2))
    trainer.stats = stats
    trainer._ensure_trackers(2)
    trainer._trackers[0].ret = 9.0
    trainer.reset_episode_trackers()
    assert trainer._trackers[0].ret == 0.0
    assert trainer.stats.episodes == 25
    assert trainer.stats.wins == 25
    assert list(trainer.stats.wins_20) == []


def test_force_stage_resets_windows_ordinary_resume_does_not():
    meta = _it800_like_meta()
    forced = plan_resume(
        ckpt=meta,
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v3run1",
        run_dir="/tmp/v3run1",
        stage_idx=next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-peninsula"),
        force_stage=True,
    )
    assert forced.reset_windows
    assert forced.reset_optimizer
    assert forced.need_baseline_eval
    assert not forced.curriculum.maps_unlocked
    assert not forced.restore_seed_stream
    assert not forced.restore_rng
    assert forced.curriculum.stage.name == "nations-easy-peninsula"
    assert forced.curriculum.stage_episodes == 0
    assert not forced.curriculum.last_eval_passed
    assert forced.curriculum.consecutive_eval_passes == 0
    assert is_sentinel_score(forced.stage_best_score)
    assert forced.source_best_score == _competent_score()
    assert not forced.same_run

    ordinary = plan_resume(
        ckpt=meta,
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v2run1",
        run_dir="/tmp/v2run1",
        stage_idx=None,
        force_stage=False,
    )
    assert not ordinary.reset_windows
    assert not ordinary.reset_optimizer
    assert not ordinary.need_baseline_eval
    assert ordinary.restore_seed_stream
    assert ordinary.curriculum.stage.name == "nations-easy-small"
    assert ordinary.curriculum.stage_episodes == 9170
    assert ordinary.global_best_score == _competent_score()


def test_force_stage_new_run_does_not_copy_or_touch_source(tmp_path):
    source_dir = tmp_path / "v2run1"
    source_dir.mkdir()
    src = source_dir / "it800.pt"
    payload = {"marker": 800, **_it800_like_meta()}
    src.write_bytes(b"source-bytes")
    before = src.read_bytes()
    mtime = src.stat().st_mtime_ns

    dest_dir = tmp_path / "v3run1"
    pen = next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-peninsula")
    plan = plan_resume(
        ckpt=payload,
        resume_path=str(src),
        run_name="v3run1",
        run_dir=dest_dir,
        stage_idx=pen,
        force_stage=True,
    )
    assert src.read_bytes() == before
    assert src.stat().st_mtime_ns == mtime
    assert not dest_dir.exists()
    assert plan.source_checkpoint == str(src)
    cfg = plan.curriculum.make_env_config("preview")
    assert cfg["map"] == "Halkidiki"
    assert cfg["mapSize"] == "Compact"
    assert cfg["nations"] == 8
    assert cfg["bots"] == 16
    assert cfg["difficulty"] == "Easy"
    from agent import spaces

    assert spaces.ACTION_BOAT in cfg["allowedActions"]
    assert spaces.ACTION_BUILD in cfg["allowedActions"]
    assert spaces.ACTION_ATTACK in cfg["allowedActions"]


def test_force_stage_seed_stream_uses_new_run_namespace():
    old = SeedStream("v2run1", 1, 8)
    old.episode_index = [12, 11, 10, 9, 8, 7, 6, 5]
    fresh = SeedStream("v3run1", 1, 8)
    assert fresh.episode_index == [0] * 8
    assert fresh.run_name == "v3run1"
    assert fresh.env_seed(0) != old.env_seed(0)
    assert fresh.env_seed(0) != SeedStream("v2run1", 1, 8).env_seed(0)


def test_force_stage_requires_stage():
    with pytest.raises(SystemExit, match="--force-stage requires --stage"):
        plan_resume(
            ckpt=_it800_like_meta(),
            resume_path="/tmp/x.pt",
            run_name="v3run1",
            run_dir="/tmp/v3run1",
            stage_idx=None,
            force_stage=True,
        )


def test_resume_event_boundary_has_source_and_restored_ids():
    rec = resume_event_record(
        source_checkpoint="checkpoints/v2run1/it800.pt",
        restored_it=800,
        restored_stage="nations-easy-peninsula",
        restored_env_steps=13_107_200,
        run_name="v3run1",
        force_stage=True,
        forced_stage="nations-easy-peninsula",
        windows_reset=True,
        seed_stream_reset=True,
    )
    assert rec["event"] == "resume"
    assert rec["source_checkpoint"].endswith("it800.pt")
    assert rec["restored_it"] == 800
    assert rec["restored_stage"] == "nations-easy-peninsula"
    blob = json.dumps(rec)
    assert "800" in blob
    assert "nations-easy-peninsula" in blob


def test_training_budget_warns_when_total_steps_already_passed():
    msg = training_budget_warning(13_107_200, 10_000_000)
    assert msg is not None
    assert "additional-steps" in msg
    assert training_budget_warning(13_107_200, 23_107_200) is None


def test_collapse_guard_refuses_eval_metadata_without_override():
    ckpt = _it800_like_meta(with_eval=_zero_win_boat_eval())
    assert collapse_resume_reason(ckpt)
    with pytest.raises(CollapsedCheckpointError, match="zero-win/passive collapse"):
        guard_collapsed_resume(
            ckpt, force_stage=False, allow_collapsed=False, source="best.pt"
        )
    warns = guard_collapsed_resume(
        ckpt, force_stage=True, allow_collapsed=False, source="best.pt"
    )
    assert warns
    warns2 = guard_collapsed_resume(
        ckpt, force_stage=False, allow_collapsed=True, source="best.pt"
    )
    assert warns2


def test_unevaluated_and_competent_checkpoints_remain_usable():
    fresh = {
        "stats": {"episodes": 0, "iterations": 0, "env_steps": 0},
        "metrics": {"win_100": 0.0, "return_100": 0.0},
    }
    assert collapse_resume_reason(fresh) is None
    assert guard_collapsed_resume(
        fresh, force_stage=False, allow_collapsed=False, source="untrained.pt"
    ) == []

    competent = _it800_like_meta()
    assert collapse_resume_reason(competent) is None
    warns = guard_collapsed_resume(
        competent, force_stage=False, allow_collapsed=False, source="it800.pt"
    )
    assert not any("refusing" in w for w in warns)


def test_metrics_collapse_warns_but_does_not_refuse_unevaluated_latest():
    latest = {
        "stats": {"episodes": 16000, "iterations": 975, "env_steps": 15_974_400},
        "metrics": {"win_20": 0.0, "win_100": 0.0, "return_100": -0.877},
    }
    assert collapse_resume_reason(latest) is None
    warns = guard_collapsed_resume(
        latest, force_stage=False, allow_collapsed=False, source="latest.pt"
    )
    assert any("win_100" in w for w in warns)


def test_atomic_save_roundtrip_leaves_no_tmp(tmp_path):
    path = tmp_path / "latest.pt"
    atomic_torch_save({"schema_version": 2, "n": 3}, path)
    assert path.is_file()
    assert list(tmp_path.glob("*.tmp")) == []
    loaded = torch.load(path, weights_only=False)
    assert loaded["n"] == 3
    atomic_torch_save({"schema_version": 2, "n": 4}, path)
    assert torch.load(path, weights_only=False)["n"] == 4
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_roundtrip_keeps_global_and_stage_best(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.episodes = 42
    trainer.stats.wins = 10
    trainer.stats.wins_20.append(1.0)
    cur = Curriculum(start=5)
    path = tmp_path / "ckpt.pt"
    trainer.save(
        str(path),
        extra={
            "curriculum": cur.state_dict(),
            "best_score": list(_competent_score()),
            "global_best_score": list(_competent_score()),
            "stage_best_score": [0.2, 0.0, 0.3, 0.1],
            "stage_best_stage": cur.stage.name,
            "source_best_score": list(_competent_score()),
        },
    )
    trainer2 = PPOTrainer(Policy(), cfg)
    loaded = trainer2.load(str(path))
    assert loaded["schema_version"] == 2
    assert loaded["best_score"][0] == pytest.approx(1.0)
    assert loaded["global_best_score"][0] == pytest.approx(1.0)
    assert loaded["stage_best_score"][0] == pytest.approx(0.2)
    json.dumps(loaded["curriculum"])
    json.dumps(loaded["stats"])
    trainer.stats.reset_rolling_windows()
    trainer.save(str(tmp_path / "after-reset.pt"), extra={"curriculum": cur.state_dict()})
    trainer3 = PPOTrainer(Policy(), cfg)
    loaded3 = trainer3.load(str(tmp_path / "after-reset.pt"))
    assert trainer3.stats.episodes == 42
    assert trainer3.stats.wins == 10
    assert loaded3["stats"]["wins_20"] == []


def test_legacy_best_score_only_checkpoint_loads(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    path = tmp_path / "legacy.pt"
    trainer.save(
        str(path),
        extra={
            "best_score": [0.4, 0.1, 0.2],
            "curriculum": {"idx": 0, "mastery_version": 3, "stage_episodes": 0},
        },
    )
    loaded = PPOTrainer(Policy(), cfg).load(str(path))
    plan = plan_resume(
        ckpt=loaded,
        resume_path=str(path),
        run_name="cont",
        run_dir=path.parent,
        stage_idx=None,
        force_stage=False,
    )
    assert plan.global_best_score == (0.4, 0.1, 0.2, 0.0)


def test_collect_interrupt_does_not_commit_env_steps():
    from agent.env_client import RolloutInterrupted, StepBatch
    from agent.ppo import RolloutBuffer
    from tests.test_rl_pipeline import _fake_info, _numpy_obs

    cfg = PPOConfig(rollout_len=4, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)

    class Boom:
        k = 1

        def step(self, actions):
            raise RolloutInterrupted([0])

    with pytest.raises(RolloutInterrupted):
        trainer.collect(Boom(), _numpy_obs(1), RolloutBuffer(4, 1))
    assert trainer.stats.env_steps == 0
    assert trainer.stats.iterations == 0
    assert trainer.stats.episodes == 0


def test_dry_run_cli_writes_nothing(tmp_path, monkeypatch, capsys):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.iterations = 800
    trainer.stats.env_steps = 13_107_200
    src = tmp_path / "it800.pt"
    trainer.save(str(src), extra=_it800_like_meta())
    dest_name = "dry-v3-test"
    monkeypatch.setattr("agent.train.CKPT_DIR", tmp_path)
    monkeypatch.setattr("agent.train.LOG_DIR", tmp_path / "logs")
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--resume",
            str(src),
            "--run-name",
            dest_name,
            "--stage",
            "nations-easy-peninsula",
            "--force-stage",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert "nations-easy-peninsula" in out
    assert "force_stage=True" in out
    assert "reset_windows=True" in out
    assert "reset_optimizer=True" in out
    assert "need_baseline_eval=True" in out
    assert "pre_update_baseline=frozen sampled holdout" in out
    assert "maps_unlocked=False" in out
    assert "eval_every=0" in out
    assert "eval-every 0" in out
    assert not (tmp_path / dest_name).exists()
    assert not (tmp_path / "logs" / f"{dest_name}.jsonl").exists()
    assert src.is_file()


def test_parse_args_new_flags():
    args = parse_args(
        [
            "--resume",
            "checkpoints/v2run1/it800.pt",
            "--stage",
            "nations-easy-peninsula",
            "--force-stage",
            "--allow-collapsed-resume",
            "--dry-run",
            "--additional-steps",
            "1000000",
            "--run-name",
            "v3run1",
        ]
    )
    assert args.force_stage
    assert not args.keep_optimizer
    assert args.allow_collapsed_resume
    assert args.dry_run
    assert args.additional_steps == 1_000_000
    assert args.run_name == "v3run1"
    assert STAGES[args.stage].name == "nations-easy-peninsula"


def test_it800_metadata_force_stage_smoke():
    """Copied it800 metadata only — never writes back to v2run1."""
    meta = _copy_it800_meta()
    mtime = IT800.stat().st_mtime_ns
    pen = next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-peninsula")
    plan = plan_resume(
        ckpt=meta,
        resume_path=str(IT800),
        run_name="v3run1",
        run_dir=ROOT / "checkpoints" / "v3run1",
        stage_idx=pen,
        force_stage=True,
    )
    assert IT800.stat().st_mtime_ns == mtime
    assert plan.curriculum.stage.name == "nations-easy-peninsula"
    assert plan.curriculum.stage_episodes == 0
    assert plan.reset_windows
    assert not plan.restore_seed_stream
    assert plan.source_best_score[0] == pytest.approx(1.0)
    assert plan.source_best_score[2] == pytest.approx(0.8113855201678821)
    assert is_sentinel_score(plan.global_best_score)
    report = format_resume_plan(plan, env_steps=13_107_200, total_steps=10_000_000)
    assert "additional-steps" in report
    assert "nations-easy-peninsula" in report
    cfg = plan.curriculum.make_env_config("smoke")
    assert cfg["map"] == "Halkidiki"
    assert cfg["nations"] == 8
    assert cfg["bots"] == 16


def test_it800_schema_v2_weight_roundtrip(tmp_path):
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    mtime = IT800.stat().st_mtime_ns
    size = IT800.stat().st_size
    cfg = PPOConfig(device="cpu", rollout_len=128, num_minibatches=8)
    trainer = PPOTrainer(Policy(), cfg)
    ckpt = trainer.load(str(IT800))
    assert ckpt["schema_version"] == 2
    assert trainer.stats.iterations == 800
    assert trainer.stats.env_steps == 13_107_200
    out = tmp_path / "it800-roundtrip.pt"
    trainer.save(str(out), extra={"curriculum": ckpt.get("curriculum", {})})
    trainer2 = PPOTrainer(Policy(), cfg)
    loaded = trainer2.load(str(out))
    assert loaded["schema_version"] == 2
    assert trainer2.stats.iterations == 800
    assert IT800.stat().st_mtime_ns == mtime
    assert IT800.stat().st_size == size


def test_same_run_force_stage_keeps_global_write_bar():
    pen = next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-peninsula")
    plan = plan_resume(
        ckpt=_it800_like_meta(),
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v2run1",
        run_dir="/tmp/v2run1",
        stage_idx=pen,
        force_stage=True,
    )
    assert plan.same_run
    assert plan.global_best_score == _competent_score()
    assert any("new --run-name" in w for w in plan.warnings)
    ev = _zero_win_boat_eval()
    assert not should_write_global_best(
        _eval_score(ev), plan.global_best_score, ev
    )
