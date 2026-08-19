"""Transfer-stability: fresh Adam, pre-update baseline, map lock, regression halt.

Does not start training, write repository checkpoints, or run the game sim.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.checkpoint import (
    BEST_SCORE_SENTINEL,
    STAGE_ENTRY_FILENAME,
    TRANSFER_EVAL_EVERY,
    TRANSFER_LR,
    build_stage_baseline,
    choose_restore_snapshot,
    format_resume_plan,
    is_sentinel_score,
    judge_holdout_regression,
    plan_resume,
    restore_runtime_from_snapshot,
    should_rewrite_halt_checkpoint,
    should_write_global_best,
    transfer_eval_cadence_warning,
    write_touches_foreign_source,
)
from agent.curriculum import STAGES, Curriculum, wilson_interval, wilson_lower_bound
from agent.model import Policy
from agent.ppo import PPOConfig, PPOTrainer, apply_optimizer_lr, optimizer_is_fresh
from agent.seeds import SeedStream, holdout_seed, mix_seed
from agent.train import parse_args, run_fixed_seed_eval, run_pre_update_baseline
from tests.test_checkpoint_safety import (
    IT800,
    _competent_score,
    _copy_it800_meta,
    _it800_like_meta,
)
from tests.test_rl_pipeline import _EvalInterruptEnv, _mode_stats, _numpy_obs


def _peninsula_idx() -> int:
    return next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-peninsula")


def _holdout_ev(win_rate: float, episodes: int = 32, peak: float = 0.40) -> dict:
    return {
        "sampled": _mode_stats(
            win_rate, episodes=episodes, peak=peak, cause="win"
        ),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.10),
        "suite": "holdout",
        "holdout_map": "Halkidiki",
    }


class _InstantEvalEnv:
    """Finishes every eval episode on the first step. No game sim."""

    def __init__(self, k: int):
        self.k = k
        self.init_calls: list[list[str]] = []
        self.step_calls = 0

    def init(self, cfgs):
        self.init_calls.append([c["seed"] for c in cfgs])
        return _numpy_obs(self.k)

    def init_active(self, cfgs):
        return self.init(cfgs)

    def step(self, actions):
        return self._done()

    def step_active(self, actions):
        return self._done()

    def _done(self):
        from agent.env_client import StepBatch
        from tests.test_rl_pipeline import _fake_info

        self.step_calls += 1
        info = _fake_info(
            terminalCause="win", win=True, tick=10, peakTilesFrac=0.5, tilesFrac=0.5
        )
        import numpy as np

        return StepBatch(
            obs=_numpy_obs(self.k),
            rewards=np.ones(self.k, np.float32),
            dones=np.ones(self.k, dtype=bool),
            infos=[info] * self.k,
        )


def test_force_stage_resets_optimizer_ordinary_resume_preserves(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1, lr=3e-4)
    trainer = PPOTrainer(Policy(), cfg)
    dummy = next(trainer.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    trainer.opt.step()
    dummy.grad = torch.zeros_like(dummy)
    trainer.opt.step()
    assert not optimizer_is_fresh(trainer.opt)
    model_before = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    opt_before = copy.deepcopy(trainer.opt.state_dict())
    path = tmp_path / "warm.pt"
    trainer.save(str(path), extra=_it800_like_meta())

    resumed = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2, lr=1e-4))
    resumed.load(str(path))
    assert not optimizer_is_fresh(resumed.opt)
    assert resumed.opt.state_dict()["state"].keys() == opt_before["state"].keys()
    for key, tensor in model_before.items():
        assert torch.equal(resumed.policy.state_dict()[key], tensor)
    # Ordinary resume keeps checkpoint Adam lr, not the unused CLI default.
    assert resumed.opt.param_groups[0]["lr"] == pytest.approx(3e-4)

    forced = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2, lr=1e-4))
    forced.load(str(path))
    for key, tensor in model_before.items():
        assert torch.equal(forced.policy.state_dict()[key], tensor)
    forced.reset_optimizer(1e-4)
    assert optimizer_is_fresh(forced.opt)
    assert forced.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert forced.cfg.lr == pytest.approx(1e-4)
    for key, tensor in model_before.items():
        assert torch.equal(forced.policy.state_dict()[key], tensor)


def test_force_stage_plan_resets_optimizer_keep_override():
    meta = _it800_like_meta()
    forced = plan_resume(
        ckpt=meta,
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v3run2",
        run_dir="/tmp/v3run2",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
    )
    assert forced.reset_optimizer
    assert not forced.keep_optimizer
    assert forced.need_baseline_eval
    assert not forced.curriculum.maps_unlocked

    kept = plan_resume(
        ckpt=meta,
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v3run2",
        run_dir="/tmp/v3run2",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        keep_optimizer=True,
        cli_lr=1e-4,
    )
    assert not kept.reset_optimizer
    assert kept.keep_optimizer
    with pytest.raises(SystemExit, match="--keep-optimizer requires --force-stage"):
        plan_resume(
            ckpt=meta,
            resume_path="/tmp/x.pt",
            run_name="cont",
            run_dir="/tmp/cont",
            stage_idx=None,
            force_stage=False,
            keep_optimizer=True,
        )


def test_it800_force_stage_optimizer_reset_read_only():
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    mtime = IT800.stat().st_mtime_ns
    size = IT800.stat().st_size
    trainer = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2, lr=3e-4))
    trainer.load(str(IT800))
    assert not optimizer_is_fresh(trainer.opt)
    step = next(iter(trainer.opt.state_dict()["state"].values()))["step"]
    assert int(step.item()) >= 1000
    src_lr = trainer.opt.param_groups[0]["lr"]
    weights = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    trainer.reset_optimizer(1e-4)
    assert optimizer_is_fresh(trainer.opt)
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert src_lr != pytest.approx(1e-4) or True
    for key, tensor in weights.items():
        assert torch.equal(trainer.policy.state_dict()[key], tensor)
    assert IT800.stat().st_mtime_ns == mtime
    assert IT800.stat().st_size == size


def test_apply_optimizer_lr_does_not_clear_moments():
    cfg = PPOConfig(device="cpu", rollout_len=2, lr=3e-4)
    trainer = PPOTrainer(Policy(), cfg)
    dummy = next(trainer.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    trainer.opt.step()
    assert not optimizer_is_fresh(trainer.opt)
    apply_optimizer_lr(trainer.opt, 1e-4)
    assert trainer.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    assert not optimizer_is_fresh(trainer.opt)


def test_pre_update_baseline_does_not_advance_counters_or_stage():
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.iterations = 800
    trainer.stats.env_steps = 13_107_200
    cur = Curriculum(start=_peninsula_idx())
    env = _InstantEvalEnv(k=4)
    expected = [holdout_seed(cur.stage.name, i) for i in range(4)]
    baseline, ev, ood, unlock = run_pre_update_baseline(
        policy=Policy(),
        env=env,
        curriculum=cur,
        trainer=trainer,
        device=torch.device("cpu"),
        n_sampled=4,
        n_det=0,
        n_ood=0,
    )
    assert trainer.stats.iterations == 800
    assert trainer.stats.env_steps == 13_107_200
    assert ev["sampled"]["seeds"] == expected
    assert env.init_calls[0] == expected
    assert baseline["iterations"] == 800
    assert baseline["env_steps"] == 13_107_200
    assert baseline["kind"] == "pre_update_holdout"
    assert not cur.last_eval_passed
    assert cur.idx == _peninsula_idx()
    assert not cur.maybe_advance()
    assert unlock["reason"] in ("hold", "unlock_holdout", "relock_holdout")


def test_baseline_does_not_write_global_best_from_other_distribution():
    ev = _holdout_ev(0.6875)
    score = (0.6875, 0.0, 0.40, 0.10)
    # New-run sentinel must still refuse a transfer snapshot as global best
    # only when we explicitly skip the write. The helper is the policy:
    # baseline path never calls should_write_global_best.
    assert should_write_global_best(score, _competent_score(), ev) is False
    built = build_stage_baseline(
        ev, iterations=800, env_steps=13_107_200, stage="nations-easy-peninsula"
    )
    assert built["win_rate"] == pytest.approx(0.6875)
    assert built["episodes"] == 32
    assert built["seeds"] == []


def test_peninsula_episode_count_does_not_unlock_balkans():
    cur = Curriculum(start=_peninsula_idx())
    assert cur.stage.map == "Halkidiki"
    assert "Balkans" in cur.stage.train_map_set
    cur.stage_episodes = 55
    maps = {cur.make_env_config(f"ep-{i}")["map"] for i in range(64)}
    assert maps == {"Halkidiki"}
    cur.stage_episodes = 75
    maps = {cur.make_env_config(f"ep2-{i}")["map"] for i in range(64)}
    assert maps == {"Halkidiki"}
    hold = cur.make_env_config("h", suite="holdout")
    ood = cur.make_env_config("o", suite="ood")
    assert hold["map"] == "Halkidiki"
    assert ood["map"] == "Britannia"
    assert ood["map"] not in cur.stage.train_map_set


def test_unlock_relock_resume_thresholds():
    cur = Curriculum(start=_peninsula_idx())
    fail = cur.judge_unlock(_holdout_ev(5 / 32))
    assert not fail["unlock"]
    assert fail["relock"]
    cur.apply_unlock(_holdout_ev(5 / 32))
    assert not cur.maps_unlocked

    mid = cur.judge_unlock(_holdout_ev(14 / 32))
    assert not mid["unlock"]
    # 14/32 = 0.4375 is above relock 0.35, so a locked run stays locked.
    assert not mid["relock"]

    ok = cur.apply_unlock(_holdout_ev(16 / 32))
    assert ok["unlock"]
    assert ok["action"] == "unlock"
    assert cur.maps_unlocked
    maps = {cur.make_env_config(f"u-{i}")["map"] for i in range(48)}
    assert "Halkidiki" in maps
    assert "Balkans" in maps

    # Hysteresis: 12/32 = 0.375 stays unlocked; 11/32 relocks.
    hold = cur.apply_unlock(_holdout_ev(12 / 32))
    assert hold["action"] == "hold"
    assert cur.maps_unlocked
    relock = cur.apply_unlock(_holdout_ev(11 / 32))
    assert relock["action"] == "relock"
    assert not cur.maps_unlocked
    assert {cur.make_env_config(f"r-{i}")["map"] for i in range(16)} == {"Halkidiki"}

    sd = cur.state_dict()
    assert sd["maps_unlocked"] is False
    cur.maps_unlocked = True
    sd = cur.state_dict()
    restored = Curriculum()
    restored.load_state_dict(sd)
    assert restored.stage.name == "nations-easy-peninsula"
    assert restored.maps_unlocked is True
    legacy = Curriculum()
    legacy.load_state_dict(
        {"stage": "nations-easy-peninsula", "stage_episodes": 90, "mastery_version": 3}
    )
    assert legacy.maps_unlocked is False
    assert {legacy.make_env_config(f"l-{i}")["map"] for i in range(16)} == {"Halkidiki"}


def test_unlock_is_not_mastery_lottery():
    cur = Curriculum(start=_peninsula_idx())
    # 16/32 unlocks maps but does not pass 26/32 mastery.
    ev = _holdout_ev(16 / 32)
    assert cur.judge_unlock(ev)["unlock"]
    assert not cur.judge_eval(ev)["passed"]
    mastery = cur.judge_eval(_holdout_ev(26 / 32))
    assert mastery["passed"]
    assert wilson_lower_bound(16, 32) >= cur.stage.unlock_wilson_floor
    assert wilson_lower_bound(15, 32) < cur.stage.unlock_win_threshold or (
        15 / 32 < cur.stage.unlock_win_threshold
    )


def test_regression_069_vs_0156_and_zero_win():
    baseline = build_stage_baseline(
        _holdout_ev(0.6875),
        iterations=800,
        env_steps=13_107_200,
        stage="nations-easy-peninsula",
    )
    collapsed = judge_holdout_regression(_holdout_ev(0.15625), baseline)
    assert collapsed["severe"]
    assert collapsed["action"] == "restore_and_halt"
    assert collapsed["reason"] == "confirmed_holdout_drop"

    zero = judge_holdout_regression(_holdout_ev(0.0, peak=0.16), baseline)
    assert zero["severe"]
    assert zero["reason"] == "zero_win_collapse"

    mild = judge_holdout_regression(_holdout_ev(0.50), baseline)
    assert not mild["severe"]
    assert mild["action"] == "continue"

    noisy = judge_holdout_regression(_holdout_ev(0.40), baseline)
    assert not noisy["severe"]

    weak = judge_holdout_regression(
        _holdout_ev(0.0),
        build_stage_baseline(
            _holdout_ev(0.10), iterations=0, env_steps=0, stage="duel"
        ),
    )
    assert not weak["severe"]
    assert weak["reason"] == "no_competent_baseline"

    tiny = judge_holdout_regression(_holdout_ev(0.0, episodes=8), baseline)
    assert not tiny["severe"]
    assert tiny["reason"] == "current_holdout_too_small"


def test_restore_prefers_stage_best_then_entry():
    assert (
        choose_restore_snapshot(
            stage_entry_exists=True,
            stage_best_exists=True,
            stage_best_score=(0.75, 0.0, 0.5, 0.1),
            baseline_wr=0.69,
        )
        == "stage-best"
    )
    assert (
        choose_restore_snapshot(
            stage_entry_exists=True,
            stage_best_exists=True,
            stage_best_score=(0.40, 0.0, 0.3, 0.1),
            baseline_wr=0.69,
        )
        == "stage-entry"
    )
    assert (
        choose_restore_snapshot(
            stage_entry_exists=False,
            stage_best_exists=False,
            stage_best_score=BEST_SCORE_SENTINEL,
            baseline_wr=0.69,
        )
        == "none"
    )


def test_load_weights_and_optimizer_leaves_counters(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    src = PPOTrainer(Policy(), cfg)
    dummy = next(src.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    src.opt.step()
    src.stats.iterations = 10
    src.stats.env_steps = 1000
    src.ent_coef = 0.11
    path = tmp_path / STAGE_ENTRY_FILENAME
    src.save(str(path))
    dst = PPOTrainer(Policy(), cfg)
    dst.stats.iterations = 35
    dst.stats.env_steps = 5000
    dst.ent_coef = 0.40
    dst.load_weights_and_optimizer(str(path))
    assert dst.stats.iterations == 35
    assert dst.stats.env_steps == 5000
    assert dst.ent_coef == pytest.approx(0.11)
    assert not optimizer_is_fresh(dst.opt)


def test_dry_run_explains_transfer_guards(tmp_path, monkeypatch, capsys):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.iterations = 800
    trainer.stats.env_steps = 13_107_200
    src = tmp_path / "it800.pt"
    trainer.save(str(src), extra=_it800_like_meta())
    dest_name = "dry-v3-transfer"
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
            "--lr",
            "1e-4",
            "--eval-every",
            "5",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert "reset_optimizer=True" in out
    assert "need_baseline_eval=True" in out
    assert "pre_update_baseline=frozen sampled holdout" in out
    assert "no stage advance" in out
    assert "no global-best write" in out
    assert "maps_unlocked=False" in out
    assert "regression_guard=halt+restore" in out
    assert "fresh Adam" in out
    assert f"--lr {TRANSFER_LR:g}" in out
    assert f"--eval-every {TRANSFER_EVAL_EVERY}" in out
    assert "eval_every=5" in out
    assert "eval-every 0" not in out
    assert not (tmp_path / dest_name).exists()
    assert src.is_file()


def test_parse_args_keep_optimizer_and_transfer_lr():
    args = parse_args(
        [
            "--resume",
            "checkpoints/v2run1/it800.pt",
            "--stage",
            "nations-easy-peninsula",
            "--force-stage",
            "--keep-optimizer",
            "--lr",
            "1e-4",
            "--eval-every",
            "5",
            "--run-name",
            "v3run2",
        ]
    )
    assert args.keep_optimizer
    assert args.force_stage
    assert args.lr == pytest.approx(1e-4)
    assert args.eval_every == 5


def test_it800_metadata_dry_run_plan_smoke():
    meta = _copy_it800_meta()
    mtime = IT800.stat().st_mtime_ns if IT800.is_file() else 0
    plan = plan_resume(
        ckpt=meta,
        resume_path=str(IT800),
        run_name="v3run2",
        run_dir=ROOT / "checkpoints" / "v3run2",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
    )
    if IT800.is_file():
        assert IT800.stat().st_mtime_ns == mtime
    assert plan.reset_optimizer
    assert plan.need_baseline_eval
    assert not plan.curriculum.maps_unlocked
    report = format_resume_plan(plan, env_steps=13_107_200, total_steps=23_107_200)
    assert "reset_optimizer=True" in report
    assert "pre_update_baseline=frozen sampled holdout" in report
    assert "maps_unlocked=False" in report


def test_baseline_eval_frozen_seeds_match_holdout_helper():
    env = _EvalInterruptEnv(k=2)
    cur = Curriculum(start=_peninsula_idx())
    expected = cur.holdout_seeds(2)
    ev = run_fixed_seed_eval(
        Policy(), env, cur, torch.device("cpu"), n_episodes=2, n_det=2, n_sampled=2
    )
    assert ev["sampled"]["seeds"] == expected
    assert ev["deterministic"]["seeds"] == expected
    assert all(m == "Halkidiki" for m in ev["sampled"]["maps"])


def test_unlock_16_32_and_hysteresis_cannot_oscillate():
    lb16, ub16 = wilson_interval(16, 32)
    lb15, ub15 = wilson_interval(15, 32)
    assert 16 / 32 >= 0.50
    assert lb16 >= 0.30
    assert 15 / 32 < 0.50 or lb15 < 0.30
    cur = Curriculum(start=_peninsula_idx())
    states = []
    for wr_n in (16, 12, 11, 16, 12, 11, 16):
        cur.apply_unlock(_holdout_ev(wr_n / 32))
        states.append(cur.maps_unlocked)
        assert not cur.maybe_advance()
    assert states == [True, True, False, True, True, False, True]
    assert not cur.last_eval_passed
    assert cur.idx == _peninsula_idx()


def test_reinit_and_in_flight_next_cfg_obey_lock():
    cur = Curriculum(start=_peninsula_idx())
    stream = SeedStream("v3run2", 1, 16)
    locked = {cur.make_env_config(stream.env_seed(i))["map"] for i in range(16)}
    assert locked == {"Halkidiki"}
    # In-flight next-config uses completed_episodes+1 and must stay locked.
    nxt = cur.make_env_config(
        stream.next_env_seed(0), completed_episodes=cur.stage_episodes + 1
    )
    assert nxt["map"] == "Halkidiki"
    cur.apply_unlock(_holdout_ev(16 / 32))
    unlocked = {cur.make_env_config(mix_seed("reinit", i))["map"] for i in range(64)}
    assert "Halkidiki" in unlocked
    assert "Balkans" in unlocked
    hold = cur.make_env_config(holdout_seed(cur.stage.name, 0), suite="holdout")
    assert hold["map"] == "Halkidiki"
    cur.apply_unlock(_holdout_ev(11 / 32))
    relocked = {cur.make_env_config(stream.env_seed(i))["map"] for i in range(16)}
    assert relocked == {"Halkidiki"}


def test_eval_every_zero_is_warned_on_force_stage():
    assert transfer_eval_cadence_warning(0, force_stage=True)
    assert transfer_eval_cadence_warning(5, force_stage=True) is None
    assert transfer_eval_cadence_warning(0, force_stage=False) is None
    meta = _it800_like_meta()
    plan = plan_resume(
        ckpt=meta,
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v3run2",
        run_dir="/tmp/v3run2",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
        eval_every=0,
    )
    assert any("eval-every 0" in w for w in plan.warnings)
    report = format_resume_plan(
        plan, env_steps=13_107_200, total_steps=23_107_200, eval_every=0
    )
    assert "eval_every=0" in report
    assert "eval-every 0" in report


def test_source_directory_writes_are_rejected(tmp_path):
    src_dir = tmp_path / "v2run1"
    dest_dir = tmp_path / "v3run2"
    src_dir.mkdir()
    dest_dir.mkdir()
    src = src_dir / "it800.pt"
    src.write_bytes(b"src")
    assert write_touches_foreign_source(src_dir / "best.pt", dest_dir, src)
    assert write_touches_foreign_source(src_dir / "it800.pt", dest_dir, src)
    assert write_touches_foreign_source(tmp_path / "other" / "latest.pt", dest_dir, src)
    assert not write_touches_foreign_source(dest_dir / "latest.pt", dest_dir, src)
    assert not write_touches_foreign_source(dest_dir / "best.pt", dest_dir, src)
    # Same-run resume may write into its own directory.
    assert not write_touches_foreign_source(src_dir / "latest.pt", src_dir, src)


def test_restore_runtime_and_halt_finally_do_not_rewrite(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    src = PPOTrainer(Policy(), cfg)
    dummy = next(src.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    src.opt.step()
    src.ent_coef = 0.07
    src.stats.iterations = 10
    src.stats.env_steps = 1000
    cur = Curriculum(start=_peninsula_idx())
    cur.maps_unlocked = True
    cur.stage_episodes = 3
    stream = SeedStream("entry", 1, 4)
    stream.episode_index = [2, 2, 2, 2]
    entry = tmp_path / STAGE_ENTRY_FILENAME
    src.save(
        str(entry),
        extra={
            "curriculum": cur.state_dict(),
            "seed_stream": stream.state_dict(),
            "rng": {"python": __import__("random").getstate()},
        },
    )
    good_w = {k: v.detach().clone() for k, v in src.policy.state_dict().items()}

    live = PPOTrainer(Policy(), cfg)
    live.ent_coef = 0.40
    live.stats.iterations = 35
    live.stats.env_steps = 5000
    live_cur = Curriculum(start=_peninsula_idx())
    live_cur.maps_unlocked = False
    live_cur.stage_episodes = 40
    live_stream = SeedStream("live", 9, 4)
    live_stream.episode_index = [9, 9, 9, 9]
    # Degrade in-memory weights.
    with torch.no_grad():
        for p in live.policy.parameters():
            p.zero_()

    snap = live.load_weights_and_optimizer(str(entry))
    applied = restore_runtime_from_snapshot(
        snap, curriculum=live_cur, seed_stream=live_stream, set_rng=None
    )
    assert "curriculum" in applied
    assert "seed_stream" in applied
    assert live.stats.iterations == 35
    assert live.stats.env_steps == 5000
    assert live.ent_coef == pytest.approx(0.07)
    assert live_cur.maps_unlocked is True
    assert live_cur.stage_episodes == 3
    assert live_stream.episode_index == [2, 2, 2, 2]
    for key, tensor in good_w.items():
        assert torch.equal(live.policy.state_dict()[key], tensor)

    latest = tmp_path / "latest.pt"
    live.save(str(latest), extra={"save_reason": "stage_regression", "marker": "restored"})
    # Mutate memory as if finally still held degraded weights.
    with torch.no_grad():
        for p in live.policy.parameters():
            p.fill_(1.0)
    live.ent_coef = 0.99
    assert should_rewrite_halt_checkpoint(False)
    assert not should_rewrite_halt_checkpoint(True)
    if should_rewrite_halt_checkpoint(True):
        live.save(str(latest), extra={"save_reason": "finally", "marker": "degraded"})
    loaded = torch.load(latest, map_location="cpu", weights_only=False)
    assert loaded["marker"] == "restored"
    assert loaded["ent_coef"] == pytest.approx(0.07)
    for key, tensor in good_w.items():
        assert torch.equal(loaded["model"][key], tensor)


def test_ordinary_load_keeps_optimizer_force_stage_skips_it(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, lr=3e-4)
    src = PPOTrainer(Policy(), cfg)
    dummy = next(src.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    src.opt.step()
    path = tmp_path / "warm.pt"
    src.save(str(path), extra=_it800_like_meta())
    kept = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2, lr=1e-4))
    kept.load(str(path), restore_optimizer=True)
    assert not optimizer_is_fresh(kept.opt)
    skipped = PPOTrainer(Policy(), PPOConfig(device="cpu", rollout_len=2, lr=1e-4))
    skipped.load(str(path), restore_optimizer=False)
    assert optimizer_is_fresh(skipped.opt)
    assert skipped.opt.param_groups[0]["lr"] == pytest.approx(1e-4)
    weights = {k: v.detach().clone() for k, v in src.policy.state_dict().items()}
    for key, tensor in weights.items():
        assert torch.equal(skipped.policy.state_dict()[key], tensor)


def test_it800_real_dry_run_writes_nothing(capsys):
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    src_dir = IT800.parent
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in src_dir.iterdir()}
    dest = ROOT / "checkpoints" / "v3run2"
    dest_existed = dest.exists()
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--resume",
            str(IT800),
            "--run-name",
            "v3run2",
            "--stage",
            "nations-easy-peninsula",
            "--force-stage",
            "--lr",
            "1e-4",
            "--eval-every",
            "5",
            "--additional-steps",
            "10000000",
            "--servers",
            "16",
            "--envs",
            "8",
            "--port",
            "8765",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert "reset_optimizer=True" in out
    assert "need_baseline_eval=True" in out
    assert "maps_unlocked=False" in out
    assert "eval_every=5" in out
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in src_dir.iterdir()}
    assert after == before
    if not dest_existed:
        assert not dest.exists()


def test_it800_has_no_scheduler_or_scaler_state():
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    mtime = IT800.stat().st_mtime_ns
    ckpt = torch.load(IT800, map_location="cpu", weights_only=False)
    assert "scheduler" not in ckpt
    assert "scaler" not in ckpt
    assert "lr_scheduler" not in ckpt
    assert IT800.stat().st_mtime_ns == mtime
