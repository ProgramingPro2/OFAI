"""Elite trajectories, behavior cloning, and guarded A/B holdout gates.

Self-contained — does not start EnvServer or write repository checkpoints.
Run: .venv/bin/python -m pytest tests/test_sample_efficiency.py -q
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
from agent.bc_ab import (
    FrozenHoldoutError,
    compare_trajectory_nll,
    extract_frozen_seeds,
    histogram_entropy,
    judge_bc_holdout,
    main as ab_main,
    write_verdict,
)
from agent.imitation import (
    BCConfig,
    ProductionCheckpointError,
    behavior_clone,
    is_bc_checkpoint,
    is_production_checkpoint_path,
    load_schema_v2_policy,
    main as bc_main,
)
from agent.model import Policy
from agent.ppo import CHECKPOINT_SCHEMA_VERSION, PPOConfig, PPOTrainer, RolloutBuffer
from agent.train import parse_args
from agent.trajectories import (
    EliteTrajectoryRecorder,
    TrajectoryValidationError,
    is_official_backend,
    load_dataset,
    save_dataset,
    split_dataset,
    synthetic_elite_dataset,
    validate_dataset,
    main as traj_main,
)
from tests.test_rl_pipeline import FakeEnv, _fake_info, _numpy_obs

SEEDS = ["hold-a", "hold-b", "hold-c", "hold-d"]
_MIXED_HIST = {"spawn": 8, "attack": 20, "noop": 6, "boat": 2}
_COLLAPSE_HIST = {"noop": 36}


def _win_info() -> dict:
    return {
        "win": True,
        "terminalCause": "win",
        "tick": 200,
        "peakTilesFrac": 0.82,
    }


def _eval_payload(
    wins: int,
    n: int,
    seeds: list[str],
    hist: dict,
    *,
    schema: int = 2,
    peak: float = 0.5,
) -> dict:
    return {
        "schema_version": schema,
        "kind": "official_core_holdout",
        "seeds": list(seeds),
        "sampled": {
            "episodes": n,
            "win_rate": wins / n,
            "seeds": list(seeds),
            "action_hist": dict(hist),
            "mean_peak_tiles": peak,
            "terminal_causes": {"win": wins, "timeout": n - wins},
        },
    }


def _write_policy_ckpt(path: Path, policy: Policy) -> Path:
    torch.save(
        {"schema_version": CHECKPOINT_SCHEMA_VERSION, "model": policy.state_dict()},
        path,
    )
    return path


def test_disabled_recorder_is_noop_and_refuses_save(tmp_path):
    rec = EliteTrajectoryRecorder(enabled=False)
    payload = synthetic_elite_dataset(
        n_episodes=1, steps_per_episode=2, seed=0, scripted=True
    )
    rec.on_reset(0, seed="x")
    rec.on_step(0, {k: payload["steps"][k][:1] for k in (
        "spatial",
        "players",
        "global",
        "action_mask",
        "target_masks",
        "quantity_mask",
        "unit_mask",
        "spawn_regions",
        "build_regions",
        "boat_regions",
    )}, payload["steps"]["actions"][0])
    assert rec.on_done(0, _win_info()) is False
    assert rec.n_episodes == 0
    with pytest.raises(TrajectoryValidationError, match="DISABLED"):
        rec.save(tmp_path / "elite.pt")


def test_recorder_keeps_only_official_core_wins():
    rec = EliteTrajectoryRecorder(enabled=True, source="official-core")
    base = synthetic_elite_dataset(
        n_episodes=1, steps_per_episode=2, seed=1, scripted=True
    )
    obs_batch = {k: base["steps"][k][:1] for k in (
        "spatial",
        "players",
        "global",
        "action_mask",
        "target_masks",
        "quantity_mask",
        "unit_mask",
        "spawn_regions",
        "build_regions",
        "boat_regions",
    )}
    action = base["steps"]["actions"][0]

    rec.on_reset(0, seed="loss")
    rec.on_step(0, obs_batch, action, obs_index=0)
    assert rec.on_done(0, {"win": False, "terminalCause": "timeout", "tick": 10}) is False

    rec.on_reset(0, seed="milestone")
    rec.on_step(0, obs_batch, action, obs_index=0)
    assert rec.on_done(
        0,
        {
            "win": False,
            "stageSuccess": True,
            "terminalCause": "curriculum_success",
            "peakTilesFrac": 0.35,
        },
    ) is False

    rec.on_reset(0, seed="rust", map_name="Halkidiki", stage="duel")
    rec.source = "rust-mirror"
    rec.on_step(0, obs_batch, action, obs_index=0)
    assert rec.on_done(0, _win_info()) is False
    rec.source = "official-core"

    rec.on_reset(0, seed="win", map_name="Halkidiki", stage="duel")
    rec.on_step(0, obs_batch, action, obs_index=0)
    rec.on_step(0, obs_batch, action, obs_index=0)
    assert rec.on_done(0, _win_info()) is True
    assert rec.n_episodes == 1
    assert rec.n_steps == 2
    out = rec.to_payload()
    assert validate_dataset(out) == out["checksum"]


def test_dataset_checksum_is_deterministic(tmp_path):
    a = synthetic_elite_dataset(
        n_episodes=2, steps_per_episode=3, seed=7, scripted=True
    )
    b = synthetic_elite_dataset(
        n_episodes=2, steps_per_episode=3, seed=7, scripted=True
    )
    assert a["checksum"] == b["checksum"]
    assert validate_dataset(a) == validate_dataset(b)
    path = save_dataset(a, tmp_path / "elite.pt")
    loaded = load_dataset(path)
    assert loaded.checksum == a["checksum"]
    assert loaded.n_episodes == 2
    traj_main(["--validate", str(path)])


def test_validate_fails_closed_on_schema_and_mask(tmp_path):
    payload = synthetic_elite_dataset(
        n_episodes=1, steps_per_episode=2, seed=3, scripted=True
    )
    bad_ver = dict(payload)
    bad_ver["schema_version"] = 1
    with pytest.raises(TrajectoryValidationError, match="SCHEMA_VERSION"):
        validate_dataset(bad_ver)

    bad_kind = dict(payload)
    bad_kind["kind"] = "rollout"
    with pytest.raises(TrajectoryValidationError, match="DATASET_KIND"):
        validate_dataset(bad_kind)

    tampered = dict(payload)
    tampered["checksum"] = "0" * 32
    with pytest.raises(TrajectoryValidationError, match="CHECKSUM"):
        validate_dataset(tampered)

    illegal = {
        "schema_version": 2,
        "kind": payload["kind"],
        "source": payload["source"],
        "steps": {k: np.array(v, copy=True) for k, v in payload["steps"].items()},
        "episodes": list(payload["episodes"]),
        "checksum": "",
    }
    illegal["steps"]["action_mask"][0] = 0
    illegal["steps"]["action_mask"][0, spaces.ACTION_NOOP] = 1
    illegal["steps"]["actions"][0, 0] = spaces.ACTION_ATTACK
    with pytest.raises(TrajectoryValidationError, match="MASK"):
        validate_dataset(illegal)

    empty_region = {
        "schema_version": 2,
        "kind": payload["kind"],
        "source": payload["source"],
        "steps": {k: np.array(v, copy=True) for k, v in payload["steps"].items()},
        "episodes": list(payload["episodes"]),
        "checksum": "",
    }
    empty_region["steps"]["actions"][0] = np.array(
        [spaces.ACTION_SPAWN, 0, 0, 0, 0], dtype=np.int16
    )
    empty_region["steps"]["action_mask"][0, spaces.ACTION_SPAWN] = 1
    empty_region["steps"]["spawn_regions"][0] = 0
    with pytest.raises(TrajectoryValidationError, match="MASK"):
        validate_dataset(empty_region)

    empty_target = {
        "schema_version": 2,
        "kind": payload["kind"],
        "source": payload["source"],
        "steps": {k: np.array(v, copy=True) for k, v in payload["steps"].items()},
        "episodes": list(payload["episodes"]),
        "checksum": "",
    }
    empty_target["steps"]["actions"][1] = np.array(
        [spaces.ACTION_ATTACK, 1, 0, 3, 0], dtype=np.int16
    )
    empty_target["steps"]["action_mask"][1, spaces.ACTION_ATTACK] = 1
    empty_target["steps"]["target_masks"][1, spaces.ACTION_ATTACK] = 0
    with pytest.raises(TrajectoryValidationError, match="MASK"):
        validate_dataset(empty_target)


def test_split_by_frozen_seeds_is_deterministic():
    payload = synthetic_elite_dataset(
        n_episodes=4, steps_per_episode=2, seed=4, scripted=True, stage="duel"
    )
    hold_seeds = [payload["episodes"][2]["seed"], payload["episodes"][3]["seed"]]
    train_a, hold_a = split_dataset(payload, hold_seeds)
    train_b, hold_b = split_dataset(payload, hold_seeds)
    assert train_a["checksum"] == train_b["checksum"]
    assert hold_a["checksum"] == hold_b["checksum"]
    assert hold_a["episodes"][0]["seed"] == hold_seeds[0]
    with pytest.raises(TrajectoryValidationError, match="HOLDOUT"):
        split_dataset(payload, ["missing-seed"])


def test_bc_refuses_production_checkpoint_names(tmp_path):
    assert is_production_checkpoint_path(tmp_path / "best.pt")
    assert is_production_checkpoint_path(tmp_path / "latest.pt")
    assert is_production_checkpoint_path(tmp_path / "stage-entry.pt")
    assert is_production_checkpoint_path(tmp_path / "best-duel.pt")
    payload = synthetic_elite_dataset(
        n_episodes=1, steps_per_episode=2, seed=5, scripted=True
    )
    ds_path = save_dataset(payload, tmp_path / "elite.pt")
    with pytest.raises(ProductionCheckpointError, match="production"):
        behavior_clone(
            load_dataset(ds_path),
            out_path=tmp_path / "best.pt",
            cfg=BCConfig(epochs=1, batch_size=2, device="cpu"),
        )


def test_bc_writes_schema_v2_candidate_and_lowers_nll(tmp_path):
    payload = synthetic_elite_dataset(
        n_episodes=3, steps_per_episode=4, seed=6, scripted=True
    )
    ds_path = save_dataset(payload, tmp_path / "elite.pt")
    out = tmp_path / "bc-warmstart.pt"
    result = behavior_clone(
        load_dataset(ds_path),
        out_path=out,
        cfg=BCConfig(epochs=5, batch_size=4, seed=0, device="cpu", lr=1e-3),
    )
    assert result.final_nll < result.train_nll
    ckpt = torch.load(out, map_location="cpu", weights_only=False)
    assert ckpt["schema_version"] == 2
    assert ckpt["save_reason"] == "behavior_clone"
    assert ckpt["pending_stage_baseline"] is True
    assert ckpt["bc"]["accepted"] is False
    assert ckpt["bc"]["promoted"] is False
    assert ckpt["best_score"][0] < 0.0
    policy = Policy()
    policy.load_state_dict(ckpt["model"])
    trainer = PPOTrainer(Policy(), PPOConfig(device="cpu"))
    loaded = trainer.load(str(out), restore_optimizer=False)
    assert loaded["schema_version"] == 2
    assert trainer.stats.episodes == 0


def test_ab_requires_identical_frozen_seeds():
    base = _eval_payload(3, 4, SEEDS, _MIXED_HIST)
    other = _eval_payload(3, 4, list(reversed(SEEDS)), _MIXED_HIST)
    with pytest.raises(FrozenHoldoutError, match="SEEDS"):
        judge_bc_holdout(base, other)
    missing = dict(base)
    missing["seeds"] = []
    missing["sampled"] = dict(base["sampled"])
    missing["sampled"]["seeds"] = []
    with pytest.raises(FrozenHoldoutError, match="SEEDS"):
        extract_frozen_seeds(missing)
    disagree = dict(base)
    disagree["seeds"] = SEEDS
    disagree["sampled"] = dict(base["sampled"])
    disagree["sampled"]["seeds"] = list(reversed(SEEDS))
    with pytest.raises(FrozenHoldoutError, match="SEEDS"):
        extract_frozen_seeds(disagree)


def test_ab_fails_closed_on_eval_schema_mismatch():
    base = _eval_payload(3, 4, SEEDS, _MIXED_HIST, schema=2)
    cand = _eval_payload(3, 4, SEEDS, _MIXED_HIST, schema=1)
    with pytest.raises(FrozenHoldoutError, match="SCHEMA_VERSION"):
        judge_bc_holdout(base, cand)
    no_schema = dict(base)
    no_schema.pop("schema_version")
    with pytest.raises(FrozenHoldoutError, match="SCHEMA_VERSION"):
        judge_bc_holdout(no_schema, base)


def test_ab_rejects_win_or_diversity_regression_and_collapse():
    base = _eval_payload(3, 4, SEEDS, _MIXED_HIST)
    worse = _eval_payload(2, 4, SEEDS, _MIXED_HIST)
    verdict = judge_bc_holdout(base, worse)
    assert verdict.accepted is False
    assert verdict.reason == "win_rate_reduced"
    assert verdict.promoted is False

    same_wins = _eval_payload(3, 4, SEEDS, {"noop": 30, "attack": 1})
    div = judge_bc_holdout(base, same_wins)
    assert div.accepted is False
    assert div.reason == "diversity_reduced"
    assert histogram_entropy(same_wins["sampled"]["action_hist"]) < histogram_entropy(
        _MIXED_HIST
    )

    collapsed = _eval_payload(0, 4, SEEDS, _COLLAPSE_HIST, peak=0.0)
    col = judge_bc_holdout(base, collapsed)
    assert col.accepted is False
    assert col.reason in {"candidate_collapsed", "zero_win_regression"}


def test_ab_accepts_non_regression_but_does_not_promote(tmp_path):
    base = _eval_payload(3, 4, SEEDS, _MIXED_HIST)
    better = _eval_payload(4, 4, SEEDS, _MIXED_HIST)
    verdict = judge_bc_holdout(base, better)
    assert verdict.accepted is True
    assert verdict.promoted is False
    assert verdict.candidate_wins == 4
    dest = write_verdict(verdict, tmp_path / "bc-ab.json")
    data = json.loads(dest.read_text())
    assert data["accepted"] is True
    assert data["promoted"] is False
    with pytest.raises(ProductionCheckpointError):
        write_verdict(verdict, tmp_path / "best.pt")


def test_imitation_and_ab_cli(tmp_path):
    payload = synthetic_elite_dataset(
        n_episodes=2, steps_per_episode=3, seed=8, scripted=True
    )
    ds = save_dataset(payload, tmp_path / "elite.pt")
    out = tmp_path / "bc-candidate.pt"
    bc_main(
        [
            "--dataset",
            str(ds),
            "--out",
            str(out),
            "--epochs",
            "2",
            "--batch-size",
            "4",
            "--device",
            "cpu",
            "--seed",
            "0",
        ]
    )
    assert out.is_file()
    base = _eval_payload(2, 4, SEEDS, _MIXED_HIST)
    cand = _eval_payload(2, 4, SEEDS, _MIXED_HIST)
    b_path = tmp_path / "base.json"
    c_path = tmp_path / "cand.json"
    b_path.write_text(json.dumps(base))
    c_path.write_text(json.dumps(cand))
    verdict_path = tmp_path / "verdict.json"
    rc = ab_main(
        [
            "--baseline-eval",
            str(b_path),
            "--candidate-eval",
            str(c_path),
            "--out",
            str(verdict_path),
        ]
    )
    assert rc == 0
    assert json.loads(verdict_path.read_text())["accepted"] is True
    with pytest.raises(ProductionCheckpointError):
        ab_main(
            [
                "--baseline-eval",
                str(b_path),
                "--candidate-eval",
                str(c_path),
                "--out",
                str(tmp_path / "latest.pt"),
            ]
        )


def test_practical_ab_smoke_same_frozen_holdout(tmp_path):
    """Tiny CPU A/B: identical holdout seeds, BC NLL, official-core judge."""
    payload = synthetic_elite_dataset(
        n_episodes=6,
        steps_per_episode=6,
        seed=11,
        scripted=True,
        stage="duel",
    )
    hold_seeds = [payload["episodes"][i]["seed"] for i in (4, 5)]
    train_payload, hold_payload = split_dataset(payload, hold_seeds)
    train_path = save_dataset(train_payload, tmp_path / "train.pt")
    hold_path = save_dataset(hold_payload, tmp_path / "hold.pt")

    torch.manual_seed(21)
    init = Policy()
    init_path = _write_policy_ckpt(tmp_path / "init.pt", init)
    cand_path = tmp_path / "bc-warmstart.pt"
    result = behavior_clone(
        load_dataset(train_path),
        out_path=cand_path,
        init_ckpt=init_path,
        cfg=BCConfig(epochs=8, batch_size=8, seed=21, device="cpu", lr=2e-3),
    )
    assert result.final_nll < result.train_nll

    hold_ds = load_dataset(hold_path)
    nll_init, nll_bc = compare_trajectory_nll(
        init_path, cand_path, hold_ds, device="cpu"
    )
    assert nll_bc <= nll_init

    loaded, ckpt = load_schema_v2_policy(cand_path, torch.device("cpu"))
    assert int(ckpt["schema_version"]) == 2
    loaded.eval()

    official = judge_bc_holdout(
        _eval_payload(3, 4, hold_seeds + SEEDS[:2], _MIXED_HIST),
        _eval_payload(3, 4, hold_seeds + SEEDS[:2], _MIXED_HIST),
    )
    assert official.seeds[:2] == hold_seeds
    assert official.accepted is True
    assert official.promoted is False
    reject = judge_bc_holdout(
        _eval_payload(3, 4, hold_seeds + SEEDS[:2], _MIXED_HIST),
        _eval_payload(1, 4, hold_seeds + SEEDS[:2], _MIXED_HIST),
    )
    assert reject.accepted is False
    assert (tmp_path / "best.pt").exists() is False
    assert list(tmp_path.glob("best-*.pt")) == []


def test_official_backend_fail_closed():
    assert is_official_backend("sequential") is True
    assert is_official_backend("parallel") is True
    assert is_official_backend("official-core") is True
    assert is_official_backend("rust") is False
    assert is_official_backend("mixed") is False
    assert is_official_backend(None) is False
    assert is_official_backend(None, allow_unlabeled=True) is True


def test_recorder_bound_and_atomic_save(tmp_path):
    rec = EliteTrajectoryRecorder(enabled=True, max_episodes=1, path=tmp_path / "elite-official.pt")
    payload = synthetic_elite_dataset(
        n_episodes=1, steps_per_episode=2, seed=9, scripted=True
    )
    batch = {k: payload["steps"][k][:1] for k in (
        "spatial", "players", "global", "action_mask", "target_masks",
        "quantity_mask", "unit_mask", "spawn_regions", "build_regions", "boat_regions",
    )}
    rec.on_reset(0, seed="a")
    rec.on_step(0, batch, payload["steps"]["actions"][0], obs_index=0)
    assert rec.on_done(0, _win_info()) is True
    rec.on_reset(0, seed="b")
    rec.on_step(0, batch, payload["steps"]["actions"][0], obs_index=0)
    assert rec.on_done(0, _win_info()) is False
    assert rec.n_episodes == 1
    path = rec.flush()
    assert path is not None and path.is_file()
    with pytest.raises(TrajectoryValidationError, match="PRODUCTION"):
        rec.save(tmp_path / "best.pt")


def test_collect_keeps_complete_episode_across_rollouts(tmp_path):
    cfg = PPOConfig(rollout_len=3, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.elite = EliteTrajectoryRecorder(
        enabled=True, n_envs=1, path=tmp_path / "elite-official.pt"
    )
    scripts = [[
        (0.0, False, _fake_info()),
        (0.0, False, _fake_info()),
        (0.0, False, _fake_info()),
        (0.0, False, _fake_info()),
        (0.0, True, _fake_info(terminalCause="win", win=True, seed="span", tick=50)),
    ]]
    env = FakeEnv(1, scripts)
    env.backend = "sequential"
    obs = _numpy_obs(1)
    buf = RolloutBuffer(cfg.rollout_len, 1)
    obs, eps1 = trainer.collect(env, obs, buf)
    assert eps1 == []
    assert trainer.elite.n_episodes == 0
    assert trainer.elite._open[0] is not None
    assert len(trainer.elite._open[0].actions) == 3
    obs, eps2 = trainer.collect(env, obs, buf)
    assert len(eps2) == 1
    assert eps2[0].win is True
    assert trainer.elite.n_episodes == 1
    assert trainer.elite.n_steps == 5
    flushed = trainer.elite.flush()
    assert flushed is not None
    loaded = load_dataset(flushed)
    assert loaded.n_steps == 5
    rust = FakeEnv(1, scripts)
    rust.backend = "rust"
    trainer2 = PPOTrainer(Policy(), cfg)
    trainer2.elite = EliteTrajectoryRecorder(enabled=True, n_envs=1)
    trainer2.collect(rust, _numpy_obs(1), RolloutBuffer(cfg.rollout_len, 1))
    assert trainer2.elite.n_episodes == 0


def test_interrupt_discards_open_elite_episode():
    cfg = PPOConfig(rollout_len=2, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.elite = EliteTrajectoryRecorder(enabled=True, n_envs=1)
    env = FakeEnv(1, [[(0.0, False, _fake_info())] * 4])
    env.backend = "parallel"
    trainer.collect(env, _numpy_obs(1), RolloutBuffer(2, 1))
    assert trainer.elite._open[0] is not None
    trainer.reset_episode_trackers()
    assert trainer.elite._open[0] is None
    assert trainer.elite.n_episodes == 0


def test_record_elite_and_bc_cli(tmp_path):
    args = parse_args(
        [
            "--run-name",
            "se-cli",
            "--record-elite",
            "--elite-max-episodes",
            "8",
            "--bc-dataset",
            str(tmp_path / "elite.pt"),
            "--bc-epochs",
            "2",
            "--device",
            "cpu",
        ]
    )
    assert args.record_elite is True
    assert args.elite_max_episodes == 8
    assert args.bc_epochs == 2
    fresh = parse_args(["--run-name", "fresh"])
    assert fresh.record_elite is False
    assert fresh.bc_dataset == ""


def test_bc_resume_dry_run_requires_official_holdout(tmp_path, capsys):
    payload = synthetic_elite_dataset(
        n_episodes=2, steps_per_episode=3, seed=12, scripted=True
    )
    ds = save_dataset(payload, tmp_path / "elite.pt")
    cand = tmp_path / "bc-warmstart.pt"
    behavior_clone(
        load_dataset(ds),
        out_path=cand,
        cfg=BCConfig(epochs=1, batch_size=4, device="cpu"),
    )
    ckpt = torch.load(cand, map_location="cpu", weights_only=False)
    assert is_bc_checkpoint(ckpt)
    assert ckpt["pending_stage_baseline"] is True
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--resume",
            str(cand),
            "--run-name",
            "se-bc-dry",
            "--device",
            "cpu",
            "--record-elite",
        ]
    )
    out = capsys.readouterr().out
    assert "bc_warm_start=true" in out
    assert "pending_stage_baseline=true" in out
    assert "record_elite=true" in out
    assert "official holdout required" in out


def test_live_official_record_bc_smoke(tmp_path):
    """Tiny official-core recording path, then BC + dry-run resume."""
    ofai = ROOT / "OpenFrontIO" / "ofai" / "env" / "EnvServer.ts"
    if not ofai.is_file():
        pytest.skip("OpenFrontIO env server is not present")

    from agent.env_client import EnvServerProcess, VecEnv
    from agent.eval import scripted_expander_action

    port = 9233
    server = EnvServerProcess(port, workers=0)
    env = None
    rec = EliteTrajectoryRecorder(
        enabled=True,
        n_envs=1,
        max_episodes=1,
        max_steps=80,
        allow_unlabeled_backend=True,
        path=tmp_path / "elite-official.pt",
    )
    live_backend = None
    kept = False
    try:
        env = VecEnv("127.0.0.1", port, timeout_s=180.0)
        obs = env.init(
            [
                {
                    "map": "Halkidiki",
                    "mapSize": "Compact",
                    "nations": "disabled",
                    "difficulty": "Easy",
                    "bots": 1,
                    "seed": "se-live-0",
                    "maxTicks": 800,
                    "decisionInterval": 10,
                    "shaping": 0.0,
                    "allowedActions": [0, 1, 2, 3, 5],
                }
            ]
        )
        live_backend = env.backend
        assert rec.accepts_backend(env.backend) or env.backend in (
            None,
            "sequential",
            "parallel",
        )
        rec.on_reset(0, seed="se-live-0", map_name="Halkidiki", stage="duel")
        for _ in range(24):
            if not rec.accepts_backend(env.backend) and env.backend not in (
                None,
                "sequential",
                "parallel",
            ):
                break
            action = scripted_expander_action(obs, 0)
            rec.on_step(0, obs, action, obs_index=0)
            batch = env.step(action.reshape(1, 5))
            if bool(batch.dones[0]):
                kept = rec.on_done(0, batch.infos[0])
                break
            obs = batch.obs
        if rec.accepts_backend(live_backend) or live_backend in (None, "sequential", "parallel"):
            assert rec.source == "official-core"
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            server.stop()
        except Exception:
            pass

    if kept:
        path = rec.flush()
        assert path is not None
        ds = load_dataset(path)
        cand = tmp_path / "bc-warmstart.pt"
        result = behavior_clone(
            ds,
            out_path=cand,
            cfg=BCConfig(epochs=2, batch_size=8, device="cpu"),
        )
        assert result.final_nll <= result.train_nll + 1e-5
    else:
        payload = synthetic_elite_dataset(
            n_episodes=2, steps_per_episode=4, seed=13, scripted=True
        )
        ds_path = save_dataset(payload, tmp_path / "elite-fallback.pt")
        cand = tmp_path / "bc-warmstart.pt"
        behavior_clone(
            load_dataset(ds_path),
            out_path=cand,
            cfg=BCConfig(epochs=2, batch_size=4, device="cpu"),
        )

    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--resume",
            str(cand),
            "--run-name",
            "se-live-dry",
            "--device",
            "cpu",
            "--eval-every",
            "1",
            "--eval-episodes",
            "2",
        ]
    )
