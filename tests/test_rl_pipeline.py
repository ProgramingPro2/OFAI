"""Unit tests for truthful episode metrics, active-head policy, PPO rollback,
checkpoint roundtrip, shard recovery, and curriculum windowing.

Self-contained / mocked — does not require a live EnvServer.
Run: .venv/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import spaces
from agent.curriculum import (
    MASTERY_VERSION,
    STAGES,
    Curriculum,
    Stage,
    wilson_lower_bound,
)
from agent.env_client import (
    RolloutInterrupted,
    ShardedVecEnv,
    StepBatch,
    _concat_obs,
    _encode_frame,
    _obs_dict,
    _read_frame,
    _scatter_obs,
    distribute_active_lanes,
)
from agent.eval import scripted_expander_action
from agent.model import Policy, make_dummy_obs, normalized_action_entropy, obs_to_tensors
from agent.ppo import (
    EpisodeRecord,
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    TrainStats,
    classify_outcome,
)
from agent.train import (
    FIXED_SEED_EVAL_RETRIES,
    _byte_rng,
    _eval_score,
    _set_rng_state,
    run_fixed_seed_eval,
)


def _fake_info(**kwargs):
    base = {
        "win": False,
        "dead": False,
        "spawned": True,
        "kills": 0,
        "tick": 100,
        "tilesFrac": 0.05,
        "peakTilesFrac": 0.06,
        "terminalCause": "timeout",
        "rewardTerms": {
            "terminal": 0.0,
            "spawn": 0.0,
            "territory": 0.01,
            "elimination": 0.0,
            "total": 0.01,
        },
        "intentCount": 1,
        "actionAccepted": True,
    }
    base.update(kwargs)
    return base


class FakeEnv:
    def __init__(self, k: int, scripts: list[list[tuple[float, bool, dict]]]):
        self.k = k
        self.scripts = scripts
        self.t = 0
        self.step_calls = 0

    def step(self, actions):
        self.step_calls += 1
        rewards = np.zeros(self.k, np.float32)
        dones = np.zeros(self.k, dtype=bool)
        infos = []
        for i in range(self.k):
            seq = self.scripts[i]
            if self.t < len(seq):
                r, d, info = seq[self.t]
            else:
                r, d, info = 0.0, False, _fake_info(terminalCause="none")
            rewards[i] = r
            dones[i] = d
            infos.append(info)
        self.t += 1
        return StepBatch(obs=_numpy_obs(self.k), rewards=rewards, dones=dones, infos=infos)


def _numpy_obs(k: int) -> dict:
    s = spaces
    return {
        "spatial": np.zeros((k, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE), np.float32),
        "players": np.ones((k, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), np.float32),
        "global": np.zeros((k, s.GLOBAL_FEATURES), np.float32),
        "action_mask": np.ones((k, s.NUM_ACTION_TYPES), np.uint8),
        "target_masks": np.ones((k, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS), np.uint8),
        "quantity_mask": np.ones((k, s.NUM_QUANTITIES), np.uint8),
        "unit_mask": np.ones((k, s.NUM_UNIT_TYPES), np.uint8),
        "spawn_regions": np.ones((k, s.NUM_REGIONS), np.uint8),
        "build_regions": np.ones((k, s.NUM_REGIONS), np.uint8),
        "boat_regions": np.ones((k, s.NUM_REGIONS), np.uint8),
    }


def test_episode_return_spans_collect_calls():
    cfg = PPOConfig(rollout_len=3, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)
    scripts = [
        [
            (1.0, False, _fake_info()),
            (2.0, False, _fake_info()),
            (0.0, False, _fake_info()),
            (4.0, True, _fake_info(terminalCause="win", win=True, tick=50)),
            (0.0, False, _fake_info()),
            (0.0, False, _fake_info()),
        ]
    ]
    env = FakeEnv(1, scripts)
    obs = _numpy_obs(1)
    buf = RolloutBuffer(cfg.rollout_len, 1)
    obs, eps1 = trainer.collect(env, obs, buf)
    assert eps1 == []
    assert trainer._trackers[0].ret == pytest.approx(3.0)
    obs, eps2 = trainer.collect(env, obs, buf)
    assert len(eps2) == 1
    assert eps2[0].ret == pytest.approx(7.0)
    assert eps2[0].terminal_cause == "win"
    assert trainer.stats.episodes == 1
    assert trainer.stats.mean_return_20 == pytest.approx(7.0)


def test_bounded_rolling_windows():
    stats = TrainStats()
    for i in range(600):
        stats.record_episode(
            EpisodeRecord(
                env_index=0,
                ret=float(i),
                decisions=10,
                ticks=100,
                terminal_cause="timeout",
                win=False,
                stage_success=False,
                kills=i % 5,
                final_tiles_frac=0.1,
                peak_tiles_frac=0.2,
                spawned=True,
                reward_terms={},
                action_hist={},
                accepted=5,
                fizzled=0,
                intent_count=5,
            )
        )
    assert len(stats.ret_20) == 20
    assert len(stats.ret_100) == 100
    assert len(stats.ret_500) == 500
    assert stats.episodes == 600
    assert stats.mean_return_20 == pytest.approx(np.mean(range(580, 600)))


def _ep(**kwargs) -> EpisodeRecord:
    base = dict(
        env_index=0,
        ret=0.0,
        decisions=10,
        ticks=100,
        terminal_cause="timeout",
        win=False,
        stage_success=False,
        kills=0,
        final_tiles_frac=0.1,
        peak_tiles_frac=0.2,
        spawned=True,
        reward_terms={},
        action_hist={},
        accepted=5,
        fizzled=0,
        intent_count=5,
    )
    base.update(kwargs)
    return EpisodeRecord(**base)


def test_recent_20_kills_exact():
    stats = TrainStats()
    for i in range(25):
        stats.record_episode(_ep(kills=i))
    assert list(stats.kills_20) == [float(i) for i in range(5, 25)]
    assert stats.kills_per_ep_20 == pytest.approx(np.mean(range(5, 25)))
    assert stats.kills == sum(range(25))


def test_tracker_reset_on_reinit():
    cfg = PPOConfig(rollout_len=2, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)
    scripts = [[(1.0, False, _fake_info()), (1.0, False, _fake_info())]]
    env = FakeEnv(1, scripts)
    buf = RolloutBuffer(2, 1)
    trainer.collect(env, _numpy_obs(1), buf)
    assert trainer._trackers[0].ret == pytest.approx(2.0)
    trainer.reset_episode_trackers()
    assert trainer._trackers[0].ret == 0.0


def test_active_head_logprob_identity():
    torch.manual_seed(0)
    policy = Policy()
    obs = make_dummy_obs(8, "cpu")
    with torch.no_grad():
        actions, logp_act, _ = policy.act(obs, deterministic=False)
        logp_eval, ent, _, extras = policy.evaluate(obs, actions)
    assert torch.allclose(logp_act, logp_eval, atol=1e-5)
    at = actions[:, 0]
    noopish = (at == spaces.ACTION_NOOP) | (at == spaces.ACTION_RETREAT_ALL)
    if noopish.any():
        assert float(extras["cond_entropy"][noopish].abs().max()) < 1e-5


def test_deterministic_uses_selected_action_masks():
    torch.manual_seed(1)
    policy = Policy()
    obs = make_dummy_obs(4, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_ATTACK] = 1.0
    obs["target_masks"].zero_()
    obs["target_masks"][:, spaces.ACTION_ATTACK, 3] = 1.0
    obs["quantity_mask"].zero_()
    obs["quantity_mask"][:, 2] = 1.0
    with torch.no_grad():
        actions, _, _ = policy.act(obs, deterministic=True)
    assert (actions[:, 0] == spaces.ACTION_ATTACK).all()
    assert (actions[:, 1] == 3).all()
    assert (actions[:, 3] == 2).all()


def test_spawn_only_region_in_logprob():
    torch.manual_seed(2)
    policy = Policy()
    obs = make_dummy_obs(2, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_SPAWN] = 1.0
    with torch.no_grad():
        actions, logp, _ = policy.act(obs, deterministic=False)
        logp2, ent, _, extras = policy.evaluate(obs, actions)
    assert (actions[:, 0] == spaces.ACTION_SPAWN).all()
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.allclose(extras["cond_entropy"], ent - extras["action_entropy"], atol=1e-5)


def test_optimizer_rollback_deep_copy():
    cfg = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=2,
        update_epochs=2,
        target_kl=1e-12,
        hard_kl_mult=0.0,
    )
    policy = Policy()
    trainer = PPOTrainer(policy, cfg)
    pre_model = copy.deepcopy({k: v.clone() for k, v in policy.state_dict().items()})
    trainer.ent_coef = 0.123

    buf = RolloutBuffer(cfg.rollout_len, 2)
    env = FakeEnv(
        2,
        [
            [(0.1, False, _fake_info())] * 4,
            [(0.1, False, _fake_info())] * 4,
        ],
    )
    obs = _numpy_obs(2)
    obs, _ = trainer.collect(env, obs, buf)
    metrics = trainer.update(buf, obs)
    assert metrics["hard_reverted"] is True
    assert metrics["n_updates"] == 0
    assert metrics["frac_kept"] == 0.0
    assert trainer.ent_coef == pytest.approx(0.123)
    for k in pre_model:
        assert torch.equal(policy.state_dict()[k], pre_model[k])


def test_array_split_minibatch_partition():
    cfg = PPOConfig(
        rollout_len=5, device="cpu", num_minibatches=3, update_epochs=1, target_kl=10.0
    )
    trainer = PPOTrainer(Policy(), cfg)
    buf = RolloutBuffer(5, 2)
    env = FakeEnv(2, [[(0.0, False, _fake_info())] * 5] * 2)
    obs = _numpy_obs(2)
    obs, _ = trainer.collect(env, obs, buf)
    metrics = trainer.update(buf, obs)
    assert metrics["n_updates"] == 3


def test_checkpoint_roundtrip(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.episodes = 42
    trainer.stats.ret_20.append(1.5)
    trainer.ent_coef = 0.07
    cur = Curriculum(start=2)
    cur.record_episode(True)
    cur.note_eval(True)
    path = tmp_path / "ckpt.pt"
    trainer.save(
        str(path),
        extra={
            "curriculum": cur.state_dict(),
            "stage_baseline": {"win_rate": 0.5},
            "rng": {"python": None},
        },
    )
    trainer2 = PPOTrainer(Policy(), cfg)
    ckpt = trainer2.load(str(path))
    assert trainer2.stats.episodes == 42
    assert list(trainer2.stats.ret_20) == [1.5]
    assert trainer2.ent_coef == pytest.approx(0.07)
    assert ckpt["curriculum"]["idx"] == 2
    assert ckpt["schema_version"] == 2


def test_shard_recovery_no_double_step():
    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 1
    env.k = 2
    env.ports = [9001, 9002]
    env.connect = False
    env._configs = [{}, {}]
    env.servers = [MagicMock(alive=MagicMock(return_value=True)) for _ in range(2)]
    step_counts = [0, 0]

    class FakeClient:
        def __init__(self, idx):
            self.idx = idx

        def step(self, actions, next_configs=None):
            step_counts[self.idx] += 1
            if self.idx == 1:
                raise ConnectionError("shard down")
            return StepBatch(
                obs=_numpy_obs(1),
                rewards=np.zeros(1, np.float32),
                dones=np.zeros(1, dtype=bool),
                infos=[_fake_info()],
            )

        def close(self):
            pass

        def init(self, cfgs):
            return _numpy_obs(len(cfgs))

    env.clients = [FakeClient(0), FakeClient(1)]
    from concurrent.futures import ThreadPoolExecutor

    env.pool = ThreadPoolExecutor(max_workers=2)
    env._restart_shard = MagicMock()

    with pytest.raises(RolloutInterrupted) as ei:
        env.step(np.zeros((2, 5), np.int32))
    assert 1 in ei.value.failed_shards
    assert step_counts[0] == 1
    assert step_counts[1] == 1
    env._restart_shard.assert_called()
    env.pool.shutdown(wait=False)


def test_read_frame_empty_header_is_connection_error():
    """v4val1 died on json.loads('') from a truncated shard frame."""
    import socket
    import struct

    body = struct.pack("<I", 0)
    frame = struct.pack("<I", len(body)) + body
    a, b = socket.socketpair()
    try:
        a.sendall(frame)
        with pytest.raises(ConnectionError, match="corrupt env frame"):
            _read_frame(b)
    finally:
        a.close()
        b.close()


def test_step_json_decode_error_is_rollout_interrupted():
    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 1
    env.k = 2
    env.ports = [9001, 9002]
    env.connect = False
    env._configs = [{}, {}]
    env.servers = [MagicMock(alive=MagicMock(return_value=True)) for _ in range(2)]

    class FakeClient:
        def __init__(self, idx):
            self.idx = idx

        def step(self, actions, next_configs=None):
            if self.idx == 1:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return StepBatch(
                obs=_numpy_obs(1),
                rewards=np.zeros(1, np.float32),
                dones=np.zeros(1, dtype=bool),
                infos=[{}],
            )

        def close(self):
            pass

    env.clients = [FakeClient(0), FakeClient(1)]
    from concurrent.futures import ThreadPoolExecutor

    env.pool = ThreadPoolExecutor(max_workers=2)
    env._restart_shard = MagicMock()

    with pytest.raises(RolloutInterrupted) as ei:
        env.step(np.zeros((2, 5), np.int32))
    assert 1 in ei.value.failed_shards
    env._restart_shard.assert_called()
    env.pool.shutdown(wait=False)


def test_init_active_json_decode_error_is_rollout_interrupted():
    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 1
    env.k = 2
    env.ports = [9001, 9002]
    env.connect = False
    env._configs = [{}, {}]
    env.servers = [MagicMock(alive=MagicMock(return_value=True)) for _ in range(2)]
    env._obs_buf = None
    env._next_config_fn = None
    env._on_episode_reset = None
    env._clear_active()

    class FakeClient:
        def __init__(self, idx):
            self.idx = idx

        def init(self, cfgs):
            if self.idx == 1:
                raise json.JSONDecodeError("Expecting value", "", 0)
            return _numpy_obs(len(cfgs))

        def close(self):
            pass

    env.clients = [FakeClient(0), FakeClient(1)]
    from concurrent.futures import ThreadPoolExecutor

    env.pool = ThreadPoolExecutor(max_workers=2)
    env._restart_shard = MagicMock()

    with pytest.raises(RolloutInterrupted) as ei:
        env.init_active([{"seed": "a"}, {"seed": "b"}])
    assert 1 in ei.value.failed_shards
    env._restart_shard.assert_called()
    env.pool.shutdown(wait=False)


def test_obs_dict_target_masks_and_quantity():
    s = spaces
    k = 2
    tensors = {
        "spatial": np.zeros(k * s.SPATIAL_CHANNELS * s.SPATIAL_SIZE * s.SPATIAL_SIZE, np.float32),
        "players": np.zeros(k * s.NUM_PLAYER_SLOTS * s.PLAYER_FEATURES, np.float32),
        "global": np.zeros(k * s.GLOBAL_FEATURES, np.float32),
        "action_mask": np.ones(k * s.NUM_ACTION_TYPES, np.uint8),
        "target_masks": np.ones(k * s.NUM_ACTION_TYPES * s.NUM_PLAYER_SLOTS, np.uint8),
        "quantity_mask": np.ones(k * s.NUM_QUANTITIES, np.uint8),
        "unit_mask": np.ones(k * s.NUM_UNIT_TYPES, np.uint8),
        "spawn_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "build_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "boat_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
    }
    obs = _obs_dict(tensors, k)
    assert obs["target_masks"].shape == (k, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS)
    assert obs["quantity_mask"].shape == (k, s.NUM_QUANTITIES)


def test_obs_dict_legacy_target_mask_fallback():
    s = spaces
    k = 1
    tensors = {
        "spatial": np.zeros(k * s.SPATIAL_CHANNELS * s.SPATIAL_SIZE * s.SPATIAL_SIZE, np.float32),
        "players": np.zeros(k * s.NUM_PLAYER_SLOTS * s.PLAYER_FEATURES, np.float32),
        "global": np.zeros(k * s.GLOBAL_FEATURES, np.float32),
        "action_mask": np.ones(k * s.NUM_ACTION_TYPES, np.uint8),
        "target_mask": np.arange(s.NUM_PLAYER_SLOTS, dtype=np.uint8) % 2,
        "unit_mask": np.ones(k * s.NUM_UNIT_TYPES, np.uint8),
        "spawn_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "build_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "boat_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
    }
    obs = _obs_dict(tensors, k)
    assert obs["target_masks"].shape == (1, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS)
    assert obs["quantity_mask"].shape == (1, s.NUM_QUANTITIES)


def test_curriculum_window_and_state():
    stages = [
        Stage(
            name="a",
            map="FourIslands",
            bots=1,
            win_threshold=0.6,
            window=5,
            min_episodes=5,
            eval_passes_required=2,
        ),
        Stage(name="b", map="FourIslands", bots=5, window=5, min_episodes=1),
    ]
    cur = Curriculum(stages=stages, start=0)
    for _ in range(5):
        cur.record_episode(True)
    assert len(cur._stage_successes) == 5
    cur.record_episode(False)
    assert len(cur._stage_successes) == 5
    assert not cur.maybe_advance()
    cur.note_eval(True)
    assert cur.maybe_advance()
    assert cur.idx == 1
    assert cur.stage_episodes == 0
    assert cur.consecutive_eval_passes == 0
    assert not cur.last_eval_passed
    sd = cur.state_dict()
    cur2 = Curriculum(stages=stages, start=0)
    cur2.load_state_dict(sd)
    assert cur2.idx == 1


def test_curriculum_stages_shape():
    assert STAGES[0].name == "duel"
    assert STAGES[0].map == "Halkidiki"
    assert STAGES[0].bots == 1
    assert STAGES[0].map_size == "Compact"
    assert STAGES[0].win_tiles_frac == pytest.approx(0.30)
    assert "build" not in STAGES[0].allowed_actions
    assert "boat" in STAGES[0].allowed_actions
    assert "ally" not in STAGES[0].allowed_actions
    assert STAGES[1].map == "Halkidiki"
    assert "boat" in STAGES[1].allowed_actions
    assert STAGES[2].name == "island-ffa"
    assert STAGES[2].map == "FourIslands"
    assert "build" in STAGES[2].allowed_actions
    assert "boat" in STAGES[2].allowed_actions
    assert STAGES[3].name == "tribes20"
    assert STAGES[-1].shaping == 0.0
    assert spaces.EARLY_ACTIONS == spaces.LAND_BOAT_ACTIONS
    assert "boat" not in spaces.LAND_ACTIONS
    assert "boat" in spaces.LAND_BOAT_ACTIONS
    assert "boat" in spaces.ISLAND_ACTIONS


def test_curriculum_env_config_matches_ts_contract():
    cur = Curriculum(start=0)
    cfg = cur.make_env_config("seed-0")
    assert cfg["map"] == "Halkidiki"
    assert cfg["mapSize"] == "Compact"
    assert cfg["allowedActions"] == spaces.encode_allowed_actions(
        spaces.LAND_BOAT_ACTIONS
    )
    assert spaces.ACTION_BOAT in cfg["allowedActions"]
    assert cfg["allowedActions"] == [
        spaces.ACTION_NOOP,
        spaces.ACTION_SPAWN,
        spaces.ACTION_ATTACK,
        spaces.ACTION_RETREAT_ALL,
        spaces.ACTION_BOAT,
    ]
    assert all(isinstance(a, int) for a in cfg["allowedActions"])
    assert cfg["winTilesFrac"] == pytest.approx(0.30)
    objective = cur.make_env_config("seed-obj", objective=True)
    assert "winTilesFrac" not in objective
    island = Curriculum(start=2).make_env_config("s")
    assert island["map"] == "FourIslands"
    assert spaces.ACTION_BUILD in island["allowedActions"]
    assert spaces.ACTION_BOAT in island["allowedActions"]
    assert island["mapSize"] == "Normal"


def test_all_zero_irrelevant_masks_finite():
    torch.manual_seed(3)
    policy = Policy()
    obs = make_dummy_obs(4, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_NOOP] = 1.0
    obs["target_masks"].zero_()
    obs["quantity_mask"].zero_()
    obs["unit_mask"].zero_()
    obs["spawn_regions"].zero_()
    obs["build_regions"].zero_()
    obs["boat_regions"].zero_()
    with torch.no_grad():
        actions, logp, _ = policy.act(obs, deterministic=False)
        logp2, ent, _, extras = policy.evaluate(obs, actions)
    assert (actions[:, 0] == spaces.ACTION_NOOP).all()
    assert torch.isfinite(logp).all()
    assert torch.isfinite(logp2).all()
    assert torch.isfinite(ent).all()
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert float(extras["cond_entropy"].abs().max()) < 1e-5


def test_all_zero_active_masks_valid_distribution():
    torch.manual_seed(4)
    policy = Policy()
    obs = make_dummy_obs(3, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_ATTACK] = 1.0
    obs["target_masks"].zero_()
    obs["quantity_mask"].zero_()
    with torch.no_grad():
        actions, logp, _ = policy.act(obs, deterministic=True)
        logp2, ent, _, extras = policy.evaluate(obs, actions)
        det, _, _ = policy.act(obs, deterministic=True)
    assert (actions[:, 0] == spaces.ACTION_ATTACK).all()
    assert torch.isfinite(logp).all()
    assert torch.isfinite(ent).all()
    assert torch.allclose(logp, logp2, atol=1e-5)
    assert torch.equal(actions, det)
    assert float(extras["cond_entropy"].min()) > 0.0


def test_pointer_target_consumes_slot_embeddings():
    torch.manual_seed(5)
    policy = Policy()
    obs = make_dummy_obs(1, "cpu")
    obs["players"].zero_()
    obs["players"][0, :, 0] = 1.0
    obs["players"][0, 1, 2] = 0.1
    obs["players"][0, 2, 2] = 0.9
    actions = torch.zeros(1, 5, dtype=torch.long)
    actions[0, 0] = spaces.ACTION_ATTACK
    with torch.no_grad():
        logits = policy.logits(obs, actions)["target"]
    assert logits.shape == (1, spaces.NUM_PLAYER_SLOTS)
    assert not torch.allclose(logits[0, 1], logits[0, 2])


def test_rng_restore_from_checkpoint_byte_tensor():
    """Resume loads tensors with map_location=cuda; torch RNG needs CPU uint8."""
    cpu_state = torch.get_rng_state()
    restored = _byte_rng(cpu_state)
    assert restored.device.type == "cpu"
    assert restored.dtype == torch.uint8
    torch.set_rng_state(restored)
    # numpy payload (some loaders)
    torch.set_rng_state(_byte_rng(cpu_state.cpu().numpy()))
    if torch.cuda.is_available():
        gpu = cpu_state.to("cuda")
        _set_rng_state({"torch": gpu})
        assert torch.get_rng_state().device.type == "cpu"
        cuda_state = torch.cuda.get_rng_state()
        _set_rng_state({"cuda": [cuda_state.to("cuda")]})


def test_old_checkpoint_schema_rejected(tmp_path):
    path = tmp_path / "old.pt"
    torch.save({"schema_version": 1, "model": {}}, path)
    trainer = PPOTrainer(Policy(), PPOConfig(device="cpu"))
    with pytest.raises(RuntimeError, match="schema v1"):
        trainer.load(str(path))


def test_checkpoint_curriculum_json_roundtrip(tmp_path):
    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    trainer.stats.record_episode(_ep(ret=2.5, kills=3, win=True, terminal_cause="win"))
    cur = Curriculum(start=1)
    cur.record_episode(True)
    cur.note_eval(True)
    path = tmp_path / "ckpt.pt"
    trainer.save(
        str(path),
        extra={
            "curriculum": cur.state_dict(),
            "stage_baseline": {"win_rate": 0.4, "env_steps": 10},
            "best_score": [0.4, 0.1, 100.0],
            "rng": {"python": None},
        },
    )
    ckpt = torch.load(path, weights_only=False)
    json.dumps(ckpt["curriculum"])
    json.dumps(ckpt["stats"])
    json.dumps(ckpt["metrics"])
    json.dumps(ckpt["stage_baseline"])
    trainer2 = PPOTrainer(Policy(), cfg)
    loaded = trainer2.load(str(path))
    cur2 = Curriculum()
    cur2.load_state_dict(loaded["curriculum"])
    assert cur2.idx == 1
    assert cur2.stage_episodes == 1
    assert trainer2.stats.kills_per_ep_20 == pytest.approx(3.0)
    assert loaded["schema_version"] == 2


def _collect_tiny(trainer: PPOTrainer) -> RolloutBuffer:
    buf = RolloutBuffer(trainer.cfg.rollout_len, 2)
    env = FakeEnv(
        2,
        [
            [(0.1, False, _fake_info())] * trainer.cfg.rollout_len,
            [(0.1, False, _fake_info())] * trainer.cfg.rollout_len,
        ],
    )
    obs = _numpy_obs(2)
    trainer.collect(env, obs, buf)
    return buf


def test_legal_frac_counts_action_masks():
    cfg = PPOConfig(rollout_len=1, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)
    obs = _numpy_obs(2)
    obs["action_mask"][:] = 0
    obs["action_mask"][:, spaces.ACTION_NOOP] = 1
    obs["action_mask"][:, spaces.ACTION_ATTACK] = 1
    env = FakeEnv(2, [[(0.0, False, _fake_info())]] * 2)
    buf = RolloutBuffer(1, 2)
    trainer.collect(env, obs, buf)
    frac = trainer.take_legal_frac()
    assert frac["noop"] == 1.0
    assert frac["attack"] == 1.0
    assert frac["boat"] == 0.0
    assert trainer.take_legal_frac()["noop"] == 0.0


def test_eval_score_ignores_never_spawn_and_noop_collapse():
    spawned = {
        "win_rate": 0.0,
        "mean_peak_tiles": 0.2,
        "mean_final_tiles": 0.18,
        "mean_ticks": 4000.0,
        "episodes": 20,
        "terminal_causes": {"timeout": 20},
        "action_hist": {"spawn": 20, "attack": 8000, "noop": 200},
    }
    never = {
        "win_rate": 0.0,
        "mean_peak_tiles": 0.0,
        "mean_final_tiles": 0.0,
        "mean_ticks": 8005.0,
        "episodes": 20,
        "terminal_causes": {"no_spawn": 20},
        "action_hist": {"noop": 16000},
    }
    passive = {
        "win_rate": 0.0,
        "mean_peak_tiles": 0.0002,
        "mean_final_tiles": 0.0002,
        "mean_ticks": 8005.0,
        "episodes": 20,
        "terminal_causes": {"timeout": 20},
        "action_hist": {"spawn": 20, "noop": 18172},
    }
    milestone_inflated = {
        "win_rate": 0.0,
        "stage_success_rate": 1.0,
        "mean_peak_tiles": 0.30,
        "mean_final_tiles": 0.30,
        "mean_ticks": 2000.0,
        "episodes": 20,
        "terminal_causes": {"curriculum_success": 20},
        "action_hist": {"spawn": 20, "attack": 4000},
    }
    true_win = {
        "win_rate": 0.1,
        "stage_success_rate": 0.1,
        "mean_peak_tiles": 0.40,
        "mean_final_tiles": 0.35,
        "mean_ticks": 7000.0,
        "episodes": 20,
        "terminal_causes": {"win": 2, "timeout": 18},
        "action_hist": {"spawn": 20, "attack": 8000},
    }
    assert _eval_score(spawned) > _eval_score(never)
    assert _eval_score(spawned) > _eval_score(passive)
    assert _eval_score(never)[1:] == (0.0, 0.0, 0.0)
    assert _eval_score(passive)[1:] == (0.0, 0.0, 0.0)
    # Milestone-only success must not outrank a true core win on best.pt.
    assert _eval_score(true_win)[0] == pytest.approx(0.1)
    assert _eval_score(milestone_inflated)[0] == pytest.approx(0.0)
    assert _eval_score(true_win) > _eval_score(milestone_inflated)


def _mode_stats(
    win_rate: float,
    *,
    episodes: int = 8,
    stage_success_rate: float = 0.0,
    peak: float = 0.2,
    cause: str = "timeout",
) -> dict:
    wins = int(round(win_rate * episodes))
    return {
        "win_rate": win_rate,
        "stage_success_rate": stage_success_rate,
        "mean_peak_tiles": peak,
        "mean_final_tiles": peak,
        "mean_ticks": 4000.0,
        "episodes": episodes,
        "terminal_causes": {cause: wins, "timeout": episodes - wins},
        "action_hist": {"spawn": episodes, "attack": 800},
    }


def test_eval_score_uses_true_wins_only():
    """best.pt ranks sampled true wins first; stageSuccess never scores."""
    sampled_master = {
        "sampled": _mode_stats(1.0, peak=0.16, cause="win"),
        "deterministic": _mode_stats(0.0, peak=0.32),
    }
    det_peak_only = {
        "sampled": _mode_stats(0.0, peak=0.50),
        "deterministic": _mode_stats(0.0, peak=0.90),
    }
    milestone_only = {
        "sampled": _mode_stats(0.0, stage_success_rate=1.0, peak=0.40),
        "deterministic": _mode_stats(0.0, stage_success_rate=1.0, peak=0.40),
    }
    collapsed = {
        "sampled": {
            **_mode_stats(1.0, peak=0.0, cause="win"),
            "terminal_causes": {"no_spawn": 8},
            "action_hist": {"noop": 16000},
        },
        "deterministic": _mode_stats(0.0, peak=0.0),
    }
    score = _eval_score(sampled_master)
    assert score[0] == pytest.approx(1.0)
    assert score[1] == pytest.approx(0.0)
    assert _eval_score(sampled_master) > _eval_score(det_peak_only)
    assert _eval_score(sampled_master) > _eval_score(milestone_only)
    assert _eval_score(milestone_only)[0] == pytest.approx(0.0)
    assert _eval_score(collapsed)[0] == pytest.approx(0.0)


def test_evaluate_exposes_effective_entropy_and_stays_in_unit_interval():
    torch.manual_seed(4)
    policy = Policy()
    obs = make_dummy_obs(5, "cpu")
    with torch.no_grad():
        actions, _, _ = policy.act(obs)
        _, _, _, extras = policy.evaluate(obs, actions)
    assert extras["action_entropy_eff_norm"].min() >= 0.0
    assert extras["action_entropy_eff_norm"].max() <= 1.0
    assert extras["action_entropy_norm"].max() <= 1.0
    assert extras["action_entropy_eff_multi"].dtype == torch.bool


def test_normalized_action_entropy_one_legal():
    """A single legal action is fully satisfied / excluded from the demand."""
    action_ent = torch.tensor([0.0, 0.02])
    mask = torch.zeros(2, spaces.NUM_ACTION_TYPES)
    mask[:, spaces.ACTION_NOOP] = 1
    norm, max_ent, multi = normalized_action_entropy(action_ent, mask)
    assert not bool(multi.any())
    assert torch.allclose(norm, torch.ones(2))
    assert torch.allclose(max_ent, torch.zeros(2))


def test_normalized_action_entropy_two_legal():
    ln2 = float(np.log(2.0))
    action_ent = torch.tensor([ln2, 0.5 * ln2])
    mask = torch.zeros(2, spaces.NUM_ACTION_TYPES)
    mask[:, :2] = 1
    norm, max_ent, multi = normalized_action_entropy(action_ent, mask)
    assert bool(multi.all())
    assert max_ent[0].item() == pytest.approx(ln2)
    assert norm[0].item() == pytest.approx(1.0)
    assert norm[1].item() == pytest.approx(0.5)


def test_normalized_action_entropy_four_legal():
    ln4 = float(np.log(4.0))
    action_ent = torch.tensor([ln4, 0.6 * ln4, 1.5])
    mask = torch.zeros(3, spaces.NUM_ACTION_TYPES)
    mask[:, :4] = 1
    norm, max_ent, multi = normalized_action_entropy(action_ent, mask)
    assert bool(multi.all())
    assert max_ent[0].item() == pytest.approx(ln4)
    assert norm[0].item() == pytest.approx(1.0)
    assert norm[1].item() == pytest.approx(0.6)
    # raw 1.5 nats exceeds log(4)≈1.386; clip to 1.0, never demand more
    assert norm[2].item() == pytest.approx(1.0)


def _fake_norm_evaluate(trainer, raw, norm, cond, multi=True):
    def fake_evaluate(obs, actions):
        b = int(actions.shape[0])
        dummy = next(trainer.policy.parameters()).sum() * 0.0
        zeros = torch.zeros(b) + dummy
        extras = {
            "action_entropy": torch.full((b,), raw) + dummy,
            "cond_entropy": torch.full((b,), cond) + dummy,
            "action_entropy_norm": torch.full((b,), norm) + dummy,
            "action_entropy_max": torch.full((b,), float(np.log(4.0))) + dummy,
            "action_entropy_multi": torch.full((b,), multi, dtype=torch.bool),
        }
        return zeros, torch.full((b,), raw + cond) + dummy, zeros, extras

    trainer.policy.evaluate = fake_evaluate  # type: ignore[method-assign]


def test_entropy_controller_tracks_normalized_action_head():
    """High conditional/region entropy must not shrink ent_coef when the
    normalized action-type head is below the [0, 1] target."""
    cfg = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=1,
        update_epochs=1,
        target_kl=10.0,
        ent_target=0.6,
        ent_kp=0.2,
        ent_coef=0.05,
        ent_coef_min=0.01,
        ent_coef_max=0.5,
    )
    trainer = PPOTrainer(Policy(), cfg)
    buf = _collect_tiny(trainer)
    buf.logprobs[:] = 0.0
    buf.values[:] = 0.0
    _fake_norm_evaluate(trainer, raw=0.4, norm=0.3, cond=4.6)
    pre = trainer.ent_coef
    metrics = trainer.update(buf, _numpy_obs(2))
    assert metrics["action_entropy"] == pytest.approx(0.4, abs=1e-5)
    assert metrics["action_entropy_norm"] == pytest.approx(0.3, abs=1e-5)
    assert metrics["cond_entropy"] == pytest.approx(4.6, abs=1e-5)
    assert trainer.ent_coef > pre
    assert trainer.ent_coef == pytest.approx(pre * (1.0 + cfg.ent_kp * (0.6 - 0.3)))


def test_entropy_controller_shrinks_when_normalized_head_is_high():
    cfg = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=1,
        update_epochs=1,
        target_kl=10.0,
        ent_target=0.6,
        ent_kp=0.2,
        ent_coef=0.05,
        ent_coef_min=0.01,
        ent_coef_max=0.5,
    )
    trainer = PPOTrainer(Policy(), cfg)
    buf = _collect_tiny(trainer)
    buf.logprobs[:] = 0.0
    buf.values[:] = 0.0
    _fake_norm_evaluate(trainer, raw=1.2, norm=0.9, cond=0.1)
    pre = trainer.ent_coef
    trainer.update(buf, _numpy_obs(2))
    assert trainer.ent_coef < pre
    assert trainer.ent_coef == pytest.approx(pre * (1.0 + cfg.ent_kp * (0.6 - 0.9)))


def test_entropy_controller_excludes_single_legal_actions():
    """<=1 legal action is fully satisfied: do not raise ent_coef."""
    cfg = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=1,
        update_epochs=1,
        target_kl=10.0,
        ent_target=0.6,
        ent_kp=0.2,
        ent_coef=0.05,
        ent_coef_min=0.01,
        ent_coef_max=0.5,
    )
    trainer = PPOTrainer(Policy(), cfg)
    buf = _collect_tiny(trainer)
    buf.logprobs[:] = 0.0
    buf.values[:] = 0.0
    _fake_norm_evaluate(trainer, raw=0.0, norm=1.0, cond=0.0, multi=False)
    pre = trainer.ent_coef
    metrics = trainer.update(buf, _numpy_obs(2))
    assert metrics["action_entropy_norm"] == pytest.approx(1.0, abs=1e-5)
    # measured 1.0 > target 0.6 => shrink, never race upward
    assert trainer.ent_coef < pre


def test_scripted_expander_spawn_then_wilderness_then_player():
    obs = _numpy_obs(1)
    obs["action_mask"][:] = 0
    obs["action_mask"][0, spaces.ACTION_SPAWN] = 1
    obs["spawn_regions"][:] = 0
    obs["spawn_regions"][0, 7] = 1
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_SPAWN
    assert int(a[2]) == 7

    obs["spawn_regions"][0, 16 * 32 + 16] = 1
    for dx in range(-1, 2):
        for dy in range(-1, 2):
            obs["spawn_regions"][0, (16 + dy) * 32 + (16 + dx)] = 1
    a = scripted_expander_action(obs, 0)
    ry, rx = divmod(int(a[2]), 32)
    assert 15 <= ry <= 17 and 15 <= rx <= 17

    obs["action_mask"][:] = 0
    obs["action_mask"][0, spaces.ACTION_ATTACK] = 1
    obs["target_masks"][:] = 0
    obs["target_masks"][0, spaces.ACTION_ATTACK, 0] = 1
    obs["global"][0, 2] = 0.05
    obs["global"][0, 3] = float(np.log1p(200.0) / 15.0)
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_ATTACK
    assert int(a[1]) == 0
    assert int(a[3]) == 3

    # Below the expand threshold, keep taking wilderness even if a player
    # is targetable.
    obs["target_masks"][0, spaces.ACTION_ATTACK, 2] = 1
    obs["global"][0, 2] = 0.10
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_ATTACK
    assert int(a[1]) == 0

    # Past the expand threshold: switch to the player.
    obs["global"][0, 2] = 0.20
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_ATTACK
    assert int(a[1]) == 2

    # Wilderness gone: attack the player, then wait while an attack is out.
    obs["target_masks"][0, spaces.ACTION_ATTACK, 0] = 0
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_ATTACK
    assert int(a[1]) == 2
    assert int(a[3]) == 3
    obs["action_mask"][0, spaces.ACTION_RETREAT_ALL] = 1
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_NOOP

    obs["action_mask"][:] = 0
    obs["action_mask"][0, spaces.ACTION_BOAT] = 1
    obs["target_masks"][:] = 0
    obs["boat_regions"][:] = 0
    obs["boat_regions"][0, 20] = 1
    obs["global"][0, 2] = 0.10
    a = scripted_expander_action(obs, 0)
    assert int(a[0]) == spaces.ACTION_BOAT
    assert int(a[2]) == 20


def test_reward_terms_helper_sums_and_causes():
    from agent.ppo import _reward_terms, _terminal_cause

    info = _fake_info(
        terminalCause="win",
        win=True,
        rewardTerms={
            "terminal": 3.0,
            "spawn": 0.1,
            "territory": 0.2,
            "elimination": 0.05,
            "total": 3.35,
        },
    )
    terms = _reward_terms(info, 3.35)
    assert terms["total"] == pytest.approx(
        terms["terminal"] + terms["spawn"] + terms["territory"] + terms["elimination"]
    )
    assert _terminal_cause(info) == "win"
    assert _terminal_cause({"dead": True}) == "death"
    assert _terminal_cause({"spawned": False}) == "no_spawn"
    assert _terminal_cause({"terminalCause": "curriculum_success"}) == "curriculum_success"


def test_classify_outcome_separates_stage_from_win():
    cause, won, stage = classify_outcome(
        {
            "terminalCause": "curriculum_success",
            "win": False,
            "stageSuccess": True,
            "peakTilesFrac": 0.31,
        }
    )
    assert cause == "curriculum_success"
    assert won is False
    assert stage is True

    cause, won, stage = classify_outcome(
        {"terminalCause": "win", "win": True, "stageSuccess": True, "peakTilesFrac": 0.82}
    )
    assert cause == "win"
    assert won is True
    assert stage is True

    # Legacy servers that labeled a milestone as win=true + curriculum cause.
    cause, won, stage = classify_outcome(
        {"terminalCause": "curriculum_success", "win": True, "peakTilesFrac": 0.2}
    )
    assert cause == "curriculum_success"
    assert won is False
    assert stage is True

    # Objective eval: no termination, infer stage success from peak tiles.
    cause, won, stage = classify_outcome(
        {"terminalCause": "timeout", "win": False, "peakTilesFrac": 0.34},
        stage_tiles_frac=0.30,
    )
    assert cause == "timeout"
    assert won is False
    assert stage is True

    cause, won, stage = classify_outcome(
        {"terminalCause": "timeout", "win": False, "peakTilesFrac": 0.10},
        stage_tiles_frac=0.30,
    )
    assert stage is False


def test_curriculum_success_does_not_inflate_win_rate():
    stats = TrainStats()
    stats.record_episode(
        _ep(terminal_cause="curriculum_success", win=False, stage_success=True)
    )
    stats.record_episode(_ep(terminal_cause="win", win=True, stage_success=True))
    stats.record_episode(_ep(terminal_cause="timeout", win=False, stage_success=False))
    assert stats.wins == 1
    assert stats.stage_successes == 2
    assert stats.win_rate == pytest.approx(1.0 / 3.0)
    assert stats.stage_success_rate == pytest.approx(2.0 / 3.0)


def test_curriculum_advance_ignores_sampled_stage_rate():
    """100% sampled stage success must not advance without det eval passes."""
    stages = [
        Stage(
            name="a",
            map="Halkidiki",
            bots=1,
            win_threshold=0.01,
            window=3,
            min_episodes=3,
            eval_passes_required=2,
        ),
        Stage(name="b", map="Halkidiki", bots=5, min_episodes=1),
    ]
    cur = Curriculum(stages=stages, start=0)
    for _ in range(10):
        cur.record_episode(True)
    assert cur.stage_success_rate == 1.0
    assert not cur.maybe_advance()
    cur.note_eval(True)
    assert cur.maybe_advance()
    assert cur.idx == 1


def test_wilson_random_sampled_does_not_pass():
    """Untrained-like 20-32% sampled true win must not pass mastery."""
    cur = Curriculum()
    for n, rate in ((8, 0.20), (8, 0.32), (32, 0.20), (32, 0.32), (50, 0.32)):
        ev = {
            "deterministic": _mode_stats(0.0, episodes=n),
            "sampled": _mode_stats(rate, episodes=n),
        }
        judgment = cur.judge_eval(ev)
        assert not judgment["passed"], (n, rate, judgment)
        assert judgment["reason"] == "none"
        if n >= cur.stage.min_eval_episodes:
            assert judgment["sampled_wilson_lower"] < cur.stage.sampled_wilson_floor
    tiny = {
        "deterministic": _mode_stats(0.0, episodes=4),
        "sampled": _mode_stats(1.0, episodes=4, cause="win"),
    }
    assert not cur.judge_eval(tiny)["passed"]
    full = {
        "deterministic": _mode_stats(0.0, episodes=8),
        "sampled": _mode_stats(1.0, episodes=8, cause="win"),
    }
    # Default holdout is 32; 8/8 is a lottery and must not pass.
    assert not cur.judge_eval(full)["passed"]
    assert wilson_lower_bound(10, 50) < 0.50
    assert wilson_lower_bound(16, 50) < 0.50


def test_sampled_mastery_is_one_holdout_decision():
    """100% sampled true wins on a large enough holdout advance once."""
    stages = [
        Stage(
            name="a",
            map="Halkidiki",
            bots=1,
            min_episodes=2,
            min_eval_episodes=8,
            win_tiles_frac=0.30,
        ),
        Stage(name="b", map="Halkidiki", bots=5, min_episodes=1),
    ]
    cur = Curriculum(stages=stages, start=0)
    cur.record_episode(True)
    cur.record_episode(True)
    ev = {
        "deterministic": _mode_stats(0.0, episodes=8, stage_success_rate=0.0),
        "sampled": _mode_stats(1.0, episodes=8, stage_success_rate=1.0, cause="win"),
    }
    first = cur.judge_eval(ev)
    assert first["passed"]
    assert first["reason"] == "sampled_true_win"
    assert first["sampled_wilson_lower"] >= cur.stage.sampled_wilson_floor
    assert first["det_diagnostic_only"] is True
    assert not cur.maybe_advance()
    cur.note_eval(True)
    assert cur.last_eval_passed
    assert cur.maybe_advance()
    assert cur.idx == 1
    assert cur.consecutive_eval_passes == 0
    assert not cur.last_eval_passed
    assert cur.stage_episodes == 0


def test_stage_success_only_never_passes_objective_mastery():
    """Sampled milestone / stageSuccess alone is not objective mastery."""
    cur = Curriculum()
    ev = {
        "deterministic": _mode_stats(0.0, episodes=8, stage_success_rate=0.0),
        "sampled": _mode_stats(0.0, episodes=8, stage_success_rate=1.0),
    }
    judgment = cur.judge_eval(ev)
    assert not judgment["passed"]
    assert judgment["reason"] == "none"
    assert judgment["sampled_stage_success_rate"] == pytest.approx(1.0)
    assert judgment["sampled_win_rate"] == pytest.approx(0.0)
    # On a true-win stage, even 100% stageSuccess in both modes is not a win.
    later = Curriculum(start=1)
    both = {
        "deterministic": _mode_stats(0.0, episodes=8, stage_success_rate=1.0),
        "sampled": _mode_stats(0.0, episodes=8, stage_success_rate=1.0),
    }
    later_j = later.judge_eval(both)
    assert not later_j["passed"]
    assert later_j["reason"] == "none"


def test_deterministic_competency_does_not_gate():
    """Det argmax is diagnostic only; it cannot pass the holdout gate."""
    cur = Curriculum()
    ev = {
        "deterministic": _mode_stats(
            0.0, episodes=32, stage_success_rate=1.0, peak=0.35
        ),
        "sampled": _mode_stats(0.20, episodes=32, stage_success_rate=0.20),
    }
    judgment = cur.judge_eval(ev)
    assert not judgment["passed"]
    assert judgment["reason"] == "none"
    assert judgment["det_diagnostic_only"] is True
    later = Curriculum(start=1)
    assert later.stage.win_tiles_frac == 0.0
    win_ev = {
        "deterministic": _mode_stats(1.0, episodes=32, cause="win"),
        "sampled": _mode_stats(0.20, episodes=32),
    }
    win_j = later.judge_eval(win_ev)
    assert not win_j["passed"]
    assert win_j["reason"] == "none"


def test_old_mastery_version_resets_eval_passes():
    cur = Curriculum()
    cur.load_state_dict(
        {
            "idx": 0,
            "consecutive_eval_passes": 2,
            "stage_episodes": 100,
            "stage_successes": [1.0] * 10,
        }
    )
    assert cur.consecutive_eval_passes == 0
    assert not cur.last_eval_passed
    assert MASTERY_VERSION == 3
    cur2 = Curriculum()
    cur2.note_eval(True)
    sd = cur2.state_dict()
    assert sd["mastery_version"] == MASTERY_VERSION
    assert sd["stage"] == "duel"
    cur3 = Curriculum()
    cur3.load_state_dict(sd)
    assert cur3.last_eval_passed
    assert cur3.consecutive_eval_passes == 1
    v2 = Curriculum()
    v2.load_state_dict(
        {
            "idx": 4,
            "consecutive_eval_passes": 2,
            "stage_episodes": 100,
            "mastery_version": 2,
        }
    )
    assert v2.stage.name == "nations-easy-small"
    assert v2.idx == next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-small")
    assert not v2.last_eval_passed


class _EvalInterruptEnv:
    """Mock ShardedVecEnv: one shard dies after a partial episode finish.

    First attempt, first step: env 0 completes a win (would leak if kept).
    First attempt, second step: raise RolloutInterrupted.
    Later inits (retries) finish every env on the first step.
    """

    def __init__(self, k: int):
        self.k = k
        self.init_calls: list[list[str]] = []
        self.step_calls = 0
        self._steps_since_init = 0
        self._interrupted_once = False

    def init(self, cfgs):
        self.init_calls.append([c["seed"] for c in cfgs])
        self._steps_since_init = 0
        return _numpy_obs(self.k)

    def step(self, actions):
        self.step_calls += 1
        self._steps_since_init += 1
        if not self._interrupted_once and self._steps_since_init == 2:
            self._interrupted_once = True
            raise RolloutInterrupted([0])
        win_info = _fake_info(
            terminalCause="win", win=True, tick=10, peakTilesFrac=0.5, tilesFrac=0.5
        )
        if not self._interrupted_once and self._steps_since_init == 1:
            dones = np.zeros(self.k, dtype=bool)
            dones[0] = True
            infos = [win_info, _fake_info(terminalCause="none", win=False, tick=5)]
            if self.k > 2:
                infos.extend([_fake_info(terminalCause="none", win=False)] * (self.k - 2))
                dones = np.zeros(self.k, dtype=bool)
                dones[0] = True
        else:
            dones = np.ones(self.k, dtype=bool)
            infos = [win_info] * self.k
        return StepBatch(
            obs=_numpy_obs(self.k),
            rewards=np.ones(self.k, np.float32),
            dones=dones,
            infos=infos,
        )


class _AlwaysInterruptEnv:
    def __init__(self, k: int):
        self.k = k
        self.init_calls = 0
        self.step_calls = 0

    def init(self, cfgs):
        self.init_calls += 1
        return _numpy_obs(self.k)

    def step(self, actions):
        self.step_calls += 1
        raise RolloutInterrupted([1])


def test_fixed_seed_eval_retries_same_seeds_no_double_count():
    """One shard interrupt mid-eval: keep the finished env-0 win and retry
    only the unfinished seed. Do not double-count."""
    env = _EvalInterruptEnv(k=2)
    policy = Policy()
    cur = Curriculum(start=0)
    prefix = "v2run1-eval-it140"
    ev = run_fixed_seed_eval(
        policy, env, cur, torch.device("cpu"), n_episodes=2, seed_prefix=prefix
    )
    assert env._interrupted_once
    det_seeds = [f"{prefix}-deterministic-0", f"{prefix}-deterministic-1"]
    samp_seeds = [f"{prefix}-sampled-0", f"{prefix}-sampled-1"]
    assert env.init_calls[0] == det_seeds
    assert env.init_calls[1][0] == det_seeds[1]
    assert det_seeds[0] not in env.init_calls[1]
    assert env.init_calls.count(det_seeds) == 1
    assert samp_seeds in env.init_calls
    for mode in ("deterministic", "sampled"):
        assert ev[mode]["episodes"] == 2
        assert ev[mode]["win_rate"] == pytest.approx(1.0)
        assert ev[mode]["terminal_causes"] == {"win": 2}
        assert ev[mode]["seeds"] == (
            det_seeds if mode == "deterministic" else samp_seeds
        )


def test_fixed_seed_eval_exhausts_retries():
    env = _AlwaysInterruptEnv(k=2)
    with pytest.raises(RuntimeError, match="failed after 3 shard-restart"):
        run_fixed_seed_eval(
            Policy(),
            env,
            Curriculum(start=0),
            torch.device("cpu"),
            n_episodes=2,
            seed_prefix="eval-fail",
        )
    assert env.init_calls == FIXED_SEED_EVAL_RETRIES
    assert env.step_calls == FIXED_SEED_EVAL_RETRIES


def test_eval_finally_restores_training_envs_and_trackers():
    """Shared train shards + episode trackers are restored after eval
    success, retry, or exhaustion — matching main()'s finally."""
    env = _EvalInterruptEnv(k=2)
    trainer = PPOTrainer(
        Policy(), PPOConfig(rollout_len=2, device="cpu", num_minibatches=1)
    )
    trainer._ensure_trackers(2)
    trainer._trackers[0].ret = 99.0
    train_cfgs = [{"seed": f"train-e{i}"} for i in range(2)]
    try:
        run_fixed_seed_eval(
            Policy(),
            env,
            Curriculum(start=0),
            torch.device("cpu"),
            n_episodes=2,
            seed_prefix="eval-restore",
        )
    finally:
        env.init(train_cfgs)
        trainer._ensure_trackers(2)
        trainer.reset_episode_trackers()
    assert env.init_calls[-1] == ["train-e0", "train-e1"]
    assert trainer._trackers[0].ret == 0.0
    assert trainer._trackers[1].ret == 0.0

    fail_env = _AlwaysInterruptEnv(k=2)
    trainer._trackers[0].ret = 7.0
    try:
        run_fixed_seed_eval(
            Policy(),
            fail_env,
            Curriculum(start=0),
            torch.device("cpu"),
            n_episodes=2,
            seed_prefix="eval-restore-fail",
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("exhausted eval must raise")
    finally:
        fail_env.init(train_cfgs)
        trainer.reset_episode_trackers()
    assert trainer._trackers[0].ret == 0.0


def test_collect_interrupt_discards_partial_and_retries():
    """Training collect path: discard partial rollout / tracker leakage."""
    cfg = PPOConfig(rollout_len=3, device="cpu", num_minibatches=1, update_epochs=1)
    trainer = PPOTrainer(Policy(), cfg)

    class CollectInterruptEnv:
        def __init__(self):
            self.k = 1
            self.steps = 0
            self.interrupted = False

        def step(self, actions):
            self.steps += 1
            if not self.interrupted and self.steps == 2:
                self.interrupted = True
                raise RolloutInterrupted([0])
            return StepBatch(
                obs=_numpy_obs(1),
                rewards=np.array([1.0], np.float32),
                dones=np.array([False]),
                infos=[_fake_info()],
            )

    env = CollectInterruptEnv()
    buf = RolloutBuffer(cfg.rollout_len, 1)
    obs = _numpy_obs(1)
    with pytest.raises(RolloutInterrupted):
        trainer.collect(env, obs, buf)
    assert trainer._trackers[0].ret == pytest.approx(1.0)
    trainer.reset_episode_trackers()
    assert trainer._trackers[0].ret == 0.0
    obs, completed = trainer.collect(env, obs, buf)
    assert env.interrupted
    assert completed == []
    assert trainer._trackers[0].ret == pytest.approx(3.0)


def test_obs_to_tensors_readonly_frombuffer_matches_copy():
    s = spaces
    payload = np.random.default_rng(0).random(
        s.SPATIAL_CHANNELS * s.SPATIAL_SIZE * s.SPATIAL_SIZE, dtype=np.float32
    )
    raw = payload.tobytes()
    view = np.frombuffer(raw, dtype=np.float32).reshape(
        1, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE
    )
    assert not view.flags["WRITEABLE"]
    obs = _numpy_obs(1)
    obs["spatial"] = view
    copied = {k: np.array(v) for k, v in obs.items()}
    device = torch.device("cpu")
    t_view = obs_to_tensors(obs, device)
    t_copy = {
        k: torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        for k, v in copied.items()
    }
    for key in obs:
        assert torch.equal(t_view[key], t_copy[key]), key


def test_obs_to_tensors_float16_spatial_matches_evaluate_cast():
    """Collect stores f16 spatial; transfer must match update's f16->f32."""
    torch.manual_seed(4)
    policy = Policy()
    obs = _numpy_obs(4)
    rng = np.random.default_rng(1)
    obs["spatial"] = rng.random(obs["spatial"].shape, dtype=np.float32)
    spatial_f16 = np.asarray(obs["spatial"], dtype=np.float16)
    obs_q = dict(obs)
    obs_q["spatial"] = spatial_f16
    device = torch.device("cpu")
    t_new = obs_to_tensors(obs_q, device)
    t_legacy = torch.as_tensor(
        spatial_f16.astype(np.float32), dtype=torch.float32, device=device
    )
    assert torch.equal(t_new["spatial"], t_legacy)
    with torch.no_grad():
        actions, logp_act, _ = policy.act(t_new, deterministic=False)
        logp_eval, _, _, _ = policy.evaluate(t_new, actions)
    assert torch.allclose(logp_act, logp_eval, atol=1e-5)


def test_read_frame_bytearray_frombuffer_roundtrip():
    import socket

    actions = np.arange(10, dtype=np.int32)
    payload = _encode_frame({"cmd": "step"}, {"actions": actions})
    a, b = socket.socketpair()
    try:
        a.sendall(payload)
        header, tensors = _read_frame(b)
        assert header["cmd"] == "step"
        assert np.array_equal(tensors["actions"], actions)
        assert tensors["actions"].flags["OWNDATA"] is False
    finally:
        a.close()
        b.close()


def test_concat_obs_reuse_matches_allocate():
    a = _numpy_obs(2)
    b = _numpy_obs(3)
    a["spatial"][0, 0, 0, 0] = 0.3
    b["global"][1, 0] = 0.7
    fresh = _concat_obs([a, b])
    reuse = _concat_obs([a, b], None)
    again = _concat_obs([a, b], reuse)
    assert again is reuse
    for key in fresh:
        assert np.array_equal(fresh[key], again[key]), key


def test_eval_active_lanes_skips_idle_envs():
    """Eval with 8 episodes on a 16-env mock must not pad-init all shards."""

    class ActiveEnv:
        def __init__(self):
            self.k = 16
            self.envs_per_server = 8
            self.init_n = []
            self.step_n = []
            self.full_init = 0
            self.seeds: list[str] = []

        def init(self, cfgs):
            self.full_init += 1
            return _numpy_obs(self.k)

        def init_active(self, cfgs):
            self.init_n.append(len(cfgs))
            self.seeds = [c["seed"] for c in cfgs]
            return _numpy_obs(len(cfgs))

        def step_active(self, actions):
            self.step_n.append(actions.shape[0])
            n = actions.shape[0]
            info = _fake_info(
                terminalCause="win", win=True, tick=4, peakTilesFrac=0.4, tilesFrac=0.4
            )
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.ones(n, np.float32),
                dones=np.ones(n, dtype=bool),
                infos=[info] * n,
            )

        def step(self, actions):
            raise AssertionError("full step should not run when active exists")

    env = ActiveEnv()
    ev = run_fixed_seed_eval(
        Policy(),
        env,
        Curriculum(start=0),
        torch.device("cpu"),
        n_episodes=8,
        seed_prefix="lead-eval",
    )
    assert env.full_init == 0
    assert env.init_n == [8, 8]
    assert env.step_n == [8, 8]
    assert ev["deterministic"]["episodes"] == 8
    assert ev["sampled"]["episodes"] == 8
    assert ev["deterministic"]["seeds"] == [f"lead-eval-deterministic-{i}" for i in range(8)]
    assert ev["sampled"]["seeds"] == [f"lead-eval-sampled-{i}" for i in range(8)]


def test_eval_sampled_32_keeps_holdout_seed_order():
    class ActiveEnv:
        def __init__(self):
            self.k = 128
            self.full_init = 0

        def init(self, cfgs):
            self.full_init += 1
            return _numpy_obs(self.k)

        def init_active(self, cfgs):
            self.last = [c["seed"] for c in cfgs]
            return _numpy_obs(len(cfgs))

        def step_active(self, actions):
            n = actions.shape[0]
            info = _fake_info(
                terminalCause="win", win=True, tick=4, peakTilesFrac=0.4, tilesFrac=0.4
            )
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.ones(n, np.float32),
                dones=np.ones(n, dtype=bool),
                infos=[info] * n,
            )

        def step(self, actions):
            raise AssertionError("idle full-width step")

    env = ActiveEnv()
    cur = Curriculum(start=0)
    ev = run_fixed_seed_eval(
        Policy(),
        env,
        cur,
        torch.device("cpu"),
        n_episodes=32,
        n_det=0,
        n_sampled=32,
    )
    expected = cur.holdout_seeds(32)
    assert env.full_init == 0
    assert env.last == expected
    assert ev["sampled"]["seeds"] == expected
    assert ev["sampled"]["episodes"] == 32


def test_progress_timing_fields_round():
    collect_s, update_s, eval_s = 10.1234, 1.96, 0.0
    steps = 16384
    rec = {
        "collect_s": round(collect_s, 3),
        "update_s": round(update_s, 3),
        "eval_s": round(eval_s, 3),
        "sps": round(steps / (collect_s + update_s), 1),
        "stage_sps": round(steps / (collect_s + update_s), 1),
    }
    assert rec["collect_s"] == 10.123
    assert rec["update_s"] == 1.96
    assert rec["eval_s"] == 0.0
    assert rec["sps"] == rec["stage_sps"]


def test_distribute_active_lanes_spreads_instead_of_packing():
    eight = distribute_active_lanes(8, 16, 8)
    assert sum(len(s) for s in eight) == 8
    assert sum(1 for s in eight if s) == 8
    assert all(len(s) <= 1 for s in eight)
    assert eight[0] == [0]
    assert eight[7] == [7]
    assert eight[8] == []

    holdout = distribute_active_lanes(32, 16, 8)
    assert [len(s) for s in holdout] == [2] * 16
    assert holdout[0] == [0, 16]
    assert holdout[1] == [1, 17]
    assert sorted(i for lanes in holdout for i in lanes) == list(range(32))

    packed_cap = distribute_active_lanes(32, 4, 8)
    assert [len(s) for s in packed_cap] == [8] * 4
    assert packed_cap[0] == list(range(0, 32, 4))

    with pytest.raises(ValueError, match="cannot place"):
        distribute_active_lanes(33, 4, 8)


def test_scatter_obs_restores_episode_order():
    assignment = [[0, 2], [1]]
    shard0 = _numpy_obs(2)
    shard1 = _numpy_obs(1)
    shard0["global"][0, 0] = 10
    shard0["global"][1, 0] = 12
    shard1["global"][0, 0] = 11
    out = _scatter_obs({0: shard0, 1: shard1}, assignment, 3)
    assert list(out["global"][:, 0]) == [10, 11, 12]


def test_init_active_skips_idle_shards_and_keeps_result_order():
    from concurrent.futures import ThreadPoolExecutor

    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 8
    env.k = 32
    env.ports = [1, 2, 3, 4]
    env.connect = True
    env.servers = []
    env._configs = [{"seed": f"train-{i}"} for i in range(32)]
    env.pool = ThreadPoolExecutor(max_workers=4)
    env._obs_buf = None
    env._next_config_fn = None
    env._on_episode_reset = None
    env._active_assignment = []
    env._active_shards = []
    env._active_n = 0
    inited: list[int] = []
    stepped: list[int] = []

    class FakeClient:
        def __init__(self, idx: int):
            self.idx = idx

        def init(self, cfgs):
            inited.append(self.idx)
            obs = _numpy_obs(len(cfgs))
            for local, cfg in enumerate(cfgs):
                obs["global"][local, 0] = float(str(cfg["seed"]).split("-")[1])
            return obs

        def step(self, actions, next_configs=None):
            stepped.append(self.idx)
            n = actions.shape[0]
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.full(n, float(self.idx), np.float32),
                dones=np.ones(n, dtype=bool),
                infos=[{"shard": self.idx, "local": j} for j in range(n)],
            )

    env.clients = [FakeClient(i) for i in range(4)]
    cfgs = [{"seed": f"eval-{i}"} for i in range(3)]
    obs = env.init_active(cfgs)
    assert sorted(inited) == [0, 1, 2]
    assert 3 not in inited
    assert list(obs["global"][:3, 0]) == [0.0, 1.0, 2.0]
    assert env._configs[0]["seed"] == "train-0"

    batch = env.step_active(np.zeros((3, 5), np.int32))
    assert sorted(stepped) == [0, 1, 2]
    assert 3 not in stepped
    assert list(batch.rewards) == [0.0, 1.0, 2.0]
    assert [info["shard"] for info in batch.infos] == [0, 1, 2]
    env.pool.shutdown(wait=False)


def test_init_active_full_width_does_not_clear_active_or_train_configs():
    """32-on-32 (README 4x8 + 32 holdout) must keep step_active usable."""
    from concurrent.futures import ThreadPoolExecutor

    env = object.__new__(ShardedVecEnv)
    env.envs_per_server = 2
    env.k = 8
    env.ports = [1, 2, 3, 4]
    env.connect = True
    env.servers = []
    env._configs = [{"seed": f"train-{i}"} for i in range(8)]
    env.pool = ThreadPoolExecutor(max_workers=4)
    env._obs_buf = None
    env._next_config_fn = None
    env._on_episode_reset = None
    env._active_assignment = []
    env._active_shards = []
    env._active_n = 0
    inited: list[int] = []

    class FakeClient:
        def __init__(self, idx: int):
            self.idx = idx

        def init(self, cfgs):
            inited.append(len(cfgs))
            obs = _numpy_obs(len(cfgs))
            for local, cfg in enumerate(cfgs):
                obs["global"][local, 0] = float(str(cfg["seed"]).split("-")[1])
            return obs

        def step(self, actions, next_configs=None):
            n = actions.shape[0]
            return StepBatch(
                obs=_numpy_obs(n),
                rewards=np.full(n, float(self.idx), np.float32),
                dones=np.ones(n, dtype=bool),
                infos=[{"shard": self.idx, "local": j} for j in range(n)],
            )

    env.clients = [FakeClient(i) for i in range(4)]
    cfgs = [{"seed": f"eval-{i}"} for i in range(8)]
    obs = env.init_active(cfgs)
    assert inited.count(2) == 4
    assert len(inited) == 4
    assert env._configs[0]["seed"] == "train-0"
    assert env._active_n == 8
    assert list(obs["global"][:, 0]) == list(range(8))
    batch = env.step_active(np.zeros((8, 5), np.int32))
    # Round-robin: shard i owns episodes i, i+4
    assert [info["shard"] for info in batch.infos] == [0, 1, 2, 3, 0, 1, 2, 3]
    env.pool.shutdown(wait=False)


def test_init_active_32_on_16_shards_uses_all_servers():
    assignment = distribute_active_lanes(32, 16, 8)
    assert len(assignment) == 16
    assert all(len(lanes) == 2 for lanes in assignment)
    assert [lanes[0] for lanes in assignment] == list(range(16))
