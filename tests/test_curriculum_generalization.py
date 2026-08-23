"""Regression tests for holdout mastery, stage migration, map sets, and seeds."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.curriculum import (
    DEFAULT_HOLDOUT_EPISODES,
    STAGES,
    V2_STAGE_NAMES,
    Curriculum,
    CurriculumRestoreError,
    Stage,
    all_curriculum_maps,
    wilson_lower_bound,
)
from agent.env_client import ShardedVecEnv, _encode_frame
from agent.maps import (
    CURATED_MAP_KEYS,
    assert_map_key,
    installed_game_map_keys,
    parse_game_map_keys,
    validate_map_list,
)
from agent.model import Policy
from agent.seeds import SeedStream, holdout_seed, mix_seed, pick_from_set
from agent.train import (
    parse_stage_arg,
    resolve_holdout_episodes,
    run_fixed_seed_eval,
)
from tests.test_rl_pipeline import _EvalInterruptEnv, _mode_stats


def test_installed_maps_include_curated_keys():
    installed = installed_game_map_keys()
    assert installed, "OpenFront Maps.gen.ts must be present"
    missing = CURATED_MAP_KEYS - installed
    assert not missing
    validate_map_list(sorted(CURATED_MAP_KEYS))
    with pytest.raises(ValueError, match="unknown OpenFront map key"):
        assert_map_key("NotARealMap")
    with pytest.raises(ValueError, match="unknown OpenFront map key"):
        Stage(name="bad", map="NotARealMap", bots=1)


def test_stage_identity_v2_idx_migrates_and_unknown_fails():
    for old_idx, name in enumerate(V2_STAGE_NAMES):
        cur = Curriculum()
        cur.load_state_dict({"idx": old_idx, "stage_episodes": 10, "mastery_version": 2})
        assert cur.stage.name == name
        assert cur.idx == next(i for i, s in enumerate(STAGES) if s.name == name)
        assert not cur.last_eval_passed
    named = Curriculum()
    named.load_state_dict({"stage": "nations-easy-small", "idx": 0, "mastery_version": 3})
    assert named.stage.name == "nations-easy-small"
    # Live v2run1 latest.pt is idx-only 5 (nations-easy-mid). Raw current
    # idx 5 is nations-easy-small after intro was inserted.
    mid = Curriculum()
    mid.load_state_dict({"idx": 5, "stage_episodes": 15, "mastery_version": 2})
    assert mid.stage.name == "nations-easy-mid"
    assert mid.idx == next(i for i, s in enumerate(STAGES) if s.name == "nations-easy-mid")
    with pytest.raises(CurriculumRestoreError, match="not in the current curriculum"):
        Curriculum().load_state_dict({"stage": "does-not-exist"})
    with pytest.raises(CurriculumRestoreError, match="outside the v2 stage table"):
        Curriculum().load_state_dict({"idx": 99})


def test_inserted_intro_stage_before_eight_nations():
    names = [s.name for s in STAGES]
    assert names.index("nations-easy-intro") < names.index("nations-easy-small")
    intro = next(s for s in STAGES if s.name == "nations-easy-intro")
    small = next(s for s in STAGES if s.name == "nations-easy-small")
    assert intro.nations == 3
    assert intro.bots < small.bots
    assert intro.difficulty == "Easy"
    pen = next(s for s in STAGES if s.name == "nations-easy-peninsula")
    mid = next(s for s in STAGES if s.name == "nations-easy-mid")
    assert names.index(pen.name) < names.index(mid.name)
    assert pen.map_size == "Compact"
    world = next(s for s in STAGES if s.name == "nations-world-easy")
    impossible = next(s for s in STAGES if s.name == "nations-impossible-world")
    assert names.index(world.name) < names.index(impossible.name)
    assert world.difficulty == "Easy"
    assert world.bots < impossible.bots


def test_holdout_seeds_frozen_and_independent_of_iteration():
    a = holdout_seed("duel", 0)
    b = holdout_seed("duel", 0)
    c = holdout_seed("duel", 1)
    d = holdout_seed("small-ffa", 0)
    assert a == b
    assert a != c
    assert a != d
    assert not c.startswith(a)
    cur = Curriculum()
    seeds = cur.holdout_seeds(DEFAULT_HOLDOUT_EPISODES)
    assert len(seeds) == DEFAULT_HOLDOUT_EPISODES
    assert len(set(seeds)) == DEFAULT_HOLDOUT_EPISODES
    cfgs = cur.holdout_configs(4)
    assert all(c["map"] == "Halkidiki" for c in cfgs)
    assert all(c["mapSize"] == "Compact" for c in cfgs)
    assert all(c["shaping"] == 0.0 for c in cfgs)


def test_train_map_set_is_canonical_until_holdout_unlock():
    cur = Curriculum(start=2)
    assert cur.stage.name == "island-ffa"
    assert cur.stage.map == "FourIslands"
    assert "Hawaii" in cur.stage.train_map_set
    cfg = cur.make_env_config("any-seed", suite="train")
    assert cfg["map"] == "FourIslands"
    # Episode-count burn-in no longer unlocks extra maps.
    cur.stage_episodes = max(cur.stage.map_burn_in, 10_000)
    maps = {cur.make_env_config(mix_seed("m", i), suite="train")["map"] for i in range(40)}
    assert maps == {"FourIslands"}
    hold = cur.make_env_config(mix_seed("h", 0), suite="holdout")
    assert hold["map"] == "FourIslands"
    ood = cur.make_env_config(mix_seed("o", 0), suite="ood")
    assert ood["map"] in cur.stage.ood_maps
    cur.maps_unlocked = True
    unlocked = {cur.make_env_config(mix_seed("m", i), suite="train")["map"] for i in range(40)}
    assert "FourIslands" in unlocked
    assert unlocked & set(cur.stage.train_maps)


def test_seed_stream_persists_and_does_not_replay():
    s = SeedStream("run", 1, 4)
    first = [s.env_seed(i) for i in range(4)]
    assert len(set(first)) == 4
    nxt = s.next_env_seed(0)
    assert nxt != s.env_seed(0)
    s.note_done(0)
    assert s.env_seed(0) == nxt
    sd = s.state_dict()
    s2 = SeedStream("run", 1, 4)
    s2.load_state_dict(sd)
    assert s2.env_seed(0) == s.env_seed(0)
    assert s2.episode_index[0] == 1
    assert s2.env_seed(1) == first[1]
    # Independent mix, not sequential suffixes / djb2 family.
    assert first[1] != first[0] + "1"
    assert mix_seed("train", "run", 1, 0, 0) != mix_seed("train", "run", 1, 1, 0)


def test_seed_stream_shard_count_change_pads():
    s = SeedStream("run", 1, 2)
    s.note_done(0)
    s.note_done(0)
    loaded = SeedStream("run", 1, 4)
    loaded.load_state_dict(s.state_dict())
    assert loaded.episode_index[:2] == [2, 0]
    assert loaded.episode_index[2:] == [0, 0]


def test_n32_wilson_gate_requires_26_wins():
    cur = Curriculum()
    assert cur.stage.min_eval_episodes == 32
    fail = {
        "deterministic": _mode_stats(1.0, episodes=8, cause="win"),
        "sampled": _mode_stats(25 / 32, episodes=32, cause="win"),
    }
    assert not cur.judge_eval(fail)["passed"]
    ok = {
        "deterministic": _mode_stats(0.0, episodes=8),
        "sampled": _mode_stats(26 / 32, episodes=32, cause="win"),
    }
    j = cur.judge_eval(ok)
    assert j["passed"]
    assert j["reason"] == "sampled_true_win"
    assert j["sampled_wilson_lower"] >= cur.stage.sampled_wilson_floor
    assert wilson_lower_bound(26, 32) >= 0.50


def test_holdout_eval_retries_same_frozen_seeds(monkeypatch):
    env = _EvalInterruptEnv(k=2)
    cur = Curriculum(start=0)
    expected = cur.holdout_seeds(2)
    ev = run_fixed_seed_eval(
        Policy(), env, cur, torch.device("cpu"), n_episodes=2, n_det=2, n_sampled=2
    )
    assert env.init_calls[0] == expected
    assert env.init_calls[1][0] == expected[1]
    assert expected[0] not in env.init_calls[1]
    # Sampled reuses the same frozen identities (fair det-vs-sampled).
    assert expected in env.init_calls
    assert ev["sampled"]["seeds"] == expected
    assert ev["deterministic"]["seeds"] == expected
    assert ev["suite"] == "holdout"


def test_all_curriculum_maps_are_real():
    keys = all_curriculum_maps()
    installed = installed_game_map_keys()
    assert keys <= installed
    assert {"Halkidiki", "FourIslands", "World", "Hawaii", "Europe"} <= keys


def test_pick_from_set_is_deterministic():
    items = ["FourIslands", "Hawaii", "Caribbean"]
    a = pick_from_set(items, "seed-a")
    b = pick_from_set(items, "seed-a")
    c = pick_from_set(items, "seed-b")
    assert a == b
    assert a in items
    assert pick_from_set(("Halkidiki",), "x") == "Halkidiki"


def test_parse_game_map_keys_local():
    source = 'export enum GameMapType {\n  Halkidiki = "Halkidiki",\n  FourIslands = "Four Islands",\n}\n'
    assert parse_game_map_keys(source) == ["Halkidiki", "FourIslands"]


def test_ood_maps_do_not_overlap_train_set():
    for stage in STAGES:
        overlap = set(stage.ood_maps) & set(stage.train_map_set)
        assert not overlap, (stage.name, overlap)
    pen = next(s for s in STAGES if s.name == "nations-easy-peninsula")
    assert "Balkans" in pen.train_map_set
    assert "Britannia" in pen.ood_maps
    with pytest.raises(ValueError, match="OOD maps overlap"):
        Stage(
            name="bad-ood",
            map="Halkidiki",
            train_maps=["Balkans"],
            ood_maps=["Balkans"],
        )


def test_completed_episodes_does_not_unlock_maps():
    cur = Curriculum(start=2)
    burn = cur.stage.map_burn_in
    assert burn > 1
    cur.stage_episodes = burn - 1
    assert cur.make_env_config("still-canonical")["map"] == "FourIslands"
    still_locked = {
        cur.make_env_config(mix_seed("m", i), completed_episodes=burn)["map"]
        for i in range(48)
    }
    assert still_locked == {"FourIslands"}


def test_resolve_holdout_episodes_raises_legacy_eight():
    assert resolve_holdout_episodes(8, DEFAULT_HOLDOUT_EPISODES) == 32
    assert resolve_holdout_episodes(32, DEFAULT_HOLDOUT_EPISODES) == 32
    assert resolve_holdout_episodes(40, DEFAULT_HOLDOUT_EPISODES) == 40
    with pytest.raises(ValueError, match="positive"):
        resolve_holdout_episodes(0, 32)


def test_parse_stage_arg_name_and_current_index():
    assert parse_stage_arg("nations-easy-small") == next(
        i for i, s in enumerate(STAGES) if s.name == "nations-easy-small"
    )
    assert parse_stage_arg("4") == 4
    assert STAGES[4].name == "nations-easy-intro"
    with pytest.raises(argparse.ArgumentTypeError, match="unknown stage"):
        parse_stage_arg("does-not-exist")


def test_v2_checkpoint_roundtrip_keeps_optimizer(tmp_path):
    from agent.ppo import PPOConfig, PPOTrainer

    cfg = PPOConfig(device="cpu", rollout_len=2, num_minibatches=1)
    trainer = PPOTrainer(Policy(), cfg)
    dummy = next(trainer.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    trainer.opt.step()
    model_before = {k: v.detach().clone() for k, v in trainer.policy.state_dict().items()}
    opt_lr = trainer.opt.state_dict()["param_groups"][0]["lr"]
    cur = Curriculum()
    cur.load_state_dict(
        {
            "idx": 4,
            "stage_episodes": 77,
            "mastery_version": 2,
            "consecutive_eval_passes": 2,
        }
    )
    assert cur.stage.name == "nations-easy-small"
    stream = SeedStream("v2run1", 1, 8)
    stream.episode_index[0] = 3
    path = tmp_path / "v2-migrate.pt"
    trainer.save(
        str(path),
        extra={
            "curriculum": cur.state_dict(),
            "seed_stream": stream.state_dict(),
        },
    )
    trainer2 = PPOTrainer(Policy(), cfg)
    ckpt = trainer2.load(str(path))
    assert ckpt["schema_version"] == 2
    restored = Curriculum()
    restored.load_state_dict(ckpt["curriculum"])
    assert restored.stage.name == "nations-easy-small"
    assert restored.idx == cur.idx
    assert restored.stage_episodes == 77
    assert not restored.last_eval_passed
    loaded_stream = SeedStream("other", 9, 8)
    loaded_stream.load_state_dict(ckpt["seed_stream"])
    assert loaded_stream.env_seed(0) == stream.env_seed(0)
    for key, tensor in model_before.items():
        assert torch.equal(trainer2.policy.state_dict()[key], tensor)
    assert trainer2.opt.state_dict()["param_groups"][0]["lr"] == opt_lr


def test_note_resets_advances_persisted_stream():
    env = ShardedVecEnv.__new__(ShardedVecEnv)
    stream = SeedStream("run", 1, 1)
    env._configs = [{"seed": stream.env_seed(0)}]
    env._on_episode_reset = lambda i, cfg: stream.note_done(i)
    nxt = [{"seed": stream.next_env_seed(0), "map": "Halkidiki"}]
    env._note_resets(np.array([True]), nxt)
    assert stream.episode_index[0] == 1
    assert env._configs[0]["seed"] == nxt[0]["seed"]
    env._note_resets(np.array([False]), nxt)
    assert stream.episode_index[0] == 1


def test_step_header_includes_next_configs():
    payload = _encode_frame(
        {"cmd": "step", "nextConfigs": [{"seed": "abc", "map": "Halkidiki"}]},
        {"actions": np.zeros((1, 5), np.int32)},
    )
    assert b"nextConfigs" in payload
    assert b"abc" in payload
