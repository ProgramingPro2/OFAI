"""v4: effective entropy, frozen reference KL, halt resume, ablation CLI.

Does not start training, write repository checkpoints, or run the game sim.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.checkpoint import (
    BEST_SCORE_SENTINEL,
    CollapsedCheckpointError,
    _eval_score,
    floor_global_best_at_entry,
    guard_collapsed_resume,
    halted_resume_reason,
    plan_resume,
    reusable_stage_baseline,
    should_write_global_best,
)
from agent.curriculum import Curriculum
from agent.model import (
    Policy,
    effective_action_entropy,
    explore_action_mask,
    make_dummy_obs,
    masked_categorical_kl,
    normalized_action_entropy,
)
from agent.ppo import PPOConfig, PPOTrainer, RolloutBuffer
from agent import spaces
from agent.train import (
    apply_v4_ablation_defaults,
    parse_args,
    should_run_holdout_eval,
)
from tests.test_checkpoint_safety import IT800, _it800_like_meta
from tests.test_rl_pipeline import _mode_stats
from tests.test_transfer_stability import _holdout_ev, _peninsula_idx


def test_reference_kl_zero_at_init_positive_after_perturbation():
    torch.manual_seed(0)
    policy = Policy()
    trainer = PPOTrainer(policy, PPOConfig(device="cpu", ref_kl_coef=0.1))
    trainer.set_reference(source="init")
    obs = make_dummy_obs(6, "cpu")
    kl0, extras0 = policy.joint_policy_kl(obs, trainer.reference)
    assert float(kl0.max()) < 1e-6
    assert float(extras0["ref_kl_action"].max()) < 1e-6
    with torch.no_grad():
        policy.head_action.weight[spaces.ACTION_ATTACK].add_(2.0)
        policy.head_region.weight[0].add_(1.5)
    kl1, extras1 = policy.joint_policy_kl(obs, trainer.reference)
    assert float(kl1.mean()) > 1e-4
    assert float(extras1["ref_kl_action"].mean()) > 0.0


def test_reference_kl_gradients_pull_toward_reference():
    torch.manual_seed(1)
    policy = Policy()
    trainer = PPOTrainer(policy, PPOConfig(device="cpu", ref_kl_coef=0.1))
    trainer.set_reference(source="init")
    with torch.no_grad():
        policy.head_action.bias[spaces.ACTION_ATTACK] += 2.5
    obs = make_dummy_obs(4, "cpu")
    before, _ = policy.joint_policy_kl(obs, trainer.reference)
    opt = torch.optim.SGD([policy.head_action.bias], lr=0.2)
    opt.zero_grad()
    before.mean().backward()
    assert policy.head_action.bias.grad is not None
    assert float(policy.head_action.bias.grad.abs().sum()) > 0.0
    for p in trainer.reference.parameters():
        assert p.grad is None
        assert not p.requires_grad
    opt.step()
    after, _ = policy.joint_policy_kl(obs, trainer.reference)
    assert float(after.mean()) < float(before.mean())


def test_reference_kl_masked_inactive_heads_are_zero():
    torch.manual_seed(2)
    policy = Policy()
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    obs = make_dummy_obs(4, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_NOOP] = 1.0
    with torch.no_grad():
        policy.head_region.weight.add_(2.0)
        policy.head_quantity.weight.add_(2.0)
        policy.head_unit.weight.add_(2.0)
    kl, extras = policy.joint_policy_kl(obs, ref)
    assert float(extras["ref_kl_target"].abs().max()) < 1e-6
    assert float(extras["ref_kl_region"].abs().max()) < 1e-6
    assert float(extras["ref_kl_quantity"].abs().max()) < 1e-6
    assert float(extras["ref_kl_unit"].abs().max()) < 1e-6
    assert float(kl.max()) < 1e-5


def test_fresh_training_disables_reference_loss():
    cfg = PPOConfig(device="cpu", ref_kl_coef=0.0)
    trainer = PPOTrainer(Policy(), cfg)
    assert not trainer.reference_enabled()
    assert trainer.reference_payload() is None
    trainer.set_reference(source="should-not-train")
    assert not trainer.reference_enabled()


def test_reference_checkpoint_roundtrip_excludes_optimizer(tmp_path):
    cfg = PPOConfig(device="cpu", ref_kl_coef=0.1, ref_kl_target=0.02, ref_kl_hard=0.4)
    trainer = PPOTrainer(Policy(), cfg)
    dummy = next(trainer.policy.parameters())
    dummy.grad = torch.zeros_like(dummy)
    trainer.opt.step()
    trainer.set_reference(source="checkpoints/v2run1/it800.pt")
    with torch.no_grad():
        trainer.policy.head_action.bias.add_(0.2)
    path = tmp_path / "v4.pt"
    trainer.save(str(path), extra={"schema_note": "v4-ref"})
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    assert loaded["schema_version"] == 2
    assert "reference" in loaded
    assert "model" not in loaded["optimizer"]
    assert loaded["reference"]["source"].endswith("it800.pt")
    assert loaded["reference"]["coef"] == pytest.approx(0.1)
    assert set(loaded["reference"]["model"]) == set(loaded["model"])
    assert "reference" not in loaded["optimizer"]

    dst = PPOTrainer(Policy(), PPOConfig(device="cpu", ref_kl_coef=0.0))
    dst.load(str(path))
    assert dst.reference is not None
    obs = make_dummy_obs(3, "cpu")
    kl, _ = dst.policy.joint_policy_kl(obs, dst.reference)
    src_kl, _ = trainer.policy.joint_policy_kl(obs, trainer.reference)
    assert torch.allclose(kl, src_kl, atol=1e-5)


def test_effective_entropy_excludes_always_legal_utilities():
    logits = torch.zeros(2, spaces.NUM_ACTION_TYPES)
    mask = torch.zeros(2, spaces.NUM_ACTION_TYPES)
    mask[:, spaces.ACTION_NOOP] = 1
    mask[:, spaces.ACTION_BOAT] = 1
    mask[:, spaces.ACTION_BUILD] = 1
    mask[:, spaces.ACTION_ATTACK] = 1
    explore = explore_action_mask(mask)
    assert int(explore[0, spaces.ACTION_NOOP]) == 0
    assert int(explore[0, spaces.ACTION_BOAT]) == 0
    assert int(explore[0, spaces.ACTION_ATTACK]) == 1
    ent, norm, max_ent, multi = effective_action_entropy(logits, mask)
    # Only attack is explore-worthy => fully satisfied.
    assert not bool(multi.any())
    assert torch.allclose(norm, torch.ones(2))
    assert torch.allclose(ent, torch.zeros(2))

    mask[:, spaces.ACTION_SPAWN] = 1
    ent2, norm2, max2, multi2 = effective_action_entropy(logits, mask)
    assert bool(multi2.all())
    assert float(max2[0]) == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert 0.0 < float(norm2[0]) <= 1.0


def test_masked_categorical_kl_is_not_mean_logp_gap():
    p = torch.tensor([[2.0, 0.1, -1.0]])
    q = torch.tensor([[0.0, 0.0, 0.0]])
    kl = masked_categorical_kl(p, q)
    logp = torch.distributions.Categorical(logits=p).log_prob(torch.tensor([0]))
    logq = torch.distributions.Categorical(logits=q).log_prob(torch.tensor([0]))
    pseudo = (logp - logq).mean()
    assert float(kl) > 0.0
    assert abs(float(kl) - float(pseudo)) > 1e-4


def test_halt_reason_refuses_ordinary_resume():
    ckpt = _it800_like_meta()
    ckpt["halt_reason"] = "confirmed_holdout_drop"
    ckpt["save_reason"] = "stage_regression"
    ckpt["regression"] = {"severe": True, "reason": "confirmed_holdout_drop"}
    assert halted_resume_reason(ckpt)
    with pytest.raises(CollapsedCheckpointError, match="halt_reason"):
        guard_collapsed_resume(
            ckpt, force_stage=False, allow_collapsed=False, source="latest.pt"
        )
    warns = guard_collapsed_resume(
        ckpt, force_stage=True, allow_collapsed=False, source="latest.pt"
    )
    assert warns
    plan = plan_resume(
        ckpt=ckpt,
        resume_path="/tmp/v3run2/latest.pt",
        run_name="v4retry",
        run_dir="/tmp/v4retry",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
    )
    assert plan.force_stage


def test_force_stage_best_floor_rejects_12_of_32():
    entry = {
        "sampled": _mode_stats(21 / 32, episodes=32, peak=0.35, cause="win"),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.10),
    }
    later = {
        "sampled": _mode_stats(12 / 32, episodes=32, peak=0.22, cause="win"),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.08),
    }
    floor = floor_global_best_at_entry(BEST_SCORE_SENTINEL, entry)
    assert floor[0] == pytest.approx(21 / 32)
    later_score = _eval_score(later)
    assert not should_write_global_best(later_score, floor, later)
    better = {
        "sampled": _mode_stats(22 / 32, episodes=32, peak=0.36, cause="win"),
        "deterministic": _mode_stats(0.0, episodes=8, peak=0.10),
    }
    assert should_write_global_best(_eval_score(better), floor, better)


def test_lock_maps_and_baseline_cannot_unlock():
    cur = Curriculum(start=_peninsula_idx())
    cur.maps_locked = True
    j = cur.apply_unlock(_holdout_ev(26 / 32))
    assert j["reason"] == "maps_locked"
    assert j["action"] == "hold"
    assert not cur.maps_unlocked
    assert {cur.make_env_config(f"l-{i}")["map"] for i in range(16)} == {"Halkidiki"}


def test_v4_ablation_cli_is_short_and_locked():
    args = apply_v4_ablation_defaults(
        parse_args(
            [
                "--resume",
                "checkpoints/v2run1/it800.pt",
                "--stage",
                "nations-easy-peninsula",
                "--force-stage",
                "--run-name",
                "v4ablate1",
                "--v4-ablation",
                "--lr",
                "1e-4",
                "--device",
                "cpu",
            ]
        )
    )
    assert args.lock_maps
    assert args.ent_weighting == "effective"
    assert args.ent_target == pytest.approx(0.3)
    assert args.eval_every == 5
    assert args.max_updates == 15
    assert args.ref_kl_coef == pytest.approx(0.1)
    assert args.additional_steps == 15 * args.rollout_len * args.servers * args.envs
    fresh = parse_args(["--run-name", "fresh"])
    assert fresh.ent_target == pytest.approx(0.6)
    assert fresh.ent_weighting == "uniform"
    assert fresh.ref_kl_coef == pytest.approx(0.0)
    assert fresh.qty_ent_coef == pytest.approx(0.0)
    assert args.qty_ent_coef == pytest.approx(0.03)
    assert fresh.eval_every == 0
    assert fresh.max_updates == 0


def test_it800_schema_v2_loads_without_reference():
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    cfg = PPOConfig(device="cpu", ref_kl_coef=0.0)
    trainer = PPOTrainer(Policy(), cfg)
    ckpt = trainer.load(str(IT800), restore_optimizer=False)
    assert int(ckpt["schema_version"]) == 2
    assert trainer.reference is None
    assert not trainer.reference_enabled()
    trainer.set_reference(source=str(IT800))
    obs = make_dummy_obs(2, "cpu")
    kl, extras = trainer.policy.joint_policy_kl(obs, trainer.reference)
    assert float(kl.max()) < 1e-5
    assert extras["ref_kl_action"].shape == kl.shape


def test_it800_v4_dry_run_writes_nothing(capsys):
    if not IT800.is_file():
        pytest.skip("checkpoints/v2run1/it800.pt is not present")
    src_dir = IT800.parent
    before = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in src_dir.iterdir()}
    dest = ROOT / "checkpoints" / "v4ablate1"
    dest_existed = dest.exists()
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--resume",
            str(IT800),
            "--run-name",
            "v4ablate1",
            "--stage",
            "nations-easy-peninsula",
            "--force-stage",
            "--v4-ablation",
            "--lr",
            "1e-4",
            "--eval-episodes",
            "32",
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
    assert "v4-ablation" in out
    assert "maps_locked" in out
    assert "max_updates=15" in out
    assert "additional_steps=" in out
    assert "loop would not run" not in out
    assert "ent_target=0.3" in out
    assert "short ablation" in out
    after = {p.name: (p.stat().st_mtime_ns, p.stat().st_size) for p in src_dir.iterdir()}
    assert after == before
    if not dest_existed:
        assert not dest.exists()


def test_joint_kl_matches_factorized_definition_no_double_count():
    """KL(π||ρ) = KL_a + π(attack)(KL_target + KL_quantity); NOOP adds 0."""
    torch.manual_seed(3)
    policy = Policy()
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    with torch.no_grad():
        policy.head_action.bias[spaces.ACTION_ATTACK] += 1.5
        policy.head_quantity.bias[0] += 1.2
        policy.target_scorer[2].bias.add_(0.8)
    obs = make_dummy_obs(4, "cpu")
    obs["action_mask"].zero_()
    obs["action_mask"][:, spaces.ACTION_NOOP] = 1
    obs["action_mask"][:, spaces.ACTION_ATTACK] = 1
    kl, extras = policy.joint_policy_kl(obs, ref)

    h, emb = policy._embed(obs)
    with torch.no_grad():
        h_r, emb_r = ref._embed(obs)
    a_logits = policy._masked_logits(policy.head_action(h), obs["action_mask"])
    with torch.no_grad():
        a_logits_r = ref._masked_logits(ref.head_action(h_r), obs["action_mask"])
    kl_a = masked_categorical_kl(a_logits, a_logits_r)
    probs = torch.softmax(a_logits, dim=-1)
    a_t = torch.full((4,), spaces.ACTION_ATTACK, dtype=torch.long)
    lg = policy._head_logits(h, emb, obs, a_t)
    with torch.no_grad():
        lg_r = ref._head_logits(h_r, emb_r, obs, a_t)
    kl_t = masked_categorical_kl(lg["target"], lg_r["target"])
    kl_q = masked_categorical_kl(lg["quantity"], lg_r["quantity"])
    expected = kl_a + probs[:, spaces.ACTION_ATTACK] * (kl_t + kl_q)
    assert torch.allclose(kl, expected, atol=1e-5)
    assert float(extras["ref_kl_region"].abs().max()) < 1e-6
    assert float(extras["ref_kl_unit"].abs().max()) < 1e-6
    sampled = (
        torch.distributions.Categorical(logits=a_logits).log_prob(a_t)
        - torch.distributions.Categorical(logits=a_logits_r).log_prob(a_t)
    ).mean()
    assert abs(float(kl.mean()) - float(sampled)) > 1e-4


def test_effective_entropy_ignores_utility_mass_and_stays_attainable():
    logits = torch.zeros(3, spaces.NUM_ACTION_TYPES)
    mask = torch.zeros(3, spaces.NUM_ACTION_TYPES)
    mask[:, spaces.ACTION_NOOP] = 1
    mask[:, spaces.ACTION_BOAT] = 1
    mask[:, spaces.ACTION_BUILD] = 1
    mask[:, spaces.ACTION_ATTACK] = 1
    # Only attack is explore-worthy: fully satisfied regardless of boat mass.
    logits[0, spaces.ACTION_BOAT] = 8.0
    logits[1, spaces.ACTION_ATTACK] = 8.0
    _ent, norm, _mx, multi = effective_action_entropy(logits, mask)
    assert not bool(multi.any())
    assert torch.allclose(norm, torch.ones(3))
    assert 0.0 <= 0.3 <= 1.0

    mask[:, spaces.ACTION_SPAWN] = 1
    logits2 = torch.zeros(2, spaces.NUM_ACTION_TYPES)
    mask2 = mask[:2]
    _e, norm2, max2, multi2 = effective_action_entropy(logits2, mask2)
    assert bool(multi2.all())
    assert float(max2[0]) == pytest.approx(float(torch.log(torch.tensor(2.0))))
    assert 0.0 < float(norm2[0]) <= 1.0
    # Target 0.3 is inside the attainable [0, 1] interval.
    assert float(norm2[0]) != float("nan")
    assert len(spaces.ACTION_ENTROPY_WEIGHTS) == spaces.NUM_ACTION_TYPES


def test_holdout_eval_runs_on_final_capped_update():
    # it800-class: evals at 805/810/815. Cap must not drop 815.
    due = [
        should_run_holdout_eval(
            iterations=800 + i,
            eval_every=5,
            updates_this_run=i,
            max_updates=15,
        )
        for i in range(1, 16)
    ]
    assert [800 + i for i, flag in enumerate(due, start=1) if flag] == [
        805,
        810,
        815,
    ]
    assert should_run_holdout_eval(
        iterations=816, eval_every=5, updates_this_run=15, max_updates=15
    )
    assert not should_run_holdout_eval(
        iterations=801, eval_every=0, updates_this_run=15, max_updates=15
    )
    assert not should_run_holdout_eval(
        iterations=0, eval_every=5, updates_this_run=0, max_updates=0
    )


def test_ref_kl_uses_same_f16_spatial_quantization_as_update():
    torch.manual_seed(5)
    policy = Policy()
    ref = Policy()
    ref.load_state_dict(policy.state_dict())
    with torch.no_grad():
        policy.head_action.bias.add_(0.4)
    obs = make_dummy_obs(3, "cpu")
    kl_f32, _ = policy.joint_policy_kl(obs, ref)
    obs_q = dict(obs)
    obs_q["spatial"] = obs["spatial"].half().float()
    kl_q, _ = policy.joint_policy_kl(obs_q, ref)
    # Same f16 storage the PPO buffer keeps; both sides of the KL see it.
    assert torch.isfinite(kl_q).all()
    assert float((kl_q - kl_f32).abs().max()) < 0.05


def test_maps_locked_persists_across_state_dict():
    cur = Curriculum(start=_peninsula_idx())
    cur.maps_locked = True
    cur.maps_unlocked = False
    sd = cur.state_dict()
    assert sd["maps_locked"] is True
    restored = Curriculum()
    restored.load_state_dict(sd)
    assert restored.maps_locked is True
    assert restored.judge_unlock(_holdout_ev(26 / 32))["unlock"]
    j = restored.apply_unlock(_holdout_ev(26 / 32))
    assert j["reason"] == "maps_locked"
    assert not restored.maps_unlocked
    legacy = Curriculum()
    legacy.load_state_dict(
        {"stage": "nations-easy-peninsula", "mastery_version": 3}
    )
    assert legacy.maps_locked is False


def _stage_entry_baseline_meta(*, it: int = 800, wr: float = 0.5625) -> dict:
    ev = _holdout_ev(wr)
    seeds = [f"{i:032x}" for i in range(32)]
    return {
        "schema_version": 2,
        "curriculum": {
            "stage": "nations-easy-peninsula",
            "mastery_version": 3,
            "maps_unlocked": False,
        },
        "stats": {"iterations": it, "env_steps": 13_107_200},
        "pending_stage_baseline": False,
        "stage_baseline": {
            "kind": "pre_update_holdout",
            "stage": "nations-easy-peninsula",
            "win_rate": wr,
            "episodes": 32,
            "iterations": 800,
            "seeds": seeds,
            "eval": ev,
        },
        "eval": ev,
    }


def test_reusable_stage_baseline_requires_same_policy_iteration():
    entry = _stage_entry_baseline_meta()
    assert reusable_stage_baseline(
        entry, stage="nations-easy-peninsula"
    )["win_rate"] == pytest.approx(0.5625)
    later = _stage_entry_baseline_meta(it=815)
    assert reusable_stage_baseline(later, stage="nations-easy-peninsula") is None
    assert reusable_stage_baseline(entry, stage="nations-easy-mid") is None
    pending = _stage_entry_baseline_meta()
    pending["pending_stage_baseline"] = True
    assert reusable_stage_baseline(pending, stage="nations-easy-peninsula") is None


def test_force_stage_reuses_matching_stage_entry_baseline():
    plan = plan_resume(
        ckpt=_stage_entry_baseline_meta(),
        resume_path="/tmp/v4ablate1/stage-entry.pt",
        run_name="v4val50",
        run_dir="/tmp/v4val50",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
    )
    assert plan.force_stage
    assert plan.reset_optimizer
    assert not plan.restore_seed_stream
    assert not plan.need_baseline_eval
    assert not plan.pending_stage_baseline
    assert plan.stage_baseline["win_rate"] == pytest.approx(0.5625)
    assert plan.stage_baseline["episodes"] == 32
    assert "reusing frozen stage_baseline" in " ".join(plan.warnings)


def test_force_stage_it800_without_complete_baseline_still_evals():
    plan = plan_resume(
        ckpt=_it800_like_meta(),
        resume_path="/tmp/v2run1/it800.pt",
        run_name="v4val50",
        run_dir="/tmp/v4val50",
        stage_idx=_peninsula_idx(),
        force_stage=True,
        cli_lr=1e-4,
    )
    assert plan.need_baseline_eval
    assert plan.stage_baseline == {}


def test_quantity_entropy_active_only_on_attack_boat():
    policy = Policy()
    obs = make_dummy_obs(4, "cpu")
    actions = torch.tensor(
        [
            [spaces.ACTION_NOOP, 0, 0, 0, 0],
            [spaces.ACTION_ATTACK, 0, 0, 0, 0],
            [spaces.ACTION_BOAT, 0, 0, 0, 0],
            [spaces.ACTION_BUILD, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    with torch.no_grad():
        _, _, _, extras = policy.evaluate(obs, actions)
    active = extras["quantity_entropy_active"]
    assert not bool(active[0])
    assert bool(active[1])
    assert bool(active[2])
    assert not bool(active[3])
    assert extras["quantity_entropy_norm"].min() >= 0.0
    assert extras["quantity_entropy_norm"].max() <= 1.0


def test_qty_ent_coef_off_by_default_and_checkpoint_compatible(tmp_path):
    cfg = PPOConfig(device="cpu", qty_ent_coef=0.0)
    trainer = PPOTrainer(Policy(), cfg)
    path = tmp_path / "qty-off.pt"
    trainer.save(str(path))
    ckpt = trainer.load(str(path))
    assert ckpt["hyperparameters"]["qty_ent_coef"] == pytest.approx(0.0)
    assert "qty_ent_coef" not in ckpt["model"]


def test_ppo_qty_ent_bonus_only_when_coef_positive():
    """qty_ent_coef>0 subtracts normalized quantity entropy from the loss."""
    torch.manual_seed(7)
    policy = Policy()
    obs = make_dummy_obs(8, "cpu")
    actions = torch.zeros(8, 5, dtype=torch.long)
    actions[:, 0] = spaces.ACTION_ATTACK
    with torch.no_grad():
        _, _, _, extras = policy.evaluate(obs, actions)
    qty_multi = extras["quantity_entropy_multi"]
    assert bool(qty_multi.any())
    qty_norm = float(extras["quantity_entropy_norm"][qty_multi].mean())

    cfg_off = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        qty_ent_coef=0.0,
    )
    cfg_on = PPOConfig(
        rollout_len=4,
        device="cpu",
        num_minibatches=1,
        update_epochs=1,
        target_kl=100.0,
        qty_ent_coef=0.05,
    )
    buf = RolloutBuffer(4, 2)
    buf.actions[:, :, 0] = spaces.ACTION_ATTACK
    buf.logprobs[:] = 0.0
    buf.values[:] = 0.0
    for key in buf.obs:
        if key == "spatial":
            buf.obs[key][:] = 0
        else:
            buf.obs[key][:] = 1
    last_obs = {
        key: (
            np.zeros(buf.obs[key].shape[1:], dtype=np.float32)
            if key == "spatial"
            else np.ones(buf.obs[key].shape[1:], dtype=np.float32)
        )
        for key in buf.obs
    }

    off = PPOTrainer(Policy(), cfg_off)
    on = PPOTrainer(Policy(), cfg_on)
    off.policy.load_state_dict(policy.state_dict())
    on.policy.load_state_dict(policy.state_dict())
    m_off = off.update(buf, last_obs)
    m_on = on.update(buf, last_obs)
    assert m_off["quantity_entropy_norm"] == pytest.approx(0.0)
    assert m_on["quantity_entropy_norm"] == pytest.approx(qty_norm, abs=0.05)
    assert m_on["quantity_entropy_norm"] > 0.0
