"""PPO training entry point.

Spawns (or connects to) env servers, then alternates rollout collection and
PPO updates. Curriculum stages advance on min_episodes plus one frozen
holdout sampled-true-win mastery decision (Wilson-guarded). Deterministic
argmax is a robustness diagnostic and never gates.

Example:
  .venv/bin/python -m agent.train --servers 4 --envs 8 --rollout-len 128 \
      --run-name dev --total-steps 1000000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import (
    BEST_SCORE_SENTINEL,
    CollapsedCheckpointError,
    STAGE_ENTRY_FILENAME,
    StageRegressionHalt,
    TRANSFER_EVAL_EVERY,
    TRANSFER_LR,
    _eval_score,
    baseline_event_record,
    build_stage_baseline,
    choose_restore_snapshot,
    finalize_event_record,
    eval_is_passive_collapse,
    eval_is_zero_win,
    format_resume_plan,
    is_sentinel_score,
    judge_holdout_regression,
    map_unlock_event_record,
    on_stage_advance_scores,
    plan_resume,
    regression_event_record,
    restore_runtime_from_snapshot,
    should_rewrite_halt_checkpoint,
    should_write_global_best,
    write_touches_foreign_source,
    should_write_stage_best,
    stage_advance_event_record,
    stage_best_filename,
    training_budget_warning,
)
from .curriculum import (
    DEFAULT_DET_EVAL_EPISODES,
    DEFAULT_HOLDOUT_EPISODES,
    DEFAULT_OOD_EVAL_EPISODES,
    STAGES,
    Curriculum,
)
from .env_client import RolloutInterrupted, ShardedVecEnv
from .model import Policy, obs_to_tensors
from . import spaces
from .ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    apply_optimizer_lr,
    classify_outcome,
    optimizer_is_fresh,
)
from .seeds import SeedStream, holdout_seed, ood_seed

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def parse_stage_arg(value: str) -> int:
    """Accept a current-table index or a stage name."""
    names = [stage.name for stage in STAGES]
    if value in names:
        return names.index(value)
    try:
        idx = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown stage {value!r}; expected an index or one of {names}"
        ) from exc
    if idx < 0 or idx >= len(STAGES):
        raise argparse.ArgumentTypeError(
            f"stage index {idx} is outside 0..{len(STAGES) - 1}"
        )
    return idx


def resolve_holdout_episodes(requested: int, minimum: int) -> int:
    """Raise a too-small `--eval-episodes` to the stage holdout minimum.

    Older commands still pass `--eval-episodes 8`. The mastery gate is
    26/32, so those runs must actually play 32 frozen holdout games.
    """
    req = int(requested)
    mn = int(minimum)
    if req < 1:
        raise ValueError("eval episodes must be positive")
    if mn < 1:
        raise ValueError("holdout minimum must be positive")
    return max(req, mn)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8, help="parallel games per server")
    p.add_argument("--servers", type=int, default=4, help="env-server processes")
    p.add_argument("--port", type=int, default=8765, help="base port (servers use port+i)")
    p.add_argument("--connect", action="store_true", help="connect to already-running servers")
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument(
        "--total-steps",
        type=int,
        default=10_000_000,
        help="absolute env decision-step target (from 0, or from checkpoint if resumed)",
    )
    p.add_argument(
        "--additional-steps",
        type=int,
        default=0,
        help="with --resume: train this many more steps beyond checkpoint env_steps "
        "(overrides --total-steps as an absolute ceiling)",
    )
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    p.add_argument(
        "--stage",
        type=parse_stage_arg,
        default=None,
        help="fresh-run stage name or current-table index (ignored on resume "
        "unless also passed with --force-stage). Numeric indices follow the "
        "current table; v2 checkpoints migrate idx through V2_STAGE_NAMES.",
    )
    p.add_argument("--force-stage", action="store_true", help="apply --stage even when resuming")
    p.add_argument(
        "--keep-optimizer",
        action="store_true",
        help="advanced: with --force-stage, keep source Adam moments/step "
        "(default force-stage is a fresh optimizer at --lr)",
    )
    p.add_argument(
        "--allow-collapsed-resume",
        action="store_true",
        help="override the refuse-to-resume guard when eval metadata "
        "shows a zero-win/passive collapse (still warns)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="load resume/config, print the plan, exit without envs or writes",
    )
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="PPO early-stop KL on active-head joint (default ~0.03)",
    )
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument(
        "--ent-target",
        type=float,
        default=0.6,
        help="mask-normalized action-type entropy target in [0, 1] "
        "(H / log(n_legal), averaged over samples with >1 legal actions)",
    )
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="iterations between frozen holdout evals (0=off)",
    )
    p.add_argument(
        "--eval-episodes",
        type=int,
        default=DEFAULT_HOLDOUT_EPISODES,
        help="sampled holdout episodes (frozen per-stage seeds; default 32; "
        "values below the stage minimum are raised to that minimum)",
    )
    p.add_argument(
        "--eval-det-episodes",
        type=int,
        default=DEFAULT_DET_EVAL_EPISODES,
        help="deterministic diagnostic episodes (does not gate; default 8)",
    )
    p.add_argument(
        "--eval-ood-episodes",
        type=int,
        default=DEFAULT_OOD_EVAL_EPISODES,
        help="optional OOD episodes logged only (0=skip; default 8)",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _byte_rng(t) -> torch.Tensor:
    """torch RNG APIs require a CPU uint8 tensor; checkpoints may load to CUDA."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.uint8)
    return t.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _set_rng_state(state: dict) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(_byte_rng(state["torch"]))
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_rng(s) for s in state["cuda"]])


# Bounded retries for shard death mid-eval. Healthy shards may already
# have stepped, so a partial attempt is never resumed or merged.
FIXED_SEED_EVAL_RETRIES = 3


def _eval_configs(
    curriculum: Curriculum,
    n_episodes: int,
    offset: int,
    *,
    suite: str,
    seed_prefix: str | None,
    mode: str = "sampled",
) -> list[dict]:
    """Frozen holdout/OOD configs, or an explicit prefix for tests.

    Frozen identities are shared across det/sampled. A test prefix keeps
    the historical `{prefix}-{mode}-{i}` form so retries stay comparable.
    """
    cfgs: list[dict] = []
    for i in range(n_episodes):
        idx = offset + i
        if seed_prefix:
            seed = f"{seed_prefix}-{mode}-{idx}"
        elif suite == "ood":
            seed = ood_seed(curriculum.stage.name, idx)
        else:
            seed = holdout_seed(curriculum.stage.name, idx)
        cfg = curriculum.make_env_config(seed, objective=True, suite=suite)
        cfg["shaping"] = 0.0
        cfgs.append(cfg)
    return cfgs


def run_fixed_seed_eval(
    policy: Policy,
    env: ShardedVecEnv,
    curriculum: Curriculum,
    device: torch.device,
    n_episodes: int,
    seed_prefix: str | None = None,
    *,
    n_det: int | None = None,
    n_sampled: int | None = None,
    suite: str = "holdout",
) -> dict:
    """Inline eval matching the active stage. Runs deterministic then sampled.

    Default seeds are frozen per (stage, index) and do not rotate with
    iteration. `seed_prefix` overrides that for tests. Milestone
    termination is omitted so `win_rate` is the core 80% FFA outcome.

    `RolloutInterrupted` is recoverable: discard the entire attempt (do not
    mix finished episodes or mid-step healthy shards), re-init every eval
    env with the same fixed seeds, and retry from the beginning.

    Episodes are spread across shards via ``init_active`` / ``step_active``
    so a 32-game holdout uses idle servers instead of packing into the
    first few processes. Idle lanes are not created or stepped.
    """
    sampled_n = n_sampled if n_sampled is not None else n_episodes
    det_n = n_det if n_det is not None else n_episodes
    last_exc: RolloutInterrupted | None = None
    for attempt in range(1, FIXED_SEED_EVAL_RETRIES + 1):
        try:
            return _fixed_seed_eval_attempt(
                policy,
                env,
                curriculum,
                device,
                sampled_n=sampled_n,
                det_n=det_n,
                seed_prefix=seed_prefix,
                suite=suite,
            )
        except RolloutInterrupted as exc:
            last_exc = exc
            print(
                f"[eval] {exc}; discarding partial eval "
                f"(attempt {attempt}/{FIXED_SEED_EVAL_RETRIES}) "
                f"and retrying from fixed seeds"
            )
    raise RuntimeError(
        f"fixed-seed eval failed after {FIXED_SEED_EVAL_RETRIES} "
        f"shard-restart attempts ({last_exc})"
    ) from last_exc


def _fixed_seed_eval_attempt(
    policy: Policy,
    env: ShardedVecEnv,
    curriculum: Curriculum,
    device: torch.device,
    *,
    sampled_n: int,
    det_n: int,
    seed_prefix: str | None,
    suite: str,
) -> dict:
    """One clean eval pass. Accumulators are local; callers retry wholesale."""
    stage = curriculum.stage
    k = env.k
    results: dict = {"deterministic": {}, "sampled": {}}
    stage_frac = float(stage.win_tiles_frac)

    for mode, n_episodes in (("deterministic", det_n), ("sampled", sampled_n)):
        if n_episodes <= 0:
            results[mode] = {
                "episodes": 0,
                "win_rate": 0.0,
                "stage_success_rate": 0.0,
                "stage": stage.name,
                "suite": suite,
                "seeds": [],
                "maps": [],
            }
            continue
        wins = 0
        stage_hits = 0
        deaths = 0
        kills = 0
        rets = []
        peaks = []
        finals = []
        ticks = []
        causes: dict[str, int] = {}
        accepted = 0
        total_a = 0
        hist: dict[str, int] = {}
        used_seeds: list[str] = []
        used_maps: list[str] = []
        remaining = n_episodes
        while remaining > 0:
            batch_n = min(k, remaining)
            offset = n_episodes - remaining
            cfgs = _eval_configs(
                curriculum,
                batch_n,
                offset,
                suite=suite,
                seed_prefix=seed_prefix,
                mode=mode,
            )
            used_seeds.extend(str(c["seed"]) for c in cfgs)
            used_maps.extend(str(c["map"]) for c in cfgs)
            pad = [
                {
                    **curriculum.make_env_config(
                        f"eval-pad-{suite}-{i}", objective=True, suite=suite
                    ),
                    "shaping": 0.0,
                }
                for i in range(env.k - batch_n)
            ]
            use_active = hasattr(env, "init_active") and hasattr(
                env, "step_active"
            )
            use_leading = hasattr(env, "init_leading") and hasattr(
                env, "step_leading"
            )
            if use_active:
                obs = env.init_active(cfgs)
            elif use_leading:
                obs = env.init_leading(cfgs)
            else:
                obs = env.init(cfgs + pad)
            work_k = int(obs["spatial"].shape[0])
            ep_ret = np.zeros(work_k, np.float32)
            active = np.zeros(work_k, dtype=bool)
            active[:batch_n] = True
            finished = 0
            while active[:batch_n].any():
                with torch.no_grad():
                    obs_t = obs_to_tensors(obs, device)
                    acts, _, _ = policy.act(obs_t, deterministic=(mode == "deterministic"))
                actions = acts.cpu().numpy().astype(np.int32)
                if use_active:
                    step = env.step_active(actions)
                elif use_leading:
                    step = env.step_leading(actions)
                else:
                    step = env.step(actions)
                ep_ret += step.rewards
                for i in range(batch_n):
                    if not active[i]:
                        continue
                    info = step.infos[i]
                    at = int(actions[i, 0])
                    name = spaces.ACTION_NAMES[at]
                    hist[name] = hist.get(name, 0) + 1
                    acc = info.get("actionAccepted", info.get("action_accepted", True))
                    total_a += 1
                    if acc:
                        accepted += 1
                    if step.dones[i]:
                        active[i] = False
                        finished += 1
                        cause, won, stage_ok = classify_outcome(info, stage_frac)
                        causes[cause] = causes.get(cause, 0) + 1
                        if won:
                            wins += 1
                        if stage_ok:
                            stage_hits += 1
                        if cause == "death":
                            deaths += 1
                        kills += int(info.get("kills", 0))
                        rets.append(float(ep_ret[i]))
                        peaks.append(float(info.get("peakTilesFrac", info.get("tilesFrac", 0))))
                        finals.append(float(info.get("tilesFrac", 0)))
                        ticks.append(int(info.get("tick", 0)))
                        ep_ret[i] = 0.0
                obs = step.obs
            remaining -= batch_n

        n = max(1, len(rets))
        results[mode] = {
            "episodes": n,
            "win_rate": wins / n,
            "stage_success_rate": stage_hits / n,
            "death_rate": deaths / n,
            "kills_per_ep": kills / n,
            "mean_return": float(np.mean(rets)) if rets else 0.0,
            "mean_peak_tiles": float(np.mean(peaks)) if peaks else 0.0,
            "mean_final_tiles": float(np.mean(finals)) if finals else 0.0,
            "mean_ticks": float(np.mean(ticks)) if ticks else 0.0,
            "terminal_causes": causes,
            "action_hist": hist,
            "accept_rate": accepted / max(1, total_a),
            "stage": stage.name,
            "suite": suite,
            "seeds": used_seeds,
            "maps": used_maps,
        }
    results["suite"] = suite
    results["holdout_map"] = stage.holdout_map
    results["holdout_map_size"] = stage.holdout_map_size
    return results


def run_pre_update_baseline(
    *,
    policy: Policy,
    env: ShardedVecEnv,
    curriculum: Curriculum,
    trainer: PPOTrainer,
    device: torch.device,
    n_sampled: int,
    n_det: int,
    n_ood: int,
) -> tuple[dict, dict, dict | None, dict]:
    """Frozen holdout before the first PPO collect/update.

    Does not increment trainer iterations/env_steps, does not call
    note_eval / maybe_advance, and does not write a global-best score.
    """
    it0 = int(trainer.stats.iterations)
    steps0 = int(trainer.stats.env_steps)
    ev = run_fixed_seed_eval(
        policy,
        env,
        curriculum,
        device,
        n_sampled,
        n_det=n_det,
        n_sampled=n_sampled,
        suite="holdout",
    )
    ood_ev = None
    if n_ood > 0 and curriculum.stage.ood_maps:
        ood_ev = run_fixed_seed_eval(
            policy,
            env,
            curriculum,
            device,
            n_ood,
            n_det=0,
            n_sampled=n_ood,
            suite="ood",
        )
    if trainer.stats.iterations != it0 or trainer.stats.env_steps != steps0:
        raise RuntimeError(
            "pre-update baseline advanced trainer counters "
            f"(it {it0}->{trainer.stats.iterations}, "
            f"env_steps {steps0}->{trainer.stats.env_steps})"
        )
    baseline = build_stage_baseline(
        ev,
        iterations=it0,
        env_steps=steps0,
        stage=curriculum.stage.name,
        kind="pre_update_holdout",
    )
    unlock = curriculum.apply_unlock(ev)
    return baseline, ev, ood_ev, unlock


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    run_dir = CKPT_DIR / args.run_name
    log_path = LOG_DIR / f"{args.run_name}.jsonl"
    ep_log_path = LOG_DIR / f"{args.run_name}.episodes.jsonl"
    resume_path = args.resume
    log_f = None
    ep_log_f = None

    start_stage = args.stage if args.stage is not None else 0
    curriculum = Curriculum(start=start_stage)
    policy = Policy()
    cfg = PPOConfig(
        rollout_len=args.rollout_len,
        lr=args.lr,
        target_kl=args.target_kl,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        ent_target=args.ent_target,
        gamma=args.gamma,
        device=args.device if torch.cuda.is_available() else "cpu",
    )
    trainer = PPOTrainer(policy, cfg)
    global_best_score: tuple = BEST_SCORE_SENTINEL
    stage_best_score: tuple = BEST_SCORE_SENTINEL
    source_best_score: tuple = BEST_SCORE_SENTINEL
    stage_baseline: dict = {}
    last_eval: dict = {}
    last_mastery: dict = {}
    pending_stage_baseline = False
    halt_reason = ""
    halt_checkpoint_committed = False
    resume_plan = None
    ckpt: dict = {}

    if resume_path:
        restore_opt = not (args.force_stage and not args.keep_optimizer)
        ckpt = trainer.load(resume_path, restore_optimizer=restore_opt)
        print(f"resumed from {resume_path} (schema={ckpt.get('schema_version', 1)})")
        try:
            resume_plan = plan_resume(
                ckpt=ckpt,
                resume_path=resume_path,
                run_name=args.run_name,
                run_dir=run_dir,
                stage_idx=args.stage,
                force_stage=args.force_stage,
                allow_collapsed=args.allow_collapsed_resume,
                default_start=start_stage,
                keep_optimizer=args.keep_optimizer,
                cli_lr=args.lr,
                eval_every=args.eval_every,
            )
        except CollapsedCheckpointError:
            raise
        curriculum = resume_plan.curriculum
        global_best_score = resume_plan.global_best_score
        stage_best_score = resume_plan.stage_best_score
        source_best_score = resume_plan.source_best_score
        stage_baseline = resume_plan.stage_baseline
        last_eval = resume_plan.last_eval
        pending_stage_baseline = resume_plan.pending_stage_baseline
        if resume_plan.restore_rng and "rng" in ckpt:
            _set_rng_state(ckpt["rng"])
        if resume_plan.reset_windows:
            trainer.stats.reset_rolling_windows()
        if resume_plan.reset_optimizer:
            trainer.reset_optimizer(args.lr)
            print(
                f"[train] --force-stage: fresh Adam at lr={args.lr:g} "
                f"(fresh={optimizer_is_fresh(trainer.opt)})"
            )
        elif resume_plan.keep_optimizer:
            apply_optimizer_lr(trainer.opt, args.lr)
            trainer.cfg.lr = float(args.lr)
            print(
                f"[train] --keep-optimizer: inherited Adam moments; "
                f"param-group lr set to {args.lr:g}"
            )
        for warn in resume_plan.warnings:
            print(warn)
        if args.force_stage:
            print(
                f"[train] --force-stage {curriculum.stage.name}: "
                "stage-local curriculum/mastery/windows/seed-stream reset; "
                "global-best metadata retained; "
                f"writing to {run_dir}"
            )
        else:
            print(
                f"[train] restored stage={curriculum.stage.name} "
                f"idx={curriculum.idx} episodes={curriculum.stage_episodes}"
            )
        if args.additional_steps > 0:
            args.total_steps = trainer.stats.env_steps + args.additional_steps
            print(
                f"[train] --additional-steps={args.additional_steps} -> "
                f"total_steps={args.total_steps}"
            )
        budget = training_budget_warning(
            trainer.stats.env_steps, args.total_steps
        )
        if budget:
            print(budget)
    elif args.stage is not None:
        curriculum = Curriculum(start=args.stage)

    if args.dry_run:
        env_steps = trainer.stats.env_steps
        if resume_plan is None:
            resume_plan = plan_resume(
                ckpt={"stats": trainer.stats.state_dict()},
                resume_path=resume_path or "(fresh)",
                run_name=args.run_name,
                run_dir=run_dir,
                stage_idx=args.stage,
                force_stage=False,
                allow_collapsed=True,
                default_start=start_stage,
            )
            resume_plan.curriculum = curriculum
        print(
            format_resume_plan(
                resume_plan,
                env_steps=env_steps,
                total_steps=args.total_steps,
                eval_every=args.eval_every,
            )
        )
        if resume_plan.force_stage:
            print(
                f"[dry-run] transfer recommendation: --lr {TRANSFER_LR:g} "
                f"--eval-every {TRANSFER_EVAL_EVERY} "
                "(do not change global defaults for unrelated fresh runs)"
            )
            print(
                "[dry-run] CLI lr after optimizer reset = "
                f"{args.lr:g}; eval_every={args.eval_every}"
            )
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh run must not silently append to a prior run's logs.
    log_mode = "a" if resume_path else "w"
    log_f = open(log_path, log_mode, buffering=1)
    ep_log_f = open(ep_log_path, log_mode, buffering=1)
    if resume_plan is not None:
        log_f.write(json.dumps(resume_plan.event) + "\n")

    env = ShardedVecEnv(
        num_servers=args.servers,
        base_port=args.port,
        envs_per_server=args.envs,
        connect=args.connect,
    )
    total_envs = env.k
    seed_stream = SeedStream(args.run_name, args.seed, total_envs)
    if resume_path and resume_plan is not None and resume_plan.restore_seed_stream:
        raw_stream = ckpt.get("seed_stream")
        if isinstance(raw_stream, dict):
            seed_stream.load_state_dict(raw_stream)
    elif resume_path and (resume_plan is None or not resume_plan.restore_seed_stream):
        if resume_plan is not None and resume_plan.force_stage:
            print(
                "[train] --force-stage: new seed stream for "
                f"{args.run_name} (source stream not restored)"
            )
        else:
            print(
                "[train] no seed_stream in checkpoint; starting mixed counters "
                "at 0 (v2 it{N} replay strings are not reused)"
            )
    print(
        f"[train] batch={cfg.rollout_len * total_envs} "
        f"num_minibatches={cfg.num_minibatches} epochs={cfg.update_epochs} "
        f"target_kl={cfg.target_kl} stage={curriculum.stage.name} "
        f"holdout={curriculum.stage.holdout_map} "
        f"train_maps={curriculum.stage.train_map_set}"
    )

    def attach_train_hooks() -> None:
        def next_cfg(env_i: int) -> dict:
            # +1: this env is about to complete, so the next episode should
            # see the post-completion burn-in count.
            return curriculum.make_env_config(
                seed_stream.next_env_seed(env_i),
                completed_episodes=curriculum.stage_episodes + 1,
            )

        def on_reset(env_i: int, _cfg: dict) -> None:
            seed_stream.note_done(env_i)

        env.set_episode_hooks(next_cfg, on_reset)

    def reinit_envs() -> dict:
        attach_train_hooks()
        cfgs = [
            curriculum.make_env_config(seed_stream.env_seed(i))
            for i in range(total_envs)
        ]
        obs_ = env.init(cfgs)
        trainer._ensure_trackers(total_envs)
        trainer.reset_episode_trackers()
        return obs_

    def save_ckpt(path: Path, **extra) -> None:
        source = resume_plan.source_checkpoint if resume_plan is not None else ""
        if write_touches_foreign_source(path, run_dir, source):
            raise RuntimeError(
                f"refusing to write {path} into source checkpoint directory "
                f"{Path(source).resolve().parent}"
            )
        trainer.save(
            str(path),
            extra={
                "curriculum": curriculum.state_dict(),
                "stage_baseline": stage_baseline,
                "best_score": list(global_best_score),
                "global_best_score": list(global_best_score),
                "stage_best_score": list(stage_best_score),
                "stage_best_stage": curriculum.stage.name,
                "source_best_score": list(source_best_score),
                "source_checkpoint": (
                    resume_plan.source_checkpoint if resume_plan is not None else ""
                ),
                "last_eval": last_eval,
                "rng": _rng_state(),
                "seed_stream": seed_stream.state_dict(),
                "curriculum_suite": curriculum.suite_meta(),
                "pending_stage_baseline": pending_stage_baseline,
                "halt_reason": halt_reason,
                **extra,
            },
        )

    if not resume_path:
        save_ckpt(run_dir / "untrained.pt")

    obs = reinit_envs()
    buf = RolloutBuffer(cfg.rollout_len, total_envs)
    steps_per_iter = cfg.rollout_len * total_envs
    device = torch.device(cfg.device)

    def persist_stage_entry(ev: dict | None = None, **extra) -> None:
        save_ckpt(
            run_dir / STAGE_ENTRY_FILENAME,
            eval=ev or {},
            stage_entry=True,
            **extra,
        )

    def apply_map_unlock(ev: dict, *, it: int) -> dict:
        judgment = curriculum.apply_unlock(ev)
        if judgment["action"] != "hold":
            rec = map_unlock_event_record(
                it=it, stage=curriculum.stage.name, judgment=judgment
            )
            log_f.write(json.dumps(rec) + "\n")
            pbar.write(
                f"[curriculum] maps {judgment['action']} "
                f"stage={curriculum.stage.name} "
                f"samp_win={judgment['sampled_win_rate']} "
                f"wilson_lb={judgment['sampled_wilson_lower']} "
                f"maps_unlocked={judgment['maps_unlocked']}"
            )
        return judgment

    def restore_on_regression(judgment: dict) -> str:
        entry = run_dir / STAGE_ENTRY_FILENAME
        best_path = run_dir / stage_best_filename(curriculum.stage.name)
        which = choose_restore_snapshot(
            stage_entry_exists=entry.is_file(),
            stage_best_exists=best_path.is_file(),
            stage_best_score=stage_best_score,
            baseline_wr=float((stage_baseline or {}).get("win_rate") or 0.0),
        )
        src = best_path if which == "stage-best" else entry
        if which == "none" or not src.is_file():
            raise RuntimeError(
                "stage regression halt requested but no stage-entry/stage-best "
                "snapshot exists"
            )
        snap = trainer.load_weights_and_optimizer(str(src))
        restore_runtime_from_snapshot(
            snap,
            curriculum=curriculum,
            seed_stream=seed_stream,
            set_rng=_set_rng_state,
        )
        return str(src)

    def run_force_stage_baseline() -> None:
        nonlocal stage_baseline, last_eval, pending_stage_baseline, obs
        nonlocal stage_best_score
        holdout_n = resolve_holdout_episodes(
            args.eval_episodes, curriculum.stage.min_eval_episodes
        )
        it0 = trainer.stats.iterations
        steps0 = trainer.stats.env_steps
        pbar.write(
            f"[eval] pre-update baseline holdout at it={it0} "
            f"env_steps={steps0} stage={curriculum.stage.name} "
            f"n_samp={holdout_n} n_det={args.eval_det_episodes} "
            f"map={curriculum.stage.holdout_map} seeds=frozen "
            "(no collect/update, no stage advance, no global-best write)"
        )
        env.set_episode_hooks(None)
        try:
            baseline, ev, ood_ev, unlock = run_pre_update_baseline(
                policy=policy,
                env=env,
                curriculum=curriculum,
                trainer=trainer,
                device=device,
                n_sampled=holdout_n,
                n_det=args.eval_det_episodes,
                n_ood=args.eval_ood_episodes,
            )
        finally:
            obs = reinit_envs()
        if trainer.stats.iterations != it0 or trainer.stats.env_steps != steps0:
            raise RuntimeError("baseline eval advanced trainer counters")
        stage_baseline = baseline
        last_eval = ev
        pending_stage_baseline = False
        score = _eval_score(ev)
        if should_write_stage_best(score, stage_best_score, ev):
            stage_best_score = score
            save_ckpt(
                run_dir / stage_best_filename(curriculum.stage.name),
                eval=ev,
                mastery={"baseline": True, "passed": False},
                ood=ood_ev,
                unlock=unlock,
                best_kind="stage",
            )
        persist_stage_entry(
            ev,
            mastery={"baseline": True, "passed": False},
            ood=ood_ev,
            unlock=unlock,
            best_kind="stage_entry",
        )
        if unlock.get("action") not in (None, "hold") and log_f is not None:
            log_f.write(
                json.dumps(
                    map_unlock_event_record(
                        it=it0, stage=curriculum.stage.name, judgment=unlock
                    )
                )
                + "\n"
            )
        if log_f is not None:
            log_f.write(
                json.dumps(
                    baseline_event_record(
                        it=it0,
                        env_steps=steps0,
                        stage=curriculum.stage.name,
                        baseline=baseline,
                        unlock=unlock,
                    )
                )
                + "\n"
            )
        pbar.write(
            f"[eval] stage-entry baseline samp_win={baseline['win_rate']:.3f} "
            f"n={baseline['episodes']} "
            f"unlock={unlock.get('action')} "
            f"maps_unlocked={curriculum.maps_unlocked} "
            f"it={trainer.stats.iterations} env_steps={trainer.stats.env_steps}"
        )

    pbar = tqdm(
        total=args.total_steps,
        initial=trainer.stats.env_steps,
        unit="step",
        dynamic_ncols=True,
        desc=curriculum.stage.name,
    )
    eval_s = 0.0
    finalize_reason = "done"

    try:
        if pending_stage_baseline or (
            resume_plan is not None and resume_plan.need_baseline_eval
        ):
            run_force_stage_baseline()

        while trainer.stats.env_steps < args.total_steps:
            t0 = time.time()
            try:
                obs, completed = trainer.collect(env, obs, buf)
            except RolloutInterrupted as exc:
                pbar.write(
                    f"[env] {exc}; discarding partial rollout and resetting trackers"
                )
                obs = reinit_envs()
                continue
            collect_s = time.time() - t0

            for ep in completed:
                curriculum.record_episode(ep.stage_success)
                ep_log_f.write(
                    json.dumps(
                        {
                            "it": trainer.stats.iterations,
                            "env_steps": trainer.stats.env_steps,
                            "stage": curriculum.stage.name,
                            "env_index": ep.env_index,
                            "return": round(ep.ret, 6),
                            "decisions": ep.decisions,
                            "ticks": ep.ticks,
                            "terminal_cause": ep.terminal_cause,
                            "win": ep.win,
                            "stage_success": ep.stage_success,
                            "kills": ep.kills,
                            "final_tiles_frac": ep.final_tiles_frac,
                            "peak_tiles_frac": ep.peak_tiles_frac,
                            "spawned": ep.spawned,
                            "reward_terms": ep.reward_terms,
                            "action_hist": ep.action_hist,
                            "accepted": ep.accepted,
                            "fizzled": ep.fizzled,
                            "intent_count": ep.intent_count,
                            "map": ep.map,
                            "seed": ep.seed,
                        }
                    )
                    + "\n"
                )

            t0 = time.time()
            metrics = trainer.update(buf, obs)
            update_s = time.time() - t0

            sps = steps_per_iter / max(1e-9, collect_s + update_s)
            trainer.stats.sps = sps
            st = trainer.stats

            prev_stage = curriculum.stage.name
            prev_idx = curriculum.idx
            if curriculum.maybe_advance():
                pre_reset = st.rolling_snapshot(prev_stage)
                # New stage: holdout baseline is taken before the next
                # collect. Rolling train windows are not a transfer bar.
                stage_baseline = {}
                pending_stage_baseline = True
                global_best_score, stage_best_score = on_stage_advance_scores(
                    global_best_score
                )
                st.reset_rolling_windows()
                last_eval = {}
                log_f.write(
                    json.dumps(
                        stage_advance_event_record(
                            it=st.iterations,
                            prev_stage=prev_stage,
                            new_stage=curriculum.stage.name,
                            snapshot=pre_reset,
                            reason=str(last_mastery.get("reason", "unknown")),
                        )
                    )
                    + "\n"
                )
                pbar.write(
                    f"[curriculum] advanced {prev_stage} -> {curriculum.stage.name} "
                    f"idx={prev_idx}->{curriculum.idx} "
                    f"reason={last_mastery.get('reason', 'unknown')} "
                    f"modes={last_mastery.get('eval_modes')} "
                    f"n_det={last_mastery.get('det_episodes')} "
                    f"n_samp={last_mastery.get('sampled_episodes')} "
                    f"samp_win={last_mastery.get('sampled_win_rate')} "
                    f"wilson_lb={last_mastery.get('sampled_wilson_lower')} "
                    f"samp_thresh={last_mastery.get('sampled_win_threshold')} "
                    f"wilson_floor={last_mastery.get('sampled_wilson_floor')} "
                    f"det_win={last_mastery.get('det_win_rate')} "
                    f"det_thresh={last_mastery.get('det_threshold')} "
                    f"gate={last_mastery.get('gate')} "
                    f"holdout={last_mastery.get('holdout_map')} "
                    f"global_best={list(global_best_score)}"
                )
                pbar.set_description(curriculum.stage.name)
                save_ckpt(run_dir / "latest.pt")
                obs = reinit_envs()
                persist_stage_entry()
                run_force_stage_baseline()

            if args.eval_every and st.iterations > 0 and st.iterations % args.eval_every == 0:
                suite_meta = curriculum.suite_meta()
                holdout_n = resolve_holdout_episodes(
                    args.eval_episodes, curriculum.stage.min_eval_episodes
                )
                if holdout_n != args.eval_episodes:
                    pbar.write(
                        f"[eval] --eval-episodes={args.eval_episodes} raised to "
                        f"{holdout_n} (stage {curriculum.stage.name} holdout minimum)"
                    )
                pbar.write(
                    f"[eval] holdout eval at it={st.iterations} "
                    f"stage={curriculum.stage.name} "
                    f"n_samp={holdout_n} n_det={args.eval_det_episodes} "
                    f"map={suite_meta['holdout_map']} "
                    f"seeds=frozen"
                )
                t_eval = time.time()
                ood_ev = None
                try:
                    env.set_episode_hooks(None)
                    ev = run_fixed_seed_eval(
                        policy,
                        env,
                        curriculum,
                        device,
                        holdout_n,
                        n_det=args.eval_det_episodes,
                        n_sampled=holdout_n,
                        suite="holdout",
                    )
                    det = ev["deterministic"]
                    samp = ev["sampled"]
                    judgment = curriculum.judge_eval(ev)
                    curriculum.note_eval(judgment["passed"])
                    last_mastery = judgment
                    if args.eval_ood_episodes > 0 and curriculum.stage.ood_maps:
                        ood_ev = run_fixed_seed_eval(
                            policy,
                            env,
                            curriculum,
                            device,
                            args.eval_ood_episodes,
                            n_det=0,
                            n_sampled=args.eval_ood_episodes,
                            suite="ood",
                        )
                    last_eval = ev
                    apply_map_unlock(ev, it=st.iterations)
                    if int((stage_baseline or {}).get("episodes") or 0) < (
                        curriculum.stage.min_eval_episodes
                    ):
                        stage_baseline = build_stage_baseline(
                            ev,
                            iterations=st.iterations,
                            env_steps=st.env_steps,
                            stage=curriculum.stage.name,
                            kind="first_holdout",
                        )
                        persist_stage_entry(ev, mastery=judgment, ood=ood_ev)
                    else:
                        guard = judge_holdout_regression(ev, stage_baseline)
                        if (
                            not is_sentinel_score(stage_best_score)
                            and float(stage_best_score[0])
                            > float(stage_baseline.get("win_rate") or 0.0)
                        ):
                            guard_best = judge_holdout_regression(
                                ev,
                                {
                                    "win_rate": float(stage_best_score[0]),
                                    "episodes": int(
                                        stage_baseline.get("episodes")
                                        or (ev.get("sampled") or {}).get("episodes")
                                        or 32
                                    ),
                                    "kind": "stage_best",
                                },
                            )
                            if guard_best["severe"]:
                                guard = guard_best
                        if guard["severe"]:
                            restored = restore_on_regression(guard)
                            halt_reason = guard["reason"]
                            log_f.write(
                                json.dumps(
                                    regression_event_record(
                                        it=st.iterations,
                                        env_steps=st.env_steps,
                                        stage=curriculum.stage.name,
                                        judgment=guard,
                                        restored_from=restored,
                                    )
                                )
                                + "\n"
                            )
                            pbar.write(
                                f"[eval] STAGE REGRESSION {guard['reason']}: "
                                f"holdout {guard['current_win_rate']} vs "
                                f"baseline {guard['baseline_win_rate']}; "
                                f"restored {restored} and halting "
                                "(no rollback loop; source/global best untouched)"
                            )
                            save_ckpt(
                                run_dir / "latest.pt",
                                eval=ev,
                                mastery=judgment,
                                ood=ood_ev,
                                regression=guard,
                                save_reason="stage_regression",
                            )
                            # latest.pt now holds the restored snapshot.
                            # Do not let the outer finally re-serialize
                            # whatever is still in memory after this point.
                            halt_checkpoint_committed = True
                            save_ckpt(
                                run_dir / "final.pt",
                                eval=ev,
                                mastery=judgment,
                                ood=ood_ev,
                                regression=guard,
                                save_reason="stage_regression",
                            )
                            raise StageRegressionHalt(guard["reason"])
                    score = _eval_score(ev)
                    if should_write_stage_best(score, stage_best_score, ev):
                        stage_best_score = score
                        save_ckpt(
                            run_dir / stage_best_filename(curriculum.stage.name),
                            eval=ev,
                            mastery=judgment,
                            ood=ood_ev,
                            best_kind="stage",
                        )
                        pbar.write(
                            f"[eval] new {stage_best_filename(curriculum.stage.name)} "
                            f"samp_win={samp['win_rate']:.3f} "
                            f"samp_peak={samp['mean_peak_tiles']:.3f}"
                        )
                    if should_write_global_best(score, global_best_score, ev):
                        global_best_score = score
                        save_ckpt(
                            run_dir / "best.pt",
                            eval=ev,
                            mastery=judgment,
                            ood=ood_ev,
                            best_kind="global",
                        )
                        pbar.write(
                            f"[eval] new best.pt samp_win={samp['win_rate']:.3f} "
                            f"det_win={det['win_rate']:.3f} "
                            f"samp_peak={samp['mean_peak_tiles']:.3f} "
                            f"det_peak={det['mean_peak_tiles']:.3f}"
                        )
                    elif eval_is_zero_win(ev) or eval_is_passive_collapse(ev):
                        pbar.write(
                            "[eval] keeping best.pt "
                            f"(global={list(global_best_score)}; "
                            f"new samp_win={samp['win_rate']:.3f} "
                            "is zero-win or collapsed)"
                        )
                    ood_wr = (
                        (ood_ev or {}).get("sampled", {}).get("win_rate")
                        if ood_ev
                        else None
                    )
                    pbar.write(
                        f"[eval] suite=holdout n_samp={holdout_n} "
                        f"n_det={args.eval_det_episodes} "
                        f"map={suite_meta['holdout_map']} "
                        f"samp_win={samp['win_rate']:.3f} "
                        f"det_win={det['win_rate']:.3f} "
                        f"samp_stage={samp.get('stage_success_rate', 0.0):.3f} "
                        f"det_stage={det.get('stage_success_rate', 0.0):.3f} "
                        f"wilson_lb={judgment['sampled_wilson_lower']:.3f} "
                        f"samp_thresh={judgment['sampled_win_threshold']} "
                        f"wilson_floor={judgment['sampled_wilson_floor']} "
                        f"passed={judgment['passed']} reason={judgment['reason']} "
                        f"ood_win={ood_wr}"
                    )
                    log_f.write(
                        json.dumps(
                            {
                                "it": st.iterations,
                                "eval": ev,
                                "ood": ood_ev,
                                "mastery": judgment,
                                "curriculum_suite": suite_meta,
                                "seed_stream": seed_stream.summary(),
                            }
                        )
                        + "\n"
                    )
                finally:
                    eval_s = time.time() - t_eval
                    # Eval (including retries/failure) re-inits the shared
                    # shards; always restore train envs and zero episode
                    # trackers so GAE/returns cannot leak across the eval.
                    obs = reinit_envs()
            else:
                eval_s = 0.0

            record = {
                "it": st.iterations,
                "env_steps": st.env_steps,
                "episodes": st.episodes,
                "stage": curriculum.stage.name,
                "stage_episodes": curriculum.stage_episodes,
                "stage_success_rate": round(curriculum.stage_success_rate, 4),
                "stage_win_rate": round(curriculum.stage_success_rate, 4),
                "train_maps": curriculum.stage.train_map_set,
                "maps_unlocked": curriculum.maps_unlocked,
                "holdout_map": curriculum.stage.holdout_map,
                "ood_maps": list(curriculum.stage.ood_maps),
                "seed_stream": seed_stream.summary(),
                "sps": round(sps, 1),
                "stage_sps": round(sps, 1),
                "collect_s": round(collect_s, 3),
                "update_s": round(update_s, 3),
                "eval_s": round(eval_s, 3),
                **st.rolling_snapshot(curriculum.stage.name),
                **{
                    k: round(v, 5) if isinstance(v, float) else v
                    for k, v in metrics.items()
                },
                "legal_frac": trainer.take_legal_frac(),
            }
            log_f.write(json.dumps(record) + "\n")

            pbar.update(steps_per_iter)
            pbar.set_postfix(
                sps=f"{sps:.0f}",
                collect_s=f"{collect_s:.1f}",
                update_s=f"{update_s:.1f}",
                eval_s=f"{eval_s:.1f}",
                win20=f"{st.win_rate_20:.3f}",
                ss20=f"{st.stage_success_rate_20:.3f}",
                kill20=f"{st.kills_per_ep_20:.2f}",
                ret20=f"{st.mean_return_20:.3f}",
                acc20=f"{st.accept_rate_20:.2f}",
                eps=st.episodes,
                kl=f"{metrics.get('approx_kl', 0.0):.3f}",
                kept=f"{metrics.get('frac_kept', 0.0) * 100:.0f}%",
                aH=f"{metrics.get('action_entropy', 0.0):.2f}",
                nH=f"{metrics.get('action_entropy_norm', 0.0):.2f}",
                e=f"{metrics.get('ent_coef', 0.0):.3f}",
            )
            if st.iterations % 5 == 0:
                save_ckpt(run_dir / "latest.pt")
            if st.iterations % 100 == 0:
                save_ckpt(run_dir / f"it{st.iterations}.pt")
    except StageRegressionHalt as exc:
        finalize_reason = "stage_regression"
        if halt_checkpoint_committed:
            pbar.write(
                f"stage regression halt: {exc}; restored latest.pt already "
                "committed (finally will not rewrite)"
            )
        else:
            pbar.write(f"stage regression halt: {exc}; saving restored latest.pt")
    except KeyboardInterrupt:
        finalize_reason = "keyboard_interrupt"
        pbar.write("interrupted; saving latest.pt and final.pt")
    except Exception as exc:
        finalize_reason = "crash"
        pbar.write(f"training crashed: {exc!r}; saving latest.pt and final.pt")
        raise
    finally:
        pbar.close()
        # Newest recoverable (committed) trainer state — not a partial
        # collect. latest.pt must not rewind to the last %5 snapshot.
        # After a regression restore the snapshot is already on disk;
        # rewriting here would replace it with whatever is still in
        # memory (including a failed/partial restore).
        if should_rewrite_halt_checkpoint(halt_checkpoint_committed):
            save_ckpt(run_dir / "latest.pt", save_reason=finalize_reason)
            save_ckpt(run_dir / "final.pt", save_reason=finalize_reason)
        if log_f is not None:
            log_f.write(
                json.dumps(
                    finalize_event_record(
                        it=trainer.stats.iterations,
                        stage=curriculum.stage.name,
                        env_steps=trainer.stats.env_steps,
                        reason=finalize_reason,
                    )
                )
                + "\n"
            )
            log_f.close()
        if ep_log_f is not None:
            ep_log_f.close()
        env.close()

    print(f"done. checkpoints in {run_dir}, log {log_path}, episodes {ep_log_path}")


if __name__ == "__main__":
    main()
