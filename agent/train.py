"""PPO training entry point.

Spawns (or connects to) env servers, then alternates rollout collection and
PPO updates. Curriculum stages advance on min_episodes plus one frozen
holdout sampled-true-win mastery decision (Wilson-guarded). Deterministic
argmax is a robustness diagnostic and never gates.

Example:
  .venv/bin/python -m agent.train --servers 4 --envs 8 --workers auto \
      --rollout-len 128 --run-name dev --total-steps 1000000
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
    floor_global_best_at_entry,
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
    cannot_pass_mastery_gate,
)
from .backend import select_training_backend
from .env_client import RolloutInterrupted, ShardedVecEnv, resolve_env_workers
from .model import Policy, obs_to_tensors
from . import spaces
from .imitation import (
    BCConfig,
    assert_safe_output_path,
    behavior_clone,
    is_bc_checkpoint,
)
from .ppo import (
    PPOConfig,
    PPOTrainer,
    RolloutBuffer,
    apply_optimizer_lr,
    classify_outcome,
    optimizer_is_fresh,
)
from .trajectories import EliteTrajectoryRecorder, load_dataset
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


def parse_workers_arg(value: str) -> int | str:
    """Accept ``auto`` / ``safe`` or a non-negative worker count."""
    key = str(value).strip().lower()
    if key in ("auto", "safe"):
        return "auto"
    try:
        n = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown workers {value!r}; expected auto, safe, or an integer >= 0"
        ) from exc
    if n < 0:
        raise argparse.ArgumentTypeError("workers must be >= 0")
    return n


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
    p.add_argument(
        "--workers",
        type=parse_workers_arg,
        default="auto",
        help="actor threads per env-server: N, 0/1=sequential oracle, "
        "auto=resource-safe pick (default). Hard-capped at --envs and 32. "
        "OFAI_ENV_WORKERS supplies N when this is auto.",
    )
    p.add_argument(
        "--backend",
        type=str,
        choices=("official", "rust", "auto"),
        default="auto",
        help="simulator backend. auto (default) selects rust iff every remaining "
        "stage is schema-v2 and fully supported, else official. rust requests "
        "the same fail-closed check. official always uses the TypeScript core. "
        "Unsupported features never approximate or switch mid-episode. "
        "agent.eval stays official-core regardless of this flag.",
    )
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
    p.add_argument(
        "--ent-weighting",
        type=str,
        choices=("uniform", "effective"),
        default="uniform",
        help="uniform: legal-set entropy (fresh default). effective: "
        "exclude always-legal utility types from the entropy bonus",
    )
    p.add_argument(
        "--ref-kl-coef",
        type=float,
        default=0.0,
        help="stage-entry reference KL coefficient (0=off; transfer uses 0.1)",
    )
    p.add_argument(
        "--ref-kl-target",
        type=float,
        default=0.02,
        help="reference KL below this is not penalized",
    )
    p.add_argument(
        "--ref-kl-hard",
        type=float,
        default=0.50,
        help="rollback the PPO update when reference KL exceeds this",
    )
    p.add_argument(
        "--lock-maps",
        action="store_true",
        help="never unlock extra train maps this run",
    )
    p.add_argument(
        "--max-updates",
        type=int,
        default=0,
        help="stop after this many PPO updates this run (0=no cap)",
    )
    p.add_argument(
        "--v4-ablation",
        action="store_true",
        help="bounded transfer ablation: maps locked, reference retention, "
        "eval every 5, 15-update cap, effective entropy, ent-target 0.3",
    )
    p.add_argument(
        "--record-elite",
        action="store_true",
        help="opt-in: record official-core winning episodes (masks/actions/"
        "outcomes) to an atomic elite dataset. Disabled by default.",
    )
    p.add_argument(
        "--elite-path",
        type=str,
        default="",
        help="elite dataset path (default: checkpoints/<run>/elite-official.pt)",
    )
    p.add_argument(
        "--elite-max-episodes",
        type=int,
        default=64,
        help="maximum official-core wins kept in the elite dataset",
    )
    p.add_argument(
        "--elite-max-steps",
        type=int,
        default=100_000,
        help="maximum decision rows kept in the elite dataset",
    )
    p.add_argument(
        "--elite-allow-unlabeled",
        action="store_true",
        help="treat a missing env backend header as official-core (legacy)",
    )
    p.add_argument(
        "--bc-dataset",
        type=str,
        default="",
        help="validated elite dataset to behavior-clone before PPO fine-tune",
    )
    p.add_argument(
        "--bc-out",
        type=str,
        default="",
        help="BC candidate path (default: checkpoints/<run>/bc-warmstart.pt)",
    )
    p.add_argument(
        "--bc-epochs",
        type=int,
        default=4,
        help="behavior-cloning epochs when --bc-dataset is set",
    )
    return p.parse_args(argv)


def apply_v4_ablation_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Opt-in transfer experiment. Does not change fresh-run defaults."""
    if not getattr(args, "v4_ablation", False):
        return args
    args.lock_maps = True
    args.ent_weighting = "effective"
    if float(args.ent_target) == 0.6:
        args.ent_target = 0.3
    if int(args.eval_every) == 0:
        args.eval_every = TRANSFER_EVAL_EVERY
    if int(args.max_updates) <= 0:
        args.max_updates = 15
    if float(args.ref_kl_coef) <= 0.0:
        args.ref_kl_coef = 0.1
    # it800-class resumes already exceed the 10M default ceiling.
    if int(args.additional_steps) <= 0:
        farm = max(1, int(args.servers) * int(args.envs))
        args.additional_steps = int(args.max_updates) * int(args.rollout_len) * farm
    return args


def apply_bc_fine_tune_semantics(
    *,
    ckpt: dict,
    resume_plan,
    trainer: PPOTrainer,
    args: argparse.Namespace,
) -> list[str]:
    """BC candidates must take the official stage-entry holdout.

    Does not write best.pt/latest.pt. Clears inherited eval/baseline so a
    source policy's holdout cannot skip or auto-promote the clone.
    """
    notes = [
        "[train] BC warm-start: stage-entry official holdout is required "
        "before PPO collect/update",
        "[train] BC warm-start: best.pt/latest.pt only via the normal "
        "official scoring gate (never auto-promoted)",
    ]
    resume_plan.pending_stage_baseline = True
    resume_plan.need_baseline_eval = True
    resume_plan.stage_baseline = {}
    resume_plan.last_eval = {}
    if not args.keep_optimizer:
        trainer.reset_optimizer(args.lr)
        resume_plan.reset_optimizer = True
        notes.append(
            f"[train] BC warm-start: fresh Adam at lr={args.lr:g} "
            f"(fresh={optimizer_is_fresh(trainer.opt)})"
        )
    trainer.set_reference(source=str(resume_plan.source_checkpoint or args.resume))
    if trainer.cfg.ref_kl_coef > 0:
        notes.append(
            f"[train] BC warm-start: stage-entry reference frozen "
            f"ref_kl_coef={trainer.cfg.ref_kl_coef:g} "
            f"hard={trainer.cfg.ref_kl_hard:g}"
        )
    if int(args.eval_every) == 0:
        notes.append(
            "[train] BC warm-start: --eval-every 0; after the required "
            "pre-update holdout the regression guard will not run"
        )
    bc = ckpt.get("bc") if isinstance(ckpt.get("bc"), dict) else {}
    if bc.get("accepted") or bc.get("promoted"):
        notes.append(
            "[train] warning: candidate metadata claimed accepted/promoted; "
            "ignoring — official holdout still required"
        )
    return notes


def should_run_holdout_eval(
    *,
    iterations: int,
    eval_every: int,
    updates_this_run: int = 0,
    max_updates: int = 0,
) -> bool:
    """Periodic holdout, plus a final holdout when a run hits --max-updates.

    The 15-update ablation must measure update 15. Breaking before this
    check dropped the last cadence eval (it800: 815) and the jsonl row.
    """
    if int(eval_every) <= 0:
        return False
    if int(iterations) > 0 and int(iterations) % int(eval_every) == 0:
        return True
    return int(max_updates) > 0 and int(updates_this_run) >= int(max_updates)


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


# Consecutive shard deaths with no newly completed episode. Completing
# at least one seed resets the counter so a flaky shard cannot discard
# the rest of a holdout.
FIXED_SEED_EVAL_RETRIES = 3

EVAL_PURPOSE_SCORE = "score"
EVAL_PURPOSE_MASTERY = "mastery"
EVAL_PURPOSE_BEST = "best-checkpoint"
EVAL_PURPOSE_GATE = "gate"
_SCORE_EVAL_PURPOSES = frozenset(
    {EVAL_PURPOSE_SCORE, EVAL_PURPOSE_MASTERY, EVAL_PURPOSE_BEST, "best"}
)
_GATE_EVAL_PURPOSES = frozenset({EVAL_PURPOSE_GATE, "gate-only"})


def eval_allows_early_reject(purpose: str) -> bool:
    """Gate-only evals may stop; scoring / mastery / best-checkpoint cannot."""
    key = str(purpose or EVAL_PURPOSE_SCORE).strip().lower()
    if key in _GATE_EVAL_PURPOSES:
        return True
    if key in _SCORE_EVAL_PURPOSES:
        return False
    raise ValueError(
        f"unknown eval purpose {purpose!r}; expected one of "
        f"{sorted(_SCORE_EVAL_PURPOSES | _GATE_EVAL_PURPOSES)}"
    )


class PendingSeedScheduler:
    """Deterministic leftover-seed queue for one eval mode.

    Seeds stay in planned order. Completed results are kept across shard
    interrupts; only running / pending seeds are retried. Recording the
    same seed twice is a no-op so failures cannot double-count.
    """

    def __init__(self, configs: list[dict]):
        self.configs = list(configs)
        self._order = [str(c["seed"]) for c in self.configs]
        self._by_seed = {str(c["seed"]): dict(c) for c in self.configs}
        self.completed: dict[str, dict] = {}
        self._pending = list(self._order)
        self._running: list[str] = []

    @property
    def planned(self) -> int:
        return len(self._order)

    @property
    def pending_count(self) -> int:
        return len(self._pending)

    @property
    def completed_count(self) -> int:
        return len(self.completed)

    def has_pending(self) -> bool:
        return bool(self._pending)

    def claim(self, n: int) -> list[dict]:
        if n <= 0:
            return []
        take = self._pending[:n]
        self._pending = self._pending[n:]
        self._running = list(take)
        return [self._by_seed[seed] for seed in take]

    def complete(self, record: dict) -> None:
        seed = str(record["seed"])
        if seed in self.completed:
            return
        self.completed[seed] = dict(record)
        if seed in self._running:
            self._running.remove(seed)

    def release_incomplete(self) -> None:
        """Return unfinished running seeds to pending in planned order."""
        leftover = set(self._running) | set(self._pending)
        self._running = []
        self._pending = [
            seed
            for seed in self._order
            if seed in leftover and seed not in self.completed
        ]

    def abandon_running(self) -> None:
        """Drop in-flight seeds without completing or retrying them."""
        self._running = []

    def wins(self) -> int:
        return sum(1 for rec in self.completed.values() if rec.get("won"))

    def summarize(self, *, stage: str, suite: str, early_rejected: bool = False) -> dict:
        recs = [self.completed[seed] for seed in self._order if seed in self.completed]
        n = len(recs)
        wins = sum(1 for rec in recs if rec.get("won"))
        stage_hits = sum(1 for rec in recs if rec.get("stage_ok"))
        deaths = sum(1 for rec in recs if rec.get("death"))
        kills = sum(int(rec.get("kills") or 0) for rec in recs)
        rets = [float(rec["ret"]) for rec in recs]
        peaks = [float(rec["peak"]) for rec in recs]
        finals = [float(rec["final"]) for rec in recs]
        ticks = [int(rec["ticks"]) for rec in recs]
        causes: dict[str, int] = {}
        hist: dict[str, int] = {}
        accepted = 0
        total_a = 0
        for rec in recs:
            cause = str(rec.get("cause") or "timeout")
            causes[cause] = causes.get(cause, 0) + 1
            for name, count in (rec.get("action_hist") or {}).items():
                hist[name] = hist.get(name, 0) + int(count)
            accepted += int(rec.get("accepted") or 0)
            total_a += int(rec.get("total_a") or 0)
        return {
            "episodes": n,
            "win_rate": (wins / n) if n else 0.0,
            "stage_success_rate": (stage_hits / n) if n else 0.0,
            "death_rate": (deaths / n) if n else 0.0,
            "kills_per_ep": (kills / n) if n else 0.0,
            "mean_return": float(np.mean(rets)) if rets else 0.0,
            "mean_peak_tiles": float(np.mean(peaks)) if peaks else 0.0,
            "mean_final_tiles": float(np.mean(finals)) if finals else 0.0,
            "mean_ticks": float(np.mean(ticks)) if ticks else 0.0,
            "terminal_causes": causes,
            "action_hist": hist,
            "accept_rate": accepted / max(1, total_a),
            "stage": stage,
            "suite": suite,
            "seeds": [rec["seed"] for rec in recs],
            "maps": [rec.get("map", "") for rec in recs],
            "early_rejected": bool(early_rejected),
            "pending": list(self._pending),
        }


def _empty_eval_mode(*, stage: str, suite: str) -> dict:
    return {
        "episodes": 0,
        "win_rate": 0.0,
        "stage_success_rate": 0.0,
        "stage": stage,
        "suite": suite,
        "seeds": [],
        "maps": [],
        "early_rejected": False,
        "pending": [],
    }


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
    purpose: str = EVAL_PURPOSE_SCORE,
) -> dict:
    """Inline eval matching the active stage. Runs deterministic then sampled.

    Default seeds are frozen per (stage, index) and do not rotate with
    iteration. `seed_prefix` overrides that for tests. Milestone
    termination is omitted so `win_rate` is the core 80% FFA outcome.

    Completed episodes are kept across ``RolloutInterrupted``. Only
    failed and still-pending seeds are re-inited; a seed is never
    counted twice. ``purpose="gate"`` may stop when the mastery bar is
    already unreachable. Scoring, mastery, and best-checkpoint purposes
    always finish the planned seed list. Results are aggregated in
    planned seed order so lane count and retries do not change the
    reported identities.
    """
    sampled_n = n_sampled if n_sampled is not None else n_episodes
    det_n = n_det if n_det is not None else n_episodes
    allow_early = eval_allows_early_reject(purpose)
    stage = curriculum.stage
    results: dict = {"deterministic": {}, "sampled": {}}
    early_rejected = False
    if allow_early:
        modes: tuple[tuple[str, int], ...] = (
            ("sampled", sampled_n),
            ("deterministic", det_n),
        )
    else:
        modes = (("deterministic", det_n), ("sampled", sampled_n))

    for mode, mode_n in modes:
        if mode_n <= 0 or (early_rejected and mode == "deterministic"):
            results[mode] = _empty_eval_mode(stage=stage.name, suite=suite)
            continue
        scheduler = PendingSeedScheduler(
            _eval_configs(
                curriculum,
                mode_n,
                0,
                suite=suite,
                seed_prefix=seed_prefix,
                mode=mode,
            )
        )
        last_exc: RolloutInterrupted | None = None
        consecutive_fails = 0
        while scheduler.has_pending():
            if allow_early and mode == "sampled" and _sampled_gate_unreachable(
                scheduler, curriculum
            ):
                early_rejected = True
                scheduler.abandon_running()
                break
            completed_before = scheduler.completed_count
            try:
                finished = _run_pending_eval_batch(
                    policy,
                    env,
                    curriculum,
                    device,
                    scheduler,
                    mode=mode,
                    suite=suite,
                    allow_early=allow_early,
                )
            except RolloutInterrupted as exc:
                last_exc = exc
                made_progress = scheduler.completed_count > completed_before
                scheduler.release_incomplete()
                if made_progress:
                    consecutive_fails = 0
                else:
                    consecutive_fails += 1
                print(
                    f"[eval] {exc}; keeping {scheduler.completed_count} completed "
                    f"{mode} seed(s), retrying {scheduler.pending_count} pending "
                    f"(fail {consecutive_fails}/{FIXED_SEED_EVAL_RETRIES})"
                )
                if consecutive_fails >= FIXED_SEED_EVAL_RETRIES:
                    raise RuntimeError(
                        f"fixed-seed eval failed after {FIXED_SEED_EVAL_RETRIES} "
                        f"shard-restart attempts ({last_exc})"
                    ) from last_exc
                continue
            if finished:
                consecutive_fails = 0
            if allow_early and mode == "sampled" and _sampled_gate_unreachable(
                scheduler, curriculum
            ):
                early_rejected = True
                scheduler.abandon_running()
                break
        results[mode] = scheduler.summarize(
            stage=stage.name, suite=suite, early_rejected=early_rejected
        )
    results["suite"] = suite
    results["purpose"] = purpose
    results["early_rejected"] = bool(early_rejected)
    results["holdout_map"] = stage.holdout_map
    results["holdout_map_size"] = stage.holdout_map_size
    return results


def _sampled_gate_unreachable(
    scheduler: PendingSeedScheduler, curriculum: Curriculum
) -> bool:
    stage = curriculum.stage
    return cannot_pass_mastery_gate(
        scheduler.wins(),
        scheduler.completed_count,
        scheduler.planned,
        min_episodes=stage.min_eval_episodes,
        win_threshold=stage.sampled_win_threshold,
        wilson_floor=stage.sampled_wilson_floor,
        wilson_z=stage.sampled_wilson_z,
    )


def _step_active_lanes(env: ShardedVecEnv, actions: np.ndarray, live: np.ndarray):
    """Call ``step_active`` with a live mask when the env supports it."""
    try:
        return env.step_active(actions, live=live)
    except TypeError:
        return env.step_active(actions)


def _run_pending_eval_batch(
    policy: Policy,
    env: ShardedVecEnv,
    curriculum: Curriculum,
    device: torch.device,
    scheduler: PendingSeedScheduler,
    *,
    mode: str,
    suite: str,
    allow_early: bool,
) -> int:
    """Init the next pending prefix and run those lanes to completion.

    Returns how many new episodes finished. On shard death the claimed
    unfinished seeds stay on the scheduler as running so the caller can
    requeue them; completed records are already stored.
    """
    stage = curriculum.stage
    stage_frac = float(stage.win_tiles_frac)
    batch_n = min(int(env.k), scheduler.pending_count)
    if batch_n <= 0:
        return 0
    cfgs = scheduler.claim(batch_n)
    completed_before = scheduler.completed_count
    pad = [
        {
            **curriculum.make_env_config(
                f"eval-pad-{suite}-{i}", objective=True, suite=suite
            ),
            "shaping": 0.0,
        }
        for i in range(max(0, int(env.k) - batch_n))
    ]
    use_active = hasattr(env, "init_active") and hasattr(env, "step_active")
    use_leading = hasattr(env, "init_leading") and hasattr(env, "step_leading")
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
    lane_hist: list[dict[str, int]] = [{} for _ in range(batch_n)]
    lane_accepted = np.zeros(batch_n, dtype=np.int32)
    lane_total = np.zeros(batch_n, dtype=np.int32)
    while active[:batch_n].any():
        if allow_early and mode == "sampled" and _sampled_gate_unreachable(
            scheduler, curriculum
        ):
            scheduler.abandon_running()
            break
        with torch.no_grad():
            obs_t = obs_to_tensors(obs, device)
            acts, _, _ = policy.act(obs_t, deterministic=(mode == "deterministic"))
        actions = acts.cpu().numpy().astype(np.int32)
        if use_active:
            step = _step_active_lanes(env, actions, active[:work_k])
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
            lane_hist[i][name] = lane_hist[i].get(name, 0) + 1
            lane_total[i] += 1
            acc = info.get("actionAccepted", info.get("action_accepted", True))
            if acc:
                lane_accepted[i] += 1
            if not step.dones[i]:
                continue
            active[i] = False
            cause, won, stage_ok = classify_outcome(info, stage_frac)
            scheduler.complete(
                {
                    "seed": str(cfgs[i]["seed"]),
                    "map": str(cfgs[i]["map"]),
                    "won": bool(won),
                    "stage_ok": bool(stage_ok),
                    "death": cause == "death",
                    "kills": int(info.get("kills", 0)),
                    "ret": float(ep_ret[i]),
                    "peak": float(info.get("peakTilesFrac", info.get("tilesFrac", 0))),
                    "final": float(info.get("tilesFrac", 0)),
                    "ticks": int(info.get("tick", 0)),
                    "cause": cause,
                    "action_hist": dict(lane_hist[i]),
                    "accepted": int(lane_accepted[i]),
                    "total_a": int(lane_total[i]),
                }
            )
            ep_ret[i] = 0.0
            if allow_early and mode == "sampled" and _sampled_gate_unreachable(
                scheduler, curriculum
            ):
                scheduler.abandon_running()
                active[:batch_n] = False
                break
        obs = step.obs
    return scheduler.completed_count - completed_before


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
    # Never unlock extra train maps from the pre-update baseline. A
    # 21/32 peninsula zero-shot previously cleared the 16/32 bar.
    unlock = {
        "unlock": False,
        "relock": False,
        "reason": "pre_update_baseline",
        "action": "hold",
        "maps_unlocked": curriculum.maps_unlocked,
        "prev_maps_unlocked": curriculum.maps_unlocked,
        "sampled_win_rate": baseline.get("win_rate"),
        "sampled_episodes": baseline.get("episodes"),
        "stage": curriculum.stage.name,
    }
    return baseline, ev, ood_ev, unlock


def main(argv: list[str] | None = None) -> None:
    args = apply_v4_ablation_defaults(parse_args(argv))
    worker_res = resolve_env_workers(
        args.workers,
        envs_per_server=args.envs,
        num_servers=args.servers,
    )
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
        ent_weighting=args.ent_weighting,
        ref_kl_coef=args.ref_kl_coef,
        ref_kl_target=args.ref_kl_target,
        ref_kl_hard=args.ref_kl_hard,
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

    if args.bc_dataset:
        bc_ds = load_dataset(args.bc_dataset, validate=True)
        print(
            f"[train] bc dataset schema={bc_ds.schema_version} "
            f"episodes={bc_ds.n_episodes} steps={bc_ds.n_steps} "
            f"checksum={bc_ds.checksum}"
        )
        if args.dry_run:
            print(
                f"[dry-run] bc_dataset={args.bc_dataset} "
                f"bc_out={args.bc_out or (run_dir / 'bc-warmstart.pt')} "
                f"bc_epochs={args.bc_epochs} init={resume_path or '(fresh)'} "
                "(no write)"
            )
        else:
            run_dir.mkdir(parents=True, exist_ok=True)
            bc_out = Path(args.bc_out) if args.bc_out else run_dir / "bc-warmstart.pt"
            assert_safe_output_path(bc_out)
            result = behavior_clone(
                bc_ds,
                out_path=bc_out,
                init_ckpt=resume_path or None,
                cfg=BCConfig(
                    epochs=int(args.bc_epochs),
                    seed=int(args.seed),
                    device=cfg.device,
                    lr=float(args.lr),
                ),
            )
            print(
                f"[train] wrote BC candidate {result.path} "
                f"nll {result.train_nll:.4f}->{result.final_nll:.4f} "
                "accepted=false promoted=false"
            )
            resume_path = str(result.path)

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
            trainer.set_reference(source=str(resume_path))
            print(
                f"[train] --force-stage {curriculum.stage.name}: "
                "stage-local curriculum/mastery/windows/seed-stream reset; "
                "global-best metadata retained; "
                f"writing to {run_dir}"
            )
            if trainer.cfg.ref_kl_coef > 0:
                print(
                    f"[train] stage-entry reference frozen from {resume_path} "
                    f"ref_kl_coef={trainer.cfg.ref_kl_coef:g} "
                    f"target={trainer.cfg.ref_kl_target:g} "
                    f"hard={trainer.cfg.ref_kl_hard:g}"
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
        if is_bc_checkpoint(ckpt):
            for note in apply_bc_fine_tune_semantics(
                ckpt=ckpt,
                resume_plan=resume_plan,
                trainer=trainer,
                args=args,
            ):
                print(note)
            pending_stage_baseline = True
            stage_baseline = {}
            last_eval = {}
            global_best_score = BEST_SCORE_SENTINEL
            stage_best_score = BEST_SCORE_SENTINEL
    elif args.stage is not None:
        curriculum = Curriculum(start=args.stage)

    if args.lock_maps:
        curriculum.maps_locked = True
        curriculum.maps_unlocked = False

    backend_decision = select_training_backend(
        getattr(args, "backend", "official"),
        STAGES[curriculum.idx :],
    )
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
        if args.v4_ablation:
            print(
                "[dry-run] v4-ablation: maps_locked "
                f"ref_kl_coef={args.ref_kl_coef:g} "
                f"ref_kl_target={args.ref_kl_target:g} "
                f"ent_target={args.ent_target:g} "
                f"ent_weighting={args.ent_weighting} "
                f"eval_every={args.eval_every} "
                f"max_updates={args.max_updates} "
                f"additional_steps={args.additional_steps} "
                "(short ablation, not 10M steps)"
            )
            print(
                "[dry-run] ablation success: holdout >= baseline and "
                "attack share does not collapse to boat/build/diplo/noop; "
                "failure: monotonic holdout drop or utility-action takeover"
            )
        if args.record_elite:
            elite_dest = args.elite_path or str(run_dir / "elite-official.pt")
            print(
                f"[dry-run] record_elite=true path={elite_dest} "
                f"max_episodes={args.elite_max_episodes} "
                f"max_steps={args.elite_max_steps} "
                f"allow_unlabeled={args.elite_allow_unlabeled} "
                "(official backend only; atomic write; no production names)"
            )
        if resume_path and is_bc_checkpoint(ckpt):
            print(
                "[dry-run] bc_warm_start=true pending_stage_baseline=true "
                "official holdout required before PPO; "
                "best.pt only via official scoring gate"
            )
        print(worker_res.log_line())
        print(backend_decision.log_line())
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    # Fresh run must not silently append to a prior run's logs.
    log_mode = "a" if resume_path else "w"
    log_f = open(log_path, log_mode, buffering=1)
    ep_log_f = open(ep_log_path, log_mode, buffering=1)
    if resume_plan is not None:
        log_f.write(json.dumps(resume_plan.event) + "\n")

    print(worker_res.log_line())
    print(backend_decision.log_line())
    if backend_decision.backend != "official":
        print(
            "[env] rust selected because every remaining stage is "
            "schema-v2 supported; agent.eval stays official-core"
        )
    env = ShardedVecEnv(
        num_servers=args.servers,
        base_port=args.port,
        envs_per_server=args.envs,
        connect=args.connect,
        workers=worker_res.workers,
        backend=backend_decision.backend,
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

    if args.record_elite:
        elite_path = (
            Path(args.elite_path) if args.elite_path else run_dir / "elite-official.pt"
        )
        assert_safe_output_path(elite_path)
        trainer.elite = EliteTrajectoryRecorder(
            enabled=True,
            n_envs=total_envs,
            max_episodes=int(args.elite_max_episodes),
            max_steps=int(args.elite_max_steps),
            allow_unlabeled_backend=bool(args.elite_allow_unlabeled),
            path=elite_path,
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
    print(f"[env] {env.backend_summary()}")
    if trainer.elite is not None:
        if trainer.elite.accepts_backend(env.backend):
            print(
                f"[train] elite recording enabled backend={env.backend} "
                f"path={trainer.elite.path} max_episodes={trainer.elite.max_episodes}"
            )
        else:
            trainer.elite.enabled = False
            print(
                f"[train] elite recording disabled: backend={env.backend!r} "
                "is not official-core"
            )
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
        nonlocal stage_best_score, global_best_score
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
        eval_failed = False
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
        except BaseException:
            eval_failed = True
            raise
        finally:
            try:
                obs = reinit_envs()
            except Exception:
                # A dead shard during reinit must not hide the eval error
                # (JSONDecodeError on a truncated frame previously did).
                if not eval_failed:
                    raise
        if trainer.stats.iterations != it0 or trainer.stats.env_steps != steps0:
            raise RuntimeError("baseline eval advanced trainer counters")
        stage_baseline = baseline
        last_eval = ev
        pending_stage_baseline = False
        global_best_score = floor_global_best_at_entry(global_best_score, ev)
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
    updates_this_run = 0

    try:
        if pending_stage_baseline or (
            resume_plan is not None and resume_plan.need_baseline_eval
        ):
            run_force_stage_baseline()
        elif (
            resume_plan is not None
            and resume_plan.force_stage
            and stage_baseline
        ):
            reused_ev = last_eval if last_eval else dict(
                stage_baseline.get("eval") or {}
            )
            if reused_ev:
                global_best_score = floor_global_best_at_entry(
                    global_best_score, reused_ev
                )
            persist_stage_entry(
                reused_ev,
                mastery={"baseline": True, "passed": False, "reused": True},
                unlock={
                    "action": "hold",
                    "reason": "reused_stage_baseline",
                    "maps_unlocked": False,
                },
                best_kind="stage_entry",
            )
            if log_f is not None:
                log_f.write(
                    json.dumps(
                        baseline_event_record(
                            it=int(trainer.stats.iterations),
                            env_steps=int(trainer.stats.env_steps),
                            stage=curriculum.stage.name,
                            baseline=stage_baseline,
                            unlock={
                                "action": "hold",
                                "reason": "reused_stage_baseline",
                                "maps_unlocked": False,
                            },
                        )
                    )
                    + "\n"
                )
            pbar.write(
                f"[eval] reused stage-entry baseline "
                f"samp_win={float(stage_baseline.get('win_rate') or 0.0):.3f} "
                f"n={int(stage_baseline.get('episodes') or 0)} "
                f"it={trainer.stats.iterations} "
                f"env_steps={trainer.stats.env_steps} "
                f"(no pre-update holdout)"
            )

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
            if trainer.elite is not None:
                flushed = trainer.elite.flush()
                if flushed is not None:
                    pbar.write(
                        f"[elite] wrote {flushed} episodes={trainer.elite.n_episodes} "
                        f"steps={trainer.elite.n_steps}"
                    )

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
            updates_this_run += 1
            capped = bool(args.max_updates) and updates_this_run >= int(
                args.max_updates
            )
            if capped:
                finalize_reason = "max_updates"
                pbar.write(
                    f"[train] reached --max-updates={args.max_updates}; "
                    "running due/final holdout then stopping"
                )

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

            if should_run_holdout_eval(
                iterations=st.iterations,
                eval_every=args.eval_every,
                updates_this_run=updates_this_run,
                max_updates=args.max_updates,
            ):
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
            if capped:
                break
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
        if getattr(trainer, "elite", None) is not None:
            flushed = trainer.elite.flush()
            if flushed is not None:
                print(
                    f"[elite] final {flushed} episodes={trainer.elite.n_episodes} "
                    f"steps={trainer.elite.n_steps}"
                )
        env.close()

    print(f"done. checkpoints in {run_dir}, log {log_path}, episodes {ep_log_path}")


if __name__ == "__main__":
    main()
