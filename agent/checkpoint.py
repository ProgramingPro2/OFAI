"""Checkpoint ranking, resume planning, and collapse guardrails.

`best.pt` is the run-global sampled-true-win ranking. Stage transitions
and `--force-stage` never reset that write-bar to a sentinel, and a
zero-win or passively collapsed eval cannot replace it. Stage-best files
(`best-<stage>.pt`) track the current stage only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .curriculum import (
    Curriculum,
    CurriculumRestoreError,
    STAGES,
    sampled_binomial,
    wilson_interval,
)

BEST_SCORE_SENTINEL: tuple[float, float, float, float] = (-1.0, -1.0, -1.0, -1.0)

# Boat-spam collapse that still holds a sliver of land (confirmed
# v2run1 best.pt after the det-only advance): 0 true wins, boat-heavy
# action mix, weak peak. Stricter than never-spawn / NOOP collapse.
_BOAT_SPAM_FRAC = 0.20
_BOAT_SPAM_PEAK = 0.20


def normalize_best_score(raw) -> tuple[float, float, float, float]:
    """Pad/trim a checkpoint best_score to the 4-tuple ranking."""
    if raw is None:
        return BEST_SCORE_SENTINEL
    vals = [float(x) for x in list(raw)]
    if len(vals) < 4:
        vals.extend([0.0] * (4 - len(vals)))
    return (vals[0], vals[1], vals[2], vals[3])


def is_sentinel_score(score: tuple) -> bool:
    return float(score[0]) < 0.0


def stage_best_filename(stage_name: str) -> str:
    return f"best-{stage_name}.pt"


STAGE_ENTRY_FILENAME = "stage-entry.pt"

# Holdout regression vs the stage-entry / stage-best sampled bar.
# Tuned to the observed v3run1 collapse (baseline ~0.69 → 0.156 / 0.0)
# without halting on ordinary transfer noise (0.69 → 0.50).
REGRESSION_RELATIVE_FLOOR = 0.50
REGRESSION_ABS_DROP = 0.30
REGRESSION_MIN_BASELINE = 0.25
REGRESSION_MIN_EPISODES = 32
TRANSFER_EVAL_EVERY = 5
TRANSFER_LR = 1e-4


def _sampled_block(ev: dict | None) -> dict:
    if not ev:
        return {}
    if "sampled" in ev or "deterministic" in ev:
        return dict(ev.get("sampled") or {})
    return dict(ev)


def _mode_collapsed(mode: dict) -> bool:
    """True when a mode is never-spawn / spawn-then-NOOP / no territory."""
    peak = float(mode.get("mean_peak_tiles", 0.0))
    causes = mode.get("terminal_causes") or {}
    n = max(1, int(mode.get("episodes", 1)))
    no_spawn = float(causes.get("no_spawn", 0)) / n
    hist = mode.get("action_hist") or {}
    total_a = float(sum(hist.values()) or 1.0)
    noop_frac = float(hist.get("noop", 0)) / total_a
    return (
        peak < 0.01
        or no_spawn >= 0.5
        or (noop_frac >= 0.85 and peak < 0.02)
    )


def _eval_score(ev: dict) -> tuple:
    """best.pt ranking for the deployed (sampled) policy.

    Order: sampled true win, det true win, sampled peak, det peak.
    Stage-success / milestone rates never enter this score. Collapse
    zeros that mode's contribution.
    """
    if "sampled" in ev or "deterministic" in ev:
        samp = ev.get("sampled") or {}
        det = ev.get("deterministic") or {}
    else:
        samp = ev
        det = ev
    samp_wr = float(samp.get("win_rate", 0.0))
    det_wr = float(det.get("win_rate", 0.0))
    samp_peak = float(samp.get("mean_peak_tiles", 0.0))
    det_peak = float(det.get("mean_peak_tiles", 0.0))
    if _mode_collapsed(samp):
        samp_wr = 0.0
        samp_peak = 0.0
    if _mode_collapsed(det):
        det_wr = 0.0
        det_peak = 0.0
    return (samp_wr, det_wr, samp_peak, det_peak)


def checkpoint_eval_payload(ckpt: dict) -> dict | None:
    """Eval block used for collapse detection. Unevaluated -> None."""
    ev = ckpt.get("eval")
    if ev:
        return ev
    ev = ckpt.get("last_eval")
    if ev:
        return ev
    return None


def eval_is_zero_win(ev: dict | None) -> bool:
    samp = _sampled_block(ev)
    if int(samp.get("episodes") or 0) <= 0:
        return False
    return float(samp.get("win_rate") or 0.0) <= 0.0


def eval_is_passive_collapse(ev: dict | None) -> bool:
    """Zero-objective / never-spawn / NOOP / boat-spam collapse."""
    samp = _sampled_block(ev)
    if int(samp.get("episodes") or 0) <= 0:
        return False
    if _mode_collapsed(samp):
        return True
    hist = samp.get("action_hist") or {}
    total = float(sum(hist.values()) or 0.0)
    if total <= 0:
        return False
    boat_frac = float(hist.get("boat", 0)) / total
    peak = float(samp.get("mean_peak_tiles") or 0.0)
    wr = float(samp.get("win_rate") or 0.0)
    return wr <= 0.0 and boat_frac >= _BOAT_SPAM_FRAC and peak < _BOAT_SPAM_PEAK


def should_write_global_best(
    new_score: tuple,
    global_best: tuple,
    ev: dict | None = None,
) -> bool:
    """Refuse zero-win / collapsed writes, including vs a sentinel bar.

    A first eval after stage advance used to reset the bar to
    (-1,-1,-1,-1), so a 0-win score replaced `best.pt`. That must not
    happen even if a caller still passes the sentinel.
    """
    if ev is not None:
        if eval_is_passive_collapse(ev) or eval_is_zero_win(ev):
            return False
    elif float(new_score[0]) <= 0.0:
        return False
    return tuple(new_score) > tuple(global_best)


def should_write_stage_best(
    new_score: tuple,
    stage_best: tuple,
    ev: dict | None = None,
) -> bool:
    """Stage-local best. Collapsed evals never write; 0-win peak can."""
    if ev is not None and eval_is_passive_collapse(ev):
        return False
    if ev is None and _score_looks_collapsed(new_score):
        return False
    return tuple(new_score) > tuple(stage_best)


def _score_looks_collapsed(score: tuple) -> bool:
    return float(score[0]) <= 0.0 and float(score[2]) < 0.01


def on_stage_advance_scores(
    global_best: tuple,
) -> tuple[tuple[float, float, float, float], tuple[float, float, float, float]]:
    """Keep the global write-bar; reset only the stage-local bar."""
    return normalize_best_score(global_best), BEST_SCORE_SENTINEL


def resume_event_record(
    *,
    source_checkpoint: str,
    restored_it: int,
    restored_stage: str,
    restored_env_steps: int,
    run_name: str,
    force_stage: bool,
    forced_stage: str | None = None,
    windows_reset: bool = False,
    seed_stream_reset: bool = False,
    reset_optimizer: bool = False,
    keep_optimizer: bool = False,
    need_baseline_eval: bool = False,
    cli_lr: float | None = None,
) -> dict:
    return {
        "event": "resume",
        "source_checkpoint": source_checkpoint,
        "restored_it": int(restored_it),
        "restored_stage": restored_stage,
        "restored_env_steps": int(restored_env_steps),
        "run_name": run_name,
        "force_stage": bool(force_stage),
        "forced_stage": forced_stage,
        "windows_reset": bool(windows_reset),
        "seed_stream_reset": bool(seed_stream_reset),
        "reset_optimizer": bool(reset_optimizer),
        "keep_optimizer": bool(keep_optimizer),
        "need_baseline_eval": bool(need_baseline_eval),
        "cli_lr": cli_lr,
    }


def finalize_event_record(
    *,
    it: int,
    stage: str,
    env_steps: int,
    reason: str,
) -> dict:
    return {
        "event": "finalize",
        "it": int(it),
        "stage": stage,
        "env_steps": int(env_steps),
        "reason": reason,
    }


def stage_advance_event_record(
    *,
    it: int,
    prev_stage: str,
    new_stage: str,
    snapshot: dict,
    reason: str = "mastery",
) -> dict:
    return {
        "event": "stage_advance",
        "it": int(it),
        "prev_stage": prev_stage,
        "stage": new_stage,
        "reason": reason,
        "pre_reset_metrics": snapshot,
    }


def metrics_look_collapsed(ckpt: dict) -> str | None:
    """Soft signal from rolling metrics. Never used to refuse a resume."""
    stats = ckpt.get("stats") or {}
    episodes = int(stats.get("episodes") or 0)
    if episodes < 20:
        return None
    metrics = ckpt.get("metrics") or {}
    win100 = float(metrics.get("win_100") or 0.0)
    ret100 = float(metrics.get("return_100") or 0.0)
    if win100 <= 0.0 and ret100 < 0.0:
        return (
            f"rolling win_100={win100} return_100={ret100} "
            f"over {episodes} lifetime episodes"
        )
    return None


def halted_resume_reason(ckpt: dict) -> str | None:
    """Ordinary resume must not continue a regression-halted run."""
    halt = str(ckpt.get("halt_reason") or "").strip()
    if halt:
        return f"halt_reason={halt}"
    if ckpt.get("save_reason") == "stage_regression":
        return "save_reason=stage_regression"
    if ckpt.get("regression"):
        return "stage_regression metadata"
    return None


def collapse_resume_reason(ckpt: dict) -> str | None:
    """Hard refuse reason from eval metadata, or None if usable."""
    halted = halted_resume_reason(ckpt)
    if halted:
        return halted
    ev = checkpoint_eval_payload(ckpt)
    if ev is None:
        return None
    samp = _sampled_block(ev)
    if int(samp.get("episodes") or 0) <= 0:
        return None
    if eval_is_passive_collapse(ev):
        wr = float(samp.get("win_rate") or 0.0)
        peak = float(samp.get("mean_peak_tiles") or 0.0)
        return (
            f"eval metadata is a zero-win/passive collapse "
            f"(sampled win_rate={wr}, peak={peak})"
        )
    return None


def floor_global_best_at_entry(
    global_best: tuple,
    entry_ev: dict | None,
) -> tuple[float, float, float, float]:
    """Force-stage write bar starts at the stage-entry holdout score.

    A later 12/32 eval cannot become `best.pt` when entry was 21/32.
    Does not write best.pt itself.
    """
    if not entry_ev:
        return normalize_best_score(global_best)
    entry_score = _eval_score(entry_ev)
    current = normalize_best_score(global_best)
    if is_sentinel_score(current) or tuple(entry_score) > tuple(current):
        return entry_score
    return current


def zero_win_resume_warning(ckpt: dict) -> str | None:
    ev = checkpoint_eval_payload(ckpt)
    if ev is None or eval_is_passive_collapse(ev):
        return None
    if eval_is_zero_win(ev):
        samp = _sampled_block(ev)
        return (
            "eval metadata has sampled win_rate=0 "
            f"(episodes={samp.get('episodes')}, "
            f"peak={samp.get('mean_peak_tiles')})"
        )
    return None


class CollapsedCheckpointError(SystemExit):
    """Resume refused because eval metadata shows collapse."""


def guard_collapsed_resume(
    ckpt: dict,
    *,
    force_stage: bool,
    allow_collapsed: bool,
    source: str,
) -> list[str]:
    """Return warning strings. Raises CollapsedCheckpointError to refuse."""
    warnings: list[str] = []
    hard = collapse_resume_reason(ckpt)
    soft = zero_win_resume_warning(ckpt)
    metric = metrics_look_collapsed(ckpt)
    if hard:
        msg = (
            f"[train] refusing to resume {source}: {hard}. "
            "Pass --force-stage with an explicit --stage, or "
            "--allow-collapsed-resume, if this is intentional."
        )
        if force_stage or allow_collapsed:
            warnings.append(msg.replace("refusing to resume", "resuming collapsed"))
        else:
            raise CollapsedCheckpointError(msg)
    elif soft:
        warnings.append(f"[train] warning: {source}: {soft}")
    if metric:
        warnings.append(f"[train] warning: {source}: {metric}")
    return warnings


def same_run_directory(resume_path: str | Path, run_dir: str | Path) -> bool:
    return Path(resume_path).resolve().parent == Path(run_dir).resolve()


def training_budget_warning(env_steps: int, total_steps: int) -> str | None:
    if env_steps >= total_steps:
        return (
            f"[train] checkpoint env_steps={env_steps} >= "
            f"total_steps={total_steps}; the loop would not run. "
            "Use --additional-steps N or raise --total-steps."
        )
    return None


def transfer_eval_cadence_warning(eval_every: int, *, force_stage: bool) -> str | None:
    """Zero eval cadence disables post-baseline unlock and regression halt."""
    if force_stage and int(eval_every) == 0:
        return (
            "[train] --force-stage with --eval-every 0: map unlock/relock "
            "and the holdout regression guard will not run after the "
            f"pre-update baseline. Use --eval-every {TRANSFER_EVAL_EVERY} "
            "for transfer recovery."
        )
    return None


def should_rewrite_halt_checkpoint(committed: bool) -> bool:
    """False once a regression restore has already written latest.pt."""
    return not bool(committed)


def write_touches_foreign_source(
    dest: str | Path,
    run_dir: str | Path,
    source_checkpoint: str | Path | None,
) -> bool:
    """True when dest would write into another run's checkpoint directory."""
    if not source_checkpoint:
        return False
    dest_p = Path(dest).resolve()
    run_p = Path(run_dir).resolve()
    src_p = Path(source_checkpoint).resolve().parent
    try:
        dest_p.relative_to(run_p)
    except ValueError:
        return True
    if src_p == run_p:
        return False
    try:
        dest_p.relative_to(src_p)
        return True
    except ValueError:
        return False


def restore_runtime_from_snapshot(
    ckpt: dict,
    *,
    curriculum,
    seed_stream=None,
    set_rng=None,
) -> list[str]:
    """Restore curriculum / seed stream / RNG from a stage snapshot.

    Trainer stats and env_steps stay with the caller. Missing fields are
    skipped so older stage-entry files remain loadable.
    """
    applied: list[str] = []
    raw_cur = ckpt.get("curriculum")
    if isinstance(raw_cur, dict):
        curriculum.load_state_dict(raw_cur)
        applied.append("curriculum")
    raw_stream = ckpt.get("seed_stream")
    if seed_stream is not None and isinstance(raw_stream, dict):
        seed_stream.load_state_dict(raw_stream)
        applied.append("seed_stream")
    raw_rng = ckpt.get("rng")
    if set_rng is not None and isinstance(raw_rng, dict):
        set_rng(raw_rng)
        applied.append("rng")
    return applied


@dataclass
class ResumePlan:
    curriculum: Curriculum
    global_best_score: tuple[float, float, float, float]
    stage_best_score: tuple[float, float, float, float]
    source_best_score: tuple[float, float, float, float]
    stage_baseline: dict
    last_eval: dict
    restore_rng: bool
    restore_seed_stream: bool
    reset_windows: bool
    source_checkpoint: str
    force_stage: bool
    forced_stage_name: str | None
    same_run: bool
    reset_optimizer: bool
    keep_optimizer: bool
    need_baseline_eval: bool
    pending_stage_baseline: bool
    event: dict
    warnings: list[str] = field(default_factory=list)


def plan_resume(
    *,
    ckpt: dict,
    resume_path: str,
    run_name: str,
    run_dir: str | Path,
    stage_idx: int | None,
    force_stage: bool,
    allow_collapsed: bool = False,
    default_start: int = 0,
    keep_optimizer: bool = False,
    cli_lr: float | None = None,
    eval_every: int | None = None,
) -> ResumePlan:
    """Build resume/force-stage state from checkpoint metadata.

    Does not load weights, write files, or mutate the source checkpoint.
    """
    if force_stage and stage_idx is None:
        raise SystemExit("[train] --force-stage requires --stage")
    if keep_optimizer and not force_stage:
        raise SystemExit("[train] --keep-optimizer requires --force-stage")

    source = str(resume_path)
    same_run = same_run_directory(resume_path, run_dir)
    warnings = list(
        guard_collapsed_resume(
            ckpt,
            force_stage=force_stage,
            allow_collapsed=allow_collapsed,
            source=source,
        )
    )
    if force_stage and same_run:
        warnings.append(
            f"[train] warning: --run-name writes into the source directory "
            f"{Path(run_dir).resolve()}; v2 artifacts are not protected. "
            "Use a new --run-name."
        )

    loaded_global = normalize_best_score(
        ckpt.get("global_best_score", ckpt.get("best_score", BEST_SCORE_SENTINEL))
    )
    loaded_stage = normalize_best_score(
        ckpt.get("stage_best_score", BEST_SCORE_SENTINEL)
    )
    mastery_ver = int((ckpt.get("curriculum") or {}).get("mastery_version", 1))
    if mastery_ver < 2:
        loaded_global = BEST_SCORE_SENTINEL
        warnings.append(
            "[train] mastery_version < 2: reset best.pt ranking "
            "(sampled true-win first)"
        )
    if mastery_ver < 3:
        warnings.append(
            "[train] mastery_version < 3: holdout gate is one sampled "
            "Wilson decision; previous consecutive/det passes discarded"
        )

    start = default_start if stage_idx is None else int(stage_idx)
    reset_optimizer = bool(force_stage) and not bool(keep_optimizer)
    pending_stage_baseline = bool(
        force_stage or ckpt.get("pending_stage_baseline")
    )
    need_baseline_eval = pending_stage_baseline
    if force_stage:
        curriculum = Curriculum(start=int(stage_idx))
        global_best = loaded_global if same_run else BEST_SCORE_SENTINEL
        stage_best = BEST_SCORE_SENTINEL
        restore_rng = False
        restore_seed_stream = False
        reset_windows = True
        stage_baseline: dict = {}
        last_eval: dict = {}
        forced_name = curriculum.stage.name
        reused = reusable_stage_baseline(ckpt, stage=forced_name)
        if reused is not None:
            pending_stage_baseline = False
            need_baseline_eval = False
            stage_baseline = reused
            last_eval = dict(
                reused.get("eval") or checkpoint_eval_payload(ckpt) or {}
            )
            warnings.append(
                "[train] --force-stage: reusing frozen stage_baseline "
                f"from checkpoint (stage={forced_name} it={int(reused.get('iterations') or 0)} "
                f"n={int(reused.get('episodes') or 0)}); skipping pre-update holdout"
            )
        if keep_optimizer:
            warnings.append(
                "[train] --keep-optimizer: inheriting Adam moments/step from "
                "the source checkpoint (advanced; default force-stage resets)"
            )
        else:
            warnings.append(
                "[train] --force-stage: fresh Adam (no inherited moments/step); "
                f"using CLI lr={cli_lr}"
            )
        if reused is None:
            warnings.append(
                "[train] --force-stage: pre-update frozen sampled holdout will "
                "run before the first PPO collect/update; it does not advance "
                "the stage or write best.pt"
            )
        warnings.append(
            "[train] --force-stage: extra train maps stay locked on the "
            "canonical map until a sampled holdout clears the unlock bar"
        )
        warnings.append(
            "[train] --force-stage: holdout regression guard will "
            "restore stage-entry/stage-best weights and halt on a "
            "confirmed collapse (no rollback loop)"
        )
        if eval_every is not None:
            cadence = transfer_eval_cadence_warning(
                int(eval_every), force_stage=True
            )
            if cadence:
                warnings.append(cadence)
    else:
        curriculum = Curriculum(start=start)
        if "curriculum" in ckpt:
            try:
                curriculum.load_state_dict(ckpt["curriculum"])
            except CurriculumRestoreError as exc:
                raise SystemExit(f"[train] cannot restore curriculum: {exc}") from exc
        stored_stage = str(ckpt.get("stage_best_stage") or "")
        stage_best = (
            loaded_stage
            if stored_stage in ("", curriculum.stage.name)
            else BEST_SCORE_SENTINEL
        )
        global_best = loaded_global
        restore_rng = "rng" in ckpt
        restore_seed_stream = isinstance(ckpt.get("seed_stream"), dict)
        reset_windows = False
        stage_baseline = dict(ckpt.get("stage_baseline") or {})
        last_eval = dict(checkpoint_eval_payload(ckpt) or {})
        forced_name = None

    stats = ckpt.get("stats") or {}
    event = resume_event_record(
        source_checkpoint=source,
        restored_it=int(stats.get("iterations") or 0),
        restored_stage=curriculum.stage.name,
        restored_env_steps=int(stats.get("env_steps") or 0),
        run_name=run_name,
        force_stage=force_stage,
        forced_stage=forced_name,
        windows_reset=reset_windows,
        seed_stream_reset=force_stage,
        reset_optimizer=reset_optimizer,
        keep_optimizer=keep_optimizer,
        need_baseline_eval=need_baseline_eval,
        cli_lr=cli_lr,
    )
    return ResumePlan(
        curriculum=curriculum,
        global_best_score=global_best,
        stage_best_score=stage_best,
        source_best_score=loaded_global,
        stage_baseline=stage_baseline,
        last_eval=last_eval,
        restore_rng=restore_rng,
        restore_seed_stream=restore_seed_stream,
        reset_windows=reset_windows,
        source_checkpoint=source,
        force_stage=force_stage,
        forced_stage_name=forced_name,
        same_run=same_run,
        reset_optimizer=reset_optimizer,
        keep_optimizer=bool(keep_optimizer),
        need_baseline_eval=need_baseline_eval,
        pending_stage_baseline=pending_stage_baseline,
        event=event,
        warnings=warnings,
    )


def format_resume_plan(
    plan: ResumePlan,
    *,
    env_steps: int,
    total_steps: int,
    eval_every: int | None = None,
) -> str:
    stage = plan.curriculum.stage
    lines = [
        f"[dry-run] source={plan.source_checkpoint}",
        f"[dry-run] run_name destination same_run={plan.same_run}",
        f"[dry-run] stage={stage.name} idx={plan.curriculum.idx}",
        f"[dry-run] map={stage.map} map_size={stage.map_size} "
        f"nations={stage.nations} bots={stage.bots} "
        f"difficulty={stage.difficulty}",
        f"[dry-run] allowed_actions={stage.allowed_actions}",
        f"[dry-run] stage_episodes={plan.curriculum.stage_episodes} "
        f"last_eval_passed={plan.curriculum.last_eval_passed}",
        f"[dry-run] force_stage={plan.force_stage} "
        f"reset_windows={plan.reset_windows} "
        f"restore_seed_stream={plan.restore_seed_stream} "
        f"restore_rng={plan.restore_rng}",
        f"[dry-run] reset_optimizer={plan.reset_optimizer} "
        f"keep_optimizer={plan.keep_optimizer} "
        f"need_baseline_eval={plan.need_baseline_eval}",
        f"[dry-run] maps_unlocked={plan.curriculum.maps_unlocked} "
        f"(canonical until 26/32 mastery unlock; baseline never unlocks)",
        (
            "[dry-run] pre_update_baseline=frozen sampled holdout before "
            "first collect/update; no stage advance; no global-best write"
            if plan.need_baseline_eval
            else "[dry-run] pre_update_baseline=skip"
        ),
        (
            "[dry-run] regression_guard=halt+restore vs "
            "stage-entry/stage-best holdout"
            if plan.force_stage
            else "[dry-run] regression_guard=stage-local holdout vs baseline"
        ),
        f"[dry-run] global_best={list(plan.global_best_score)} "
        f"stage_best={list(plan.stage_best_score)} "
        f"source_best={list(plan.source_best_score)}",
        f"[dry-run] env_steps={env_steps} total_steps={total_steps}",
    ]
    if eval_every is not None:
        lines.append(
            f"[dry-run] eval_every={eval_every} "
            f"(0=off; transfer recovery uses {TRANSFER_EVAL_EVERY})"
        )
        cadence = transfer_eval_cadence_warning(
            int(eval_every), force_stage=plan.force_stage
        )
        if cadence:
            lines.append(cadence)
    budget = training_budget_warning(env_steps, total_steps)
    if budget:
        lines.append(budget)
    lines.extend(plan.warnings)
    return "\n".join(lines)


def forced_stage_env_preview(plan: ResumePlan, seed: str = "dry-run") -> dict:
    return plan.curriculum.make_env_config(seed)


def stage_index_by_name(name: str) -> int:
    for i, stage in enumerate(STAGES):
        if stage.name == name:
            return i
    raise KeyError(name)


def holdout_reference_from_baseline(baseline: dict | None) -> dict:
    """Normalize a persisted stage baseline to sampled holdout fields."""
    raw = dict(baseline or {})
    if "sampled" in raw or "deterministic" in raw:
        samp = _sampled_block(raw)
        wins, n, wr = sampled_binomial(samp)
        peak = float(samp.get("mean_peak_tiles") or 0.0)
    else:
        wr = float(raw.get("win_rate") or 0.0)
        n = int(raw.get("episodes") or raw.get("sampled_episodes") or 0)
        wins = int(round(wr * n)) if n else 0
        peak = float(raw.get("mean_peak_tiles") or 0.0)
    lb, ub = wilson_interval(wins, n) if n else (0.0, 1.0)
    return {
        "win_rate": wr,
        "episodes": n,
        "wins": wins,
        "mean_peak_tiles": peak,
        "wilson_lower": lb,
        "wilson_upper": ub,
        "kind": raw.get("kind") or "holdout",
        "stage": raw.get("stage") or "",
        "iterations": int(raw.get("iterations") or 0),
        "env_steps": int(raw.get("env_steps") or 0),
    }


def reusable_stage_baseline(
    ckpt: dict,
    *,
    stage: str,
    min_episodes: int = 32,
) -> dict | None:
    """Frozen holdout that already describes this checkpoint's policy.

    ``--force-stage`` normally re-measures the loaded weights. A stage-entry
    snapshot (same stage, same iteration, complete frozen seeds, not
    pending) is reusable so an independent validation does not replay the
    multi-hour pre-update holdout. A later ``latest.pt`` (it != baseline it)
    does not qualify and must not inherit a stale entry bar as a skip.
    """
    if ckpt.get("pending_stage_baseline"):
        return None
    stored = dict(ckpt.get("stage_baseline") or {})
    stats_it = int((ckpt.get("stats") or {}).get("iterations") or -1)
    base_it = int(stored.get("iterations") or -2)
    seeds = list(stored.get("seeds") or [])
    n = int(stored.get("episodes") or 0)
    if (
        str(stored.get("stage") or "") != str(stage)
        or n < int(min_episodes)
        or len(seeds) < int(min_episodes)
        or base_it != stats_it
        or str(stored.get("kind") or "") != "pre_update_holdout"
    ):
        return None
    return stored


def build_stage_baseline(
    ev: dict,
    *,
    iterations: int,
    env_steps: int,
    stage: str,
    kind: str = "pre_update_holdout",
) -> dict:
    """Persist a frozen sampled holdout as the stage-local reference."""
    samp = _sampled_block(ev)
    wins, n, wr = sampled_binomial(samp)
    lb, ub = wilson_interval(wins, n) if n else (0.0, 1.0)
    return {
        "kind": kind,
        "stage": stage,
        "win_rate": wr,
        "episodes": n,
        "wins": wins,
        "mean_peak_tiles": float(samp.get("mean_peak_tiles") or 0.0),
        "wilson_lower": round(lb, 4),
        "wilson_upper": round(ub, 4),
        "env_steps": int(env_steps),
        "iterations": int(iterations),
        "seeds": list(samp.get("seeds") or ev.get("seeds") or []),
        "eval": ev,
    }


def judge_holdout_regression(
    current_ev: dict | None,
    baseline: dict | None,
    *,
    relative_floor: float = REGRESSION_RELATIVE_FLOOR,
    abs_drop: float = REGRESSION_ABS_DROP,
    min_baseline_wr: float = REGRESSION_MIN_BASELINE,
    min_episodes: int = REGRESSION_MIN_EPISODES,
) -> dict:
    """Severe confirmed holdout drop vs stage-entry / stage-best baseline.

    Uses frozen sampled holdout evidence only. Rolling train windows
    never enter this decision. Halt+restore (not continue-after-rollback)
    is the lowest-loop-risk action: the observed collapse had healthy
    per-update KL, so retrying the same settings would loop.
    """
    cur_samp = _sampled_block(current_ev)
    cur_wins, cur_n, cur_wr = sampled_binomial(cur_samp)
    cur_lb, cur_ub = wilson_interval(cur_wins, cur_n) if cur_n else (0.0, 1.0)
    ref = holdout_reference_from_baseline(baseline)
    base_wr = float(ref["win_rate"])
    base_n = int(ref["episodes"])
    base_lb = float(ref["wilson_lower"])
    competent = base_n >= min_episodes and base_wr >= min_baseline_wr
    enough = cur_n >= min_episodes
    zero_win = enough and cur_wr <= 0.0
    relative = enough and cur_wr <= base_wr * relative_floor
    dropped = enough and (base_wr - cur_wr) >= abs_drop
    confirmed = enough and cur_n > 0 and cur_ub < base_lb
    severe = bool(
        competent
        and enough
        and (
            zero_win
            or (relative and dropped and confirmed)
        )
    )
    if not competent:
        reason = "no_competent_baseline"
    elif not enough:
        reason = "current_holdout_too_small"
    elif severe and zero_win:
        reason = "zero_win_collapse"
    elif severe:
        reason = "confirmed_holdout_drop"
    else:
        reason = "within_tolerance"
    return {
        "severe": severe,
        "reason": reason,
        "action": "restore_and_halt" if severe else "continue",
        "baseline_win_rate": round(base_wr, 4),
        "baseline_episodes": base_n,
        "baseline_wilson_lower": round(base_lb, 4),
        "current_win_rate": round(cur_wr, 4),
        "current_episodes": cur_n,
        "current_wilson_upper": round(cur_ub, 4),
        "current_wilson_lower": round(cur_lb, 4),
        "relative_floor": relative_floor,
        "abs_drop": abs_drop,
        "min_baseline_wr": min_baseline_wr,
    }


def choose_restore_snapshot(
    *,
    stage_entry_exists: bool,
    stage_best_exists: bool,
    stage_best_score: tuple,
    baseline_wr: float,
) -> str:
    """Prefer stage-best weights when they beat the entry holdout."""
    if stage_best_exists and float(stage_best_score[0]) > float(baseline_wr):
        return "stage-best"
    if stage_entry_exists:
        return "stage-entry"
    if stage_best_exists:
        return "stage-best"
    return "none"


def baseline_event_record(
    *,
    it: int,
    env_steps: int,
    stage: str,
    baseline: dict,
    unlock: dict | None = None,
) -> dict:
    return {
        "event": "stage_baseline",
        "it": int(it),
        "env_steps": int(env_steps),
        "stage": stage,
        "kind": baseline.get("kind"),
        "win_rate": baseline.get("win_rate"),
        "episodes": baseline.get("episodes"),
        "seeds": list(baseline.get("seeds") or []),
        "unlock": unlock or {},
        "wrote_global_best": False,
        "advanced_stage": False,
    }


def map_unlock_event_record(
    *,
    it: int,
    stage: str,
    judgment: dict,
) -> dict:
    return {
        "event": "map_unlock",
        "it": int(it),
        "stage": stage,
        **judgment,
    }


def regression_event_record(
    *,
    it: int,
    env_steps: int,
    stage: str,
    judgment: dict,
    restored_from: str,
) -> dict:
    return {
        "event": "stage_regression",
        "it": int(it),
        "env_steps": int(env_steps),
        "stage": stage,
        "restored_from": restored_from,
        **judgment,
    }


class StageRegressionHalt(Exception):
    """Severe holdout regression; weights already restored. Stop the run."""
