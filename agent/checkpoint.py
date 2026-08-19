"""Checkpoint ranking, resume planning, and collapse guardrails.

`best.pt` is the run-global sampled-true-win ranking. Stage transitions
and `--force-stage` never reset that write-bar to a sentinel, and a
zero-win or passively collapsed eval cannot replace it. Stage-best files
(`best-<stage>.pt`) track the current stage only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .curriculum import Curriculum, CurriculumRestoreError, STAGES

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


def collapse_resume_reason(ckpt: dict) -> str | None:
    """Hard refuse reason from eval metadata, or None if usable."""
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
) -> ResumePlan:
    """Build resume/force-stage state from checkpoint metadata.

    Does not load weights, write files, or mutate the source checkpoint.
    """
    if force_stage and stage_idx is None:
        raise SystemExit("[train] --force-stage requires --stage")

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
        event=event,
        warnings=warnings,
    )


def format_resume_plan(plan: ResumePlan, *, env_steps: int, total_steps: int) -> str:
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
        f"[dry-run] global_best={list(plan.global_best_score)} "
        f"stage_best={list(plan.stage_best_score)} "
        f"source_best={list(plan.source_best_score)}",
        f"[dry-run] env_steps={env_steps} total_steps={total_steps}",
    ]
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
