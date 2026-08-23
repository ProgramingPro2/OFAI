"""Guarded A/B of a BC candidate against a frozen official-core holdout.

Compares identical seeds. Rejects a win-rate or diversity regression.
Never writes or replaces production checkpoints. Official-core accept
requires eval payloads; trajectory NLL is a diagnostic only.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from . import spaces
from .checkpoint import eval_is_passive_collapse, eval_is_zero_win
from .curriculum import sampled_binomial, wilson_interval
from .imitation import (
    ProductionCheckpointError,
    dataset_mean_nll,
    is_production_checkpoint_path,
    load_schema_v2_policy,
    resolve_device,
)
from .seeds import holdout_seed
from .trajectories import (
    EliteDataset,
    TrajectoryValidationError,
    frozen_stage_holdout_seeds,
    load_dataset,
)

DIVERSITY_EPS = 1e-9


class FrozenHoldoutError(ValueError):
    """Fail-closed mismatch of frozen seeds, schema, or eval identity."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class ABVerdict:
    accepted: bool
    reason: str
    baseline_wins: int
    candidate_wins: int
    episodes: int
    baseline_win_rate: float
    candidate_win_rate: float
    baseline_diversity: float
    candidate_diversity: float
    baseline_wilson: tuple[float, float]
    candidate_wilson: tuple[float, float]
    seeds: list[str]
    promoted: bool = False
    nll_baseline: float | None = None
    nll_candidate: float | None = None
    extras: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["baseline_wilson"] = [float(x) for x in self.baseline_wilson]
        d["candidate_wilson"] = [float(x) for x in self.candidate_wilson]
        d["promoted"] = False
        return d


def histogram_entropy(hist: dict | None) -> float:
    """Shannon entropy over ACTION_NAMES. Deterministic key order."""
    counts = np.array(
        [float((hist or {}).get(name, 0.0)) for name in spaces.ACTION_NAMES],
        dtype=np.float64,
    )
    total = float(counts.sum())
    if total <= 0.0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _sampled_block(ev: dict) -> dict:
    if "sampled" in ev or "deterministic" in ev:
        return dict(ev.get("sampled") or {})
    return dict(ev)


def extract_frozen_seeds(ev: dict) -> list[str]:
    top = [str(s) for s in list(ev.get("seeds") or [])]
    samp = _sampled_block(ev)
    inner = [str(s) for s in list(samp.get("seeds") or [])]
    if top and inner and top != inner:
        raise FrozenHoldoutError(
            "SEEDS",
            "top-level seeds and sampled.seeds disagree",
        )
    seeds = top or inner
    if not seeds:
        raise FrozenHoldoutError("SEEDS", "eval payload has no frozen seeds")
    if len(set(seeds)) != len(seeds):
        raise FrozenHoldoutError("SEEDS", "frozen seeds are not unique")
    return seeds


def _require_matching_schema(baseline: dict, candidate: dict) -> None:
    b = baseline.get("schema_version")
    c = candidate.get("schema_version")
    if b is None or c is None:
        raise FrozenHoldoutError(
            "SCHEMA_VERSION",
            "both eval payloads must declare schema_version",
        )
    if int(b) != int(c):
        raise FrozenHoldoutError(
            "SCHEMA_VERSION",
            f"baseline schema={b} candidate schema={c}",
        )
    if int(b) != 2:
        raise FrozenHoldoutError(
            "SCHEMA_VERSION",
            f"eval schema_version={b}; require 2",
        )


def judge_bc_holdout(
    baseline_eval: dict,
    candidate_eval: dict,
    *,
    min_episodes: int = 1,
    nll_baseline: float | None = None,
    nll_candidate: float | None = None,
) -> ABVerdict:
    """Accept BC only when official-core holdout quality does not regress.

    Seed/schema mismatches raise. Quality regressions return accepted=False.
    `promoted` is always False: this harness never replaces production.
    """
    _require_matching_schema(baseline_eval, candidate_eval)
    b_seeds = extract_frozen_seeds(baseline_eval)
    c_seeds = extract_frozen_seeds(candidate_eval)
    if b_seeds != c_seeds:
        raise FrozenHoldoutError(
            "SEEDS",
            "baseline and candidate frozen holdout seeds differ",
        )
    b_samp = _sampled_block(baseline_eval)
    c_samp = _sampled_block(candidate_eval)
    b_wins, b_n, b_wr = sampled_binomial(b_samp)
    c_wins, c_n, c_wr = sampled_binomial(c_samp)
    if b_n != c_n:
        raise FrozenHoldoutError(
            "EPISODES",
            f"baseline episodes={b_n} candidate episodes={c_n}",
        )
    if b_n < int(min_episodes):
        raise FrozenHoldoutError(
            "EPISODES",
            f"holdout episodes={b_n} < min_episodes={min_episodes}",
        )
    b_div = histogram_entropy(b_samp.get("action_hist"))
    c_div = histogram_entropy(c_samp.get("action_hist"))
    b_w = wilson_interval(b_wins, b_n)
    c_w = wilson_interval(c_wins, c_n)
    extras = {
        "kind": "official_core_holdout",
        "collapsed": bool(eval_is_passive_collapse({"sampled": c_samp})),
        "zero_win": bool(eval_is_zero_win({"sampled": c_samp})),
    }
    accepted = True
    reason = "accepted"
    if eval_is_passive_collapse({"sampled": c_samp}):
        accepted = False
        reason = "candidate_collapsed"
    elif (not eval_is_zero_win({"sampled": b_samp})) and eval_is_zero_win(
        {"sampled": c_samp}
    ):
        accepted = False
        reason = "zero_win_regression"
    elif c_wins < b_wins:
        accepted = False
        reason = "win_rate_reduced"
    elif c_div + DIVERSITY_EPS < b_div:
        accepted = False
        reason = "diversity_reduced"
    return ABVerdict(
        accepted=accepted,
        reason=reason,
        baseline_wins=b_wins,
        candidate_wins=c_wins,
        episodes=b_n,
        baseline_win_rate=b_wr,
        candidate_win_rate=c_wr,
        baseline_diversity=b_div,
        candidate_diversity=c_div,
        baseline_wilson=b_w,
        candidate_wilson=c_w,
        seeds=list(b_seeds),
        promoted=False,
        nll_baseline=nll_baseline,
        nll_candidate=nll_candidate,
        extras=extras,
    )


def write_verdict(verdict: ABVerdict, path: str | Path) -> Path:
    dest = Path(path)
    if dest.suffix == ".pt" or is_production_checkpoint_path(dest):
        raise ProductionCheckpointError(
            f"A/B verdict must not write a checkpoint ({dest.name}); "
            "use a JSON path"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(verdict.to_dict(), indent=2, sort_keys=True) + "\n")
    return dest


def load_eval_payload(path: str | Path) -> dict:
    raw = Path(path).read_text()
    ev = json.loads(raw)
    if not isinstance(ev, dict):
        raise FrozenHoldoutError("SCHEMA", "eval file is not a JSON object")
    return ev


def compare_trajectory_nll(
    baseline_ckpt: str | Path,
    candidate_ckpt: str | Path,
    dataset: EliteDataset,
    *,
    device: str = "cpu",
    batch_size: int = 64,
) -> tuple[float, float]:
    """Diagnostic NLL on a frozen trajectory holdout. Not a promotion gate."""
    dev = resolve_device(device)
    base_p, _ = load_schema_v2_policy(baseline_ckpt, dev)
    cand_p, _ = load_schema_v2_policy(candidate_ckpt, dev)
    base_nll = dataset_mean_nll(base_p, dataset, dev, batch_size=batch_size)
    cand_nll = dataset_mean_nll(cand_p, dataset, dev, batch_size=batch_size)
    return float(base_nll), float(cand_nll)


def holdout_seed_list(stage: str, n: int) -> list[str]:
    return frozen_stage_holdout_seeds(stage, n)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Guard BC against a frozen official-core holdout"
    )
    p.add_argument("--baseline-eval", type=str, default="")
    p.add_argument("--candidate-eval", type=str, default="")
    p.add_argument("--baseline-ckpt", type=str, default="")
    p.add_argument("--candidate-ckpt", type=str, default="")
    p.add_argument("--dataset", type=str, default="", help="optional NLL holdout")
    p.add_argument("--out", type=str, default="")
    p.add_argument("--min-episodes", type=int, default=1)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--stage", type=str, default="")
    p.add_argument(
        "--expect-seeds",
        type=int,
        default=0,
        help="if set with --stage, require that many holdout_seed identities",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.out and (
        is_production_checkpoint_path(args.out) or Path(args.out).suffix == ".pt"
    ):
        raise ProductionCheckpointError(
            "A/B --out must be a JSON verdict, not a checkpoint"
        )
    nll_b = nll_c = None
    dataset = None
    if args.dataset:
        dataset = load_dataset(args.dataset, validate=True)
    if args.baseline_ckpt and args.candidate_ckpt and dataset is not None:
        nll_b, nll_c = compare_trajectory_nll(
            args.baseline_ckpt,
            args.candidate_ckpt,
            dataset,
            device=args.device,
        )
    if not args.baseline_eval or not args.candidate_eval:
        if nll_b is None or nll_c is None:
            raise SystemExit(
                "provide --baseline-eval and --candidate-eval, or "
                "--baseline-ckpt --candidate-ckpt --dataset for NLL-only smoke"
            )
        # NLL-only is informational and never an official accept.
        verdict = ABVerdict(
            accepted=False,
            reason="nll_only_not_official_core",
            baseline_wins=0,
            candidate_wins=0,
            episodes=0,
            baseline_win_rate=0.0,
            candidate_win_rate=0.0,
            baseline_diversity=0.0,
            candidate_diversity=0.0,
            baseline_wilson=(0.0, 1.0),
            candidate_wilson=(0.0, 1.0),
            seeds=[],
            promoted=False,
            nll_baseline=nll_b,
            nll_candidate=nll_c,
            extras={"kind": "trajectory_nll_diagnostic"},
        )
    else:
        baseline = load_eval_payload(args.baseline_eval)
        candidate = load_eval_payload(args.candidate_eval)
        if args.stage and args.expect_seeds:
            expected = [holdout_seed(args.stage, i) for i in range(args.expect_seeds)]
            got = extract_frozen_seeds(baseline)
            if got != expected:
                raise FrozenHoldoutError(
                    "SEEDS",
                    "baseline seeds do not match frozen holdout_seed identities",
                )
        verdict = judge_bc_holdout(
            baseline,
            candidate,
            min_episodes=args.min_episodes,
            nll_baseline=nll_b,
            nll_candidate=nll_c,
        )
    if args.out:
        write_verdict(verdict, args.out)
    print(
        f"ab accepted={verdict.accepted} reason={verdict.reason} "
        f"wins={verdict.candidate_wins}/{verdict.baseline_wins} "
        f"div={verdict.candidate_diversity:.4f}/{verdict.baseline_diversity:.4f} "
        f"promoted=false"
    )
    return 0 if verdict.accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
