"""Evaluation: run N episodes with deterministic and/or sampled policy.

Training and deployment use the sampled PPO policy. Deterministic
argmax is a robustness / greedy diagnostic, not the definition of
policy quality. `win_rate` is the core 80% FFA objective only;
`stage_success_rate` is a separate competency flag.

Reports win rate, Wilson 95% interval, terminal causes, full return,
kills, ticks, territory, action histogram, acceptance rate, plus
attack-target / quantity / legal-action / territory-effect telemetry.

Stage mode (`--stage`) uses frozen curriculum `holdout_seed` /
`ood_seed` identities and stage-derived map/config/actions. Manual
flags (`--map`, `--episodes`, `--mode`, …) stay backward compatible
when `--stage` is omitted.

`--scripted expander` runs a deterministic patient-expander/attacker
instead of a checkpoint (used to prove the land-duel env is winnable).

Example:
  .venv/bin/python -m agent.eval --ckpt checkpoints/run/latest.pt \
      --episodes 20 --map Halkidiki --bots 1 --map-size Compact
  .venv/bin/python -m agent.eval --ckpt checkpoints/run/latest.pt \
      --stage nations-impossible-world --suite holdout \
      --sampled-episodes 64 --det-episodes 8 --workers 2 --json-out eval.json
  .venv/bin/python -m agent.eval --scripted expander --episodes 20 \
      --map Halkidiki --map-size Compact --bots 1 --max-ticks 8000
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import spaces
from .curriculum import (
    DEFAULT_DET_EVAL_EPISODES,
    DEFAULT_HOLDOUT_EPISODES,
    DEFAULT_OOD_EVAL_EPISODES,
    STAGES,
    Curriculum,
    wilson_interval,
)
from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors
from .ppo import classify_outcome
from .seeds import holdout_seed, ood_seed

QUANTITY_PCTS = (5, 15, 30, 50, 80)
# Wilderness capture rate is border-limited, not troop-limited. 50% leaves
# enough standing army to keep a live attack without starving income.
# Player attacks wait while retreat_all is legal so we do not restart combat.
EXPAND_QTY = 3
PLAYER_QTY = 3
EFFECT_WINDOW = 3
TILE_EPS = 1e-4
REGION_GRID = spaces.REGION_GRID

GOAL_STAGE = "nations-impossible-world"
GOAL_HOLDOUT_SAMPLED = 64
GOAL_HOLDOUT_DET = 8
GOAL_OOD_SAMPLED = 16
WILSON_Z = 1.96
MANUAL_SUITE = "manual"
ALLOWED_SUITES = ("holdout", "ood")


@dataclass(frozen=True)
class EvalPlan:
    """Resolved eval request: either manual CLI or frozen stage suite."""

    stage: str
    suite: str
    sampled_n: int
    det_n: int
    workers: int
    envs: int
    sampled_configs: list[dict] = field(default_factory=list)
    det_configs: list[dict] = field(default_factory=list)
    allowed_actions: list[str] = field(default_factory=list)
    map: str = ""
    map_size: str = ""
    nations: int | str = "disabled"
    difficulty: str = "Easy"
    bots: int = 1
    max_ticks: int = 8000
    stage_tiles_frac: float = 0.0
    scripted: str = ""
    mode: str = "both"

    @property
    def is_stage(self) -> bool:
        return bool(self.stage) and self.suite in ALLOWED_SUITES


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument(
        "--scripted",
        choices=["", "expander"],
        default="",
        help="deterministic scripted policy (no checkpoint required)",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument(
        "--sampled-episodes",
        type=int,
        default=None,
        help="sampled (deployed-policy) episode count; required split in --stage mode",
    )
    p.add_argument(
        "--det-episodes",
        type=int,
        default=None,
        help="deterministic diagnostic episode count; ignored by --mode sampled",
    )
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--connect", action="store_true")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="EnvServer actor threads (official always passes --workers)",
    )
    p.add_argument("--map", type=str, default="Halkidiki")
    p.add_argument("--map-size", type=str, default="Compact")
    p.add_argument("--nations", type=str, default="disabled")
    p.add_argument("--difficulty", type=str, default="Easy")
    p.add_argument("--bots", type=int, default=1)
    p.add_argument("--max-ticks", type=int, default=8000)
    p.add_argument("--shaping", type=float, default=0.0)
    p.add_argument(
        "--stage",
        type=str,
        default="",
        help="curriculum stage name (frozen holdout/OOD seeds and stage config)",
    )
    p.add_argument(
        "--suite",
        choices=ALLOWED_SUITES,
        default="",
        help="holdout (canonical map) or ood; requires --stage",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="write machine-readable JSON report to this path",
    )
    p.add_argument(
        "--win-tiles-frac",
        type=float,
        default=0.0,
        help="env competency flag (stageSuccess when peak tiles reach this). "
        "Does not terminate. 0=off; true-objective eval should leave this 0 "
        "and use --stage-tiles-frac to report the metric",
    )
    p.add_argument(
        "--stage-tiles-frac",
        type=float,
        default=0.0,
        help="report stage_success when peakTilesFrac reaches this value "
        "without terminating the episode (0=off)",
    )
    p.add_argument(
        "--allowed-actions",
        type=str,
        default=None,
        help="comma-separated action names passed as allowedActions. "
        "Manual default is EARLY_ACTIONS. When --stage is set, the stage "
        "mask is used (FULL for Impossible) unless this flag is explicit.",
    )
    p.add_argument(
        "--mode",
        choices=["both", "deterministic", "sampled"],
        default="both",
        help="sampled is the deployed policy; deterministic is a greedy diagnostic",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def stage_index(name: str) -> int:
    for i, stage in enumerate(STAGES):
        if stage.name == name:
            return i
    known = [s.name for s in STAGES]
    raise ValueError(f"unknown stage {name!r}; known: {known}")


def curriculum_for_stage(name: str) -> Curriculum:
    return Curriculum(start=stage_index(name))


def frozen_suite_seeds(stage: str, suite: str, n: int) -> list[str]:
    """Frozen holdout_seed / ood_seed identities for a stage suite."""
    fn = ood_seed if suite == "ood" else holdout_seed
    return [fn(stage, i) for i in range(int(n))]


def parse_allowed_actions_csv(raw: str) -> list[str]:
    return [a.strip() for a in str(raw).split(",") if a.strip()]


def wilson_report(wins: int, n: int, z: float = WILSON_Z) -> dict[str, float | int]:
    """Wilson 95% interval plus binomial counts for a win rate."""
    wins_i = max(0, min(int(wins), int(n)))
    n_i = max(0, int(n))
    lo, hi = wilson_interval(wins_i, n_i, z=z)
    return {
        "wins": wins_i,
        "episodes": n_i,
        "win_rate": (wins_i / n_i) if n_i else 0.0,
        "wilson_low": float(lo),
        "wilson_high": float(hi),
        "wilson_z": float(z),
    }


def _default_counts(stage: str, suite: str) -> tuple[int, int]:
    if stage == GOAL_STAGE:
        if suite == "ood":
            return GOAL_OOD_SAMPLED, 0
        return GOAL_HOLDOUT_SAMPLED, GOAL_HOLDOUT_DET
    if suite == "ood":
        return DEFAULT_OOD_EVAL_EPISODES, 0
    return DEFAULT_HOLDOUT_EPISODES, DEFAULT_DET_EVAL_EPISODES


def _apply_mode_counts(
    *,
    mode: str,
    sampled_n: int,
    det_n: int,
    sampled_explicit: bool,
    det_explicit: bool,
) -> tuple[int, int]:
    if mode == "sampled":
        if det_explicit and det_n > 0:
            raise ValueError("--mode sampled cannot be combined with --det-episodes > 0")
        return sampled_n, 0
    if mode == "deterministic":
        if sampled_explicit and sampled_n > 0:
            raise ValueError(
                "--mode deterministic cannot be combined with --sampled-episodes > 0"
            )
        return 0, det_n
    return sampled_n, det_n


def resolve_eval_plan(args: argparse.Namespace) -> EvalPlan:
    """Validate flags and build frozen configs. Manual CLI stays as-is."""
    if args.suite and not args.stage:
        raise ValueError("--suite requires --stage")
    if int(args.workers) < 0:
        raise ValueError("--workers must be >= 0")
    if int(args.envs) < 1:
        raise ValueError("--envs must be >= 1")
    if args.sampled_episodes is not None and int(args.sampled_episodes) < 0:
        raise ValueError("--sampled-episodes must be >= 0")
    if args.det_episodes is not None and int(args.det_episodes) < 0:
        raise ValueError("--det-episodes must be >= 0")
    if int(args.episodes) < 0:
        raise ValueError("--episodes must be >= 0")

    sampled_explicit = args.sampled_episodes is not None
    det_explicit = args.det_episodes is not None
    allowed_explicit = args.allowed_actions is not None

    if args.stage:
        suite = args.suite or "holdout"
        cur = curriculum_for_stage(args.stage)
        stage = cur.stage
        def_s, def_d = _default_counts(args.stage, suite)
        sampled_n = int(args.sampled_episodes) if sampled_explicit else def_s
        det_n = int(args.det_episodes) if det_explicit else def_d
        sampled_n, det_n = _apply_mode_counts(
            mode=args.mode,
            sampled_n=sampled_n,
            det_n=det_n,
            sampled_explicit=sampled_explicit,
            det_explicit=det_explicit,
        )
        if sampled_n + det_n <= 0:
            raise ValueError("stage eval needs at least one sampled or det episode")
        if suite == "ood" and not stage.ood_maps:
            raise ValueError(f"stage {args.stage!r} has no OOD maps")

        if allowed_explicit:
            allowed = parse_allowed_actions_csv(args.allowed_actions)
        else:
            allowed = list(stage.allowed_actions)

        n_hold = max(sampled_n, det_n)
        if suite == "ood":
            all_cfgs = cur.ood_configs(n_hold)
        else:
            all_cfgs = cur.holdout_configs(n_hold)
        sampled_cfgs = [dict(c) for c in all_cfgs[:sampled_n]]
        det_cfgs = [dict(c) for c in all_cfgs[:det_n]]
        if allowed_explicit:
            encoded = spaces.encode_allowed_actions(allowed)
            for cfg in sampled_cfgs + det_cfgs:
                cfg["allowedActions"] = list(encoded)
        tiles = (
            float(args.stage_tiles_frac)
            if float(args.stage_tiles_frac) > 0
            else float(stage.win_tiles_frac or 0.0)
        )
        preview = (sampled_cfgs or det_cfgs)[0]
        return EvalPlan(
            stage=stage.name,
            suite=suite,
            sampled_n=sampled_n,
            det_n=det_n,
            workers=int(args.workers),
            envs=int(args.envs),
            sampled_configs=sampled_cfgs,
            det_configs=det_cfgs,
            allowed_actions=allowed,
            map=str(preview.get("map", stage.map)),
            map_size=str(preview.get("mapSize", stage.map_size)),
            nations=preview.get("nations", stage.nations),
            difficulty=str(preview.get("difficulty", stage.difficulty)),
            bots=int(preview.get("bots", stage.bots)),
            max_ticks=int(preview.get("maxTicks", stage.max_ticks)),
            stage_tiles_frac=tiles,
            scripted=str(args.scripted or ""),
            mode=str(args.mode),
        )

    sampled_n = int(args.episodes)
    det_n = int(args.episodes)
    if sampled_explicit:
        sampled_n = int(args.sampled_episodes)
    if det_explicit:
        det_n = int(args.det_episodes)
    sampled_n, det_n = _apply_mode_counts(
        mode=args.mode,
        sampled_n=sampled_n,
        det_n=det_n,
        sampled_explicit=sampled_explicit or args.mode != "deterministic",
        det_explicit=det_explicit or args.mode != "sampled",
    )
    if args.scripted:
        sampled_n = int(args.sampled_episodes if sampled_explicit else args.episodes)
        det_n = 0
    if sampled_n + det_n <= 0:
        raise ValueError("eval needs at least one episode")

    if allowed_explicit:
        allowed = parse_allowed_actions_csv(args.allowed_actions)
    else:
        allowed = list(spaces.EARLY_ACTIONS)

    nations: int | str = args.nations
    if nations not in ("default", "disabled"):
        nations = int(nations)
    encoded = spaces.encode_allowed_actions(allowed)

    def _manual(n: int, deterministic: bool) -> list[dict]:
        """Keep the historical eval-{tag}-{batch_start}-{lane} seed formula."""
        tag = "det" if deterministic else "samp"
        cfgs: list[dict] = []
        remaining = n
        while remaining > 0:
            batch_n = min(int(args.envs), remaining)
            batch_start = n - remaining
            for i in range(batch_n):
                cfg = {
                    "map": args.map,
                    "mapSize": args.map_size,
                    "nations": nations,
                    "difficulty": args.difficulty,
                    "bots": args.bots,
                    "seed": f"eval-{tag}-{batch_start}-{i}",
                    "maxTicks": args.max_ticks,
                    "decisionInterval": 10,
                    "shaping": args.shaping,
                    "allowedActions": list(encoded),
                }
                if args.win_tiles_frac > 0:
                    cfg["winTilesFrac"] = args.win_tiles_frac
                cfgs.append(cfg)
            remaining -= batch_n
        return cfgs

    sampled_cfgs = _manual(sampled_n, False)
    det_cfgs = _manual(det_n, True)
    return EvalPlan(
        stage="",
        suite=MANUAL_SUITE,
        sampled_n=sampled_n,
        det_n=det_n,
        workers=int(args.workers),
        envs=int(args.envs),
        sampled_configs=sampled_cfgs,
        det_configs=det_cfgs,
        allowed_actions=allowed,
        map=str(args.map),
        map_size=str(args.map_size),
        nations=nations,
        difficulty=str(args.difficulty),
        bots=int(args.bots),
        max_ticks=int(args.max_ticks),
        stage_tiles_frac=float(args.stage_tiles_frac),
        scripted=str(args.scripted or ""),
        mode=str(args.mode),
    )


def _densest_spawn_region(spawn_regions: np.ndarray) -> int:
    """Prefer the largest cluster of legal spawn cells (mainland, not a spit)."""
    grid = np.asarray(spawn_regions, dtype=np.float32).reshape(REGION_GRID, REGION_GRID)
    if not bool(grid.any()):
        return 0
    kernel = np.ones((5, 5), dtype=np.float32)
    padded = np.pad(grid, 2, mode="constant")
    dens = np.zeros_like(grid)
    for dy in range(5):
        for dx in range(5):
            dens += padded[dy : dy + REGION_GRID, dx : dx + REGION_GRID] * kernel[dy, dx]
    dens = np.where(grid > 0, dens, -1.0)
    flat = dens.reshape(-1)
    max_d = float(flat.max())
    candidates = np.flatnonzero(flat == max_d)
    ys, xs = np.nonzero(grid > 0)
    cy, cx = float(ys.mean()), float(xs.mean())
    best = int(candidates[0])
    best_d = 1e18
    for idx in candidates:
        y, x = divmod(int(idx), REGION_GRID)
        d = (y - cy) ** 2 + (x - cx) ** 2
        if d < best_d:
            best_d = d
            best = int(idx)
    return best


def scripted_expander_action(obs: dict, i: int) -> np.ndarray:
    """Patient expander/attacker for the Halkidiki Compact land duel.

    Spawn immediately. Always take legal wilderness (50%) before fighting
    the bot — switching to player attacks at a tiny blob leaves the Easy
    bot a free path to the 80% tile win. Once wilderness is gone, send a
    player attack and NOOP while `retreat_all` is legal (outgoing attack
    in flight). `actionAccepted` only means the intent was translated.
    """
    mask = obs["action_mask"][i]
    if mask[spaces.ACTION_SPAWN]:
        region = _densest_spawn_region(obs["spawn_regions"][i])
        return np.array([spaces.ACTION_SPAWN, 0, region, 0, 0], dtype=np.int32)
    tiles = float(obs["global"][i][2])
    wild = False
    player_slots = np.array([], dtype=np.int64)
    if mask[spaces.ACTION_ATTACK]:
        tm = obs["target_masks"][i, spaces.ACTION_ATTACK]
        wild = bool(tm[0])
        player_slots = np.flatnonzero(tm[1:]) + 1
        if player_slots.size and tiles >= 0.12:
            if mask[spaces.ACTION_RETREAT_ALL]:
                return np.array(
                    [spaces.ACTION_NOOP, 0, 0, 0, 0], dtype=np.int32
                )
            return np.array(
                [spaces.ACTION_ATTACK, int(player_slots[0]), 0, PLAYER_QTY, 0],
                dtype=np.int32,
            )
        if wild and (
            player_slots.size > 0
            or not mask[spaces.ACTION_BOAT]
            or tiles < 0.015
        ):
            return np.array(
                [spaces.ACTION_ATTACK, 0, 0, EXPAND_QTY, 0],
                dtype=np.int32,
            )
    if (
        mask[spaces.ACTION_BOAT]
        and tiles >= 0.015
        and player_slots.size == 0
        and not mask[spaces.ACTION_RETREAT_ALL]
    ):
        if np.asarray(obs["boat_regions"][i]).any():
            region = _densest_spawn_region(obs["boat_regions"][i])
            return np.array(
                [spaces.ACTION_BOAT, 0, region, EXPAND_QTY, 0],
                dtype=np.int32,
            )
    if wild:
        return np.array(
            [spaces.ACTION_ATTACK, 0, 0, EXPAND_QTY, 0],
            dtype=np.int32,
        )
    return np.array([spaces.ACTION_NOOP, 0, 0, 0, 0], dtype=np.int32)


def _resolve_pending(
    pending: list[tuple[str, int, float]],
    tiles: float,
    grew: Counter[str],
    shrank: Counter[str],
    unchanged: Counter[str],
) -> list[tuple[str, int, float]]:
    still: list[tuple[str, int, float]] = []
    for kind, steps_left, tf0 in pending:
        delta = tiles - tf0
        if delta > TILE_EPS:
            grew[kind] += 1
        elif delta < -TILE_EPS:
            shrank[kind] += 1
        elif steps_left <= 1:
            unchanged[kind] += 1
        else:
            still.append((kind, steps_left - 1, tf0))
    return still


def _hygiene(hist: Counter[str], causes: Counter[str], total_a: int) -> dict:
    n_term = max(1, sum(causes.values()))
    return {
        "noop_frac": hist.get("noop", 0) / max(1, total_a),
        "spawn_frac": hist.get("spawn", 0) / max(1, total_a),
        "no_spawn_rate": causes.get("no_spawn", 0) / n_term,
        "timeout_rate": causes.get("timeout", 0) / n_term,
        "death_rate": causes.get("death", 0) / n_term,
    }


def _summarize(
    *,
    planned_seeds: list[str],
    records: dict[str, dict],
    wins: int,
    stage_hits: int,
    deaths: int,
    kills_total: int,
    returns: dict[str, float],
    ticks_survived: dict[str, int],
    peaks: dict[str, float],
    finals: dict[str, float],
    causes: Counter[str],
    hist: Counter[str],
    accepted: int,
    total_a: int,
    legal_counts: np.ndarray,
    legal_n: int,
    attack_wild: int,
    attack_player: int,
    accepted_wild: int,
    accepted_player: int,
    qty_hist: Counter[str],
    grew: Counter[str],
    shrank: Counter[str],
    unchanged: Counter[str],
    maps: list[str],
) -> dict:
    ordered = [records[s] for s in planned_seeds if s in records]
    n = len(ordered)
    wr = wilson_report(wins, n)
    legal_frac = {
        spaces.ACTION_NAMES[i]: round(float(legal_counts[i]) / max(1, legal_n), 4)
        for i in range(spaces.NUM_ACTION_TYPES)
    }

    def _effect(kind: str) -> dict:
        g, s, u = grew[kind], shrank[kind], unchanged[kind]
        resolved = g + s + u
        return {
            "grew": g,
            "shrank": s,
            "unchanged": u,
            "resolved": resolved,
            "affect_rate": (g + s) / max(1, resolved),
        }

    ret_list = [returns[s] for s in planned_seeds if s in returns]
    tick_list = [ticks_survived[s] for s in planned_seeds if s in ticks_survived]
    peak_list = [peaks[s] for s in planned_seeds if s in peaks]
    final_list = [finals[s] for s in planned_seeds if s in finals]
    return {
        "episodes": n,
        "wins": wr["wins"],
        "win_rate": wr["win_rate"],
        "wilson_low": wr["wilson_low"],
        "wilson_high": wr["wilson_high"],
        "wilson_z": wr["wilson_z"],
        "stage_success_rate": stage_hits / max(1, n),
        "death_rate": deaths / max(1, n),
        "kills_per_ep": kills_total / max(1, n),
        "mean_return": float(np.mean(ret_list)) if ret_list else 0.0,
        "mean_ticks": float(np.mean(tick_list)) if tick_list else 0.0,
        "mean_peak_tiles": float(np.mean(peak_list)) if peak_list else 0.0,
        "mean_final_tiles": float(np.mean(final_list)) if final_list else 0.0,
        "terminal_causes": dict(causes),
        "action_hist": dict(hist),
        "accept_rate": accepted / max(1, total_a),
        "legal_frac": legal_frac,
        "attack_targets": {
            "wilderness": attack_wild,
            "player": attack_player,
            "accepted_wilderness": accepted_wild,
            "accepted_player": accepted_player,
        },
        "quantity_hist": dict(qty_hist),
        "territory_effect": {
            "wilderness": _effect("wilderness"),
            "player": _effect("player"),
        },
        "hygiene": _hygiene(hist, causes, total_a),
        "seeds": list(planned_seeds),
        "maps": list(maps),
    }


def _empty_mode() -> dict:
    wr = wilson_report(0, 0)
    return {
        "episodes": 0,
        "wins": 0,
        "win_rate": 0.0,
        "wilson_low": wr["wilson_low"],
        "wilson_high": wr["wilson_high"],
        "wilson_z": wr["wilson_z"],
        "stage_success_rate": 0.0,
        "death_rate": 0.0,
        "kills_per_ep": 0.0,
        "mean_return": 0.0,
        "mean_ticks": 0.0,
        "mean_peak_tiles": 0.0,
        "mean_final_tiles": 0.0,
        "terminal_causes": {},
        "action_hist": {},
        "accept_rate": 0.0,
        "legal_frac": {},
        "attack_targets": {
            "wilderness": 0,
            "player": 0,
            "accepted_wilderness": 0,
            "accepted_player": 0,
        },
        "quantity_hist": {},
        "territory_effect": {
            "wilderness": {
                "grew": 0,
                "shrank": 0,
                "unchanged": 0,
                "resolved": 0,
                "affect_rate": 0.0,
            },
            "player": {
                "grew": 0,
                "shrank": 0,
                "unchanged": 0,
                "resolved": 0,
                "affect_rate": 0.0,
            },
        },
        "hygiene": _hygiene(Counter(), Counter(), 0),
        "seeds": [],
        "maps": [],
    }


def _run_configs(
    policy: Policy | None,
    env: VecEnv,
    configs: list[dict],
    *,
    envs: int,
    device: torch.device,
    deterministic: bool,
    scripted: str = "",
    stage_tiles_frac: float = 0.0,
) -> dict:
    if not configs:
        return _empty_mode()

    planned_seeds = [str(c["seed"]) for c in configs]
    maps = [str(c.get("map", "")) for c in configs]
    records: dict[str, dict] = {}
    wins = 0
    stage_hits = 0
    deaths = 0
    kills_total = 0
    returns: dict[str, float] = {}
    ticks_survived: dict[str, int] = {}
    peaks: dict[str, float] = {}
    finals: dict[str, float] = {}
    causes: Counter[str] = Counter()
    hist: Counter[str] = Counter()
    accepted = 0
    total_a = 0
    legal_counts = np.zeros(spaces.NUM_ACTION_TYPES, np.int64)
    legal_n = 0
    attack_wild = 0
    attack_player = 0
    accepted_wild = 0
    accepted_player = 0
    qty_hist: Counter[str] = Counter()
    grew: Counter[str] = Counter()
    shrank: Counter[str] = Counter()
    unchanged: Counter[str] = Counter()

    idx = 0
    while idx < len(configs):
        batch = configs[idx : idx + max(1, int(envs))]
        batch_n = len(batch)
        obs = env.init(batch)
        active = np.ones(batch_n, dtype=bool)
        ep_ret = np.zeros(batch_n, np.float32)
        pending: list[list[tuple[str, int, float]]] = [[] for _ in range(batch_n)]
        while active.any():
            if scripted == "expander":
                actions = np.stack(
                    [scripted_expander_action(obs, i) for i in range(batch_n)]
                ).astype(np.int32)
            else:
                assert policy is not None
                with torch.no_grad():
                    obs_t = obs_to_tensors(obs, device)
                    actions_t, _, _ = policy.act(obs_t, deterministic=deterministic)
                actions = actions_t.cpu().numpy().astype(np.int32)
            mask = np.asarray(obs["action_mask"], dtype=np.int64)
            for i in range(batch_n):
                if not active[i]:
                    continue
                legal_counts += mask[i]
                legal_n += 1
            step = env.step(actions)
            ep_ret += step.rewards
            for i in range(batch_n):
                if not active[i]:
                    continue
                info = step.infos[i]
                at = int(actions[i, 0])
                hist[spaces.ACTION_NAMES[at]] += 1
                total_a += 1
                acc = info.get("actionAccepted", info.get("action_accepted", True))
                if acc:
                    accepted += 1
                tiles = float(info.get("tilesFrac", 0.0))
                pending[i] = _resolve_pending(
                    pending[i], tiles, grew, shrank, unchanged
                )
                if at == spaces.ACTION_ATTACK:
                    tgt = int(actions[i, 1])
                    q = int(actions[i, 3])
                    q_label = (
                        f"{QUANTITY_PCTS[q]}%"
                        if 0 <= q < len(QUANTITY_PCTS)
                        else str(q)
                    )
                    qty_hist[q_label] += 1
                    kind = "wilderness" if tgt == 0 else "player"
                    if kind == "wilderness":
                        attack_wild += 1
                        if acc:
                            accepted_wild += 1
                    else:
                        attack_player += 1
                        if acc:
                            accepted_player += 1
                    pending[i].append((kind, EFFECT_WINDOW, tiles))
                if step.dones[i]:
                    active[i] = False
                    seed = str(batch[i]["seed"])
                    cause, won, stage_ok = classify_outcome(
                        info, float(stage_tiles_frac)
                    )
                    causes[cause] += 1
                    if won:
                        wins += 1
                    if stage_ok:
                        stage_hits += 1
                    if cause == "death":
                        deaths += 1
                    kills_total += int(info.get("kills", 0))
                    ticks_survived[seed] = int(info.get("tick", 0))
                    returns[seed] = float(ep_ret[i])
                    peaks[seed] = float(
                        info.get("peakTilesFrac", info.get("tilesFrac", 0))
                    )
                    finals[seed] = float(info.get("tilesFrac", 0))
                    records[seed] = {
                        "seed": seed,
                        "won": bool(won),
                        "cause": cause,
                    }
            obs = step.obs
        idx += batch_n

    return _summarize(
        planned_seeds=planned_seeds,
        records=records,
        wins=wins,
        stage_hits=stage_hits,
        deaths=deaths,
        kills_total=kills_total,
        returns=returns,
        ticks_survived=ticks_survived,
        peaks=peaks,
        finals=finals,
        causes=causes,
        hist=hist,
        accepted=accepted,
        total_a=total_a,
        legal_counts=legal_counts,
        legal_n=legal_n,
        attack_wild=attack_wild,
        attack_player=attack_player,
        accepted_wild=accepted_wild,
        accepted_player=accepted_player,
        qty_hist=qty_hist,
        grew=grew,
        shrank=shrank,
        unchanged=unchanged,
        maps=maps,
    )


def _print(label: str, r: dict) -> None:
    print(f"--- {label} ---")
    print(f"  episodes : {r['episodes']}")
    print(
        f"  win_rate : {r['win_rate']:.3f}  "
        f"(core 80% FFA only; {r.get('wins', 0)}/{r['episodes']})"
    )
    if r.get("episodes"):
        print(
            f"  wilson95 : [{r['wilson_low']:.3f}, {r['wilson_high']:.3f}] "
            f"z={r.get('wilson_z', WILSON_Z)}"
        )
    print(f"  stage_success_rate : {r['stage_success_rate']:.3f}")
    print(f"  death    : {r['death_rate']:.3f}")
    print(f"  kills/ep : {r['kills_per_ep']:.2f}")
    print(f"  return   : {r['mean_return']:.4f}")
    print(f"  ticks    : {r['mean_ticks']:.0f}")
    print(
        f"  peak/final tiles : {r['mean_peak_tiles']:.4f} / {r['mean_final_tiles']:.4f}"
    )
    print(f"  accept   : {r['accept_rate']:.3f}")
    print(f"  causes   : {r['terminal_causes']}")
    print(f"  actions  : {r['action_hist']}")
    print(f"  legal    : {r['legal_frac']}")
    print(f"  attacks  : {r['attack_targets']}")
    print(f"  quantity : {r['quantity_hist']}")
    print(f"  territory effect : {r['territory_effect']}")
    print(f"  hygiene  : {r.get('hygiene', {})}")
    seeds = r.get("seeds") or []
    if seeds:
        print(f"  seeds    : {len(seeds)} (first={seeds[0]!s})")


def _json_ready(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def build_report(plan: EvalPlan, modes: dict[str, dict]) -> dict:
    sampled = modes.get("sampled") or _empty_mode()
    det = modes.get("deterministic") or _empty_mode()
    return {
        "stage": plan.stage,
        "suite": plan.suite,
        "workers": plan.workers,
        "envs": plan.envs,
        "allowed_actions": list(plan.allowed_actions),
        "config": {
            "map": plan.map,
            "mapSize": plan.map_size,
            "nations": plan.nations,
            "difficulty": plan.difficulty,
            "bots": plan.bots,
            "maxTicks": plan.max_ticks,
            "allowedActions": spaces.encode_allowed_actions(plan.allowed_actions),
        },
        "sampled": sampled,
        "deterministic": det,
        "scripted": plan.scripted,
        "goal_holdout_sampled": GOAL_HOLDOUT_SAMPLED,
        "goal_holdout_det": GOAL_HOLDOUT_DET,
        "goal_ood_sampled": GOAL_OOD_SAMPLED,
    }


def run_eval(
    policy: Policy | None,
    env: VecEnv,
    plan: EvalPlan,
    device: torch.device,
) -> dict:
    """Run sampled/det modes from a resolved plan (no server lifecycle)."""
    modes: dict[str, dict] = {}
    print(
        f"eval stage={plan.stage or 'manual'} suite={plan.suite} "
        f"map={plan.map} size={plan.map_size} bots={plan.bots} "
        f"difficulty={plan.difficulty} workers={plan.workers} "
        f"scripted={plan.scripted or 'off'} actions={plan.allowed_actions}"
    )
    if plan.scripted:
        if plan.det_n:
            modes["deterministic"] = _run_configs(
                policy,
                env,
                plan.det_configs,
                envs=plan.envs,
                device=device,
                deterministic=True,
                scripted=plan.scripted,
                stage_tiles_frac=plan.stage_tiles_frac,
            )
            _print(f"scripted-{plan.scripted}-det", modes["deterministic"])
        else:
            modes["deterministic"] = _empty_mode()
        if plan.sampled_n:
            modes["sampled"] = _run_configs(
                policy,
                env,
                plan.sampled_configs,
                envs=plan.envs,
                device=device,
                deterministic=True,
                scripted=plan.scripted,
                stage_tiles_frac=plan.stage_tiles_frac,
            )
            _print(f"scripted-{plan.scripted}", modes["sampled"])
        else:
            modes["sampled"] = _empty_mode()
        return build_report(plan, modes)

    print(
        f"eval stage={plan.stage or 'manual'} suite={plan.suite} "
        f"map={plan.map} size={plan.map_size} bots={plan.bots} "
        f"difficulty={plan.difficulty} workers={plan.workers} "
        f"scripted=off actions={plan.allowed_actions}"
    )
    if plan.det_n:
        modes["deterministic"] = _run_configs(
            policy,
            env,
            plan.det_configs,
            envs=plan.envs,
            device=device,
            deterministic=True,
            scripted="",
            stage_tiles_frac=plan.stage_tiles_frac,
        )
        _print("deterministic", modes["deterministic"])
    else:
        modes["deterministic"] = _empty_mode()
    if plan.sampled_n:
        modes["sampled"] = _run_configs(
            policy,
            env,
            plan.sampled_configs,
            envs=plan.envs,
            device=device,
            deterministic=False,
            scripted="",
            stage_tiles_frac=plan.stage_tiles_frac,
        )
        _print("sampled", modes["sampled"])
    else:
        modes["sampled"] = _empty_mode()
    return build_report(plan, modes)


def spawn_eval_server(port: int, workers: int) -> EnvServerProcess:
    """Start an official EnvServer with an explicit --workers value."""
    return EnvServerProcess(port, workers=workers)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    try:
        plan = resolve_eval_plan(args)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if not args.scripted and not args.ckpt:
        raise SystemExit("provide --ckpt or --scripted expander")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy: Policy | None = None
    if not args.scripted:
        policy = Policy().to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        policy.eval()

    server = None if args.connect else spawn_eval_server(args.port, plan.workers)
    env = VecEnv("127.0.0.1", args.port)
    try:
        report = run_eval(policy, env, plan, device)
    finally:
        env.close()
        if server is not None:
            server.stop()

    if args.json_out:
        path = Path(args.json_out)
        path.write_text(
            json.dumps(_json_ready(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"json_out={path}")
    return report


if __name__ == "__main__":
    main()
