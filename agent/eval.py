"""Evaluation: run N episodes with deterministic and/or sampled policy.

Training and deployment use the sampled PPO policy. Deterministic
argmax is a robustness / greedy diagnostic, not the definition of
policy quality. `win_rate` is the core 80% FFA objective only;
`stage_success_rate` is a separate competency flag.

Reports win rate, terminal causes, full return, kills, ticks, territory,
action histogram, acceptance rate, plus attack-target / quantity /
legal-action / territory-effect telemetry.

`--scripted expander` runs a deterministic patient-expander/attacker
instead of a checkpoint (used to prove the land-duel env is winnable).

Example:
  .venv/bin/python -m agent.eval --ckpt checkpoints/run/latest.pt \
      --episodes 20 --map Halkidiki --bots 1 --map-size Compact
  .venv/bin/python -m agent.eval --scripted expander --episodes 20 \
      --map Halkidiki --map-size Compact --bots 1 --max-ticks 8000
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from . import spaces
from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors
from .ppo import classify_outcome

QUANTITY_PCTS = (5, 15, 30, 50, 80)
# Wilderness capture rate is border-limited, not troop-limited. 50% leaves
# enough standing army to keep a live attack without starving income.
# Player attacks wait while retreat_all is legal so we do not restart combat.
EXPAND_QTY = 3
PLAYER_QTY = 3
EFFECT_WINDOW = 3
TILE_EPS = 1e-4
REGION_GRID = spaces.REGION_GRID


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="")
    p.add_argument(
        "--scripted",
        choices=["", "expander"],
        default="",
        help="deterministic scripted policy (no checkpoint required)",
    )
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--connect", action="store_true")
    p.add_argument("--map", type=str, default="Halkidiki")
    p.add_argument("--map-size", type=str, default="Compact")
    p.add_argument("--nations", type=str, default="disabled")
    p.add_argument("--difficulty", type=str, default="Easy")
    p.add_argument("--bots", type=int, default=1)
    p.add_argument("--max-ticks", type=int, default=8000)
    p.add_argument("--shaping", type=float, default=0.0)
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
        default=",".join(spaces.EARLY_ACTIONS),
        help="comma-separated action names passed as allowedActions",
    )
    p.add_argument(
        "--mode",
        choices=["both", "deterministic", "sampled"],
        default="both",
        help="sampled is the deployed policy; deterministic is a greedy diagnostic",
    )
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


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


def _run_mode(
    policy: Policy | None,
    env: VecEnv,
    args: argparse.Namespace,
    nations: int | str,
    device: torch.device,
    deterministic: bool,
    scripted: str = "",
) -> dict:
    remaining = args.episodes
    wins = 0
    stage_hits = 0
    deaths = 0
    kills_total = 0
    finished = 0
    returns: list[float] = []
    ticks_survived: list[int] = []
    peaks: list[float] = []
    finals: list[float] = []
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
    allowed = spaces.encode_allowed_actions(
        [a.strip() for a in args.allowed_actions.split(",") if a.strip()]
    )

    while remaining > 0:
        batch_n = min(args.envs, remaining)
        cfgs = [
            {
                "map": args.map,
                "mapSize": args.map_size,
                "nations": nations,
                "difficulty": args.difficulty,
                "bots": args.bots,
                "seed": f"eval-{('det' if deterministic else 'samp')}-{args.episodes - remaining}-{i}",
                "maxTicks": args.max_ticks,
                "decisionInterval": 10,
                "shaping": args.shaping,
                "allowedActions": allowed,
                **(
                    {"winTilesFrac": args.win_tiles_frac}
                    if args.win_tiles_frac > 0
                    else {}
                ),
            }
            for i in range(batch_n)
        ]
        obs = env.init(cfgs)
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
                    q_label = f"{QUANTITY_PCTS[q]}%" if 0 <= q < len(QUANTITY_PCTS) else str(q)
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
                    finished += 1
                    cause, won, stage_ok = classify_outcome(
                        info, float(args.stage_tiles_frac)
                    )
                    causes[cause] += 1
                    if won:
                        wins += 1
                    if stage_ok:
                        stage_hits += 1
                    if cause == "death":
                        deaths += 1
                    kills_total += int(info.get("kills", 0))
                    ticks_survived.append(int(info.get("tick", 0)))
                    returns.append(float(ep_ret[i]))
                    peaks.append(float(info.get("peakTilesFrac", info.get("tilesFrac", 0))))
                    finals.append(float(info.get("tilesFrac", 0)))
            obs = step.obs
        remaining -= batch_n

    n = max(1, finished)
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

    return {
        "episodes": finished,
        "win_rate": wins / n,
        "stage_success_rate": stage_hits / n,
        "death_rate": deaths / n,
        "kills_per_ep": kills_total / n,
        "mean_return": float(np.mean(returns)) if returns else 0.0,
        "mean_ticks": float(np.mean(ticks_survived)) if ticks_survived else 0.0,
        "mean_peak_tiles": float(np.mean(peaks)) if peaks else 0.0,
        "mean_final_tiles": float(np.mean(finals)) if finals else 0.0,
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
    }


def _print(label: str, r: dict) -> None:
    print(f"--- {label} ---")
    print(f"  episodes : {r['episodes']}")
    print(f"  win_rate : {r['win_rate']:.3f}  (core 80% FFA only)")
    print(f"  stage_success_rate : {r['stage_success_rate']:.3f}")
    print(f"  death    : {r['death_rate']:.3f}")
    print(f"  kills/ep : {r['kills_per_ep']:.2f}")
    print(f"  return   : {r['mean_return']:.4f}")
    print(f"  ticks    : {r['mean_ticks']:.0f}")
    print(f"  peak/final tiles : {r['mean_peak_tiles']:.4f} / {r['mean_final_tiles']:.4f}")
    print(f"  accept   : {r['accept_rate']:.3f}")
    print(f"  causes   : {r['terminal_causes']}")
    print(f"  actions  : {r['action_hist']}")
    print(f"  legal    : {r['legal_frac']}")
    print(f"  attacks  : {r['attack_targets']}")
    print(f"  quantity : {r['quantity_hist']}")
    print(f"  territory effect : {r['territory_effect']}")


def main() -> None:
    args = parse_args()
    if not args.scripted and not args.ckpt:
        raise SystemExit("provide --ckpt or --scripted expander")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy: Policy | None = None
    if not args.scripted:
        policy = Policy().to(device)
        ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
        policy.load_state_dict(ckpt["model"])
        policy.eval()

    nations: int | str = args.nations
    if nations not in ("default", "disabled"):
        nations = int(nations)

    server = None if args.connect else EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)
    try:
        if args.scripted:
            modes = [(f"scripted-{args.scripted}", True)]
        else:
            modes = (
                [("deterministic", True), ("sampled", False)]
                if args.mode == "both"
                else [(args.mode, args.mode == "deterministic")]
            )
        print(
            f"eval map={args.map} size={args.map_size} bots={args.bots} "
            f"difficulty={args.difficulty} scripted={args.scripted or 'off'}"
        )
        for label, det in modes:
            r = _run_mode(
                policy, env, args, nations, device, det, scripted=args.scripted
            )
            _print(label, r)
    finally:
        env.close()
        if server is not None:
            server.stop()


if __name__ == "__main__":
    main()
