"""Behavior trace for a trained checkpoint.

Two phases, both against a dedicated env server (separate port; does not
touch any running training shard):

  Phase A  run the policy with sampled actions (as in training) and report the
           action-type histogram plus, for every ATTACK-wilderness decision,
           whether territory actually grew afterwards (i.e. the attack fired).
  Phase B  controlled probe: force ATTACK-wilderness at specific regions and
           measure which regions actually expand the agent. This isolates the
           region-gate (`regionHasAdjacentWilderness`) from policy learning.

Example:
  .venv/bin/python -m agent.trace --ckpt checkpoints/run6/latest.pt --port 9500
"""

from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch

from . import spaces
from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors

ACTION_NAMES = [
    "noop", "spawn", "attack", "retreat", "build",
    "boat", "ally", "break", "embargo",
]
GRID = spaces.REGION_GRID
ATTACK = spaces.ACTION_ATTACK
SPAWN = spaces.ACTION_SPAWN
NOOP = spaces.ACTION_NOOP


def tribes_cfg(seed: str) -> dict:
    return {
        "map": "FourIslands",
        "mapSize": "Normal",
        "nations": "disabled",
        "difficulty": "Easy",
        "bots": 20,
        "seed": seed,
        "maxTicks": 6000,
        "decisionInterval": 10,
        "shaping": 5.0,
    }


def own_regions(obs: dict, i: int) -> np.ndarray:
    return np.flatnonzero(obs["build_regions"][i])


def region_troops(obs: dict, i: int) -> float:
    # global[3] = log1p(troops)/15  ->  troops
    return float(np.expm1(obs["global"][i][3] * 15.0))


def bot_action(obs: dict, i: int) -> np.ndarray:
    """Patient expander: spawn, then NOOP to build a real army, and only
    commit to a wilderness attack once troops are high (send 80%)."""
    spawned = obs["build_regions"][i].any()
    if not spawned:
        sr = np.flatnonzero(obs["spawn_regions"][i])
        return np.array([SPAWN, 0, int(sr[0]) if sr.size else 0, 0, 0], dtype=np.int32)
    troops = region_troops(obs, i)
    if obs["action_mask"][i][ATTACK] and obs["target_mask"][i][0] and troops > 3000:
        own = own_regions(obs, i)
        region = int(own[0]) if own.size else 0
        return np.array([ATTACK, 0, region, 4, 0], dtype=np.int32)  # 80%
    return np.array([NOOP, 0, 0, 0, 0], dtype=np.int32)


def phase_a(policy: Policy, env: VecEnv, device: torch.device, decisions: int, bot: bool) -> None:
    k = env.k
    counts: Counter[int] = Counter()
    attack_wild = 0
    # per-env pending wilderness-attack checks: [steps_left, tilesFrac_at_attack]
    pending: list[list] = [[] for _ in range(k)]
    wild_fired = 0
    wild_fizzled = 0
    ep_returns = np.zeros(k)
    episodes = 0
    deaths = 0
    spawned_eps = 0
    death_ticks: list[int] = []
    final_tiles: list[float] = []

    for _ in range(decisions):
        if bot:
            actions = np.stack([bot_action(obs_global[0], i) for i in range(k)]).astype(np.int32)
        else:
            obs_t = obs_to_tensors(obs_global[0], device)
            with torch.no_grad():
                acts, _, _ = policy(obs_t)
            actions = acts.cpu().numpy().astype(np.int32)
        step = env.step(actions)
        for i in range(k):
            a = actions[i]
            at = int(a[0])
            counts[at] += 1
            tf = float(step.infos[i]["tilesFrac"])
            # resolve any pending wilderness-attack checks for this env
            still = []
            for steps_left, tf0 in pending[i]:
                if tf > tf0 + 1e-4:
                    wild_fired += 1
                elif steps_left <= 1:
                    wild_fizzled += 1
                else:
                    still.append((steps_left - 1, tf0))
            pending[i] = still
            if at == ATTACK and int(a[1]) == 0:
                attack_wild += 1
                pending[i].append((3, tf))
            ep_returns[i] += float(step.rewards[i])
            if step.dones[i]:
                episodes += 1
                if step.infos[i].get("spawned"):
                    spawned_eps += 1
                if step.infos[i].get("dead"):
                    deaths += 1
                    death_ticks.append(int(step.infos[i].get("tick", 0)))
                final_tiles.append(float(step.infos[i]["tilesFrac"]))
                ep_returns[i] = 0.0
                obs_global[0] = spliced(obs_global[0], env, i)
        obs_global[0] = to_writable(step.obs)

    total = sum(counts.values())
    label = "hardcoded expand-bot" if bot else "policy (sampled)"
    print(f"\n=== Phase A: action distribution ({label}, tribes) ===")
    for idx in range(spaces.NUM_ACTION_TYPES):
        n = counts.get(idx, 0)
        print(f"  {ACTION_NAMES[idx]:8s} {n:6d}  {100.0 * n / max(1, total):5.1f}%")
    print(f"  decisions={total} episodes={episodes} spawned={spawned_eps} deaths={deaths}")
    if death_ticks:
        print(f"  death tick: mean {np.mean(death_ticks):.0f}  median {np.median(death_ticks):.0f}  (6000=timeout)")
    if final_tiles:
        print(f"  final tilesFrac: mean {np.mean(final_tiles):.4f}  max {max(final_tiles):.4f}")
    print(f"  wilderness attacks: {attack_wild}  fired {wild_fired}  fizzled {wild_fizzled}")


# module-level scratch so phase_a can splice resets without threading args
obs_global: list[dict] = [{}]


def to_writable(obs: dict) -> dict:
    # wire buffers are read-only frombuffer views; copy so we can splice resets
    return {k: np.array(v, copy=True) for k, v in obs.items()}


def spliced(obs: dict, env: VecEnv, i: int) -> dict:
    fresh = env.reset(i)
    for key in obs:
        obs[key][i] = fresh[key][0]
    return obs


def phase_b(env: VecEnv, trials: int) -> None:
    k = env.k
    results = {"own": [0, 0], "adj": [0, 0], "far": [0, 0]}  # [fired, total]
    pending: list[list] = [[] for _ in range(k)]  # (steps_left, tf0, kind)
    for _ in range(trials):
        # choose a probe action per env
        acts = np.zeros((k, 5), dtype=np.int32)
        for i in range(k):
            own = own_regions(obs_global[0], i)
            troops = region_troops(obs_global[0], i)
            spawned = own.size > 0
            if not spawned or troops < 25:
                # drive toward a spawn / let troops regen
                if not spawned:
                    sr = np.flatnonzero(obs_global[0]["spawn_regions"][i])
                    acts[i] = [SPAWN, 0, int(sr[0]) if sr.size else 0, 0, 0]
                else:
                    acts[i] = [NOOP, 0, 0, 0, 0]
                continue
            r = int(own[0])
            kind = "own"
            roll = np.random.rand()
            if roll < 0.34:
                # adjacent region (4-neighbour on the GRID x GRID coarse grid)
                cands = []
                x, y = r % GRID, r // GRID
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < GRID and 0 <= ny < GRID:
                        cands.append(ny * GRID + nx)
                if cands:
                    r = int(np.random.choice(cands))
                    kind = "adj"
            elif roll < 0.78:
                # far region: mask-legal (has wilderness) but not own/adjacent
                sr = np.flatnonzero(obs_global[0]["spawn_regions"][i])
                x, y = r % GRID, r // GRID
                adj = {r, x+1+y*GRID if x+1 < GRID else -1, x-1+y*GRID if x-1 >= 0 else -1,
                       x+(y+1)*GRID if y+1 < GRID else -1, x+(y-1)*GRID if y-1 >= 0 else -1}
                far = [int(v) for v in sr if int(v) not in adj]
                if far:
                    r = int(np.random.choice(far))
                    kind = "far"
            acts[i] = [ATTACK, 0, r, 2, 0]  # quantity idx 2 = 0.3 of troops
            pending[i].append((3, float(obs_global[0]["global"][i][2]), kind))
        step = env.step(acts)
        for i in range(k):
            tf = float(step.infos[i]["tilesFrac"])
            still = []
            for steps_left, tf0, kind in pending[i]:
                if tf > tf0 + 1e-4:
                    results[kind][0] += 1
                    results[kind][1] += 1
                elif steps_left <= 1:
                    results[kind][1] += 1
                else:
                    still.append((steps_left - 1, tf0, kind))
            pending[i] = still
            if step.dones[i]:
                obs_global[0] = spliced(obs_global[0], env, i)
        obs_global[0] = to_writable(step.obs)

    print("\n=== Phase B: forced ATTACK->wilderness, by region choice ===")
    print("  (own = my region, adj = 4-neighbour, far = mask-legal wilderness far away)")
    for kind in ("own", "adj", "far"):
        fired, total = results[kind]
        pct = 100.0 * fired / max(1, total)
        print(f"  {kind:4s}: fired {fired:4d}/{total:4d}  = {pct:5.1f}%")


def phase_c(env: VecEnv, target_troops: float, qty_idx: int = 3) -> None:
    """Decisive probe: build a real army, then attack wilderness once and
    measure EXACT tiles conquered (tilesFrac * landTiles) over the next few
    decisions. Tells us whether the expansion mechanic works when the agent
    commits enough troops."""
    k = env.k
    land = np.expm1(obs_global[0]["global"][:, 7] * 15.0)  # per-env land tiles
    state = ["spawn"] * k
    troops_at = [0.0] * k
    base_tiles = [0.0] * k
    mcount = [0] * k
    results: list[tuple[float, float]] = []  # (troops committed, tiles gained)
    qty_frac = [0.05, 0.15, 0.3, 0.5, 0.8][qty_idx]

    for _ in range(600):
        acts = np.zeros((k, 5), dtype=np.int32)
        for i in range(k):
            if state[i] == "done":
                acts[i] = [NOOP, 0, 0, 0, 0]
                continue
            spawned = obs_global[0]["build_regions"][i].any()
            troops = region_troops(obs_global[0], i)
            if not spawned:
                sr = np.flatnonzero(obs_global[0]["spawn_regions"][i])
                acts[i] = [SPAWN, 0, int(sr[0]) if sr.size else 0, 0, 0]
                continue
            if state[i] == "spawn":
                state[i] = "build"
            if state[i] == "build":
                if troops >= target_troops:
                    own = own_regions(obs_global[0], i)
                    reg = int(own[0]) if own.size else 0
                    acts[i] = [ATTACK, 0, reg, qty_idx, 0]
                    troops_at[i] = troops * qty_frac
                    base_tiles[i] = float(obs_global[0]["global"][i][2]) * land[i]
                    state[i] = "measure"
                    mcount[i] = 0
                else:
                    acts[i] = [NOOP, 0, 0, 0, 0]
            elif state[i] == "measure":
                acts[i] = [NOOP, 0, 0, 0, 0]
        step = env.step(acts)
        for i in range(k):
            if state[i] == "measure":
                mcount[i] += 1
                tiles = float(step.infos[i]["tilesFrac"]) * land[i]
                if mcount[i] >= 4:
                    results.append((troops_at[i], tiles - base_tiles[i]))
                    state[i] = "done"
            if step.dones[i]:
                state[i] = "done"
        obs_global[0] = to_writable(step.obs)
        if all(s == "done" for s in state):
            break

    print(f"\n=== Phase C: wilderness attack with a real army (target {target_troops:.0f} troops, send {qty_frac:.0%}) ===")
    if not results:
        print("  no attacks completed")
        return
    for tr, gained in results:
        print(f"  committed {tr:7.0f} troops -> net {gained:+7.1f} tiles")
    arr = np.array([g for _, g in results])
    print(f"  mean net tiles gained: {arr.mean():+.1f}   (attacks that gained>0: {(arr>0).sum()}/{len(arr)})")


def phase_d(env: VecEnv, decisions: int) -> None:
    """Ground-truth trajectory for the patient expander on env 0: troops,
    exact tiles, and whether it borders wilderness, every few decisions."""
    k = env.k
    land = np.expm1(obs_global[0]["global"][:, 7] * 15.0)
    print("\n=== Phase D: patient-expander trajectory (env 0) ===")
    print("  dec | action  | troops | tiles | bordersWild | note")
    prev_tiles = 0.0
    for t in range(decisions):
        acts = np.stack([bot_action(obs_global[0], i) for i in range(k)]).astype(np.int32)
        pre_troops = region_troops(obs_global[0], 0)
        borders_wild = bool(obs_global[0]["target_mask"][0][0])
        step = env.step(acts)
        tiles = float(step.infos[0]["tilesFrac"]) * land[0]
        a = acts[0]
        note = ""
        if tiles > prev_tiles + 0.5:
            note = f"+{tiles - prev_tiles:.0f} tiles"
        if tiles < prev_tiles - 0.5:
            note = f"{tiles - prev_tiles:.0f} tiles (lost)"
        if t % 15 == 0 or note:
            print(f"  {t:4d} | {ACTION_NAMES[int(a[0])]:7s} | {pre_troops:6.0f} | {tiles:5.0f} | {str(borders_wild):5s} | {note}")
        prev_tiles = tiles
        for i in range(k):
            if step.dones[i]:
                obs_global[0] = spliced(obs_global[0], env, i)
                if i == 0:
                    print(f"  ---- env0 done at dec {t} (dead={step.infos[0].get('dead')}) ----")
                    prev_tiles = 0.0
        obs_global[0] = to_writable(step.obs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--port", type=int, default=9500)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--trials", type=int, default=120)
    ap.add_argument("--probe-troops", type=float, default=400.0)
    ap.add_argument("--phases", type=str, default="abc", help="which phases to run")
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    server = EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)
    try:
        if "a" in args.phases:
            obs = env.init([tribes_cfg(f"trace-a-{i}") for i in range(args.envs)])
            obs_global[0] = to_writable(obs)
            phase_a(policy, env, device, args.decisions, bot=False)

            obs = env.init([tribes_cfg(f"trace-bot-{i}") for i in range(args.envs)])
            obs_global[0] = to_writable(obs)
            phase_a(policy, env, device, args.decisions, bot=True)

        if "b" in args.phases:
            obs = env.init([tribes_cfg(f"trace-b-{i}") for i in range(args.envs)])
            obs_global[0] = to_writable(obs)
            phase_b(env, args.trials)

        if "c" in args.phases:
            obs = env.init([tribes_cfg(f"trace-c-{i}") for i in range(args.envs)])
            obs_global[0] = to_writable(obs)
            phase_c(env, args.probe_troops)

        if "d" in args.phases:
            obs = env.init([tribes_cfg(f"trace-d-{i}") for i in range(args.envs)])
            obs_global[0] = to_writable(obs)
            phase_d(env, args.decisions)
    finally:
        env.close()
        server.stop()


if __name__ == "__main__":
    main()
