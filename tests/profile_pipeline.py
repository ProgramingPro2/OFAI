"""Micro-profile of the training hot loop: env stepping vs policy forward
vs PPO update, plus server x env layouts at a fixed total env count.

Run from workspace root (uses unused ports; keep away from an active
v2run1 range 8980-8995):

  .venv/bin/python tests/profile_pipeline.py
  .venv/bin/python tests/profile_pipeline.py --stage nations-easy-small
  .venv/bin/python tests/profile_pipeline.py --layouts 1x8,2x4,4x2 --iters 12
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import spaces
from agent.curriculum import STAGES
from agent.env_client import EnvServerProcess, ShardedVecEnv, VecEnv
from agent.model import Policy, obs_to_tensors
from agent.ppo import PPOConfig, PPOTrainer, RolloutBuffer

# Far from v2run1 (8980-8995) and leftover 8790 watch/debug servers.
DEFAULT_PORT = 9120


def _stage_cfg(name: str) -> dict:
    stage = next(s for s in STAGES if s.name == name)
    return {
        "map": stage.map,
        "mapSize": stage.map_size,
        "nations": stage.nations,
        "difficulty": stage.difficulty,
        "bots": stage.bots,
        "maxTicks": stage.max_ticks,
        "decisionInterval": stage.decision_interval,
        "shaping": 0,
        "allowedActions": list(stage.allowed_actions),
    }


def _median_ms(samples: list[float]) -> float:
    return statistics.median(samples) * 1000.0


def _masked_actions(obs: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    acts = np.zeros((k, 5), dtype=np.int32)
    for i in range(k):
        legal = np.flatnonzero(obs["action_mask"][i])
        acts[i, 0] = int(rng.choice(legal)) if legal.size else 0
        acts[i, 1] = int(rng.integers(0, spaces.NUM_PLAYER_SLOTS))
        acts[i, 2] = int(rng.integers(0, spaces.NUM_REGIONS))
        acts[i, 3] = int(rng.integers(0, spaces.NUM_QUANTITIES))
        acts[i, 4] = int(rng.integers(0, spaces.NUM_UNIT_TYPES))
    return acts


def _env_step_times(
    env, k: int, iters: int, warmup: int, obs: dict | None = None
) -> list[float]:
    rng = np.random.default_rng(0)
    batch_obs = obs
    if batch_obs is None:
        actions = np.zeros((k, 5), dtype=np.int32)
    for _ in range(warmup):
        actions = (
            _masked_actions(batch_obs, k, rng)
            if batch_obs is not None
            else np.zeros((k, 5), dtype=np.int32)
        )
        stepped = env.step(actions)
        batch_obs = stepped.obs
    times: list[float] = []
    for _ in range(iters):
        actions = (
            _masked_actions(batch_obs, k, rng)
            if batch_obs is not None
            else np.zeros((k, 5), dtype=np.int32)
        )
        t0 = time.perf_counter()
        stepped = env.step(actions)
        times.append(time.perf_counter() - t0)
        batch_obs = stepped.obs
    return times


def _parse_layouts(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        servers, envs = part.strip().lower().split("x")
        out.append((int(servers), int(envs)))
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="OFAI training pipeline profiler")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--stage", type=str, default="duel", choices=[s.name for s in STAGES])
    p.add_argument("--k", type=int, default=8, help="envs on the single-server benches")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--layouts",
        type=str,
        default="",
        help="comma layouts at equal total envs, e.g. 1x8,2x4,4x2",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="decision steps after init before timing (mid-episode cost)",
    )
    args = p.parse_args()

    cfg = _stage_cfg(args.stage)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(
        f"stage={args.stage} map={cfg['map']} size={cfg['mapSize']} "
        f"nations={cfg['nations']} bots={cfg['bots']} port={args.port} device={device}"
    )

    server = EnvServerProcess(args.port)
    try:
        env = VecEnv("127.0.0.1", args.port)
        obs = env.init([{**cfg, "seed": f"prof-{i}"} for i in range(args.k)])
        policy = Policy().to(device).eval()
        if args.burnin:
            _env_step_times(env, args.k, args.burnin, 0, obs)
            print(f"burn-in {args.burnin} decision steps")

        step_t = _env_step_times(env, args.k, args.iters, args.warmup, obs)
        step_ms = _median_ms(step_t)
        print(
            f"env.step (K={args.k})         : {step_ms:7.1f} ms median  "
            f"({args.k * 1000 / step_ms:5.1f} env-steps/s)"
        )

        obs_t = obs_to_tensors(obs, device)
        with torch.no_grad():
            for _ in range(args.warmup):
                policy(obs_t)
        if device.type == "cuda":
            torch.cuda.synchronize()
        fwd: list[float] = []
        with torch.no_grad():
            for _ in range(args.iters):
                t0 = time.perf_counter()
                policy(obs_t)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                fwd.append(time.perf_counter() - t0)
        print(f"policy forward (K={args.k})   : {_median_ms(fwd):7.1f} ms median")

        xfer: list[float] = []
        for _ in range(args.iters):
            t0 = time.perf_counter()
            obs_to_tensors(obs, device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            xfer.append(time.perf_counter() - t0)
        print(f"obs_to_tensors (K={args.k})   : {_median_ms(xfer):7.1f} ms median")

        spatial_bytes = obs["spatial"].nbytes
        total = sum(v.nbytes for v in obs.values())
        print(
            f"obs payload/env: {total / args.k / 1024:.0f} KiB "
            f"(spatial {spatial_bytes / args.k / 1024:.0f} KiB)"
        )

        for kk in (1, 2, 4):
            kobs = env.init([{**cfg, "seed": f"prof-k{kk}-{i}"} for i in range(kk)])
            times = _env_step_times(env, kk, max(8, args.iters // 2), 2, kobs)
            ms = _median_ms(times)
            print(
                f"env.step (K={kk})         : {ms:7.1f} ms median  "
                f"({kk * 1000 / ms:5.1f} env-steps/s)"
            )

        # Isolated PPO update on dummy rollout (no extra EnvServers).
        rollout_len = 32
        k_upd = min(args.k, 8)
        trainer = PPOTrainer(
            Policy().to(device),
            PPOConfig(
                rollout_len=rollout_len,
                num_minibatches=4,
                update_epochs=2,
                device=str(device),
            ),
        )
        buf = RolloutBuffer(rollout_len, k_upd)
        dummy = {key: np.repeat(obs[key][:1], k_upd, axis=0) for key in obs}
        if dummy["spatial"].shape[0] != k_upd:
            dummy = {
                key: np.resize(obs[key], (k_upd,) + obs[key].shape[1:])
                for key in obs
            }
        for t in range(rollout_len):
            buf.obs["spatial"][t] = dummy["spatial"].astype(np.float16)
            for key in buf.obs:
                if key != "spatial":
                    buf.obs[key][t] = dummy[key]
        buf.actions[:] = 0
        buf.logprobs[:] = 0
        buf.values[:] = 0
        buf.rewards[:] = 0
        buf.dones[:] = 0
        last = dummy
        # warmup
        trainer.update(buf, last)
        upd: list[float] = []
        for _ in range(max(5, args.iters // 2)):
            t0 = time.perf_counter()
            trainer.update(buf, last)
            if device.type == "cuda":
                torch.cuda.synchronize()
            upd.append(time.perf_counter() - t0)
        print(
            f"PPO update (T={rollout_len} K={k_upd}): {_median_ms(upd):7.1f} ms median"
        )

        env.close()
    finally:
        server.stop()

    layouts = _parse_layouts(args.layouts) if args.layouts else []
    if not layouts:
        return
    totals = {s * e for s, e in layouts}
    if len(totals) != 1:
        raise SystemExit("layouts must share the same total env count")
    print(f"\nlayout compare total_envs={next(iter(totals))} stage={args.stage}")
    base = args.port + 20
    for servers, envs_each in layouts:
        sharded = ShardedVecEnv(servers, base, envs_each)
        try:
            n = servers * envs_each
            lobs = sharded.init(
                [{**cfg, "seed": f"lay-{servers}x{envs_each}-{i}"} for i in range(n)]
            )
            if args.burnin:
                _env_step_times(sharded, n, args.burnin, 0, lobs)
            times = _env_step_times(sharded, n, args.iters, args.warmup, lobs)
            ms = _median_ms(times)
            print(
                f"  {servers}x{envs_each}              : {ms:7.1f} ms median  "
                f"({n * 1000 / ms:5.1f} env-steps/s)"
            )
        finally:
            sharded.close()
        base += servers


if __name__ == "__main__":
    main()
