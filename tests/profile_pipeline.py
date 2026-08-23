"""Structured profiler for the training hot loop and farm topologies.

Reports per-stage throughput, latency percentiles, resets, shard failures,
and server-side simulation / mask-observation / serialization time.

Representative baseline (duel, peninsula, nations; 16x8 vs 32x4 vs 64x2):

  .venv/bin/python tests/profile_pipeline.py --baseline
  .venv/bin/python tests/profile_pipeline.py --baseline --quick
  .venv/bin/python tests/profile_pipeline.py --stage nations-easy-small
  .venv/bin/python tests/profile_pipeline.py --layouts 16x8,32x4,64x2 --iters 6

Uses unused ports; keep away from an active v2run1 range 8980-8995.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import spaces
from agent.curriculum import STAGES
from agent.env_client import (
    EnvServerProcess,
    RolloutInterrupted,
    ShardedVecEnv,
    VecEnv,
)
from agent.model import Policy, obs_to_tensors
from agent.ppo import PPOConfig, PPOTrainer, RolloutBuffer

# Far from v2run1 (8980-8995) and leftover 8790 watch/debug servers.
DEFAULT_PORT = 9120
REPRESENTATIVE_STAGES = (
    "duel",
    "nations-easy-peninsula",
    "nations-easy-small",
)
REPRESENTATIVE_LAYOUTS = ((16, 8), (32, 4), (64, 2))
QUICK_LAYOUTS = ((1, 4), (2, 2), (4, 1))
PROFILE_KEYS = ("simMs", "maskObsMs", "serializeMs", "resetMs", "resets")


def stage_cfg(name: str) -> dict:
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


def parse_layouts(spec: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        servers, envs = part.lower().split("x")
        out.append((int(servers), int(envs)))
    return out


def format_layout(servers: int, envs_each: int) -> str:
    return f"{servers}x{envs_each}"


def percentile(samples: list[float], p: float) -> float:
    if not samples:
        return 0.0
    if len(samples) == 1:
        return float(samples[0])
    ordered = sorted(samples)
    idx = min(len(ordered) - 1, max(0, round((p / 100.0) * (len(ordered) - 1))))
    return float(ordered[idx])


def latency_stats(samples: list[float]) -> dict[str, float]:
    if not samples:
        return {"n": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "n": float(len(samples)),
        "mean_ms": statistics.fmean(samples) * 1000.0,
        "p50_ms": percentile(samples, 50) * 1000.0,
        "p95_ms": percentile(samples, 95) * 1000.0,
        "p99_ms": percentile(samples, 99) * 1000.0,
    }


def empty_profile_acc() -> dict[str, float]:
    return {key: 0.0 for key in PROFILE_KEYS}


def accumulate_profile(acc: dict[str, float], profile: dict | None) -> None:
    if not profile:
        return
    for key in PROFILE_KEYS:
        acc[key] += float(profile.get(key) or 0)


def mean_profile(acc: dict[str, float], n: int) -> dict[str, float]:
    denom = max(n, 1)
    return {
        "sim_ms": acc["simMs"] / denom,
        "mask_obs_ms": acc["maskObsMs"] / denom,
        "serialize_ms": acc["serializeMs"] / denom,
        "reset_ms": acc["resetMs"] / denom,
        "resets": acc["resets"],
    }


def masked_actions(obs: dict, k: int, rng: np.random.Generator) -> np.ndarray:
    acts = np.zeros((k, 5), dtype=np.int32)
    for i in range(k):
        legal = np.flatnonzero(obs["action_mask"][i])
        acts[i, 0] = int(rng.choice(legal)) if legal.size else 0
        acts[i, 1] = int(rng.integers(0, spaces.NUM_PLAYER_SLOTS))
        acts[i, 2] = int(rng.integers(0, spaces.NUM_REGIONS))
        acts[i, 3] = int(rng.integers(0, spaces.NUM_QUANTITIES))
        acts[i, 4] = int(rng.integers(0, spaces.NUM_UNIT_TYPES))
    return acts


def step_env(
    env,
    k: int,
    iters: int,
    warmup: int,
    obs: dict | None = None,
) -> dict[str, Any]:
    rng = np.random.default_rng(0)
    batch_obs = obs
    profile_acc = empty_profile_acc()
    resets = 0
    shard_failures = 0
    if batch_obs is None:
        actions = np.zeros((k, 5), dtype=np.int32)
    for _ in range(warmup):
        actions = (
            masked_actions(batch_obs, k, rng)
            if batch_obs is not None
            else np.zeros((k, 5), dtype=np.int32)
        )
        try:
            stepped = env.step(actions)
        except RolloutInterrupted:
            shard_failures += 1
            raise
        batch_obs = stepped.obs
        resets += int(np.asarray(stepped.dones).sum())
        accumulate_profile(profile_acc, getattr(stepped, "profile", None))
    times: list[float] = []
    encode_times: list[float] = []
    timed_resets = 0
    timed_acc = empty_profile_acc()
    for _ in range(iters):
        t_enc = time.perf_counter()
        actions = (
            masked_actions(batch_obs, k, rng)
            if batch_obs is not None
            else np.zeros((k, 5), dtype=np.int32)
        )
        encode_times.append(time.perf_counter() - t_enc)
        t0 = time.perf_counter()
        try:
            stepped = env.step(actions)
        except RolloutInterrupted:
            shard_failures += 1
            raise
        times.append(time.perf_counter() - t0)
        batch_obs = stepped.obs
        timed_resets += int(np.asarray(stepped.dones).sum())
        accumulate_profile(timed_acc, getattr(stepped, "profile", None))
    stats = latency_stats(times)
    env_steps = k * max(len(times), 1)
    p50 = stats["p50_ms"] if stats["p50_ms"] > 0 else 1.0
    return {
        "latency": stats,
        "env_steps_per_s": env_steps * 1000.0 / p50,
        "resets": float(resets + timed_resets),
        "shard_failures": shard_failures,
        "client_action_build_ms": statistics.fmean(encode_times) * 1000.0
        if encode_times
        else 0.0,
        **mean_profile(timed_acc, len(times)),
    }


def print_step_report(label: str, k: int, report: dict[str, Any]) -> None:
    lat = report["latency"]
    print(
        f"{label:<28}: {lat['p50_ms']:7.1f} ms p50  "
        f"{lat['p95_ms']:7.1f} ms p95  "
        f"{report['env_steps_per_s']:6.1f} env-steps/s  "
        f"resets={int(report['resets'])} fail={report['shard_failures']}"
    )
    print(
        f"{'':28}  sim={report['sim_ms']:6.1f} ms  "
        f"mask/obs={report['mask_obs_ms']:6.1f} ms  "
        f"serialize={report['serialize_ms']:6.1f} ms  "
        f"reset={report['reset_ms']:6.1f} ms  "
        f"K={k}"
    )


def run_single_server(
    *,
    stage: str,
    port: int,
    k: int,
    iters: int,
    warmup: int,
    burnin: int,
    device: torch.device,
    include_policy: bool,
    workers: int = 0,
) -> dict[str, Any]:
    cfg = stage_cfg(stage)
    print(
        f"stage={stage} map={cfg['map']} size={cfg['mapSize']} "
        f"nations={cfg['nations']} bots={cfg['bots']} port={port} device={device}"
    )
    server = EnvServerProcess(port, workers=workers)
    out: dict[str, Any] = {
        "stage": stage,
        "topology": format_layout(1, k),
        "config": cfg,
    }
    try:
        env = VecEnv("127.0.0.1", port)
        obs = env.init([{**cfg, "seed": f"prof-{i}"} for i in range(k)])
        if burnin:
            step_env(env, k, burnin, 0, obs)
            print(f"burn-in {burnin} decision steps")
        report = step_env(env, k, iters, warmup, obs)
        print_step_report(f"env.step (K={k})", k, report)
        out["env_step"] = report

        if include_policy:
            policy = Policy().to(device).eval()
            obs_t = obs_to_tensors(obs, device)
            with torch.no_grad():
                for _ in range(warmup):
                    policy(obs_t)
            if device.type == "cuda":
                torch.cuda.synchronize()
            fwd: list[float] = []
            with torch.no_grad():
                for _ in range(iters):
                    t0 = time.perf_counter()
                    policy(obs_t)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    fwd.append(time.perf_counter() - t0)
            print(f"policy forward (K={k})   : {percentile(fwd, 50) * 1000:7.1f} ms p50")
            out["policy_forward"] = latency_stats(fwd)

            xfer: list[float] = []
            for _ in range(iters):
                t0 = time.perf_counter()
                obs_to_tensors(obs, device)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                xfer.append(time.perf_counter() - t0)
            print(f"obs_to_tensors (K={k})   : {percentile(xfer, 50) * 1000:7.1f} ms p50")
            out["obs_to_tensors"] = latency_stats(xfer)

            spatial_bytes = obs["spatial"].nbytes
            total = sum(v.nbytes for v in obs.values())
            out["obs_kib_per_env"] = total / k / 1024.0
            out["spatial_kib_per_env"] = spatial_bytes / k / 1024.0
            print(
                f"obs payload/env: {out['obs_kib_per_env']:.0f} KiB "
                f"(spatial {out['spatial_kib_per_env']:.0f} KiB)"
            )

            for kk in (1, 2, 4):
                kobs = env.init([{**cfg, "seed": f"prof-k{kk}-{i}"} for i in range(kk)])
                kreport = step_env(env, kk, max(8, iters // 2), 2, kobs)
                print_step_report(f"env.step (K={kk})", kk, kreport)
                out[f"env_step_k{kk}"] = kreport

            rollout_len = 32
            k_upd = min(k, 8)
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
            trainer.update(buf, last)
            upd: list[float] = []
            for _ in range(max(5, iters // 2)):
                t0 = time.perf_counter()
                trainer.update(buf, last)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                upd.append(time.perf_counter() - t0)
            print(
                f"PPO update (T={rollout_len} K={k_upd}): "
                f"{percentile(upd, 50) * 1000:7.1f} ms p50"
            )
            out["ppo_update"] = latency_stats(upd)

        env.close()
    finally:
        server.stop()
    return out


def run_layout(
    *,
    stage: str,
    servers: int,
    envs_each: int,
    port: int,
    iters: int,
    warmup: int,
    burnin: int,
    workers: int = 0,
) -> dict[str, Any]:
    cfg = stage_cfg(stage)
    n = servers * envs_each
    topology = format_layout(servers, envs_each)
    sharded = ShardedVecEnv(servers, port, envs_each, workers=workers)
    try:
        lobs = sharded.init(
            [{**cfg, "seed": f"lay-{topology}-{i}"} for i in range(n)]
        )
        if burnin:
            step_env(sharded, n, burnin, 0, lobs)
        report = step_env(sharded, n, iters, warmup, lobs)
        print_step_report(f"  {topology}", n, report)
        return {
            "stage": stage,
            "topology": topology,
            "servers": servers,
            "envs_each": envs_each,
            "total_envs": n,
            "workers": workers,
            **report,
        }
    except Exception as exc:
        print(f"  {topology}              : FAILED {type(exc).__name__}: {exc}")
        return {
            "stage": stage,
            "topology": topology,
            "servers": servers,
            "envs_each": envs_each,
            "total_envs": n,
            "workers": workers,
            "error": f"{type(exc).__name__}: {exc}",
        }
    finally:
        sharded.close()


def run_layout_compare(
    *,
    stage: str,
    layouts: list[tuple[int, int]],
    port: int,
    iters: int,
    warmup: int,
    burnin: int,
    workers: int = 0,
) -> list[dict[str, Any]]:
    totals = {s * e for s, e in layouts}
    if len(totals) != 1:
        raise SystemExit("layouts must share the same total env count")
    print(
        f"\nlayout compare total_envs={next(iter(totals))} "
        f"stage={stage} workers={workers}"
    )
    rows: list[dict[str, Any]] = []
    base = port
    for servers, envs_each in layouts:
        rows.append(
            run_layout(
                stage=stage,
                servers=servers,
                envs_each=envs_each,
                port=base,
                iters=iters,
                warmup=warmup,
                burnin=burnin,
                workers=workers,
            )
        )
        base += servers + 1
    return rows


def pick_fastest_stable(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    ok = [r for r in rows if "error" not in r and r.get("env_steps_per_s")]
    if not ok:
        return None
    return max(ok, key=lambda r: float(r["env_steps_per_s"]))


def run_baseline(
    *,
    stages: list[str],
    layouts: list[tuple[int, int]],
    port: int,
    k: int,
    iters: int,
    warmup: int,
    burnin: int,
    device: torch.device,
    skip_layouts: bool,
    workers: int = 0,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "kind": "baseline",
        "stages": stages,
        "layouts": [format_layout(s, e) for s, e in layouts],
        "workers": workers,
        "single_server": [],
        "topology": [],
        "fastest_stable": {},
    }
    for i, stage in enumerate(stages):
        print(f"\n=== stage {stage} ===")
        report["single_server"].append(
            run_single_server(
                stage=stage,
                port=port + i,
                k=k,
                iters=iters,
                warmup=warmup,
                burnin=burnin,
                device=device,
                include_policy=i == 0,
                workers=workers,
            )
        )
        if skip_layouts:
            continue
        layout_port = port + 40 + i * 80
        rows = run_layout_compare(
            stage=stage,
            layouts=layouts,
            port=layout_port,
            iters=max(4, iters // 2),
            warmup=max(1, warmup // 2),
            burnin=burnin,
            workers=workers,
        )
        report["topology"].extend(rows)
        best = pick_fastest_stable(rows)
        if best:
            report["fastest_stable"][stage] = {
                "topology": best["topology"],
                "env_steps_per_s": best["env_steps_per_s"],
                "p50_ms": best["latency"]["p50_ms"],
            }
            print(
                f"  fastest stable {stage}: {best['topology']} "
                f"{best['env_steps_per_s']:.1f} env-steps/s"
            )
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="OFAI training pipeline profiler")
    p.add_argument("--port", type=int, default=DEFAULT_PORT)
    p.add_argument("--stage", type=str, default="duel", choices=[s.name for s in STAGES])
    p.add_argument(
        "--stages",
        type=str,
        default="",
        help="comma stage list; default representative: duel,peninsula,nations",
    )
    p.add_argument("--k", type=int, default=8, help="envs on the single-server benches")
    p.add_argument("--iters", type=int, default=12)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument(
        "--layouts",
        type=str,
        default="",
        help="comma layouts at equal total envs, e.g. 16x8,32x4,64x2",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--burnin",
        type=int,
        default=0,
        help="decision steps after init before timing (mid-episode cost)",
    )
    p.add_argument(
        "--baseline",
        action="store_true",
        help="profile representative stages and 16x8/32x4/64x2 topologies",
    )
    p.add_argument(
        "--quick",
        action="store_true",
        help="smaller K/iters/layouts for a focused smoke profile",
    )
    p.add_argument(
        "--skip-layouts",
        action="store_true",
        help="baseline single-server stages only",
    )
    p.add_argument("--json-out", type=str, default="")
    p.add_argument(
        "--workers",
        type=int,
        default=0,
        help="official-core actor threads per EnvServer (0/1=sequential oracle)",
    )
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    iters = 4 if args.quick and args.iters == 12 else args.iters
    warmup = 1 if args.quick and args.warmup == 3 else args.warmup
    k = 2 if args.quick and args.k == 8 else args.k

    if args.baseline:
        stages = (
            [s.strip() for s in args.stages.split(",") if s.strip()]
            if args.stages
            else list(REPRESENTATIVE_STAGES)
        )
        if args.quick:
            stages = stages[:1]
        layouts = (
            parse_layouts(args.layouts)
            if args.layouts
            else (list(QUICK_LAYOUTS) if args.quick else list(REPRESENTATIVE_LAYOUTS))
        )
        report = run_baseline(
            stages=stages,
            layouts=layouts,
            port=args.port,
            k=k,
            iters=iters,
            warmup=warmup,
            burnin=args.burnin,
            device=device,
            skip_layouts=args.skip_layouts,
            workers=args.workers,
        )
        if args.json_out:
            Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
            print(f"\nwrote {args.json_out}")
        return

    single = run_single_server(
        stage=args.stage,
        port=args.port,
        k=k,
        iters=iters,
        warmup=warmup,
        burnin=args.burnin,
        device=device,
        include_policy=True,
        workers=args.workers,
    )
    layouts = parse_layouts(args.layouts) if args.layouts else []
    report: dict[str, Any] = {"kind": "single", "single_server": [single], "topology": []}
    if layouts:
        report["topology"] = run_layout_compare(
            stage=args.stage,
            layouts=layouts,
            port=args.port + 20,
            iters=iters,
            warmup=warmup,
            burnin=args.burnin,
            workers=args.workers,
        )
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(report, indent=2) + "\n")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
