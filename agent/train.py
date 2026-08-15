"""PPO training entry point.

Spawns (or connects to) env servers, then alternates rollout collection and
PPO updates. Curriculum stages advance on rolling win rate.

Throughput notes:
- --servers N spawns N independent Node env-server processes stepped in
  parallel (one sim per process, so CPU cores get used).
- --compile cudagraph wraps the rollout-time policy with CUDA graphs; the
  update pass stays eager. Weights are shared, so learning is unaffected.

Example:
  .venv/bin/python -m agent.train --servers 4 --envs 8 --rollout-len 128 \
      --run-name dev --total-steps 1000000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .curriculum import Curriculum
from .env_client import ShardedVecEnv
from .model import Policy
from .ppo import PPOConfig, PPOTrainer, RolloutBuffer

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8, help="parallel games per server")
    p.add_argument("--servers", type=int, default=4, help="env-server processes")
    p.add_argument("--port", type=int, default=8765, help="base port (servers use port+i)")
    p.add_argument("--connect", action="store_true", help="connect to already-running servers")
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=10_000_000, help="env decision steps")
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    p.add_argument("--stage", type=int, default=0, help="curriculum stage index to start at")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="iterations between evals (0=off)")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--compile",
        choices=["cudagraph", "inductor", "off"],
        default="cudagraph",
        help="torch.compile mode for rollout inference",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    run_dir = CKPT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.run_name}.jsonl"
    log_f = open(log_path, "a", buffering=1)

    curriculum = Curriculum(start=args.stage)
    policy = Policy()
    cfg = PPOConfig(
        rollout_len=args.rollout_len,
        lr=args.lr,
        device=args.device if torch.cuda.is_available() else "cpu",
    )

    rollout_policy = None
    if args.compile != "off" and cfg.device == "cuda":
        mode = "reduce-overhead" if args.compile == "cudagraph" else "default"
        rollout_policy = torch.compile(policy, mode=mode)

    trainer = PPOTrainer(policy, cfg, rollout_policy=rollout_policy)
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume}")

    env = ShardedVecEnv(
        num_servers=args.servers,
        base_port=args.port,
        envs_per_server=args.envs,
        connect=args.connect,
    )
    total_envs = env.k

    def reinit_envs() -> dict:
        cfgs = [
            curriculum.make_env_config(
                f"{args.run_name}-s{args.seed}-e{i}-it{trainer.stats.iterations}"
            )
            for i in range(total_envs)
        ]
        return env.init(cfgs)

    obs = reinit_envs()
    buf = RolloutBuffer(cfg.rollout_len, total_envs)
    stage_base_episodes = 0
    steps_per_iter = cfg.rollout_len * total_envs

    pbar = tqdm(
        total=args.total_steps,
        initial=trainer.stats.env_steps,
        unit="step",
        dynamic_ncols=True,
        desc=curriculum.stage.name,
    )

    try:
        while trainer.stats.env_steps < args.total_steps:
            t0 = time.time()
            obs = trainer.collect(env, obs, buf)
            collect_s = time.time() - t0
            t0 = time.time()
            metrics = trainer.update(buf, obs)
            update_s = time.time() - t0

            sps = steps_per_iter / max(1e-9, collect_s + update_s)
            trainer.stats.sps = sps
            st = trainer.stats
            episodes_in_stage = st.episodes - stage_base_episodes
            if curriculum.maybe_advance(st.win_rate, st.episodes, episodes_in_stage):
                stage_base_episodes = st.episodes
                pbar.write(
                    f"[curriculum] advanced to stage {curriculum.idx}: {curriculum.stage.name}"
                )
                pbar.set_description(curriculum.stage.name)
                obs = reinit_envs()

            record = {
                "it": st.iterations,
                "env_steps": st.env_steps,
                "episodes": st.episodes,
                "win_rate": round(st.win_rate, 4),
                "mean_return": round(st.mean_return, 4),
                "kills_total": st.kills,
                "sps": round(sps, 1),
                "stage": curriculum.stage.name,
                **{k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()},
            }
            log_f.write(json.dumps(record) + "\n")

            pbar.update(steps_per_iter)
            pbar.set_postfix(
                sps=f"{sps:.0f}",
                winrate=f"{st.win_rate:.3f}",
                ret=f"{st.mean_return:.3f}",
                eps=st.episodes,
                collect=f"{collect_s:.1f}s",
                update=f"{update_s:.1f}s",
            )
            if st.iterations % 5 == 0:
                trainer.save(str(run_dir / "latest.pt"))
            if st.iterations % 100 == 0:
                trainer.save(str(run_dir / f"it{st.iterations}.pt"))
    except KeyboardInterrupt:
        pbar.write("interrupted; saving checkpoint")
    finally:
        pbar.close()
        trainer.save(str(run_dir / "final.pt"))
        env.close()
        log_f.close()

    print(f"done. checkpoints in {run_dir}, log {log_path}")


if __name__ == "__main__":
    main()
