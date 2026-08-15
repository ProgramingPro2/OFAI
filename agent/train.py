"""PPO training entry point.

Spawns (or connects to) an env server, then alternates rollout collection and
PPO updates. Curriculum stages advance on rolling win rate.

Example:
  .venv/bin/python -m agent.train --envs 8 --rollout-len 128 \
      --run-name dev --total-steps 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .curriculum import Curriculum
from .env_client import EnvServerProcess, VecEnv
from .model import Policy
from .ppo import PPOConfig, PPOTrainer, RolloutBuffer

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8, help="parallel games")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--connect", action="store_true", help="connect to an already-running server")
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=10_000_000, help="env decision steps")
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    p.add_argument("--stage", type=int, default=0, help="curriculum stage index to start at")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eval-every", type=int, default=0, help="iterations between evals (0=off)")
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

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
    trainer = PPOTrainer(policy, cfg)
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume}")

    server: EnvServerProcess | None = None
    if not args.connect:
        server = EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)

    def reinit_envs() -> dict:
        cfgs = [
            curriculum.make_env_config(
                f"{args.run_name}-s{args.seed}-e{i}-it{trainer.stats.iterations}"
            )
            for i in range(args.envs)
        ]
        return env.init(cfgs)

    obs = reinit_envs()
    buf = RolloutBuffer(cfg.rollout_len, args.envs)
    stage_base_episodes = 0
    t_start = time.time()

    while trainer.stats.env_steps < args.total_steps:
        t0 = time.time()
        obs = trainer.collect(env, obs, buf)
        collect_s = time.time() - t0
        t0 = time.time()
        metrics = trainer.update(buf, obs)
        update_s = time.time() - t0

        sps = cfg.rollout_len * args.envs / max(1e-9, collect_s + update_s)
        trainer.stats.sps = sps
        st = trainer.stats
        episodes_in_stage = st.episodes - stage_base_episodes
        if curriculum.maybe_advance(st.win_rate, st.episodes, episodes_in_stage):
            stage_base_episodes = st.episodes
            print(
                f"[curriculum] advanced to stage {curriculum.idx}: {curriculum.stage.name}"
            )
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
        if st.iterations % 5 == 0:
            print(
                f"it={st.iterations} steps={st.env_steps} eps={st.episodes} "
                f"winrate={st.win_rate:.3f} ret={st.mean_return:.3f} "
                f"sps={sps:.0f} stage={curriculum.stage.name} "
                f"({collect_s:.1f}s collect, {update_s:.1f}s update)"
            )
            trainer.save(str(run_dir / "latest.pt"))
        if st.iterations % 100 == 0:
            trainer.save(str(run_dir / f"it{st.iterations}.pt"))

    trainer.save(str(run_dir / "final.pt"))
    env.close()
    if server is not None:
        server.stop()
    elapsed = time.time() - t_start
    print(
        f"done in {elapsed / 60:.1f}min. checkpoints in {run_dir}, log {log_path}"
    )


if __name__ == "__main__":
    main()
