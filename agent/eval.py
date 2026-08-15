"""Evaluation: run N episodes with a (near-)greedy policy and report win rate.

Example:
  .venv/bin/python -m agent.eval --ckpt checkpoints/run/latest.pt \
      --episodes 20 --map Halkidiki --difficulty Hard
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--envs", type=int, default=4)
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--connect", action="store_true")
    p.add_argument("--map", type=str, default="FourIslands")
    p.add_argument("--map-size", type=str, default="Normal")
    p.add_argument("--nations", type=str, default="default")
    p.add_argument("--difficulty", type=str, default="Easy")
    p.add_argument("--bots", type=int, default=20)
    p.add_argument("--max-ticks", type=int, default=12000)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()

    nations: int | str = args.nations
    if nations not in ("default", "disabled"):
        nations = int(nations)

    server = None if args.connect else EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)

    remaining = args.episodes
    wins = 0
    deaths = 0
    kills_total = 0
    finished = 0
    ticks_survived: list[int] = []

    try:
        while remaining > 0:
            batch_n = min(args.envs, remaining)
            cfgs = [
                {
                    "map": args.map,
                    "mapSize": args.map_size,
                    "nations": nations,
                    "difficulty": args.difficulty,
                    "bots": args.bots,
                    "seed": f"eval-{args.episodes - remaining}-{i}",
                    "maxTicks": args.max_ticks,
                    "decisionInterval": 10,
                    "shaping": 0,
                }
                for i in range(batch_n)
            ]
            obs = env.init(cfgs)
            active = np.ones(batch_n, dtype=bool)
            while active.any():
                with torch.no_grad():
                    obs_t = obs_to_tensors(obs, device)
                    logits = policy.logits(obs_t)
                    heads = ["action", "target", "region", "quantity", "unit"]
                    actions = torch.stack(
                        [logits[h].argmax(dim=-1) for h in heads], dim=1
                    )
                step = env.step(actions.cpu().numpy().astype(np.int32))
                for i in range(batch_n):
                    if active[i] and step.dones[i]:
                        active[i] = False
                        finished += 1
                        info = step.infos[i]
                        if info.get("win"):
                            wins += 1
                        if info.get("dead"):
                            deaths += 1
                        kills_total += int(info.get("kills", 0))
                        ticks_survived.append(int(info.get("tick", 0)))
                obs = step.obs
            remaining -= batch_n
    finally:
        env.close()
        if server is not None:
            server.stop()

    n = max(1, finished)
    print(f"eval episodes={finished} map={args.map} difficulty={args.difficulty}")
    print(f"  win_rate : {wins / n:.3f} ({wins}/{n})")
    print(f"  death    : {deaths / n:.3f}")
    print(f"  kills/ep : {kills_total / n:.2f}")
    print(f"  ticks    : mean {np.mean(ticks_survived):.0f}")


if __name__ == "__main__":
    main()
