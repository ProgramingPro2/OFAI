"""Micro-profile of the training hot loop: env server stepping vs Python
policy forward vs PPO update. Run from workspace root:

  .venv/bin/python tests/profile_pipeline.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import spaces
from agent.env_client import EnvServerProcess, VecEnv
from agent.model import Policy, obs_to_tensors
import torch

CFG = {
    "map": "FourIslands",
    "mapSize": "Normal",
    "nations": 2,
    "difficulty": "Easy",
    "bots": 10,
    "seed": "prof",
    "maxTicks": 4000,
    "decisionInterval": 10,
    "shaping": 0,
}


def main() -> None:
    k = 8
    port = 8792
    server = EnvServerProcess(port)
    try:
        env = VecEnv("127.0.0.1", port)
        obs = env.init([{**CFG, "seed": f"prof-{i}"} for i in range(k)])
        policy = Policy().cuda().eval()
        rng = np.random.default_rng(0)

        # 1) Raw step cost with noop actions (sim + obs + wire)
        actions = np.zeros((k, 5), dtype=np.int32)
        for _ in range(3):
            env.step(actions)  # warmup
        t0 = time.time()
        n = 30
        for _ in range(n):
            batch = env.step(actions)
        step_ms = (time.time() - t0) / n * 1000
        print(f"env.step (K={k})       : {step_ms:7.1f} ms  ({k * 1000 / step_ms:5.1f} env-steps/s)")

        # 2) Policy forward (GPU) on a real obs batch
        obs_t = obs_to_tensors(obs, torch.device("cuda"))
        with torch.no_grad():
            for _ in range(3):
                policy(obs_t)
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            for _ in range(n):
                policy(obs_t)
        torch.cuda.synchronize()
        fwd_ms = (time.time() - t0) / n * 1000
        print(f"policy forward (K={k}) : {fwd_ms:7.1f} ms")

        # 3) Combined train-style loop (masked random actions)
        t0 = time.time()
        for _ in range(n):
            legal = [np.flatnonzero(obs["action_mask"][i]) for i in range(k)]
            acts = np.stack(
                [
                    np.array(
                        [
                            rng.choice(legal[i]),
                            rng.integers(0, spaces.NUM_PLAYER_SLOTS),
                            rng.integers(0, spaces.NUM_REGIONS),
                            rng.integers(0, spaces.NUM_QUANTITIES),
                            rng.integers(0, spaces.NUM_UNIT_TYPES),
                        ]
                    )
                    for i in range(k)
                ]
            ).astype(np.int32)
            batch = env.step(acts)
            obs = batch.obs
        loop_ms = (time.time() - t0) / n * 1000
        print(f"train-style loop (K={k}): {loop_ms:7.1f} ms  ({k * 1000 / loop_ms:5.1f} env-steps/s)")

        # 4) Payload size per step reply
        spatial_bytes = obs["spatial"].nbytes
        total = sum(v.nbytes for v in obs.values())
        print(f"obs payload/env: {total / 1024:.0f} KiB (spatial {spatial_bytes / 1024:.0f} KiB)")

        # 5) Per-env step cost scaling (isolates serialization vs sim)
        for kk in (1, 2, 4):
            env.init([{**CFG, "seed": f"prof-{kk}-{i}"} for i in range(kk)])
            acts = np.zeros((kk, 5), dtype=np.int32)
            env.step(acts)
            t0 = time.time()
            for _ in range(20):
                env.step(acts)
            ms = (time.time() - t0) / 20 * 1000
            print(f"env.step (K={kk})       : {ms:7.1f} ms  ({kk * 1000 / ms:5.1f} env-steps/s)")

        env.close()
    finally:
        server.stop()


if __name__ == "__main__":
    main()
