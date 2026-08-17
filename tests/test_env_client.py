"""Integration smoke test: boots EnvServer, inits 2 envs, steps random
actions, validates shapes/dtypes and episode termination.

Run from workspace root: .venv/bin/python tests/test_env_client.py
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import spaces
from agent.env_client import EnvServerProcess, VecEnv

CFG = {
    "map": "FourIslands",
    "mapSize": "Normal",
    "nations": 2,
    "difficulty": "Easy",
    "bots": 5,
    "seed": "py-smoke",
    "maxTicks": 1500,
    "decisionInterval": 10,
    "shaping": 0,
}


def main() -> None:
    port = 8791
    server = EnvServerProcess(port)
    try:
        env = VecEnv("127.0.0.1", port)
        cfgs = []
        for i in range(2):
            cfgs.append({**CFG, "seed": f"py-smoke-{i}"})
        obs = env.init(cfgs)
        assert obs["spatial"].shape == (
            2,
            spaces.SPATIAL_CHANNELS,
            spaces.SPATIAL_SIZE,
            spaces.SPATIAL_SIZE,
        ), obs["spatial"].shape
        assert obs["players"].shape == (2, spaces.NUM_PLAYER_SLOTS, spaces.PLAYER_FEATURES)
        assert obs["global"].shape == (2, spaces.GLOBAL_FEATURES)
        assert obs["action_mask"].shape == (2, spaces.NUM_ACTION_TYPES)
        assert obs["target_masks"].shape == (
            2,
            spaces.NUM_ACTION_TYPES,
            spaces.NUM_PLAYER_SLOTS,
        )
        assert obs["target_masks"].dtype == np.uint8
        assert obs["quantity_mask"].shape == (2, spaces.NUM_QUANTITIES)
        assert obs["quantity_mask"].dtype == np.uint8
        assert obs["spawn_regions"].shape == (2, spaces.NUM_REGIONS)
        # Spawn phase: spawn action must be legal.
        assert obs["action_mask"][:, spaces.ACTION_SPAWN].all()
        assert obs["spawn_regions"].sum() > 0

        rng = np.random.default_rng(0)
        dones_seen = np.zeros(2, dtype=bool)
        t0 = time.time()
        steps = 0
        for _ in range(400):
            k = env.k
            actions = np.stack(
                [
                    np.array(
                        [
                            # Choose only among legal action types.
                            rng.choice(np.flatnonzero(obs["action_mask"][i])),
                            rng.integers(0, spaces.NUM_PLAYER_SLOTS),
                            rng.integers(0, spaces.NUM_REGIONS),
                            rng.integers(0, spaces.NUM_QUANTITIES),
                            rng.integers(0, spaces.NUM_UNIT_TYPES),
                        ]
                    )
                    for i in range(k)
                ]
            ).astype(np.int32)
            batch = env.step(actions)
            assert batch.rewards.shape == (k,)
            assert batch.obs["spatial"].shape[0] == k
            assert batch.obs["target_masks"].shape == (
                k,
                spaces.NUM_ACTION_TYPES,
                spaces.NUM_PLAYER_SLOTS,
            )
            for info in batch.infos:
                assert "terminalCause" in info
                assert "peakTilesFrac" in info
                assert "stageSuccess" in info
                assert "rewardTerms" in info
                if info.get("terminalCause") == "curriculum_success":
                    assert info.get("win") is False
                    assert info.get("stageSuccess") is True
                if info.get("win"):
                    assert info.get("terminalCause") == "win"
                terms = info["rewardTerms"]
                assert abs(
                    terms["total"]
                    - (
                        terms["terminal"]
                        + terms["spawn"]
                        + terms["territory"]
                        + terms["elimination"]
                    )
                ) < 1e-5
                assert "intentCount" in info
                assert "actionAccepted" in info
            dones_seen |= batch.dones
            steps += 1
            if dones_seen.all():
                break
            obs = batch.obs
        dt = time.time() - t0
        print(
            f"steps={steps} ({steps / dt:.1f} vec-steps/s, "
            f"{steps * env.k / dt:.0f} env-steps/s) dones={dones_seen}"
        )
        assert dones_seen.all(), "episodes never terminated"
        env.close()
    finally:
        server.stop()
    print("PASS")


if __name__ == "__main__":
    main()
