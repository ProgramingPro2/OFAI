# OFAI

Reinforcement-learning agent for OpenFront. Trains headlessly against the
game's built-in AI (Nations/Tribes) using the deterministic TypeScript core
as the environment and a PyTorch PPO stack as the learner.

See `docs/PLAN.md` for architecture, observation/action spaces, reward
design, curriculum, and roadmap.

## Clone

Two repositories: this trainer, and a fork of OpenFrontIO on the `ofai`
branch (the headless env harness lives there).

```bash
git clone https://github.com/ProgramingPro2/OFAI.git
cd OFAI
git clone -b ofai https://github.com/ProgramingPro2/OpenFrontIO.git
cd OpenFrontIO && npm install && cd ..
```

## Quick start

```bash
# Python deps live in .venv (torch, numpy, msgpack, tqdm, ...)
source .venv/bin/activate

# Train: --servers N env-server processes x --envs K games each, stepped in
# parallel. A tqdm progress bar shows sps / winrate / stage live.
python -m agent.train --servers 4 --envs 8 --rollout-len 128 --run-name run1

# Evaluate
python -m agent.eval --ckpt checkpoints/run1/latest.pt --episodes 20

# Tests
cd OpenFrontIO && npx vitest run ofai/tests/env.test.ts
python tests/test_env_client.py

# Throughput micro-profile (env step vs policy forward)
python tests/profile_pipeline.py
```

Key flags: `--compile cudagraph|inductor|off` (CUDA-graph rollout inference,
default on), `--servers`, `--envs`, `--total-steps`, `--resume`, `--stage`.
