# OFAI

Reinforcement-learning agent for OpenFront. Trains headlessly against the
game's built-in AI (Nations/Tribes) using the deterministic TypeScript core
as the environment and a PyTorch PPO stack as the learner.

See `docs/PLAN.md` for architecture, observation/action spaces, reward
design, curriculum, and roadmap.

## Quick start

```bash
# Python deps live in .venv (torch, numpy, msgpack, ...)
source .venv/bin/activate

# Train
python -m agent.train --envs 8 --rollout-len 128 --run-name run1

# Evaluate
python -m agent.eval --ckpt checkpoints/run1/latest.pt --episodes 20

# Tests
cd OpenFrontIO && npx vitest run ofai/tests/env.test.ts
python tests/test_env_client.py
```
