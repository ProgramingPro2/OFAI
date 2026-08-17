# OFAI

Reinforcement-learning agent for OpenFront. Trains headlessly against the
game's built-in AI (Nations/Tribes) using the deterministic TypeScript core
as the environment and a PyTorch PPO stack as the learner.

See `docs/PLAN.md` for architecture, observation/action spaces, reward
protocol, curriculum, checkpoint schema, and roadmap.

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
source .venv/bin/activate

# Train (N env-server processes x K games each). Fresh runs truncate their
# JSONL logs; resume appends. Episode-level metrics go to logs/<run>.episodes.jsonl.
python -m agent.train --servers 4 --envs 8 --rollout-len 128 --run-name run1 \
    --eval-every 25 --total-steps 2000000

# Resume with an absolute step ceiling, or add more steps beyond the checkpoint:
python -m agent.train --resume checkpoints/run1/latest.pt --run-name run1 \
    --additional-steps 1000000

# Evaluate true 80% FFA objective (no milestone termination).
# Sampled is the deployed policy; --mode deterministic is a greedy diagnostic.
python -m agent.eval --ckpt checkpoints/run1/best.pt --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --stage-tiles-frac 0.30

# Watch
python -m agent.watch --ckpt checkpoints/run1/latest.pt --greedy --save match.mp4

# Unit tests (mocked; no EnvServer required)
python -m pytest tests/test_rl_pipeline.py -q

# Live env smoke (requires OpenFrontIO env server)
python tests/test_env_client.py
```

Key flags: `--servers`, `--envs`, `--total-steps`, `--additional-steps`,
`--resume`, `--stage` / `--force-stage`, `--eval-every`, `--target-kl`
(default 0.03 active-joint), `--ent-target` (default 0.6 mask-normalized
action-type entropy in [0, 1]). There is no `--compile` path; rollout
always uses the eager policy.
