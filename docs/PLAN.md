# OFAI — OpenFront RL Agent: Architecture & Plan

## Goal

Train an agent to play OpenFront (openfront.io) at a high level via deep
reinforcement learning, training headlessly against the game's built-in
Nation/Tribe AI, with an opt-in, user-confirmed live path for supervised
private matches.

## System overview

```
+--------------------+        binary TCP frames        +----------------------+
|  Node/tsx process  |  <----------------------------> |  Python trainer      |
|  ofai/env/         |   obs tensors / action vectors  |  agent/              |
|  AgentEnv x K      |                                 |  Policy (CNN+MLP)    |
|  (deterministic    |                                 |  PPO + GAE           |
|   src/core sim)    |                                 |  Curriculum          |
+--------------------+                                 +----------------------+
```

- The game core (`src/core`) runs natively in Node; no browser, no network.
- The agent is a Human player in a Singleplayer FFA game; opponents are
  Nations (difficulty-scaled AI) and Tribes (filler bots) that live inside
  the sim.
- Decisions happen every `decisionInterval` ticks (default 10 = 1s game
  time); intents are fed as turns to `GameRunner`.

## Verified architecture (Python)

### Policy (`agent/model.py`)

- Spatial CNN + **per-slot player embeddings** (masked mean-pool for trunk
  context) + global MLP → tanh-bounded trunk.
- **Pointer-style target head**: scores `(trunk, candidate_slot_emb)` per
  player slot, masked by action-conditioned `target_masks[b, action, :]`.
- Single autoregressive API: `act(obs, deterministic=False)` used by
  train/eval/watch/trace.
- **Active-head logprobs / entropy**:
  - always: action type
  - spawn → region
  - attack → target + quantity
  - retreat / noop → none
  - build → unit + region
  - boat → region + quantity
  - diplomacy (ally/break/embargo) → target
- `evaluate()` matches `act()` exactly. Deterministic argmax uses the
  **selected** action's target/region/quantity masks (not union masks).

### Observations (wire contract)

| Tensor | Shape | Notes |
|--------|-------|-------|
| spatial | [K,10,64,64] f32 | land/mountain/mine/enemy/ally/border/fallout/defense/structures |
| players | [K,16,14] f32 | slot 0 = agent |
| global | [K,10] f32 | tick progress, spawn, tiles/troops/gold, … |
| action_mask | [K,9] u8 | |
| **target_masks** | [K,9,16] u8 | per action type (replaces flat `target_mask`) |
| **quantity_mask** | [K,5] u8 | |
| unit_mask | [K,10] u8 | |
| spawn/build/boat_regions | [K,1024] u8 | unchanged |

Python accepts a legacy flat `target_mask` (broadcast) for older traces.

### Step info (wire)

Existing fields remain. Added:

- `terminalCause`: `none|win|curriculum_success|death|loss_alive|timeout|no_spawn`
- `win`: core FFA win only (80% land / last standing)
- `stageSuccess`: curriculum tile milestone and/or a true win
- `peakTilesFrac`
- `rewardTerms`: `{terminal, spawn, territory, elimination, total}`
- `intentCount`, `actionAccepted`

`actionAccepted` means the action translated to at least one core intent
(or was NOOP), not that territory changed. Eval telemetry distinguishes
accepted attacks from later tile growth/shrink.

Terminal rewards: `win=3.0`, `curriculum_success=1.0`, every non-objective
terminal `-1.0`, one-time spawn `+0.1`. Shaping is a bounded potential
(territory + 0.25×elimination) and anneals to 0 after early stages. A
milestone return cannot exceed a core win.

`EnvConfig.allowedActions` gates which action types the env marks legal.
During the spawn window the only legal action is SPAWN (NOOP is illegal
until the agent has land); a never-spawn at `maxTicks` is `no_spawn`.

### Metrics (`agent/ppo.py`)

- Persistent per-env episode trackers across `collect()` calls (full
  undiscounted return, decisions, terminal cause, territory, kills,
  reward-term sums, action histograms, accept/fizzle).
- Reset only on real `done` or explicit env reinit / `RolloutInterrupted`.
- Bounded rolling windows **20 / 100 / 500** (not unbounded lists) plus
  separate cumulative counters.
- `logs/<run>.jsonl` — aggregate per iteration.
- `logs/<run>.episodes.jsonl` — one record per completed episode.
- Fresh runs open logs with `"w"` (no silent append); `--resume` appends.

### PPO / recovery

- Adam `(β1,β2)=(0.9,0.95)`, `eps=1e-5`, default `lr=3e-4` (1e-4 left
  active-head KL near 1e-4); spatial buffer **float16** (not uint8).
- Deep-copied optimizer snapshot; hard revert restores model+optimizer,
  does **not** adapt `ent_coef`, reports **zero** kept updates.
- Advantages NaN-guarded + std-floor normalized.
- `np.array_split` for exact `num_minibatches` partitions.
- Early-stop on **aggregate active-head KL** (running mean); catastrophic
  net at `target_kl * hard_kl_mult`.
- Default `target_kl=0.03` (was 0.3 under 5-head summed accounting).
- Logs raw `action_entropy`, `action_entropy_max` (`log(n_legal)`),
  `action_entropy_norm` (`H / log(n_legal)` over samples with >1 legal
  actions), `cond_entropy`, and `ent_coef`.
- `ent_target` is mask-normalized in `[0, 1]` (default 0.6). Samples with
  `<=1` legal action are excluded (treated as fully satisfied). A raw-nats
  target such as 1.5 is unattainable on land stages (2–4 legal ⇒ max
  0.69–1.39) and races `ent_coef` toward uniform/NOOP.
- Entropy bonus and controller use normalized action-type entropy only.
  Conditional / 1024-way region entropy is logged, not bonused.
- Shard failure: restart dead shards, raise `RolloutInterrupted`, discard
  partial rollout, reset trackers — **never** retry-all (no double-step).

### Checkpoint schema (`schema_version: 2`)

```
model, optimizer, ent_coef, stats (bounded windows + cumulatives),
hyperparameters, metrics snapshot,
curriculum {idx, stage_episodes, stage_successes, consecutive_eval_passes,
mastery_version},
stage_baseline, best_score (sampled true win, det true win, sampled peak,
det peak), rng {python, numpy, torch, cuda}
```

Resume restores the above. Fresh `--stage` is explicit; on resume use
`--force-stage` to override. `--additional-steps` extends past checkpoint
`env_steps`; otherwise `--total-steps` is an absolute ceiling.

### Curriculum (`agent/curriculum.py`)

| Stage | Map | Bots | maxTicks | shaping | Actions |
|-------|-----|------|----------|---------|---------|
| duel | Halkidiki Compact | 1 | 8000 | 1.0 | land+boat, stage milestone 30% tiles |
| small-ffa | Halkidiki Compact | 5 | 10000 | 0.6 | land+boat |
| island-ffa | FourIslands Normal | 10 | 12000 | 0.4 | +build+boat |
| tribes20 | FourIslands Normal | 20 | 16000 | 0.25 | full |
| nations-* | … | … | … | →0 | full |

Halkidiki is peninsulas: land-only cannot reach an opponent on another
finger, and player ATTACK is legal only with a shared border (intents
without a border translate then retreat). Duel/small-ffa therefore include
boat (no Port required). The core FFA win is always 80% of land.

Duel sets `winTilesFrac=0.30` as a **non-terminating** competency flag
(`stageSuccess=true`, `win=false`). Terminating at `0.15` ended every
smoke episode and collapsed true-objective sampled wins (untrained 32%
→ final 2%). Expansion signal is territory shaping, which anneals to 0
in later stages. `REWARD_CURRICULUM_SUCCESS=1.0` is reserved below
`REWARD_WIN=3.0` if a discrete milestone bonus is ever reintroduced;
it must not create a positive non-win return.

`--eval-every` plays the true objective (no milestone termination) on
fixed seeds, both deterministic and sampled. `win` / `win_rate` are
core 80% FFA only. `stageSuccess` / `stage_success_rate` are peak tiles
reaching the stage milestone (or a true win) and never substitute for
`win`.

Advancement uses `min_episodes` plus **two consecutive** mastery passes.
A pass is either (A) deterministic competency (`stage_success_rate` on
milestone stages, else true `win_rate`, at `win_threshold`) or (B)
sampled **true-win** mastery (`win_rate >= 0.8` and Wilson lower bound
above `sampled_wilson_floor`, default 0.50). Untrained random on the
duel is ~20–32% and cannot pass (B). Sampled milestone / stageSuccess
alone never passes. Training-time sampled rates are logged only.

Deterministic argmax is a robustness / greedy diagnostic, not the
definition of policy quality. Training and deployment use the sampled
PPO policy. A 0% deterministic win with 100% sampled true wins is
mastery of the intended policy, not a reason to repeat the stage.

`best.pt` ranks sampled true `win_rate`, then deterministic true
`win_rate`, then sampled / det peak tiles. Collapse (never-spawn,
spawn-then-NOOP, near-zero territory) zeros that mode. Milestone rates
do not enter the score. Stage advance resets `best_score` so a new
stage can replace the previous best.

## Historical notes (bugs that shaped the design)

Earlier runs hit: mask-inconsistent region sampling; uint8 spatial
quantization KL walls; kill-farming from misaligned rewards; 5-head
summed KL budgets that were either too tight or too loose; compiled
rollout logprob drift; and retry-all shard recovery that double-stepped
healthy envs. The current stack removes those failure modes on the
Python side. Reward exclusivity and anti-farming live in AgentEnv.

## Layout

```
OpenFrontIO/            upstream repo, branch ofai
  ofai/env/             TS harness (ObsExtractor / AgentEnv / EnvServer)
agent/                  Python trainer
docs/PLAN.md            this file
tests/                  pytest unit tests + live smoke
checkpoints/, logs/     training artifacts
```

## How to run

```bash
.venv/bin/python -m agent.train --servers 4 --envs 8 --rollout-len 128 \
    --run-name run1 --eval-every 25

.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt --episodes 50 \
    --map Halkidiki --map-size Compact --bots 1 --stage-tiles-frac 0.30
.venv/bin/python -m agent.eval --scripted expander --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --max-ticks 8000

.venv/bin/python -m pytest tests/test_rl_pipeline.py -q
.venv/bin/python tests/test_env_client.py   # needs live EnvServer
```

## Roadmap (not yet built)

- Async/pipelined collection; imitation pretraining; self-play league.
- Live bridge (gated, explicit user confirmation; ToS-aware private lobbies).
- Emoji taunts (flavor).
