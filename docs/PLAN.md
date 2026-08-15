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

## Key facts established during research

- Tick rate: 10 Hz (`Config.msPerTick() = 100`).
- FFA win condition: >80% of non-fallout land tiles (`WinCheckExecution`).
- Singleplayer spawn phase ends the moment the human picks a spawn
  (`SpawnExecution`), which keeps episodes fast.
- The sim is deterministic per seed. **Caveat found the hard way:**
  `loadTerrainMap()` caches mutable `GameMapImpl` state module-level, so a
  second game on the same map inherits the first game's territory. We bypass
  it with `ofai/env/TerrainCache.ts` (caches raw bins, fresh map per game).
- Full-game sim throughput: ~3000 ticks/s on a small map (single process).
- Throughput (measured 2026-08-14, RTX 3090): ~150-350 decision-steps/s
  with 4 env servers x 4 envs (sim cost scales with in-game activity; a
  passive policy steps faster than a fighting one). Before the perf pass
  the same setup managed ~25-45 env-steps/s with obs extraction + wire
  dominating. Key changes:
  - Obs planes downsampled to 64x64 (was 128x128): 4x less wire/CPU/GPU.
  - ObsExtractor maintains per-bin counters incrementally from the sim's
    packed tile deltas (O(changed tiles) per tick instead of O(map)).
  - ShardedVecEnv: N env-server processes stepped concurrently (threads),
    one batched GPU forward for all shards.
  - torch.compile(reduce-overhead) for rollout inference: 12.5ms -> ~1ms
    per batch of 64 (launch-overhead bound net).
  - Rollout buffer stores spatial planes as uint8 (values are in [0,1]),
    4x less RAM; dequantized in update().
- PPO stability (debugged over four instrumented runs 2026-08-14):
  1. Rollout sampling is mask-consistent with evaluate() (region head
     conditioned on the sampled action type — a union-mask mismatch produced
     -1e9 logprobs and exploded KL).
  2. Rollout logprobs are computed on the same uint8-quantized obs stored in
     the buffer (quantized-vs-raw input mismatch otherwise reads as KL).
  3. The trunk embedding is tanh-bounded; with an unbounded ReLU trunk the
     policy eventually drifted enough that the compiled-vs-eager rollout
     forward micro-divergence (~1e-3) became visible in logprobs, tripping
     the KL stopper every minibatch and freezing learning (entropy then
     collapsed via near-clip-boundary noise updates).
  4. A startup probe compares compiled vs eager logprobs and falls back to
     eager rollout if they diverge (> 0.02).
  5. Updates use value clipping (±1.0 around rollout values) and KL
     early-stopping (0.05). Healthy 150k-step run: entropy stable ~6.9,
     value_loss ~0.006, KL ~0.02, 16/16 minibatch updates per iter.
  6. Early curriculum stages use territory-delta shaping (5.0/3.0/2.0) so
     the reward signal is not ~100% deaths.

## Layout

```
OpenFrontIO/            upstream repo, branch ofai (pinned at v0.33.4)
  ofai/env/             TS harness: AgentEnv, ObsExtractor, ActionTranslator,
                        TerrainCache, EnvServer (TCP), framing
  ofai/tests/           vitest smoke + determinism tests
agent/                  Python: env_client, model, ppo, curriculum, train, eval
docs/PLAN.md            this file
tests/                  Python integration tests
checkpoints/, logs/     training artifacts
```

## Observation space (fixed shapes)

- `spatial` [10,64,64] f32 planes: land, mountain, mine, enemy, ally,
  my border, fallout, defense bonus, my structures, enemy structures.
- `players` [16,14] f32: slot 0 = agent; others sorted by territory.
  Features: exists, is_self, tiles_frac, log troops, log gold, alive,
  is_nation, is_bot, allied, embargoed, incoming attack troops, shares
  border, relation, has_spawned.
- `global` [10] f32: tick progress, spawn phase, my tiles/troops/gold,
  incoming troops, players alive, map size, has port, game over.
- Legality masks: `action_mask` [9], `target_mask` [16], `unit_mask` [10],
  and three region masks [1024] for spawn/build/boat.

## Action space (factorized, 5 ints per decision)

- action type (9): noop, spawn, attack, retreat_all, build, boat, ally,
  break_ally, embargo.
- target player slot (16)
- region (32x32 = 1024 coarse grid)
- quantity (5 troop fractions: 5/15/30/50/80%)
- unit type (10): City, DefensePost, SAMLauncher, MissileSilo, Port,
  Factory, AtomBomb, HydrogenBomb, MIRV, Warship.

Invalid actions (mask violations) translate to no-ops server-side.

## Reward

Win +1; eliminated -1; alive when someone else wins -0.5; timeout alive
-0.25; per-kill +1/(num opponents). Optional territory-delta shaping via
`shaping` env config (default 0).

## Curriculum (agent/curriculum.py)

1. `tribes`: FourIslands, no nations, 20 bots
2. `nations-easy-small`: FourIslands, 8 Easy nations
3. `nations-easy-mid`: Halkidiki, 12 Easy nations
4. `nations-medium-mid`: same, Medium
5. `nations-hard-mid`: same, Hard
6. `nations-impossible-world`: World, default nations, Impossible

Advance when rolling 100-episode win rate clears the stage threshold.

## How to run

```bash
# Train (spawns its own env servers: N processes x K games each)
.venv/bin/python -m agent.train --servers 4 --envs 8 --rollout-len 128 \
    --run-name run1

# Evaluate a checkpoint
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/latest.pt \
    --episodes 20 --map Halkidiki --difficulty Hard

# Env harness tests
cd OpenFrontIO && npx vitest run ofai/tests/env.test.ts

# Python integration test
.venv/bin/python tests/test_env_client.py

# Throughput micro-profile
.venv/bin/python tests/profile_pipeline.py
```

## Roadmap (not yet built)

- **Throughput**: async/pipelined collection (overlap GPU forward with env
  steps); `Compact` maps for early stages; more servers if cores allow.
- **Imitation pretraining**: replay downloader (`docs/API.md` full-game
  endpoint) -> intent extraction -> behavior cloning before PPO.
- **Self-play league**: after beating Impossible nations, mix past
  checkpoints in as extra Human opponents (env already supports N humans by
  config extension).
- **Live bridge (gated, explicit user confirmation required)**: Tampermonkey
  userscript bridging the page's WebSocket to a localhost policy server.
  Note: the OpenFront ToS prohibits automation on official servers; the
  intended scope is a handful of supervised private-lobby games with
  friends. Human-paced action cap (1 action/2s), no chat, kill switch.
