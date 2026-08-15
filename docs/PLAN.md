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
  Env throughput currently ~30 decision-steps/s (~300 ticks/s) with 4 envs,
  dominated by Python round-trips and obs extraction; scale by running more
  env-server processes.

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

- `spatial` [10,128,128] f32 planes: land, mountain, mine, enemy, ally,
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
# Train (spawns its own env server)
.venv/bin/python -m agent.train --envs 8 --rollout-len 128 --run-name run1

# Evaluate a checkpoint
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/latest.pt \
    --episodes 20 --map Halkidiki --difficulty Hard

# Env harness tests
cd OpenFrontIO && npx vitest run ofai/tests/env.test.ts

# Python integration test
.venv/bin/python tests/test_env_client.py
```

## Roadmap (not yet built)

- **Throughput**: shard K envs across N env-server processes; batch obs
  extraction; consider `Compact` maps for early stages.
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
