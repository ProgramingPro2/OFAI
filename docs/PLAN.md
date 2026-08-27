# OFAI — OpenFront RL Agent: Architecture & Plan

## Goal

Train an agent to play OpenFront (openfront.io) at a high level via deep
reinforcement learning, training headlessly against the game's built-in
Nation/Tribe AI, with an opt-in, user-confirmed live path for supervised
private matches.

## System overview

```
+--------------------+        binary TCP frames        +----------------------+
|  Official TS core  |  <----------------------------> |  Python trainer      |
|  ofai/env/         |   schema-v2 obs / actions       |  agent/              |
|  sequential|parallel|                                |  Policy (CNN+MLP)    |
|  (src/core oracle) |                                 |  PPO + GAE           |
+--------------------+                                 |  Curriculum          |
|  Rust mirror sim/  |  fail-closed fallback=official  |                      |
|  (boats/nations/   |  never mid-episode switch       |                      |
|   later systems)   |  eval stays official TS         |                      |
+--------------------+                                 +----------------------+
```

- The official game core (`src/core`) runs natively in Node; no browser, no
  public multiplayer. Parallel actors own complete games; sequential is
  the oracle. Training `--backend auto` (default) uses the Rust package
  when every remaining stage is schema-v2 supported; otherwise it stays
  official. `python -m agent.eval` is always the official core.
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
| action_mask | [K,9] u8 | effective: type can emit a state-changing intent now |
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
Action masks are **effective**: BOAT requires a `canBuildTransportShip`
destination plus troops; BUILD requires at least one unit `canBuild` can
place; ALLY/BREAK_ALLY/EMBARGO require an effective target; ATTACK
requires wilderness or a bordering attackable player plus troops. Factor
masks use the same ActionTranslator samplers as intent emission, so a
legal non-NOOP sample cannot silently no-op. Head order is unchanged
(schema v2 / `it800.pt` loadable). During the spawn window the only
legal action is SPAWN (NOOP is illegal until the agent has land); a
never-spawn at `maxTicks` is `no_spawn`.

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
  active-head KL near 1e-4). `--force-stage` loads policy/value weights
  then **resets Adam** (empty moments, step 0) at the CLI `--lr`.
  Ordinary same-stage resume still restores optimizer exactly.
  `--keep-optimizer` is an advanced override only. Spatial buffer
  **float16** (not uint8).
- Deep-copied optimizer snapshot; hard revert restores model+optimizer,
  does **not** adapt `ent_coef`, reports **zero** kept updates.
- Advantages NaN-guarded + std-floor normalized.
- `np.array_split` for exact `num_minibatches` partitions.
- Early-stop on **aggregate active-head KL** (running mean); catastrophic
  net at `target_kl * hard_kl_mult`.
- Default `target_kl=0.03` (was 0.3 under 5-head summed accounting).
- Logs raw `action_entropy`, `action_entropy_max` (`log(n_legal)`),
  `action_entropy_norm` (`H / log(n_legal)` over samples with >1 legal
  actions), `action_entropy_eff` / `action_entropy_eff_norm` (explore-worthy
  types only), `cond_entropy`, `ent_coef`, and `ref_kl` (plus per-head
  `ref_kl_*` when a stage-entry reference is set).
- `ent_target` is mask-normalized in `[0, 1]` (fresh default 0.6). Samples
  with `<=1` counted action are excluded (treated as fully satisfied). A
  raw-nats target such as 1.5 is unattainable on land stages (2–4 legal ⇒
  max 0.69–1.39) and races `ent_coef` toward uniform/NOOP.
- Entropy bonus and controller use normalized action-type entropy only.
  Fresh runs keep `ent_weighting=uniform` (all legal types). Transfer /
  `--v4-ablation` uses `effective`: NOOP / boat / build / diplomacy are
  dropped from the bonus so always-legal utilities are not rewarded for
  being diverse. Attack/spawn remain the exploration set; the normalized
  value stays in `[0, 1]`. Conditional / 1024-way region entropy is
  logged, not bonused.
- **Stage-entry reference retention.** `--force-stage` freezes a copy of
  the loaded policy (e.g. `it800.pt`) that is not an optimizer parameter.
  The loss is `coef * relu(KL(π || π_ref) - target)` using the exact
  mask-aware factorized KL
  `KL(π_a||ρ_a) + Σ_t π_a(t) Σ_{h ∈ H(t)} KL(π_h|t || ρ_h|t)`,
  not `mean(logp - ref_logp)`. Inactive heads contribute 0. Exceeding
  `ref_kl_hard` rolls the update back. Fresh training leaves `ref_kl_coef=0`.
  Checkpoints store `reference` metadata/weights for resume only; schema
  stays 2.
- Shard failure: restart dead shards, raise `RolloutInterrupted`, discard
  partial rollout, reset trackers — **never** retry-all (no double-step).

### Checkpoint schema (`schema_version: 2`)

```
model, optimizer, ent_coef, stats (bounded windows + cumulatives),
hyperparameters, metrics snapshot,
curriculum {idx, stage (name), stage_episodes, stage_successes,
last_eval_passed, consecutive_eval_passes, maps_unlocked, mastery_version},
seed_stream {run_name, master_seed, n_envs, episode_index[]},
curriculum_suite {canonical_map, train_maps, holdout_map, ood_maps, gate},
stage_baseline (sampled holdout at stage entry), pending_stage_baseline,
best_score (sampled true win, det true win, sampled peak,
det peak), rng {python, numpy, torch, cuda}
```

Resume restores the above. Curriculum identity is the stage **name**.
v2 checkpoints that only stored `idx` migrate through `V2_STAGE_NAMES`
(duel … nations-impossible-world), so a live `nations-easy-small`
checkpoint (`idx=4`) lands on that name rather than the inserted
`nations-easy-intro`. Inserted stages cannot silently load the wrong
stage; an unknown name or idx raises `CurriculumRestoreError`. Fresh
`--stage` accepts a name or a **current-table** index.
A missing `seed_stream` starts mixed counters at 0 (new hash namespace;
old `it{N}` strings are not reused). Fresh `--stage` is explicit; on
resume use `--force-stage` to override. `--additional-steps` extends past
checkpoint `env_steps`; otherwise `--total-steps` is an absolute ceiling.

`schema_version` stays **2** (model/observation tensors unchanged).
`mastery_version` is 3 (one frozen sampled-Wilson decision).

### Curriculum (`agent/curriculum.py`)

| Stage | Canonical map | Nations / bots | Notes |
|-------|---------------|----------------|-------|
| duel | Halkidiki Compact | 0 / 1 | land+boat, milestone 30% tiles |
| small-ffa | Halkidiki Compact | 0 / 5 | land+boat |
| island-ffa | FourIslands Normal | 0 / 10 | +build; island set after burn-in |
| tribes20 | FourIslands Normal | 0 / 20 | full actions |
| nations-easy-intro | FourIslands Normal | 3 / 8 | Easy nations, fewer bots |
| nations-easy-small | FourIslands Normal | 8 / 20 | previous first-nation stage |
| nations-easy-peninsula | Halkidiki Compact | 8 / 16 | topology/size soften |
| nations-easy-mid | Halkidiki Normal | 12 / 40 | peninsula Normal set |
| nations-medium-mid | Halkidiki Normal | 12 / 40 | Medium |
| nations-hard-mid | Halkidiki Normal | 12 / 40 | Hard |
| nations-world-easy | World Normal | 12 / 20 | Easy continent before Impossible |
| nations-impossible-world | World Normal | default / 100 | Impossible |

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

**Map sets.** Early stages stay on one canonical map. Extra train maps
(FourIslands/Hawaii/Caribbean/FaroeIslands, then Halkidiki/Balkans,
then World/Europe/NorthAmerica) unlock only after a **sampled frozen
holdout** clears the **mastery** bar: `win_rate >= 0.80` and Wilson
lower bound `>= 0.50` (26/32). The pre-update `--force-stage` baseline
never unlocks maps (a 21/32 peninsula zero-shot previously cleared the
old 16/32 bar). Relock uses hysteresis (`win_rate < 0.35`, about 11/32)
so 0.35–0.80 noise does not flip-flop. `map_burn_in` remains in suite
metadata only. Selection among unlocked maps is deterministic from the
episode seed. Holdout stays on the canonical map. Arcade, tournament,
The Box, and Giant World are never used. Tensor/action schema does not
change.

**Force-stage transfer.** `--force-stage` with a new `--run-name` is
isolated from the source directory. It resets stage-local curriculum,
windows, and seed stream; keeps global-best metadata for same-run
only; **does not inherit Adam moments**. Before the first PPO
collect/update it runs a frozen sampled holdout, writes
`stage-entry.pt` plus `stage_baseline`, and must not increment
`iterations`/`env_steps`, advance the stage, write `best.pt`, or
unlock extra maps. The run-global `best.pt` write bar is floored at
the stage-entry holdout score so 12/32 cannot become `best.pt` after
a 21/32 entry. A stage-local holdout regression guard compares later
evals to that baseline (or a better stage-best). Ordinary resume of a
halted run (`halt_reason` / `stage_regression`) is refused unless
`--force-stage` or `--allow-collapsed-resume`. On a confirmed collapse (the
observed ~0.69 → 0.156 / 0-win pattern: current `<= 50%` of baseline,
absolute drop `>= 0.30`, and Wilson intervals disjoint, or zero-win
against a competent bar) the run restores `stage-entry.pt` /
`best-<stage>.pt` policy/value, optimizer, `ent_coef`, RNG,
curriculum, and seed-stream, writes `latest.pt` atomically, and
**halts**. The outer `finally` must not rewrite that snapshot from
in-memory weights. Restore-and-continue would loop: per-update KL
stayed healthy while the policy drifted. Source checkpoints and
run-global `best.pt` are never overwritten; writes stay inside the
new `--run-name` directory. A weak or undersized baseline never
halts. `--eval-every 0` skips later unlock/regression after the
pre-update baseline and is warned on `--force-stage`. Missing
`maps_unlocked` / `stage_baseline` on old schema-v2 metadata loads
as locked / empty. Recommended recovery uses the source **16×8**
farm (128 envs, batch 16384) with `--lr 1e-4`, `--eval-every 5`,
`--additional-steps`, and a free `--port`; global defaults for
fresh training stay `3e-4` and `eval-every 0`. A 4×8 farm would
cut the minibatch 4× and change PPO noise/KL.

**v4 ablation.** `--v4-ablation` is a bounded transfer experiment: maps
locked, pre-update baseline first, stage-entry reference retention on,
eval every 5, 15-update cap if `--max-updates` is unset (the cap still
runs the due/final holdout and writes the last jsonl row; it does not
break before eval), `--additional-steps`
sized for that cap (so `it800` is not blocked by the 10M ceiling),
effective entropy, recommended `--ent-target 0.3`. Compare `stage_baseline`
holdout vs later evals (`win_rate`, action_hist / attack share,
`legal_frac`, `ref_kl`). Success: holdout does not fall below entry
and the policy does not become boat/build/diplomacy/NOOP. Failure: the
v3 monotonic 21/32 → 12 → 11 → 7 drop while per-update KL stays
0.003–0.006. Do not start with 10M steps. Schema v2 / `it800.pt`
weights stay loadable. No LSTM / self-play / slot-schema break.

**Holdout gate.** `--eval-every` plays the true objective on a **frozen
per-stage holdout** (canonical map; blake2 mix of
`("holdout", stage_name, index)`, default 32 sampled episodes). Seeds do
not rotate with iteration; retries replay the same identities. Values of
`--eval-episodes` below the stage minimum (including leftover `--eval-episodes 8`
commands) are raised to 32 so the 26/32 bar is actually played.
Advancement is **one** sampled true-win decision: `win_rate >= 0.8` and
Wilson lower bound `>= 0.50` (26/32). Deterministic argmax (default 8
episodes, same holdout maps/seeds) is logged only and cannot pass.
Sampled milestone / stageSuccess never passes. Training-time rates are
logged only. Optional OOD episodes (ArchipelagoSea / Balkans / Britannia
/ SouthAmerica, never overlapping that stage's train set) are logged and
never gate.

Eval uses `init_active` / `step_active`: episodes are round-robin
spread across as many shards as possible (a 32-game holdout on a 16×8
farm uses all 16 servers, two games each). Idle shards and unused
lanes are not stepped. Results stay in seed/episode order. After eval,
training `init()` restores the full shard layout.

**Seed stream.** Python owns train episode identity. Each env has a
persisted `episode_index`. Seeds are blake2 mixes of
`(train, run, master_seed, env_i, episode_index)` — not sequential
`e{i}` + djb2. On `done`, the step frame may include `nextConfigs`
(seed + map). TypeScript falls back to `{seed}-r{n}` only for watch /
clients that omit the field. Shard restart re-inits from the **current**
per-env configs (updated on each done), so a dead shard does not replay
the original `init()` batch. Interrupt / resume restores counters and
continues; it does not jump back to `it{N}`.

Deterministic argmax is a robustness / greedy diagnostic, not the
definition of policy quality. Training and deployment use the sampled
PPO policy. A 0% deterministic win with 100% sampled true wins is
mastery of the intended policy, not a reason to repeat the stage.

`best.pt` ranks sampled true `win_rate`, then deterministic true
`win_rate`, then sampled / det peak tiles. Collapse (never-spawn,
spawn-then-NOOP, near-zero territory) zeros that mode. Milestone rates
do not enter the score. `best.pt` is the run-global ranking and is
never reset to a sentinel on stage advance or `--force-stage`; a
zero-win or passively collapsed eval cannot replace it. Stage-best
files (`best-<stage>.pt`) reset per stage. Rolling train
`win_20/100/500` windows reset on stage transition / `--force-stage`.
Clean shutdown writes `latest.pt` and `final.pt`.

**16-slot observability.** Slot 0 is the agent. Priority fill keeps
bordering / attacking / allied opponents individually targetable. When
more than 15 opponents exist, slots 1–14 stay individuals and slot 15
is an untargetable overflow aggregate (both Nation+Bot feature flags).
Opponent `sharesBorderWith` / attackability / incoming troops are
computed once per extract and reused for player features and target
masks; ties break by `smallID` then id. v2 checkpoints remain loadable
(schema_version 2, same tensor shapes). Residual: more than 14
simultaneous border/attackers still cannot all be targeted; World-scale
games remain mostly anonymous mass. A v3 schema would be required for
full correction.

**Headless GameRunner.** AgentEnv constructs `GameRunner` with
`skipNamePlacement: true` so HUD `placeName` / `placeSpawnName` is
skipped. Packed tile/player/attack updates, hashes, rewards, masks,
and `renderFrame` still emit. `createGameRunner()` (OpenFront client)
keeps the default name-placement path.

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
.venv/bin/python -m agent.train --servers 8 --envs 8 --workers 4 \
    --backend auto --rollout-len 128 --run-name run1 \
    --eval-every 25 --total-steps 12000000

.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt --episodes 50 \
    --map Halkidiki --map-size Compact --bots 1 --stage-tiles-frac 0.30 --workers 2
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite holdout \
    --sampled-episodes 64 --det-episodes 8 --workers 2 --json-out eval-holdout.json
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite ood \
    --sampled-episodes 16 --det-episodes 0 --workers 2 --json-out eval-ood.json
.venv/bin/python -m agent.eval --scripted expander --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --max-ticks 8000

.venv/bin/python -m pytest tests/test_rl_pipeline.py tests/test_baseline_gates.py \
    tests/test_backend_select.py tests/test_eval_scheduler.py \
    tests/test_eval_harness.py tests/test_curriculum_generalization.py -q
.venv/bin/python tests/test_env_client.py   # needs live EnvServer
.venv/bin/python tests/profile_pipeline.py --baseline --quick
.venv/bin/python tests/profile_pipeline.py --stage duel --layouts 8x8 --workers 2
```

From `OpenFrontIO/`:

```bash
npx vitest run ofai/tests/env.test.ts ofai/tests/fidelity.test.ts \
    ofai/tests/runtime.test.ts
```

Rust capability / golden gates:

```bash
cargo test --manifest-path sim/Cargo.toml
cargo run --manifest-path sim/Cargo.toml -- --capabilities
```

## Baseline gates

Before runtime changes, record stage and topology numbers from
`tests/profile_pipeline.py`. Compare `16x8`, `32x4`, and `64x2` at 128
envs on `duel`, `nations-easy-peninsula`, and `nations-easy-small`.
Reports include env-steps/s, latency percentiles, resets, shard
failures, plus server `simMs` / `maskObsMs` / `serializeMs`.

Fidelity: `AgentEnv.strongStateDigest()` (test-only unless
`enableDigest`) must match committed goldens and randomized dual-env
traces at every decision boundary. Tile state, observations, masks,
rewards, intent counts, and terminal causes are part of the digest; the
existing 10-tick core hash remains in `info.hash`. Semantic drift fails
the gate and keeps an optimized backend disabled.

## Simulator backends and fallback

`agent/backend.py` is fail-closed. `--backend auto` (train default)
queries the Rust capability document and the union of remaining
curriculum stages. If an unknown capability, a non-v2 schema, or a
missing runtime is required, the selector returns official **before**
`ShardedVecEnv` creates episodes. A Rust farm that later sees an
unsupported `nextConfig` raises rather than approximating or switching
mid-episode. `--backend official` always uses the TypeScript core.
Standalone `agent.eval` is official-only.

Implemented official parity: sequential and parallel TypeScript actors
are byte-equal on goldens and randomized traces (obs, masks, rewards,
digests, terminals). Rust parity covers spawn / wilderness / player
land combat / boats / structures / diplomacy / tribe and nation AI /
rails / nukes / doomsday terminals when the capability document marks
those slices supported.

Worker topology is chosen by quality-to-wall-clock, not raw SPS. On a
32-core host the measured overnight pick is **8 servers × 8 envs × 2
workers** (64 envs, batch 8192, ~1387 duel env-steps/s, 0 shard
failures). Sequential 4x8 is the cheap-tick baseline; 16x8 remains the
historical transfer farm. `auto` workers shrink to the CPU/memory
budget and can fall back to sequential.

Optional official-core elite recording and BC warm-start exist behind
`--record-elite` / `--bc-dataset`. `best.pt` is never auto-promoted.
Accept requires the existing official holdout gate (win rate, Wilson,
diversity, reference-KL). This validation rejected BC for production;
use a fresh official-core run or the strongest validated non-BC
checkpoint.

## Roadmap (not yet built)

- Async/pipelined collection; self-play league.
- Live bridge (gated, explicit user confirmation; ToS-aware private lobbies).
- Emoji taunts (flavor).
