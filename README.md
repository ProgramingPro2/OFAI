# OFAI

Reinforcement-learning agent for OpenFront. Trains headlessly against the
game's built-in AI (Nations/Tribes) using a schema-v2 env (Rust sim by
default when supported, official TypeScript core otherwise) and a
PyTorch PPO stack as the learner. Standalone eval is always the official
core.

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
python -m agent.train --servers 8 --envs 8 --workers 4 --backend auto \
    --rollout-len 128 --run-name run1 \
    --eval-every 25 --eval-episodes 32 --total-steps 12000000

# Resume with an absolute step ceiling, or add more steps beyond the checkpoint:
python -m agent.train --resume checkpoints/run1/latest.pt --run-name run1 \
    --additional-steps 1000000

# Evaluate true 80% FFA objective (no milestone termination).
# Sampled is the deployed policy; --mode deterministic is a greedy diagnostic.
python -m agent.eval --ckpt checkpoints/run1/best.pt --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --stage-tiles-frac 0.30 --workers 2
# Frozen Impossible holdout / SouthAmerica OOD (see docs/eval.md)
python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite holdout \
    --sampled-episodes 64 --det-episodes 8 --workers 2 --json-out eval-holdout.json
python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite ood \
    --sampled-episodes 16 --det-episodes 0 --workers 2 --json-out eval-ood.json

# Watch (sampled policy; recommended fast preview).
# Interactive: one fullscreen pygame window for dropdown configuration,
# playback, and a timeline-scrubbed post-match replay. --frame-skip N
# still steps the policy every decision but only pulls the full-resolution
# map every Nth frame (plus
# first and last). --fps is the display/video rate of those rendered
# frames; in-window [ ] changes playback speed only. Videos stream to
# ffmpeg so long 1500x1500 games stay memory-bounded.
# --greedy is a deterministic diagnostic, not the deployed policy.
python -m agent.watch --ckpt checkpoints/run1/latest.pt \
    --preset tribes20 --scale 1 --fps 60 --frame-skip 5

# Headless video: nations only at the map's default nation count.
python -m agent.watch --ckpt checkpoints/run1/latest.pt \
    --opponents nations --difficulty Hard --map World \
    --save match.mp4 --scale 1 --fps 60 --frame-skip 5

# Mixed custom match: exactly 8 nations plus 20 tribe filler bots.
python -m agent.watch --ckpt checkpoints/run1/latest.pt \
    --opponents mixed --map FourIslands --map-size Normal \
    --nations 8 --bots 20 --difficulty Easy

# Unit tests (mocked; no EnvServer required)
python -m pytest tests/test_rl_pipeline.py tests/test_v4_redesign.py \
    tests/test_baseline_gates.py tests/test_backend_select.py \
    tests/test_env_workers.py tests/test_eval_scheduler.py \
    tests/test_eval_harness.py tests/test_sample_efficiency.py -q

# Live env smoke (requires OpenFrontIO env server)
python tests/test_env_client.py

# TypeScript env + replay-fidelity + sequential/parallel parity
# (from OpenFrontIO/)
npx vitest run ofai/tests/env.test.ts ofai/tests/fidelity.test.ts \
    ofai/tests/runtime.test.ts

# Rust capability / golden gates
cargo test --manifest-path sim/Cargo.toml

# Structured stage / topology baseline (writes JSON if --json-out is set)
python tests/profile_pipeline.py --baseline --quick
python tests/profile_pipeline.py --stage duel --layouts 4x8,8x8 --workers 2
python tests/profile_pipeline.py --baseline --layouts 16x8,32x4,64x2
```

`agent.watch` supports every curriculum stage via `--preset`: `duel`,
`small-ffa`, `island-ffa`, `tribes20`, `nations-easy-intro`,
`nations-easy-small`, `nations-easy-peninsula`, `nations-easy-mid`,
`nations-medium-mid`, `nations-hard-mid`, `nations-world-easy`, and
`nations-impossible-world`. Explicit match flags override preset values.
`--game-type` (also `--opponents`) selects `tribes`, `nations`, or `mixed`;
`--nations` accepts an exact count, `default`, or `disabled`, while `--bots`
is the tribe filler count from 0 to 400. Difficulties are `Easy`, `Medium`, `Hard`, and
`Impossible`; map sizes are `Compact` and `Normal`. Maps use OpenFront
`GameMapType` keys such as `Halkidiki`, `FourIslands`, and `World`.

Watch runs are reproducible by default: episode game IDs are `watch-0`,
`watch-1`, and so on, and sampled policy actions use PyTorch seed 0. Use
`--seed` and `--policy-seed` to choose another deterministic sequence. The
OpenFront agent environment supports single-player FFA only; team and network
lobby types are not watch options.

The interactive viewer is a single pygame window. It starts in true
fullscreen on `--display N` (default 0) and stays fullscreen during play.
CLI flags prepopulate the in-window form (checkpoint, preset, game type,
difficulty, map, nations/bots, seeds, max ticks, episodes, scale, fps,
frame skip, speed). Finite choices use clickable dropdowns (hover,
keyboard, scroll, click-away / Esc). Numbers stay text boxes or steppers.
Start and Quit are on that form. `--save` remains headless and does not
open the UI. `--episodes` is the headless video count; the window plays
one match, stays open, and uses New for the next seed.

Playback and camera (HUD stays fixed in screen pixels; only the map zooms):
- Space pause/resume; `.` or `n` step one decision while paused
- `[` / `]` slower and faster (0.25x to 8x, current speed always shown)
- Mouse wheel or `+`/`-` zoom (aspect ratio preserved); arrows/WASD or
  drag to pan; `0` / Home / middle-click recenter
- Esc opens a menu (Resume / Configure / Quit; arrows + Enter select);
  F11 or Alt+Enter leave or re-enter fullscreen; Q confirms quit; the
  window close button exits. The mouse is never grabbed.

After a match the window stays open. Replay keeps a bounded sliding window
of frames (96MB / 360-frame cap) so the end of a long match stays
seekable. A timeline scrollbar shows progress, the buffered window, and
any evicted prefix. Click or drag to seek in buffered frames; PageUp /
PageDown jump; `,` / `.` step; End jumps to the last frame. Evicted
frames are not instant-seekable: clicking that region rebuilds from the
recorded action tape when the env is still up, otherwise seek clamps to
the earliest buffered frame. Play/pause and speed chips sit on the replay
bar. Restart uses the same game seed and the policy RNG from match start;
New increments the episode suffix (`watch-1`, …); Configure returns to
the form.

Use `--windowed` or `--window-size WIDTHxHEIGHT` to start windowed
(oversized requests are clamped to the usable display). `--scale` controls
the native map and saved-video resolution; display fitting does not alter
video output.

Key flags: `--servers`, `--envs`, `--workers` (`auto` / `N`; `0`/`1` is
the sequential official oracle), `--backend` (`auto` default: rust when every remaining stage is
schema-v2 supported, else official; `official` forces the TypeScript
core; `rust` is the same fail-closed check as auto), `--total-steps`, `--additional-steps`,
`--resume`, `--stage` / `--force-stage` (name or current-table index),
`--keep-optimizer` (advanced; default `--force-stage` is a fresh Adam
at `--lr`), `--allow-collapsed-resume`, `--dry-run`,
`--eval-every`, `--eval-episodes`
(default 32 frozen holdout; values below the stage minimum are raised),
`--eval-det-episodes` (diagnostic),
`--eval-ood-episodes` (logged only), `--target-kl`
(default 0.03 active-joint), `--ent-target` (default 0.6 mask-normalized
action-type entropy in [0, 1]; transfer should pass `0.3`),
`--ent-weighting` (`uniform` fresh default, `effective` for transfer),
`--ref-kl-coef` / `--ref-kl-target` / `--ref-kl-hard` (stage-entry
reference retention; `0` coef disables it), `--lock-maps`,
`--max-updates`, `--v4-ablation`. There is no `--compile` path; rollout
always uses the eager policy.

`--force-stage` loads policy/value weights, resets Adam unless
`--keep-optimizer`, then runs a frozen sampled holdout **before** the
first collect/update. That baseline cannot advance the stage, write
`best.pt`, or unlock extra train maps. Extra maps unlock only at the
sampled mastery bar (26/32 and Wilson >= 0.50). The force-stage
run-global `best.pt` write bar is floored at the stage-entry holdout
score so a later 12/32 cannot replace 21/32. A confirmed holdout
collapse versus the stage-entry/stage-best bar restores those weights,
optimizer, `ent_coef`, RNG, curriculum, and seed stream, writes
`latest.pt`, and stops without the shutdown `finally` rewriting the
restored file. Ordinary resume of a checkpoint with `halt_reason` /
`stage_regression` metadata is refused unless `--force-stage` or
`--allow-collapsed-resume`. `--eval-every 0` is warned on
`--force-stage` (no later unlock/regression).

v3 peninsula transfer collapsed with healthy per-update KL (21/32 ->
12/32 -> 11/32 -> 7/32) because boat/build/diplomacy were mask-legal
without a valid intent and entropy rewarded those always-legal
utilities. v4 makes action masks mean "this selected action can emit a
state-changing intent now," freezes a stage-entry reference KL
(`ref_kl`), and uses effectiveness-aware entropy on transfer. Schema
stays v2; `it800.pt` remains loadable.

Recommended first transfer experiment (short ablation, not 10M steps).
Dry-run first; do not use these as new global fresh-run defaults:

```bash
python -m agent.train --dry-run --resume checkpoints/v2run1/it800.pt \
    --stage nations-easy-peninsula --force-stage --run-name v4ablate1 \
    --v4-ablation --ent-target 0.3 --lr 1e-4 --eval-every 5 \
    --eval-episodes 32 --servers 16 --envs 8 --port 8765

python -m agent.train --resume checkpoints/v2run1/it800.pt \
    --stage nations-easy-peninsula --force-stage --run-name v4ablate1 \
    --v4-ablation --ent-target 0.3 --lr 1e-4 --eval-every 5 \
    --eval-episodes 32 --servers 16 --envs 8 --port 8765
```

`--v4-ablation` locks maps, enables reference retention (`ref_kl_coef`
0.1), sets `--eval-every 5` and `--max-updates 15` if unset, sizes
`--additional-steps` for that update cap (so `it800` is not blocked by
the 10M ceiling), and uses effective entropy. The update cap still
runs the due/final holdout (it800: 805/810/815) before stopping. Ablation success: later
holdout >= stage-entry
baseline and attack share does not collapse into boat/build/diplomacy/NOOP.
Failure: monotonic holdout drop (the v3 21->12->11->7 pattern) or
utility-action takeover while per-update KL stays small.

Distribution-transfer recovery (do not use these as new global
defaults): `--lr 1e-4 --eval-every 5` with a **new** `--run-name`,
`--additional-steps` past the checkpoint `env_steps`, and the source
**16×8** farm. Example (read-only dry-run first):

```bash
python -m agent.train --dry-run --resume checkpoints/v2run1/it800.pt \
    --stage nations-easy-peninsula --force-stage --run-name v3run2 \
    --lr 1e-4 --eval-every 5 --eval-episodes 32 \
    --additional-steps 10000000 --servers 16 --envs 8 --port 8765
```

Curriculum resume is keyed by stage name. A schema-v2 checkpoint that
only stored `idx` is migrated through the frozen v2 name table so
inserted stages cannot silently remap. Unknown names fail clearly.
Train maps, holdout suite, OOD suite, unlock state, stage-entry
baseline, and seed-stream counters are written to checkpoint extras
and JSONL logs. Old schema-v2 checkpoints load without
`maps_unlocked` (stay locked) or holdout `stage_baseline` (first
later holdout becomes the guard).

## Baseline and fidelity gates

`tests/profile_pipeline.py` times env stepping, policy forward, and PPO
update, then compares farm layouts at a fixed env count. It reports
throughput, p50/p95/p99 step latency, auto-resets, shard failures, and
server-side simulation / mask-observation / serialization time.
Representative stages are `duel`, `nations-easy-peninsula`, and
`nations-easy-small`. Representative topologies are `16x8`, `32x4`, and
`64x2` (128 envs). Use `--quick` for a small smoke; `--json-out` writes
the structured report. Speed claims for later runtime work should use
the fastest stable measured topology, not an assumption.

Replay fidelity lives in `OpenFrontIO/ofai/tests/fidelity.test.ts`.
`AgentEnv.strongStateDigest()` hashes tile state, observations, masks,
rewards, intent counts, and terminal causes at every decision boundary,
alongside the official 10-tick core hash. Committed goldens under
`OpenFrontIO/ofai/tests/fixtures/` cover spawn, wilderness/player
combat, boats, structures, diplomacy, nations, and terminals. Randomized
dual-actor traces must stay byte-equal. `EnvConfig.enableDigest` puts
those digests on step info for tests only; training leaves it off.
Regenerate goldens with `UPDATE_OFAI_GOLDEN=1` only when a deliberate
semantic change is intended. Any digest drift blocks an optimized
runtime from being enabled.

## Simulator backends

Training `--backend auto` (default) selects the schema-v2 Rust mirror
(`sim/`) when **every** remaining curriculum stage is supported in the
capability document (`cargo run --manifest-path sim/Cargo.toml -- --capabilities`).
`--backend official` always spawns `OpenFrontIO/ofai/env/EnvServer.ts`.
Standalone `python -m agent.eval` is official-core only and is the
checkpoint-quality authority for Impossible wins. Unsupported or unknown
capabilities select official **before episode creation**. Mid-episode
switching and approximation are refused.

The official farm has two actor modes that must stay byte-equal:
sequential (`--workers 0`/`1`, the oracle) and parallel (`--workers N>1`,
each worker owns complete games).

On this 32-core host, measured duel layouts (no shard failures):

| Farm | Workers | Envs | env-steps/s | Notes |
|------|---------|------|-------------|-------|
| 2x8 | 1 | 16 | 781 | sequential baseline |
| 2x8 | 4 | 16 | 930 | fastest small farm; batch 2048 is thin |
| 4x8 | 1 | 32 | 711 | sequential, batch 4096 |
| 4x8 | 4 | 32 | 670 | workers add overhead on cheap duel ticks |
| 8x8 | 2 | 64 | 1387 | best quality-to-wall-clock; batch 8192 |
| 16x8 / 32x4 / 64x2 | 1 | 128 | 366 / 340 / 396 | nations-easy-small process topologies |

Choose 8x8 workers=2 for overnight official-core training: more
diversity per update than 4x8, higher stable SPS than sequential 16x8
nations (mask/obs dominated), and no fidelity loss versus the oracle.
`16x8` remains the historical transfer farm (batch 16384). Nations-stage
wall-clock is still dominated by official mask/obs, not PPO.

Optional elite recording (`--record-elite`) keeps official-core wins
only, atomically, with episode/step caps. Behavior-cloning a candidate
never writes `best.pt`; accept requires the existing official holdout
gate (win rate, Wilson, reference-KL, diversity). NLL-only is not an
accept. This validation rejected BC for production because no BC
candidate cleared that frozen official-core gate.

Limitations: a Rust farm that cannot declare a required capability stays
on official. Parallel official workers fall back to sequential if a
worker farm fails to init. Do not point this stack at public
multiplayer servers.
