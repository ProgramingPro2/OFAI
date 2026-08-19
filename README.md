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
    --eval-every 25 --eval-episodes 32 --total-steps 2000000

# Resume with an absolute step ceiling, or add more steps beyond the checkpoint:
python -m agent.train --resume checkpoints/run1/latest.pt --run-name run1 \
    --additional-steps 1000000

# Evaluate true 80% FFA objective (no milestone termination).
# Sampled is the deployed policy; --mode deterministic is a greedy diagnostic.
python -m agent.eval --ckpt checkpoints/run1/best.pt --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --stage-tiles-frac 0.30

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
python -m pytest tests/test_rl_pipeline.py -q

# Live env smoke (requires OpenFrontIO env server)
python tests/test_env_client.py
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

Key flags: `--servers`, `--envs`, `--total-steps`, `--additional-steps`,
`--resume`, `--stage` / `--force-stage` (name or current-table index),
`--allow-collapsed-resume`, `--dry-run`,
`--eval-every`, `--eval-episodes`
(default 32 frozen holdout; values below the stage minimum are raised),
`--eval-det-episodes` (diagnostic),
`--eval-ood-episodes` (logged only), `--target-kl`
(default 0.03 active-joint), `--ent-target` (default 0.6 mask-normalized
action-type entropy in [0, 1]). There is no `--compile` path; rollout
always uses the eager policy.

Curriculum resume is keyed by stage name. A schema-v2 checkpoint that
only stored `idx` is migrated through the frozen v2 name table so
inserted stages cannot silently remap. Unknown names fail clearly.
Train maps, holdout suite, OOD suite, and seed-stream counters are
written to checkpoint extras and JSONL logs.
