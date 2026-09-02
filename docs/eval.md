# Official-core evaluation

`python -m agent.eval` is the standalone official TypeScript-core
evaluator. It does not start training, does not talk to public
multiplayer, and does not use the Rust mirror.

When `agent.train` collects on Rust, periodic holdout/mastery still
defaults to this official core (`--eval-backend auto`). Use
`--eval-backend train` only to score the collect farm on purpose.

Sampled PPO is the deployed policy. Deterministic argmax is a greedy
diagnostic only. `win_rate` is the core 80% FFA win; it is reported
with a Wilson 95% interval.

## Manual CLI (unchanged)

Explicit map / bot / action flags still work. The historical default
action mask is `EARLY_ACTIONS` (land + boat). Seeds stay
`eval-{samp|det}-{batch_start}-{lane}`.

```bash
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --episodes 20 --map Halkidiki --map-size Compact --bots 1 \
    --stage-tiles-frac 0.30 --workers 2
.venv/bin/python -m agent.eval --scripted expander --episodes 20 \
    --map Halkidiki --map-size Compact --bots 1 --max-ticks 8000
```

## Stage suites (frozen curriculum identities)

`--stage` loads the curriculum row: map, size, nations, difficulty,
bots, tick cap, and **stage allowed actions**. Impossible uses
`FULL_ACTIONS`; the EARLY default is not applied. `--suite holdout`
plays the canonical map on `holdout_seed(stage, 0..n-1)`. `--suite ood`
plays the stage OOD set on `ood_seed(stage, 0..n-1)`.

`--sampled-episodes` and `--det-episodes` are independent. Results are
aggregated in seed order, so `--envs` (lane count) does not change
identities or win rate. `--workers` is passed to `EnvServerProcess`.

`--json-out PATH` writes a machine-readable report (`stage`, `suite`,
`workers`, `config`, `sampled`, `deterministic`, Wilson bounds,
`hygiene`, `seeds`, `episode_records`). `--actions-out PATH` writes
per-episode 5-int action tapes (seed, config, terminal) for
`NATION_EP_ACTIONS` replay in `sim/tests/nation_episodes.rs`.
`--holdout-offset N` skips the first N frozen holdout/OOD seeds so
`--det-episodes 1 --holdout-offset 1` records only holdout[1].

```bash
# Goal holdout: 64 sampled + 8 det on World Normal Impossible
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite holdout \
    --sampled-episodes 64 --det-episodes 8 \
    --workers 2 --envs 8 --json-out eval-holdout.json

# Goal OOD: 16 sampled SouthAmerica Impossible
.venv/bin/python -m agent.eval --ckpt checkpoints/run1/best.pt \
    --stage nations-impossible-world --suite ood \
    --sampled-episodes 16 --det-episodes 0 \
    --workers 2 --envs 8 --json-out eval-ood.json
```

Omitting the episode flags on `nations-impossible-world` uses the same
counts (64/8 holdout, 16/0 OOD). Other stages default to the curriculum
32/8 holdout and 8/0 OOD sizes.

`--suite` without `--stage` is rejected. `--stage` with `--mode sampled`
runs only sampled episodes; `--mode deterministic` runs only det.

## Hygiene

Each mode report includes `accept_rate`, `action_hist`, `legal_frac`,
`terminal_causes`, and `hygiene` (`noop_frac`, `spawn_frac`,
`no_spawn_rate`, `timeout_rate`, `death_rate`).
