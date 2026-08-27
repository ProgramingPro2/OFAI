//! Full-episode official-vs-Rust Nation differential harness.
//!
//! Plays one deterministic mask-aware action tape on Rust from reset to an
//! exclusive terminal (or env maxTicks), then replays the same tape on
//! official. Every decision boundary compares tile/core/strong/obs/step
//! digests, reward, intents, acceptance, terminal cause, win, tick,
//! tilesFrac, and stageSuccess. A mid-episode official fallback fails.

use std::time::Instant;

use ofai_sim::actions::ActionVec;
use ofai_sim::env::AgentEnv;
use ofai_sim::obs::ObsBuffers;
use ofai_sim::spec::{
    EnvConfig, ACTION_ALLY, ACTION_ATTACK, ACTION_BOAT, ACTION_BREAK_ALLY, ACTION_BUILD,
    ACTION_EMBARGO, ACTION_NOOP, ACTION_RETREAT_ALL, ACTION_SPAWN, NUM_PLAYER_SLOTS, NUM_QUANTITIES,
    NUM_REGIONS, NUM_UNIT_TYPES,
};
use ofai_sim::terrain::TerrainCache;

const FULL_ACTIONS: &[&str] = &[
    "noop",
    "spawn",
    "attack",
    "retreat_all",
    "build",
    "boat",
    "ally",
    "break_ally",
    "embargo",
];

struct EpisodeSpec {
    name: &'static str,
    stage: &'static str,
    map: &'static str,
    map_size: &'static str,
    nations: serde_json::Value,
    bots: u32,
    difficulty: &'static str,
    gold: Option<i64>,
    max_ticks: u32,
    seed: &'static str,
    win_tiles_frac: f64,
    shaping: f64,
}

struct EpisodeOutcome {
    name: &'static str,
    stage: &'static str,
    seed: &'static str,
    steps: usize,
    terminal: String,
    win: bool,
    elapsed_ms: u128,
}

fn maps_dir() -> std::path::PathBuf {
    TerrainCache::default_maps_dir()
}

fn first_set_index(mask: &[u8]) -> i32 {
    mask.iter()
        .position(|&b| b != 0)
        .map(|i| i as i32)
        .unwrap_or(0)
}

fn first_legal_target(mask: &[u8], row: usize, slots: usize) -> i32 {
    let base = row * slots;
    for i in 0..slots {
        if mask.get(base + i).copied().unwrap_or(0) != 0 {
            return i as i32;
        }
    }
    1
}

/// Deterministic mask-aware tape: spawn first, then cycle combat / build /
/// naval / diplomacy so later systems are exercised when legal.
fn episode_action(obs: &ObsBuffers, step: usize) -> ActionVec {
    if obs.action_mask[ACTION_SPAWN as usize] != 0 {
        return ActionVec {
            action_type: ACTION_SPAWN,
            target: 0,
            region: first_set_index(&obs.spawn_regions).max(2),
            quantity: 0,
            unit: 0,
        };
    }
    let prefer = match step % 11 {
        1 | 6 => ACTION_BUILD,
        2 => ACTION_BOAT,
        3 => ACTION_ALLY,
        4 => ACTION_EMBARGO,
        5 => ACTION_BREAK_ALLY,
        8 => ACTION_RETREAT_ALL,
        _ => ACTION_ATTACK,
    };
    let chosen = if obs.action_mask.get(prefer as usize).copied().unwrap_or(0) != 0 {
        prefer
    } else if obs.action_mask[ACTION_ATTACK as usize] != 0 {
        ACTION_ATTACK
    } else if obs.action_mask[ACTION_BUILD as usize] != 0 {
        ACTION_BUILD
    } else if obs.action_mask[ACTION_BOAT as usize] != 0 {
        ACTION_BOAT
    } else {
        ACTION_NOOP
    };
    let region = match chosen {
        ACTION_SPAWN => first_set_index(&obs.spawn_regions),
        ACTION_BUILD => first_set_index(&obs.build_regions),
        ACTION_BOAT => first_set_index(&obs.boat_regions),
        _ => (step % NUM_REGIONS) as i32,
    };
    let unit = if chosen == ACTION_BUILD {
        let idx = (step / 3) % NUM_UNIT_TYPES;
        if obs.unit_mask.get(idx).copied().unwrap_or(0) != 0 {
            idx as i32
        } else {
            first_set_index(&obs.unit_mask)
        }
    } else {
        first_set_index(&obs.unit_mask)
    };
    ActionVec {
        action_type: chosen,
        target: first_legal_target(&obs.target_masks, chosen as usize, NUM_PLAYER_SLOTS),
        region,
        quantity: first_set_index(&obs.quantity_mask).min((NUM_QUANTITIES - 1) as i32),
        unit,
    }
}

fn official_replay_boundaries(
    raw_cfg: &serde_json::Value,
    actions: &[[i32; 5]],
) -> Result<(String, Vec<serde_json::Value>), String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../OpenFrontIO");
    let name = raw_cfg
        .get("seed")
        .and_then(|s| s.as_str())
        .unwrap_or("ep");
    let dump = std::env::temp_dir().join(format!(
        "ofai-nation-replay-{}-{}.json",
        std::process::id(),
        name
    ));
    let _ = std::fs::remove_file(&dump);
    let payload = serde_json::json!({
        "config": raw_cfg,
        "actions": actions,
        "outPath": dump.to_string_lossy(),
    });
    let out = std::process::Command::new("npx")
        .args(["tsx", "ofai/tests/fidelity.ts", "--replay-stdin"])
        .current_dir(&root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            use std::io::Write;
            if let Some(mut stdin) = child.stdin.take() {
                stdin.write_all(payload.to_string().as_bytes())?;
            }
            child.wait_with_output()
        })
        .map_err(|e| e.to_string())?;
    if !out.status.success() {
        return Err(format!(
            "official replay failed: {}",
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    let raw_json = std::fs::read_to_string(&dump).unwrap_or_else(|_| {
        let stdout = String::from_utf8_lossy(&out.stdout);
        stdout
            .rfind("{\"init\"")
            .map(|i| stdout[i..].to_string())
            .unwrap_or_default()
    });
    let v: serde_json::Value = serde_json::from_str(&raw_json).map_err(|e| {
        format!(
            "official json: {e} dump={} dump_bytes={} stdout={} stderr={}",
            dump.display(),
            raw_json.len(),
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })?;
    let init = v["init"]["tileDigest"].as_str().unwrap_or("").to_string();
    let bounds = v["boundaries"].as_array().cloned().unwrap_or_default();
    Ok((init, bounds))
}

fn assert_boundary(label: &str, i: usize, got: &ofai_sim::digest::DecisionBoundary, exp: &serde_json::Value) {
    let field = |k: &str| -> String {
        match &exp[k] {
            serde_json::Value::Null => "null".into(),
            v => v.to_string(),
        }
    };
    if got.tile_digest != exp["tileDigest"].as_str().unwrap_or("") {
        panic!(
            "{label} step {i} FIRST MISMATCH tileDigest rust_tick={} off_tick={} rust_hash={} off_hash={}\nrust_tile={}\noff_tile={}",
            got.tick,
            field("tick"),
            field("coreHash"),
            got.core_hash.map(|h| h.to_string()).unwrap_or_else(|| "null".into()),
            got.tile_digest,
            exp["tileDigest"].as_str().unwrap_or(""),
        );
    }
    assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "{label} step {i} FIRST MISMATCH coreHash");
    assert_eq!(
        got.obs_digest,
        exp["obsDigest"].as_str().unwrap_or(""),
        "{label} step {i} FIRST MISMATCH obsDigest (masks+obs)"
    );
    assert_eq!(
        got.step_digest,
        exp["stepDigest"].as_str().unwrap_or(""),
        "{label} step {i} FIRST MISMATCH stepDigest"
    );
    assert_eq!(
        got.digest,
        exp["digest"].as_str().unwrap_or(""),
        "{label} step {i} FIRST MISMATCH digest"
    );
    assert_eq!(got.reward, exp["reward"].as_f64().unwrap(), "{label} step {i} FIRST MISMATCH reward");
    assert_eq!(
        got.intent_count as u64,
        exp["intentCount"].as_u64().unwrap(),
        "{label} step {i} FIRST MISMATCH intentCount"
    );
    assert_eq!(
        got.action_accepted,
        exp["actionAccepted"].as_bool().unwrap(),
        "{label} step {i} FIRST MISMATCH actionAccepted"
    );
    assert_eq!(
        got.terminal_cause.as_str(),
        exp["terminalCause"].as_str().unwrap(),
        "{label} step {i} FIRST MISMATCH terminalCause"
    );
    assert_eq!(got.win, exp["win"].as_bool().unwrap(), "{label} step {i} FIRST MISMATCH win");
    assert_eq!(got.tick as u64, exp["tick"].as_u64().unwrap(), "{label} step {i} FIRST MISMATCH tick");
    if let Some(tf) = exp["tilesFrac"].as_f64() {
        assert_eq!(got.tiles_frac, tf, "{label} step {i} FIRST MISMATCH tilesFrac");
    }
    if let Some(ss) = exp["stageSuccess"].as_bool() {
        assert_eq!(got.stage_success, ss, "{label} step {i} FIRST MISMATCH stageSuccess");
    }
}

fn episode_raw(spec: &EpisodeSpec) -> serde_json::Value {
    let mut raw = serde_json::json!({
        "map": spec.map,
        "mapSize": spec.map_size,
        "nations": spec.nations,
        "bots": spec.bots,
        "difficulty": spec.difficulty,
        "seed": spec.seed,
        "maxTicks": spec.max_ticks,
        "decisionInterval": 10,
        "shaping": spec.shaping,
        "enableDigest": true,
        "allowedActions": FULL_ACTIONS,
        "winTilesFrac": spec.win_tiles_frac,
    });
    if let Some(g) = spec.gold {
        raw["startingGold"] = serde_json::json!(g);
    }
    raw
}

fn run_episode(cache: &mut TerrainCache, spec: &EpisodeSpec) -> EpisodeOutcome {
    let t0 = Instant::now();
    let raw = episode_raw(spec);
    let cfg = EnvConfig::from_json(&raw).unwrap_or_else(|e| panic!("{} cfg: {e}", spec.name));
    let mut rust = AgentEnv::create(cfg, cache).unwrap_or_else(|e| panic!("{} create: {e}", spec.name));
    assert_eq!(
        rust.strong_state_digest().tile_digest.is_empty(),
        false,
        "{} missing rust digest",
        spec.name
    );
    let rust_init = rust.strong_state_digest();
    let mut actions = Vec::new();
    let mut rust_bounds = Vec::new();
    let mut terminal = "none".to_string();
    let mut win = false;
    loop {
        let i = actions.len();
        let act = episode_action(rust.peek_obs(), i);
        actions.push([act.action_type, act.target, act.region, act.quantity, act.unit]);
        let step = rust.step(act);
        if let Some(reason) = step.info.fallback.as_deref() {
            panic!("{} rust fallback at step {i} tick={}: {reason}", spec.name, step.info.tick);
        }
        rust_bounds.push(rust.strong_state_digest());
        terminal = step.info.terminal_cause.as_str().to_string();
        win = step.info.win;
        if i > 0 && i % 40 == 0 {
            eprintln!(
                "  {} rust step {i} tick={} terminal={}",
                spec.name, step.info.tick, terminal
            );
        }
        if step.done {
            break;
        }
        if rust.game().ticks() >= spec.max_ticks {
            panic!("{} rust exceeded maxTicks without terminal", spec.name);
        }
    }
    let (off_init, off_bounds) = official_replay_boundaries(&raw, &actions)
        .unwrap_or_else(|e| panic!("{} official replay: {e}", spec.name));
    assert_eq!(rust_init.tile_digest, off_init, "{} init tileDigest", spec.name);
    assert_eq!(off_bounds.len(), rust_bounds.len(), "{} boundary count rust={} official={}", spec.name, rust_bounds.len(), off_bounds.len());
    for (i, (got, exp)) in rust_bounds.iter().zip(off_bounds.iter()).enumerate() {
        assert_boundary(spec.name, i, got, exp);
    }
    let last = rust_bounds.last().expect("episode produced a boundary");
    assert_ne!(last.terminal_cause.as_str(), "none", "{} ended without exclusive terminal", spec.name);
    EpisodeOutcome {
        name: spec.name,
        stage: spec.stage,
        seed: spec.seed,
        steps: rust_bounds.len(),
        terminal,
        win,
        elapsed_ms: t0.elapsed().as_millis(),
    }
}

fn nation_episodes() -> Vec<EpisodeSpec> {
    // High-gold later-system tapes first: compact maps, full action cycle.
    let later = [
        EpisodeSpec {
            name: "later-sam-impossible",
            stage: "later-systems",
            map: "Halkidiki",
            map_size: "Compact",
            nations: serde_json::json!(3),
            bots: 2,
            difficulty: "Impossible",
            gold: Some(50_000_000),
            max_ticks: 800,
            seed: "nation-ep-later-sam-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "later-mirv-hard",
            stage: "later-systems",
            map: "FourIslands",
            map_size: "Normal",
            nations: serde_json::json!(2),
            bots: 2,
            difficulty: "Hard",
            gold: Some(50_000_000),
            max_ticks: 800,
            seed: "nation-ep-later-mirv-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "later-diplomacy-easy",
            stage: "later-systems",
            map: "Halkidiki",
            map_size: "Compact",
            nations: serde_json::json!(3),
            bots: 4,
            difficulty: "Easy",
            gold: Some(5_000_000),
            max_ticks: 800,
            seed: "nation-ep-later-dip-0",
            win_tiles_frac: 0.0,
            shaping: 0.2,
        },
    ];

    // Every Nation curriculum stage on its canonical map, two frozen seeds
    // on the cheaper intro/peninsula rows, one seed on heavier rows.
    let curriculum = [
        EpisodeSpec {
            name: "nations-easy-intro-s0",
            stage: "nations-easy-intro",
            map: "FourIslands",
            map_size: "Normal",
            nations: serde_json::json!(3),
            bots: 8,
            difficulty: "Easy",
            gold: None,
            max_ticks: 800,
            seed: "nation-ep-intro-0",
            win_tiles_frac: 0.0,
            shaping: 0.2,
        },
        EpisodeSpec {
            name: "nations-easy-intro-s1",
            stage: "nations-easy-intro",
            map: "FourIslands",
            map_size: "Normal",
            nations: serde_json::json!(3),
            bots: 8,
            difficulty: "Easy",
            gold: None,
            max_ticks: 800,
            seed: "nation-ep-intro-1",
            win_tiles_frac: 0.0,
            shaping: 0.2,
        },
        EpisodeSpec {
            name: "nations-easy-small-s0",
            stage: "nations-easy-small",
            map: "FourIslands",
            map_size: "Normal",
            nations: serde_json::json!(8),
            bots: 20,
            difficulty: "Easy",
            gold: None,
            max_ticks: 700,
            seed: "nation-ep-small-0",
            win_tiles_frac: 0.0,
            shaping: 0.15,
        },
        EpisodeSpec {
            name: "nations-easy-peninsula-s0",
            stage: "nations-easy-peninsula",
            map: "Halkidiki",
            map_size: "Compact",
            nations: serde_json::json!(8),
            bots: 16,
            difficulty: "Easy",
            gold: None,
            max_ticks: 800,
            seed: "nation-ep-pen-0",
            win_tiles_frac: 0.0,
            shaping: 0.12,
        },
        EpisodeSpec {
            name: "nations-easy-peninsula-s1",
            stage: "nations-easy-peninsula",
            map: "Halkidiki",
            map_size: "Compact",
            nations: serde_json::json!(8),
            bots: 16,
            difficulty: "Easy",
            gold: None,
            max_ticks: 800,
            seed: "nation-ep-pen-1",
            win_tiles_frac: 0.0,
            shaping: 0.12,
        },
        EpisodeSpec {
            name: "nations-easy-peninsula-holdout",
            stage: "nations-easy-peninsula",
            map: "Halkidiki",
            map_size: "Compact",
            nations: serde_json::json!(8),
            bots: 16,
            difficulty: "Easy",
            gold: None,
            max_ticks: 700,
            seed: "nation-ep-pen-holdout-0",
            win_tiles_frac: 0.0,
            shaping: 0.12,
        },
        EpisodeSpec {
            name: "nations-easy-mid-s0",
            stage: "nations-easy-mid",
            map: "Halkidiki",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 40,
            difficulty: "Easy",
            gold: None,
            max_ticks: 600,
            seed: "nation-ep-emid-0",
            win_tiles_frac: 0.0,
            shaping: 0.1,
        },
        EpisodeSpec {
            name: "nations-medium-mid-s0",
            stage: "nations-medium-mid",
            map: "Halkidiki",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 40,
            difficulty: "Medium",
            gold: None,
            max_ticks: 600,
            seed: "nation-ep-mmid-0",
            win_tiles_frac: 0.0,
            shaping: 0.05,
        },
        EpisodeSpec {
            name: "nations-hard-mid-s0",
            stage: "nations-hard-mid",
            map: "Halkidiki",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 40,
            difficulty: "Hard",
            gold: None,
            max_ticks: 600,
            seed: "nation-ep-hmid-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "nations-easy-mid-ood-southamerica",
            stage: "nations-easy-mid",
            map: "SouthAmerica",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 40,
            difficulty: "Easy",
            gold: None,
            max_ticks: 500,
            seed: "nation-ep-emid-ood-0",
            win_tiles_frac: 0.0,
            shaping: 0.1,
        },
        EpisodeSpec {
            name: "nations-world-easy-s0",
            stage: "nations-world-easy",
            map: "World",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 20,
            difficulty: "Easy",
            gold: None,
            max_ticks: 400,
            seed: "nation-ep-world-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "nations-world-easy-ood-southamerica",
            stage: "nations-world-easy",
            map: "SouthAmerica",
            map_size: "Normal",
            nations: serde_json::json!(12),
            bots: 20,
            difficulty: "Easy",
            gold: None,
            max_ticks: 400,
            seed: "nation-ep-world-ood-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "nations-impossible-world-s0",
            stage: "nations-impossible-world",
            map: "World",
            map_size: "Normal",
            nations: serde_json::json!("default"),
            bots: 100,
            difficulty: "Impossible",
            gold: None,
            max_ticks: 300,
            seed: "nation-ep-imp-world-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
        EpisodeSpec {
            name: "nations-easy-peninsula-train-balkans",
            stage: "nations-easy-peninsula",
            map: "Balkans",
            map_size: "Compact",
            nations: serde_json::json!(8),
            bots: 16,
            difficulty: "Easy",
            gold: None,
            max_ticks: 600,
            seed: "nation-ep-pen-balkans-0",
            win_tiles_frac: 0.0,
            shaping: 0.12,
        },
        EpisodeSpec {
            name: "nations-impossible-southamerica-s0",
            stage: "nations-impossible-world",
            map: "SouthAmerica",
            map_size: "Normal",
            nations: serde_json::json!("default"),
            bots: 100,
            difficulty: "Impossible",
            gold: None,
            max_ticks: 300,
            seed: "nation-ep-imp-sa-0",
            win_tiles_frac: 0.0,
            shaping: 0.0,
        },
    ];

    later.into_iter().chain(curriculum).collect()
}

#[test]
fn full_episode_nation_curriculum_parity() {
    let mut cases = nation_episodes();
    if let Ok(filter) = std::env::var("NATION_EP") {
        cases.retain(|c| c.name.contains(&filter) || c.stage.contains(&filter));
        assert!(!cases.is_empty(), "NATION_EP={filter} matched no episodes");
    }
    let mut cache = TerrainCache::new(maps_dir());
    let mut outcomes = Vec::new();
    let suite_t0 = Instant::now();
    for spec in &cases {
        eprintln!(
            "episode {} stage={} map={} {} nations={} bots={} {} ticks<={}",
            spec.name, spec.stage, spec.map, spec.map_size, spec.nations, spec.bots, spec.difficulty, spec.max_ticks
        );
        outcomes.push(run_episode(&mut cache, spec));
    }
    let mut terminals: std::collections::BTreeMap<String, usize> = std::collections::BTreeMap::new();
    let mut stages: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut seeds: std::collections::BTreeSet<&str> = std::collections::BTreeSet::new();
    let mut boundaries = 0usize;
    for o in &outcomes {
        *terminals.entry(o.terminal.clone()).or_insert(0) += 1;
        stages.insert(o.stage);
        seeds.insert(o.seed);
        boundaries += o.steps;
        eprintln!(
            "  ok {} steps={} terminal={} win={} {}ms",
            o.name, o.steps, o.terminal, o.win, o.elapsed_ms
        );
    }
    let suite_ms = suite_t0.elapsed().as_millis();
    eprintln!(
        "NATION FULL-EPISODE SUMMARY episodes={} stages={} seeds={} boundaries={} terminals={:?} suite_ms={} wins={}",
        outcomes.len(),
        stages.len(),
        seeds.len(),
        boundaries,
        terminals,
        suite_ms,
        outcomes.iter().filter(|o| o.win).count()
    );
    assert!(!outcomes.is_empty());
    if std::env::var("NATION_EP").is_err() {
        assert!(
            stages.contains("nations-easy-intro")
                && stages.contains("nations-impossible-world")
                && stages.contains("nations-world-easy"),
            "curriculum stage coverage incomplete: {stages:?}"
        );
    }
}
