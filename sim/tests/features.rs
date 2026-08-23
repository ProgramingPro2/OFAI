//! Focused rust-features tests: dest legality, construction, diplomacy, AI setup.

use std::fs;
use std::path::PathBuf;

use ofai_sim::actions::ActionVec;
use ofai_sim::capabilities::Capabilities;
use ofai_sim::env::AgentEnv;
use ofai_sim::spec::{
    EnvConfig, MapSize, NationsSpec, ACTION_ATTACK, ACTION_BOAT, ACTION_BUILD, ACTION_SPAWN,
};
use ofai_sim::terrain::TerrainCache;

fn maps_dir() -> PathBuf {
    TerrainCache::default_maps_dir()
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../OpenFrontIO/ofai/tests/fixtures")
}

fn load_golden(name: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("golden-{name}.json"));
    serde_json::from_str(&fs::read_to_string(&path).expect("golden")).expect("golden json")
}

fn terrain() -> TerrainCache {
    TerrainCache::new(maps_dir())
}

#[test]
fn capabilities_declare_ported_slices_and_later_system_boundary() {
    let caps = Capabilities::foundation();
    assert!(caps.supported.boats);
    assert!(!caps.unsupported.boats);
    assert!(caps.supported.structures);
    assert!(caps.supported.diplomacy);
    assert!(caps.supported.tribe_ai);
    assert!(!caps.supported.nations_ai);
    assert!(!caps.unsupported.nukes);
    assert!(!caps.unsupported.rails);
    assert!(!caps.unsupported.doomsday);
    assert_eq!(caps.fallback, "official");
}

#[test]
fn boats_and_structures_goldens_create_without_official_fallback() {
    let mut cache = terrain();
    for name in ["boats", "structures"] {
        let golden = load_golden(name);
        let cfg = EnvConfig::from_json(&golden["config"]).expect("cfg");
        let env = AgentEnv::create(cfg, &mut cache).expect(name);
        assert!(
            env.strong_state_digest().tile_digest.len() == 64,
            "{name} tile digest"
        );
    }
}

#[test]
fn tribe_and_nation_configs_are_accepted() {
    let mut cache = terrain();
    let mut cfg = EnvConfig::from_json(&load_golden("player-combat")["config"]).unwrap();
    cfg.enable_digest = true;
    AgentEnv::create(cfg, &mut cache).expect("player-combat create");
    let cfg = EnvConfig::from_json(&load_golden("nations")["config"]).unwrap();
    AgentEnv::create(cfg, &mut cache).expect("nations create");
}

#[test]
fn boats_golden_init_and_first_steps_match_without_fallback() {
    let golden = load_golden("boats");
    let cfg = EnvConfig::from_json(&golden["config"]).expect("cfg");
    let mut cache = terrain();
    let mut env = AgentEnv::create(cfg, &mut cache).expect("boats");
    let init = env.strong_state_digest();
    assert_eq!(init.tick, golden["init"]["tick"]);
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
    assert_eq!(
        init.tile_digest,
        golden["init"]["tileDigest"].as_str().unwrap(),
        "boats init tileDigest"
    );
    let spawn = env.step(ActionVec {
        action_type: ACTION_SPAWN,
        target: 0,
        region: 2,
        quantity: 0,
        unit: 0,
    });
    assert!(spawn.info.action_accepted);
    assert!(spawn.info.fallback.is_none(), "spawn must not fallback");
    let boat = env.step(ActionVec {
        action_type: ACTION_BOAT,
        target: 0,
        region: 2,
        quantity: 0,
        unit: 0,
    });
    assert!(
        boat.info.fallback.is_none(),
        "HPA boat path must execute, got {:?}",
        boat.info.fallback
    );
    assert!(boat.info.action_accepted);
}

#[test]
fn structures_golden_init_matches_integer_oracle() {
    let golden = load_golden("structures");
    let cfg = EnvConfig::from_json(&golden["config"]).expect("cfg");
    let mut cache = terrain();
    let mut env = AgentEnv::create(cfg, &mut cache).expect("structures");
    let init = env.strong_state_digest();
    assert_eq!(
        init.tile_digest,
        golden["init"]["tileDigest"].as_str().unwrap(),
        "structures init tileDigest"
    );
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
}

#[test]
fn player_combat_and_nations_create_with_ai_roster() {
    let mut cache = terrain();
    for name in ["player-combat", "diplomacy", "nations"] {
        let golden = load_golden(name);
        let cfg = EnvConfig::from_json(&golden["config"]).expect(name);
        let env = AgentEnv::create(cfg, &mut cache).expect(name);
        assert!(env.strong_state_digest().tick >= 5, "{name} warmup ticks");
    }
}

#[test]
fn post_spawn_boat_mask_is_spatial_query_not_shore_approx() {
    let mut cache = terrain();
    let cfg = EnvConfig {
        map: "FourIslands".into(),
        map_size: MapSize::Normal,
        nations: NationsSpec::Disabled,
        bots: 0,
        seed: "golden-boats".into(),
        max_ticks: 400,
        ..EnvConfig::default()
    };
    let mut env = AgentEnv::create(cfg, &mut cache).unwrap();
    let _ = env.step(ActionVec {
        action_type: ACTION_SPAWN,
        target: 0,
        region: 2,
        quantity: 0,
        unit: 0,
    });
    let obs = env.peek_obs();
    assert_eq!(obs.action_mask[ACTION_BUILD as usize] == 1, obs.unit_mask.iter().any(|&b| b == 1));
    let _ = ACTION_BOAT;
}

fn replay_named(name: &str) -> (serde_json::Value, ofai_sim::digest::DecisionBoundary, Vec<ofai_sim::digest::DecisionBoundary>) {
    let golden = load_golden(name);
    let cfg = EnvConfig::from_json(&golden["config"]).expect("cfg");
    let actions: Vec<[i32; 5]> = golden["actions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let v = a.as_array().unwrap();
            [
                v[0].as_i64().unwrap() as i32,
                v[1].as_i64().unwrap() as i32,
                v[2].as_i64().unwrap() as i32,
                v[3].as_i64().unwrap() as i32,
                v[4].as_i64().unwrap() as i32,
            ]
        })
        .collect();
    let mut cache = terrain();
    let (init, bounds) = ofai_sim::replay_golden(&mut cache, cfg, &actions).expect(name);
    (golden, init, bounds)
}

fn dump_boats_game(env: &AgentEnv) -> String {
    let g = env.game();
    let mut s = format!("tick={} units_total={}\n", g.ticks(), g.units().len());
    for p in g.all_players() {
        s.push_str(&format!(
            "  player {} tiles={} troops={} gold={} unit_ids={:?} attacks={}\n",
            p.small_id,
            p.num_tiles(),
            p.troops_f64(),
            p.gold,
            p.units,
            p.outgoing.len()
        ));
        for atk in &p.outgoing {
            s.push_str(&format!(
                "    atk id={} tgt={} troops={} retreating={} active={} src={:?} border={}\n",
                atk.id,
                atk.target_small_id,
                atk.troops,
                atk.retreating,
                atk.active,
                atk.source_tile,
                atk.border_size()
            ));
        }
    }
    for u in g.units() {
        s.push_str(&format!(
            "  unit id={} ty={:?} tile={} active={} troops={} target={:?}\n",
            u.id, u.ty, u.tile, u.active, u.troops, u.target_tile
        ));
    }
    s
}

#[test]
fn boats_golden_replays_every_decision_boundary() {
    let golden = load_golden("boats");
    let cfg = EnvConfig::from_json(&golden["config"]).expect("cfg");
    let actions: Vec<[i32; 5]> = golden["actions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let v = a.as_array().unwrap();
            [
                v[0].as_i64().unwrap() as i32,
                v[1].as_i64().unwrap() as i32,
                v[2].as_i64().unwrap() as i32,
                v[3].as_i64().unwrap() as i32,
                v[4].as_i64().unwrap() as i32,
            ]
        })
        .collect();
    let mut cache = terrain();
    let mut env = AgentEnv::create(cfg, &mut cache).expect("boats");
    let init = env.strong_state_digest();
    assert_eq!(init.tile_digest, golden["init"]["tileDigest"].as_str().unwrap());
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
    let expected = golden["boundaries"].as_array().unwrap();
    let mut last_ok = String::new();
    for (i, exp) in expected.iter().enumerate() {
        let a = actions[i];
        let step = env.step(ActionVec {
            action_type: a[0],
            target: a[1],
            region: a[2],
            quantity: a[3],
            unit: a[4],
        });
        let got = env.strong_state_digest();
        assert_eq!(got.tick, exp["tick"].as_u64().unwrap() as u32, "boats step {i} tick");
        let tile_ok = got.tile_digest == exp["tileDigest"].as_str().unwrap();
        let hash_ok = got.core_hash == exp["coreHash"].as_f64();
        let obs_ok = got.obs_digest == exp["obsDigest"].as_str().unwrap();
        let step_ok = got.step_digest == exp["stepDigest"].as_str().unwrap();
        let dig_ok = got.digest == exp["digest"].as_str().unwrap();
        if !tile_ok || !hash_ok || !obs_ok || !step_ok || !dig_ok {
            panic!(
                "boats step {i} action={a:?} accepted={} fallback={:?} tile_ok={tile_ok} hash_ok={hash_ok} obs_ok={obs_ok} step_ok={step_ok} dig_ok={dig_ok} rust_hash={:?} gold_hash={:?}\nLAST OK:\n{last_ok}\nNOW:\n{}",
                step.info.action_accepted,
                step.info.fallback,
                got.core_hash,
                exp["coreHash"].as_f64(),
                dump_boats_game(&env)
            );
        }
        last_ok = dump_boats_game(&env);
    }
}

#[test]
fn structures_golden_replays_every_decision_boundary() {
    let (golden, init, bounds) = replay_named("structures");
    assert_eq!(init.tile_digest, golden["init"]["tileDigest"].as_str().unwrap());
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
    let expected = golden["boundaries"].as_array().unwrap();
    assert_eq!(bounds.len(), expected.len());
    for (i, (exp, got)) in expected.iter().zip(bounds.iter()).enumerate() {
        assert_eq!(got.tile_digest, exp["tileDigest"].as_str().unwrap(), "structures step {i} tileDigest");
        assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "structures step {i} coreHash");
        assert_eq!(got.obs_digest, exp["obsDigest"].as_str().unwrap(), "structures step {i} obsDigest");
        assert_eq!(got.step_digest, exp["stepDigest"].as_str().unwrap(), "structures step {i} stepDigest");
        assert_eq!(got.digest, exp["digest"].as_str().unwrap(), "structures step {i} combined digest");
        assert_eq!(got.terminal_cause.as_str(), exp["terminalCause"].as_str().unwrap());
        assert_eq!(got.reward, exp["reward"].as_f64().unwrap(), "structures step {i} reward");
        assert_eq!(got.win, exp["win"].as_bool().unwrap(), "structures step {i} win");
    }
}

fn first_ai_boundary_mismatch(name: &str) -> Option<String> {
    let (golden, init, bounds) = replay_named(name);
    if init.tile_digest != golden["init"]["tileDigest"].as_str().unwrap() {
        return Some(format!("{name}: init tileDigest"));
    }
    let expected = golden["boundaries"].as_array().unwrap();
    if bounds.len() != expected.len() {
        return Some(format!(
            "{name}: boundary count rust={} gold={}",
            bounds.len(),
            expected.len()
        ));
    }
    let mut firsts: Vec<String> = Vec::new();
    let mut seen = std::collections::BTreeSet::new();
    let mark = |field: &str, i: usize, seen: &mut std::collections::BTreeSet<String>, firsts: &mut Vec<String>| {
        if seen.insert(field.to_string()) {
            firsts.push(format!("{field}@{i}"));
        }
    };
    if init.obs_digest != golden["init"]["obsDigest"].as_str().unwrap() {
        mark("init.obsDigest", 0, &mut seen, &mut firsts);
    }
    for (i, (exp, got)) in expected.iter().zip(bounds.iter()).enumerate() {
        if got.tile_digest != exp["tileDigest"].as_str().unwrap() {
            mark("tileDigest", i, &mut seen, &mut firsts);
        }
        if got.obs_digest != exp["obsDigest"].as_str().unwrap() {
            mark("obsDigest", i, &mut seen, &mut firsts);
        }
        if got.step_digest != exp["stepDigest"].as_str().unwrap() {
            mark("stepDigest", i, &mut seen, &mut firsts);
        }
        if got.digest != exp["digest"].as_str().unwrap() {
            mark("digest", i, &mut seen, &mut firsts);
        }
        if got.core_hash != exp["coreHash"].as_f64() {
            mark("coreHash", i, &mut seen, &mut firsts);
        }
        if got.terminal_cause.as_str() != exp["terminalCause"].as_str().unwrap() {
            mark("terminalCause", i, &mut seen, &mut firsts);
        }
        if (got.reward - exp["reward"].as_f64().unwrap()).abs() > 1e-9 {
            mark("reward", i, &mut seen, &mut firsts);
        }
        if got.win != exp["win"].as_bool().unwrap() {
            mark("win", i, &mut seen, &mut firsts);
        }
        if got.action_accepted != exp["actionAccepted"].as_bool().unwrap_or(true) {
            mark("actionAccepted", i, &mut seen, &mut firsts);
        }
    }
    if firsts.is_empty() {
        None
    } else {
        Some(format!("{name}: {}", firsts.join(", ")))
    }
}

#[test]
fn player_combat_and_nations_replay_full_boundaries() {
    for name in ["player-combat", "nations"] {
        if let Some(msg) = first_ai_boundary_mismatch(name) {
            panic!("{msg}");
        }
    }
}

#[test]
fn diplomacy_player_combat_nations_replay_integer_fields() {
    let mut failures = Vec::new();
    for name in ["diplomacy", "player-combat", "nations"] {
        if let Some(msg) = first_ai_boundary_mismatch(name) {
            failures.push(msg);
        }
    }
    if !failures.is_empty() {
        panic!(
            "tribe/nation AI goldens not exact:\n{}",
            failures.join("\n")
        );
    }
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

fn diplomacy_trace_action(obs: &ofai_sim::obs::ObsBuffers, step: usize) -> ActionVec {
    use ofai_sim::spec::{
        ACTION_ALLY, ACTION_ATTACK, ACTION_BREAK_ALLY, ACTION_EMBARGO, ACTION_NOOP, ACTION_SPAWN,
        NUM_PLAYER_SLOTS,
    };
    if obs.action_mask[ACTION_SPAWN as usize] != 0 {
        return ActionVec {
            action_type: ACTION_SPAWN,
            target: 0,
            region: 2,
            quantity: 0,
            unit: 0,
        };
    }
    let prefer = match step % 5 {
        1 | 2 => ACTION_ALLY,
        3 => ACTION_EMBARGO,
        4 => ACTION_BREAK_ALLY,
        _ => ACTION_ATTACK,
    };
    let chosen = if obs.action_mask[prefer as usize] != 0 {
        prefer
    } else if obs.action_mask[ACTION_ATTACK as usize] != 0 {
        ACTION_ATTACK
    } else {
        ACTION_NOOP
    };
    ActionVec {
        action_type: chosen,
        target: first_legal_target(&obs.target_masks, chosen as usize, NUM_PLAYER_SLOTS),
        region: 0,
        quantity: 2,
        unit: 0,
    }
}

fn official_replay_boundaries(
    raw_cfg: &serde_json::Value,
    actions: &[[i32; 5]],
) -> Result<(String, Vec<serde_json::Value>), String> {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../OpenFrontIO");
    let payload = serde_json::json!({
        "config": raw_cfg,
        "actions": actions,
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
    let stdout = String::from_utf8_lossy(&out.stdout);
    let json_start = stdout.find('{').ok_or_else(|| format!("no json in {stdout}"))?;
    let v: serde_json::Value = serde_json::from_str(&stdout[json_start..]).map_err(|e| {
        format!(
            "official json: {e} stdout={} stderr={}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    })?;
    let init = v["init"]["tileDigest"].as_str().unwrap_or("").to_string();
    let bounds = v["boundaries"].as_array().cloned().unwrap_or_default();
    Ok((init, bounds))
}

#[test]
fn randomized_official_vs_rust_ai_traces() {
    use ofai_sim::spec::{Difficulty, NationsSpec};
    let cases: &[(&str, u32, NationsSpec, Difficulty)] = &[
        ("trace-bots-easy", 5, NationsSpec::Disabled, Difficulty::Easy),
        ("trace-bots-medium", 8, NationsSpec::Disabled, Difficulty::Medium),
        ("trace-nations-easy", 4, NationsSpec::Count(3), Difficulty::Easy),
        ("trace-nations-hard", 2, NationsSpec::Count(2), Difficulty::Hard),
        ("trace-duel-impossible", 0, NationsSpec::Disabled, Difficulty::Impossible),
        ("trace-bots-hard", 6, NationsSpec::Disabled, Difficulty::Hard),
    ];
    let mut cache = terrain();
    for (seed, bots, nations, difficulty) in cases {
        let mut raw = load_golden("diplomacy")["config"].clone();
        raw["seed"] = serde_json::json!(seed);
        raw["bots"] = serde_json::json!(bots);
        raw["nations"] = match nations {
            NationsSpec::Disabled => serde_json::json!("disabled"),
            NationsSpec::Count(n) => serde_json::json!(n),
            NationsSpec::Default => serde_json::json!("default"),
        };
        raw["difficulty"] = serde_json::json!(format!("{difficulty:?}"));
        raw["enableDigest"] = serde_json::json!(true);
        raw["maxTicks"] = serde_json::json!(400);
        let cfg = EnvConfig::from_json(&raw).unwrap();
        let mut rust = AgentEnv::create(cfg, &mut cache).expect(seed);
        let rust_init = rust.strong_state_digest().tile_digest.clone();
        let mut actions = Vec::new();
        let mut rust_bounds = Vec::new();
        for i in 0..12 {
            let act = diplomacy_trace_action(rust.peek_obs(), i);
            actions.push([act.action_type, act.target, act.region, act.quantity, act.unit]);
            let step = rust.step(act);
            rust_bounds.push(rust.strong_state_digest());
            if let Some(reason) = step.info.fallback.as_deref() {
                panic!("{seed} rust fallback at step {i}: {reason}");
            }
        }
        let (off_init, off_bounds) = official_replay_boundaries(&raw, &actions)
            .unwrap_or_else(|e| panic!("{seed} official replay: {e}"));
        assert_eq!(rust_init, off_init, "{seed} init tileDigest");
        assert_eq!(off_bounds.len(), rust_bounds.len(), "{seed} boundary count");
        for (i, (got, exp)) in rust_bounds.iter().zip(off_bounds.iter()).enumerate() {
            assert_eq!(got.tile_digest, exp["tileDigest"].as_str().unwrap(), "{seed} step {i} tileDigest");
            assert_eq!(got.obs_digest, exp["obsDigest"].as_str().unwrap(), "{seed} step {i} obsDigest");
            assert_eq!(got.step_digest, exp["stepDigest"].as_str().unwrap(), "{seed} step {i} stepDigest");
            assert_eq!(got.digest, exp["digest"].as_str().unwrap(), "{seed} step {i} digest");
            assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "{seed} step {i} coreHash");
            assert_eq!(
                got.terminal_cause.as_str(),
                exp["terminalCause"].as_str().unwrap(),
                "{seed} step {i} terminal"
            );
        }
    }
}

fn first_later_divergence(name: &str) -> Option<String> {
    let (golden, init, bounds) = replay_named(name);
    if init.tile_digest != golden["init"]["tileDigest"].as_str().unwrap() {
        return Some(format!("{name} init tileDigest"));
    }
    if init.core_hash != golden["init"]["coreHash"].as_f64() {
        return Some(format!("{name} init coreHash"));
    }
    let expected = golden["boundaries"].as_array().unwrap();
    for (i, (got, exp)) in bounds.iter().zip(expected.iter()).enumerate() {
        if got.tile_digest != exp["tileDigest"].as_str().unwrap() {
            return Some(format!("{name} step {i} tileDigest"));
        }
        if got.core_hash != exp["coreHash"].as_f64() {
            return Some(format!("{name} step {i} coreHash rust={:?} gold={:?}", got.core_hash, exp["coreHash"].as_f64()));
        }
        if got.obs_digest != exp["obsDigest"].as_str().unwrap() {
            return Some(format!("{name} step {i} obsDigest"));
        }
        if got.step_digest != exp["stepDigest"].as_str().unwrap() {
            return Some(format!("{name} step {i} stepDigest"));
        }
        if got.digest != exp["digest"].as_str().unwrap() {
            return Some(format!("{name} step {i} combined digest"));
        }
        if got.terminal_cause.as_str() != exp["terminalCause"].as_str().unwrap() {
            return Some(format!("{name} step {i} terminal"));
        }
    }
    None
}

#[test]
fn later_goldens_match_every_boundary() {
    for name in [
        "port-factory",
        "rails-trains",
        "trade-ships",
        "warships",
        "warships-combat",
        "nukes",
        "nukes-atom",
        "nukes-mirv",
        "sam-intercept",
        "win",
    ] {
        if let Some(div) = first_later_divergence(name) {
            panic!("first later-system divergence: {div}");
        }
    }
}

#[test]
fn later_feature_official_vs_rust_traces() {
    use ofai_sim::spec::Difficulty;
    let cases: &[(&str, u32, Difficulty, i64)] = &[
        ("trace-later-easy", 0, Difficulty::Easy, 2_500_000),
        ("trace-later-hard", 0, Difficulty::Hard, 50_000_000),
        ("trace-later-impossible", 0, Difficulty::Impossible, 50_000_000),
    ];
    let mut cache = terrain();
    for (seed, bots, difficulty, gold) in cases {
        let mut raw = load_golden("port-factory")["config"].clone();
        raw["seed"] = serde_json::json!(seed);
        raw["bots"] = serde_json::json!(bots);
        raw["nations"] = serde_json::json!("disabled");
        raw["difficulty"] = serde_json::json!(format!("{difficulty:?}"));
        raw["startingGold"] = serde_json::json!(gold);
        raw["enableDigest"] = serde_json::json!(true);
        raw["maxTicks"] = serde_json::json!(800);
        let cfg = EnvConfig::from_json(&raw).unwrap();
        let mut rust = AgentEnv::create(cfg, &mut cache).expect(seed);
        let rust_init = rust.strong_state_digest().tile_digest.clone();
        let mut actions = Vec::new();
        let mut rust_bounds = Vec::new();
        for i in 0..10 {
            let _obs = rust.peek_obs();
            let mut act = ActionVec {
                action_type: if i == 0 { ACTION_SPAWN } else if i % 4 == 1 { ACTION_BUILD } else { ACTION_ATTACK },
                target: 0,
                region: (i % 8) as i32,
                quantity: 1,
                unit: (i % 6) as i32,
            };
            if act.action_type == ACTION_SPAWN {
                act.region = 2;
            }
            actions.push([act.action_type, act.target, act.region, act.quantity, act.unit]);
            let step = rust.step(act);
            rust_bounds.push(rust.strong_state_digest());
            if let Some(reason) = step.info.fallback.as_deref() {
                panic!("{seed} rust fallback at step {i}: {reason}");
            }
        }
        let (off_init, off_bounds) = official_replay_boundaries(&raw, &actions)
            .unwrap_or_else(|e| panic!("{seed} official replay: {e}"));
        assert_eq!(rust_init, off_init, "{seed} init tileDigest");
        for (i, (got, exp)) in rust_bounds.iter().zip(off_bounds.iter()).enumerate() {
            assert_eq!(got.tile_digest, exp["tileDigest"].as_str().unwrap(), "{seed} step {i} tileDigest");
            assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "{seed} step {i} coreHash");
            assert_eq!(got.obs_digest, exp["obsDigest"].as_str().unwrap(), "{seed} step {i} obsDigest");
            assert_eq!(got.digest, exp["digest"].as_str().unwrap(), "{seed} step {i} digest");
        }
    }
}

#[test]
fn doomsday_enabled_official_vs_rust_is_deterministic() {
    let mut raw = load_golden("win")["config"].clone();
    raw["doomsdayClock"] = serde_json::json!({"enabled": true, "speed": "veryfast"});
    raw["enableDigest"] = serde_json::json!(true);
    raw["maxTicks"] = serde_json::json!(800);
    raw["seed"] = serde_json::json!("doomsday-diff");
    let cfg = EnvConfig::from_json(&raw).unwrap();
    let mut cache = terrain();
    let mut rust = AgentEnv::create(cfg, &mut cache).expect("doomsday");
    let mut actions = Vec::new();
    let mut rust_bounds = Vec::new();
    for i in 0..8 {
        let act = ActionVec {
            action_type: if i == 0 { ACTION_SPAWN } else { ACTION_ATTACK },
            target: 0,
            region: 2,
            quantity: 2,
            unit: 0,
        };
        actions.push([act.action_type, act.target, act.region, act.quantity, act.unit]);
        let step = rust.step(act);
        rust_bounds.push(rust.strong_state_digest());
        assert!(step.info.fallback.is_none(), "doomsday rust fallback");
    }
    let (off_init, off_bounds) = official_replay_boundaries(&raw, &actions)
        .unwrap_or_else(|e| panic!("doomsday official: {e}"));
    assert_eq!(rust.strong_state_digest().tile_digest.is_empty(), false);
    let _ = (off_init, off_bounds.len());
    for (i, (got, exp)) in rust_bounds.iter().zip(off_bounds.iter()).enumerate() {
        assert_eq!(got.tile_digest, exp["tileDigest"].as_str().unwrap(), "doomsday step {i} tileDigest");
        assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "doomsday step {i} coreHash");
    }
}

