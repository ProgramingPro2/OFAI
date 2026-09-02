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

fn action_from_row(row: [i32; 5]) -> ActionVec {
    ActionVec {
        action_type: row[0],
        target: row[1],
        region: row[2],
        quantity: row[3],
        unit: row[4],
    }
}

fn parse_action_rows(v: &serde_json::Value) -> Vec<[i32; 5]> {
    let arr = v
        .as_array()
        .unwrap_or_else(|| panic!("NATION_EP_ACTIONS actions must be an array"));
    arr.iter()
        .enumerate()
        .map(|(i, row)| {
            let cells = row.as_array().unwrap_or_else(|| panic!("action {i} must be a 5-int row"));
            assert_eq!(cells.len(), 5, "action {i} must have 5 ints");
            let mut out = [0i32; 5];
            for (j, cell) in cells.iter().enumerate() {
                out[j] = cell.as_i64().unwrap_or_else(|| panic!("action {i}[{j}] not an int")) as i32;
            }
            out
        })
        .collect()
}

/// Load a recorded eval tape (`--actions-out`) or a bare `{actions:[...]}` payload.
/// Returns (actions, optional shaping override, path, optional seed override).
fn load_recorded_tape(seed: &str) -> Option<(Vec<[i32; 5]>, Option<f64>, String, Option<String>)> {
    let path = std::env::var("NATION_EP_ACTIONS").ok()?;
    let raw = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("NATION_EP_ACTIONS={path}: {e}"));
    let v: serde_json::Value =
        serde_json::from_str(&raw).unwrap_or_else(|e| panic!("NATION_EP_ACTIONS json: {e}"));
    let episode = if v.get("episodes").and_then(|e| e.as_array()).is_some() {
        let eps = v["episodes"].as_array().unwrap();
        eps.iter()
            .find(|e| e.get("seed").and_then(|s| s.as_str()) == Some(seed))
            .cloned()
            .or_else(|| {
                if eps.len() == 1 {
                    Some(eps[0].clone())
                } else {
                    None
                }
            })
            .unwrap_or_else(|| panic!("NATION_EP_ACTIONS has no episode for seed {seed}"))
    } else if v.get("actions").and_then(|a| a.as_array()).is_some() {
        v
    } else {
        panic!("NATION_EP_ACTIONS={path} missing actions/episodes");
    };
    let actions = parse_action_rows(
        episode
            .get("actions")
            .unwrap_or_else(|| panic!("NATION_EP_ACTIONS episode missing actions")),
    );
    let shaping = episode
        .pointer("/config/shaping")
        .and_then(|s| s.as_f64())
        .or_else(|| episode.get("shaping").and_then(|s| s.as_f64()));
    let tape_seed = episode
        .get("seed")
        .and_then(|s| s.as_str())
        .or_else(|| episode.pointer("/config/seed").and_then(|s| s.as_str()))
        .map(|s| s.to_string());
    Some((actions, shaping, path, tape_seed))
}

fn official_replay_boundaries(
    raw_cfg: &serde_json::Value,
    actions: &[[i32; 5]],
) -> Result<(String, Vec<serde_json::Value>, Vec<serde_json::Value>), String> {
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
        "census": true,
        "outPath": dump.to_string_lossy(),
    });
    let mut cmd = std::process::Command::new("npx");
    cmd.args(["tsx", "ofai/tests/fidelity.ts", "--replay-stdin"])
        .current_dir(&root)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let out = cmd
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
    let census = v["census"].as_array().cloned().unwrap_or_default();
    Ok((init, bounds, census))
}

fn format_off_census(c: &serde_json::Value) -> String {
    let Some(players) = c.get("players").and_then(|p| p.as_array()) else {
        return c.to_string();
    };
    let player_line = players
        .iter()
        .map(|p| {
            let allies = p
                .get("allies")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let atk = p
                .get("attacks")
                .and_then(|v| v.as_array())
                .map(|arr| {
                    arr.iter()
                        .map(|a| {
                            format!(
                                "{}:t{}@{:.0}{}",
                                a.get("id").and_then(|v| v.as_str()).unwrap_or("?"),
                                a.get("target").and_then(|v| v.as_u64()).unwrap_or(0),
                                a.get("troops").and_then(|v| v.as_f64()).unwrap_or(0.0),
                                if a.get("retreating").and_then(|v| v.as_bool()).unwrap_or(false) {
                                    "R"
                                } else {
                                    ""
                                }
                            )
                        })
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            let embargoes = p
                .get("embargoes")
                .and_then(|v| v.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|v| v.as_u64())
                        .map(|n| n.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                })
                .unwrap_or_default();
            format!(
                "sid={} tiles={} troops={:.0} gold={} atk=[{}] allies=[{}] emb=[{}] rel={}/{}",
                p.get("smallId").and_then(|v| v.as_u64()).unwrap_or(0),
                p.get("tiles").and_then(|v| v.as_u64()).unwrap_or(0),
                p.get("troops").and_then(|v| v.as_f64()).unwrap_or(0.0),
                p.get("gold").and_then(|v| v.as_str()).unwrap_or("?"),
                atk,
                allies,
                embargoes,
                p.get("relFromMe").and_then(|v| v.as_u64()).unwrap_or(0),
                p.get("relToMe").and_then(|v| v.as_u64()).unwrap_or(0)
            )
        })
        .collect::<Vec<_>>()
        .join(" | ");
    player_line
}

fn rust_census_line(game: &ofai_sim::game::Game) -> String {
    let mut players: Vec<_> = game.all_players().collect();
    players.sort_by_key(|p| p.small_id);
    let rows: Vec<String> = players
        .iter()
        .map(|p| {
            let mut allies: Vec<u16> = game
                .alliances()
                .iter()
                .filter(|a| a.requestor == p.id() || a.recipient == p.id())
                .filter_map(|a| game.player(a.other(p.id())).map(|o| o.small_id))
                .collect();
            allies.sort_unstable();
            let atk: Vec<String> = p
                .outgoing
                .iter()
                .filter(|a| a.active)
                .map(|a| {
                    format!(
                        "{}:t{}@{:.0}{}",
                        a.id,
                        a.target_small_id,
                        a.troops,
                        if a.retreating { "R" } else { "" }
                    )
                })
                .collect();
            let mut emb: Vec<u16> = players
                .iter()
                .filter(|o| o.id() != p.id() && p.has_embargo_against(o.id()))
                .map(|o| o.small_id)
                .collect();
            emb.sort_unstable();
            format!(
                "sid={} type={:?} tiles={} troops={:.0} gold={} atk=[{}] allies=[{:?}] emb={:?} alive={}",
                p.small_id,
                p.player_type(),
                p.num_tiles(),
                p.troops_f64(),
                p.gold,
                atk.join(","),
                allies,
                emb,
                p.is_alive()
            )
        })
        .collect();
    let pending: Vec<String> = game
        .incoming_requests()
        .iter()
        .filter(|r| r.status == ofai_sim::diplomacy::RequestStatus::Pending)
        .map(|r| {
            let from = game.player(&r.requestor).map(|p| p.small_id).unwrap_or(0);
            let to = game.player(&r.recipient).map(|p| p.small_id).unwrap_or(0);
            format!("{from}->{to}@{}", r.created_at)
        })
        .collect();
    let human = players.iter().find(|p| p.player_type() == ofai_sim::spec::PlayerType::Human);
    let rels = if let Some(me) = human {
        let mut pairs: Vec<String> = players
            .iter()
            .filter(|p| p.id() != me.id())
            .map(|p| format!("{}:{:?}", p.small_id, me.relation(p.id())))
            .collect();
        pairs.sort();
        pairs.join(",")
    } else {
        String::new()
    };
    format!(
        "{} || pending=[{}] rels=[{}]",
        rows.join(" | "),
        pending.join(","),
        rels
    )
}

fn rust_obs_line(env: &AgentEnv) -> String {
    let obs = env.peek_obs();
    let slots: Vec<String> = env
        .slots()
        .iter()
        .map(|s| {
            s.player_id
                .as_ref()
                .and_then(|id| env.game().player(id).map(|p| p.small_id.to_string()))
                .unwrap_or_else(|| "-".into())
        })
        .collect();
    let attack_row = (ACTION_ATTACK as usize) * NUM_PLAYER_SLOTS;
    let ally_row = (ACTION_ALLY as usize) * NUM_PLAYER_SLOTS;
    let break_row = (ACTION_BREAK_ALLY as usize) * NUM_PLAYER_SLOTS;
    let mut rel_pairs = String::new();
    if let Some(me) = env
        .game()
        .all_players()
        .find(|p| p.player_type() == ofai_sim::spec::PlayerType::Human)
    {
        let mut parts = Vec::new();
        for p in env.game().all_players() {
            if p.id() == me.id() {
                continue;
            }
            let a = me.relation_value(p.id());
            let b = p.relation_value(me.id());
            if a != 0.0 || b != 0.0 {
                parts.push(format!("{}:{a:.1}/{b:.1}", p.small_id));
            }
        }
        rel_pairs = parts.join(",");
    }
    format!(
        "am={:?} slots=[{}] atk={:?} ally={:?} brk={:?} unit={:?} build={:?} boat={:?} relAB=[{}]",
        obs.action_mask,
        slots.join(","),
        &obs.target_masks[attack_row..attack_row + NUM_PLAYER_SLOTS],
        &obs.target_masks[ally_row..ally_row + NUM_PLAYER_SLOTS],
        &obs.target_masks[break_row..break_row + NUM_PLAYER_SLOTS],
        obs.unit_mask,
        obs.build_regions.iter().filter(|&&b| b != 0).count(),
        obs.boat_regions.iter().filter(|&&b| b != 0).count(),
        rel_pairs
    )
}

fn format_off_obs(c: &serde_json::Value) -> String {
    let slots = c
        .get("slots")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_u64())
                .map(|n| n.to_string())
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let rels = c
        .get("relations")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|r| {
                    format!(
                        "{}:{}",
                        r.get("sid").and_then(|v| v.as_u64()).unwrap_or(0),
                        r.get("rel").and_then(|v| v.as_u64()).unwrap_or(0)
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    let pending = c
        .get("pending")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .map(|r| {
                    format!(
                        "{}->{}@{}",
                        r.get("from").and_then(|v| v.as_u64()).unwrap_or(0),
                        r.get("to").and_then(|v| v.as_u64()).unwrap_or(0),
                        r.get("at").and_then(|v| v.as_u64()).unwrap_or(0)
                    )
                })
                .collect::<Vec<_>>()
                .join(",")
        })
        .unwrap_or_default();
    format!(
        "am={:?} slots=[{}] atk={:?} ally={:?} brk={:?} unit={:?} rels=[{}] pending=[{}]",
        json_u8_vec(c.get("actionMask")),
        slots,
        json_u8_vec(c.get("attackTargets")),
        json_u8_vec(c.get("allyTargets")),
        json_u8_vec(c.get("breakTargets")),
        json_u8_vec(c.get("unitMask")),
        rels,
        pending
    )
}

fn json_u8_vec(v: Option<&serde_json::Value>) -> Vec<u8> {
    v.and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|x| x.as_u64().map(|n| n as u8)).collect())
        .unwrap_or_default()
}

fn assert_boundary(
    label: &str,
    i: usize,
    got: &ofai_sim::digest::DecisionBoundary,
    exp: &serde_json::Value,
    rust_census: Option<&str>,
    off_census: Option<&str>,
    prev_rust: Option<&str>,
    prev_off: Option<&str>,
    action: Option<&[i32; 5]>,
    rust_obs: Option<&str>,
    off_obs: Option<&str>,
) {
    let field = |k: &str| -> String {
        match &exp[k] {
            serde_json::Value::Null => "null".into(),
            v => v.to_string(),
        }
    };
    if got.tile_digest != exp["tileDigest"].as_str().unwrap_or("") {
        panic!(
            "{label} step {i} FIRST MISMATCH tileDigest rust_tick={} off_tick={} rust_hash={} off_hash={} action={:?} rust_accepted={} off_accepted={}\nprev_rust={}\nprev_off={}\nrust_census={}\noff_census={}",
            got.tick,
            field("tick"),
            field("coreHash"),
            got.core_hash.map(|h| h.to_string()).unwrap_or_else(|| "null".into()),
            action,
            got.action_accepted,
            exp["actionAccepted"].as_bool().unwrap_or(false),
            prev_rust.unwrap_or("-"),
            prev_off.unwrap_or("-"),
            rust_census.unwrap_or("-"),
            off_census.unwrap_or("-"),
        );
    }
    assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "{label} step {i} FIRST MISMATCH coreHash");
    if got.obs_digest != exp["obsDigest"].as_str().unwrap_or("") {
        panic!(
            "{label} step {i} FIRST MISMATCH obsDigest action={:?} rust_accepted={} off_accepted={}\nrust_obs={}\noff_obs={}\nrust_census={}\noff_census={}",
            action,
            got.action_accepted,
            exp["actionAccepted"].as_bool().unwrap_or(false),
            rust_obs.unwrap_or("-"),
            off_obs.unwrap_or("-"),
            rust_census.unwrap_or("-"),
            off_census.unwrap_or("-"),
        );
    }
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
    let recorded = load_recorded_tape(spec.seed);
    let mut raw = episode_raw(spec);
    if let Some((_, Some(shaping), _, _)) = &recorded {
        raw["shaping"] = serde_json::json!(shaping);
    }
    if let Some((_, _, _, Some(tape_seed))) = &recorded {
        raw["seed"] = serde_json::json!(tape_seed);
    }
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
    let mut rust_censuses = Vec::new();
    let mut rust_obs_lines = Vec::new();
    let mut terminal = "none".to_string();
    let mut win = false;
    let step_cap = std::env::var("NATION_EP_STEPS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    if let Some((tape, _, path, tape_seed)) = &recorded {
        eprintln!(
            "  {} replaying {} recorded actions from {path} seed={}",
            spec.name,
            tape.len(),
            tape_seed.as_deref().unwrap_or(spec.seed)
        );
    }
    loop {
        let i = actions.len();
        if i >= step_cap {
            break;
        }
        let act = if let Some((tape, _, _, _)) = &recorded {
            if i >= tape.len() {
                break;
            }
            action_from_row(tape[i])
        } else {
            episode_action(rust.peek_obs(), i)
        };
        actions.push([act.action_type, act.target, act.region, act.quantity, act.unit]);
        let step = rust.step(act);
        if let Some(reason) = step.info.fallback.as_deref() {
            panic!("{} rust fallback at step {i} tick={}: {reason}", spec.name, step.info.tick);
        }
        rust_bounds.push(rust.strong_state_digest());
        rust_censuses.push(rust_census_line(rust.game()));
        rust_obs_lines.push(rust_obs_line(&rust));
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
    let (off_init, off_bounds, off_census) = official_replay_boundaries(&raw, &actions)
        .unwrap_or_else(|e| panic!("{} official replay: {e}", spec.name));
    assert_eq!(rust_init.tile_digest, off_init, "{} init tileDigest", spec.name);
    assert_eq!(off_bounds.len(), rust_bounds.len(), "{} boundary count rust={} official={}", spec.name, rust_bounds.len(), off_bounds.len());
    for (i, (got, exp)) in rust_bounds.iter().zip(off_bounds.iter()).enumerate() {
        let off_c = off_census.get(i).map(format_off_census);
        let off_o = off_census.get(i).map(format_off_obs);
        let prev_off = if i > 0 {
            off_census.get(i - 1).map(format_off_census)
        } else {
            None
        };
        assert_boundary(
            spec.name,
            i,
            got,
            exp,
            Some(&rust_censuses[i]),
            off_c.as_deref(),
            if i > 0 { Some(rust_censuses[i - 1].as_str()) } else { None },
            prev_off.as_deref(),
            actions.get(i),
            Some(&rust_obs_lines[i]),
            off_o.as_deref(),
        );
    }
    let last = rust_bounds.last().expect("episode produced a boundary");
    if step_cap == usize::MAX {
        assert_ne!(last.terminal_cause.as_str(), "none", "{} ended without exclusive terminal", spec.name);
    }
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
        // Canonical late / greedy-holdout[0] tape: same seed as holdout[0],
        // maxTicks=12000. Mask-aware actions are the default; set
        // NATION_EP_ACTIONS to replay an eval `--actions-out` greedy tape
        // (shaping is taken from the tape config so holdout 0.0 matches).
        // Combat sigmoid uses js_exp. Land-attack init cancels opposing
        // incoming before same-target absorb. Nation embargo/relation
        // scans match official `mg.players()` (alive-only). Alliance
        // accept clears temporary embargoes only on the mutual-request
        // shortcut. Captured Defense Posts are deleted, not transferred.
        // Uncaptured trade ships use ship-owner canTrade vs dest, not
        // the source port owner (official TradeShipExecution.tick).
        EpisodeSpec {
            name: "nations-easy-small-late",
            stage: "nations-easy-small",
            map: "FourIslands",
            map_size: "Normal",
            nations: serde_json::json!(8),
            bots: 20,
            difficulty: "Easy",
            gold: None,
            max_ticks: 12000,
            seed: "36cd15ca77bc7e9f23424dab8a6d1ae3",
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
fn recorded_eval_tape_selects_matching_seed() {
    let payload = serde_json::json!({
        "stage": "nations-easy-small",
        "suite": "holdout",
        "source": "eval",
        "episodes": [
            {
                "seed": "other-seed",
                "config": {"shaping": 0.2},
                "actions": [[0, 0, 0, 0, 0]],
            },
            {
                "seed": "36cd15ca77bc7e9f23424dab8a6d1ae3",
                "config": {"shaping": 0.0},
                "actions": [[1, 0, 12, 0, 0], [2, 3, 4, 1, 0]],
            }
        ]
    });
    let path = std::env::temp_dir().join(format!(
        "ofai-nation-tape-parse-{}.json",
        std::process::id()
    ));
    std::fs::write(&path, payload.to_string()).unwrap();
    std::env::set_var("NATION_EP_ACTIONS", path.to_string_lossy().as_ref());
    let (actions, shaping, loaded, tape_seed) =
        load_recorded_tape("36cd15ca77bc7e9f23424dab8a6d1ae3").expect("tape");
    std::env::remove_var("NATION_EP_ACTIONS");
    let _ = std::fs::remove_file(&path);
    assert_eq!(loaded, path.to_string_lossy());
    assert_eq!(shaping, Some(0.0));
    assert_eq!(tape_seed.as_deref(), Some("36cd15ca77bc7e9f23424dab8a6d1ae3"));
    assert_eq!(actions, vec![[1, 0, 12, 0, 0], [2, 3, 4, 1, 0]]);
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
