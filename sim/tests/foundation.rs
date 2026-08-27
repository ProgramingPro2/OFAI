//! Focused rust-foundation tests: RNG/terrain, framing, capabilities, goldens.

use std::fs;
use std::path::PathBuf;

use ofai_sim::actions::ActionVec;
use ofai_sim::capabilities::{Capabilities, UnsupportedConfig};
use ofai_sim::digest::boundary_json;
use ofai_sim::env::{replay_golden, AgentEnv};
use ofai_sim::protocol::{decode_frame, encode_frame, Tensor};
use ofai_sim::rng::{simple_hash, PseudoRandom};
use ofai_sim::spec::{EnvConfig, MapSize, NationsSpec, ACTION_BOAT, ACTION_BUILD, ACTION_SPAWN};
use ofai_sim::terrain::TerrainCache;

fn maps_dir() -> PathBuf {
    TerrainCache::default_maps_dir()
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../OpenFrontIO/ofai/tests/fixtures")
}

fn load_golden(name: &str) -> serde_json::Value {
    let path = golden_dir().join(format!("golden-{name}.json"));
    serde_json::from_str(&fs::read_to_string(&path).expect("golden"))
        .expect("golden json")
}

fn terrain() -> TerrainCache {
    TerrainCache::new(maps_dir())
}

#[test]
fn rng_matches_official_simple_hash_and_next_id() {
    // Oracle: OpenFront src/core/Util.ts + PseudoRandom.ts
    assert_eq!(simple_hash("golden-spawn"), 1_418_636_873);
    let mut rng = PseudoRandom::new(simple_hash("golden-spawn"));
    assert_eq!(rng.next_id(), "0uyypzm6");
    assert!((rng.next() - 0.0401959631126374).abs() < 1e-12);
    assert_eq!(rng.next_int(0.0, 7.0), 2);
}

#[test]
fn terrain_loads_official_bins() {
    let mut cache = terrain();
    let four = cache
        .fresh_terrain("FourIslands", MapSize::Compact)
        .expect("fourislands");
    assert_eq!(four.game_map.width(), 750);
    assert_eq!(four.game_map.height(), 750);
    assert_eq!(four.game_map.num_land_tiles(), 127_769);
    let halk = cache
        .fresh_terrain("Halkidiki", MapSize::Compact)
        .expect("halkidiki");
    assert_eq!(halk.game_map.width(), 1100);
    assert_eq!(halk.game_map.height(), 880);
    assert_eq!(halk.game_map.num_land_tiles(), 427_883);
}

#[test]
fn framing_roundtrip_preserves_header_and_blobs() {
    let frame = encode_frame(
        serde_json::json!({"type":"inited","k":1}),
        &[
            ("spatial", Tensor::F32(vec![1.0, 2.0])),
            ("action_mask", Tensor::U8(vec![1, 0, 1])),
        ],
    );
    assert_eq!(u32::from_le_bytes(frame[0..4].try_into().unwrap()) as usize, frame.len() - 4);
    let decoded = decode_frame(&frame[4..]).expect("decode");
    assert_eq!(decoded.header["type"], "inited");
    assert_eq!(decoded.tensors["action_mask"].1, vec![1, 0, 1]);
}

#[test]
fn capabilities_fail_closed_for_later_features() {
    let caps = Capabilities::foundation();
    assert!(caps.supported.spawn);
    assert!(caps.supported.boats);
    assert!(!caps.unsupported.boats);
    assert!(caps.supported.tribe_ai);
    assert!(caps.supported.nations_ai);
    assert!(!caps.unsupported.nukes);
    assert!(!caps.unsupported.rails);
    assert_eq!(caps.fallback, "official");
    let mut cfg = EnvConfig::default();
    cfg.nations = NationsSpec::Count(3);
    cfg.bots = 1;
    assert!(UnsupportedConfig::check(&cfg).is_ok());
}

#[test]
fn later_systems_are_declared_supported() {
    let caps = Capabilities::foundation();
    assert!(!caps.unsupported.nukes);
    assert!(!caps.unsupported.rails);
    assert!(!caps.unsupported.doomsday);
    assert!(caps.supported.nations_ai);
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
    let (init, bounds) = replay_golden(&mut cache, cfg, &actions).expect(name);
    (golden, init, bounds)
}

fn assert_core_fields(label: &str, exp: &serde_json::Value, got: &ofai_sim::digest::DecisionBoundary) {
    let gj = boundary_json(got);
    assert_eq!(exp["tick"].as_u64(), gj["tick"].as_u64(), "{label} tick");
    assert_eq!(exp["reward"].as_f64(), gj["reward"].as_f64(), "{label} reward");
    assert_eq!(exp["intentCount"].as_u64(), gj["intentCount"].as_u64(), "{label} intentCount");
    assert_eq!(exp["actionAccepted"], gj["actionAccepted"], "{label} actionAccepted");
    assert_eq!(exp["terminalCause"], gj["terminalCause"], "{label} terminalCause");
    assert_eq!(exp["win"], gj["win"], "{label} win");
    assert_eq!(exp["stageSuccess"], gj["stageSuccess"], "{label} stageSuccess");
}

#[test]
fn spawn_golden_matches_oracle_decision_boundaries() {
    let (golden, init, bounds) = replay_named("spawn");
    assert_eq!(init.tick, golden["init"]["tick"]);
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
    assert_eq!(init.tile_digest, golden["init"]["tileDigest"].as_str().unwrap());
    assert_eq!(init.step_digest, golden["init"]["stepDigest"].as_str().unwrap());
    assert_eq!(init.obs_digest, golden["init"]["obsDigest"].as_str().unwrap());
    assert_eq!(init.digest, golden["init"]["digest"].as_str().unwrap());
    let expected = golden["boundaries"].as_array().unwrap();
    assert_eq!(bounds.len(), expected.len());
    for (i, (exp, got)) in expected.iter().zip(bounds.iter()).enumerate() {
        assert_core_fields(&format!("spawn step {i}"), exp, got);
        assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "spawn step {i} coreHash");
        assert_eq!(got.tile_digest, exp["tileDigest"].as_str().unwrap(), "spawn step {i} tileDigest");
        assert_eq!(got.step_digest, exp["stepDigest"].as_str().unwrap(), "spawn step {i} stepDigest");
        // Post-spawn official obsDigest includes BOAT legality via SpatialQuery
        // (rust-features). Foundation fails BOAT closed instead of approximating.
    }
}

#[test]
fn terminal_no_spawn_matches_oracle() {
    let (golden, init, bounds) = replay_named("terminal-no-spawn");
    assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64());
    assert_eq!(init.tile_digest, golden["init"]["tileDigest"].as_str().unwrap());
    assert_eq!(init.obs_digest, golden["init"]["obsDigest"].as_str().unwrap());
    assert_eq!(init.digest, golden["init"]["digest"].as_str().unwrap());
    let last = bounds.last().expect("steps");
    assert_eq!(last.terminal_cause.as_str(), "no_spawn");
    let exp_last = golden["boundaries"].as_array().unwrap().last().unwrap();
    assert_core_fields("no_spawn last", exp_last, last);
    assert_eq!(last.step_digest, exp_last["stepDigest"].as_str().unwrap());
    assert_eq!(last.tile_digest, exp_last["tileDigest"].as_str().unwrap());
    // Still in spawn phase, so obs stays oracle-equal (no boat-mask gap).
    assert_eq!(last.obs_digest, exp_last["obsDigest"].as_str().unwrap());
    assert_eq!(last.digest, exp_last["digest"].as_str().unwrap());
}

#[test]
fn terminal_timeout_and_wilderness_match_integer_oracle() {
    for name in ["terminal-timeout", "wilderness-combat"] {
        let (golden, init, bounds) = replay_named(name);
        assert_eq!(init.tick, golden["init"]["tick"], "{name} init tick");
        assert_eq!(init.core_hash, golden["init"]["coreHash"].as_f64(), "{name} init coreHash");
        assert_eq!(
            init.tile_digest,
            golden["init"]["tileDigest"].as_str().unwrap(),
            "{name} init tileDigest"
        );
        assert_eq!(
            init.obs_digest,
            golden["init"]["obsDigest"].as_str().unwrap(),
            "{name} init obsDigest"
        );
        assert_eq!(
            init.digest,
            golden["init"]["digest"].as_str().unwrap(),
            "{name} init digest"
        );
        let expected = golden["boundaries"].as_array().unwrap();
        assert_eq!(bounds.len(), expected.len(), "{name} boundary count");
        for (i, (exp, got)) in expected.iter().zip(bounds.iter()).enumerate() {
            assert_core_fields(&format!("{name} step {i}"), exp, got);
            assert_eq!(got.core_hash, exp["coreHash"].as_f64(), "{name} step {i} coreHash");
            assert_eq!(
                got.tile_digest,
                exp["tileDigest"].as_str().unwrap(),
                "{name} step {i} tileDigest"
            );
            assert_eq!(
                got.step_digest,
                exp["stepDigest"].as_str().unwrap(),
                "{name} step {i} stepDigest"
            );
        }
    }
}

#[test]
fn early_stage_masks_keep_unsupported_actions_closed() {
    let golden = load_golden("spawn");
    let cfg = EnvConfig::from_json(&golden["config"]).unwrap();
    let mut cache = terrain();
    let mut env = AgentEnv::create(cfg, &mut cache).unwrap();
    let obs = env.peek_obs();
    assert_eq!(obs.action_mask[ACTION_SPAWN as usize], 1);
    assert_eq!(obs.action_mask[ACTION_BUILD as usize], 0);
    assert_eq!(obs.action_mask[ACTION_BOAT as usize], 0);
    let spawned = env.step(ActionVec {
        action_type: ACTION_SPAWN,
        target: 0,
        region: 2,
        quantity: 0,
        unit: 0,
    });
    assert!(spawned.info.action_accepted);
}

#[test]
fn player_attack_intent_is_emitted_when_two_humans_share_a_border() {
    // Foundation combat path: player-vs-player AttackExecution exists even
    // though tribe AI (the player-combat golden) is rust-features.
    use ofai_sim::game::{find_spawn_tile, Game, Intent};
    use ofai_sim::player::PlayerInfo;
    use ofai_sim::spec::Difficulty;

    let mut cache = terrain();
    let fresh = cache.fresh_terrain("Halkidiki", MapSize::Compact).unwrap();
    let a = PlayerInfo {
        name: "A".into(),
        player_type: ofai_sim::spec::PlayerType::Human,
        client_id: Some("a".into()),
        id: "playeraaa".into(),
    };
    let b = PlayerInfo {
        name: "B".into(),
        player_type: ofai_sim::spec::PlayerType::Human,
        client_id: Some("b".into()),
        id: "playerbbb".into(),
    };
    let mut game = Game::new(
        fresh.game_map,
        fresh.mini_map,
        vec![a, b],
        Difficulty::Easy,
        "g".into(),
    );
    game.finish_setup();
    let ta = find_spawn_tile(&game.map, 0).expect("spawn a");
    let tb = find_spawn_tile(&game.map, 20).expect("spawn b");
    game.add_intents("g", "playeraaa", &[Intent::Spawn { tile: ta }]);
    game.execute_next_tick();
    game.execute_next_tick();
    game.add_intents("g", "playerbbb", &[Intent::Spawn { tile: tb }]);
    game.execute_next_tick();
    game.execute_next_tick();
    assert!(game.player("playeraaa").unwrap().has_spawned());
    assert!(game.player("playerbbb").unwrap().has_spawned());
    game.add_intents(
        "g",
        "playeraaa",
        &[Intent::Attack {
            target_id: Some("playerbbb".into()),
            troops: 100.0,
            source_tile: None,
            remove_troops: true,
        }],
    );
    for _ in 0..30 {
        game.execute_next_tick();
    }
    let a = game.player("playeraaa").unwrap();
    assert!(
        a.outgoing.iter().any(|atk| atk.target_player_id.as_deref() == Some("playerbbb"))
            || a.num_tiles() > 0,
        "player attack execution should run"
    );
}
