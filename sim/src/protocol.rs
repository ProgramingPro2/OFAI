//! Length-prefixed schema-v2 TCP framing (ofai/env/framing.ts).

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Instant;

use crate::actions::ActionVec;
use crate::capabilities::Capabilities;
use crate::env::AgentEnv;
use crate::obs::ObsBuffers;
use crate::spec::{
    EnvConfig, GLOBAL_FEATURES, NUM_ACTION_TYPES, NUM_PLAYER_SLOTS, NUM_QUANTITIES, NUM_REGIONS,
    NUM_UNIT_TYPES, PLAYER_FEATURES, SPATIAL_CHANNELS, SPATIAL_SIZE, TARGET_MASKS_SIZE,
};
use crate::terrain::TerrainCache;

pub fn encode_frame(header: serde_json::Value, tensors: &[(&str, Tensor)]) -> Vec<u8> {
    let mut specs = serde_json::Map::new();
    let mut blob = Vec::new();
    for (name, t) in tensors {
        let data = t.bytes();
        specs.insert(
            (*name).to_string(),
            serde_json::json!({
                "dtype": t.dtype(),
                "shape": [t.len()],
                "offset": blob.len(),
                "length": data.len(),
            }),
        );
        blob.extend_from_slice(&data);
    }
    let mut obj = header.as_object().cloned().unwrap_or_default();
    obj.insert("tensors".into(), serde_json::Value::Object(specs));
    let header_bytes = serde_json::to_vec(&obj).expect("header");
    let total = 4 + header_bytes.len() + blob.len();
    let mut out = Vec::with_capacity(8 + header_bytes.len() + blob.len());
    out.extend_from_slice(&(total as u32).to_le_bytes());
    out.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    out.extend_from_slice(&blob);
    out
}

pub enum Tensor {
    F32(Vec<f32>),
    U8(Vec<u8>),
    I32(Vec<i32>),
}

impl Tensor {
    fn dtype(&self) -> &'static str {
        match self {
            Self::F32(_) => "f32",
            Self::U8(_) => "u8",
            Self::I32(_) => "i32",
        }
    }
    fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::U8(v) => v.len(),
            Self::I32(v) => v.len(),
        }
    }
    fn bytes(&self) -> Vec<u8> {
        match self {
            Self::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Self::U8(v) => v.clone(),
            Self::I32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

pub struct Frame {
    pub header: serde_json::Value,
    pub tensors: HashMap<String, (String, Vec<u8>)>,
}

pub fn decode_frame(payload: &[u8]) -> Result<Frame, String> {
    if payload.len() < 4 {
        return Err("short frame".into());
    }
    let header_len = u32::from_le_bytes(payload[0..4].try_into().unwrap()) as usize;
    if payload.len() < 4 + header_len {
        return Err("short header".into());
    }
    let header: serde_json::Value = serde_json::from_slice(&payload[4..4 + header_len])
        .map_err(|e| format!("header json: {e}"))?;
    let blob = &payload[4 + header_len..];
    let mut tensors = HashMap::new();
    if let Some(map) = header.get("tensors").and_then(|t| t.as_object()) {
        for (name, spec) in map {
            let off = spec.get("offset").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let len = spec.get("length").and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let dtype = spec
                .get("dtype")
                .and_then(|x| x.as_str())
                .unwrap_or("u8")
                .to_string();
            if off + len > blob.len() {
                return Err(format!("tensor {name} out of range"));
            }
            tensors.insert(name.clone(), (dtype, blob[off..off + len].to_vec()));
        }
    }
    Ok(Frame { header, tensors })
}

fn obs_tensors(batch: &[ObsBuffers]) -> Vec<(&'static str, Tensor)> {
    let k = batch.len();
    let plane = SPATIAL_CHANNELS * SPATIAL_SIZE * SPATIAL_SIZE;
    let mut spatial = vec![0f32; k * plane];
    let mut players = vec![0f32; k * NUM_PLAYER_SLOTS * PLAYER_FEATURES];
    let mut global = vec![0f32; k * GLOBAL_FEATURES];
    let mut action_mask = vec![0u8; k * NUM_ACTION_TYPES];
    let mut target_masks = vec![0u8; k * TARGET_MASKS_SIZE];
    let mut quantity_mask = vec![0u8; k * NUM_QUANTITIES];
    let mut unit_mask = vec![0u8; k * NUM_UNIT_TYPES];
    let mut spawn = vec![0u8; k * NUM_REGIONS];
    let mut build = vec![0u8; k * NUM_REGIONS];
    let mut boat = vec![0u8; k * NUM_REGIONS];
    for (i, o) in batch.iter().enumerate() {
        spatial[i * plane..i * plane + plane].copy_from_slice(&o.spatial);
        players[i * NUM_PLAYER_SLOTS * PLAYER_FEATURES
            ..(i + 1) * NUM_PLAYER_SLOTS * PLAYER_FEATURES]
            .copy_from_slice(&o.players);
        global[i * GLOBAL_FEATURES..(i + 1) * GLOBAL_FEATURES].copy_from_slice(&o.global);
        action_mask[i * NUM_ACTION_TYPES..(i + 1) * NUM_ACTION_TYPES]
            .copy_from_slice(&o.action_mask);
        target_masks[i * TARGET_MASKS_SIZE..(i + 1) * TARGET_MASKS_SIZE]
            .copy_from_slice(&o.target_masks);
        quantity_mask[i * NUM_QUANTITIES..(i + 1) * NUM_QUANTITIES]
            .copy_from_slice(&o.quantity_mask);
        unit_mask[i * NUM_UNIT_TYPES..(i + 1) * NUM_UNIT_TYPES].copy_from_slice(&o.unit_mask);
        spawn[i * NUM_REGIONS..(i + 1) * NUM_REGIONS].copy_from_slice(&o.spawn_regions);
        build[i * NUM_REGIONS..(i + 1) * NUM_REGIONS].copy_from_slice(&o.build_regions);
        boat[i * NUM_REGIONS..(i + 1) * NUM_REGIONS].copy_from_slice(&o.boat_regions);
    }
    vec![
        ("spatial", Tensor::F32(spatial)),
        ("players", Tensor::F32(players)),
        ("global", Tensor::F32(global)),
        ("action_mask", Tensor::U8(action_mask)),
        ("target_masks", Tensor::U8(target_masks)),
        ("quantity_mask", Tensor::U8(quantity_mask)),
        ("unit_mask", Tensor::U8(unit_mask)),
        ("spawn_regions", Tensor::U8(spawn)),
        ("build_regions", Tensor::U8(build)),
        ("boat_regions", Tensor::U8(boat)),
    ]
}

pub struct Server {
    envs: Vec<AgentEnv>,
    terrain: TerrainCache,
}

impl Server {
    pub fn new(maps_dir: std::path::PathBuf) -> Self {
        Self {
            envs: Vec::new(),
            terrain: TerrainCache::new(maps_dir),
        }
    }

    pub fn handle(&mut self, frame: &Frame) -> Result<Vec<u8>, String> {
        let cmd = frame
            .header
            .get("cmd")
            .and_then(|c| c.as_str())
            .unwrap_or("");
        match cmd {
            "init" => self.init(frame),
            "step" => self.step(frame),
            "reset" => self.reset(frame),
            "frame" => self.frame_cmd(frame),
            "capabilities" => Ok(encode_frame(
                serde_json::json!({
                    "type": "capabilities",
                    "capabilities": Capabilities::foundation(),
                }),
                &[],
            )),
            "close" => Ok(encode_frame(serde_json::json!({"type":"closed"}), &[])),
            _ => Ok(encode_frame(
                serde_json::json!({"type":"error","error": format!("unknown cmd {cmd}")}),
                &[],
            )),
        }
    }

    fn init(&mut self, frame: &Frame) -> Result<Vec<u8>, String> {
        let configs = frame
            .header
            .get("envs")
            .and_then(|e| e.as_array())
            .ok_or_else(|| "init missing envs".to_string())?;
        self.envs.clear();
        for c in configs {
            let cfg = EnvConfig::from_json(c)?;
            self.envs
                .push(AgentEnv::create(cfg, &mut self.terrain)?);
        }
        let obs: Vec<_> = self.envs.iter().map(|e| e.peek_obs().clone()).collect();
        Ok(encode_frame(
            serde_json::json!({
                "type": "inited",
                "k": self.envs.len(),
                "backend": "rust",
                "capabilities": Capabilities::foundation(),
                "fallback": "official",
            }),
            &obs_tensors(&obs),
        ))
    }

    fn step(&mut self, frame: &Frame) -> Result<Vec<u8>, String> {
        let actions = frame
            .tensors
            .get("actions")
            .ok_or_else(|| "step missing actions".to_string())?;
        let k = self.envs.len();
        if actions.1.len() < k * 20 {
            return Err("actions blob too short".into());
        }
        let mut rewards = Vec::new();
        let mut dones = Vec::new();
        let mut infos = Vec::new();
        let mut obs_list = Vec::new();
        let mut profile = serde_json::json!({
            "simMs": 0.0, "maskObsMs": 0.0, "serializeMs": 0.0, "resetMs": 0.0, "resets": 0.0
        });
        for i in 0..k {
            let off = i * 20;
            let action = ActionVec {
                action_type: i32::from_le_bytes(actions.1[off..off + 4].try_into().unwrap()),
                target: i32::from_le_bytes(actions.1[off + 4..off + 8].try_into().unwrap()),
                region: i32::from_le_bytes(actions.1[off + 8..off + 12].try_into().unwrap()),
                quantity: i32::from_le_bytes(actions.1[off + 12..off + 16].try_into().unwrap()),
                unit: i32::from_le_bytes(actions.1[off + 16..off + 20].try_into().unwrap()),
            };
            let t0 = Instant::now();
            let result = self.envs[i].step(action);
            let sim_ms = t0.elapsed().as_secs_f64() * 1000.0;
            profile["simMs"] = serde_json::json!(profile["simMs"].as_f64().unwrap_or(0.0) + sim_ms);
            rewards.push(result.reward);
            dones.push(if result.done { 1 } else { 0 });
            infos.push(info_json(&result.info));
            if result.done {
                let t_reset = Instant::now();
                let next = frame
                    .header
                    .get("nextConfigs")
                    .and_then(|a| a.as_array())
                    .and_then(|a| a.get(i));
                let (spec, fallback) = resolve_auto_reset(next, &self.envs[i]);
                if fallback {
                    self.envs[i].reset_count += 1;
                }
                self.envs[i].reset(&mut self.terrain, spec.as_ref())?;
                profile["resetMs"] =
                    serde_json::json!(profile["resetMs"].as_f64().unwrap_or(0.0) + t_reset.elapsed().as_secs_f64() * 1000.0);
                profile["resets"] = serde_json::json!(profile["resets"].as_f64().unwrap_or(0.0) + 1.0);
                obs_list.push(self.envs[i].peek_obs().clone());
            } else {
                obs_list.push(result.obs);
            }
        }
        let t_ser = Instant::now();
        let tensors = obs_tensors(&obs_list);
        profile["serializeMs"] = serde_json::json!(t_ser.elapsed().as_secs_f64() * 1000.0);
        Ok(encode_frame(
            serde_json::json!({
                "type": "step",
                "rewards": rewards,
                "dones": dones,
                "infos": infos,
                "profile": profile,
            }),
            &tensors,
        ))
    }

    fn reset(&mut self, frame: &Frame) -> Result<Vec<u8>, String> {
        let index = frame
            .header
            .get("index")
            .and_then(|i| i.as_u64())
            .unwrap_or(0) as usize;
        let env = self
            .envs
            .get_mut(index)
            .ok_or_else(|| format!("bad reset index {index}"))?;
        let patch = frame.header.get("config").cloned().or_else(|| {
            frame
                .header
                .get("seed")
                .and_then(|s| s.as_str())
                .map(|s| serde_json::Value::String(s.to_string()))
        });
        env.reset(&mut self.terrain, patch.as_ref())?;
        Ok(encode_frame(
            serde_json::json!({"type":"reset","index": index}),
            &obs_tensors(&[env.peek_obs().clone()]),
        ))
    }

    fn frame_cmd(&mut self, frame: &Frame) -> Result<Vec<u8>, String> {
        let index = frame
            .header
            .get("index")
            .and_then(|i| i.as_u64())
            .unwrap_or(0) as usize;
        let env = self
            .envs
            .get(index)
            .ok_or_else(|| format!("bad frame index {index}"))?;
        let (w, h, cells, extra) = env.render_frame();
        Ok(encode_frame(
            serde_json::json!({
                "type": "frame",
                "width": w,
                "height": h,
                "players": extra["players"],
                "units": extra["units"],
                "tick": extra["tick"],
            }),
            &[("cells", Tensor::U8(cells))],
        ))
    }
}

fn resolve_auto_reset(
    next: Option<&serde_json::Value>,
    env: &AgentEnv,
) -> (Option<serde_json::Value>, bool) {
    match next {
        Some(serde_json::Value::String(s)) if !s.is_empty() => {
            (Some(serde_json::Value::String(s.clone())), false)
        }
        Some(v) if v.is_object() => (Some(v.clone()), false),
        _ => (
            Some(serde_json::Value::String(format!(
                "{}-r{}",
                env.seed(),
                env.reset_count
            ))),
            true,
        ),
    }
}

fn info_json(info: &crate::env::StepInfo) -> serde_json::Value {
    let mut v = serde_json::json!({
        "tick": info.tick,
        "win": info.win,
        "stageSuccess": info.stage_success,
        "dead": info.dead,
        "kills": info.kills,
        "tilesFrac": info.tiles_frac,
        "spawned": info.spawned,
        "hash": info.hash,
        "terminalCause": info.terminal_cause.as_str(),
        "peakTilesFrac": info.peak_tiles_frac,
        "rewardTerms": {
            "terminal": info.reward_terms.terminal,
            "spawn": info.reward_terms.spawn,
            "territory": info.reward_terms.territory,
            "elimination": info.reward_terms.elimination,
            "total": info.reward_terms.total,
        },
        "intentCount": info.intent_count,
        "actionAccepted": info.action_accepted,
        "map": info.map,
        "mapSize": info.map_size,
        "seed": info.seed,
    });
    if let Some(d) = &info.digest {
        v["digest"] = serde_json::Value::String(d.clone());
    }
    if let Some(d) = &info.tile_digest {
        v["tileDigest"] = serde_json::Value::String(d.clone());
    }
    if let Some(d) = &info.obs_digest {
        v["obsDigest"] = serde_json::Value::String(d.clone());
    }
    if let Some(d) = &info.step_digest {
        v["stepDigest"] = serde_json::Value::String(d.clone());
    }
    if let Some(f) = &info.fallback {
        v["fallback"] = serde_json::Value::String(f.clone());
        v["unsupported"] = serde_json::Value::String(f.clone());
    }
    v
}

pub fn serve(port: u16, maps_dir: std::path::PathBuf) -> Result<(), String> {
    let listener = TcpListener::bind(("0.0.0.0", port)).map_err(|e| e.to_string())?;
    eprintln!("[ofai-sim] listening on port {port}");
    println!("ENV_SERVER_READY port={port}");
    let mut server = Server::new(maps_dir);
    for stream in listener.incoming() {
        let mut stream = stream.map_err(|e| e.to_string())?;
        eprintln!("[ofai-sim] client connected");
        if let Err(e) = handle_client(&mut server, &mut stream) {
            eprintln!("[ofai-sim] client error: {e}");
        }
    }
    Ok(())
}

fn handle_client(server: &mut Server, stream: &mut TcpStream) -> Result<(), String> {
    loop {
        let mut len_buf = [0u8; 4];
        match stream.read_exact(&mut len_buf) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(e) => return Err(e.to_string()),
        }
        let total = u32::from_le_bytes(len_buf) as usize;
        let mut payload = vec![0u8; total];
        stream.read_exact(&mut payload).map_err(|e| e.to_string())?;
        let frame = decode_frame(&payload)?;
        if frame.header.get("cmd").and_then(|c| c.as_str()) == Some("close") {
            let reply = encode_frame(serde_json::json!({"type":"closed"}), &[]);
            stream.write_all(&reply).map_err(|e| e.to_string())?;
            return Ok(());
        }
        let reply = match server.handle(&frame) {
            Ok(r) => r,
            Err(e) => encode_frame(
                serde_json::json!({
                    "type": "error",
                    "error": e,
                    "fallback": "official",
                    "capabilities": Capabilities::foundation(),
                }),
                &[],
            ),
        };
        stream.write_all(&reply).map_err(|e| e.to_string())?;
    }
}
