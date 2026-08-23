//! Schema-v2 constants mirrored from OpenFrontIO/ofai/env/spec.ts.

pub const SPATIAL_CHANNELS: usize = 10;
pub const SPATIAL_SIZE: usize = 64;
pub const NUM_PLAYER_SLOTS: usize = 16;
pub const PLAYER_FEATURES: usize = 14;
pub const GLOBAL_FEATURES: usize = 10;
pub const REGION_GRID: usize = 32;
pub const NUM_REGIONS: usize = REGION_GRID * REGION_GRID;
pub const NUM_ACTION_TYPES: usize = 9;
pub const NUM_QUANTITIES: usize = 5;
pub const NUM_UNIT_TYPES: usize = 10;
pub const TARGET_MASKS_SIZE: usize = NUM_ACTION_TYPES * NUM_PLAYER_SLOTS;
pub const ACTION_HEADS: usize = 5;

pub const ACTION_NOOP: i32 = 0;
pub const ACTION_SPAWN: i32 = 1;
pub const ACTION_ATTACK: i32 = 2;
pub const ACTION_RETREAT_ALL: i32 = 3;
pub const ACTION_BUILD: i32 = 4;
pub const ACTION_BOAT: i32 = 5;
pub const ACTION_ALLY: i32 = 6;
pub const ACTION_BREAK_ALLY: i32 = 7;
pub const ACTION_EMBARGO: i32 = 8;

pub const TROOP_FRACTIONS: [f64; 5] = [0.05, 0.15, 0.3, 0.5, 0.8];

pub const REWARD_WIN: f64 = 3.0;
pub const REWARD_CURRICULUM_SUCCESS: f64 = 1.0;
pub const REWARD_DEATH: f64 = -1.0;
pub const REWARD_LOSS_ALIVE: f64 = -1.0;
pub const REWARD_TIMEOUT: f64 = -1.0;
pub const REWARD_NO_SPAWN: f64 = -1.0;
pub const REWARD_SPAWN: f64 = 0.1;
pub const ELIMINATION_SHAPING_WEIGHT: f64 = 0.25;

pub const CH_LAND: usize = 0;
pub const CH_MOUNTAIN: usize = 1;
pub const CH_MINE: usize = 2;
pub const CH_ENEMY: usize = 3;
pub const CH_ALLY: usize = 4;
pub const CH_MY_BORDER: usize = 5;
pub const CH_FALLOUT: usize = 6;
pub const CH_DEFENSE: usize = 7;
pub const CH_MY_STRUCTURES: usize = 8;
pub const CH_ENEMY_STRUCTURES: usize = 9;

pub const AGENT_CLIENT_ID: &str = "agent";
pub const TERRA_NULLIUS_ID: Option<&str> = None;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminalCause {
    None,
    Win,
    CurriculumSuccess,
    Death,
    LossAlive,
    Timeout,
    NoSpawn,
}

impl TerminalCause {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Win => "win",
            Self::CurriculumSuccess => "curriculum_success",
            Self::Death => "death",
            Self::LossAlive => "loss_alive",
            Self::Timeout => "timeout",
            Self::NoSpawn => "no_spawn",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerType {
    Human,
    Bot,
    Nation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerrainKind {
    Plains,
    Highland,
    Mountain,
    Ocean,
    Impassable,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Difficulty {
    Easy,
    Medium,
    Hard,
    Impossible,
}

impl Difficulty {
    pub fn parse(s: &str) -> Self {
        match s {
            "Easy" => Self::Easy,
            "Hard" => Self::Hard,
            "Impossible" => Self::Impossible,
            _ => Self::Medium,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MapSize {
    Normal,
    Compact,
}

impl MapSize {
    pub fn parse(s: &str) -> Self {
        if s.eq_ignore_ascii_case("compact") {
            Self::Compact
        } else {
            Self::Normal
        }
    }
}

#[derive(Clone, Debug)]
pub enum NationsSpec {
    Disabled,
    Default,
    Count(u32),
}

impl NationsSpec {
    pub fn from_json(v: &serde_json::Value) -> Self {
        match v {
            serde_json::Value::String(s) if s == "disabled" => Self::Disabled,
            serde_json::Value::String(s) if s == "default" => Self::Default,
            serde_json::Value::Number(n) => {
                Self::Count(n.as_u64().unwrap_or(0) as u32)
            }
            _ => Self::Default,
        }
    }

    pub fn is_disabled(&self) -> bool {
        matches!(self, Self::Disabled)
    }
}

#[derive(Clone, Debug)]
pub struct EnvConfig {
    pub map: String,
    pub map_size: MapSize,
    pub nations: NationsSpec,
    pub difficulty: Difficulty,
    pub bots: u32,
    pub seed: String,
    pub max_ticks: u32,
    pub decision_interval: u32,
    pub shaping: f64,
    pub allowed_actions: Option<Vec<i32>>,
    pub win_tiles_frac: Option<f64>,
    pub enable_digest: bool,
    pub starting_gold: Option<i64>,
    pub disabled_units: Vec<String>,
    pub max_timer_value: Option<u32>,
    pub doomsday_enabled: bool,
    pub doomsday_speed: String,
    pub water_nukes: bool,
}

impl Default for EnvConfig {
    fn default() -> Self {
        Self {
            map: "FourIslands".into(),
            map_size: MapSize::Normal,
            nations: NationsSpec::Default,
            difficulty: Difficulty::Medium,
            bots: 20,
            seed: "ofai-seed-0".into(),
            max_ticks: 12000,
            decision_interval: 10,
            shaping: 0.0,
            allowed_actions: None,
            win_tiles_frac: None,
            enable_digest: false,
            starting_gold: None,
            disabled_units: Vec::new(),
            max_timer_value: None,
            doomsday_enabled: false,
            doomsday_speed: "normal".into(),
            water_nukes: false,
        }
    }
}

impl EnvConfig {
    pub fn from_json(v: &serde_json::Value) -> Result<Self, String> {
        let mut cfg = Self::default();
        if let Some(m) = v.get("map").and_then(|x| x.as_str()) {
            cfg.map = m.to_string();
        }
        if let Some(s) = v.get("mapSize").and_then(|x| x.as_str()) {
            cfg.map_size = MapSize::parse(s);
        }
        if let Some(n) = v.get("nations") {
            cfg.nations = NationsSpec::from_json(n);
        }
        if let Some(d) = v.get("difficulty").and_then(|x| x.as_str()) {
            cfg.difficulty = Difficulty::parse(d);
        }
        if let Some(b) = v.get("bots").and_then(|x| x.as_u64()) {
            cfg.bots = b as u32;
        }
        if let Some(s) = v.get("seed").and_then(|x| x.as_str()) {
            cfg.seed = s.to_string();
        }
        if let Some(t) = v.get("maxTicks").and_then(|x| x.as_u64()) {
            cfg.max_ticks = t as u32;
        }
        if let Some(t) = v.get("decisionInterval").and_then(|x| x.as_u64()) {
            cfg.decision_interval = t as u32;
        }
        if let Some(s) = v.get("shaping").and_then(|x| x.as_f64()) {
            cfg.shaping = s;
        }
        if let Some(w) = v.get("winTilesFrac").and_then(|x| x.as_f64()) {
            cfg.win_tiles_frac = Some(w);
        }
        if let Some(d) = v.get("enableDigest").and_then(|x| x.as_bool()) {
            cfg.enable_digest = d;
        }
        if let Some(g) = v.get("startingGold").and_then(|x| x.as_i64()) {
            cfg.starting_gold = Some(g);
        }
        if let Some(arr) = v.get("disabledUnits").and_then(|x| x.as_array()) {
            cfg.disabled_units = arr
                .iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect();
        }
        if let Some(t) = v.get("maxTimerValue").and_then(|x| x.as_u64()) {
            cfg.max_timer_value = Some(t as u32);
        }
        if let Some(w) = v.get("waterNukes").and_then(|x| x.as_bool()) {
            cfg.water_nukes = w;
        }
        if let Some(dd) = v.get("doomsdayClock") {
            if let Some(en) = dd.get("enabled").and_then(|x| x.as_bool()) {
                cfg.doomsday_enabled = en;
            }
            if let Some(sp) = dd.get("speed").and_then(|x| x.as_str()) {
                cfg.doomsday_speed = sp.to_string();
            }
        }
        if let Some(arr) = v.get("allowedActions").and_then(|x| x.as_array()) {
            let mut out = Vec::new();
            for a in arr {
                if let Some(n) = a.as_i64() {
                    out.push(n as i32);
                } else if let Some(s) = a.as_str() {
                    if let Some(id) = action_name_to_id(s) {
                        out.push(id);
                    }
                }
            }
            cfg.allowed_actions = Some(out);
        }
        Ok(cfg)
    }

    pub fn apply_patch(&mut self, next: &serde_json::Value) {
        if let Ok(patch) = Self::from_json(next) {
            if next.get("map").is_some() {
                self.map = patch.map;
            }
            if next.get("mapSize").is_some() {
                self.map_size = patch.map_size;
            }
            if next.get("nations").is_some() {
                self.nations = patch.nations;
            }
            if next.get("difficulty").is_some() {
                self.difficulty = patch.difficulty;
            }
            if next.get("bots").is_some() {
                self.bots = patch.bots;
            }
            if next.get("seed").is_some() {
                self.seed = patch.seed;
            }
            if next.get("maxTicks").is_some() {
                self.max_ticks = patch.max_ticks;
            }
            if next.get("decisionInterval").is_some() {
                self.decision_interval = patch.decision_interval;
            }
            if next.get("shaping").is_some() {
                self.shaping = patch.shaping;
            }
            if next.get("winTilesFrac").is_some() {
                self.win_tiles_frac = patch.win_tiles_frac;
            }
            if next.get("enableDigest").is_some() {
                self.enable_digest = patch.enable_digest;
            }
            if next.get("allowedActions").is_some() {
                self.allowed_actions = patch.allowed_actions;
            }
            if next.get("startingGold").is_some() {
                self.starting_gold = patch.starting_gold;
            }
            if next.get("disabledUnits").is_some() {
                self.disabled_units = patch.disabled_units;
            }
            if next.get("maxTimerValue").is_some() {
                self.max_timer_value = patch.max_timer_value;
            }
            if next.get("doomsdayClock").is_some() {
                self.doomsday_enabled = patch.doomsday_enabled;
                self.doomsday_speed = patch.doomsday_speed;
            }
            if next.get("waterNukes").is_some() {
                self.water_nukes = patch.water_nukes;
            }
        }
    }
}

pub fn action_name_to_id(name: &str) -> Option<i32> {
    Some(match name {
        "noop" => ACTION_NOOP,
        "spawn" => ACTION_SPAWN,
        "attack" => ACTION_ATTACK,
        "retreat_all" => ACTION_RETREAT_ALL,
        "build" => ACTION_BUILD,
        "boat" => ACTION_BOAT,
        "ally" => ACTION_ALLY,
        "break_ally" => ACTION_BREAK_ALLY,
        "embargo" => ACTION_EMBARGO,
        _ => return None,
    })
}

pub fn quantize_reward(reward: f64) -> f64 {
    (reward * 1_000_000.0).round() / 1_000_000.0
}
