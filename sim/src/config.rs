//! Economy and combat formulas from OpenFront Config.ts.

use crate::js_math::js_pow;
use crate::spec::{Difficulty, PlayerType, TerrainKind};

const DEFENSE_DEBUFF_MIDPOINT: f64 = 150_000.0;
const DEFENSE_DEBUFF_DECAY_RATE: f64 = std::f64::consts::LN_2 / 50_000.0;
pub const DEFAULT_SPAWN_IMMUNITY_TICKS: u32 = 50;
pub const START_MANPOWER_HUMAN: f64 = 25_000.0;
pub const CITY_TROOP_INCREASE: f64 = 250_000.0;

pub fn within(value: f64, min: f64, max: f64) -> f64 {
    value.max(min).min(max)
}

pub fn sigmoid(value: f64, decay_rate: f64, midpoint: f64) -> f64 {
    1.0 / (1.0 + (-decay_rate * (value - midpoint)).exp())
}

pub fn start_manpower(player_type: PlayerType, difficulty: Difficulty) -> f64 {
    match player_type {
        PlayerType::Bot => 10_000.0,
        PlayerType::Nation => match difficulty {
            Difficulty::Easy => 12_500.0,
            Difficulty::Medium => 18_750.0,
            Difficulty::Hard => 25_000.0,
            Difficulty::Impossible => 31_250.0,
        },
        PlayerType::Human => START_MANPOWER_HUMAN,
    }
}

pub fn max_troops(
    player_type: PlayerType,
    difficulty: Difficulty,
    tiles: u32,
    city_levels: f64,
) -> f64 {
    let mut max =
        2.0 * (js_pow(tiles as f64, 0.6) * 1000.0 + 50_000.0) + city_levels * CITY_TROOP_INCREASE;
    match player_type {
        PlayerType::Bot => max /= 3.0,
        PlayerType::Human => {}
        PlayerType::Nation => {
            max *= match difficulty {
                Difficulty::Easy => 0.5,
                Difficulty::Medium => 0.75,
                Difficulty::Hard => 1.0,
                Difficulty::Impossible => 1.25,
            };
        }
    }
    max
}

pub fn troop_increase_rate(
    player_type: PlayerType,
    difficulty: Difficulty,
    troops: f64,
    tiles: u32,
    city_levels: f64,
) -> f64 {
    let max = max_troops(player_type, difficulty, tiles, city_levels);
    let mut to_add = 10.0 + js_pow(troops, 0.73) / 4.0;
    let ratio = 1.0 - troops / max;
    to_add *= ratio;
    if player_type == PlayerType::Bot {
        to_add *= 0.5;
    }
    if player_type == PlayerType::Nation {
        to_add *= match difficulty {
            Difficulty::Easy => 0.9,
            Difficulty::Medium => 0.95,
            Difficulty::Hard => 1.0,
            Difficulty::Impossible => 1.05,
        };
    }
    (troops + to_add).min(max) - troops
}

pub fn gold_addition_rate(player_type: PlayerType, gold_multiplier: f64) -> i64 {
    let base = if player_type == PlayerType::Bot {
        50.0
    } else {
        100.0
    };
    (base * gold_multiplier).floor() as i64
}

pub struct AttackOutcome {
    pub attacker_troop_loss: f64,
    pub defender_troop_loss: f64,
    pub tiles_per_tick_used: f64,
}

pub fn attack_logic(
    attack_troops: f64,
    attacker_type: PlayerType,
    attacker_tiles: u32,
    defender_is_player: bool,
    defender_type: PlayerType,
    defender_troops: f64,
    defender_tiles: u32,
    defender_traitor: bool,
    terrain: TerrainKind,
    has_defense_post: bool,
    fallout_mod: Option<f64>,
) -> AttackOutcome {
    let (mut mag, mut speed) = terrain_mag_speed(terrain);
    if defender_is_player && has_defense_post {
        mag *= 5.0;
        speed *= 3.0;
    }
    if let Some(fm) = fallout_mod {
        mag *= fm;
        speed *= fm;
    }
    if defender_is_player
        && matches!(attacker_type, PlayerType::Human | PlayerType::Nation)
        && defender_type == PlayerType::Bot
    {
        mag *= 0.7;
    }
    if !defender_is_player {
        return AttackOutcome {
            attacker_troop_loss: if attacker_type == PlayerType::Bot {
                mag / 10.0
            } else {
                mag / 5.0
            },
            defender_troop_loss: 0.0,
            tiles_per_tick_used: within((2000.0 * speed.max(10.0)) / attack_troops, 5.0, 100.0),
        };
    }
    let defense_sig = 1.0 - sigmoid(defender_tiles as f64, DEFENSE_DEBUFF_DECAY_RATE, DEFENSE_DEBUFF_MIDPOINT);
    let large_def_speed = 0.7 + 0.3 * defense_sig;
    let large_def_atk = 0.7 + 0.3 * defense_sig;
    let large_attack_bonus = if attacker_tiles > 100_000 {
        (100_000.0 / attacker_tiles as f64).sqrt().powf(0.7)
    } else {
        1.0
    };
    let large_atk_speed = if attacker_tiles > 100_000 {
        (100_000.0 / attacker_tiles as f64).powf(0.6)
    } else {
        1.0
    };
    let defender_troop_loss = defender_troops / defender_tiles.max(1) as f64;
    let traitor_mod = if defender_traitor { 0.5 } else { 1.0 };
    let current = within(defender_troops / attack_troops, 0.6, 2.0)
        * mag
        * 0.8
        * large_def_atk
        * large_attack_bonus
        * traitor_mod;
    let alt = 1.3 * defender_troop_loss * (mag / 100.0) * traitor_mod;
    let attacker_troop_loss = 0.6 * current + 0.4 * alt;
    let tiles = within(defender_troops / (5.0 * attack_troops), 0.2, 1.5)
        * speed
        * large_def_speed
        * large_atk_speed
        * if defender_traitor { 0.8 } else { 1.0 };
    AttackOutcome {
        attacker_troop_loss,
        defender_troop_loss,
        tiles_per_tick_used: tiles,
    }
}

pub fn attack_tiles_per_tick(
    attack_troops: f64,
    defender_is_player: bool,
    defender_troops: f64,
    num_adj: f64,
) -> f64 {
    if defender_is_player {
        within((5.0 * attack_troops / defender_troops) * 2.0, 0.01, 0.5) * num_adj * 3.0
    } else {
        num_adj * 2.0
    }
}

fn terrain_mag_speed(terrain: TerrainKind) -> (f64, f64) {
    match terrain {
        TerrainKind::Plains => (80.0, 16.5),
        TerrainKind::Highland => (100.0, 20.0),
        TerrainKind::Mountain => (120.0, 25.0),
        TerrainKind::Impassable | TerrainKind::Ocean => (0.0, 0.0),
    }
}

/// Cheapest structure cost used to keep BUILD masked off in early stages.
pub fn cheapest_structure_cost() -> i64 {
    50_000
}

pub const TRAIN_STATION_MIN_RANGE: u32 = 15;
pub const TRAIN_STATION_MAX_RANGE: u32 = 110;
pub const SILO_COOLDOWN: u32 = 90;
pub const SAM_COOLDOWN: u32 = 90;
pub const DEFAULT_SAM_RANGE: f64 = 70.0;
pub const MAX_SAM_RANGE: f64 = 150.0;
pub const DEFAULT_SAM_MISSILE_SPEED: u32 = 12;
pub const DEFAULT_NUKE_TARGETABLE_RANGE: u32 = 150;
pub const TRADE_SHIP_SHORT_RANGE: u32 = 200;
pub const WARSHIP_SHELL_RATE: u32 = 20;
pub const WARSHIP_SHELL_DAMAGE: i32 = 250;
pub const WARSHIP_TARGETING_RANGE: u32 = 130;
pub const WARSHIP_PATROL_RANGE: u32 = 100;
pub const SHELL_LIFETIME: u32 = 50;
pub const SAFE_FROM_PIRATES_COOLDOWN: u32 = 20;
pub const WARSHIP_MAX_HEALTH: i32 = 1000;
pub const WARSHIP_RETREAT_HEALTH_PERCENT: i32 = 40;
pub const WARSHIP_PASSIVE_HEALING: i32 = 1;
pub const WARSHIP_PASSIVE_HEALING_RANGE: u32 = 20;
pub const WARSHIP_DOCKING_RANGE: u32 = 3;
pub const WARSHIP_PORT_HEALING_BONUS: i32 = 5;
pub const WARSHIP_PORT_SWITCH: f64 = 0.7;
pub const MIRV_COOLDOWN_TICKS: u32 = 300;

pub fn train_spawn_rate(num_factories: u32) -> i32 {
    ((num_factories + 10) * 15) as i32
}

pub fn train_gold(rel: &str, cities_visited: u32) -> i64 {
    let visited = cities_visited.saturating_sub(9);
    let base = match rel {
        "ally" => 35_000,
        "self" => 10_000,
        _ => 25_000,
    };
    (base - visited as i64 * 5_000).max(5_000)
}

pub fn trade_ship_gold(dist: u32) -> i64 {
    // Official Config.tradeShipShortRangeDebuff() is 300, distinct from the
    // 200-tile spawn proximity check (TRADE_SHIP_SHORT_RANGE).
    let debuff = 300.0;
    let base = 75_000.0 / (1.0 + (-0.03 * (dist as f64 - debuff)).exp()) + 50.0 * dist as f64;
    base.floor() as i64
}

pub fn trade_ship_spawn_rate(rejections: u32, num_ships: u32) -> i32 {
    let decay = std::f64::consts::LN_2 / 50.0;
    let base = 1.0 - sigmoid(num_ships as f64, decay, 400.0);
    let pity = 1.0 / (rejections as f64 + 1.0);
    ((100.0 * pity) / base).floor() as i32
}

pub fn nuke_speed(ty: crate::units::UnitType) -> u32 {
    match ty {
        crate::units::UnitType::Mirv => 15,
        crate::units::UnitType::MirvWarhead => 22,
        _ => 10,
    }
}

pub fn nuke_magnitudes(ty: crate::units::UnitType) -> (u32, u32) {
    match ty {
        crate::units::UnitType::MirvWarhead => (12, 18),
        crate::units::UnitType::AtomBomb => (12, 30),
        crate::units::UnitType::HydrogenBomb => (80, 100),
        _ => (12, 30),
    }
}

pub fn sam_range(level: u32) -> f64 {
    MAX_SAM_RANGE - 480.0 / (level as f64 + 5.0)
}

pub fn mirv_cost(launched: u32) -> i64 {
    25_000_000 + launched as i64 * 15_000_000
}

pub fn proximity_bonus_ports(n: usize) -> f64 {
    // Official Config.proximityBonusPortsNb returns a float; PortExecution
    // compares `i < bonus`, so index 5 is included when n/3 == 5.333.
    within(n as f64 / 3.0, 4.0, n as f64)
}
