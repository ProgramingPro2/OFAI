//! Official Doomsday Clock math and tick (off unless GameConfig enables it).

use crate::config::max_troops;
use crate::game::Game;
use crate::spec::PlayerType;
use crate::units::city_levels;

const LEVELS: [i32; 7] = [200, 400, 700, 1100, 1700, 2500, 3500];
const WARN: i32 = 30;
const DRAIN_START: i32 = 2;
const DRAIN_MAX: i32 = 5;
const DRAIN_RAMP: i32 = 90;
const DRAIN_FLOOR: i32 = 5;
const FLOOR_START: i32 = 40;
const FLOOR_DECAY: i32 = 90;
const ROT_DEATH: i32 = 150;
const DRAIN_CURVE_SCALE: i64 = 1_000_000;

fn schedule(speed: &str) -> (i32, [i32; 7], [i32; 7]) {
    match speed {
        "slow" => (600, [240; 7], [70, 70, 70, 70, 70, 70, 0]),
        "fast" => (600, [102; 7], [31, 31, 31, 31, 31, 31, 0]),
        "veryfast" => (600, [36; 7], [8, 8, 8, 8, 8, 8, 0]),
        _ => (600, [168; 7], [54, 54, 54, 54, 54, 54, 0]),
    }
}

fn required_bp(speed: &str, elapsed: i32) -> i32 {
    let (grace, ramps, pauses) = schedule(speed);
    if elapsed <= grace {
        return 0;
    }
    let mut t = elapsed - grace;
    let mut prev = 0;
    for i in 0..LEVELS.len() {
        let ramp = ramps[i];
        let target = LEVELS[i];
        if t < ramp {
            return prev + ((target - prev) * t) / ramp;
        }
        t -= ramp;
        if t < pauses[i] {
            return target;
        }
        t -= pauses[i];
        prev = target;
    }
    LEVELS[LEVELS.len() - 1]
}

fn required_tiles(speed: &str, land: i32, elapsed: i32) -> i32 {
    if land <= 0 {
        return 0;
    }
    (required_bp(speed, elapsed) * land) / 10000
}

fn troop_floor(max_troops: i64, seconds_past_warn: i32) -> i64 {
    let span = FLOOR_START - DRAIN_FLOOR;
    let t = seconds_past_warn.max(0);
    let pct = if span > 0 && FLOOR_DECAY > 0 && t < FLOOR_DECAY {
        FLOOR_START - (span * t) / FLOOR_DECAY
    } else {
        DRAIN_FLOOR
    };
    (max_troops * pct as i64) / 100
}

fn drain(max_troops: i64, seconds_past_warn: i32) -> i64 {
    let t = seconds_past_warn.max(0);
    let span = DRAIN_MAX - DRAIN_START;
    let pct = if DRAIN_RAMP > 0 && t < DRAIN_RAMP {
        DRAIN_START + (span * t) / DRAIN_RAMP
    } else {
        DRAIN_MAX
    };
    ((max_troops * pct as i64) / 100).max(1)
}

pub fn tick(game: &mut Game, ticks: u32) {
    if ticks % 10 != 0 || !game.later.doomsday_enabled {
        return;
    }
    if game.in_spawn_phase() {
        return;
    }
    let elapsed = (game.ticks_since_start() / 10) as i32;
    let land = game.map.num_land_tiles() as i32 - game.map.num_tiles_with_fallout() as i32;
    let required = required_tiles(&game.later.doomsday_speed, land, elapsed);
    let mut sides: Vec<(String, u32)> = game
        .all_players()
        .filter(|p| p.is_alive() && p.player_type() != PlayerType::Bot)
        .map(|p| (p.id().to_string(), p.num_tiles()))
        .collect();
    if sides.len() < 2 {
        for p in game.players_mut_ids() {
            if let Some(pl) = game.player_mut(&p) {
                pl.in_doomsday = false;
                pl.doomsday_ticks = 0;
            }
        }
        return;
    }
    let mut leader = 0usize;
    for i in 1..sides.len() {
        if sides[i].1 > sides[leader].1 {
            leader = i;
        }
    }
    for (i, (pid, tiles)) in sides.iter().enumerate() {
        if i != leader && (*tiles as i32) < required {
            let cities = city_levels(game.units(), pid);
            let diff = game.difficulty();
            let (pt, tiles, under) = {
                let Some(p) = game.player_mut(pid) else {
                    continue;
                };
                p.in_doomsday = true;
                p.doomsday_ticks = p.doomsday_ticks.saturating_add(10);
                (p.player_type(), p.num_tiles(), (p.doomsday_ticks / 10) as i32)
            };
            if under >= WARN {
                let past = under - WARN;
                let max = max_troops(pt, diff, tiles, cities) as i64;
                let floor = troop_floor(max, past);
                let chunk = drain(max, past);
                if let Some(p) = game.player_mut(pid) {
                    let removable = (p.troops - floor).max(0);
                    p.troops -= chunk.min(removable);
                }
            }
        } else if let Some(p) = game.player_mut(pid) {
            p.in_doomsday = false;
            p.doomsday_ticks = 0;
        }
    }
    let _ = DRAIN_CURVE_SCALE;
    let _ = ROT_DEATH;
}

impl Game {
    pub(crate) fn players_mut_ids(&self) -> Vec<String> {
        self.all_players().map(|p| p.id().to_string()).collect()
    }

    pub(crate) fn difficulty(&self) -> crate::spec::Difficulty {
        self.difficulty
    }
}
