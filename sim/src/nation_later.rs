//! Official Nation later-system hooks: structures, warships, nukes, MIRV.

use std::collections::{HashMap, HashSet};

use indexmap::IndexSet;

use crate::config::{
    nuke_magnitudes, nuke_speed, sam_range, train_gold, DEFAULT_NUKE_TARGETABLE_RANGE,
    MAX_SAM_RANGE, SAM_COOLDOWN, TRAIN_STATION_MAX_RANGE, TRAIN_STATION_MIN_RANGE,
};
use crate::diplomacy::Relation;
use crate::game::Game;
use crate::later::LaterExec;
use crate::map::TileRef;
use crate::parabola::ParabolaPath;
use crate::rng::PseudoRandom;
use crate::spec::{Difficulty, PlayerType};
use crate::units::UnitType;

const MIRV_COOLDOWN: u32 = 300;
const FIRST_SILO_RATIO: f64 = 0.4;
const MAX_SILOS: u32 = 3;
const FACTORY_COASTAL: f64 = 0.33;
const HIGH_GOLD: i64 = 3_000_000;
const TILES_PER_CITY: u32 = 2000;
const SHARED_WATER_TTL: u32 = 30;
const OCEAN_SENTINEL: i32 = -1;
const MIN_PORT_WATER_COMPONENT_SIZE: u32 = 3000;
const MAX_NATION_SILO_UPGRADE_LEVEL: u32 = 5;
const NUKE_EVENT_MAX_AGE: u32 = 600;

#[derive(Default)]
pub struct NationLater {
    pub last_structure: Option<u32>,
    pub placements: u32,
    pub high_gold: Option<bool>,
    pub post_save_up: Option<u32>,
    pub tracked_transports: IndexSet<u32>,
    pub tracked_trade: IndexSet<u32>,
    pub tracked_incoming: IndexSet<u32>,
    pub dealt_incoming: IndexSet<u32>,
    pub is_hydro: Option<bool>,
    pub atom_perceived: Option<i64>,
    pub hydro_perceived: Option<i64>,
    pub recent_nukes: Vec<(u32, TileRef, UnitType)>,
    pub last_emoji_sent: HashMap<String, u32>,
    /// Official `NationExecution.embargoMalusApplied`.
    pub embargo_malus_applied: HashSet<String>,
    /// Official `NationEmojiBehavior.gameOver`.
    pub emoji_game_over: bool,
    /// Official `Player.outgoingEmojis_` last-send tick per recipient id (`*` = all).
    pub outgoing_emoji_at: HashMap<String, u32>,
    /// Official `AiAttackBehavior.botAttackTroopsSent`. Reset only when
    /// `attackBots` finds at least one nearby unfriendly bot; leftover budget
    /// applies to later bot-target sends (weakest/hated/assist) on later ticks.
    pub bot_attack_troops_sent: f64,
}

impl Game {
    pub(crate) fn nation_later_mut(&mut self, pid: &str) -> &mut NationLater {
        self.later.nations.entry(pid.to_string()).or_default()
    }

    pub(crate) fn nation_rng(&mut self, idx: usize) -> Option<&mut PseudoRandom> {
        match &mut self.execs[idx] {
            crate::game::Exec::Nation(n) => Some(&mut n.rng),
            _ => None,
        }
    }
}

pub fn handle_structures(game: &mut Game, pid: &str, idx: usize) {
    let _ = has_high_starting_gold(game, pid);
    let placements = game
        .later
        .nations
        .get(pid)
        .map(|s| s.placements)
        .unwrap_or(0);
    if placements > 0 && !game.is_unit_disabled(UnitType::DefensePost) {
        if try_build_defense_post(game, pid, idx) {
            return;
        }
        if defense_post_needed(game, pid) {
            return;
        }
    }
    if is_on_structure_cooldown(game, pid) {
        return;
    }
    if is_in_post_save_up_blocked_phase(game, pid) {
        return;
    }
    if do_handle_structures(game, pid, idx) {
        let now = game.ticks();
        let st = game.later.nations.entry(pid.to_string()).or_default();
        st.last_structure = Some(now);
        st.placements += 1;
    }
}

fn rebuild_shared_water(game: &mut Game) {
    let mut player_water: Vec<(String, bool, HashSet<i32>)> = Vec::new();
    let mut lake_partners: HashMap<i32, Vec<String>> = HashMap::new();
    let pids: Vec<String> = game
        .all_players()
        .filter(|p| p.player_type() != PlayerType::Bot)
        .map(|p| p.id().to_string())
        .collect();
    for pid in &pids {
        let Some(p) = game.player(pid) else {
            continue;
        };
        let mut has_ocean = false;
        let mut lakes = HashSet::new();
        let mut nbuf = [0u32; 4];
        for &tile in &p.border_tiles {
            if !game.map.is_shore(tile) {
                continue;
            }
            let n = game.map.neighbors4(tile, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if !game.map.is_water(nb) {
                    continue;
                }
                if game.map.is_ocean(nb) {
                    has_ocean = true;
                    continue;
                }
                if let Some(comp) = game.water.water_id(nb) {
                    lakes.insert(comp);
                }
            }
        }
        for &c in &lakes {
            lake_partners.entry(c).or_default().push(pid.clone());
        }
        player_water.push((pid.clone(), has_ocean, lakes));
    }
    let mut result = HashMap::new();
    for (pid, has_ocean, lakes) in player_water {
        let mut shared = HashSet::new();
        if has_ocean {
            shared.insert(OCEAN_SENTINEL);
        }
        for c in lakes {
            let Some(partners) = lake_partners.get(&c) else {
                continue;
            };
            if partners
                .iter()
                .any(|other| other != &pid && game.can_trade(&pid, other))
            {
                shared.insert(c);
            }
        }
        result.insert(pid, if shared.is_empty() { None } else { Some(shared) });
    }
    game.later.shared_water = result;
    game.later.shared_water_tick = Some(game.ticks());
}

fn shared_water_components(game: &mut Game, pid: &str) -> Option<HashSet<i32>> {
    let stale = match game.later.shared_water_tick {
        None => true,
        Some(t) => game.ticks().saturating_sub(t) >= SHARED_WATER_TTL,
    };
    if stale {
        rebuild_shared_water(game);
    }
    game.later.shared_water.get(pid).cloned().flatten()
}

fn do_handle_structures(game: &mut Game, pid: &str, idx: usize) -> bool {
    let cities_disabled = game.is_unit_disabled(UnitType::City);
    let city_owned = owned_count(game, pid, UnitType::City);
    let city_count = if cities_disabled {
        let tiles = game.player(pid).map(|p| p.num_tiles()).unwrap_or(0);
        (tiles / TILES_PER_CITY).max(1)
    } else {
        city_owned
    };
    let has_coast = shared_water_components(game, pid).is_some();
    let placements = game
        .later
        .nations
        .get(pid)
        .map(|s| s.placements)
        .unwrap_or(0);
    let silos_enabled = !game.is_unit_disabled(UnitType::MissileSilo);
    if placements == 0
        && matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible)
        && !game.is_unit_disabled(UnitType::AtomBomb)
        && silos_enabled
        && !game.is_unit_disabled(UnitType::SamLauncher)
        && has_high_starting_gold(game, pid)
        && maybe_spawn_structure(game, pid, idx, UnitType::SamLauncher)
    {
        return true;
    }
    if !cities_disabled && city_owned == 0 && is_high_nation_density(game) {
        let preferred = if has_coast && !game.is_unit_disabled(UnitType::Port) {
            UnitType::Port
        } else {
            UnitType::Factory
        };
        if !game.is_unit_disabled(preferred) && maybe_spawn_structure(game, pid, idx, preferred) {
            return true;
        }
    }
    let nukes_enabled = !game.is_unit_disabled(UnitType::AtomBomb)
        || !game.is_unit_disabled(UnitType::HydrogenBomb)
        || !game.is_unit_disabled(UnitType::Mirv);
    for ty in [
        UnitType::Port,
        UnitType::Factory,
        UnitType::SamLauncher,
        UnitType::MissileSilo,
    ] {
        if game.is_unit_disabled(ty) {
            continue;
        }
        if ty == UnitType::Port && !has_coast {
            continue;
        }
        if !nukes_enabled && matches!(ty, UnitType::MissileSilo | UnitType::SamLauncher) {
            continue;
        }
        if !silos_enabled && ty == UnitType::SamLauncher {
            continue;
        }
        if should_build(game, pid, ty, city_count, has_coast)
            && maybe_spawn_structure(game, pid, idx, ty)
        {
            return true;
        }
    }
    !cities_disabled && maybe_spawn_structure(game, pid, idx, UnitType::City)
}

fn has_high_starting_gold(game: &mut Game, pid: &str) -> bool {
    if let Some(v) = game.later.nations.get(pid).and_then(|s| s.high_gold) {
        return v;
    }
    let v = game.later.starting_gold >= HIGH_GOLD;
    game.later.nations.entry(pid.to_string()).or_default().high_gold = Some(v);
    v
}

fn is_high_nation_density(game: &Game) -> bool {
    let land = game.map.num_land_tiles();
    if land == 0 {
        return false;
    }
    let nations = game
        .all_players()
        .filter(|p| p.player_type() == PlayerType::Nation)
        .count();
    nations as f64 / land as f64 > 1.0 / 7500.0
}

fn is_on_structure_cooldown(game: &Game, pid: &str) -> bool {
    let Some(st) = game.later.nations.get(pid) else {
        return false;
    };
    let (Some(last), Some(true)) = (st.last_structure, st.high_gold) else {
        return false;
    };
    let gap = match st.placements {
        0 | 1 => 0,
        2 => 250,
        3 => 150,
        4 => 100,
        _ => 0,
    };
    gap > 0 && game.ticks().saturating_sub(last) < gap
}

fn is_in_post_save_up_blocked_phase(game: &mut Game, pid: &str) -> bool {
    if game.is_unit_disabled(UnitType::MissileSilo) {
        return false;
    }
    let target = save_up_target(game, pid);
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    let now = game.ticks();
    let st = game.later.nations.entry(pid.to_string()).or_default();
    if st.post_save_up.is_none() {
        if gold < target {
            return false;
        }
        st.post_save_up = Some(now);
    }
    let start = st.post_save_up.unwrap();
    let elapsed = now.saturating_sub(start);
    elapsed % 300 >= 150
}

fn maybe_spawn_structure(game: &mut Game, pid: &str, idx: usize, ty: UnitType) -> bool {
    let perceived = perceived_cost(game, pid, ty);
    if game.player(pid).map(|p| p.gold).unwrap_or(0) < perceived {
        return false;
    }
    let density = structure_density(game, pid);
    let existing = owned_count(game, pid, ty);
    if density > 1.0 / 1500.0 && is_upgradable(ty) {
        if maybe_upgrade_structure(game, pid, idx, ty) {
            return true;
        }
        if existing > 0 {
            return false;
        }
    }
    let Some(tile) = structure_spawn_tile(game, pid, idx, ty) else {
        return false;
    };
    if game.can_build(pid, ty, tile).is_none() {
        return false;
    }
    game.uninit_construction(pid, ty, tile);
    true
}

fn is_upgradable(ty: UnitType) -> bool {
    matches!(
        ty,
        UnitType::City
            | UnitType::Port
            | UnitType::Factory
            | UnitType::SamLauncher
            | UnitType::MissileSilo
    )
}

fn structure_density(game: &Game, pid: &str) -> f64 {
    let tiles = game.player(pid).map(|p| p.num_tiles()).unwrap_or(0);
    if tiles == 0 {
        return 0.0;
    }
    let n = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty.is_structure() && u.owner_id == pid)
        .count();
    n as f64 / tiles as f64
}

fn maybe_upgrade_structure(game: &mut Game, pid: &str, idx: usize, ty: UnitType) -> bool {
    let ids: Vec<u32> = game
        .units()
        .iter()
        .filter(|u| {
            u.active
                && u.ty == ty
                && u.owner_id == pid
                && !u.under_construction
                && !u.marked_for_deletion
        })
        .map(|u| u.id)
        .collect();
    if ids.is_empty() {
        return false;
    }
    let Some(id) = find_best_upgrade(game, pid, idx, &ids) else {
        return false;
    };
    let cost = game.structure_cost(pid, ty);
    if game.player(pid).map(|p| p.gold).unwrap_or(0) < cost {
        return false;
    }
    game.enqueue_later(LaterExec::Upgrade {
        active: true,
        owner: pid.to_string(),
        unit_id: id,
        inited: false,
    });
    true
}

fn find_best_upgrade(game: &mut Game, pid: &str, idx: usize, ids: &[u32]) -> Option<u32> {
    let chance = match game.difficulty {
        Difficulty::Easy => 70,
        Difficulty::Medium => 40,
        Difficulty::Hard => 25,
        Difficulty::Impossible => 10,
    };
    if game
        .nation_rng(idx)
        .map(|r| r.next_int(0.0, 100.0) < chance)
        .unwrap_or(false)
    {
        return game.nation_rng(idx).map(|r| r.rand_element(ids));
    }
    let range = crate::config::sam_range(1);
    let range2 = range * range;
    let sams: Vec<(TileRef, u32)> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::SamLauncher && u.owner_id == pid)
        .map(|u| (u.tile, u.level))
        .collect();
    let tiles: Vec<(u32, TileRef)> = ids
        .iter()
        .filter_map(|&id| {
            game.units()
                .iter()
                .find(|u| u.id == id)
                .map(|u| (id, u.tile))
        })
        .collect();
    let mut scored = Vec::new();
    for (id, tile) in tiles {
        let mut score = 0.0;
        for (sam_tile, sam_lv) in &sams {
            let r = crate::config::sam_range(*sam_lv);
            let dx = game.map.x(tile) as f64 - game.map.x(*sam_tile) as f64;
            let dy = game.map.y(tile) as f64 - game.map.y(*sam_tile) as f64;
            if dx * dx + dy * dy <= r * r {
                score += 10.0;
                if *sam_lv > 1 {
                    score += (*sam_lv as f64 - 1.0) * 7.5;
                }
            }
        }
        let jitter = game
            .nation_rng(idx)
            .map(|r| r.next_int(0.0, 5.0))
            .unwrap_or(0) as f64;
        scored.push((id, score + jitter));
    }
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if scored.len() >= 2
        && game.nation_rng(idx).map(|r| r.chance(2)).unwrap_or(false)
    {
        let pick = if scored.len() >= 3 {
            game.nation_rng(idx)
                .map(|r| r.next_int(1.0, 3.0) as usize)
                .unwrap_or(1)
        } else {
            1
        };
        return scored.get(pick).map(|s| s.0);
    }
    let _ = (pid, range2);
    scored.first().map(|s| s.0)
}

fn owned_count(game: &Game, pid: &str, ty: UnitType) -> u32 {
    game.units()
        .iter()
        .filter(|u| u.active && u.ty == ty && u.owner_id == pid)
        .map(|u| {
            if u.under_construction {
                1
            } else {
                u.level
            }
        })
        .sum()
}

fn should_build(game: &Game, pid: &str, ty: UnitType, cities: u32, coast: bool) -> bool {
    let owned = owned_count(game, pid, ty);
    let mut ratio = match ty {
        UnitType::Port | UnitType::Factory => 0.75,
        UnitType::SamLauncher => match game.difficulty {
            Difficulty::Easy => 0.15,
            Difficulty::Medium => 0.2,
            Difficulty::Hard => 0.25,
            Difficulty::Impossible => 0.3,
        },
        UnitType::MissileSilo => 0.2,
        UnitType::City => 1.0,
        _ => return false,
    };
    if ty == UnitType::Factory && coast && !game.is_unit_disabled(UnitType::Port) {
        ratio *= FACTORY_COASTAL;
    }
    if ty == UnitType::MissileSilo {
        if owned >= MAX_SILOS {
            return false;
        }
        if owned == 0 {
            ratio = FIRST_SILO_RATIO;
        }
    }
    owned < (cities as f64 * ratio).floor() as u32
}

fn perceived_cost(game: &Game, pid: &str, ty: UnitType) -> i64 {
    let real = game.structure_cost(pid, ty);
    let target = save_up_target(game, pid);
    if target == 0 || game.player(pid).map(|p| p.gold).unwrap_or(0) >= target {
        return real;
    }
    let owned = owned_count(game, pid, ty) as f64;
    // Official NationStructureBehavior.getStructureRatios / CITY_PERCEIVED.
    let inc = match ty {
        UnitType::SamLauncher => 0.3,
        UnitType::City | UnitType::Port | UnitType::Factory | UnitType::MissileSilo => 1.0,
        _ => 0.1,
    };
    (real as f64 * (1.0 + inc * owned)).ceil() as i64
}

fn save_up_target(game: &Game, pid: &str) -> i64 {
    if game.is_unit_disabled(UnitType::MissileSilo) {
        return game.structure_cost(pid, UnitType::SamLauncher);
    }
    if !game.is_unit_disabled(UnitType::Mirv) {
        return game.structure_cost(pid, UnitType::Mirv)
            + game.structure_cost(pid, UnitType::HydrogenBomb);
    }
    if !game.is_unit_disabled(UnitType::HydrogenBomb) {
        return game.structure_cost(pid, UnitType::HydrogenBomb) * 5;
    }
    if !game.is_unit_disabled(UnitType::AtomBomb) {
        return game.structure_cost(pid, UnitType::AtomBomb) * 20;
    }
    game.structure_cost(pid, UnitType::SamLauncher)
}

fn structure_spawn_tile(
    game: &mut Game,
    pid: &str,
    idx: usize,
    ty: UnitType,
) -> Option<TileRef> {
    let tiles = if ty == UnitType::Port {
        rand_coastal_tiles(game, pid, idx, 25)
    } else {
        rand_territory_tiles(game, pid, idx, 25)
    };
    if tiles.is_empty() {
        return None;
    }
    let use_connect = should_use_connectivity(game, idx, ty);
    // Official `samLauncherValue` rolls coverage weighting once per placement.
    let sam_coverage = if ty == UnitType::SamLauncher {
        Some(sam_use_coverage_weighting(game, idx))
    } else {
        None
    };
    let small = game.player(pid).map(|p| p.small_id).unwrap_or(0);
    let mut best = None;
    let mut best_v = 0.0;
    for &t in &tiles {
        let v = if let Some(coverage) = sam_coverage {
            sam_tile_value(game, pid, t, coverage)
        } else {
            structure_tile_value(game, pid, ty, t, use_connect)
        };
        if best.is_some() && v <= best_v {
            continue;
        }
        if game.map.owner_id(t) != small || game.can_build(pid, ty, t).is_none() {
            continue;
        }
        best = Some(t);
        best_v = v;
    }
    best
}

fn sam_use_coverage_weighting(game: &mut Game, idx: usize) -> bool {
    if game.difficulty == Difficulty::Easy {
        return false;
    }
    game.nation_rng(idx)
        .map(|r| r.next_int(0.0, 100.0) < 25)
        .unwrap_or(false)
}

fn sam_tile_value(game: &Game, pid: &str, tile: TileRef, use_coverage: bool) -> f64 {
    let (border_sp, struct_sp) = spacing_constants();
    let mut w = game.map.magnitude(tile) as f64;
    if let Some(d) = game
        .player(pid)
        .and_then(|p| closest_two_dist(game, &p.border_tiles.iter().copied().collect::<Vec<_>>(), tile))
    {
        w += d.min(border_sp);
    }
    let same: Vec<TileRef> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::SamLauncher && u.owner_id == pid && u.tile != tile)
        .map(|u| u.tile)
        .collect();
    if let Some(d) = closest_two_dist(game, &same, tile) {
        w += d.min(struct_sp);
    }
    if game.difficulty == Difficulty::Easy {
        return w;
    }
    let weight_by_level = matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible);
    let protect: Vec<(TileRef, f64)> = game
        .units()
        .iter()
        .filter(|u| {
            u.active
                && u.owner_id == pid
                && matches!(
                    u.ty,
                    UnitType::City | UnitType::Factory | UnitType::MissileSilo | UnitType::Port
                )
        })
        .map(|u| (u.tile, if weight_by_level { u.level as f64 } else { 1.0 }))
        .collect();
    let range = crate::config::sam_range(1);
    let range2 = range * range;
    let existing_sams: Vec<(TileRef, u32)> = if use_coverage {
        game.units()
            .iter()
            .filter(|u| u.active && u.ty == UnitType::SamLauncher && u.owner_id == pid)
            .map(|u| (u.tile, u.level))
            .collect()
    } else {
        Vec::new()
    };
    for (ptile, weight) in protect {
        let dx = game.map.x(tile) as f64 - game.map.x(ptile) as f64;
        let dy = game.map.y(tile) as f64 - game.map.y(ptile) as f64;
        if dx * dx + dy * dy > range2 {
            continue;
        }
        if use_coverage {
            let mut coverage = 0.0;
            for (sam_tile, sam_lv) in &existing_sams {
                let r = crate::config::sam_range(*sam_lv);
                let sx = game.map.x(ptile) as f64 - game.map.x(*sam_tile) as f64;
                let sy = game.map.y(ptile) as f64 - game.map.y(*sam_tile) as f64;
                if sx * sx + sy * sy <= r * r {
                    coverage += *sam_lv as f64;
                }
            }
            w += struct_sp * weight * (1.0 / (1.0 + coverage));
        } else {
            w += struct_sp * weight;
        }
    }
    w
}

fn should_use_connectivity(game: &mut Game, idx: usize, ty: UnitType) -> bool {
    if !matches!(ty, UnitType::City | UnitType::Factory) {
        return false;
    }
    let chance = match game.difficulty {
        Difficulty::Easy => 0,
        Difficulty::Medium => 60,
        Difficulty::Hard => 75,
        Difficulty::Impossible => 100,
    };
    game.nation_rng(idx)
        .map(|r| r.next_int(0.0, 100.0) < chance)
        .unwrap_or(false)
}

fn structure_tile_value(
    game: &Game,
    pid: &str,
    ty: UnitType,
    tile: TileRef,
    use_connect: bool,
) -> f64 {
    let (border_sp, struct_sp) = spacing_constants();
    let same: Vec<TileRef> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == ty && u.owner_id == pid && u.tile != tile)
        .map(|u| u.tile)
        .collect();
    if ty == UnitType::Port {
        // Official `portValue` is distance to other ports only — no elevation.
        return closest_tile_dist(game, same.into_iter(), tile);
    }
    let mut w = game.map.magnitude(tile) as f64;
    let border = game
        .player(pid)
        .map(|p| closest_tile_dist(game, p.border_tiles.iter().copied(), tile))
        .unwrap_or(f64::INFINITY);
    w += border.min(border_sp);
    if let Some((other, _)) = closest_two_tiles(game, &same, &[tile]) {
        let d = game.map.manhattan(other, tile) as f64;
        let cap = if ty == UnitType::Factory {
            TRAIN_STATION_MAX_RANGE as f64
        } else {
            struct_sp
        };
        w += d.min(cap);
    }
    let cross_ty = if ty == UnitType::City {
        UnitType::Factory
    } else if ty == UnitType::Factory {
        UnitType::City
    } else {
        ty
    };
    if cross_ty != ty {
        let cross: Vec<TileRef> = game
            .units()
            .iter()
            .filter(|u| u.active && u.ty == cross_ty && u.owner_id == pid)
            .map(|u| u.tile)
            .collect();
        if let Some((other, _)) = closest_two_tiles(game, &cross, &[tile]) {
            let d = game.map.manhattan(other, tile) as f64;
            w += d.min(struct_sp);
        }
    }
    if use_connect {
        w += connectivity_score(game, pid, tile) * struct_sp;
    }
    w
}

fn reachable_stations(game: &Game, pid: &str) -> Vec<(TileRef, Option<u32>, f64)> {
    let max_gold = train_gold("ally", 0).max(1) as f64;
    let self_w = train_gold("self", 0) as f64 / max_gold;
    let mut out = Vec::new();
    let push_owned = |game: &Game, owner: &str, weight: f64, out: &mut Vec<(TileRef, Option<u32>, f64)>| {
        for u in game.units() {
            if !u.active || u.owner_id != owner {
                continue;
            }
            if !matches!(u.ty, UnitType::City | UnitType::Port | UnitType::Factory) {
                continue;
            }
            let Some(sid) = game.later.rail.find_by_unit(u.id) else {
                continue;
            };
            let cluster = game.later.rail.stations.get(&sid).and_then(|s| s.cluster);
            out.push((u.tile, cluster, weight));
        }
    };
    push_owned(game, pid, self_w, &mut out);
    for n in game.nearby(pid).into_iter().flatten() {
        let Some(other) = game.player(&n) else {
            continue;
        };
        if other.player_type() == PlayerType::Bot {
            continue;
        }
        if !game.can_trade(pid, &n) {
            continue;
        }
        let rel = if game.is_allied(pid, &n) {
            "ally"
        } else {
            "other"
        };
        let w = train_gold(rel, 0) as f64 / max_gold;
        push_owned(game, &n, w, &mut out);
    }
    out
}

fn spacing_constants() -> (f64, f64) {
    let border = nuke_magnitudes(UnitType::AtomBomb).1 as f64;
    (border, border * 2.0)
}

fn closest_tile_dist(game: &Game, refs: impl Iterator<Item = TileRef>, tile: TileRef) -> f64 {
    let mut best = f64::INFINITY;
    for r in refs {
        let d = game.map.manhattan(r, tile) as f64;
        if d < best {
            best = d;
        }
    }
    best
}

fn closest_two_dist(game: &Game, others: &[TileRef], tile: TileRef) -> Option<f64> {
    if others.is_empty() {
        return None;
    }
    Some(closest_tile_dist(game, others.iter().copied(), tile))
}

fn connectivity_score(game: &Game, pid: &str, tile: TileRef) -> f64 {
    let min2 = (TRAIN_STATION_MIN_RANGE as i64) * (TRAIN_STATION_MIN_RANGE as i64);
    let max2 = (TRAIN_STATION_MAX_RANGE as i64) * (TRAIN_STATION_MAX_RANGE as i64);
    let stations = reachable_stations(game, pid);
    let mut clusters: HashMap<u32, f64> = HashMap::new();
    let mut isolated = 0.0;
    for (st, cluster, weight) in stations {
        let dx = game.map.x(tile) as i64 - game.map.x(st) as i64;
        let dy = game.map.y(tile) as i64 - game.map.y(st) as i64;
        let d2 = dx * dx + dy * dy;
        if d2 < min2 || d2 > max2 {
            continue;
        }
        if let Some(cid) = cluster {
            let e = clusters.entry(cid).or_insert(0.0);
            if weight > *e {
                *e = weight;
            }
        } else {
            isolated += weight;
        }
    }
    isolated + clusters.values().sum::<f64>()
}

fn rand_territory_tiles(game: &mut Game, pid: &str, idx: usize, n: usize) -> Vec<TileRef> {
    let Some(p) = game.player(pid) else {
        return Vec::new();
    };
    let bbox = bounding_box(game, p.border_tiles.iter().copied());
    let small = p.small_id;
    let owned: Vec<TileRef> = p.tiles.iter().copied().collect();
    let mut out = Vec::new();
    for _ in 0..n {
        if let Some(t) = rand_territory_tile(game, idx, small, bbox, &owned) {
            out.push(t);
        }
    }
    out
}

fn bounding_box(game: &Game, tiles: impl Iterator<Item = TileRef>) -> Option<(u32, u32, u32, u32)> {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    let mut any = false;
    for t in tiles {
        any = true;
        min_x = min_x.min(game.map.x(t));
        min_y = min_y.min(game.map.y(t));
        max_x = max_x.max(game.map.x(t));
        max_y = max_y.max(game.map.y(t));
    }
    any.then_some((min_x, min_y, max_x, max_y))
}

fn rand_territory_tile(
    game: &mut Game,
    idx: usize,
    small: u16,
    bbox: Option<(u32, u32, u32, u32)>,
    owned: &[TileRef],
) -> Option<TileRef> {
    let Some((min_x, min_y, max_x, max_y)) = bbox else {
        return None;
    };
    for _ in 0..100 {
        let (x, y) = {
            let Some(rng) = game.nation_rng(idx) else {
                return None;
            };
            (
                rng.next_int(min_x as f64, max_x as f64),
                rng.next_int(min_y as f64, max_y as f64),
            )
        };
        if x < 0 || y < 0 {
            continue;
        }
        let Ok(t) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if game.map.owner_id(t) == small {
            return Some(t);
        }
    }
    if (1..=100).contains(&owned.len()) {
        return game.nation_rng(idx).map(|r| r.rand_element(owned));
    }
    None
}

fn rand_coastal_tiles(game: &mut Game, pid: &str, idx: usize, n: usize) -> Vec<TileRef> {
    let shared = shared_water_components(game, pid);
    let Some(p) = game.player(pid) else {
        return Vec::new();
    };
    let easy = game.difficulty == Difficulty::Easy;
    let tiles: Vec<TileRef> = p
        .border_tiles
        .iter()
        .copied()
        .filter(|&t| {
            if !game.map.is_shore(t) {
                return false;
            }
            let Some(ref shared) = shared else {
                return false;
            };
            let mut nbuf = [0u32; 4];
            let ncount = game.map.neighbors4(t, &mut nbuf);
            for i in 0..ncount {
                let nb = nbuf[i];
                if !game.map.is_water(nb) {
                    continue;
                }
                if game.map.is_ocean(nb) {
                    return true;
                }
                let Some(comp) = game.water.water_id(nb) else {
                    continue;
                };
                if !shared.contains(&comp) {
                    continue;
                }
                if !easy {
                    let size = game.water.component_size(comp);
                    if size < MIN_PORT_WATER_COMPONENT_SIZE {
                        continue;
                    }
                }
                return true;
            }
            false
        })
        .collect();
    if tiles.len() <= n {
        return tiles;
    }
    let Some(rng) = game.nation_rng(idx) else {
        return Vec::new();
    };
    let mut remaining = tiles;
    let mut out = Vec::new();
    for _ in 0..n {
        if remaining.is_empty() {
            break;
        }
        let i = rng.next_int(0.0, remaining.len() as f64) as usize;
        out.push(remaining.remove(i.min(remaining.len() - 1)));
    }
    out
}

fn incoming_land_attacks(game: &Game, pid: &str) -> Vec<(u16, f64)> {
    game.all_players()
        .flat_map(|o| {
            o.outgoing.iter().filter_map(move |a| {
                if a.active
                    && a.source_tile.is_none()
                    && a.target_player_id.as_deref() == Some(pid)
                {
                    Some((o.small_id, a.troops))
                } else {
                    None
                }
            })
        })
        .collect()
}

fn attack_front_tiles(game: &Game, pid: &str, attackers: &HashSet<u16>) -> Vec<TileRef> {
    let Some(p) = game.player(pid) else {
        return Vec::new();
    };
    if attackers.is_empty() {
        return Vec::new();
    }
    let mut front = Vec::new();
    let mut nbuf = [0u32; 4];
    for &border in &p.border_tiles {
        let n = game.map.neighbors4(border, &mut nbuf);
        for i in 0..n {
            if attackers.contains(&game.map.owner_id(nbuf[i])) {
                front.push(border);
                break;
            }
        }
    }
    front
}

fn count_defense_posts_near_front(
    game: &Game,
    pid: &str,
    front: &[TileRef],
    cap: u32,
) -> u32 {
    if front.is_empty() {
        return 0;
    }
    let (border_sp, _) = spacing_constants();
    let range2 = (border_sp * 1.5) * (border_sp * 1.5);
    let mut count = 0u32;
    for dp in game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::DefensePost && u.owner_id == pid)
    {
        for &ft in front {
            let dx = game.map.x(dp.tile) as f64 - game.map.x(ft) as f64;
            let dy = game.map.y(dp.tile) as f64 - game.map.y(ft) as f64;
            if dx * dx + dy * dy <= range2 {
                count += 1;
                if count >= cap {
                    return count;
                }
                break;
            }
        }
    }
    count
}

fn sample_tiles_near_front(
    game: &mut Game,
    pid: &str,
    idx: usize,
    front: &[TileRef],
    count: usize,
) -> Vec<TileRef> {
    if front.is_empty() {
        return Vec::new();
    }
    let (border_sp, _) = spacing_constants();
    let search_radius = (border_sp * 1.5).ceil() as i32;
    let min_border = (border_sp * 0.75).ceil() as f64;
    let max_border = (border_sp * 1.5).ceil() as f64;
    let spread2 = (border_sp * 1.5) * (border_sp * 1.5);
    let existing: Vec<TileRef> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::DefensePost && u.owner_id == pid)
        .map(|u| u.tile)
        .collect();
    let anchors: Vec<TileRef> = if existing.is_empty() {
        front.to_vec()
    } else {
        let filtered: Vec<TileRef> = front
            .iter()
            .copied()
            .filter(|&ft| {
                !existing.iter().any(|&dp| {
                    let dx = game.map.x(ft) as f64 - game.map.x(dp) as f64;
                    let dy = game.map.y(ft) as f64 - game.map.y(dp) as f64;
                    dx * dx + dy * dy < spread2
                })
            })
            .collect();
        if filtered.is_empty() {
            front.to_vec()
        } else {
            filtered
        }
    };
    let small = game.player(pid).map(|p| p.small_id).unwrap_or(0);
    let border_tiles: Vec<TileRef> = game
        .player(pid)
        .map(|p| p.border_tiles.iter().copied().collect())
        .unwrap_or_default();
    let mut result = Vec::new();
    let mut attempt = 0;
    while attempt < count * 6 && result.len() < count {
        attempt += 1;
        let Some(anchor) = game.nation_rng(idx).map(|r| r.rand_element(&anchors)) else {
            break;
        };
        let ax = game.map.x(anchor) as i32;
        let ay = game.map.y(anchor) as i32;
        let (x, y) = {
            let Some(rng) = game.nation_rng(idx) else {
                break;
            };
            (
                rng.next_int((ax - search_radius) as f64, (ax + search_radius + 1) as f64),
                rng.next_int((ay - search_radius) as f64, (ay + search_radius + 1) as f64),
            )
        };
        if x < 0
            || y < 0
            || (x as u32) >= game.map.width()
            || (y as u32) >= game.map.height()
        {
            continue;
        }
        let Ok(t) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if game.map.owner_id(t) != small {
            continue;
        }
        let border_dist = closest_tile_dist(game, border_tiles.iter().copied(), t);
        if border_dist < min_border || border_dist > max_border {
            continue;
        }
        if game.can_build(pid, UnitType::DefensePost, t).is_none() {
            continue;
        }
        result.push(t);
    }
    if !result.is_empty() {
        return result;
    }
    let mut fallback = Vec::new();
    let mut attempt = 0;
    while attempt < count * 4 && fallback.len() < count {
        attempt += 1;
        let Some(anchor) = game.nation_rng(idx).map(|r| r.rand_element(&anchors)) else {
            break;
        };
        let ax = game.map.x(anchor) as i32;
        let ay = game.map.y(anchor) as i32;
        let (x, y) = {
            let Some(rng) = game.nation_rng(idx) else {
                break;
            };
            (
                rng.next_int((ax - search_radius) as f64, (ax + search_radius + 1) as f64),
                rng.next_int((ay - search_radius) as f64, (ay + search_radius + 1) as f64),
            )
        };
        if x < 0
            || y < 0
            || (x as u32) >= game.map.width()
            || (y as u32) >= game.map.height()
        {
            continue;
        }
        let Ok(t) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if game.map.owner_id(t) != small {
            continue;
        }
        fallback.push(t);
    }
    fallback
}

fn try_build_defense_post(game: &mut Game, pid: &str, idx: usize) -> bool {
    if game.difficulty == Difficulty::Easy {
        return false;
    }
    if game.difficulty == Difficulty::Medium {
        let ok = game.nation_rng(idx).map(|r| r.chance(2)).unwrap_or(false);
        if !ok {
            return false;
        }
    }
    let land = incoming_land_attacks(game, pid);
    if land.is_empty() {
        return false;
    }
    let ours = game.player(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
    if ours <= 0.0 {
        return false;
    }
    let incoming: f64 = land.iter().map(|(_, t)| *t).sum();
    let ratio = incoming / ours;
    if ratio < 0.35 {
        return false;
    }
    let allowed = if game.difficulty == Difficulty::Medium {
        1
    } else {
        (ratio / 0.4).ceil() as u32
    };
    let attackers: HashSet<u16> = land.iter().map(|(sid, _)| *sid).collect();
    let front = attack_front_tiles(game, pid, &attackers);
    if count_defense_posts_near_front(game, pid, &front, allowed) >= allowed {
        return false;
    }
    let cost = game.structure_cost(pid, UnitType::DefensePost);
    if game.player(pid).map(|p| p.gold).unwrap_or(0) < cost {
        return false;
    }
    let tiles = sample_tiles_near_front(game, pid, idx, &front, 25);
    for t in tiles {
        if game.can_build(pid, UnitType::DefensePost, t).is_some() {
            game.uninit_construction(pid, UnitType::DefensePost, t);
            return true;
        }
    }
    false
}

fn defense_post_needed(game: &Game, pid: &str) -> bool {
    if game.difficulty == Difficulty::Easy {
        return false;
    }
    let land = incoming_land_attacks(game, pid);
    if land.is_empty() {
        return false;
    }
    let ours = game.player(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
    if ours <= 0.0 {
        return false;
    }
    let incoming: f64 = land.iter().map(|(_, t)| *t).sum();
    incoming / ours >= 0.35
}

pub fn maybe_spawn_warship(game: &mut Game, pid: &str, idx: usize) {
    if game.is_unit_disabled(UnitType::Warship) {
        return;
    }
    let chance = {
        let Some(rng) = game.nation_rng(idx) else {
            return;
        };
        rng.chance(50)
    };
    if !chance {
        return;
    }
    // Official `player.units(Port)` walks Player._units insertion order,
    // including captured ports appended on tile conquest. Global `game.units()`
    // is id order and picks the wrong port for randElement.
    let owned: Vec<u32> = game.player(pid).map(|p| p.units.clone()).unwrap_or_default();
    let ports: Vec<TileRef> = owned
        .iter()
        .filter_map(|&id| {
            game.units()
                .iter()
                .find(|u| u.id == id && u.ty == UnitType::Port)
                .map(|u| u.tile)
        })
        .collect();
    let ships = owned
        .iter()
        .filter(|&&id| {
            game.units()
                .iter()
                .any(|u| u.id == id && u.ty == UnitType::Warship)
        })
        .count();
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    let cost = game.structure_cost(pid, UnitType::Warship);
    if ports.is_empty() || ships > 0 || gold <= cost {
        return;
    }
    let port = {
        let Some(rng) = game.nation_rng(idx) else {
            return;
        };
        rng.rand_element(&ports)
    };
    let target = warship_spawn_tile(game, idx, port, 250);
    if let Some(t) = target {
        if game.can_build(pid, UnitType::Warship, t).is_some() {
            game.uninit_construction(pid, UnitType::Warship, t);
        }
    }
}

/// Official `NationWarshipBehavior.counterWarshipInfestation`.
pub fn counter_warship_infestation(game: &mut Game, pid: &str, idx: usize) {
    if game.is_unit_disabled(UnitType::Warship) {
        return;
    }
    if !matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible) {
        return;
    }
    let warships = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::Warship)
        .count();
    if warships <= 10 {
        return;
    }
    let cost = game.structure_cost(pid, UnitType::Warship);
    if cost > game.player(pid).map(|p| p.gold).unwrap_or(0) {
        return;
    }
    let ports = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::Port && u.owner_id == pid)
        .count();
    if ports == 0 {
        return;
    }
    let mine = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::Warship && u.owner_id == pid)
        .count();
    if mine >= 10 {
        return;
    }
    if !nation_is_top_gold(game, pid) {
        return;
    }
    let mut enemies: Vec<(usize, String)> = game
        .alive_players()
        .into_iter()
        .filter(|p| {
            if p.id() == pid {
                return false;
            }
            match game.player(pid) {
                Some(me) => !game.is_friendly(me, p),
                None => true,
            }
        })
        .map(|p| {
            let n = game
                .units()
                .iter()
                .filter(|u| u.active && u.ty == UnitType::Warship && u.owner_id == p.id())
                .count();
            (n, p.id().to_string())
        })
        .collect();
    enemies.retain(|(n, _)| *n > 10);
    let Some((_, eid)) = enemies.into_iter().next() else {
        return;
    };
    let ids: Vec<u32> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::Warship && u.owner_id == eid)
        .map(|u| u.id)
        .collect();
    if ids.is_empty() {
        return;
    }
    let pick = game.nation_rng(idx).map(|r| r.rand_element(&ids)).unwrap_or(ids[0]);
    let Some(tile) = game.units().iter().find(|u| u.id == pick).map(|u| u.tile) else {
        return;
    };
    if game.can_build(pid, UnitType::Warship, tile).is_none() {
        maybe_move_warship(game, pid, tile);
        return;
    }
    game.uninit_construction(pid, UnitType::Warship, tile);
}

fn nation_is_top_gold(game: &Game, pid: &str) -> bool {
    let mut golds: Vec<(i64, String)> = game
        .alive_players()
        .into_iter()
        .filter(|p| p.player_type() != PlayerType::Human)
        .map(|p| (p.gold, p.id().to_string()))
        .collect();
    golds.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    golds.iter().take(3).any(|(_, id)| id == pid)
}

fn warship_spawn_tile(game: &mut Game, idx: usize, port: TileRef, radius: i32) -> Option<TileRef> {
    let px = game.map.x(port) as i32;
    let py = game.map.y(port) as i32;
    for _ in 0..50 {
        let (x, y) = {
            let Some(rng) = game.nation_rng(idx) else {
                return None;
            };
            (
                rng.next_int((px - radius) as f64, (px + radius) as f64),
                rng.next_int((py - radius) as f64, (py + radius) as f64),
            )
        };
        if x < 0 || y < 0 {
            continue;
        }
        let Ok(t) = game.map.ref_xy(x as u32, y as u32) else {
            continue;
        };
        if game.map.is_water(t) {
            return Some(t);
        }
    }
    None
}

pub fn consider_mirv(game: &mut Game, pid: &str, idx: usize) {
    if game.is_unit_disabled(UnitType::Mirv) {
        return;
    }
    if !player_has_silo(game, pid) {
        return;
    }
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    if gold < game.structure_cost(pid, UnitType::Mirv) {
        return;
    }
    let odds = match game.difficulty {
        Difficulty::Easy => 2,
        Difficulty::Medium => 4,
        Difficulty::Hard => 8,
        Difficulty::Impossible => 16,
    };
    let hesitate = game.nation_rng(idx).map(|r| r.chance(odds)).unwrap_or(true);
    if hesitate {
        return;
    }
    if let Some(oid) = select_counter_mirv_target(game, pid) {
        if !was_recently_mirved(game, &oid) {
            maybe_send_mirv(game, pid, idx, &oid);
            return;
        }
    }
    if let Some(oid) = select_victory_denial_target(game, pid) {
        if !was_recently_mirved(game, &oid) {
            maybe_send_mirv(game, pid, idx, &oid);
            return;
        }
    }
    if let Some(oid) = select_steamroll_stop_target(game, pid) {
        if !was_recently_mirved(game, &oid) {
            maybe_send_mirv(game, pid, idx, &oid);
        }
    }
}

fn valid_mirv_targets(game: &Game, pid: &str) -> Vec<String> {
    game.all_players()
        .filter(|p| {
            p.id() != pid
                && p.is_alive()
                && p.player_type() != PlayerType::Bot
                && !game.same_team(pid, p.id())
        })
        .map(|p| p.id().to_string())
        .collect()
}

fn was_recently_mirved(game: &Game, oid: &str) -> bool {
    match game.later.recent_mirv.get(oid) {
        None => false,
        Some(&last) => game.ticks().saturating_sub(last) < MIRV_COOLDOWN,
    }
}

fn select_counter_mirv_target(game: &Game, pid: &str) -> Option<String> {
    let mut attackers: Vec<(u32, String)> = valid_mirv_targets(game, pid)
        .into_iter()
        .filter(|oid| is_inbound_mirv_from(game, pid, oid))
        .filter_map(|oid| game.player(&oid).map(|p| (p.num_tiles(), oid)))
        .collect();
    attackers.sort_by(|a, b| b.0.cmp(&a.0));
    attackers.into_iter().next().map(|(_, id)| id)
}

fn is_inbound_mirv_from(game: &Game, pid: &str, attacker: &str) -> bool {
    let my_small = game.player(pid).map(|p| p.small_id).unwrap_or(0);
    game.units().iter().any(|u| {
        if !u.active || u.ty != UnitType::Mirv || u.owner_id != attacker {
            return false;
        }
        let Some(dst) = u.target_tile else {
            return false;
        };
        if !game.map.has_owner(dst) {
            return false;
        }
        game.map.owner_id(dst) == my_small
    })
}

fn select_victory_denial_target(game: &Game, pid: &str) -> Option<String> {
    let total = game.map.num_land_tiles();
    if total == 0 {
        return None;
    }
    let team_thresh = match game.difficulty {
        Difficulty::Easy => 0.9,
        Difficulty::Medium => 0.8,
        Difficulty::Hard => 0.7,
        Difficulty::Impossible => 0.6,
    };
    let solo_thresh = match game.difficulty {
        Difficulty::Easy => 0.75,
        Difficulty::Medium => 0.65,
        Difficulty::Hard => 0.55,
        Difficulty::Impossible => 0.4,
    };
    let mut best: Option<(String, f64)> = None;
    for oid in valid_mirv_targets(game, pid) {
        let mut severity = 0.0;
        if game.cfg_team_mode() {
            if let Some(team) = game.player_team(&oid) {
                let members: Vec<(u32, String)> = game
                    .all_players()
                    .filter(|p| p.is_alive() && game.player_team(p.id()) == Some(team))
                    .map(|p| (p.num_tiles(), p.id().to_string()))
                    .collect();
                let team_tiles: u32 = members.iter().map(|m| m.0).sum();
                let team_share = team_tiles as f64 / total as f64;
                if team_share >= team_thresh {
                    let largest = members.iter().max_by_key(|(t, _)| *t).map(|(_, id)| id.as_str());
                    if largest == Some(oid.as_str()) {
                        severity = team_share;
                    }
                }
            }
        } else {
            let share = game.player(&oid).map(|p| p.num_tiles()).unwrap_or(0) as f64 / total as f64;
            if share >= solo_thresh {
                severity = share;
            }
        }
        if severity > 0.0 && best.as_ref().map(|b| severity > b.1).unwrap_or(true) {
            best = Some((oid, severity));
        }
    }
    best.map(|(id, _)| id)
}

fn select_steamroll_stop_target(game: &Game, pid: &str) -> Option<String> {
    let valid = valid_mirv_targets(game, pid);
    if valid.is_empty() {
        return None;
    }
    let min_cities = match game.difficulty {
        Difficulty::Easy => 20,
        Difficulty::Medium | Difficulty::Hard => 10,
        Difficulty::Impossible => 8,
    };
    let gap_mult = match game.difficulty {
        Difficulty::Easy => 2.0,
        Difficulty::Medium => 1.5,
        Difficulty::Hard => 1.25,
        Difficulty::Impossible => 1.15,
    };
    let mut ranked: Vec<(u32, String)> = game
        .all_players()
        .filter(|p| p.is_alive())
        .map(|p| (count_cities(game, p.id()), p.id().to_string()))
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0));
    if ranked.len() < 2 {
        return None;
    }
    let (top_cities, top_id) = &ranked[0];
    if *top_cities <= min_cities {
        return None;
    }
    let second = ranked[1].0 as f64;
    if *top_cities as f64 >= second * gap_mult && valid.iter().any(|id| id == top_id) {
        Some(top_id.clone())
    } else {
        None
    }
}

fn count_cities(game: &Game, pid: &str) -> u32 {
    game.units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::City && u.owner_id == pid)
        .count() as u32
}

fn calculate_territory_center(game: &Game, oid: &str) -> Option<TileRef> {
    let Some(p) = game.player(oid) else {
        return None;
    };
    if p.border_tiles.is_empty() {
        return None;
    }
    let mut min_x = u32::MAX;
    let mut max_x = 0u32;
    let mut min_y = u32::MAX;
    let mut max_y = 0u32;
    for &t in &p.border_tiles {
        let x = game.map.x(t);
        let y = game.map.y(t);
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    let cx = (min_x + max_x) / 2;
    let cy = (min_y + max_y) / 2;
    if let Ok(center) = game.map.ref_xy(cx, cy) {
        if game.map.owner_id(center) == p.small_id {
            return Some(center);
        }
    }
    let mut best = None;
    let mut best_d = i64::MAX;
    for &t in &p.border_tiles {
        let dx = game.map.x(t) as i64 - cx as i64;
        let dy = game.map.y(t) as i64 - cy as i64;
        let d2 = dx * dx + dy * dy;
        if d2 < best_d {
            best_d = d2;
            best = Some(t);
        }
    }
    best
}

fn maybe_send_mirv(game: &mut Game, pid: &str, idx: usize, oid: &str) {
    // Official maybeSendAttackEmoji: nations are not Human, so no RNG.
    let Some(tile) = calculate_territory_center(game, oid) else {
        return;
    };
    if game.can_build(pid, UnitType::Mirv, tile).is_none() {
        return;
    }
    game.launch_mirv(pid.to_string(), tile);
    game.later.recent_mirv.insert(oid.to_string(), game.ticks());
    // Official sendEmoji(AllPlayers, EMOJI_NUKE) then respondToMIRV.
    if let Some(rng) = game.nation_rng(idx) {
        let _ = rng.next_int(0.0, 2.0);
        if rng.chance(8) {
            let _ = rng.next_int(0.0, 8.0);
        }
    }
}

fn player_has_silo(game: &Game, pid: &str) -> bool {
    // Official `player.units(MissileSilo)` includes under-construction silos.
    game.units().iter().any(|u| {
        u.active && u.ty == UnitType::MissileSilo && u.owner_id == pid
    })
}

fn nation_is_richest(game: &Game, pid: &str) -> bool {
    let my_gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    !game.all_players().any(|o| {
        o.id() != pid && o.player_type() == PlayerType::Nation && o.gold > my_gold
    })
}

/// Official `NationNukeBehavior.findBestNukeTarget`.
fn find_best_nuke_target(game: &mut Game, pid: &str, idx: usize) -> Option<String> {
    if matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible) {
        let alive: Vec<String> = game
            .all_players()
            .filter(|p| p.is_alive())
            .map(|p| p.id().to_string())
            .collect();
        if alive.len() == 2 {
            return alive.into_iter().find(|id| id != pid);
        }
    }
    if let Some(atk) = game.incoming_attack_player(pid) {
        return Some(atk);
    }
    if game.difficulty == Difficulty::Impossible && nation_is_richest(game, pid) {
        let rolled = game.nation_rng(idx).map(|r| r.chance(2)).unwrap_or(false);
        if rolled {
            if let Some(id) = find_high_density_nuke_target(game, pid) {
                return Some(id);
            }
        }
    }
    if game.difficulty == Difficulty::Impossible && !game.cfg_team_mode() {
        let land = game
            .map
            .num_land_tiles()
            .saturating_sub(game.map.num_tiles_with_fallout());
        if land > 0 {
            if let Some(crown) = first_by_tiles(game) {
                if crown != pid && !game.is_allied(pid, &crown) {
                    let share = game.player(&crown).map(|p| p.num_tiles()).unwrap_or(0) as f64
                        / land as f64;
                    if share > 0.5 {
                        return Some(crown);
                    }
                }
            }
        }
    }
    for ally in game.ally_ids(pid) {
        if (game
            .player(pid)
            .map(|p| p.relation(&ally))
            .unwrap_or(Relation::Neutral) as i32)
            < (Relation::Friendly as i32)
        {
            continue;
        }
        for tid in game.attackers_of(&ally) {
            if tid == pid || game.is_allied(pid, &tid) {
                continue;
            }
            return Some(tid);
        }
    }
    let my_max = game.player_max_troops(pid);
    let mut hated: Vec<(f64, String)> = game
        .player(pid)
        .map(|p| {
            p.relations
                .iter()
                .filter_map(|(oid, val)| {
                    let alive = game.player(oid).map(|o| o.is_alive()).unwrap_or(false);
                    alive.then_some((*val, oid.clone()))
                })
                .collect()
        })
        .unwrap_or_default();
    hated.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    for (val, oid) in hated {
        if Relation::from_value(val) != Relation::Hostile {
            continue;
        }
        if game.is_allied(pid, &oid) {
            continue;
        }
        let other_max = game.player_max_troops(&oid);
        if my_max >= other_max * 2.0 {
            continue;
        }
        return Some(oid);
    }
    if let Some(crown) = find_ffa_crown_target(game, pid) {
        return Some(crown);
    }
    find_strongest_team_target(game, pid, idx)
}

fn find_high_density_nuke_target(game: &Game, pid: &str) -> Option<String> {
    const HIGH_DENSITY: f64 = 1.0 / 75.0;
    const MIN_LEVEL_SUM: u32 = 5;
    let mut best: Option<(String, f64)> = None;
    for o in game.all_players() {
        if o.id() == pid || o.player_type() == PlayerType::Bot || game.is_allied(pid, o.id()) {
            continue;
        }
        let tiles = o.num_tiles();
        if tiles == 0 {
            continue;
        }
        let level_sum: u32 = game
            .units()
            .iter()
            .filter(|u| u.active && u.owner_id == o.id() && u.ty.is_structure())
            .map(|u| u.level)
            .sum();
        if level_sum < MIN_LEVEL_SUM {
            continue;
        }
        let density = level_sum as f64 / tiles as f64;
        if density > HIGH_DENSITY && best.as_ref().map(|b| density > b.1).unwrap_or(true) {
            best = Some((o.id().to_string(), density));
        }
    }
    best.map(|(id, _)| id)
}

pub fn maybe_send_nuke(game: &mut Game, pid: &str, idx: usize) {
    if !player_has_silo(game, pid) {
        return;
    }
    if game.is_unit_disabled(UnitType::MissileSilo)
        || (game.is_unit_disabled(UnitType::AtomBomb) && game.is_unit_disabled(UnitType::HydrogenBomb))
    {
        return;
    }
    let Some(target) = find_best_nuke_target(game, pid, idx) else {
        return;
    };
    if game
        .player(&target)
        .map(|p| p.player_type() == PlayerType::Bot)
        .unwrap_or(false)
        || game.is_allied(pid, &target)
    {
        return;
    }
    if !nation_should_attack_nuke_target(game, pid, &target, idx) {
        return;
    }
    ensure_nuke_costs(game, pid);
    let hydro_cost = perceived_nuke_cost(game, pid, UnitType::HydrogenBomb);
    let atom_cost = perceived_nuke_cost(game, pid, UnitType::AtomBomb);
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    let is_hydro = game
        .later
        .nations
        .get(pid)
        .and_then(|s| s.is_hydro)
        .unwrap_or(false);
    let nuke_type = if !game.is_unit_disabled(UnitType::HydrogenBomb) && gold >= hydro_cost {
        UnitType::HydrogenBomb
    } else if !game.is_unit_disabled(UnitType::AtomBomb)
        && (!is_hydro || is_under_heavy_attack(game, pid))
        && gold >= atom_cost
    {
        UnitType::AtomBomb
    } else {
        return;
    };
    let range = nuke_magnitudes(nuke_type).1;
    // Official `nukeTarget.units(Structures.types)` follows Player._units
    // insertion/capture order, not global unit-id order. Equal nukeTileScore
    // ties keep the first candidate, so walking game.units() by id picks a
    // different dest (city 9 vs city 48 on greedy holdout[0] step 679).
    let structure_tiles: Vec<TileRef> = game
        .player(&target)
        .map(|p| {
            p.units
                .iter()
                .filter_map(|&id| {
                    game.units().iter().find(|u| {
                        u.id == id && u.active && u.owner_id == target && u.ty.is_structure()
                    })
                })
                .map(|u| u.tile)
                .collect()
        })
        .unwrap_or_default();
    let n_rand = if game.difficulty == Difficulty::Impossible {
        30
    } else {
        10
    };
    let random_tiles = rand_territory_tiles(game, &target, idx, n_rand);
    let mut all_tiles = IndexSet::new();
    for t in &random_tiles {
        all_tiles.insert(*t);
    }
    for t in structure_tiles {
        all_tiles.insert(t);
    }
    remove_old_nuke_events(game, pid);
    let silo_tiles: Vec<TileRef> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::MissileSilo && u.owner_id == pid)
        .map(|u| u.tile)
        .collect();
    let structures: Vec<(UnitType, u32, TileRef)> = game
        .player(&target)
        .map(|p| {
            p.units
                .iter()
                .filter_map(|&id| {
                    game.units().iter().find(|u| {
                        u.id == id && u.active && u.owner_id == target && u.ty.is_structure()
                    })
                })
                .map(|u| (u.ty, u.level, u.tile))
                .collect()
        })
        .unwrap_or_default();
    let mut best_tile: Option<TileRef> = None;
    let mut best_value: f64 = -1.0;
    let tiles: Vec<TileRef> = all_tiles.into_iter().collect();
    for tile in tiles {
        if !nuke_bounding_box_valid(game, pid, tile, range, &target) {
            continue;
        }
        let Some(spawn) = game.can_build(pid, nuke_type, tile) else {
            continue;
        };
        if game.cfg_team_mode()
            && game.difficulty != Difficulty::Easy
            && is_teammate_nuking_spot(game, pid, tile, nuke_type)
        {
            continue;
        }
        if matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible)
            && is_trajectory_interceptable(game, pid, spawn, tile, None)
        {
            continue;
        }
        let value = nuke_tile_score(game, pid, tile, &silo_tiles, &structures, nuke_type);
        if value > best_value {
            best_tile = Some(tile);
            best_value = value;
        }
    }
    if let Some(tile) = best_tile {
        if best_value > 0.0 || game.difficulty != Difficulty::Impossible {
            send_nuke(game, pid, tile, nuke_type, 0);
            return;
        }
    }
    if game.difficulty == Difficulty::Impossible {
        maybe_destroy_enemy_sam(game, pid, &target);
    }
}

fn first_by_tiles(game: &Game) -> Option<String> {
    let mut v: Vec<(u32, String)> = game
        .all_players()
        .filter(|p| p.is_alive())
        .map(|p| (p.num_tiles(), p.id().to_string()))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().next().map(|(_, id)| id)
}

fn sorted_by_tiles(game: &Game) -> Vec<String> {
    let mut v: Vec<(u32, String)> = game
        .all_players()
        .filter(|p| p.is_alive())
        .map(|p| (p.num_tiles(), p.id().to_string()))
        .collect();
    v.sort_by(|a, b| b.0.cmp(&a.0));
    v.into_iter().map(|(_, id)| id).collect()
}

fn find_ffa_crown_target(game: &Game, pid: &str) -> Option<String> {
    if game.cfg_team_mode() {
        return None;
    }
    let sorted = sorted_by_tiles(game);
    if sorted.len() <= 1 {
        return None;
    }
    let first = &sorted[0];
    if game.difficulty == Difficulty::Impossible && first == pid && sorted.len() >= 2 {
        let second = &sorted[1];
        if !game.is_allied(pid, second) {
            return Some(second.clone());
        }
    }
    if first == pid || game.is_allied(pid, first) {
        return None;
    }
    let land = game
        .map
        .num_land_tiles()
        .saturating_sub(game.map.num_tiles_with_fallout());
    if land == 0 {
        return None;
    }
    let first_share = game.player(first).map(|p| p.num_tiles()).unwrap_or(0) as f64 / land as f64;
    let my_share = game.player(pid).map(|p| p.num_tiles()).unwrap_or(0) as f64 / land as f64;
    let thresh = match game.difficulty {
        Difficulty::Easy => 0.4,
        Difficulty::Medium => 0.3,
        Difficulty::Hard => 0.2,
        Difficulty::Impossible => 0.1,
    };
    if first_share - my_share > thresh {
        Some(first.clone())
    } else {
        None
    }
}

fn find_strongest_team_target(game: &mut Game, pid: &str, idx: usize) -> Option<String> {
    if !game.cfg_team_mode() {
        return None;
    }
    let _ = (pid, idx);
    None
}

fn nation_should_attack_nuke_target(game: &mut Game, pid: &str, target: &str, idx: usize) -> bool {
    let other_human = game
        .player(target)
        .map(|p| p.player_type() == PlayerType::Human)
        .unwrap_or(false);
    let traitor = game.player(target).map(|p| p.traitor).unwrap_or(false);
    let me_bot = game
        .player(pid)
        .map(|p| p.player_type() == PlayerType::Bot)
        .unwrap_or(false);
    if !other_human || traitor || me_bot {
        return true;
    }
    match game.difficulty {
        Difficulty::Easy => game
            .nation_rng(idx)
            .map(|r| r.next_int(0.0, 4.0) == 0)
            .unwrap_or(false),
        Difficulty::Medium => game.nation_rng(idx).map(|r| !r.chance(4)).unwrap_or(true),
        _ => true,
    }
}

fn ensure_nuke_costs(game: &mut Game, pid: &str) {
    let atom = game.structure_cost(pid, UnitType::AtomBomb);
    let hydro = game.structure_cost(pid, UnitType::HydrogenBomb);
    let st = game.later.nations.entry(pid.to_string()).or_default();
    if st.atom_perceived.is_none() {
        st.atom_perceived = Some(atom);
    }
    if st.hydro_perceived.is_none() {
        st.hydro_perceived = Some(hydro);
    }
}

fn perceived_nuke_cost(game: &Game, pid: &str, ty: UnitType) -> i64 {
    let actual = game.structure_cost(pid, ty);
    let alive = game.all_players().filter(|p| p.is_alive()).count();
    if alive == 2 {
        return actual;
    }
    if game.is_unit_disabled(UnitType::Mirv) {
        return actual;
    }
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    let hydro = game.structure_cost(pid, UnitType::HydrogenBomb);
    if game.cfg_team_mode() && gold > hydro {
        return actual;
    }
    if gold > game.structure_cost(pid, UnitType::Mirv) + hydro {
        return actual;
    }
    if matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible)
        && is_under_heavy_attack(game, pid)
    {
        return actual;
    }
    let st = game.later.nations.get(pid);
    match ty {
        UnitType::AtomBomb => st.and_then(|s| s.atom_perceived).unwrap_or(actual),
        _ => st.and_then(|s| s.hydro_perceived).unwrap_or(actual),
    }
}

fn is_under_heavy_attack(game: &Game, pid: &str) -> bool {
    let incoming = game.incoming_troops_on(pid);
    let mine = game.player(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
    incoming >= mine
}

fn remove_old_nuke_events(game: &mut Game, pid: &str) {
    let tick = game.ticks();
    if let Some(st) = game.later.nations.get_mut(pid) {
        while let Some(&(t, _, _)) = st.recent_nukes.first() {
            if t + NUKE_EVENT_MAX_AGE < tick {
                st.recent_nukes.remove(0);
            } else {
                break;
            }
        }
    }
}

fn try_ref(game: &Game, x: i32, y: i32) -> Option<TileRef> {
    if x < 0 || y < 0 {
        return None;
    }
    game.map.ref_xy(x as u32, y as u32).ok()
}

fn bounding_box_tiles(game: &Game, center: TileRef, radius: u32) -> Vec<TileRef> {
    let cx = game.map.x(center) as i32;
    let cy = game.map.y(center) as i32;
    let r = radius as i32;
    let min_x = cx - r;
    let max_x = cx + r;
    let min_y = cy - r;
    let max_y = cy + r;
    let mut tiles = Vec::new();
    for x in min_x..=max_x {
        if let Some(t) = try_ref(game, x, min_y) {
            tiles.push(t);
        }
        if min_y != max_y {
            if let Some(t) = try_ref(game, x, max_y) {
                tiles.push(t);
            }
        }
    }
    for y in (min_y + 1)..max_y {
        if let Some(t) = try_ref(game, min_x, y) {
            tiles.push(t);
        }
        if min_x != max_x {
            if let Some(t) = try_ref(game, max_x, y) {
                tiles.push(t);
            }
        }
    }
    tiles
}

fn nuke_bounding_box_valid(
    game: &Game,
    pid: &str,
    tile: TileRef,
    range: u32,
    target: &str,
) -> bool {
    let mut box_tiles = bounding_box_tiles(game, tile, range);
    box_tiles.extend(bounding_box_tiles(game, tile, range / 2));
    box_tiles
        .iter()
        .all(|&t| is_valid_nuke_tile(game, pid, t, target))
}

fn is_valid_nuke_tile(game: &Game, pid: &str, t: TileRef, target: &str) -> bool {
    let owner = game.map.owner_id(t);
    if let Some(tp) = game.player(target) {
        if owner == tp.small_id {
            return true;
        }
    }
    if matches!(game.difficulty, Difficulty::Hard | Difficulty::Impossible) {
        if owner == 0 {
            return true;
        }
        if game.cfg_team_mode() {
            if let Some(op) = game.player_by_small(owner) {
                if !game.is_allied(pid, op.id()) {
                    return true;
                }
            }
        }
    }
    false
}

fn eucl2(game: &Game, a: TileRef, b: TileRef) -> i64 {
    let dx = game.map.x(a) as i64 - game.map.x(b) as i64;
    let dy = game.map.y(a) as i64 - game.map.y(b) as i64;
    dx * dx + dy * dy
}

fn is_teammate_nuking_spot(game: &Game, pid: &str, tile: TileRef, nuke_type: UnitType) -> bool {
    let our_inner = nuke_magnitudes(nuke_type).0 as i64;
    for u in game.units() {
        if !u.active || !matches!(u.ty, UnitType::AtomBomb | UnitType::HydrogenBomb) {
            continue;
        }
        if u.owner_id == pid || !game.is_allied(pid, &u.owner_id) {
            continue;
        }
        let Some(dst) = u.target_tile else {
            continue;
        };
        let other_inner = nuke_magnitudes(u.ty).0 as i64;
        let sum = our_inner + other_inner;
        if eucl2(game, tile, dst) <= sum * sum {
            return true;
        }
    }
    false
}

fn is_trajectory_interceptable(
    game: &Game,
    pid: &str,
    spawn: TileRef,
    target: TileRef,
    excluded: Option<&HashSet<u32>>,
) -> bool {
    let path = ParabolaPath::new(&game.map, spawn, target, nuke_speed(UnitType::AtomBomb), true);
    let trajectory = path.all_tiles(&game.map);
    if trajectory.is_empty() {
        return false;
    }
    let range2 = (DEFAULT_NUKE_TARGETABLE_RANGE as i64) * (DEFAULT_NUKE_TARGETABLE_RANGE as i64);
    let mut un_start = None;
    let mut un_end = None;
    for (i, &tile) in trajectory.iter().enumerate() {
        if un_start.is_none() {
            if eucl2(game, tile, spawn) > range2 {
                if eucl2(game, tile, target) < range2 {
                    break;
                }
                un_start = Some(i);
            }
        } else if eucl2(game, tile, target) < range2 {
            un_end = Some(i);
            break;
        }
    }
    let mut i = 0;
    while i < trajectory.len() {
        if let (Some(s), Some(e)) = (un_start, un_end) {
            if i == s {
                i = e;
                continue;
            }
        }
        let tile = trajectory[i];
        let nearby = game.later_grid_nearby(tile, MAX_SAM_RANGE as u32, &[UnitType::SamLauncher]);
        for (id, d2) in nearby {
            let Some(sam) = game.units().iter().find(|u| u.id == id) else {
                continue;
            };
            if sam.owner_id == pid || game.is_allied(pid, &sam.owner_id) {
                continue;
            }
            if excluded.map(|ex| ex.contains(&id)).unwrap_or(false) {
                continue;
            }
            let r = sam_range(sam.level);
            if (d2 as f64) <= r * r {
                return true;
            }
        }
        i += 1;
    }
    false
}

fn closest_two_tiles(game: &Game, xs: &[TileRef], ys: &[TileRef]) -> Option<(TileRef, TileRef)> {
    let mut x_sorted = xs.to_vec();
    let mut y_sorted = ys.to_vec();
    x_sorted.sort_by_key(|&t| game.map.x(t));
    y_sorted.sort_by_key(|&t| game.map.x(t));
    if x_sorted.is_empty() || y_sorted.is_empty() {
        return None;
    }
    let mut i = 0;
    let mut j = 0;
    let mut min_d = u32::MAX;
    let mut result = (x_sorted[0], y_sorted[0]);
    while i < x_sorted.len() && j < y_sorted.len() {
        let cx = x_sorted[i];
        let cy = y_sorted[j];
        let d = game.map.manhattan(cx, cy);
        if d < min_d {
            min_d = d;
            result = (cx, cy);
        }
        if i + 1 == x_sorted.len() {
            j += 1;
        } else if j + 1 == y_sorted.len() {
            i += 1;
        } else if game.map.x(cx) < game.map.x(cy) {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(result)
}

fn nuke_tile_score(
    game: &Game,
    pid: &str,
    tile: TileRef,
    silos: &[TileRef],
    targets: &[(UnitType, u32, TileRef)],
    nuke_type: UnitType,
) -> f64 {
    let outer = nuke_magnitudes(nuke_type).1 as i64;
    let outer2 = outer * outer;
    let mut tile_value = 0.0;
    for &(ty, level, t) in targets {
        if eucl2(game, tile, t) > outer2 {
            continue;
        }
        let lv = level as f64;
        tile_value += match ty {
            UnitType::City => 25_000.0 * lv,
            UnitType::DefensePost => 5_000.0 * lv,
            UnitType::MissileSilo => 50_000.0 * lv,
            UnitType::Port | UnitType::Factory => 15_000.0 * lv,
            _ => 0.0,
        };
    }
    if game.difficulty == Difficulty::Medium {
        let r2 = 50i64 * 50;
        if targets
            .iter()
            .any(|&(ty, _, t)| ty == UnitType::SamLauncher && eucl2(game, tile, t) <= r2)
        {
            return -1.0;
        }
    }
    if game.difficulty == Difficulty::Impossible && nuke_type == UnitType::HydrogenBomb {
        let hydro_outer = nuke_magnitudes(UnitType::HydrogenBomb).1;
        let nearby = game.later_grid_nearby(tile, hydro_outer, &[UnitType::SamLauncher]);
        for (id, d2) in nearby {
            let Some(sam) = game.units().iter().find(|u| u.id == id) else {
                continue;
            };
            if sam.level >= 5 {
                continue;
            }
            let sr = sam_range(sam.level);
            let dist = (d2 as f64).sqrt();
            if dist > sr {
                tile_value += 100_000.0 * sam.level as f64;
            }
        }
    }
    let Some((closest, _)) = closest_two_tiles(game, silos, &[tile]) else {
        return tile_value;
    };
    let dist = (eucl2(game, tile, closest) as f64).sqrt();
    let penalty = dist * 30.0;
    tile_value = (tile_value * 0.2).max(tile_value - penalty);
    if let Some(st) = game.later.nations.get(pid) {
        for &(_, recent, rty) in &st.recent_nukes {
            let inner = nuke_magnitudes(rty).0 as i64;
            if eucl2(game, tile, recent) <= inner * inner {
                tile_value -= 1_000_000.0;
            }
        }
    }
    tile_value
}

fn send_nuke(game: &mut Game, pid: &str, tile: TileRef, nuke_type: UnitType, wait: u32) {
    let tick = game.ticks();
    {
        let st = game.later.nations.entry(pid.to_string()).or_default();
        st.recent_nukes.push((tick, tile, nuke_type));
        match nuke_type {
            UnitType::AtomBomb => {
                if let Some(c) = st.atom_perceived.as_mut() {
                    *c = (*c * 150) / 100;
                }
            }
            UnitType::HydrogenBomb => {
                if let Some(c) = st.hydro_perceived.as_mut() {
                    *c = (*c * 125) / 100;
                }
            }
            _ => {}
        }
    }
    game.launch_nuke_wait(pid.to_string(), nuke_type, tile, true, wait);
}

fn maybe_destroy_enemy_sam(game: &mut Game, pid: &str, target: &str) {
    if game.is_unit_disabled(UnitType::AtomBomb) {
        return;
    }
    if game
        .units()
        .iter()
        .any(|u| u.active && u.ty == UnitType::AtomBomb && u.owner_id == pid)
    {
        return;
    }
    let atom_cost = game.structure_cost(pid, UnitType::AtomBomb);
    let mut enemy_sams: Vec<(u32, u32, TileRef)> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::SamLauncher && u.owner_id == target)
        .map(|u| (u.id, u.level, u.tile))
        .collect();
    if enemy_sams.is_empty() {
        return;
    }
    enemy_sams.sort_by_key(|(_, level, _)| *level);
    let our_silos: Vec<(u32, u32, TileRef, usize)> = game
        .units()
        .iter()
        .filter(|u| {
            u.active && u.ty == UnitType::MissileSilo && u.owner_id == pid && !u.under_construction
        })
        .map(|u| (u.id, u.level, u.tile, u.missile_queue.len()))
        .collect();
    if our_silos.is_empty() {
        return;
    }
    let mut needs_more = false;
    let mut failed: Option<(TileRef, HashSet<u32>, usize)> = None;
    for &(_sid, _slevel, target_tile) in &enemy_sams {
        let covering = find_enemy_sams_covering(game, pid, target_tile);
        let covering_ids: HashSet<u32> = covering.iter().map(|(id, _, _)| *id).collect();
        let total_int: u32 = covering.iter().map(|(_, lv, _)| *lv).sum();
        let bombs_needed = total_int as usize + 1;
        let mut available: Vec<(TileRef, u32, usize, bool)> = Vec::new();
        for &(_, level, stile, qlen) in &our_silos {
            let slots = level.saturating_sub(qlen as u32);
            if slots == 0 {
                continue;
            }
            let interceptable =
                is_trajectory_interceptable(game, pid, stile, target_tile, Some(&covering_ids));
            let path =
                ParabolaPath::new(&game.map, stile, target_tile, nuke_speed(UnitType::AtomBomb), true);
            let flight = path.all_tiles(&game.map).len();
            if flight == 0 {
                continue;
            }
            available.push((stile, slots, flight, interceptable));
        }
        available.sort_by_key(|(stile, _, _, _)| game.map.manhattan(*stile, target_tile));
        let mut launch_seq: Vec<(usize, bool)> = Vec::new();
        for &(_, slots, flight, interceptable) in &available {
            for _ in 0..slots {
                launch_seq.push((flight, interceptable));
            }
        }
        let max_spread = (SAM_COOLDOWN / 2) as usize;
        let extra = bombs_needed / 5;
        let total_bombs = bombs_needed + extra;
        let mut unblocked: Vec<(usize, usize)> = Vec::new();
        for (i, &(flight, interceptable)) in launch_seq.iter().enumerate() {
            if !interceptable {
                unblocked.push((i, flight));
            }
        }
        if unblocked.len() < total_bombs {
            if failed.is_none() {
                failed = Some((target_tile, covering_ids, total_bombs));
            }
            needs_more = true;
            continue;
        }
        let mut by_flight = unblocked.clone();
        by_flight.sort_by_key(|(_, f)| *f);
        let mut best_start = 0;
        let mut best_count = 0;
        for start in 0..by_flight.len() {
            let mut end = start;
            while end < by_flight.len() && by_flight[end].1 - by_flight[start].1 <= max_spread {
                end += 1;
            }
            if end - start > best_count {
                best_count = end - start;
                best_start = start;
            }
        }
        if best_count < total_bombs {
            if failed.is_none() {
                failed = Some((target_tile, covering_ids, total_bombs));
            }
            needs_more = true;
            continue;
        }
        let mut window: Vec<(usize, usize)> = by_flight[best_start..best_start + best_count].to_vec();
        window.sort_by_key(|(i, _)| *i);
        let selected = &window[..total_bombs];
        let selected_set: HashSet<usize> = selected.iter().map(|(i, _)| *i).collect();
        let last_idx = selected[selected.len() - 1].0;
        let bombs_to_fire = last_idx + 1;
        let selected_min = selected.iter().map(|(_, f)| *f).min().unwrap_or(0);
        let stagger = 1.max(max_spread / total_bombs);
        let mut selected_i = 0;
        let mut waits = Vec::new();
        for i in 0..bombs_to_fire {
            if selected_set.contains(&i) {
                let target_arr = selected_min + selected_i * stagger;
                let flight = launch_seq[i].0;
                waits.push(target_arr.saturating_sub(flight) as u32);
                selected_i += 1;
            } else {
                waits.push(0);
            }
        }
        let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
        if gold < atom_cost * bombs_to_fire as i64 {
            continue;
        }
        for wait in waits {
            send_nuke(game, pid, target_tile, UnitType::AtomBomb, wait);
        }
        return;
    }
    if needs_more {
        if let Some((tile, ids, total)) = failed {
            maybe_upgrade_helpful_silo(game, pid, tile, &ids, total);
        }
    }
}

fn find_enemy_sams_covering(game: &Game, pid: &str, tile: TileRef) -> Vec<(u32, u32, TileRef)> {
    let nearby = game.later_grid_nearby(tile, MAX_SAM_RANGE as u32, &[UnitType::SamLauncher]);
    let mut out = Vec::new();
    for (id, d2) in nearby {
        let Some(sam) = game.units().iter().find(|u| u.id == id) else {
            continue;
        };
        if sam.owner_id == pid || game.is_allied(pid, &sam.owner_id) {
            continue;
        }
        let r = sam_range(sam.level);
        if (d2 as f64) <= r * r {
            out.push((id, sam.level, sam.tile));
        }
    }
    out
}

fn maybe_upgrade_helpful_silo(
    game: &mut Game,
    pid: &str,
    target_tile: TileRef,
    covering: &HashSet<u32>,
    total_bombs: usize,
) {
    let silos: Vec<(u32, u32, TileRef)> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::MissileSilo && u.owner_id == pid)
        .map(|u| (u.id, u.level, u.tile))
        .collect();
    if silos.is_empty() {
        return;
    }
    let mut unblocked = Vec::new();
    for &(id, level, tile) in &silos {
        if !is_trajectory_interceptable(game, pid, tile, target_tile, Some(covering)) {
            unblocked.push((id, level, tile));
        }
    }
    if unblocked.is_empty() {
        return;
    }
    if unblocked.len() * MAX_NATION_SILO_UPGRADE_LEVEL as usize >= total_bombs {
        // reachable at max level
    } else {
        return;
    }
    let our_sams: Vec<(u32, TileRef)> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::SamLauncher && u.owner_id == pid)
        .map(|u| (u.level, u.tile))
        .collect();
    let gold = game.player(pid).map(|p| p.gold).unwrap_or(0);
    let cost = game.structure_cost(pid, UnitType::MissileSilo);
    let mut best: Option<(u32, u32)> = None;
    for &(id, level, stile) in &unblocked {
        if level >= MAX_NATION_SILO_UPGRADE_LEVEL {
            continue;
        }
        if game.is_unit_disabled(UnitType::MissileSilo) || gold < cost {
            continue;
        }
        let mut protection = 0u32;
        for &(sl, stile_sam) in &our_sams {
            let r = sam_range(sl);
            if (eucl2(game, stile, stile_sam) as f64) <= r * r {
                protection += sl;
            }
        }
        if best.map(|(_, p)| protection > p).unwrap_or(true) {
            best = Some((id, protection));
        }
    }
    if let Some((id, _)) = best {
        game.enqueue_later(LaterExec::Upgrade {
            active: true,
            owner: pid.to_string(),
            unit_id: id,
            inited: false,
        });
    }
}

pub fn track_ships_and_retaliate(game: &mut Game, pid: &str, idx: usize) {
    track_own_transports(game, pid, idx);
    track_own_trade(game, pid, idx);
    track_incoming_transports(game, pid, idx);
}

fn track_own_transports(game: &mut Game, pid: &str, idx: usize) {
    if game.is_unit_disabled(UnitType::TransportShip) {
        return;
    }
    let live: Vec<u32> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::TransportShip && u.owner_id == pid)
        .map(|u| u.id)
        .collect();
    {
        let st = game.later.nations.entry(pid.to_string()).or_default();
        for id in live {
            st.tracked_transports.insert(id);
        }
    }
    let tracked: Vec<u32> = game
        .later
        .nations
        .get(pid)
        .map(|s| s.tracked_transports.iter().copied().collect())
        .unwrap_or_default();
    for id in tracked {
        let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_transports.shift_remove(&id);
            }
            continue;
        };
        if ship.active {
            continue;
        }
        if let Some(destroyer) = ship.destroyer_id.clone() {
            maybe_retaliate_warship(game, pid, idx, ship.tile, &destroyer, true);
        }
        if let Some(st) = game.later.nations.get_mut(pid) {
            st.tracked_transports.shift_remove(&id);
        }
    }
}

fn track_own_trade(game: &mut Game, pid: &str, idx: usize) {
    let live: Vec<u32> = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::TradeShip && u.owner_id == pid)
        .map(|u| u.id)
        .collect();
    {
        let st = game.later.nations.entry(pid.to_string()).or_default();
        for id in live {
            st.tracked_trade.insert(id);
        }
    }
    let tracked: Vec<u32> = game
        .later
        .nations
        .get(pid)
        .map(|s| s.tracked_trade.iter().copied().collect())
        .unwrap_or_default();
    for id in tracked {
        let Some(ship) = game.units().iter().find(|u| u.id == id).cloned() else {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_trade.shift_remove(&id);
            }
            continue;
        };
        if !ship.active {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_trade.shift_remove(&id);
            }
            continue;
        }
        if ship.owner_id != pid {
            maybe_retaliate_warship(game, pid, idx, ship.tile, &ship.owner_id, false);
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_trade.shift_remove(&id);
            }
        }
    }
}

fn track_incoming_transports(game: &mut Game, pid: &str, idx: usize) {
    let my_small = game.player(pid).map(|p| p.small_id).unwrap_or(0);
    let incoming: Vec<u32> = game
        .units()
        .iter()
        .filter(|u| {
            if !u.active || u.ty != UnitType::TransportShip || u.retreating {
                return false;
            }
            let Some(target) = u.target_tile else {
                return false;
            };
            game.map.owner_id(target) == my_small && u.owner_id != pid
        })
        .map(|u| u.id)
        .collect();
    {
        let st = game.later.nations.entry(pid.to_string()).or_default();
        for id in incoming {
            st.tracked_incoming.insert(id);
        }
    }
    let tracked: Vec<u32> = game
        .later
        .nations
        .get(pid)
        .map(|s| s.tracked_incoming.iter().copied().collect())
        .unwrap_or_default();
    for id in tracked {
        let Some(transport) = game.units().iter().find(|u| u.id == id).cloned() else {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_incoming.shift_remove(&id);
                st.dealt_incoming.shift_remove(&id);
            }
            continue;
        };
        let target = transport.target_tile;
        if !transport.active || target.is_none() || transport.retreating {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.tracked_incoming.shift_remove(&id);
                st.dealt_incoming.shift_remove(&id);
            }
            continue;
        }
        if game
            .later
            .nations
            .get(pid)
            .map(|s| s.dealt_incoming.contains(&id))
            .unwrap_or(false)
        {
            continue;
        }
        let target = target.unwrap();
        let dist = game.map.manhattan(transport.tile, target);
        if dist < 20 {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.dealt_incoming.insert(id);
            }
            continue;
        }
        if game.is_allied(pid, &transport.owner_id) {
            continue;
        }
        if warship_covers_tile(game, pid, target) {
            if let Some(st) = game.later.nations.get_mut(pid) {
                st.dealt_incoming.insert(id);
            }
            continue;
        }
        let Some(ocean) = warship_spawn_tile(game, idx, target, 30) else {
            continue;
        };
        maybe_retaliate_warship(game, pid, idx, ocean, &transport.owner_id, true);
        if let Some(st) = game.later.nations.get_mut(pid) {
            st.dealt_incoming.insert(id);
        }
        break;
    }
}

fn warship_covers_tile(game: &Game, pid: &str, target: TileRef) -> bool {
    for u in game.units() {
        if !u.active || u.ty != UnitType::Warship || u.owner_id != pid {
            continue;
        }
        let dx = game.map.x(u.tile) as i64 - game.map.x(target) as i64;
        let dy = game.map.y(u.tile) as i64 - game.map.y(target) as i64;
        if dx * dx + dy * dy <= 90 * 90 {
            return true;
        }
        if let Some(patrol) = u.patrol_tile {
            if game.map.manhattan(target, patrol) < 90 {
                return true;
            }
        }
    }
    false
}

fn maybe_retaliate_warship(
    game: &mut Game,
    pid: &str,
    idx: usize,
    tile: TileRef,
    enemy: &str,
    transport: bool,
) {
    if enemy == pid {
        return;
    }
    let ships = game
        .units()
        .iter()
        .filter(|u| u.active && u.ty == UnitType::Warship && u.owner_id == pid)
        .count();
    if ships >= 10 {
        maybe_move_warship(game, pid, tile);
        return;
    }
    let roll_ok = match game.difficulty {
        Difficulty::Easy => false,
        Difficulty::Medium => game
            .nation_rng(idx)
            .map(|r| r.next_int(0.0, 100.0) < 15)
            .unwrap_or(false),
        Difficulty::Hard => game
            .nation_rng(idx)
            .map(|r| r.next_int(0.0, 100.0) < 50)
            .unwrap_or(false),
        Difficulty::Impossible => game
            .nation_rng(idx)
            .map(|r| r.next_int(0.0, 100.0) < 80)
            .unwrap_or(false),
    };
    if !roll_ok {
        return;
    }
    if game.can_build(pid, UnitType::Warship, tile).is_none() {
        maybe_move_warship(game, pid, tile);
        return;
    }
    game.uninit_construction(pid, UnitType::Warship, tile);
    if let Some(p) = game.player_mut(pid) {
        p.update_relation(enemy, if transport { -15.0 } else { -7.5 });
    }
}

fn maybe_move_warship(game: &mut Game, pid: &str, tile: TileRef) {
    if !game.map.is_water(tile) {
        return;
    }
    let mut best: Option<(u32, u32)> = None;
    for u in game.units() {
        if !u.active || u.ty != UnitType::Warship || u.owner_id != pid {
            continue;
        }
        let Some(patrol) = u.patrol_tile else {
            continue;
        };
        if game.map.manhattan(u.tile, patrol) >= 130 {
            continue;
        }
        let d = game.map.manhattan(u.tile, tile);
        if best.map(|(bd, _)| d < bd).unwrap_or(true) {
            best = Some((d, u.id));
        }
    }
    if let Some((_, id)) = best {
        if let Some(u) = game.units_mut().iter_mut().find(|u| u.id == id) {
            u.patrol_tile = Some(tile);
        }
    }
}

impl Game {
    pub(crate) fn uninit_construction(&mut self, pid: &str, ty: UnitType, tile: TileRef) {
        self.uninit.push(crate::game::Exec::Construction(Self::construction_exec(
            pid, ty, tile,
        )));
    }
}
