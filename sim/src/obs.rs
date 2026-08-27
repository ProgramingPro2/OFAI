//! Fixed-size schema-v2 observations and early-stage masks.

use std::collections::HashSet;

use crate::game::{find_spawn_tile, Game};
use crate::map::TileRef;
use crate::spec::{
    Difficulty, PlayerType, ACTION_ALLY, ACTION_ATTACK, ACTION_BOAT, ACTION_BREAK_ALLY,
    ACTION_BUILD, ACTION_EMBARGO, ACTION_NOOP, ACTION_RETREAT_ALL, ACTION_SPAWN, CH_ALLY,
    CH_DEFENSE, CH_ENEMY, CH_FALLOUT, CH_MINE, CH_MOUNTAIN, CH_MY_BORDER, CH_MY_STRUCTURES,
    CH_ENEMY_STRUCTURES, GLOBAL_FEATURES, NUM_ACTION_TYPES, NUM_PLAYER_SLOTS, NUM_QUANTITIES,
    NUM_REGIONS, NUM_UNIT_TYPES, PLAYER_FEATURES, REGION_GRID, SPATIAL_CHANNELS, SPATIAL_SIZE,
    TARGET_MASKS_SIZE, TROOP_FRACTIONS,
};
use crate::units::{UnitType, UNIT_HEAD_ORDER};

/// JS `Float32Array` stores mathematical 0 as +0, never -0. Digests hash bytes.
fn obs_f32(x: f64) -> f32 {
    let y = x as f32;
    if y == 0.0 {
        0.0
    } else {
        y
    }
}

const PLANE: usize = SPATIAL_SIZE * SPATIAL_SIZE;

#[derive(Clone)]
pub struct ObsBuffers {
    pub spatial: Vec<f32>,
    pub players: Vec<f32>,
    pub global: Vec<f32>,
    pub action_mask: Vec<u8>,
    pub target_masks: Vec<u8>,
    pub quantity_mask: Vec<u8>,
    pub unit_mask: Vec<u8>,
    pub spawn_regions: Vec<u8>,
    pub build_regions: Vec<u8>,
    pub boat_regions: Vec<u8>,
}

impl ObsBuffers {
    pub fn new() -> Self {
        Self {
            spatial: vec![0.0; SPATIAL_CHANNELS * PLANE],
            players: vec![0.0; NUM_PLAYER_SLOTS * PLAYER_FEATURES],
            global: vec![0.0; GLOBAL_FEATURES],
            action_mask: vec![0; NUM_ACTION_TYPES],
            target_masks: vec![0; TARGET_MASKS_SIZE],
            quantity_mask: vec![0; NUM_QUANTITIES],
            unit_mask: vec![0; NUM_UNIT_TYPES],
            spawn_regions: vec![0; NUM_REGIONS],
            build_regions: vec![0; NUM_REGIONS],
            boat_regions: vec![0; NUM_REGIONS],
        }
    }
}

#[derive(Clone)]
pub struct SlotInfo {
    pub player_id: Option<String>,
    pub shares_border: bool,
    pub can_attack: bool,
    pub ally_signal: bool,
}

pub struct ObsExtractor {
    land_bins: Vec<f32>,
    mountain_bins: Vec<f32>,
    counts: Vec<f32>,
    mine: Vec<f32>,
    enemy: Vec<f32>,
    ally: Vec<f32>,
    my_border: Vec<f32>,
    fallout: Vec<f32>,
    defense: Vec<f32>,
    region_my: Vec<i32>,
    region_unowned: Vec<i32>,
    region_enemy_shore: Vec<i32>,
    wild_adj: Vec<u8>,
    wild_count: i32,
    map_w: u32,
    map_h: u32,
    my_small: u16,
    ally_sids: HashSet<u16>,
    // Full-res tile-state mirror. Bit 15 is the extractor-only
    // "my-border already counted" sentinel (official ObsExtractor).
    mirror: Vec<u16>,
}

impl ObsExtractor {
    pub fn new() -> Self {
        Self {
            land_bins: vec![0.0; PLANE],
            mountain_bins: vec![0.0; PLANE],
            counts: vec![0.0; PLANE],
            mine: vec![0.0; PLANE],
            enemy: vec![0.0; PLANE],
            ally: vec![0.0; PLANE],
            my_border: vec![0.0; PLANE],
            fallout: vec![0.0; PLANE],
            defense: vec![0.0; PLANE],
            region_my: vec![0; NUM_REGIONS],
            region_unowned: vec![0; NUM_REGIONS],
            region_enemy_shore: vec![0; NUM_REGIONS],
            wild_adj: Vec::new(),
            wild_count: 0,
            map_w: 0,
            map_h: 0,
            my_small: 0,
            ally_sids: HashSet::new(),
            mirror: Vec::new(),
        }
    }

    pub fn has_adjacent_wilderness(&self) -> bool {
        self.wild_count > 0
    }

    fn current_ally_sids(game: &Game, me_id: &str) -> HashSet<u16> {
        game.alliances()
            .iter()
            .filter(|a| a.requestor == me_id || a.recipient == me_id)
            .filter_map(|a| game.player(a.other(me_id)).map(|p| p.small_id))
            .collect()
    }

    pub fn build_static(&mut self, game: &Game, me_id: &str) {
        let map = &game.map;
        self.map_w = map.width();
        self.map_h = map.height();
        self.my_small = game.player(me_id).map(|p| p.small_id).unwrap_or(0);
        self.ally_sids = Self::current_ally_sids(game, me_id);
        self.land_bins.fill(0.0);
        self.mountain_bins.fill(0.0);
        self.counts.fill(0.0);
        self.mine.fill(0.0);
        self.enemy.fill(0.0);
        self.ally.fill(0.0);
        self.my_border.fill(0.0);
        self.fallout.fill(0.0);
        self.defense.fill(0.0);
        self.region_my.fill(0);
        self.region_unowned.fill(0);
        self.region_enemy_shore.fill(0);
        let n = map.len();
        self.mirror = map.tile_state_buffer().to_vec();
        self.wild_adj = vec![0; n];
        self.wild_count = 0;
        for r in 0..n as u32 {
            let bin = self.bin_of(r);
            self.counts[bin] += 1.0;
            let land = map.is_land(r);
            if land {
                self.land_bins[bin] += 1.0;
                match map.terrain_type(r) {
                    crate::spec::TerrainKind::Highland | crate::spec::TerrainKind::Mountain => {
                        self.mountain_bins[bin] += 1.0;
                    }
                    _ => {}
                }
            }
            self.accumulate_initial(game, r, self.mirror[r as usize], land);
        }
        for i in 0..PLANE {
            if self.counts[i] > 0.0 {
                self.land_bins[i] /= self.counts[i];
                self.mountain_bins[i] /= self.counts[i];
            }
        }
        for r in 0..n as u32 {
            self.refresh_wild(game, r);
        }
    }

    fn accumulate_initial(&mut self, game: &Game, r: TileRef, s: u16, land: bool) {
        let bin = self.bin_of(r);
        let owner = s & 0x0fff;
        let region = self.region_of(r);
        if owner != 0 {
            if owner == self.my_small {
                self.mine[bin] += 1.0;
                self.region_my[region] += 1;
                if game.map.is_border(r) {
                    self.my_border[bin] += 1.0;
                    self.mirror[r as usize] |= 0x8000;
                }
            } else if self.ally_sids.contains(&owner) {
                self.ally[bin] += 1.0;
            } else {
                self.enemy[bin] += 1.0;
            }
        } else if land {
            self.region_unowned[region] += 1;
        }
        if s & 0x2000 != 0 {
            self.fallout[bin] += 1.0;
        }
        if s & 0x4000 != 0 {
            self.defense[bin] += 1.0;
        }
        if owner != self.my_small && game.map.is_shoreline(r) {
            self.region_enemy_shore[region] += 1;
        }
    }

    /// Official `ObsExtractor.applyTileUpdates`, including the bit-15 wipe
    /// that leaves conquered my-border tiles counted.
    pub fn apply_tile_updates(&mut self, game: &Game, packed: &[u32]) {
        if self.mirror.is_empty() || packed.is_empty() {
            return;
        }
        let mut i = 0;
        while i + 1 < packed.len() {
            let r = packed[i];
            let s = (packed[i + 1] & 0xffff) as u16;
            let old = self.mirror.get(r as usize).copied().unwrap_or(0);
            self.transition(game, r, old, s);
            if (r as usize) < self.mirror.len() {
                self.mirror[r as usize] = s;
            }
            let mut nbuf = [0u32; 4];
            let n = game.map.neighbors4(r, &mut nbuf);
            let saved = [nbuf[0], nbuf[1], nbuf[2], nbuf[3]];
            for j in 0..n {
                self.refresh_border(&game.map, saved[j]);
            }
            self.refresh_border(&game.map, r);
            self.refresh_wild(game, r);
            for j in 0..n {
                self.refresh_wild(game, saved[j]);
            }
            i += 2;
        }
    }

    fn transition(&mut self, game: &Game, r: TileRef, old_s: u16, new_s: u16) {
        if old_s == new_s {
            return;
        }
        let bin = self.bin_of(r);
        let region = self.region_of(r);
        let land = game.map.is_land(r);

        let old_owner = old_s & 0x0fff;
        if old_owner != 0 {
            if old_owner == self.my_small {
                self.mine[bin] -= 1.0;
                self.region_my[region] -= 1;
            } else if self.ally_sids.contains(&old_owner) {
                self.ally[bin] -= 1.0;
            } else {
                self.enemy[bin] -= 1.0;
            }
        } else if land {
            self.region_unowned[region] -= 1;
        }
        if old_s & 0x2000 != 0 {
            self.fallout[bin] -= 1.0;
        }
        if old_s & 0x4000 != 0 {
            self.defense[bin] -= 1.0;
        }
        let old_shore = old_owner != self.my_small && land && game.map.is_shoreline(r);
        if old_shore {
            self.region_enemy_shore[region] -= 1;
        }

        let new_owner = new_s & 0x0fff;
        if new_owner != 0 {
            if new_owner == self.my_small {
                self.mine[bin] += 1.0;
                self.region_my[region] += 1;
            } else if self.ally_sids.contains(&new_owner) {
                self.ally[bin] += 1.0;
            } else {
                self.enemy[bin] += 1.0;
            }
        } else if land {
            self.region_unowned[region] += 1;
        }
        if new_s & 0x2000 != 0 {
            self.fallout[bin] += 1.0;
        }
        if new_s & 0x4000 != 0 {
            self.defense[bin] += 1.0;
        }
        let new_shore = new_owner != self.my_small && land && game.map.is_shoreline(r);
        if new_shore {
            self.region_enemy_shore[region] += 1;
        }
    }

    fn refresh_border(&mut self, map: &crate::map::GameMap, r: TileRef) {
        let Some(s) = self.mirror.get(r as usize).copied() else {
            return;
        };
        let owner = s & 0x0fff;
        let bin = self.bin_of(r);
        let now_border = owner == self.my_small && owner != 0 && map.is_border(r);
        let counted = s & 0x8000 != 0;
        if now_border && !counted {
            self.mirror[r as usize] = s | 0x8000;
            self.my_border[bin] += 1.0;
        } else if !now_border && counted {
            self.mirror[r as usize] = s & !0x8000;
            self.my_border[bin] -= 1.0;
        }
    }

    fn refresh_wild(&mut self, game: &Game, r: TileRef) {
        let now = self.compute_wild(game, r);
        let was = self.wild_adj.get(r as usize).copied().unwrap_or(0) != 0;
        if now == was {
            return;
        }
        if (r as usize) < self.wild_adj.len() {
            self.wild_adj[r as usize] = if now { 1 } else { 0 };
        }
        self.wild_count += if now { 1 } else { -1 };
    }

    fn compute_wild(&self, game: &Game, r: TileRef) -> bool {
        if game.map.owner_id(r) != self.my_small {
            return false;
        }
        let mut nbuf = [0u32; 4];
        let n = game.map.neighbors4(r, &mut nbuf);
        for i in 0..n {
            let t = nbuf[i];
            if game.map.is_land(t) && !game.map.is_impassable(t) && !game.map.has_owner(t) {
                return true;
            }
        }
        false
    }

    fn bin_of(&self, r: TileRef) -> usize {
        let x = (r % self.map_w) as usize;
        let y = (r / self.map_w) as usize;
        let bx = ((x * SPATIAL_SIZE) / self.map_w as usize).min(SPATIAL_SIZE - 1);
        let by = ((y * SPATIAL_SIZE) / self.map_h as usize).min(SPATIAL_SIZE - 1);
        by * SPATIAL_SIZE + bx
    }

    fn region_of(&self, r: TileRef) -> usize {
        let x = (r % self.map_w) as usize;
        let y = (r / self.map_w) as usize;
        let rx = ((x * REGION_GRID) / self.map_w as usize).min(REGION_GRID - 1);
        let ry = ((y * REGION_GRID) / self.map_h as usize).min(REGION_GRID - 1);
        ry * REGION_GRID + rx
    }

    pub fn extract(
        &mut self,
        game: &Game,
        me_id: &str,
        out: &mut ObsBuffers,
        max_ticks: u32,
        difficulty: Difficulty,
    ) -> Vec<SlotInfo> {
        let _ = difficulty;
        let Some(me) = game.player(me_id) else {
            return vec![SlotInfo {
                player_id: None,
                shares_border: false,
                can_attack: false,
                ally_signal: false,
            }; NUM_PLAYER_SLOTS];
        };
        // Alliance changes reclassify ally/enemy planes; rebuild like official.
        let allies_now = Self::current_ally_sids(game, me_id);
        if allies_now != self.ally_sids {
            self.build_static(game, me_id);
        }

        out.spatial[CH_MY_STRUCTURES * PLANE..(CH_ENEMY_STRUCTURES + 1) * PLANE].fill(0.0);
        out.spatial[..PLANE].copy_from_slice(&self.land_bins);
        out.spatial[CH_MOUNTAIN * PLANE..CH_MOUNTAIN * PLANE + PLANE]
            .copy_from_slice(&self.mountain_bins);
        for i in 0..PLANE {
            let c = if self.counts[i] > 0.0 { self.counts[i] } else { 1.0 };
            out.spatial[CH_MINE * PLANE + i] = self.mine[i] / c;
            out.spatial[CH_ENEMY * PLANE + i] = self.enemy[i] / c;
            out.spatial[CH_ALLY * PLANE + i] = self.ally[i] / c;
            out.spatial[CH_MY_BORDER * PLANE + i] = self.my_border[i] / c;
            out.spatial[CH_FALLOUT * PLANE + i] = self.fallout[i] / c;
            out.spatial[CH_DEFENSE * PLANE + i] = self.defense[i] / c;
        }
        let my_sid = me.small_id;
        for u in game.units() {
            // Official `game.units(...RASTER_UNIT_TYPES)` spreads into the
            // 3-arg overload, so only City / DefensePost / SAMLauncher are
            // painted. Port, Factory, silo, and naval types stay off the
            // raster so later goldens stay byte-identical.
            if !u.active
                || !matches!(
                    u.ty,
                    UnitType::City | UnitType::DefensePost | UnitType::SamLauncher
                )
            {
                continue;
            }
            let bin = self.bin_of(u.tile);
            let ch = if u.owner_small == my_sid {
                CH_MY_STRUCTURES
            } else {
                CH_ENEMY_STRUCTURES
            };
            let i = ch * PLANE + bin;
            out.spatial[i] = (out.spatial[i] + 0.34).min(1.0);
        }

        for r in 0..NUM_REGIONS {
            out.spawn_regions[r] = if self.region_unowned[r] > 0 { 1 } else { 0 };
            out.build_regions[r] = if self.region_my[r] > 0 { 1 } else { 0 };
            out.boat_regions[r] = if self.region_enemy_shore[r] > 0 { 1 } else { 0 };
        }

        let land = game.map.num_land_tiles().max(1) as f64;
        out.players.fill(0.0);
        write_self(&mut out.players, me, land);

        let mut metas: Vec<(i64, u16, String, bool, bool, bool)> = Vec::new();
        for p in game.alive_players() {
            if p.id() == me.id() {
                continue;
            }
            let shares = game.shares_border(me, p.small_id);
            let can = game.can_attack_player(me, p);
            let allied = game.is_allied(me.id(), p.id());
            let incoming_ally = game.incoming_from(me.id(), p.id());
            let incoming_troops = incoming_troops_from(game, me.id(), p.id());
            let ally_signal = incoming_ally || game.can_send_ally(me.id(), p.id());
            let mut pri = p.num_tiles() as i64;
            if shares || can {
                pri += 1_000_000;
            }
            if incoming_troops > 0.0 {
                pri += 500_000;
            }
            if allied {
                pri += 250_000;
            }
            if ally_signal {
                pri += 100_000;
            }
            metas.push((pri, p.small_id, p.id().to_string(), shares, can && shares, ally_signal));
        }
        metas.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
        let overflow = metas.len() > NUM_PLAYER_SLOTS - 1;
        let n = if overflow {
            NUM_PLAYER_SLOTS - 2
        } else {
            metas.len().min(NUM_PLAYER_SLOTS - 1)
        };
        let mut slots = vec![SlotInfo {
            player_id: Some(me.id().to_string()),
            shares_border: false,
            can_attack: false,
            ally_signal: false,
        }];
        for meta in metas.iter().take(n) {
            if let Some(p) = game.player(&meta.2) {
                write_other(&mut out.players, slots.len(), me, p, land, meta.3, game);
            }
            slots.push(SlotInfo {
                player_id: Some(meta.2.clone()),
                shares_border: meta.3,
                can_attack: meta.4,
                ally_signal: meta.5,
            });
        }
        while slots.len() < NUM_PLAYER_SLOTS {
            slots.push(SlotInfo {
                player_id: None,
                shares_border: false,
                can_attack: false,
                ally_signal: false,
            });
        }
        if overflow {
            write_overflow_aggregate(&mut out.players, me, land, game, &metas[n..]);
        }

        // Incoming attacks on me live on other players' outgoing lists.
        let mut my_incoming = 0.0;
        for p in game.all_players() {
            if p.id() == me.id() {
                continue;
            }
            for a in &p.outgoing {
                if a.target_player_id.as_deref() == Some(me.id()) && a.active {
                    my_incoming += a.troops;
                }
            }
        }
        // Official `game.players()` is alive-only, so g[6] is 0 or 1.
        let listed = game.alive_players();
        let alive_count = listed.len() as f64;
        let listed_len = listed.len().max(1) as f64;
        out.global[0] = obs_f32((game.ticks() as f64 / max_ticks.max(1) as f64).clamp(0.0, 1.0));
        out.global[1] = if game.in_spawn_phase() { 1.0 } else { 0.0 };
        out.global[2] = obs_f32(me.num_tiles() as f64 / land);
        out.global[3] = obs_f32(me.troops_f64().ln_1p() / 15.0);
        out.global[4] = obs_f32((me.gold as f64).ln_1p() / 20.0);
        out.global[5] = obs_f32(my_incoming.ln_1p() / 15.0);
        out.global[6] = obs_f32(alive_count / listed_len);
        out.global[7] = obs_f32(land.ln_1p() / 15.0);
        out.global[8] = if game.has_port(me.id()) { 1.0 } else { 0.0 };
        out.global[9] = if game.winner().is_some() { 1.0 } else { 0.0 };

        slots
    }
}

fn incoming_troops_from(game: &Game, me_id: &str, other_id: &str) -> f64 {
    game.player(other_id)
        .map(|p| {
            p.outgoing
                .iter()
                .filter(|a| a.active && a.target_player_id.as_deref() == Some(me_id))
                .map(|a| a.troops)
                .sum()
        })
        .unwrap_or(0.0)
}

fn write_overflow_aggregate(
    players: &mut [f32],
    me: &crate::player::Player,
    land: f64,
    game: &Game,
    rest: &[(i64, u16, String, bool, bool, bool)],
) {
    let mut tiles = 0.0;
    let mut troops = 0.0;
    let mut gold = 0.0;
    let mut incoming = 0.0;
    let mut alive = 0u32;
    let mut spawned = 0u32;
    let mut nation_tiles = 0.0;
    let mut bot_tiles = 0.0;
    let mut border = 0u32;
    let mut relation_sum = 0.0;
    for meta in rest {
        let Some(p) = game.player(&meta.2) else {
            continue;
        };
        let owned = p.num_tiles() as f64;
        tiles += owned;
        troops += p.troops_f64();
        gold += p.gold as f64;
        if p.is_alive() {
            alive += 1;
        }
        if p.has_spawned() {
            spawned += 1;
        }
        if p.player_type() == PlayerType::Nation {
            nation_tiles += owned;
        }
        if p.player_type() == PlayerType::Bot {
            bot_tiles += owned;
        }
        if meta.3 {
            border += 1;
        }
        relation_sum += me.relation(p.id()) as i32 as f64;
        incoming += incoming_troops_from(game, me.id(), p.id());
    }
    let _ = (nation_tiles, bot_tiles);
    let b = (NUM_PLAYER_SLOTS - 1) * PLAYER_FEATURES;
    players[b] = 1.0;
    players[b + 1] = 0.0;
    players[b + 2] = obs_f32(tiles / land);
    players[b + 3] = obs_f32(troops.ln_1p() / 15.0);
    players[b + 4] = obs_f32(gold.ln_1p() / 20.0);
    players[b + 5] = if alive > 0 { 1.0 } else { 0.0 };
    players[b + 6] = 1.0;
    players[b + 7] = 1.0;
    players[b + 8] = 0.0;
    players[b + 9] = 0.0;
    players[b + 10] = obs_f32(incoming.ln_1p() / 15.0);
    players[b + 11] = if border > 0 { 1.0 } else { 0.0 };
    players[b + 12] = if rest.is_empty() {
        0.0
    } else {
        obs_f32(relation_sum / rest.len() as f64 / 3.0)
    };
    players[b + 13] = if spawned > 0 { 1.0 } else { 0.0 };
}

fn write_self(players: &mut [f32], me: &crate::player::Player, land: f64) {
    players[0] = 1.0;
    players[1] = 1.0;
    players[2] = obs_f32(me.num_tiles() as f64 / land);
    players[3] = obs_f32(me.troops_f64().ln_1p() / 15.0);
    players[4] = obs_f32((me.gold as f64).ln_1p() / 20.0);
    players[5] = if me.is_alive() { 1.0 } else { 0.0 };
    players[6] = if me.player_type() == PlayerType::Nation { 1.0 } else { 0.0 };
    players[7] = if me.player_type() == PlayerType::Bot { 1.0 } else { 0.0 };
    players[8] = 0.0;
    players[9] = 0.0;
    players[10] = 0.0;
    players[11] = 0.0;
    players[12] = 3.0 / 3.0; // Relation.Friendly
    players[13] = if me.has_spawned() { 1.0 } else { 0.0 };
}

fn write_other(
    players: &mut [f32],
    slot: usize,
    me: &crate::player::Player,
    p: &crate::player::Player,
    land: f64,
    shares: bool,
    game: &Game,
) {
    let b = slot * PLAYER_FEATURES;
    players[b] = 1.0;
    players[b + 1] = 0.0;
    players[b + 2] = obs_f32(p.num_tiles() as f64 / land);
    players[b + 3] = obs_f32(p.troops_f64().ln_1p() / 15.0);
    players[b + 4] = obs_f32((p.gold as f64).ln_1p() / 20.0);
    players[b + 5] = if p.is_alive() { 1.0 } else { 0.0 };
    players[b + 6] = if p.player_type() == PlayerType::Nation { 1.0 } else { 0.0 };
    players[b + 7] = if p.player_type() == PlayerType::Bot { 1.0 } else { 0.0 };
    let incoming = game
        .all_players()
        .filter(|o| o.id() == p.id())
        .flat_map(|o| o.outgoing.iter())
        .filter(|a| a.target_player_id.as_deref() == Some(me.id()) && a.active)
        .map(|a| a.troops)
        .sum::<f64>();
    players[b + 8] = if game.is_allied(me.id(), p.id()) { 1.0 } else { 0.0 };
    players[b + 9] = if me.has_embargo_against(p.id()) { 1.0 } else { 0.0 };
    players[b + 10] = obs_f32(incoming.ln_1p() / 15.0);
    players[b + 11] = if shares { 1.0 } else { 0.0 };
    players[b + 12] = obs_f32(me.relation(p.id()) as i32 as f64 / 3.0);
    players[b + 13] = if p.has_spawned() { 1.0 } else { 0.0 };
}

pub fn fill_masks(
    game: &mut Game,
    me_id: &str,
    slots: &[SlotInfo],
    obs: &mut ObsBuffers,
    wild: bool,
    allowed: Option<&[i32]>,
) {
    obs.action_mask.fill(0);
    obs.target_masks.fill(0);
    obs.unit_mask.fill(0);
    obs.quantity_mask.fill(0);
    let Some(me) = game.player(me_id) else {
        return;
    };
    let spawned = me.has_spawned();
    let in_spawn = game.in_spawn_phase();
    if in_spawn && !spawned {
        let any = fill_spawn_regions(game, &mut obs.spawn_regions);
        obs.action_mask[ACTION_SPAWN as usize] = if any { 1 } else { 0 };
        apply_allowed(obs, allowed, in_spawn, spawned);
        return;
    }
    obs.action_mask[ACTION_NOOP as usize] = 1;
    if !spawned {
        apply_allowed(obs, allowed, in_spawn, spawned);
        return;
    }
    let mut any_q = false;
    for (q, frac) in TROOP_FRACTIONS.iter().enumerate() {
        if (me.troops_f64() * frac).floor() >= 1.0 {
            obs.quantity_mask[q] = 1;
            any_q = true;
        }
    }
    let attack_row = (ACTION_ATTACK as usize) * NUM_PLAYER_SLOTS;
    let mut any_attack = false;
    if wild {
        obs.target_masks[attack_row] = 1;
        any_attack = true;
    }
    for (i, slot) in slots.iter().enumerate().skip(1) {
        if slot.can_attack && slot.shares_border {
            obs.target_masks[attack_row + i] = 1;
            any_attack = true;
        }
    }
    obs.action_mask[ACTION_ATTACK as usize] = if any_attack && any_q { 1 } else { 0 };
    obs.action_mask[ACTION_RETREAT_ALL as usize] =
        if me.outgoing.iter().any(|a| a.active && !a.retreating) {
            1
        } else {
            0
        };

    // Official ActionTranslator.fillEffectiveBuildMasks.
    let gold = me.gold;
    let mut pending_struct = Vec::new();
    let mut pending_warship = Vec::new();
    let mut pending_nukes = Vec::new();
    for (i, ty) in UNIT_HEAD_ORDER.iter().enumerate() {
        obs.unit_mask[i] = 0;
        if game.is_unit_disabled(*ty) {
            continue;
        }
        if gold < game.structure_cost(me_id, *ty) {
            continue;
        }
        if *ty == UnitType::Warship {
            pending_warship.push(i);
        } else if ty.is_nuke() {
            pending_nukes.push(i);
        } else {
            pending_struct.push(i);
        }
    }
    for i in pending_struct {
        if game.find_build_anywhere(me_id, UNIT_HEAD_ORDER[i]).is_some() {
            obs.unit_mask[i] = 1;
        }
    }
    let any_structure = UNIT_HEAD_ORDER.iter().enumerate().any(|(u, t)| {
        obs.unit_mask[u] == 1 && *t != UnitType::Warship && !t.is_nuke()
    });
    if !any_structure {
        obs.build_regions.fill(0);
    }
    if !pending_warship.is_empty() {
        let ready_port = game.units().iter().any(|u| {
            u.active && !u.under_construction && u.ty == UnitType::Port && u.owner_id == me_id
        });
        if ready_port {
            if let Some(tile) = game.find_build_anywhere(me_id, UnitType::Warship) {
                for i in &pending_warship {
                    obs.unit_mask[*i] = 1;
                }
                obs.build_regions[crate::game::region_of_tile(&game.map, tile)] = 1;
            }
        }
    }
    if !pending_nukes.is_empty() {
        let ready_silo = game.units().iter().any(|u| {
            u.active
                && !u.under_construction
                && u.ty == UnitType::MissileSilo
                && u.owner_id == me_id
                && u.missile_queue.len() != u.level as usize
        });
        if ready_silo {
            for i in pending_nukes {
                let ty = UNIT_HEAD_ORDER[i];
                if let Some(tile) = game.find_build_anywhere(me_id, ty) {
                    obs.unit_mask[i] = 1;
                    obs.build_regions[crate::game::region_of_tile(&game.map, tile)] = 1;
                }
            }
        }
    }
    let any_build = obs.unit_mask.iter().any(|&b| b != 0);
    if !any_build {
        obs.build_regions.fill(0);
        obs.action_mask[ACTION_BUILD as usize] = 0;
    } else {
        if !obs.build_regions.iter().any(|&b| b != 0) {
            obs.build_regions[0] = 1;
        }
        obs.action_mask[ACTION_BUILD as usize] = 1;
    }

    let mut any_boat = false;
    for r in 0..NUM_REGIONS {
        if obs.boat_regions[r] == 0 {
            continue;
        }
        let ok = game.find_boat_in_region(me_id, r as i32).is_some();
        obs.boat_regions[r] = if ok { 1 } else { 0 };
        if ok {
            any_boat = true;
        }
    }
    obs.action_mask[ACTION_BOAT as usize] = if any_boat && any_q { 1 } else { 0 };

    let ally_row = (ACTION_ALLY as usize) * NUM_PLAYER_SLOTS;
    let break_row = (ACTION_BREAK_ALLY as usize) * NUM_PLAYER_SLOTS;
    let emb_row = (ACTION_EMBARGO as usize) * NUM_PLAYER_SLOTS;
    let mut any_ally = false;
    let mut any_break = false;
    let mut any_emb = false;
    for (i, slot) in slots.iter().enumerate().skip(1) {
        let Some(tid) = &slot.player_id else {
            continue;
        };
        if slot.ally_signal || game.can_send_ally(me_id, tid) {
            obs.target_masks[ally_row + i] = 1;
            any_ally = true;
        }
        if game.is_allied(me_id, tid) {
            obs.target_masks[break_row + i] = 1;
            any_break = true;
        } else if tid != me_id {
            obs.target_masks[emb_row + i] = 1;
            any_emb = true;
        }
    }
    obs.action_mask[ACTION_ALLY as usize] = if any_ally { 1 } else { 0 };
    obs.action_mask[ACTION_BREAK_ALLY as usize] = if any_break { 1 } else { 0 };
    obs.action_mask[ACTION_EMBARGO as usize] = if any_emb { 1 } else { 0 };
    apply_allowed(obs, allowed, in_spawn, spawned);
}

fn fill_spawn_regions(game: &Game, out: &mut [u8]) -> bool {
    let mut any = false;
    for r in 0..NUM_REGIONS {
        if out[r] == 0 {
            continue;
        }
        let ok = find_spawn_tile(&game.map, r as i32).is_some();
        out[r] = if ok { 1 } else { 0 };
        if ok {
            any = true;
        }
    }
    any
}

fn apply_allowed(obs: &mut ObsBuffers, allowed: Option<&[i32]>, in_spawn: bool, spawned: bool) {
    let Some(allow) = allowed else {
        return;
    };
    let force_noop = !(in_spawn && !spawned);
    let force_spawn = in_spawn && !spawned;
    for a in 0..NUM_ACTION_TYPES {
        let extra = (a as i32 == ACTION_NOOP && force_noop) || (a as i32 == ACTION_SPAWN && force_spawn);
        if !allow.contains(&(a as i32)) && !extra {
            obs.action_mask[a] = 0;
        }
    }
}
