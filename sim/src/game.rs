//! Game state, tick loop, spawn, economy, and land-combat executions.

use std::collections::{HashMap, HashSet};

use indexmap::{IndexMap, IndexSet};

use crate::config::{
    attack_logic, attack_tiles_per_tick, cheapest_structure_cost, gold_addition_rate,
    start_manpower, troop_increase_rate, DEFAULT_SPAWN_IMMUNITY_TICKS,
};
use crate::diplomacy::{
    alliance_expires_at, attack_relation_delta, can_send_alliance_request, request_expired,
    Alliance, AllianceRequest, Relation, RequestStatus,
};
use crate::map::{GameMap, TileRef, WaterComponents};
use crate::hpa::WaterChain;
use crate::later::{LaterExec, LaterState};
use crate::path::{PathStatus, PathStepper};
use crate::unit_grid::UnitGrid;
use crate::player::{Attack, Player, PlayerInfo};
use crate::rng::{simple_hash, PseudoRandom};
use crate::spatial::{can_build_transport_ship, target_transport_tile};
use crate::spec::{Difficulty, PlayerType, TerrainKind};
use crate::units::{
    city_levels, construction_duration, structure_count, unit_cost, Unit, UnitType, BOAT_MAX_NUMBER,
    DEFENSE_POST_RANGE, STRUCTURE_MIN_DIST, TRAITOR_DURATION,
};

const MALUS_FOR_RETREAT: f64 = 25.0;
const RETREAT_DELAY: u32 = 20;

#[derive(Clone, Debug)]
pub enum Intent {
    Spawn { tile: TileRef },
    Attack {
        target_id: Option<String>,
        troops: f64,
        source_tile: Option<TileRef>,
        remove_troops: bool,
    },
    Retreat { attack_id: String },
    Boat { dst: TileRef, troops: f64 },
    Build { unit: UnitType, tile: TileRef },
    AllianceRequest { recipient: String },
    BreakAlliance { recipient: String },
    Embargo { target: String, start: bool },
}

pub(crate) enum Exec {
    Spawn(SpawnExec),
    Player(PlayerExec),
    Attack(AttackExec),
    Retreat(RetreatExec),
    WinCheck(WinCheckExec),
    Boat(BoatExec),
    Construction(ConstructionExec),
    AllianceReq(AllianceReqExec),
    BreakAlly(BreakAllyExec),
    Embargo(EmbargoExec),
    Tribe(TribeExec),
    Nation(NationExec),
    Later(LaterExec),
}

struct SpawnExec {
    active: bool,
    player_id: String,
    game_id: String,
    tile: Option<TileRef>,
    info: Option<PlayerInfo>,
    rng: PseudoRandom,
}

struct PlayerExec {
    active: bool,
    player_id: String,
    last_calc: u32,
    inited: bool,
}

struct AttackExec {
    active: bool,
    inited: bool,
    start_troops: f64,
    owner_id: String,
    target_id: Option<String>,
    attack_id: Option<String>,
    owner_small: u16,
    target_small: u16,
    target_is_player: bool,
    source_tile: Option<TileRef>,
    remove_troops: bool,
    heap: FlatHeap,
    rng: PseudoRandom,
}

struct RetreatExec {
    active: bool,
    inited: bool,
    player_id: String,
    attack_id: String,
    start_tick: u32,
    ordered: bool,
}

struct WinCheckExec {
    active: bool,
    inited: bool,
}

struct BoatExec {
    active: bool,
    inited: bool,
    owner_id: String,
    ref_tile: TileRef,
    troops: f64,
    dst: Option<TileRef>,
    src: Option<TileRef>,
    unit_id: Option<u32>,
    last_move: u32,
    stepper: PathStepper,
}

pub(crate) struct ConstructionExec {
    active: bool,
    inited: bool,
    player_id: String,
    ty: UnitType,
    tile: TileRef,
    unit_id: Option<u32>,
    ticks_left: u32,
}

struct AllianceReqExec {
    active: bool,
    inited: bool,
    requestor: String,
    recipient: String,
}

struct BreakAllyExec {
    active: bool,
    inited: bool,
    requestor: String,
    recipient: String,
}

struct EmbargoExec {
    active: bool,
    inited: bool,
    player_id: String,
    target_id: String,
    start: bool,
}

struct TribeExec {
    active: bool,
    inited: bool,
    player_id: String,
    attack_rate: u32,
    attack_tick: u32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    behavior: bool,
    neighbors_tn: bool,
    rng: PseudoRandom,
}

pub(crate) struct NationExec {
    active: bool,
    inited: bool,
    player_id: String,
    spawn_x: Option<u32>,
    spawn_y: Option<u32>,
    attack_rate: u32,
    attack_tick: u32,
    trigger_ratio: f64,
    reserve_ratio: f64,
    expand_ratio: f64,
    behaviors: bool,
    spawn_added: bool,
    pub(crate) rng: PseudoRandom,
}

#[derive(Clone, Debug)]
pub struct NationSpec {
    pub name: String,
    pub spawn_x: Option<u32>,
    pub spawn_y: Option<u32>,
    pub id: String,
}

pub struct Game {
    pub map: GameMap,
    pub water: WaterComponents,
    pub mini_map: GameMap,
    pub mini_water: WaterComponents,
    players: IndexMap<String, Player>,
    by_small: HashMap<u16, String>,
    next_small: u16,
    ticks: u32,
    start_tick: Option<u32>,
    winner: Option<String>,
    pub(crate) execs: Vec<Exec>,
    pub(crate) uninit: Vec<Exec>,
    last_hash: Option<f64>,
    pub(crate) difficulty: Difficulty,
    gold_mult: f64,
    pending_kills: HashMap<String, u32>,
    pub(crate) units: Vec<Unit>,
    next_unit_id: u32,
    alliances: Vec<Alliance>,
    next_alliance_id: u32,
    incoming_requests: Vec<AllianceRequest>,
    game_id: String,
    pub fallback: Option<String>,
    water_chain: WaterChain,
    boat_cache: Vec<Option<TileRef>>,
    pub(crate) later: LaterState,
    tile_update_pairs: Vec<u32>,
}

impl Game {
    pub fn new(
        map: GameMap,
        mini_map: GameMap,
        humans: Vec<PlayerInfo>,
        difficulty: Difficulty,
        game_id: String,
    ) -> Self {
        let water = WaterComponents::compute(&map);
        let mini_water = WaterComponents::compute(&mini_map);
        let water_chain = WaterChain::new(&mini_map);
        let later = LaterState::new(UnitGrid::new(&map));
        let mut game = Self {
            map,
            water,
            mini_map,
            mini_water,
            players: IndexMap::new(),
            by_small: HashMap::new(),
            next_small: 1,
            ticks: 0,
            start_tick: None,
            winner: None,
            execs: Vec::new(),
            uninit: Vec::new(),
            last_hash: None,
            difficulty,
            gold_mult: 1.0,
            pending_kills: HashMap::new(),
            units: Vec::new(),
            next_unit_id: 1,
            alliances: Vec::new(),
            next_alliance_id: 1,
            incoming_requests: Vec::new(),
            game_id,
            fallback: None,
            water_chain,
            boat_cache: vec![None; crate::spec::NUM_REGIONS],
            later,
            tile_update_pairs: Vec::new(),
        };
        for info in humans {
            game.add_player(info);
        }
        game
    }

    pub fn add_player(&mut self, info: PlayerInfo) -> u16 {
        let sid = self.next_small;
        self.next_small += 1;
        let seed = simple_hash(&info.id);
        let troops = start_manpower(info.player_type, self.difficulty);
        let id = info.id.clone();
        let p = Player::new(sid, info, troops, seed);
        self.by_small.insert(sid, id.clone());
        self.players.insert(id, p);
        sid
    }

    pub fn ticks(&self) -> u32 {
        self.ticks
    }
    pub fn in_spawn_phase(&self) -> bool {
        self.start_tick.is_none()
    }
    pub fn last_hash(&self) -> Option<f64> {
        self.last_hash
    }
    pub fn winner(&self) -> Option<&str> {
        self.winner.as_deref()
    }
    pub fn pending_kills(&self, pid: &str) -> u32 {
        self.pending_kills.get(pid).copied().unwrap_or(0)
    }
    pub fn take_kills(&mut self, pid: &str) -> u32 {
        self.pending_kills.remove(pid).unwrap_or(0)
    }

    pub fn player(&self, id: &str) -> Option<&Player> {
        self.players.get(id)
    }
    pub fn player_mut(&mut self, id: &str) -> Option<&mut Player> {
        self.players.get_mut(id)
    }
    pub fn player_by_client(&self, client: &str) -> Option<&Player> {
        self.players.values().find(|p| p.info.client_id.as_deref() == Some(client))
    }
    pub fn player_by_small(&self, sid: u16) -> Option<&Player> {
        self.by_small.get(&sid).and_then(|id| self.players.get(id))
    }
    pub fn all_players(&self) -> impl Iterator<Item = &Player> {
        self.players.values()
    }
    pub fn alive_players(&self) -> Vec<&Player> {
        self.players.values().filter(|p| p.is_alive()).collect()
    }

    pub fn end_spawn_phase(&mut self) {
        if self.start_tick.is_none() {
            self.start_tick = Some(self.ticks);
        }
    }

    pub fn ticks_since_start(&self) -> u32 {
        match self.start_tick {
            None => 0,
            Some(s) => self.ticks.saturating_sub(s),
        }
    }

    pub fn spawn_immunity_active(&self) -> bool {
        self.in_spawn_phase() || self.ticks_since_start() < DEFAULT_SPAWN_IMMUNITY_TICKS
    }

    pub fn is_immune(&self, p: &Player) -> bool {
        match p.player_type() {
            PlayerType::Human => self.spawn_immunity_active(),
            PlayerType::Nation => {
                self.in_spawn_phase() || self.ticks_since_start() < DEFAULT_SPAWN_IMMUNITY_TICKS
            }
            PlayerType::Bot => false,
        }
    }

    pub fn is_allied(&self, a: &str, b: &str) -> bool {
        self.alliances.iter().any(|al| al.involves(a, b))
    }

    pub fn is_friendly(&self, a: &Player, b: &Player) -> bool {
        if a.id() == b.id() {
            return true;
        }
        if b.disconnected {
            return false;
        }
        self.is_allied(a.id(), b.id())
    }

    pub fn can_attack_player(&self, attacker: &Player, target: &Player) -> bool {
        if self.is_friendly(attacker, target) {
            return false;
        }
        if attacker.player_type() != PlayerType::Human {
            return true;
        }
        !self.is_immune(target)
    }

    pub fn units(&self) -> &[Unit] {
        &self.units
    }

    pub fn active_units(&self) -> impl Iterator<Item = &Unit> {
        self.units.iter().filter(|u| u.active)
    }

    pub fn fallback_reason(&self) -> Option<&str> {
        self.fallback.as_deref()
    }

    pub fn mark_official(&mut self, reason: &str) {
        if self.fallback.is_none() {
            self.fallback = Some(reason.to_string());
        }
    }

    pub fn shares_border(&self, a: &Player, b_small: u16) -> bool {
        let mut nbuf = [0u32; 4];
        for &tile in &a.border_tiles {
            let n = self.map.neighbors4(tile, &mut nbuf);
            for i in 0..n {
                if self.map.owner_id(nbuf[i]) == b_small {
                    return true;
                }
            }
        }
        false
    }

    pub fn add_intents(&mut self, game_id: &str, owner_id: &str, intents: &[Intent]) {
        for intent in intents {
            match intent {
                Intent::Spawn { tile } => {
                    self.uninit.push(Exec::Spawn(SpawnExec {
                        active: true,
                        player_id: owner_id.to_string(),
                        game_id: game_id.to_string(),
                        tile: Some(*tile),
                        info: None,
                        rng: PseudoRandom::new(
                            simple_hash(owner_id).wrapping_add(simple_hash(game_id)),
                        ),
                    }));
                }
                Intent::Attack {
                    target_id,
                    troops,
                    source_tile,
                    remove_troops,
                } => {
                    self.uninit.push(Exec::Attack(AttackExec {
                        active: true,
                        inited: false,
                        start_troops: *troops,
                        owner_id: owner_id.to_string(),
                        target_id: target_id.clone(),
                        attack_id: None,
                        owner_small: 0,
                        target_small: 0,
                        target_is_player: target_id.is_some(),
                        source_tile: *source_tile,
                        remove_troops: *remove_troops,
                        heap: FlatHeap::new(),
                        rng: PseudoRandom::new(123),
                    }));
                }
                Intent::Retreat { attack_id } => {
                    self.uninit.push(Exec::Retreat(RetreatExec {
                        active: true,
                        inited: false,
                        player_id: owner_id.to_string(),
                        attack_id: attack_id.clone(),
                        start_tick: 0,
                        ordered: false,
                    }));
                }
                Intent::Boat { dst, troops } => {
                    self.uninit.push(Exec::Boat(BoatExec {
                        active: true,
                        inited: false,
                        owner_id: owner_id.to_string(),
                        ref_tile: *dst,
                        troops: *troops,
                        dst: None,
                        src: None,
                        unit_id: None,
                        last_move: 0,
                        stepper: PathStepper::new(),
                    }));
                }
                Intent::Build { unit, tile } => {
                    {
                        self.uninit.push(Exec::Construction(ConstructionExec {
                            active: true,
                            inited: false,
                            player_id: owner_id.to_string(),
                            ty: *unit,
                            tile: *tile,
                            unit_id: None,
                            ticks_left: 0,
                        }));
                    }
                }
                Intent::AllianceRequest { recipient } => {
                    self.uninit.push(Exec::AllianceReq(AllianceReqExec {
                        active: true,
                        inited: false,
                        requestor: owner_id.to_string(),
                        recipient: recipient.clone(),
                    }));
                }
                Intent::BreakAlliance { recipient } => {
                    self.uninit.push(Exec::BreakAlly(BreakAllyExec {
                        active: true,
                        inited: false,
                        requestor: owner_id.to_string(),
                        recipient: recipient.clone(),
                    }));
                }
                Intent::Embargo { target, start } => {
                    self.uninit.push(Exec::Embargo(EmbargoExec {
                        active: true,
                        inited: false,
                        player_id: owner_id.to_string(),
                        target_id: target.clone(),
                        start: *start,
                    }));
                }
            }
        }
    }

    pub fn execute_next_tick(&mut self) {
        let t = self.ticks;
        let in_spawn = self.in_spawn_phase();
        let mut i = 0;
        while i < self.execs.len() {
            if self.should_run(&self.execs[i], in_spawn) {
                self.tick_exec(i, t);
            }
            i += 1;
        }
        let pending = std::mem::take(&mut self.uninit);
        let still_spawn = self.in_spawn_phase();
        for mut e in pending {
            if still_spawn && !during_spawn(&e) {
                self.uninit.push(e);
                continue;
            }
            self.init_exec(&mut e, t);
            self.execs.push(e);
        }
        let still = self.in_spawn_phase();
        self.execs.retain(|e| keep_exec(e, still));
        if self.ticks % 10 == 0 {
            self.last_hash = Some(self.core_hash());
        }
        self.ticks += 1;
    }

    fn should_run(&self, e: &Exec, in_spawn: bool) -> bool {
        is_active(e) && (!in_spawn || during_spawn(e))
    }

    fn core_hash(&self) -> f64 {
        let mut hash = 1.0;
        for p in self.players.values() {
            let mut ph = p.core_hash();
            for uid in &p.units {
                if let Some(u) = self.units.iter().find(|u| u.id == *uid && u.active) {
                    ph += u.hash_f64();
                }
            }
            hash += ph;
        }
        hash
    }

    pub fn core_hash_now(&self) -> f64 {
        self.core_hash()
    }

    fn init_exec(&mut self, e: &mut Exec, ticks: u32) {
        match e {
            Exec::Spawn(_) => {}
            Exec::Player(p) => {
                if let Some(pl) = self.players.get(&p.player_id) {
                    p.last_calc = ticks + (simple_hash(pl.id()) as u32 % 20);
                }
                p.inited = true;
            }
            Exec::Attack(a) => self.init_attack(a, ticks),
            Exec::Retreat(r) => {
                r.start_tick = ticks;
                r.inited = true;
            }
            Exec::WinCheck(w) => w.inited = true,
            Exec::Boat(b) => self.init_boat(b, ticks),
            Exec::Construction(c) => {
                c.inited = true;
            }
            Exec::AllianceReq(a) => {
                a.inited = true;
            }
            Exec::BreakAlly(b) => {
                b.inited = true;
            }
            Exec::Embargo(e) => {
                e.inited = true;
            }
            Exec::Tribe(t) => {
                t.inited = true;
            }
            Exec::Later(e) => match e {
                crate::later::LaterExec::Port {
                    rng,
                    check_offset,
                    inited,
                    ..
                } => {
                    *rng = Some(PseudoRandom::new(ticks as i32));
                    *check_offset = ticks % 10;
                    *inited = true;
                }
                crate::later::LaterExec::TrainStation {
                    spawn_trains,
                    rng,
                    inited,
                    ..
                } => {
                    if *spawn_trains {
                        *rng = Some(PseudoRandom::new(ticks as i32));
                    }
                    *inited = true;
                }
                crate::later::LaterExec::Warship {
                    rng,
                    inited,
                    unit_id,
                    owner,
                    patrol,
                    active,
                    ..
                } => {
                    *rng = Some(PseudoRandom::new(ticks as i32));
                    // Official WarshipExecution.init builds the unit this tick.
                    if !*inited {
                        *inited = true;
                        let o = owner.clone();
                        let p = *patrol;
                        if let Some(spawn) = self.can_build(&o, UnitType::Warship, p) {
                            if let Some(id) = self.build_unit(&o, UnitType::Warship, spawn, 0.0) {
                                if let Some(u) = self.units.iter_mut().find(|u| u.id == id) {
                                    u.patrol_tile = Some(p);
                                    u.target_tile = None;
                                }
                                *unit_id = Some(id);
                            } else {
                                *active = false;
                            }
                        } else {
                            *active = false;
                        }
                    }
                }
                crate::later::LaterExec::Mirv { rng, owner, .. } => {
                    *rng = Some(PseudoRandom::new(ticks as i32 + simple_hash(owner)));
                }
                crate::later::LaterExec::Train { .. } => {
                    self.init_train(e);
                }
                crate::later::LaterExec::Upgrade {
                    owner,
                    unit_id,
                    inited,
                    active,
                    ..
                } => {
                    let pid = owner.clone();
                    let uid = *unit_id;
                    if let Some(ty) = self.units.iter().find(|u| u.id == uid).map(|u| u.ty) {
                        self.upgrade_structure_unit(&pid, uid, ty);
                    }
                    *inited = true;
                    *active = false;
                }
                _ => {}
            }
            Exec::Nation(n) => {
                n.inited = true;
                if n.attack_rate == 0 {
                    let seed = simple_hash(&n.player_id).wrapping_add(simple_hash(&self.game_id));
                    let mut rng = PseudoRandom::new(seed);
                    n.trigger_ratio = rng.next_int(50.0, 60.0) as f64 / 100.0;
                    n.reserve_ratio = rng.next_int(30.0, 40.0) as f64 / 100.0;
                    n.expand_ratio = rng.next_int(10.0, 20.0) as f64 / 100.0;
                    n.attack_rate = match self.difficulty {
                        Difficulty::Easy => rng.next_int(65.0, 100.0) as u32,
                        Difficulty::Medium => rng.next_int(55.0, 70.0) as u32,
                        Difficulty::Hard => rng.next_int(45.0, 60.0) as u32,
                        Difficulty::Impossible => rng.next_int(30.0, 50.0) as u32,
                    };
                    n.attack_tick = rng.next_int(0.0, n.attack_rate as f64) as u32;
                    n.rng = rng;
                }
            }
        }
    }

    fn tick_exec(&mut self, idx: usize, ticks: u32) {
        match &self.execs[idx] {
            Exec::Spawn(_) => self.tick_spawn(idx, ticks),
            Exec::Player(_) => self.tick_player(idx, ticks),
            Exec::Attack(_) => self.tick_attack(idx, ticks),
            Exec::Retreat(_) => self.tick_retreat(idx, ticks),
            Exec::WinCheck(_) => self.tick_win(idx, ticks),
            Exec::Boat(_) => self.tick_boat(idx, ticks),
            Exec::Construction(_) => self.tick_construction(idx, ticks),
            Exec::AllianceReq(_) => self.tick_alliance_req(idx, ticks),
            Exec::BreakAlly(_) => self.tick_break_ally(idx, ticks),
            Exec::Embargo(_) => self.tick_embargo(idx, ticks),
            Exec::Tribe(_) => self.tick_tribe(idx, ticks),
            Exec::Nation(_) => self.tick_nation(idx, ticks),
            Exec::Later(_) => crate::later_tick::tick_later(self, idx, ticks),
        }
    }

    fn tick_spawn(&mut self, idx: usize, _ticks: u32) {
        let Exec::Spawn(s) = &self.execs[idx] else {
            return;
        };
        let player_id = s.player_id.clone();
        let tile = s.tile;
        let info = s.info.clone();
        let game_id = s.game_id.clone();
        let mut rng = s.rng.clone();
        if let Some(e) = self.execs.get_mut(idx) {
            if let Exec::Spawn(s) = e {
                s.active = false;
            }
        }
        if let Some(info) = info {
            if !self.players.contains_key(&player_id) {
                self.add_player(info);
            }
        }
        let already = self.players.get(&player_id).map(|p| p.has_spawned()).unwrap_or(true);
        if !self.in_spawn_phase() && already {
            return;
        }
        if tile.is_some_and(|t| !self.map.is_valid_ref(t as i32)) {
            return;
        }
        // Official SpawnExecution.tick always drops existing territory before
        // placing the new blob. Without this, spawn-phase hops accumulate land.
        let owned: Vec<TileRef> = self
            .players
            .get(&player_id)
            .map(|p| p.tiles.iter().copied().collect())
            .unwrap_or_default();
        for t in owned {
            self.relinquish_tile(&player_id, t);
        }
        let pty = self.players.get(&player_id).map(|p| p.player_type());
        let center = if let Some(t) = tile {
            t
        } else {
            match self.random_spawn_center(&mut rng, &player_id) {
                Some(t) => t,
                None => return,
            }
        };
        let require_all = tile.is_none();
        let tiles = spawn_tiles(&self.map, center, require_all);
        if tiles.is_empty() {
            return;
        }
        for t in tiles {
            self.conquer(&player_id, t);
        }
        let need_player_exec = self
            .players
            .get(&player_id)
            .map(|p| !p.has_spawned())
            .unwrap_or(false);
        if let Some(p) = self.players.get_mut(&player_id) {
            p.spawn_tile = Some(center);
        }
        if need_player_exec {
            self.uninit.push(Exec::Player(PlayerExec {
                active: true,
                player_id: player_id.clone(),
                last_calc: 0,
                inited: false,
            }));
            if pty == Some(PlayerType::Bot) {
                self.queue_tribe_exec(&player_id);
            }
        }
        if pty == Some(PlayerType::Human) {
            self.end_spawn_phase();
        }
        let _ = game_id;
    }

    fn tick_player(&mut self, idx: usize, ticks: u32) {
        let Exec::Player(pe) = &self.execs[idx] else {
            return;
        };
        let pid = pe.player_id.clone();
        let Some(p) = self.players.get(&pid) else {
            return;
        };
        if !p.is_alive() {
            if let Some(pl) = self.players.get_mut(&pid) {
                pl.gold = 0;
            }
            if let Exec::Player(pe) = &mut self.execs[idx] {
                pe.active = false;
            }
            return;
        }
        let cities = city_levels(&self.units, &pid);
        let inc = troop_increase_rate(
            p.player_type(),
            self.difficulty,
            p.troops_f64(),
            p.num_tiles(),
            cities,
        );
        let gold = gold_addition_rate(p.player_type(), self.gold_mult);
        if let Some(pl) = self.players.get_mut(&pid) {
            pl.add_troops(inc);
            pl.add_gold(gold);
            pl.decay_relations();
        }
        self.capture_or_delete_structures(&pid);
        self.expire_alliances_for(&pid);
        let _ = ticks;
    }

    fn init_attack(&mut self, a: &mut AttackExec, _ticks: u32) {
        a.inited = true;
        let Some(owner) = self.players.get(&a.owner_id) else {
            a.active = false;
            return;
        };
        a.owner_small = owner.small_id;
        if let Some(tid) = &a.target_id {
            if !self.players.contains_key(tid) {
                a.active = false;
                return;
            }
            if tid == &a.owner_id {
                a.active = false;
                return;
            }
            let target = self.players.get(tid).unwrap();
            if self.is_friendly(owner, target) {
                a.active = false;
                return;
            }
            if !self.can_attack_player(owner, target) {
                a.active = false;
                return;
            }
            a.target_small = target.small_id;
            a.target_is_player = true;
        } else {
            a.target_small = 0;
            a.target_is_player = false;
        }
        let owner_type = owner.player_type();
        let troops = if a.remove_troops {
            a.start_troops.min(owner.troops_f64()).max(0.0)
        } else {
            a.start_troops.max(0.0)
        };
        a.start_troops = troops;
        let src = a.source_tile;
        if let Some(tid) = a.target_id.clone() {
            self.reject_incoming_from(&a.owner_id, &tid);
        }
        if let Some(owner) = self.players.get_mut(&a.owner_id) {
            if a.remove_troops {
                owner.remove_troops(troops.floor() as i64);
            }
            let id = owner.rng.next_id();
            let atk = Attack {
                id: id.clone(),
                target_small_id: a.target_small,
                target_player_id: a.target_id.clone(),
                attacker_small_id: a.owner_small,
                troops,
                retreating: false,
                retreated: false,
                active: true,
                source_tile: src,
                border: Default::default(),
            };
            a.attack_id = Some(id.clone());
            owner.outgoing.push(atk);
            owner.attacks_sent += 1;
            // Official AttackExecution: a land attack (sourceTile == null)
            // absorbs every existing outgoing attack on the same target,
            // including boat-sourced ones. A new boat attack is not merged.
            if src.is_none() {
                let mut extra = 0.0;
                for other in owner.outgoing.iter_mut() {
                    if other.id != id && other.target_player_id == a.target_id && other.active {
                        extra += other.troops;
                        other.active = false;
                    }
                }
                if extra > 0.0 {
                    if let Some(merged) = owner.outgoing.iter_mut().find(|x| x.id == id) {
                        merged.troops += extra;
                    }
                    owner.outgoing.retain(|x| x.active);
                }
            }
        }
        self.cancel_opposing(&a.owner_id, a.target_id.as_deref(), a.attack_id.as_deref());
        if let Some(src) = a.source_tile {
            self.add_neighbors(a, src);
        } else {
            self.refresh_to_conquer(a);
        }
        if let Some(tid) = a.target_id.clone() {
            if let Some(def) = self.players.get_mut(&tid) {
                if owner_type != PlayerType::Bot && def.player_type() != PlayerType::Bot {
                    def.add_embargo(&a.owner_id, self.ticks, true);
                }
                def.update_relation(&a.owner_id, attack_relation_delta(self.difficulty));
            }
        }
    }

    fn cancel_opposing(&mut self, owner_id: &str, target_id: Option<&str>, attack_id: Option<&str>) {
        let Some(tid) = target_id else {
            return;
        };
        let Some(atk_id) = attack_id else {
            return;
        };
        let my_troops = self
            .players
            .get(owner_id)
            .and_then(|p| p.outgoing.iter().find(|a| a.id == atk_id))
            .map(|a| a.troops)
            .unwrap_or(0.0);
        let Some(target) = self.players.get_mut(tid) else {
            return;
        };
        let mut opp = None;
        for (i, at) in target.outgoing.iter().enumerate() {
            if at.target_player_id.as_deref() == Some(owner_id) && at.active {
                opp = Some((i, at.troops, at.id.clone()));
                break;
            }
        }
        if let Some((i, ot, _oid)) = opp {
            if ot > my_troops {
                target.outgoing[i].troops -= my_troops;
                if let Some(owner) = self.players.get_mut(owner_id) {
                    if let Some(a) = owner.outgoing.iter_mut().find(|a| a.id == atk_id) {
                        a.active = false;
                    }
                    owner.outgoing.retain(|a| a.active);
                }
            } else {
                target.outgoing[i].active = false;
                target.outgoing.retain(|a| a.active);
                if let Some(owner) = self.players.get_mut(owner_id) {
                    if let Some(a) = owner.outgoing.iter_mut().find(|a| a.id == atk_id) {
                        a.troops -= ot;
                    }
                }
            }
        }
    }

    fn refresh_to_conquer(&mut self, a: &mut AttackExec) {
        a.heap.clear();
        let Some(atk_id) = a.attack_id.clone() else {
            return;
        };
        if let Some(owner) = self.players.get_mut(&a.owner_id) {
            if let Some(atk) = owner.outgoing.iter_mut().find(|x| x.id == atk_id) {
                atk.clear_border();
            }
        }
        let borders: Vec<TileRef> = self
            .players
            .get(&a.owner_id)
            .map(|p| p.border_tiles.iter().copied().collect())
            .unwrap_or_default();
        for t in borders {
            self.add_neighbors(a, t);
        }
    }

    fn add_neighbors(&mut self, a: &mut AttackExec, tile: TileRef) {
        let tick_now = self.ticks as f64;
        let mut nbuf = [0u32; 4];
        let n = self.map.neighbors4(tile, &mut nbuf);
        for i in 0..n {
            let neighbor = nbuf[i];
            if self.map.is_water(neighbor)
                || self.map.is_impassable(neighbor)
                || self.map.owner_id(neighbor) != a.target_small
            {
                continue;
            }
            if let Some(atk_id) = &a.attack_id {
                if let Some(owner) = self.players.get_mut(&a.owner_id) {
                    if let Some(atk) = owner.outgoing.iter_mut().find(|x| x.id == *atk_id) {
                        atk.add_border_tile(neighbor);
                    }
                }
            }
            let mut nbuf2 = [0u32; 4];
            let n2 = self.map.neighbors4(neighbor, &mut nbuf2);
            let mut owned = 0;
            for j in 0..n2 {
                if self.map.owner_id(nbuf2[j]) == a.owner_small {
                    owned += 1;
                }
            }
            let mag = match self.map.terrain_type(neighbor) {
                TerrainKind::Plains => 1.0,
                TerrainKind::Highland => 1.5,
                TerrainKind::Mountain => 2.0,
                _ => 0.0,
            };
            let pri = (a.rng.next_int(0.0, 7.0) as f64 + 10.0)
                * (1.0 - owned as f64 * 0.5 + mag / 2.0)
                + tick_now;
            a.heap.enqueue(neighbor, pri as f32);
        }
    }

    fn tick_attack(&mut self, idx: usize, _ticks: u32) {
        let (owner_id, atk_id, target_id, target_is_player, owner_small, target_small) = {
            let Exec::Attack(a) = &self.execs[idx] else {
                return;
            };
            if !a.inited || !a.active {
                return;
            }
            (
                a.owner_id.clone(),
                a.attack_id.clone(),
                a.target_id.clone(),
                a.target_is_player,
                a.owner_small,
                a.target_small,
            )
        };
        let Some(atk_id) = atk_id else {
            return;
        };
        let Some(owner) = self.players.get(&owner_id) else {
            return;
        };
        let Some(atk) = owner.outgoing.iter().find(|x| x.id == atk_id).cloned() else {
            if let Exec::Attack(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        };
        if atk.retreated {
            self.retreat_attack(&owner_id, &atk_id, target_is_player, true);
            if let Exec::Attack(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        }
        if atk.retreating {
            return;
        }
        if !atk.active {
            if let Exec::Attack(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        }
        let mut troop_count = atk.troops;
        let border = atk.border_size() as f64;
        let extra = {
            let Exec::Attack(a) = &mut self.execs[idx] else {
                return;
            };
            a.rng.next_int(0.0, 5.0) as f64
        };
        let def_troops = target_id
            .as_ref()
            .and_then(|id| self.players.get(id))
            .map(|p| p.troops_f64())
            .unwrap_or(0.0);
        let mut num_tiles = attack_tiles_per_tick(
            troop_count,
            target_is_player,
            def_troops.max(1.0),
            border + extra,
        );
        while num_tiles > 0.0 {
            if troop_count < 1.0 {
                self.delete_attack(&owner_id, &atk_id);
                if let Exec::Attack(a) = &mut self.execs[idx] {
                    a.active = false;
                }
                return;
            }
            let tile = {
                let Exec::Attack(a) = &mut self.execs[idx] else {
                    return;
                };
                if a.heap.is_empty() {
                    None
                } else {
                    Some(a.heap.dequeue())
                }
            };
            let Some(tile) = tile else {
                let mut tmp = self.take_attack_exec(idx);
                self.refresh_to_conquer(&mut tmp);
                self.put_attack_exec(idx, tmp);
                self.retreat_attack(&owner_id, &atk_id, target_is_player, false);
                if let Exec::Attack(a) = &mut self.execs[idx] {
                    a.active = false;
                }
                return;
            };
            if let Some(owner) = self.players.get_mut(&owner_id) {
                if let Some(atk) = owner.outgoing.iter_mut().find(|x| x.id == atk_id) {
                    atk.remove_border_tile(tile);
                }
            }
            let mut on_border = false;
            let mut nbuf = [0u32; 4];
            let n = self.map.neighbors4(tile, &mut nbuf);
            for i in 0..n {
                if self.map.owner_id(nbuf[i]) == owner_small {
                    on_border = true;
                    break;
                }
            }
            if self.map.owner_id(tile) != target_small || !on_border {
                continue;
            }
            if !self.map.is_land(tile) || self.map.is_impassable(tile) {
                continue;
            }
            {
                let mut tmp = self.take_attack_exec(idx);
                self.add_neighbors(&mut tmp, tile);
                self.put_attack_exec(idx, tmp);
            }
            let terrain = self.map.terrain_type(tile);
            let (atk_type, atk_tiles) = self
                .players
                .get(&owner_id)
                .map(|p| (p.player_type(), p.num_tiles()))
                .unwrap_or((PlayerType::Human, 0));
            let (def_type, def_tiles, def_troops, traitor) = target_id
                .as_ref()
                .and_then(|id| self.players.get(id))
                .map(|p| (p.player_type(), p.num_tiles(), p.troops_f64(), p.traitor))
                .unwrap_or((PlayerType::Human, 0, 0.0, false));
            let fallout = if self.map.has_fallout(tile) {
                Some(5.0 - (self.map.num_tiles_with_fallout() as f64 / self.map.num_land_tiles().max(1) as f64) * 2.0)
            } else {
                None
            };
            let has_dp = target_is_player
                && target_id
                    .as_ref()
                    .map(|tid| self.has_defense_post(tid, tile))
                    .unwrap_or(false);
            let out = attack_logic(
                troop_count,
                atk_type,
                atk_tiles,
                target_is_player,
                def_type,
                def_troops,
                def_tiles,
                traitor,
                terrain,
                has_dp,
                fallout,
            );
            num_tiles -= out.tiles_per_tick_used;
            troop_count -= out.attacker_troop_loss;
            if let Some(owner) = self.players.get_mut(&owner_id) {
                if let Some(atk) = owner.outgoing.iter_mut().find(|x| x.id == atk_id) {
                    atk.troops = troop_count.max(0.0);
                }
            }
            if let Some(tid) = &target_id {
                if let Some(def) = self.players.get_mut(tid) {
                    def.remove_troops(out.defender_troop_loss.floor() as i64);
                }
            }
            self.conquer(&owner_id, tile);
            if target_is_player {
                if let Some(tid) = &target_id {
                    if let Some(def) = self.players.get(tid) {
                        if def.num_tiles() < 100 {
                            self.conquer_player(&owner_id, tid);
                        }
                    }
                }
            }
        }
        // Correct defender troop loss: official floors via toInt once per tile.
        // Applied inside the loop via remove_troops(floor). The duplicate draft
        // above is cleaned by only calling once — see the single call remaining.
        let _ = cheapest_structure_cost;
    }

    fn retreat_attack(
        &mut self,
        owner_id: &str,
        atk_id: &str,
        target_is_player: bool,
        canceled: bool,
    ) {
        // Official `retreat()` default malus is 0; canceled player attacks use 25.
        let malus = if canceled && target_is_player {
            MALUS_FOR_RETREAT
        } else {
            0.0
        };
        if let Some(owner) = self.players.get_mut(owner_id) {
            let survivors = owner
                .outgoing
                .iter()
                .find(|a| a.id == atk_id)
                .map(|atk| atk.troops - atk.troops * (malus / 100.0));
            if let Some(surv) = survivors {
                owner.add_troops(surv);
                if let Some(atk) = owner.outgoing.iter_mut().find(|a| a.id == atk_id) {
                    atk.active = false;
                }
            }
            owner.outgoing.retain(|a| a.active);
        }
    }

    fn delete_attack(&mut self, owner_id: &str, atk_id: &str) {
        if let Some(owner) = self.players.get_mut(owner_id) {
            if let Some(atk) = owner.outgoing.iter_mut().find(|a| a.id == atk_id) {
                atk.active = false;
            }
            owner.outgoing.retain(|a| a.active);
        }
    }

    fn tick_retreat(&mut self, idx: usize, ticks: u32) {
        let Exec::Retreat(r) = &self.execs[idx] else {
            return;
        };
        let pid = r.player_id.clone();
        let aid = r.attack_id.clone();
        let start = r.start_tick;
        if !r.ordered {
            if let Some(p) = self.players.get_mut(&pid) {
                if let Some(a) = p.outgoing.iter_mut().find(|a| a.id == aid) {
                    a.retreating = true;
                }
            }
            if let Exec::Retreat(r) = &mut self.execs[idx] {
                r.ordered = true;
            }
        }
        if ticks >= start + RETREAT_DELAY {
            if let Some(p) = self.players.get_mut(&pid) {
                if let Some(a) = p.outgoing.iter_mut().find(|a| a.id == aid) {
                    a.retreated = true;
                }
            }
            if let Exec::Retreat(r) = &mut self.execs[idx] {
                r.active = false;
            }
        }
    }

    fn tick_win(&mut self, idx: usize, ticks: u32) {
        if ticks % 10 != 0 {
            return;
        }
        if self.in_spawn_phase() {
            return;
        }
        let mut alive: Vec<(String, u32)> = self
            .players
            .values()
            .filter(|p| p.is_alive())
            .map(|p| (p.id().to_string(), p.num_tiles()))
            .collect();
        if alive.is_empty() {
            return;
        }
        alive.sort_by(|a, b| b.1.cmp(&a.1));
        let land = self.map.num_land_tiles().saturating_sub(self.map.num_tiles_with_fallout()).max(1);
        let frac = alive[0].1 as f64 / land as f64 * 100.0;
        let elapsed = self.ticks_since_start() / 10;
        let timer_done = self
            .later
            .max_timer
            .map(|m| elapsed >= m * 60)
            .unwrap_or(false);
        if frac > 80.0 || timer_done || elapsed >= 170 * 60 {
            self.winner = Some(alive[0].0.clone());
            if let Exec::WinCheck(w) = &mut self.execs[idx] {
                w.active = false;
            }
        }
    }

    pub fn conquer(&mut self, owner_id: &str, tile: TileRef) {
        if !self.map.is_land(tile) || self.map.is_impassable(tile) {
            return;
        }
        let prev = self.map.owner_id(tile);
        if prev != 0 {
            if let Some(pid) = self.by_small.get(&prev).cloned() {
                if let Some(p) = self.players.get_mut(&pid) {
                    p.last_tile_change = self.ticks;
                    p.tiles.shift_remove(&tile);
                    p.border_tiles.shift_remove(&tile);
                }
            }
        }
        let sid = self.players.get(owner_id).map(|p| p.small_id).unwrap_or(0);
        self.map.set_owner_id(tile, sid);
        if let Some(p) = self.players.get_mut(owner_id) {
            p.tiles.insert(tile);
            p.last_tile_change = self.ticks;
        }
        self.map.set_fallout(tile, false);
        self.update_borders(tile);
        self.record_tile_update(tile);
    }

    pub(crate) fn relinquish_tile(&mut self, player_id: &str, tile: TileRef) {
        let Some(p) = self.players.get_mut(player_id) else {
            return;
        };
        p.tiles.shift_remove(&tile);
        p.border_tiles.shift_remove(&tile);
        p.last_tile_change = self.ticks;
        self.map.set_owner_id(tile, 0);
        self.update_borders(tile);
        self.record_tile_update(tile);
    }

    pub(crate) fn record_tile_update(&mut self, tile: TileRef) {
        self.tile_update_pairs.push(tile);
        self.tile_update_pairs.push(self.map.tile_state(tile) as u32);
    }

    pub fn drain_packed_tile_updates(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.tile_update_pairs)
    }

    /// Official `GameImpl.setFallout`: no-op if fallout is already set.
    pub fn set_fallout(&mut self, tile: TileRef, value: bool) {
        if value && self.map.has_owner(tile) {
            return;
        }
        if self.map.has_fallout(tile) {
            return;
        }
        self.map.set_fallout(tile, value);
        self.record_tile_update(tile);
    }

    fn update_borders(&mut self, tile: TileRef) {
        self.update_border_status(tile);
        let mut nbuf = [0u32; 4];
        let n = self.map.neighbors4(tile, &mut nbuf);
        for i in 0..n {
            self.update_border_status(nbuf[i]);
        }
    }

    fn update_border_status(&mut self, t: TileRef) {
        if !self.map.has_owner(t) {
            return;
        }
        let sid = self.map.owner_id(t);
        let is_b = self.map.is_border(t);
        if let Some(pid) = self.by_small.get(&sid).cloned() {
            if let Some(p) = self.players.get_mut(&pid) {
                if is_b {
                    p.border_tiles.insert(t);
                } else {
                    p.border_tiles.shift_remove(&t);
                }
            }
        }
    }

    fn conquer_player(&mut self, conqueror: &str, conquered: &str) {
        if conqueror == conquered {
            return;
        }
        // Official GameImpl.conquerPlayer gold split, then leftover sweeps.
        if let Some(v) = self.players.get(conquered) {
            let skip = v.attacks_sent == 0 && v.player_type() == PlayerType::Human;
            let all = v.gold;
            let captured = if skip {
                0
            } else if matches!(v.player_type(), PlayerType::Bot | PlayerType::Nation) {
                all
            } else {
                all / 2
            };
            if !skip {
                self.add_gold(conqueror, captured);
                if let Some(v) = self.players.get_mut(conquered) {
                    v.gold = 0;
                }
            }
        }
        for _ in 0..10 {
            let remaining: Vec<TileRef> = self
                .players
                .get(conquered)
                .map(|p| p.tiles.iter().copied().collect())
                .unwrap_or_default();
            if remaining.is_empty() {
                break;
            }
            let mut nbuf = [0u32; 4];
            for tile in remaining {
                let n = self.map.neighbors4(tile, &mut nbuf);
                let mut borders_me = false;
                for i in 0..n {
                    if self.map.owner_id(nbuf[i])
                        == self.players.get(conqueror).map(|p| p.small_id).unwrap_or(0)
                    {
                        borders_me = true;
                        break;
                    }
                }
                if borders_me {
                    self.conquer(conqueror, tile);
                    continue;
                }
                let mut other: Option<String> = None;
                for i in 0..n {
                    let oid = self.map.owner_id(nbuf[i]);
                    if oid == 0 {
                        continue;
                    }
                    if let Some(op) = self.by_small.get(&oid).cloned() {
                        if op != conquered && !self.is_allied(&op, conquered) {
                            other = Some(op);
                            break;
                        }
                    }
                }
                if let Some(op) = other {
                    self.conquer(&op, tile);
                }
            }
        }
        *self.pending_kills.entry(conqueror.to_string()).or_default() += 1;
        if self.alive_players().iter().filter(|p| p.id() != conquered).count() <= 1 {
            self.winner = Some(conqueror.to_string());
        }
    }

    fn take_attack_exec(&mut self, idx: usize) -> AttackExec {
        match &mut self.execs[idx] {
            Exec::Attack(a) => AttackExec {
                active: a.active,
                inited: a.inited,
                start_troops: a.start_troops,
                owner_id: a.owner_id.clone(),
                target_id: a.target_id.clone(),
                attack_id: a.attack_id.clone(),
                owner_small: a.owner_small,
                target_small: a.target_small,
                target_is_player: a.target_is_player,
                source_tile: a.source_tile,
                remove_troops: a.remove_troops,
                heap: std::mem::replace(&mut a.heap, FlatHeap::new()),
                rng: a.rng.clone(),
            },
            _ => AttackExec {
                active: false,
                inited: false,
                start_troops: 0.0,
                owner_id: String::new(),
                target_id: None,
                attack_id: None,
                owner_small: 0,
                target_small: 0,
                target_is_player: false,
                source_tile: None,
                remove_troops: true,
                heap: FlatHeap::new(),
                rng: PseudoRandom::new(123),
            },
        }
    }

    fn put_attack_exec(&mut self, idx: usize, tmp: AttackExec) {
        if let Exec::Attack(a) = &mut self.execs[idx] {
            a.heap = tmp.heap;
            a.rng = tmp.rng;
            a.active = tmp.active;
        }
    }

    pub fn structure_affordable(&self, player_id: &str) -> bool {
        self.players
            .get(player_id)
            .map(|p| p.gold >= cheapest_structure_cost())
            .unwrap_or(false)
    }

    pub fn game_id(&self) -> &str {
        &self.game_id
    }

    pub fn finish_setup(&mut self) {
        if !self.execs.iter().any(|e| matches!(e, Exec::WinCheck(_)))
            && !self.uninit.iter().any(|e| matches!(e, Exec::WinCheck(_)))
        {
            self.uninit.push(Exec::WinCheck(WinCheckExec {
                active: true,
                inited: false,
            }));
        }
        if !self.later.unit_disabled(UnitType::Factory) {
            self.uninit.push(Exec::Later(LaterExec::RailRecompute { active: true }));
        }
        if self.later.doomsday_enabled {
            self.uninit.push(Exec::Later(LaterExec::Doomsday { active: true }));
        }
    }

    pub(crate) fn construction_exec(pid: &str, ty: UnitType, tile: TileRef) -> ConstructionExec {
        ConstructionExec {
            active: true,
            inited: false,
            player_id: pid.to_string(),
            ty,
            tile,
            unit_id: None,
            ticks_left: 0,
        }
    }

    pub fn apply_env(&mut self, cfg: &crate::spec::EnvConfig) {
        self.later.starting_gold = cfg.starting_gold.unwrap_or(0);
        self.later.max_timer = cfg.max_timer_value;
        self.later.doomsday_enabled = cfg.doomsday_enabled;
        self.later.doomsday_speed = cfg.doomsday_speed.clone();
        self.later.water_nukes = cfg.water_nukes;
        self.later.disabled = cfg
            .disabled_units
            .iter()
            .filter_map(|n| UnitType::from_official_name(n))
            .collect();
        if self.later.starting_gold > 0 {
            for p in self.players.values_mut() {
                if p.player_type() != crate::spec::PlayerType::Bot {
                    p.gold = self.later.starting_gold;
                }
            }
        }
    }

    pub(crate) fn push_later(&mut self, e: LaterExec) {
        self.uninit.push(Exec::Later(e));
    }

    pub(crate) fn later_grid_nearby(
        &self,
        tile: TileRef,
        range: u32,
        types: &[UnitType],
    ) -> Vec<(u32, i64)> {
        self.later
            .grid
            .nearby(&self.map, tile, range, types, false, &self.units)
    }

    pub(crate) fn later_has_nearby(&self, tile: TileRef, range: u32, ty: UnitType) -> bool {
        self.later
            .grid
            .has_nearby(&self.map, tile, range, ty, false, &self.units)
    }

    pub fn is_unit_disabled(&self, ty: UnitType) -> bool {
        self.later.unit_disabled(ty)
    }

    pub(crate) fn move_unit(&mut self, id: u32, tile: TileRef) {
        if let Some(u) = self.units.iter_mut().find(|u| u.id == id) {
            u.last_tile = u.tile;
            u.tile = tile;
        }
        if let Some(u) = self.units.iter().find(|u| u.id == id).cloned() {
            self.later.grid.update_cell(&self.map, &u);
        }
    }

    pub(crate) fn add_gold(&mut self, player_id: &str, amount: i64) {
        if let Some(p) = self.players.get_mut(player_id) {
            p.gold = p.gold.saturating_add(amount);
        }
    }

    pub(crate) fn upgrade_structure_unit(&mut self, player_id: &str, unit_id: u32, ty: UnitType) {
        let cost = self.structure_cost(player_id, ty);
        self.add_gold(player_id, -cost);
        if let Some(p) = self.players.get_mut(player_id) {
            p.record_constructed(ty);
        }
        if let Some(u) = self.units.iter_mut().find(|u| u.id == unit_id) {
            u.level = u.level.saturating_add(1);
        }
    }

    pub(crate) fn swap_later(&mut self, idx: usize, repl: LaterExec) -> Option<LaterExec> {
        match std::mem::replace(&mut self.execs[idx], Exec::Later(repl)) {
            Exec::Later(e) => Some(e),
            other => {
                self.execs[idx] = other;
                None
            }
        }
    }

    pub(crate) fn water_step(&mut self, from: TileRef, to: TileRef) -> (PathStatus, Option<TileRef>) {
        let mut stepper = PathStepper::new();
        stepper.next(&mut self.water_chain, &self.map, &self.mini_map, from, to)
    }

    pub(crate) fn water_step_with(
        &mut self,
        stepper: &mut PathStepper,
        from: TileRef,
        to: TileRef,
    ) -> (PathStatus, Option<TileRef>) {
        stepper.next(&mut self.water_chain, &self.map, &self.mini_map, from, to)
    }

    pub(crate) fn can_trade(&self, a: &str, b: &str) -> bool {
        if a == b {
            return true;
        }
        let Some(pa) = self.players.get(a) else {
            return false;
        };
        !pa.has_embargo_against(b)
            && self
                .players
                .get(b)
                .map(|pb| !pb.has_embargo_against(a))
                .unwrap_or(false)
    }

    pub fn queue_nation(&mut self, spec: NationSpec) {
        if !self.players.contains_key(&spec.id) {
            self.add_player(PlayerInfo {
                name: spec.name.clone(),
                player_type: PlayerType::Nation,
                client_id: None,
                id: spec.id.clone(),
            });
        }
        self.uninit.push(Exec::Nation(NationExec {
            active: true,
            inited: false,
            player_id: spec.id.clone(),
            spawn_x: spec.spawn_x,
            spawn_y: spec.spawn_y,
            attack_rate: 0,
            attack_tick: 0,
            trigger_ratio: 0.0,
            reserve_ratio: 0.0,
            expand_ratio: 0.0,
            behaviors: false,
            spawn_added: false,
            rng: PseudoRandom::new(0),
        }));
    }

    pub fn queue_bot_spawn(&mut self, info: PlayerInfo) {
        let seed = simple_hash(&info.id).wrapping_add(simple_hash(&self.game_id));
        self.uninit.push(Exec::Spawn(SpawnExec {
            active: true,
            player_id: info.id.clone(),
            game_id: self.game_id.clone(),
            tile: None,
            info: Some(info),
            rng: PseudoRandom::new(seed),
        }));
    }

    fn queue_tribe_exec(&mut self, player_id: &str) {
        let mut rng = PseudoRandom::new(simple_hash(player_id));
        let attack_rate = rng.next_int(40.0, 80.0) as u32;
        let attack_tick = rng.next_int(0.0, attack_rate as f64) as u32;
        self.uninit.push(Exec::Tribe(TribeExec {
            active: true,
            inited: false,
            player_id: player_id.to_string(),
            attack_rate,
            attack_tick,
            trigger_ratio: rng.next_int(50.0, 60.0) as f64 / 100.0,
            reserve_ratio: rng.next_int(30.0, 40.0) as f64 / 100.0,
            expand_ratio: rng.next_int(10.0, 20.0) as f64 / 100.0,
            behavior: false,
            neighbors_tn: true,
            rng,
        }));
    }

    fn random_spawn_center(&mut self, rng: &mut PseudoRandom, player_id: &str) -> Option<TileRef> {
        for _ in 0..1000 {
            let x = rng.next_int(0.0, self.map.width() as f64) as u32;
            let y = rng.next_int(0.0, self.map.height() as f64) as u32;
            let Ok(center) = self.map.ref_xy(x, y) else {
                continue;
            };
            if !self.map.is_land(center) || self.map.has_owner(center) || self.map.is_border(center)
            {
                continue;
            }
            let too_close = self.players.values().any(|p| {
                if p.id() == player_id {
                    return false;
                }
                p.spawn_tile
                    .map(|s| self.map.manhattan(s, center) < 30)
                    .unwrap_or(false)
            });
            if too_close {
                continue;
            }
            if spawn_tiles(&self.map, center, true).is_empty() {
                continue;
            }
            return Some(center);
        }
        None
    }

    pub(crate) fn apply_nuke_blast_relations(&mut self, owner: &str, dst: TileRef, ty: UnitType) {
        if ty == UnitType::MirvWarhead {
            return;
        }
        let (inner, outer) = crate::config::nuke_magnitudes(ty);
        let inner2 = inner as i64 * inner as i64;
        let outer2 = outer as i64 * outer as i64;
        let cx = self.map.x(dst) as i32;
        let cy = self.map.y(dst) as i32;
        let r = outer as i32;
        let mut weights: HashMap<u16, f64> = HashMap::new();
        for y in (cy - r).max(0)..=cy + r {
            for x in (cx - r).max(0)..=cx + r {
                let Ok(t) = self.map.ref_xy(x as u32, y as u32) else {
                    continue;
                };
                let dx = self.map.x(t) as i64 - self.map.x(dst) as i64;
                let dy = self.map.y(t) as i64 - self.map.y(dst) as i64;
                let d2 = dx * dx + dy * dy;
                if d2 > outer2 {
                    continue;
                }
                let oid = self.map.owner_id(t);
                if oid == 0 {
                    continue;
                }
                let w = if d2 <= inner2 { 1.0 } else { 0.5 };
                *weights.entry(oid).or_insert(0.0) += w;
            }
        }
        let mut break_sids: HashSet<u16> = weights
            .into_iter()
            .filter(|(_, w)| *w > 100.0)
            .map(|(id, _)| id)
            .collect();
        for u in &self.units {
            if !u.active || !u.ty.is_structure() || u.under_construction {
                continue;
            }
            let dx = self.map.x(u.tile) as i64 - self.map.x(dst) as i64;
            let dy = self.map.y(u.tile) as i64 - self.map.y(dst) as i64;
            if dx * dx + dy * dy <= outer2 {
                break_sids.insert(u.owner_small);
            }
        }
        let attacked: Vec<String> = break_sids
            .iter()
            .filter_map(|&sid| self.player_by_small(sid).map(|p| p.id().to_string()))
            .collect();
        for tid in &attacked {
            self.reject_incoming_from(owner, tid);
        }
        let ticks = self.ticks;
        for tid in attacked {
            if tid == owner {
                continue;
            }
            let pending_from_launcher = self.incoming_requests.iter().any(|r| {
                r.recipient == tid
                    && r.requestor == owner
                    && r.status == RequestStatus::Pending
            });
            if pending_from_launcher {
                self.reject_incoming_from(&tid, owner);
                continue;
            }
            if let Some(al) = self.alliances.iter().find(|a| a.involves(owner, &tid)).cloned() {
                self.break_alliance(&al, owner, ticks);
            }
            if let Some(p) = self.players.get_mut(&tid) {
                p.update_relation(owner, -100.0);
            }
        }
    }

    fn reject_incoming_from(&mut self, me: &str, other: &str) {
        let mut reject = None;
        for (i, r) in self.incoming_requests.iter().enumerate() {
            if r.recipient == me && r.requestor == other && r.status == RequestStatus::Pending {
                reject = Some(i);
                break;
            }
        }
        if let Some(i) = reject {
            let req = self.incoming_requests[i].clone();
            self.incoming_requests[i].status = RequestStatus::Rejected;
            if let Some(p) = self.players.get_mut(&req.requestor) {
                p.past_outgoing_requests.push(req);
            }
        }
    }

    fn has_defense_post(&self, owner_id: &str, tile: TileRef) -> bool {
        let r2 = (DEFENSE_POST_RANGE * DEFENSE_POST_RANGE) as i64;
        self.units.iter().any(|u| {
            u.active
                && !u.under_construction
                && u.ty == UnitType::DefensePost
                && u.owner_id == owner_id
                && {
                    let dx = self.map.x(u.tile) as i64 - self.map.x(tile) as i64;
                    let dy = self.map.y(u.tile) as i64 - self.map.y(tile) as i64;
                    dx * dx + dy * dy <= r2
                }
        })
    }

    pub fn boat_count(&self, player_id: &str) -> u32 {
        self.units
            .iter()
            .filter(|u| u.active && u.ty == UnitType::TransportShip && u.owner_id == player_id)
            .count() as u32
    }

    pub fn has_port(&self, player_id: &str) -> bool {
        // Official obs g[8] is `me.units(Port).length > 0`, including
        // under-construction ports.
        self.units
            .iter()
            .any(|u| u.active && u.ty == UnitType::Port && u.owner_id == player_id)
    }

    pub fn structure_cost(&self, player_id: &str, ty: UnitType) -> i64 {
        if ty == UnitType::Mirv {
            return crate::config::mirv_cost(self.later.num_mirvs);
        }
        let Some(p) = self.players.get(player_id) else {
            return i64::MAX;
        };
        let types: &[UnitType] = match ty {
            UnitType::Port | UnitType::Factory => &[UnitType::Port, UnitType::Factory],
            _ => std::slice::from_ref(&ty),
        };
        let n: u32 = types
            .iter()
            .map(|t| {
                let owned = structure_count(&self.units, player_id, *t);
                owned.min(p.units_constructed(*t))
            })
            .sum();
        unit_cost(ty, n)
    }

    pub fn can_build(&mut self, player_id: &str, ty: UnitType, tile: TileRef) -> Option<TileRef> {
        if self.later.unit_disabled(ty) {
            return None;
        }
        let Some(p) = self.players.get(player_id) else {
            return None;
        };
        if !p.is_alive() && !matches!(ty, UnitType::MirvWarhead) {
            return None;
        }
        let cost = self.structure_cost(player_id, ty);
        let gold = p.gold;
        if gold < cost {
            return None;
        }
        match ty {
            UnitType::TransportShip => self.can_build_boat(player_id, tile),
            UnitType::Port => self.port_spawn(player_id, tile),
            UnitType::City
            | UnitType::DefensePost
            | UnitType::SamLauncher
            | UnitType::MissileSilo
            | UnitType::Factory => self.land_structure_spawn(player_id, tile),
            UnitType::Warship => self.warship_spawn(player_id, tile),
            UnitType::Mirv => {
                if !self.map.has_owner(tile) {
                    return None;
                }
                self.nuke_spawn(player_id, tile)
            }
            UnitType::AtomBomb | UnitType::HydrogenBomb => self.nuke_spawn(player_id, tile),
            UnitType::MirvWarhead => Some(tile),
            UnitType::TradeShip => self.land_structure_spawn(player_id, tile).or(Some(tile)),
            UnitType::Train => {
                if self.map.is_land(tile) && !self.map.is_impassable(tile) {
                    Some(tile)
                } else {
                    None
                }
            }
            _ => None,
        }
    }

    fn can_build_boat(&mut self, player_id: &str, tile: TileRef) -> Option<TileRef> {
        let Some(p) = self.players.get(player_id) else {
            return None;
        };
        let owner_small = p.small_id;
        let shores = p.shore_tiles(&self.map);
        let border: Vec<TileRef> = p.border_tiles.iter().copied().collect();
        let boats = self.boat_count(player_id);
        let target_sid = self.map.owner_id(tile);
        let allowed = if target_sid == 0 {
            true
        } else if let Some(other) = self.player_by_small(target_sid) {
            self.can_attack_player(p, other)
        } else {
            true
        };
        can_build_transport_ship(
            &mut self.water_chain,
            &self.map,
            &self.mini_map,
            &self.mini_water,
            boats,
            owner_small,
            &shores,
            border.into_iter(),
            tile,
            |_| allowed,
        )
    }

    fn land_structure_spawn(&self, player_id: &str, tile: TileRef) -> Option<TileRef> {
        self.valid_structure_tiles(player_id, tile).first().copied()
    }

    fn warship_spawn(&self, player_id: &str, tile: TileRef) -> Option<TileRef> {
        if !self.map.is_water(tile) {
            return None;
        }
        let comp = self.water.of_tile(&self.map, tile);
        let mut best: Option<(u32, TileRef)> = None;
        for u in &self.units {
            if !u.active || u.under_construction || u.ty != UnitType::Port || u.owner_id != player_id
            {
                continue;
            }
            let pcomp = self.water.of_tile(&self.map, u.tile);
            if comp.is_none() || pcomp != comp {
                continue;
            }
            let d = self.map.manhattan(u.tile, tile);
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, u.tile));
            }
        }
        best.map(|(_, t)| t)
    }

    fn nuke_spawn(&self, player_id: &str, tile: TileRef) -> Option<TileRef> {
        if self.spawn_immunity_active() {
            return None;
        }
        if self.map.is_impassable(tile) {
            return None;
        }
        let mut silos: Vec<(u32, TileRef)> = self
            .units
            .iter()
            .filter(|u| {
                u.active
                    && u.ty == UnitType::MissileSilo
                    && u.owner_id == player_id
                    && !u.under_construction
                    && u.missile_queue.len() != u.level as usize
            })
            .map(|u| (self.map.manhattan(u.tile, tile), u.tile))
            .collect();
        silos.sort_by_key(|(d, _)| *d);
        silos.first().map(|(_, t)| *t)
    }

    fn port_spawn(&self, player_id: &str, tile: TileRef) -> Option<TileRef> {
        let Some(p) = self.players.get(player_id) else {
            return None;
        };
        let sid = p.small_id;
        let mut shores: Vec<TileRef> = self
            .map
            .bfs_manhattan(tile, crate::units::RADIUS_PORT_SPAWN)
            .into_iter()
            .filter(|&t| self.map.owner_id(t) == sid && self.map.is_shore(t))
            .collect();
        shores.sort_by_key(|&t| self.map.manhattan(t, tile));
        let valid: HashSet<TileRef> = self.valid_structure_tiles(player_id, tile).into_iter().collect();
        shores.into_iter().find(|t| valid.contains(t))
    }

    fn valid_structure_tiles(&self, player_id: &str, tile: TileRef) -> Vec<TileRef> {
        let Some(p) = self.players.get(player_id) else {
            return Vec::new();
        };
        if self.map.owner_id(tile) != p.small_id {
            return Vec::new();
        }
        let sid = p.small_id;
        let search_r2 = 15i64 * 15;
        let euclid2 = |a: TileRef, b: TileRef| -> i64 {
            let dx = self.map.x(a) as i64 - self.map.x(b) as i64;
            let dy = self.map.y(a) as i64 - self.map.y(b) as i64;
            dx * dx + dy * dy
        };
        // Official `validStructureSpawnTiles`: BFS only through owned tiles
        // with integer euclid d2 < 15^2 (not the center-offset radius).
        let mut seen: HashSet<TileRef> = HashSet::new();
        let mut stack = Vec::new();
        let mut nearby = Vec::new();
        if euclid2(tile, tile) < search_r2 {
            seen.insert(tile);
            stack.push(tile);
            nearby.push(tile);
        }
        let mut nbuf = [0u32; 4];
        while let Some(curr) = stack.pop() {
            let n = self.map.neighbors4(curr, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if seen.contains(&nb) {
                    continue;
                }
                if self.map.owner_id(nb) == sid && euclid2(tile, nb) < search_r2 {
                    seen.insert(nb);
                    stack.push(nb);
                    nearby.push(nb);
                }
            }
        }
        let min_d2 = (STRUCTURE_MIN_DIST * STRUCTURE_MIN_DIST) as i64;
        let structs: Vec<TileRef> = self
            .units
            .iter()
            .filter(|u| u.active && u.ty.is_structure())
            .map(|u| u.tile)
            .collect();
        let mut valid: Vec<TileRef> = nearby
            .into_iter()
            .filter(|&t| structs.iter().all(|&s| euclid2(s, t) >= min_d2))
            .collect();
        valid.sort_by_key(|&t| euclid2(t, tile));
        valid
    }

    pub(crate) fn build_unit(&mut self, player_id: &str, ty: UnitType, tile: TileRef, troops: f64) -> Option<u32> {
        let cost = self.structure_cost(player_id, ty);
        let Some(p) = self.players.get_mut(player_id) else {
            return None;
        };
        p.remove_gold(cost);
        p.record_constructed(ty);
        let sid = p.small_id;
        let id = self.next_unit_id;
        self.next_unit_id += 1;
        let u = Unit::new(id, ty, sid, player_id.to_string(), tile, troops, None);
        p.units.push(id);
        self.units.push(u);
        if let Some(built) = self.units.last() {
            self.later.grid.add(&self.map, built);
        }
        Some(id)
    }

    fn init_boat(&mut self, b: &mut BoatExec, ticks: u32) {
        b.inited = true;
        b.last_move = ticks;
        if self.boat_count(&b.owner_id) >= BOAT_MAX_NUMBER {
            b.active = false;
            return;
        }
        let target_sid = self.map.owner_id(b.ref_tile);
        let (owner_small, owner_type, troops_avail, shores, border, allowed, reject_tid) = {
            let Some(owner) = self.players.get(&b.owner_id) else {
                b.active = false;
                return;
            };
            if target_sid == owner.small_id {
                b.active = false;
                return;
            }
            let allowed = if target_sid == 0 {
                true
            } else if let Some(target) = self.player_by_small(target_sid) {
                self.can_attack_player(owner, target)
            } else {
                true
            };
            let reject_tid = if target_sid != 0 {
                self.player_by_small(target_sid).and_then(|target| {
                    if owner.player_type() != PlayerType::Bot && target.player_type() != PlayerType::Bot {
                        Some(target.id().to_string())
                    } else {
                        None
                    }
                })
            } else {
                None
            };
            (
                owner.small_id,
                owner.player_type(),
                owner.troops_f64(),
                owner.shore_tiles(&self.map),
                owner.border_tiles.iter().copied().collect::<Vec<_>>(),
                allowed,
                reject_tid,
            )
        };
        if !allowed {
            b.active = false;
            return;
        }
        let _ = owner_type;
        if let Some(tid) = reject_tid {
            self.reject_incoming_from(&b.owner_id, &tid);
        }
        let troops = b.troops.min(troops_avail).max(0.0);
        b.troops = troops;
        let dst = target_transport_tile(
            &self.map,
            &self.mini_map,
            &self.mini_water,
            target_sid,
            border.iter().copied(),
            b.ref_tile,
        );
        let Some(dst) = dst else {
            b.active = false;
            return;
        };
        let boats = self.boat_count(&b.owner_id);
        let src = can_build_transport_ship(
            &mut self.water_chain,
            &self.map,
            &self.mini_map,
            &self.mini_water,
            boats,
            owner_small,
            &shores,
            border.into_iter(),
            dst,
            |_| true,
        );
        let Some(src) = src else {
            b.active = false;
            return;
        };
        if let Some(p) = self.players.get_mut(&b.owner_id) {
            p.remove_troops(troops.floor() as i64);
        }
        let uid = self.build_unit(&b.owner_id, UnitType::TransportShip, src, troops);
        b.dst = Some(dst);
        b.src = Some(src);
        b.unit_id = uid;
        if let Some(u) = self.units.iter_mut().find(|u| Some(u.id) == uid) {
            u.target_tile = Some(dst);
        }
    }

    fn tick_boat(&mut self, idx: usize, ticks: u32) {
        let (owner_id, dst, unit_id, last_move) = {
            let Exec::Boat(b) = &self.execs[idx] else {
                return;
            };
            if !b.active || !b.inited {
                return;
            }
            (b.owner_id.clone(), b.dst, b.unit_id, b.last_move)
        };
        let Some(dst) = dst else {
            if let Exec::Boat(b) = &mut self.execs[idx] {
                b.active = false;
            }
            return;
        };
        let Some(uid) = unit_id else {
            if let Exec::Boat(b) = &mut self.execs[idx] {
                b.active = false;
            }
            return;
        };
        let Some(boat) = self.units.iter().find(|u| u.id == uid && u.active) else {
            if let Exec::Boat(b) = &mut self.execs[idx] {
                b.active = false;
            }
            return;
        };
        if ticks.saturating_sub(last_move) < 1 {
            return;
        }
        let from = boat.tile;
        let troops = boat.troops;
        let (status, node) = {
            let Exec::Boat(b) = &mut self.execs[idx] else {
                return;
            };
            b.last_move = ticks;
            b.stepper
                .next(&mut self.water_chain, &self.map, &self.mini_map, from, dst)
        };
        match status {
            PathStatus::Complete => {
                if self.map.owner_id(dst) == self.players.get(&owner_id).map(|p| p.small_id).unwrap_or(0)
                {
                    if let Some(p) = self.players.get_mut(&owner_id) {
                        p.add_troops(troops - troops * (MALUS_FOR_RETREAT / 100.0));
                    }
                    self.delete_unit(uid);
                    if let Exec::Boat(b) = &mut self.execs[idx] {
                        b.active = false;
                    }
                    return;
                }
                self.conquer(&owner_id, dst);
                let target_id = {
                    let sid = self.map.owner_id(dst);
                    if sid == 0 {
                        None
                    } else {
                        self.by_small.get(&sid).cloned()
                    }
                };
                // Landing may have conquered dest; attack the previous owner if still a player tile neighbor.
                let prev_target = {
                    let mut nbuf = [0u32; 4];
                    let n = self.map.neighbors4(dst, &mut nbuf);
                    let mut tid = target_id;
                    if tid.as_deref() == Some(owner_id.as_str()) {
                        tid = None;
                    }
                    for i in 0..n {
                        let os = self.map.owner_id(nbuf[i]);
                        if os != 0 && os != self.players.get(&owner_id).map(|p| p.small_id).unwrap_or(0)
                        {
                            tid = self.by_small.get(&os).cloned();
                            break;
                        }
                    }
                    tid
                };
                self.uninit.push(Exec::Attack(AttackExec {
                    active: true,
                    inited: false,
                    start_troops: troops,
                    owner_id: owner_id.clone(),
                    target_id: prev_target.clone(),
                    attack_id: None,
                    owner_small: 0,
                    target_small: 0,
                    target_is_player: prev_target.is_some(),
                    source_tile: Some(dst),
                    remove_troops: false,
                    heap: FlatHeap::new(),
                    rng: PseudoRandom::new(123),
                }));
                self.delete_unit(uid);
                if let Exec::Boat(b) = &mut self.execs[idx] {
                    b.active = false;
                }
            }
            PathStatus::Next => {
                if let Some(n) = node {
                    self.move_unit(uid, n);
                }
            }
            PathStatus::NotFound => {
                if let Some(p) = self.players.get_mut(&owner_id) {
                    p.add_troops(troops);
                }
                self.delete_unit(uid);
                if let Exec::Boat(b) = &mut self.execs[idx] {
                    b.active = false;
                }
            }
        }
    }

    pub(crate) fn capture_unit(&mut self, new_owner: &str, id: u32) {
        let Some(u) = self.units.iter().find(|u| u.id == id && u.active) else {
            return;
        };
        let old = u.owner_id.clone();
        if old == new_owner {
            return;
        }
        let Some(np) = self.players.get(new_owner) else {
            return;
        };
        let sid = np.small_id;
        if let Some(op) = self.players.get_mut(&old) {
            op.units.retain(|&x| x != id);
        }
        if let Some(np) = self.players.get_mut(new_owner) {
            np.units.push(id);
        }
        if let Some(u) = self.units.iter_mut().find(|u| u.id == id) {
            u.owner_id = new_owner.to_string();
            u.owner_small = sid;
        }
    }

    pub(crate) fn delete_unit(&mut self, id: u32) {
        self.delete_unit_by(id, None);
    }

    pub(crate) fn delete_unit_by(&mut self, id: u32, destroyer: Option<&str>) {
        let owner = self
            .units
            .iter()
            .find(|u| u.id == id && u.active)
            .map(|u| u.owner_id.clone());
        if let Some(u) = self.units.iter().find(|u| u.id == id) {
            self.later.grid.remove(&self.map, u);
            self.later.rail.remove_unit(id);
        }
        if let Some(u) = self.units.iter_mut().find(|u| u.id == id) {
            u.active = false;
            if let Some(d) = destroyer {
                u.destroyer_id = Some(d.to_string());
            }
        }
        if let Some(owner) = owner {
            if let Some(p) = self.players.get_mut(&owner) {
                p.units.retain(|&uid| uid != id);
            }
        }
    }

    fn tick_construction(&mut self, idx: usize, _ticks: u32) {
        let (pid, ty, tile, unit_id, left) = {
            let Exec::Construction(c) = &self.execs[idx] else {
                return;
            };
            if !c.active {
                return;
            }
            (c.player_id.clone(), c.ty, c.tile, c.unit_id, c.ticks_left)
        };
        if !ty.is_structure() {
            match ty {
                UnitType::Warship => self.launch_warship(pid, tile),
                UnitType::Mirv => self.launch_mirv(pid, tile),
                UnitType::AtomBomb | UnitType::HydrogenBomb => {
                    self.launch_nuke(pid, ty, tile, true)
                }
                _ => {}
            }
            if let Exec::Construction(c) = &mut self.execs[idx] {
                c.active = false;
            }
            return;
        }
        if unit_id.is_none() {
            let spawn = self.can_build(&pid, ty, tile);
            let Some(spawn) = spawn else {
                if let Exec::Construction(c) = &mut self.execs[idx] {
                    c.active = false;
                }
                return;
            };
            let Some(uid) = self.build_unit(&pid, ty, spawn, 0.0) else {
                if let Exec::Construction(c) = &mut self.execs[idx] {
                    c.active = false;
                }
                return;
            };
            let dur = construction_duration(ty);
            if let Some(u) = self.units.iter_mut().find(|u| u.id == uid) {
                u.under_construction = dur > 0;
            }
            if let Exec::Construction(c) = &mut self.execs[idx] {
                c.unit_id = Some(uid);
                c.ticks_left = dur;
            }
            if dur == 0 {
                self.complete_structure(uid, ty);
                if let Exec::Construction(c) = &mut self.execs[idx] {
                    c.active = false;
                }
            }
            return;
        }
        let uid = unit_id.unwrap();
        let still = self.units.iter().any(|u| u.id == uid && u.active);
        if !still {
            if let Exec::Construction(c) = &mut self.execs[idx] {
                c.active = false;
            }
            return;
        }
        if left == 0 {
            self.complete_structure(uid, ty);
            if let Exec::Construction(c) = &mut self.execs[idx] {
                c.active = false;
            }
            return;
        }
        if let Exec::Construction(c) = &mut self.execs[idx] {
            c.ticks_left = left.saturating_sub(1);
        }
    }

    fn complete_structure(&mut self, uid: u32, ty: UnitType) {
        if let Some(u) = self.units.iter_mut().find(|u| u.id == uid) {
            u.under_construction = false;
        }
        self.complete_later_structure(uid, ty);
    }

    fn tick_alliance_req(&mut self, idx: usize, ticks: u32) {
        let Exec::AllianceReq(a) = &self.execs[idx] else {
            return;
        };
        let requestor = a.requestor.clone();
        let recipient = a.recipient.clone();
        if !self.players.contains_key(&recipient) {
            if let Exec::AllianceReq(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        }
        let incoming = self.incoming_requests.iter().find(|r| {
            r.requestor == recipient && r.recipient == requestor && r.status == RequestStatus::Pending
        });
        if incoming.is_some() {
            if let Some(p) = self.players.get_mut(&requestor) {
                p.update_relation(&recipient, 100.0);
            }
            if let Some(p) = self.players.get_mut(&recipient) {
                p.update_relation(&requestor, 100.0);
            }
            self.accept_alliance(&recipient, &requestor, ticks);
            if let Exec::AllianceReq(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        }
        let can = {
            let Some(req) = self.players.get(&requestor) else {
                return;
            };
            let Some(rec) = self.players.get(&recipient) else {
                return;
            };
            let last = req
                .past_outgoing_requests
                .iter()
                .filter(|r| r.recipient == recipient)
                .map(|r| r.created_at)
                .max();
            can_send_alliance_request(
                ticks,
                rec.is_alive(),
                req.disconnected,
                rec.disconnected,
                self.is_friendly(req, rec),
                req.outgoing_requests
                    .iter()
                    .any(|r| r.recipient == recipient && r.status == RequestStatus::Pending),
                self.incoming_requests.iter().any(|r| {
                    r.recipient == requestor && r.requestor == recipient && r.status == RequestStatus::Pending
                }),
                last,
            )
        };
        if !can {
            if let Exec::AllianceReq(a) = &mut self.execs[idx] {
                a.active = false;
            }
            return;
        }
        let req = AllianceRequest {
            requestor: requestor.clone(),
            recipient: recipient.clone(),
            created_at: ticks,
            status: RequestStatus::Pending,
        };
        if let Some(p) = self.players.get_mut(&requestor) {
            p.outgoing_requests.push(req.clone());
        }
        self.incoming_requests.push(req);
    }

    fn tick_break_ally(&mut self, idx: usize, ticks: u32) {
        let Exec::BreakAlly(b) = &self.execs[idx] else {
            return;
        };
        let requestor = b.requestor.clone();
        let recipient = b.recipient.clone();
        if let Some(al) = self
            .alliances
            .iter()
            .find(|a| a.involves(&requestor, &recipient))
            .cloned()
        {
            self.break_alliance(&al, &requestor, ticks);
            if let Some(rec) = self.players.get_mut(&recipient) {
                rec.update_relation(&requestor, -100.0);
            }
            let neighbors: Vec<String> = self
                .nearby_player_ids(&requestor)
                .into_iter()
                .filter(|id| id != &recipient)
                .collect();
            for n in neighbors {
                if let Some(p) = self.players.get_mut(&n) {
                    p.update_relation(&requestor, -40.0);
                }
            }
        }
        if let Exec::BreakAlly(b) = &mut self.execs[idx] {
            b.active = false;
        }
    }

    fn tick_embargo(&mut self, idx: usize, ticks: u32) {
        let Exec::Embargo(e) = &self.execs[idx] else {
            return;
        };
        let pid = e.player_id.clone();
        let tid = e.target_id.clone();
        let start = e.start;
        if let Some(p) = self.players.get_mut(&pid) {
            if start {
                p.add_embargo(&tid, ticks, false);
            } else {
                p.stop_embargo(&tid);
            }
        }
        if let Exec::Embargo(e) = &mut self.execs[idx] {
            e.active = false;
        }
    }

    fn accept_alliance(&mut self, a: &str, b: &str, ticks: u32) {
        let past = self
            .incoming_requests
            .iter()
            .find(|r| r.involves_ids(a, b) && r.status == RequestStatus::Pending)
            .cloned();
        for r in &mut self.incoming_requests {
            if r.involves_ids(a, b) {
                r.status = RequestStatus::Accepted;
            }
        }
        // Official acceptAllianceRequest creates the alliance only
        // and records the request on the requestor's pastOutgoing list
        // so canSendAllianceRequest cooldown starts from createdAt.
        if let Some(req) = past {
            if let Some(p) = self.players.get_mut(&req.requestor) {
                p.past_outgoing_requests.push(req.clone());
                p.outgoing_requests.retain(|r| {
                    !(r.requestor == req.requestor && r.recipient == req.recipient)
                });
            }
        }
        if let Some(p) = self.players.get_mut(a) {
            p.end_temporary_embargo(b);
        }
        if let Some(p) = self.players.get_mut(b) {
            p.end_temporary_embargo(a);
        }
        self.alliances.push(Alliance {
            id: self.next_alliance_id,
            requestor: a.to_string(),
            recipient: b.to_string(),
            created_at: ticks,
            expires_at: alliance_expires_at(ticks),
            ext_requestor: false,
            ext_recipient: false,
        });
        self.next_alliance_id += 1;
    }

    pub(crate) fn break_alliance(&mut self, al: &Alliance, breaker: &str, ticks: u32) {
        self.alliances.retain(|x| x.id != al.id);
        if let Some(p) = self.players.get_mut(breaker) {
            p.traitor = true;
            p.marked_traitor_tick = ticks as i32;
        }
    }

    fn expire_alliances_for(&mut self, pid: &str) {
        let ticks = self.ticks;
        let expired: Vec<u32> = self
            .alliances
            .iter()
            .filter(|a| a.involves(pid, a.other(pid)) && ticks >= a.expires_at)
            .map(|a| a.id)
            .collect();
        self.alliances.retain(|a| !expired.contains(&a.id));
        self.incoming_requests.retain(|r| {
            if r.recipient != pid && r.requestor != pid {
                return true;
            }
            if r.status != RequestStatus::Pending {
                return false;
            }
            !request_expired(ticks, r.created_at)
        });
        if let Some(p) = self.players.get_mut(pid) {
            if p.traitor && p.marked_traitor_tick >= 0 {
                if ticks.saturating_sub(p.marked_traitor_tick as u32) >= TRAITOR_DURATION {
                    p.traitor = false;
                    p.marked_traitor_tick = -1;
                }
            }
            p.embargoes.retain(|_, e| {
                !e.temporary
                    || ticks.saturating_sub(e.created_at) < crate::units::TEMPORARY_EMBARGO_DURATION
            });
        }
    }

    fn capture_or_delete_structures(&mut self, pid: &str) {
        let sid = self.players.get(pid).map(|p| p.small_id).unwrap_or(0);
        let mut dead = Vec::new();
        for u in &mut self.units {
            if !u.active || u.owner_id != pid || !u.ty.is_structure() {
                continue;
            }
            let owner = self.map.owner_id(u.tile);
            if owner == 0 {
                u.active = false;
                dead.push(u.id);
            } else if owner != sid {
                if let Some(new_id) = self.by_small.get(&owner).cloned() {
                    u.owner_id = new_id;
                    u.owner_small = owner;
                }
            }
        }
        let _ = dead;
    }

    /// Official `Player.nearby()`: land-border owners plus shore-reachable
    /// owners (every 10th shore, 5 tiles out). `None` is TerraNullius.
    /// Insertion order matches the official Set.
    pub fn nearby(&self, player_id: &str) -> Vec<Option<String>> {
        let Some(p) = self.players.get(player_id) else {
            return Vec::new();
        };
        let sid = p.small_id;
        let mut ns: IndexSet<Option<String>> = IndexSet::new();
        let mut nbuf = [0u32; 4];
        for &t in &p.border_tiles {
            let n = self.map.neighbors4(t, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if !self.map.is_land(nb) || self.map.is_impassable(nb) {
                    continue;
                }
                if !self.map.has_owner(nb) && self.map.has_fallout(nb) {
                    continue;
                }
                let os = self.map.owner_id(nb);
                if os != sid {
                    ns.insert(self.owner_id_opt(os));
                }
            }
        }
        let mut shore_idx = 0i32;
        let dirs: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        for &border in &p.border_tiles {
            if !self.map.is_shore(border) {
                continue;
            }
            if shore_idx % 10 != 0 {
                shore_idx += 1;
                continue;
            }
            shore_idx += 1;
            let bx = self.map.x(border) as i32;
            let by = self.map.y(border) as i32;
            for (dx, dy) in dirs {
                let x1 = bx + dx;
                let y1 = by + dy;
                if x1 < 0 || y1 < 0 {
                    continue;
                }
                let Ok(w) = self.map.ref_xy(x1 as u32, y1 as u32) else {
                    continue;
                };
                if !self.map.is_water(w) {
                    continue;
                }
                let nx = bx + dx * 5;
                let ny = by + dy * 5;
                if nx < 0 || ny < 0 {
                    continue;
                }
                let Ok(tile) = self.map.ref_xy(nx as u32, ny as u32) else {
                    continue;
                };
                if !self.map.is_land(tile) || self.map.is_impassable(tile) {
                    continue;
                }
                if !self.map.has_owner(tile) && self.map.has_fallout(tile) {
                    continue;
                }
                let os = self.map.owner_id(tile);
                if os != sid {
                    ns.insert(self.owner_id_opt(os));
                }
            }
        }
        ns.into_iter().collect()
    }

    fn owner_id_opt(&self, small: u16) -> Option<String> {
        if small == 0 {
            None
        } else {
            self.by_small.get(&small).cloned()
        }
    }

    pub fn nearby_player_ids(&self, player_id: &str) -> Vec<String> {
        self.nearby(player_id).into_iter().flatten().collect()
    }

    pub fn incoming_from(&self, me: &str, other: &str) -> bool {
        self.incoming_requests.iter().any(|r| {
            r.recipient == me && r.requestor == other && r.status == RequestStatus::Pending
        })
    }

    pub fn can_send_ally(&self, me: &str, other: &str) -> bool {
        let Some(req) = self.players.get(me) else {
            return false;
        };
        let Some(rec) = self.players.get(other) else {
            return false;
        };
        if !req.is_alive() {
            return false;
        }
        if self.incoming_from(me, other) {
            return true;
        }
        let last = req
            .past_outgoing_requests
            .iter()
            .filter(|r| r.recipient == other)
            .map(|r| r.created_at)
            .max();
        can_send_alliance_request(
            self.ticks,
            req.is_alive(),
            req.disconnected,
            rec.disconnected,
            self.is_friendly(req, rec),
            req.outgoing_requests
                .iter()
                .any(|r| r.recipient == other && r.status == RequestStatus::Pending),
            self.incoming_from(me, other),
            last,
        )
    }

    fn tick_tribe(&mut self, idx: usize, ticks: u32) {
        let (pid, rate, atick, trigger, reserve, expand, behaved, neighbors_tn) = {
            let Exec::Tribe(t) = &self.execs[idx] else {
                return;
            };
            if !t.active {
                return;
            }
            (
                t.player_id.clone(),
                t.attack_rate,
                t.attack_tick,
                t.trigger_ratio,
                t.reserve_ratio,
                t.expand_ratio,
                t.behavior,
                t.neighbors_tn,
            )
        };
        if ticks % rate.max(1) != atick {
            return;
        }
        if !self.players.get(&pid).map(|p| p.is_alive()).unwrap_or(false) {
            if let Exec::Tribe(t) = &mut self.execs[idx] {
                t.active = false;
            }
            return;
        }
        if !behaved {
            // Official first tick: AiAttackBehavior.sendAttack(terraNullius).
            let mut rng = {
                let Exec::Tribe(t) = &self.execs[idx] else {
                    return;
                };
                t.rng.clone()
            };
            self.ai_send_attack_official(&pid, None, expand, reserve, &mut rng, false);
            if let Exec::Tribe(t) = &mut self.execs[idx] {
                t.behavior = true;
                t.rng = rng;
            }
            return;
        }
        self.tribe_accept_alliances(&pid, ticks);
        self.tribe_delete_next_structure(&pid);
        self.tribe_maybe_attack(&pid, trigger, reserve, expand, neighbors_tn, idx, ticks);
    }

    fn tribe_delete_next_structure(&mut self, pid: &str) {
        // Official deleteNextStructure: first unmarked structure, if any.
        // Training tribes rarely own structures; keep the scan for parity.
        let Some(uid) = self.units.iter().find(|u| {
            u.active && u.owner_id == pid && u.ty.is_structure() && !u.under_construction
        }).map(|u| u.id.clone()) else {
            return;
        };
        let _ = uid;
    }

    fn tribe_maybe_attack(
        &mut self,
        pid: &str,
        trigger: f64,
        reserve: f64,
        expand: f64,
        neighbors_tn: bool,
        idx: usize,
        ticks: u32,
    ) {
        let mut rng = {
            let Exec::Tribe(t) = &self.execs[idx] else {
                return;
            };
            t.rng.clone()
        };
        if let Some(tid) = self.neighbor_traitor_to_attack(pid, &mut rng) {
            let friendly = self.is_allied(pid, &tid);
            let odds = if friendly { 6 } else { 3 };
            if rng.chance(odds) {
                if friendly {
                    if let Some(al) = self.alliances.iter().find(|a| a.involves(pid, &tid)).cloned() {
                        self.break_alliance(&al, pid, ticks);
                    }
                }
                if self.ai_send_attack_official(pid, Some(tid), expand, reserve, &mut rng, false) {
                    if let Exec::Tribe(t) = &mut self.execs[idx] {
                        t.rng = rng;
                    }
                    return;
                }
            }
        }
        let mut still_tn = neighbors_tn;
        if still_tn {
            if self.nearby(pid).iter().any(|n| n.is_none()) {
                if self.ai_send_attack_official(pid, None, expand, reserve, &mut rng, false) {
                    if let Exec::Tribe(t) = &mut self.execs[idx] {
                        t.rng = rng;
                        t.neighbors_tn = true;
                    }
                    return;
                }
            } else {
                still_tn = false;
            }
        }
        self.tribe_attack_random_target(pid, trigger, reserve, expand, &mut rng);
        if let Exec::Tribe(t) = &mut self.execs[idx] {
            t.rng = rng;
            t.neighbors_tn = still_tn;
        }
    }

    fn neighbor_traitor_to_attack(&self, pid: &str, rng: &mut PseudoRandom) -> Option<String> {
        let traitors: Vec<String> = self
            .nearby(pid)
            .into_iter()
            .flatten()
            .filter(|n| {
                !self.is_allied(pid, n)
                    && self.players.get(n).map(|p| p.traitor).unwrap_or(false)
            })
            .collect();
        if traitors.is_empty() {
            return None;
        }
        Some(rng.rand_element(&traitors))
    }

    fn tribe_attack_random_target(
        &mut self,
        pid: &str,
        trigger: f64,
        reserve: f64,
        expand: f64,
        rng: &mut PseudoRandom,
    ) {
        let Some(p) = self.players.get(pid) else {
            return;
        };
        let cities = city_levels(&self.units, pid);
        let max = crate::config::max_troops(p.player_type(), self.difficulty, p.num_tiles(), cities);
        // Official attackRandomTarget: trigger only, no chance(10) fallback.
        if max > 0.0 && p.troops_f64() / max < trigger {
            return;
        }
        if let Some(atk) = self.incoming_attack_player(pid) {
            if self.ai_send_attack_official(pid, Some(atk), expand, reserve, rng, true) {
                return;
            }
        }
        if let Some(tid) = self.neighbor_traitor_to_attack(pid, rng) {
            if rng.chance(3) && self.ai_send_attack_official(pid, Some(tid), expand, reserve, rng, false) {
                return;
            }
        }
        let neighbors = self.nearby(pid);
        for n in rng.shuffle_array(&neighbors) {
            let Some(tid) = n else {
                continue;
            };
            if self.is_allied(pid, &tid) {
                continue;
            }
            if let Some(other) = self.players.get(&tid) {
                if matches!(other.player_type(), PlayerType::Nation | PlayerType::Human) {
                    if rng.chance(2) {
                        continue;
                    }
                }
            }
            if self.ai_send_attack_official(pid, Some(tid), expand, reserve, rng, false) {
                return;
            }
        }
    }

    /// Official `AiAttackBehavior.attackBestTarget` (nation path).
    fn nation_attack_best_target(
        &mut self,
        pid: &str,
        trigger: f64,
        reserve: f64,
        expand: f64,
        rng: &mut PseudoRandom,
    ) {
        if self.nation_has_neighboring_bot_with_structures(pid)
            && self.nation_attack_bots(pid, expand, reserve, rng)
        {
            return;
        }
        let Some(p) = self.players.get(pid) else {
            return;
        };
        let cities = city_levels(&self.units, pid);
        let max = crate::config::max_troops(p.player_type(), self.difficulty, p.num_tiles(), cities);
        let ratio = if max > 0.0 { p.troops_f64() / max } else { 0.0 };
        if ratio < reserve {
            return;
        }
        if ratio < trigger && !rng.chance(10) {
            return;
        }
        let nearby = self.nearby(pid);
        let mut enemies: Vec<String> = nearby
            .iter()
            .flatten()
            .filter(|id| !self.is_allied(pid, id))
            .cloned()
            .collect();
        let mut friends: Vec<String> = nearby
            .iter()
            .flatten()
            .filter(|id| self.is_allied(pid, id))
            .cloned()
            .collect();
        enemies.sort_by_key(|id| self.players.get(id).map(|o| o.troops).unwrap_or(0));
        friends.sort_by_key(|id| self.players.get(id).map(|o| o.troops).unwrap_or(0));

        // Impossible strategy order.
        if let Some(atk) = self.incoming_attack_player(pid) {
            if self.ai_send_attack_official(pid, Some(atk), expand, reserve, rng, true) {
                return;
            }
        }
        if self.nation_attack_bots(pid, expand, reserve, rng) {
            return;
        }
        if let Some(id) = self.nation_find_very_weak(pid, &enemies) {
            if self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false) {
                return;
            }
        }
        if self.nation_assist_allies(pid, expand, reserve, rng) {
            return;
        }
        if let Some(id) = self.nation_find_traitor(pid, &enemies) {
            if self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false) {
                return;
            }
        }
        if let Some(id) = self.nation_find_afk(pid, &enemies) {
            if self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false) {
                return;
            }
        }
        if self.nation_maybe_betray_and_attack(pid, &friends, &enemies, expand, reserve, rng) {
            return;
        }
        if let Some(id) = self.nation_find_victim(pid, &enemies) {
            if self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false) {
                return;
            }
        }
        if self.nation_bordering_nuked(pid)
            && self.ai_send_attack_official(pid, None, expand, reserve, rng, false)
        {
            return;
        }
        if self.nation_attack_hated(pid, expand, reserve, rng) {
            return;
        }
        if let Some(id) = enemies.first().cloned() {
            let other_t = self.players.get(&id).map(|o| o.troops_f64()).unwrap_or(0.0);
            let me_t = self.players.get(pid).map(|o| o.troops_f64()).unwrap_or(0.0);
            if other_t < me_t
                && self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false)
            {
                return;
            }
        }
        if enemies.is_empty() {
            if let Some(id) = self.nation_nearest_island_enemy(pid, rng) {
                let _ = self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false);
            }
        }
    }

    fn nation_has_neighboring_bot_with_structures(&self, pid: &str) -> bool {
        self.nearby(pid).into_iter().flatten().any(|id| {
            !self.is_allied(pid, &id)
                && self
                    .players
                    .get(&id)
                    .map(|o| o.player_type() == PlayerType::Bot)
                    .unwrap_or(false)
                && self
                    .units
                    .iter()
                    .any(|u| u.active && u.owner_id == id && u.ty.is_structure())
        })
    }

    fn nation_attack_bots(
        &mut self,
        pid: &str,
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
    ) -> bool {
        let mut bots: Vec<String> = self
            .nearby(pid)
            .into_iter()
            .flatten()
            .filter(|id| {
                !self.is_allied(pid, id)
                    && self
                        .players
                        .get(id)
                        .map(|o| o.player_type() == PlayerType::Bot)
                        .unwrap_or(false)
            })
            .collect();
        if bots.is_empty() {
            return false;
        }
        bots.sort_by(|a, b| {
            let a_st = self
                .units
                .iter()
                .any(|u| u.active && u.owner_id == *a && u.ty.is_structure());
            let b_st = self
                .units
                .iter()
                .any(|u| u.active && u.owner_id == *b && u.ty.is_structure());
            if a_st != b_st {
                return b_st.cmp(&a_st);
            }
            let da = self
                .players
                .get(a)
                .map(|o| o.troops_f64() / o.num_tiles().max(1) as f64)
                .unwrap_or(0.0);
            let db = self
                .players
                .get(b)
                .map(|o| o.troops_f64() / o.num_tiles().max(1) as f64)
                .unwrap_or(0.0);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        });
        let limit = match self.difficulty {
            Difficulty::Easy => 1,
            Difficulty::Medium => {
                if rng.chance(2) {
                    1
                } else {
                    2
                }
            }
            Difficulty::Hard => 3,
            Difficulty::Impossible => 100,
        };
        let mut sent_any = false;
        for id in bots.into_iter().take(limit) {
            if self.ai_send_attack_official(pid, Some(id), expand, reserve, rng, false) {
                sent_any = true;
            }
        }
        sent_any
    }

    fn nation_find_very_weak(&self, pid: &str, enemies: &[String]) -> Option<String> {
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        enemies.iter().find(|id| {
            let Some(o) = self.players.get(*id) else {
                return false;
            };
            let cities = city_levels(&self.units, id);
            let max = crate::config::max_troops(o.player_type(), self.difficulty, o.num_tiles(), cities);
            o.troops_f64() < max * 0.15 && o.troops_f64() < me_t * 1.2
        }).cloned()
    }

    fn nation_find_traitor(&self, pid: &str, enemies: &[String]) -> Option<String> {
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        enemies.iter().find(|id| {
            self.players.get(*id).map(|o| o.traitor && o.troops_f64() < me_t * 1.2).unwrap_or(false)
        }).cloned()
    }

    fn nation_find_afk(&self, pid: &str, enemies: &[String]) -> Option<String> {
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        enemies.iter().find(|id| {
            self.players
                .get(*id)
                .map(|o| o.disconnected && o.troops_f64() < me_t * 3.0)
                .unwrap_or(false)
        }).cloned()
    }

    fn nation_find_victim(&self, pid: &str, enemies: &[String]) -> Option<String> {
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        enemies.iter().find(|id| {
            let Some(o) = self.players.get(*id) else {
                return false;
            };
            if o.troops_f64() > me_t * 1.2 {
                return false;
            }
            let incoming = self.incoming_troops_on(id);
            incoming > o.troops_f64() * 0.5
        }).cloned()
    }

    pub(crate) fn incoming_troops_on(&self, pid: &str) -> f64 {
        self.players
            .values()
            .flat_map(|o| o.outgoing.iter())
            .filter(|a| a.active && a.target_player_id.as_deref() == Some(pid))
            .map(|a| a.troops)
            .sum()
    }

    fn nation_assist_allies(
        &mut self,
        pid: &str,
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
    ) -> bool {
        let allies: Vec<String> = self
            .alliances
            .iter()
            .filter(|a| a.requestor == pid || a.recipient == pid)
            .map(|a| a.other(pid).to_string())
            .collect();
        for ally in allies {
            if (self.players.get(pid).map(|p| p.relation(&ally)).unwrap_or(crate::diplomacy::Relation::Neutral) as i32)
                < (crate::diplomacy::Relation::Friendly as i32)
            {
                continue;
            }
            let targets: Vec<String> = self
                .players
                .values()
                .flat_map(|o| o.outgoing.iter())
                .filter(|a| a.active && a.target_player_id.as_deref() == Some(ally.as_str()))
                .filter_map(|a| {
                    self.players
                        .values()
                        .find(|p| p.small_id == a.attacker_small_id)
                        .map(|p| p.id().to_string())
                })
                .collect();
            // Official uses ally.targets(); incoming attackers are the common assist case.
            for tid in targets {
                if tid == pid || self.is_allied(pid, &tid) {
                    continue;
                }
                if self.ai_send_attack_official(pid, Some(tid), expand, reserve, rng, false) {
                    if let Some(p) = self.players.get_mut(pid) {
                        p.update_relation(&ally, -20.0);
                    }
                    return true;
                }
            }
        }
        false
    }

    fn nation_maybe_betray_and_attack(
        &mut self,
        pid: &str,
        friends: &[String],
        enemies: &[String],
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
    ) -> bool {
        let border_n = friends.len() + enemies.len();
        for friend in friends {
            if self.nation_should_betray(pid, friend, border_n) {
                if let Some(al) = self.alliances.iter().find(|a| a.involves(pid, friend)).cloned() {
                    self.break_alliance(&al, pid, self.ticks());
                }
                return self.ai_send_attack_official(pid, Some(friend.clone()), expand, reserve, rng, true);
            }
        }
        false
    }

    fn nation_should_betray(&self, pid: &str, other: &str, bordering: usize) -> bool {
        if !self.is_allied(pid, other) {
            return false;
        }
        let Some(me) = self.players.get(pid) else {
            return false;
        };
        let Some(o) = self.players.get(other) else {
            return false;
        };
        if matches!(self.difficulty, Difficulty::Hard | Difficulty::Impossible) {
            let cities = city_levels(&self.units, other);
            let max = crate::config::max_troops(o.player_type(), self.difficulty, o.num_tiles(), cities);
            let outgoing: f64 = o.outgoing.iter().filter(|a| a.active).map(|a| a.troops).sum();
            if o.troops_f64() + outgoing < max * 0.2 && o.troops_f64() < me.troops_f64() {
                return true;
            }
            if o.traitor && o.troops_f64() < me.troops_f64() * 1.2 {
                return true;
            }
            if bordering == 1 && o.troops_f64() * 3.0 < me.troops_f64() {
                return true;
            }
        }
        false
    }

    fn nation_bordering_nuked(&self, pid: &str) -> bool {
        if self.is_unit_disabled(UnitType::MissileSilo) {
            return false;
        }
        let Some(p) = self.players.get(pid) else {
            return false;
        };
        let mut nbuf = [0u32; 4];
        for &t in &p.border_tiles {
            let n = self.map.neighbors4(t, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if self.map.is_land(nb) && !self.map.has_owner(nb) && self.map.has_fallout(nb) {
                    return true;
                }
            }
        }
        false
    }

    fn nation_attack_hated(
        &mut self,
        pid: &str,
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
    ) -> bool {
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        let mut rels: Vec<(String, f64)> = self
            .players
            .get(pid)
            .map(|p| p.relations.iter().map(|(k, v)| (k.clone(), *v)).collect())
            .unwrap_or_default();
        rels.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        for (oid, val) in rels {
            if crate::diplomacy::Relation::from_value(val) != crate::diplomacy::Relation::Hostile {
                continue;
            }
            if self.is_allied(pid, &oid) {
                continue;
            }
            let other_t = self.players.get(&oid).map(|o| o.troops_f64()).unwrap_or(0.0);
            if other_t > me_t * 3.0 {
                continue;
            }
            if self.ai_send_attack_official(pid, Some(oid), expand, reserve, rng, false) {
                return true;
            }
        }
        false
    }

    fn nation_nearest_island_enemy(
        &mut self,
        pid: &str,
        rng: &mut PseudoRandom,
    ) -> Option<String> {
        if self.is_unit_disabled(UnitType::TransportShip) || self.boat_count(pid) >= BOAT_MAX_NUMBER {
            return None;
        }
        let Some(p) = self.players.get(pid) else {
            return None;
        };
        if !p.border_tiles.iter().any(|&t| self.map.is_shore(t)) {
            return None;
        }
        let me_t = p.troops_f64();
        let my_center = player_border_center(&self.map, p);
        let my_shores: Vec<TileRef> = p
            .border_tiles
            .iter()
            .copied()
            .filter(|&t| self.map.is_shore(t))
            .collect();
        let mut ranked: Vec<(u32, String)> = self
            .players
            .values()
            .filter(|o| o.id() != pid && !self.is_allied(pid, o.id()) && o.troops_f64() < me_t)
            .map(|o| {
                let c = player_border_center(&self.map, o);
                let dx = my_center.0.abs_diff(c.0);
                let dy = my_center.1.abs_diff(c.1);
                (dx + dy, o.id().to_string())
            })
            .collect();
        ranked.sort_by_key(|(d, _)| *d);
        let mut reachable = Vec::new();
        for (_, oid) in ranked {
            let their: Vec<TileRef> = self
                .players
                .get(&oid)
                .map(|o| {
                    o.border_tiles
                        .iter()
                        .copied()
                        .filter(|&t| self.map.is_shore(t))
                        .collect()
                })
                .unwrap_or_default();
            let Some((_, dst)) = closest_two_tiles(&self.map, &my_shores, &their) else {
                continue;
            };
            if self.can_build_boat(pid, dst).is_some() {
                reachable.push(oid);
                if reachable.len() >= 2 {
                    break;
                }
            }
        }
        if reachable.is_empty() {
            return None;
        }
        if reachable.len() >= 2 && rng.chance(3) {
            return Some(reachable[1].clone());
        }
        Some(reachable[0].clone())
    }

    pub(crate) fn incoming_attack_player(&self, pid: &str) -> Option<String> {
        let bot = self.players.get(pid).map(|p| p.player_type() == PlayerType::Bot).unwrap_or(false);
        let mut best_t = 0.0;
        let mut best: Option<String> = None;
        for p in self.players.values() {
            if p.id() == pid {
                continue;
            }
            if self.is_allied(pid, p.id()) {
                continue;
            }
            if !bot && p.player_type() == PlayerType::Bot {
                continue;
            }
            for a in &p.outgoing {
                if a.active && a.target_player_id.as_deref() == Some(pid) && a.troops > best_t {
                    best_t = a.troops;
                    best = Some(p.id().to_string());
                }
            }
        }
        best
    }

    fn ai_should_attack(
        &self,
        owner: &str,
        target: Option<&str>,
        rng: &mut PseudoRandom,
    ) -> bool {
        let Some(tid) = target else {
            return true;
        };
        let Some(other) = self.players.get(tid) else {
            return true;
        };
        let Some(me) = self.players.get(owner) else {
            return true;
        };
        if other.player_type() != PlayerType::Human || other.traitor || me.player_type() == PlayerType::Bot
        {
            return true;
        }
        match self.difficulty {
            Difficulty::Easy => rng.next_int(0.0, 4.0) == 0,
            Difficulty::Medium => !rng.chance(4),
            _ => true,
        }
    }

    /// Official `AiAttackBehavior.sendAttack`.
    fn ai_send_attack_official(
        &mut self,
        pid: &str,
        target: Option<String>,
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
        force: bool,
    ) -> bool {
        if !force && !self.ai_should_attack(pid, target.as_deref(), rng) {
            return false;
        }
        let land = match target.as_deref() {
            None => self.has_land_border_with_terra_nullius(pid),
            Some(tid) => {
                let Some(other) = self.players.get(tid) else {
                    return false;
                };
                let os = other.small_id;
                self.players
                    .get(pid)
                    .map(|p| self.shares_border(p, os))
                    .unwrap_or(false)
            }
        };
        if land {
            self.ai_send_land_attack(pid, target, expand, reserve, rng)
        } else {
            self.ai_send_boat_attack(pid, target, rng)
        }
    }

    /// Official `NationEmojiBehavior.maybeSendAttackEmoji` after a valid
    /// `calculateAttackTroops` against a player.
    fn maybe_send_attack_emoji(&mut self, pid: &str, target: &str, rng: &mut PseudoRandom) {
        let Some(other) = self.players.get(target) else {
            return;
        };
        if other.player_type() != PlayerType::Human {
            return;
        }
        let ticks = self.ticks();
        {
            let st = self.later.nations.entry(pid.to_string()).or_default();
            if let Some(&last) = st.last_emoji_sent.get(target) {
                if ticks.saturating_sub(last) <= 300 {
                    return;
                }
            }
            st.last_emoji_sent.insert(target.to_string(), ticks);
        }
        let rel = self
            .players
            .get(pid)
            .map(|p| p.relation(target))
            .unwrap_or(Relation::Neutral);
        let ok = if (rel as i32) >= (Relation::Neutral as i32) {
            rng.chance(2)
        } else {
            rng.chance(4)
        };
        if !ok {
            return;
        }
        // Official sendEmoji then randElement of a 1-emoji list.
        let _ = rng.next_int(0.0, 1.0);
    }

    fn ai_send_land_attack(
        &mut self,
        pid: &str,
        target: Option<String>,
        expand: f64,
        reserve: f64,
        rng: &mut PseudoRandom,
    ) -> bool {
        let Some(p) = self.players.get(pid) else {
            return false;
        };
        let cities = city_levels(&self.units, pid);
        let max = crate::config::max_troops(p.player_type(), self.difficulty, p.num_tiles(), cities);
        let bot_with_structs = target
            .as_ref()
            .and_then(|t| self.players.get(t))
            .map(|o| {
                o.player_type() == PlayerType::Bot
                    && self
                        .units
                        .iter()
                        .any(|u| u.active && u.owner_id == o.id() && u.ty.is_structure())
            })
            .unwrap_or(false);
        let use_reserve = target.is_some() && !bot_with_structs;
        let keep = max * if use_reserve { reserve } else { expand };
        let mut troops = if target
            .as_ref()
            .and_then(|t| self.players.get(t))
            .map(|o| o.player_type() == PlayerType::Bot)
            .unwrap_or(false)
            && p.player_type() != PlayerType::Bot
        {
            let other_t = target
                .as_ref()
                .and_then(|t| self.players.get(t))
                .map(|o| o.troops_f64())
                .unwrap_or(0.0);
            (p.troops_f64() - keep).min(other_t * 4.0)
        } else {
            p.troops_f64() - keep
        };
        if target.is_some() {
            troops = troops.min(self.troop_send_cap(pid));
        }
        if troops < 1.0 {
            return false;
        }
        if let Some(tid) = target.as_deref() {
            if self.is_attack_too_weak(pid, tid, troops) {
                return false;
            }
            self.maybe_send_attack_emoji(pid, tid, rng);
        }
        self.ai_send_attack(pid, target, troops);
        true
    }

    fn ai_send_boat_attack(
        &mut self,
        pid: &str,
        target: Option<String>,
        rng: &mut PseudoRandom,
    ) -> bool {
        if self.is_unit_disabled(UnitType::TransportShip) {
            return false;
        }
        if self.boat_count(pid) >= BOAT_MAX_NUMBER {
            return false;
        }
        let (mut troops, my_shores) = {
            let Some(p) = self.players.get(pid) else {
                return false;
            };
            let troops = p.troops_f64() / 5.0;
            let shores: Vec<TileRef> = p
                .border_tiles
                .iter()
                .copied()
                .filter(|&t| self.map.is_shore(t))
                .collect();
            (troops, shores)
        };
        if target.is_some() {
            troops = troops.min(self.troop_send_cap(pid));
        }
        if troops < 1.0 {
            return false;
        }
        if let Some(tid) = target.as_deref() {
            if self.is_attack_too_weak(pid, tid, troops) {
                return false;
            }
        }
        if my_shores.is_empty() {
            return false;
        }
        let dst = if let Some(tid) = target.as_deref() {
            let Some(other) = self.players.get(tid) else {
                return false;
            };
            let their: Vec<TileRef> = other
                .border_tiles
                .iter()
                .copied()
                .filter(|&t| self.map.is_shore(t))
                .collect();
            let Some(pair) = closest_two_tiles(&self.map, &my_shores, &their) else {
                return false;
            };
            pair.1
        } else {
            return self.ai_send_boat_to_nearby_tn_bool(pid);
        };
        if self.can_build_boat(pid, dst).is_none() {
            return false;
        }
        if let Some(tid) = target.as_deref() {
            self.maybe_send_attack_emoji(pid, tid, rng);
        }
        self.uninit.push(Exec::Boat(BoatExec {
            active: true,
            inited: false,
            owner_id: pid.to_string(),
            ref_tile: dst,
            troops,
            dst: None,
            src: None,
            unit_id: None,
            last_move: 0,
            stepper: PathStepper::new(),
        }));
        true
    }

    fn ai_send_boat_to_nearby_tn_bool(&mut self, pid: &str) -> bool {
        let before = self.uninit.len();
        self.ai_send_boat_to_nearby_tn(pid);
        self.uninit.len() > before
    }

    fn ai_send_tn_attack(&mut self, pid: &str, expand_ratio: f64) {
        if self.has_land_tn_border(pid) {
            let Some(p) = self.players.get(pid) else {
                return;
            };
            let cities = city_levels(&self.units, pid);
            let max = crate::config::max_troops(
                p.player_type(),
                self.difficulty,
                p.num_tiles(),
                cities,
            );
            // TN uses expandRatio (not reserve) as the kept troop floor.
            let keep = max * expand_ratio;
            let troops = p.troops_f64() - keep;
            if troops >= 1.0 {
                self.ai_send_attack(pid, None, troops);
            }
            return;
        }
        self.ai_send_boat_to_nearby_tn(pid);
    }

    fn ai_send_boat_to_nearby_tn(&mut self, pid: &str) {
        if self.is_unit_disabled(UnitType::TransportShip) {
            return;
        }
        if self.boat_count(pid) >= BOAT_MAX_NUMBER {
            return;
        }
        let Some(p) = self.players.get(pid) else {
            return;
        };
        let troops = p.troops_f64() / 5.0;
        if troops < 1.0 {
            return;
        }
        let shores: Vec<TileRef> = p
            .border_tiles
            .iter()
            .copied()
            .filter(|&t| self.map.is_shore(t))
            .collect();
        let dirs: [(i32, i32); 4] = [(0, -1), (0, 1), (-1, 0), (1, 0)];
        for (i, &border) in shores.iter().enumerate() {
            if i % 10 != 0 {
                continue;
            }
            let bx = self.map.x(border) as i32;
            let by = self.map.y(border) as i32;
            for (dx, dy) in dirs {
            let x1 = bx + dx;
            let y1 = by + dy;
            if x1 < 0 || y1 < 0 {
                continue;
            }
            let Ok(w) = self.map.ref_xy(x1 as u32, y1 as u32) else {
                continue;
            };
                if !self.map.is_water(w) {
                    continue;
                }
                let nx = bx + dx * 5;
                let ny = by + dy * 5;
                if nx < 0 || ny < 0 {
                    continue;
                }
                let Ok(tile) = self.map.ref_xy(nx as u32, ny as u32) else {
                    continue;
                };
                if !self.map.is_land(tile)
                    || self.map.is_impassable(tile)
                    || self.map.has_owner(tile)
                    || self.map.has_fallout(tile)
                {
                    continue;
                }
                if self.can_build_boat(pid, tile).is_none() {
                    continue;
                }
                self.uninit.push(Exec::Boat(BoatExec {
                    active: true,
                    inited: false,
                    owner_id: pid.to_string(),
                    ref_tile: tile,
                    troops,
                    dst: None,
                    src: None,
                    unit_id: None,
                    last_move: 0,
                    stepper: PathStepper::new(),
                }));
                return;
            }
        }
    }

    fn tribe_accept_alliances(&mut self, pid: &str, ticks: u32) {
        let incoming: Vec<String> = self
            .incoming_requests
            .iter()
            .filter(|r| r.recipient == pid && r.status == RequestStatus::Pending)
            .map(|r| r.requestor.clone())
            .collect();
        for other in incoming {
            self.accept_alliance(pid, &other, ticks);
        }
    }

    fn has_land_tn_border(&self, player_id: &str) -> bool {
        let Some(p) = self.players.get(player_id) else {
            return false;
        };
        let mut nbuf = [0u32; 4];
        for &t in &p.border_tiles {
            let n = self.map.neighbors4(t, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if self.map.is_land(nb)
                    && !self.map.is_impassable(nb)
                    && !self.map.has_owner(nb)
                    && !self.map.has_fallout(nb)
                {
                    return true;
                }
            }
        }
        false
    }

    /// Official `hasLandBorderWithTerraNullius`: fallout tiles still count.
    fn has_land_border_with_terra_nullius(&self, player_id: &str) -> bool {
        let Some(p) = self.players.get(player_id) else {
            return false;
        };
        let mut nbuf = [0u32; 4];
        for &t in &p.border_tiles {
            let n = self.map.neighbors4(t, &mut nbuf);
            for i in 0..n {
                let nb = nbuf[i];
                if self.map.is_land(nb) && !self.map.is_impassable(nb) && !self.map.has_owner(nb) {
                    return true;
                }
            }
        }
        false
    }

    fn ai_send_attack(&mut self, owner: &str, target: Option<String>, troops: f64) {
        if troops < 1.0 {
            return;
        }
        self.uninit.push(Exec::Attack(AttackExec {
            active: true,
            inited: false,
            start_troops: troops,
            owner_id: owner.to_string(),
            target_id: target,
            attack_id: None,
            owner_small: 0,
            target_small: 0,
            target_is_player: true,
            source_tile: None,
            remove_troops: true,
            heap: FlatHeap::new(),
            rng: PseudoRandom::new(123),
        }));
        if let Some(a) = self.uninit.last_mut() {
            if let Exec::Attack(atk) = a {
                atk.target_is_player = atk.target_id.is_some();
            }
        }
    }

    fn tick_nation(&mut self, idx: usize, ticks: u32) {
        let (pid, spawn_x, spawn_y, rate, atick, trigger, reserve, expand, behaviors, spawn_added) = {
            let Exec::Nation(n) = &self.execs[idx] else {
                return;
            };
            if !n.active {
                return;
            }
            (
                n.player_id.clone(),
                n.spawn_x,
                n.spawn_y,
                n.attack_rate.max(1),
                n.attack_tick,
                n.trigger_ratio,
                n.reserve_ratio,
                n.expand_ratio,
                n.behaviors,
                n.spawn_added,
            )
        };
        if !self.players.contains_key(&pid) {
            return;
        }
        if self.in_spawn_phase() {
            if self.players.get(&pid).map(|p| p.has_spawned()).unwrap_or(false) {
                if ticks % rate != atick {
                    return;
                }
            } else if spawn_added {
                return;
            }
            let tile = self.nation_spawn_tile(idx, spawn_x, spawn_y);
            if let Some(tile) = tile {
                self.uninit.push(Exec::Spawn(SpawnExec {
                    active: true,
                    player_id: pid.clone(),
                    game_id: self.game_id.clone(),
                    tile: Some(tile),
                    info: None,
                    rng: PseudoRandom::new(simple_hash(&pid).wrapping_add(simple_hash(&self.game_id))),
                }));
                if let Exec::Nation(n) = &mut self.execs[idx] {
                    n.spawn_added = true;
                }
            }
            return;
        }
        if spawn_added && !self.players.get(&pid).map(|p| p.has_spawned()).unwrap_or(false) {
            return;
        }
        if !self.players.get(&pid).map(|p| p.is_alive()).unwrap_or(false) {
            if let Exec::Nation(n) = &mut self.execs[idx] {
                n.active = false;
            }
            return;
        }
        if !behaviors {
            // NationNukeBehavior constructor: isHydroNation = random.chance(3).
            let hydro = if let Exec::Nation(n) = &mut self.execs[idx] {
                let h = n.rng.chance(3);
                n.behaviors = true;
                Some(h)
            } else {
                None
            };
            if let Some(h) = hydro {
                self.later.nations.entry(pid.clone()).or_default().is_hydro = Some(h);
            }
            let troops = self.players.get(&pid).map(|p| p.troops_f64() / 2.0).unwrap_or(0.0);
            self.ai_send_attack(&pid, None, troops);
            return;
        }
        if self.difficulty != Difficulty::Easy
            && !self.is_unit_disabled(UnitType::Warship)
            && self
                .players
                .get(&pid)
                .map(|p| p.units_constructed(UnitType::Port) > 0)
                .unwrap_or(false)
        {
            crate::nation_later::track_ships_and_retaliate(self, &pid, idx);
        }
        if ticks % rate != atick {
            let offset = ticks % rate;
            let one_third = (atick + rate / 3) % rate;
            let two_thirds = (atick + (rate * 2) / 3) % rate;
            if offset == one_third || offset == two_thirds {
                crate::nation_later::handle_structures(self, &pid, idx);
            }
            return;
        }
        self.nation_consume_emoji_rng(idx, &pid, ticks);
        self.nation_handle_alliances(&pid, idx, ticks);
        crate::nation_later::consider_mirv(self, &pid, idx);
        crate::nation_later::handle_structures(self, &pid, idx);
        crate::nation_later::maybe_spawn_warship(self, &pid, idx);
        self.nation_handle_embargoes(&pid);
        self.nation_maybe_attack(&pid, idx, trigger, reserve, expand);
        crate::nation_later::counter_warship_infestation(self, &pid, idx);
        crate::nation_later::maybe_send_nuke(self, &pid, idx);
    }

    fn nation_spawn_tile(&mut self, idx: usize, sx: Option<u32>, sy: Option<u32>) -> Option<TileRef> {
        let (Some(cx), Some(cy)) = (sx, sy) else {
            return None;
        };
        let Exec::Nation(n) = &mut self.execs[idx] else {
            return None;
        };
        let delta = 25i32;
        for _ in 0..50 {
            let x = n.rng.next_int((cx as i32 - delta) as f64, (cx as i32 + delta) as f64);
            let y = n.rng.next_int((cy as i32 - delta) as f64, (cy as i32 + delta) as f64);
            if x < 0 || y < 0 {
                continue;
            }
            let Ok(tile) = self.map.ref_xy(x as u32, y as u32) else {
                continue;
            };
            if self.map.is_land(tile) && !self.map.has_owner(tile) && !self.map.is_impassable(tile) {
                if self.map.terrain_type(tile) == TerrainKind::Mountain && n.rng.chance(2) {
                    continue;
                }
                return Some(tile);
            }
        }
        None
    }

    fn nation_consume_emoji_rng(&mut self, idx: usize, pid: &str, ticks: u32) {
        let incoming_human: Vec<(String, f64)> = self
            .players
            .values()
            .flat_map(|o| {
                o.outgoing.iter().filter_map(move |a| {
                    if a.active
                        && a.target_player_id.as_deref() == Some(pid)
                        && o.player_type() == PlayerType::Human
                    {
                        Some((o.id().to_string(), a.troops))
                    } else {
                        None
                    }
                })
            })
            .collect();
        let incoming_sum: f64 = self
            .players
            .values()
            .flat_map(|o| o.outgoing.iter())
            .filter(|a| a.active && a.target_player_id.as_deref() == Some(pid))
            .map(|a| a.troops)
            .sum();
        let incoming_any = incoming_sum > 0.0;
        let ours = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        let human_allies: Vec<String> = self
            .alliances
            .iter()
            .filter(|a| a.requestor == pid || a.recipient == pid)
            .map(|a| a.other(pid).to_string())
            .filter(|oid| {
                self.players
                    .get(oid)
                    .map(|p| p.player_type() == PlayerType::Human)
                    .unwrap_or(false)
            })
            .collect();
        let traitors: Vec<String> = self
            .players
            .values()
            .filter(|p| {
                p.player_type() == PlayerType::Human
                    && p.traitor
                    && p.id() != pid
                    && !self.is_allied(pid, p.id())
            })
            .map(|p| p.id().to_string())
            .collect();
        let greet_humans: Vec<String> = self
            .nearby(pid)
            .into_iter()
            .flatten()
            .filter(|id| {
                self.players
                    .get(id)
                    .map(|p| p.player_type() == PlayerType::Human)
                    .unwrap_or(false)
            })
            .collect();
        let mut rng = {
            let Exec::Nation(n) = &self.execs[idx] else {
                return;
            };
            n.rng.clone()
        };
        let e16 = rng.chance(16);
        if e16 && incoming_any && incoming_sum >= ours * 3.0 {
            let _ = rng.next_int(0.0, 8.0);
        }
        if rng.chance(8) {
            for (hid, troops) in &incoming_human {
                if ours > 0.0 && *troops < ours * 0.1 {
                    let list_len = if rng.chance(2) { 2.0 } else { 1.0 };
                    let skip = self
                        .later
                        .nations
                        .get(pid)
                        .and_then(|s| s.last_emoji_sent.get(hid))
                        .is_some_and(|&last| ticks.saturating_sub(last) <= 300);
                    if !skip {
                        self.later
                            .nations
                            .entry(pid.to_string())
                            .or_default()
                            .last_emoji_sent
                            .insert(hid.clone(), ticks);
                        let _ = rng.next_int(0.0, list_len);
                    }
                }
            }
        }
        let _ = rng.chance(300);
        if rng.chance(250) && !human_allies.is_empty() {
            let _ = rng.rand_element(&human_allies);
            let _ = rng.chance(3);
        }
        if rng.chance(40) && !traitors.is_empty() {
            let _ = rng.rand_element(&traitors);
        }
        if ticks >= 6000 {
            let _ = rng.chance(10000);
        }
        if ticks <= 600 && rng.chance(250) && !greet_humans.is_empty() {
            let _ = rng.rand_element(&greet_humans);
        }
        if let Exec::Nation(n) = &mut self.execs[idx] {
            n.rng = rng;
        }
    }

    fn nation_handle_alliances(&mut self, pid: &str, idx: usize, ticks: u32) {
        let incoming: Vec<(String, u32)> = self
            .incoming_requests
            .iter()
            .filter(|r| r.recipient == pid && r.status == RequestStatus::Pending)
            .map(|r| (r.requestor.clone(), r.created_at))
            .collect();
        for (other, created) in incoming {
            if created <= 101 {
                self.reject_incoming_from(pid, &other);
                continue;
            }
            let mut rng = {
                let Exec::Nation(n) = &self.execs[idx] else {
                    return;
                };
                n.rng.clone()
            };
            let ok = self.nation_alliance_decision(pid, &other, true, &mut rng);
            if let Exec::Nation(n) = &mut self.execs[idx] {
                n.rng = rng;
            }
            if ok {
                self.accept_alliance(pid, &other, ticks);
            } else {
                self.reject_incoming_from(pid, &other);
            }
        }
        let extend: Vec<String> = self
            .alliances
            .iter()
            .filter(|a| a.involves(pid, a.other(pid)) && a.only_one_agreed_to_extend())
            .map(|a| a.other(pid).to_string())
            .collect();
        for other in extend {
            let mut rng = {
                let Exec::Nation(n) = &self.execs[idx] else {
                    return;
                };
                n.rng.clone()
            };
            let ok = self.nation_alliance_decision(pid, &other, true, &mut rng);
            if let Exec::Nation(n) = &mut self.execs[idx] {
                n.rng = rng;
            }
            let _ = ok;
        }
    }

    fn nation_alliance_decision(
        &self,
        pid: &str,
        other: &str,
        is_response: bool,
        rng: &mut PseudoRandom,
    ) -> bool {
        if self.nation_is_confused(rng) {
            return rng.chance(2);
        }
        let traitor = self.players.get(other).map(|p| p.traitor).unwrap_or(false);
        if traitor && rng.next_int(0.0, 100.0) >= 10 {
            if is_response {
                let _ = rng.chance(3);
            }
            return false;
        }
        if self.nation_other_has_too_many_alliances(other) {
            return false;
        }
        if self.nation_partner_is_threat(pid, other) {
            if !is_response {
                let _ = rng.chance(6);
            }
            if is_response {
                let _ = rng.chance(6);
            }
            return true;
        }
        if self.nation_should_reject_team(rng) {
            return false;
        }
        let rel = self
            .players
            .get(pid)
            .map(|p| p.relation(other))
            .unwrap_or(crate::diplomacy::Relation::Neutral);
        if (rel as i32) < (crate::diplomacy::Relation::Neutral as i32) {
            if is_response {
                let _ = rng.chance(3);
            }
            return false;
        }
        if self.nation_partner_is_friendly(pid, other, rng) {
            let _ = rng.chance(3);
            return true;
        }
        if self.nation_already_enough_alliances(pid, other, rng) {
            return false;
        }
        if self.nation_is_earlygame(rng) {
            return true;
        }
        self.nation_partner_similarly_strong(pid, other, rng)
    }

    fn nation_is_confused(&self, rng: &mut PseudoRandom) -> bool {
        match self.difficulty {
            Difficulty::Easy => rng.chance(10),
            Difficulty::Medium => rng.chance(20),
            Difficulty::Hard => rng.chance(40),
            Difficulty::Impossible => false,
        }
    }

    fn nation_other_has_too_many_alliances(&self, other: &str) -> bool {
        if !matches!(self.difficulty, Difficulty::Hard | Difficulty::Impossible) {
            return false;
        }
        let total = self
            .players
            .values()
            .filter(|p| p.player_type() != PlayerType::Bot)
            .count();
        let n = self
            .alliances
            .iter()
            .filter(|a| a.involves(other, a.other(other)))
            .count();
        if self.difficulty == Difficulty::Hard {
            n as f64 >= total as f64 * 0.5
        } else {
            n as f64 >= total as f64 * 0.25
        }
    }

    fn nation_partner_is_threat(&self, pid: &str, other: &str) -> bool {
        let Some(me) = self.players.get(pid) else {
            return false;
        };
        let Some(them) = self.players.get(other) else {
            return false;
        };
        match self.difficulty {
            Difficulty::Easy => false,
            Difficulty::Medium => them.troops_f64() > me.troops_f64() * 2.5,
            Difficulty::Hard => {
                them.troops_f64() > me.troops_f64()
                    && self.player_max_troops(other) > self.player_max_troops(pid) * 2.0
            }
            Difficulty::Impossible => {
                let more_troops = them.troops_f64() > me.troops_f64() * 1.5;
                let more_max = them.troops_f64() > me.troops_f64()
                    && self.player_max_troops(other) > self.player_max_troops(pid) * 1.5;
                let more_tiles = them.troops_f64() > me.troops_f64()
                    && them.num_tiles() as f64 > me.num_tiles() as f64 * 1.5;
                more_troops || more_max || more_tiles
            }
        }
    }

    fn nation_should_reject_team(&self, rng: &mut PseudoRandom) -> bool {
        if self.cfg_team_mode() {
            return match self.difficulty {
                Difficulty::Easy => rng.next_int(0.0, 100.0) < 25,
                Difficulty::Medium => rng.next_int(0.0, 100.0) < 50,
                Difficulty::Hard => rng.next_int(0.0, 100.0) < 75,
                Difficulty::Impossible => true,
            };
        }
        false
    }

    pub(crate) fn cfg_team_mode(&self) -> bool {
        false
    }
    pub(crate) fn same_team(&self, _a: &str, _b: &str) -> bool {
        false
    }
    pub(crate) fn player_team(&self, _id: &str) -> Option<u32> {
        None
    }

    pub(crate) fn ally_ids(&self, pid: &str) -> Vec<String> {
        self.alliances
            .iter()
            .filter(|a| a.requestor == pid || a.recipient == pid)
            .map(|a| a.other(pid).to_string())
            .collect()
    }

    pub(crate) fn attackers_of(&self, pid: &str) -> Vec<String> {
        self.players
            .values()
            .flat_map(|o| o.outgoing.iter().filter_map(move |a| {
                if a.active && a.target_player_id.as_deref() == Some(pid) {
                    Some(o.id().to_string())
                } else {
                    None
                }
            }))
            .collect()
    }

    fn nation_partner_is_friendly(
        &self,
        pid: &str,
        other: &str,
        rng: &mut PseudoRandom,
    ) -> bool {
        let rel = self
            .players
            .get(pid)
            .map(|p| p.relation(other))
            .unwrap_or(crate::diplomacy::Relation::Neutral);
        if rel != crate::diplomacy::Relation::Friendly {
            return false;
        }
        match self.difficulty {
            Difficulty::Easy | Difficulty::Medium => true,
            Difficulty::Hard => rng.next_int(0.0, 100.0) >= 17,
            Difficulty::Impossible => rng.next_int(0.0, 100.0) >= 33,
        }
    }

    fn nation_already_enough_alliances(
        &self,
        pid: &str,
        other: &str,
        rng: &mut PseudoRandom,
    ) -> bool {
        match self.difficulty {
            Difficulty::Easy => false,
            Difficulty::Medium => {
                self.alliance_count(pid) >= rng.next_int(4.0, 6.0) as usize
            }
            Difficulty::Hard | Difficulty::Impossible => {
                let bordering: Vec<String> = self
                    .nearby(pid)
                    .into_iter()
                    .flatten()
                    .filter(|id| {
                        self.players
                            .get(id)
                            .map(|p| p.player_type() != PlayerType::Bot)
                            .unwrap_or(false)
                    })
                    .collect();
                let friends = bordering
                    .iter()
                    .filter(|id| self.is_allied(pid, id) || self.is_friendly_ids(pid, id))
                    .count();
                if bordering.len() >= 2 && bordering.iter().any(|id| id == other) {
                    return bordering.len() <= friends + 1;
                }
                let mine = self.alliance_count(pid);
                if self.difficulty == Difficulty::Hard {
                    mine >= rng.next_int(3.0, 5.0) as usize
                } else {
                    mine >= rng.next_int(2.0, 4.0) as usize
                }
            }
        }
    }

    fn nation_is_earlygame(&self, rng: &mut PseudoRandom) -> bool {
        let spawn = 100u32;
        match self.difficulty {
            Difficulty::Easy => {
                self.ticks() < 3000 + spawn && rng.next_int(0.0, 100.0) >= 10
            }
            Difficulty::Medium => {
                self.ticks() < 1800 + spawn && rng.next_int(0.0, 100.0) >= 30
            }
            Difficulty::Hard => {
                self.ticks() < 1800 + spawn && rng.next_int(0.0, 100.0) >= 50
            }
            Difficulty::Impossible => {
                self.ticks() < 600 + spawn && rng.next_int(0.0, 100.0) >= 70
            }
        }
    }

    fn nation_partner_similarly_strong(
        &self,
        pid: &str,
        other: &str,
        rng: &mut PseudoRandom,
    ) -> bool {
        let (t0, t1) = match self.difficulty {
            Difficulty::Easy => (60.0, 70.0),
            Difficulty::Medium => (70.0, 80.0),
            Difficulty::Hard => (75.0, 85.0),
            Difficulty::Impossible => (80.0, 90.0),
        };
        let (s0, s1) = match self.difficulty {
            Difficulty::Easy => (70.0, 80.0),
            Difficulty::Medium => (80.0, 90.0),
            Difficulty::Hard => (85.0, 95.0),
            Difficulty::Impossible => (90.0, 100.0),
        };
        let me_out: f64 = self
            .players
            .get(pid)
            .map(|p| p.outgoing.iter().map(|a| a.troops).sum())
            .unwrap_or(0.0);
        let them_out: f64 = self
            .players
            .get(other)
            .map(|p| p.outgoing.iter().map(|a| a.troops).sum())
            .unwrap_or(0.0);
        let me_t = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0) + me_out;
        let them_t = self.players.get(other).map(|p| p.troops_f64()).unwrap_or(0.0) + them_out;
        let me_tiles = self.players.get(pid).map(|p| p.num_tiles()).unwrap_or(0);
        let them_tiles = self.players.get(other).map(|p| p.num_tiles()).unwrap_or(0);
        let troop_th = me_t * (rng.next_int(t0, t1) as f64 / 100.0);
        let tile_th = me_tiles as f64 * (rng.next_int(s0, s1) as f64 / 100.0);
        them_t > troop_th || (them_tiles as f64 > tile_th && them_t > me_t * 0.5)
    }

    fn alliance_count(&self, pid: &str) -> usize {
        self.alliances.iter().filter(|a| a.involves(pid, a.other(pid))).count()
    }

    fn is_friendly_ids(&self, a: &str, b: &str) -> bool {
        let Some(pa) = self.players.get(a) else {
            return false;
        };
        let Some(pb) = self.players.get(b) else {
            return false;
        };
        self.is_friendly(pa, pb)
    }

    pub(crate) fn player_max_troops(&self, pid: &str) -> f64 {
        let Some(p) = self.players.get(pid) else {
            return 0.0;
        };
        let cities: f64 = self
            .units
            .iter()
            .filter(|u| u.active && u.ty == UnitType::City && u.owner_id == pid && !u.under_construction)
            .map(|u| u.level as f64)
            .sum();
        crate::config::max_troops(p.player_type(), self.difficulty, p.num_tiles(), cities)
    }

    fn nation_handle_structures(&mut self, pid: &str, idx: usize) {
        // Easy: defense-post path is a no-op (no RNG). If the nation can
        // actually afford a later-system structure, fail closed rather than
        // invent a placement heuristic.
        let gold = self.players.get(pid).map(|p| p.gold).unwrap_or(0);
        let later = self.units.iter().any(|u| {
            u.active
                && u.owner_id == pid
                && matches!(
                    u.ty,
                    UnitType::MissileSilo | UnitType::Factory | UnitType::Port | UnitType::Warship
                )
        });
        let _ = (gold, idx, later);
    }

    fn nation_handle_embargoes(&mut self, pid: &str) {
        let others: Vec<String> = self
            .players
            .keys()
            .filter(|id| *id != pid)
            .cloned()
            .collect();
        for other in others {
            let rel = self
                .players
                .get(pid)
                .map(|p| p.relation(&other))
                .unwrap_or(crate::diplomacy::Relation::Neutral);
            let has = self
                .players
                .get(pid)
                .map(|p| p.has_embargo_against(&other))
                .unwrap_or(false);
            if (rel as i32) <= (crate::diplomacy::Relation::Hostile as i32) && !has {
                if let Some(p) = self.players.get_mut(pid) {
                    p.add_embargo(&other, self.ticks, false);
                }
            }
        }
    }

    fn troop_send_cap(&self, pid: &str) -> f64 {
        let Some(p) = self.players.get(pid) else {
            return f64::INFINITY;
        };
        if p.player_type() == PlayerType::Bot {
            return f64::INFINITY;
        }
        let retain = match self.difficulty {
            Difficulty::Hard => 0.75,
            Difficulty::Impossible => 0.9,
            _ => return f64::INFINITY,
        };
        let mut max_neighbor: f64 = 0.0;
        for n in self.nearby(pid).into_iter().flatten() {
            if self.is_allied(pid, &n) {
                continue;
            }
            let Some(other) = self.players.get(&n) else {
                continue;
            };
            if other.player_type() == PlayerType::Bot {
                continue;
            }
            max_neighbor = max_neighbor.max(other.troops_f64());
        }
        let cap = if max_neighbor == 0.0 {
            f64::INFINITY
        } else {
            (p.troops_f64() - (max_neighbor * retain).ceil()).max(0.0)
        };
        let incoming: f64 = self
            .players
            .values()
            .flat_map(|o| o.outgoing.iter())
            .filter(|a| a.active && a.target_player_id.as_deref() == Some(pid))
            .map(|a| a.troops)
            .sum();
        if incoming > 0.0 {
            cap.max(incoming)
        } else {
            cap
        }
    }

    fn is_attack_too_weak(&self, pid: &str, target: &str, troops: f64) -> bool {
        let Some(p) = self.players.get(pid) else {
            return false;
        };
        if p.player_type() == PlayerType::Bot {
            return false;
        }
        if !matches!(self.difficulty, Difficulty::Hard | Difficulty::Impossible) {
            return false;
        }
        let incoming = self.players.values().any(|o| {
            o.outgoing
                .iter()
                .any(|a| a.active && a.target_player_id.as_deref() == Some(pid))
        });
        if incoming {
            return false;
        }
        let other_t = self.players.get(target).map(|o| o.troops_f64()).unwrap_or(0.0);
        troops < other_t * 0.2
    }

    fn find_random_boat_target(
        &mut self,
        pid: &str,
        tile: TileRef,
        bordering_enemies: &[String],
        high_interest: bool,
        rng: &mut PseudoRandom,
    ) -> Option<TileRef> {
        let x = self.map.x(tile) as i32;
        let y = self.map.y(tile) as i32;
        let my_small = self.players.get(pid).map(|p| p.small_id)?;
        let my_troops = self.players.get(pid).map(|p| p.troops_f64()).unwrap_or(0.0);
        let mut unreachable: HashSet<String> = HashSet::new();
        for _ in 0..500 {
            let rand_x = rng.next_int((x - 150) as f64, (x + 150) as f64);
            let rand_y = rng.next_int((y - 150) as f64, (y + 150) as f64);
            if rand_x < 0 || rand_y < 0 {
                continue;
            }
            let Ok(rand_tile) = self.map.ref_xy(rand_x as u32, rand_y as u32) else {
                continue;
            };
            if !self.map.is_land(rand_tile) || self.map.is_impassable(rand_tile) {
                continue;
            }
            let os = self.map.owner_id(rand_tile);
            if os == my_small {
                continue;
            }
            let owner_id = self.owner_id_opt(os);
            if let Some(ref oid) = owner_id {
                if unreachable.contains(oid) {
                    continue;
                }
                if bordering_enemies.iter().any(|e| e == oid) {
                    continue;
                }
                let other_troops = self.players.get(oid).map(|p| p.troops_f64()).unwrap_or(0.0);
                if other_troops > my_troops {
                    continue;
                }
            }
            let matches = if high_interest {
                owner_id.as_ref().map_or(true, |oid| {
                    self.players
                        .get(oid)
                        .map(|p| p.player_type() == PlayerType::Bot)
                        .unwrap_or(false)
                })
            } else {
                owner_id
                    .as_ref()
                    .map_or(true, |oid| !self.is_allied(pid, oid))
            };
            if !matches {
                continue;
            }
            if self.can_build_boat(pid, rand_tile).is_none() {
                if let Some(oid) = owner_id {
                    unreachable.insert(oid);
                }
                continue;
            }
            return Some(rand_tile);
        }
        None
    }

    fn nation_attack_with_random_boat(
        &mut self,
        pid: &str,
        bordering_enemies: &[String],
        rng: &mut PseudoRandom,
    ) {
        if self.is_unit_disabled(UnitType::TransportShip) {
            return;
        }
        if self.boat_count(pid) >= BOAT_MAX_NUMBER {
            return;
        }
        let shores: Vec<TileRef> = self
            .players
            .get(pid)
            .map(|p| {
                p.border_tiles
                    .iter()
                    .copied()
                    .filter(|&t| self.map.is_shore(t))
                    .collect()
            })
            .unwrap_or_default();
        if shores.is_empty() {
            return;
        }
        let src = rng.rand_element(&shores);
        let mut dst = self.find_random_boat_target(pid, src, bordering_enemies, true, rng);
        if dst.is_none() {
            dst = self.find_random_boat_target(pid, src, bordering_enemies, false, rng);
        }
        let Some(dst) = dst else {
            return;
        };
        let owner_sid = self.map.owner_id(dst);
        let cap = if owner_sid == 0 {
            f64::INFINITY
        } else {
            self.troop_send_cap(pid)
        };
        let troops = self
            .players
            .get(pid)
            .map(|p| p.troops_f64() / 5.0)
            .unwrap_or(0.0)
            .min(cap);
        if troops < 1.0 {
            return;
        }
        if owner_sid != 0 {
            if let Some(other) = self.player_by_small(owner_sid) {
                let other_id = other.id().to_string();
                if self.is_attack_too_weak(pid, &other_id, troops) {
                    return;
                }
            }
        }
        self.uninit.push(Exec::Boat(BoatExec {
            active: true,
            inited: false,
            owner_id: pid.to_string(),
            ref_tile: dst,
            troops,
            dst: None,
            src: None,
            unit_id: None,
            last_move: 0,
            stepper: PathStepper::new(),
        }));
    }

    fn nation_maybe_attack(
        &mut self,
        pid: &str,
        idx: usize,
        trigger: f64,
        reserve: f64,
        expand: f64,
    ) {
        let mut rng = {
            let Exec::Nation(n) = &self.execs[idx] else {
                return;
            };
            n.rng.clone()
        };
        // Official maybeAttack: TN (land or nearby) first, then boat/alliance/best.
        let nearby = self.nearby(pid);
        let has_tn = nearby.iter().any(|n| n.is_none()) || self.has_land_tn_border(pid);
        if has_tn {
            let sent = self.ai_send_attack_official(pid, None, expand, reserve, &mut rng, false);
            if sent {
                if let Exec::Nation(n) = &mut self.execs[idx] {
                    n.rng = rng;
                }
                return;
            }
        }
        let mut enemies: Vec<String> = nearby.into_iter().flatten().collect();
        enemies.retain(|id| !self.is_allied(pid, id));
        enemies.sort_by_key(|id| self.players.get(id).map(|p| p.troops).unwrap_or(0));
        if enemies.is_empty() {
            if rng.chance(5) {
                self.nation_attack_with_random_boat(pid, &enemies, &mut rng);
            }
        } else if rng.chance(10) {
            self.nation_attack_with_random_boat(pid, &enemies, &mut rng);
            if let Exec::Nation(n) = &mut self.execs[idx] {
                n.rng = rng;
            }
            return;
        } else {
            for e in &enemies {
                if !rng.chance(30) {
                    continue;
                }
                let acceptable = self.players.get(e).map(|p| {
                    p.player_type() != PlayerType::Bot || self.difficulty == Difficulty::Easy
                }).unwrap_or(false);
                if acceptable
                    && self.can_send_ally(pid, e)
                    && self.nation_alliance_decision(pid, e, false, &mut rng)
                {
                    self.uninit.push(Exec::AllianceReq(AllianceReqExec {
                        active: true,
                        inited: false,
                        requestor: pid.to_string(),
                        recipient: e.clone(),
                    }));
                }
            }
        }
        self.nation_attack_best_target(pid, trigger, reserve, expand, &mut rng);
        if let Exec::Nation(n) = &mut self.execs[idx] {
            n.rng = rng;
        }
    }

    pub fn find_boat_in_region(&mut self, player_id: &str, region: i32) -> Option<TileRef> {
        let ri = region as usize;
        if ri < self.boat_cache.len() {
            if let Some(cached) = self.boat_cache[ri] {
                if self.can_build_boat(player_id, cached).is_some() {
                    return Some(cached);
                }
                self.boat_cache[ri] = None;
            }
        }
        let (x0, y0, x1, y1) = crate::game::region_bounds(&self.map, region);
        let mut tries = 0;
        let mut y = y0;
        while y < y1 && tries < 64 {
            let mut x = x0;
            while x < x1 && tries < 64 {
                if let Ok(tile) = self.map.ref_xy(x, y) {
                    if self.map.is_land(tile) && self.map.is_shoreline(tile) {
                        let owner = self.map.owner_id(tile);
                        if owner != self.players.get(player_id).map(|p| p.small_id).unwrap_or(0) {
                            tries += 1;
                            if self.can_build_boat(player_id, tile).is_some() {
                                if ri < self.boat_cache.len() {
                                    self.boat_cache[ri] = Some(tile);
                                }
                                return Some(tile);
                            }
                        }
                    }
                }
                x += 2;
            }
            y += 2;
        }
        None
    }

    pub fn find_build_in_region(
        &mut self,
        player_id: &str,
        ty: UnitType,
        region: i32,
    ) -> Option<TileRef> {
        if ty == UnitType::Warship {
            return self.find_warship_in_region(player_id, region);
        }
        if ty.is_nuke() {
            return self.find_nuke_in_region(player_id, ty, region);
        }
        let (x0, y0, x1, y1) = crate::game::region_bounds(&self.map, region);
        let mut tries = 0;
        let mut y = y0;
        while y < y1 && tries < 64 {
            let mut x = x0;
            while x < x1 && tries < 64 {
                if let Ok(tile) = self.map.ref_xy(x, y) {
                    if self.map.has_owner(tile)
                        && self.map.owner_id(tile)
                            == self.players.get(player_id).map(|p| p.small_id).unwrap_or(0)
                    {
                        tries += 1;
                        if let Some(t) = self.can_build(player_id, ty, tile) {
                            return Some(t);
                        }
                    }
                }
                x += 2;
            }
            y += 2;
        }
        None
    }

    pub fn find_build_anywhere(&mut self, player_id: &str, ty: UnitType) -> Option<TileRef> {
        if ty == UnitType::Warship {
            return self.find_warship_anywhere(player_id);
        }
        if ty.is_nuke() {
            return self.find_nuke_anywhere(player_id, ty);
        }
        let tiles: Vec<TileRef> = self
            .players
            .get(player_id)
            .map(|p| p.tiles.iter().copied().collect())
            .unwrap_or_default();
        for t in tiles {
            if let Some(s) = self.can_build(player_id, ty, t) {
                return Some(s);
            }
        }
        None
    }

    fn find_warship_in_region(&mut self, player_id: &str, region: i32) -> Option<TileRef> {
        let (x0, y0, x1, y1) = crate::game::region_bounds(&self.map, region);
        let mut tries = 0;
        let mut y = y0;
        while y < y1 && tries < 64 {
            let mut x = x0;
            while x < x1 && tries < 64 {
                if let Ok(tile) = self.map.ref_xy(x, y) {
                    if self.map.is_water(tile) {
                        tries += 1;
                        if self.can_build(player_id, UnitType::Warship, tile).is_some() {
                            return Some(tile);
                        }
                    }
                }
                x += 2;
            }
            y += 2;
        }
        None
    }

    fn find_nuke_in_region(
        &mut self,
        player_id: &str,
        ty: UnitType,
        region: i32,
    ) -> Option<TileRef> {
        let my_sid = self.players.get(player_id).map(|p| p.small_id).unwrap_or(0);
        let (x0, y0, x1, y1) = crate::game::region_bounds(&self.map, region);
        let mut tries = 0;
        let mut y = y0;
        while y < y1 && tries < 64 {
            let mut x = x0;
            while x < x1 && tries < 64 {
                if let Ok(tile) = self.map.ref_xy(x, y) {
                    if !self.map.is_impassable(tile)
                        && !(ty == UnitType::Mirv && !self.map.has_owner(tile))
                    {
                        let owner = self.map.owner_id(tile);
                        if owner != my_sid {
                            tries += 1;
                            if self.can_build(player_id, ty, tile).is_some() {
                                return Some(tile);
                            }
                        }
                    }
                }
                x += 2;
            }
            y += 2;
        }
        None
    }

    fn find_nuke_anywhere(&mut self, player_id: &str, ty: UnitType) -> Option<TileRef> {
        // Official `game.players()` is Map insertion order = smallID order.
        let mut others: Vec<(u16, String)> = self
            .alive_players()
            .into_iter()
            .filter(|p| p.id() != player_id)
            .map(|p| (p.small_id, p.id().to_string()))
            .collect();
        others.sort_by_key(|(sid, _)| *sid);
        for (_, oid) in others {
            let tiles: Vec<TileRef> = self
                .players
                .get(&oid)
                .map(|p| p.tiles.iter().copied().collect())
                .unwrap_or_default();
            for tile in tiles {
                if self.can_build(player_id, ty, tile).is_some() {
                    return Some(tile);
                }
            }
        }
        if ty == UnitType::Mirv {
            return None;
        }
        for r in 0..crate::spec::NUM_REGIONS as i32 {
            if let Some(tile) = self.find_nuke_in_region(player_id, ty, r) {
                return Some(tile);
            }
        }
        None
    }

    fn find_warship_anywhere(&mut self, player_id: &str) -> Option<TileRef> {
        let ports: Vec<TileRef> = self
            .units
            .iter()
            .filter(|u| {
                u.active && !u.under_construction && u.ty == UnitType::Port && u.owner_id == player_id
            })
            .map(|u| u.tile)
            .collect();
        for port in ports {
            if let Some(tile) = self.water_near_port(player_id, port) {
                return Some(tile);
            }
        }
        None
    }

    fn water_near_port(&mut self, player_id: &str, port: TileRef) -> Option<TileRef> {
        let mut nbuf = [0u32; 4];
        let n = self.map.neighbors4(port, &mut nbuf);
        for i in 0..n {
            let t = nbuf[i];
            if self.map.is_water(t) && self.can_build(player_id, UnitType::Warship, t).is_some() {
                return Some(t);
            }
        }
        let mut seen = HashSet::new();
        seen.insert(port);
        let mut queue = Vec::new();
        for i in 0..n {
            let t = nbuf[i];
            if self.map.is_water(t) && seen.insert(t) {
                queue.push(t);
            }
        }
        let mut i = 0;
        while i < queue.len() && i < 64 {
            let tile = queue[i];
            i += 1;
            if self.can_build(player_id, UnitType::Warship, tile).is_some() {
                return Some(tile);
            }
            let nn = self.map.neighbors4(tile, &mut nbuf);
            for j in 0..nn {
                let t = nbuf[j];
                if self.map.is_water(t) && seen.insert(t) {
                    queue.push(t);
                }
            }
        }
        None
    }

    pub fn incoming_requests(&self) -> &[AllianceRequest] {
        &self.incoming_requests
    }

    pub fn alliances(&self) -> &[Alliance] {
        &self.alliances
    }
}

fn player_border_center(map: &GameMap, p: &Player) -> (u32, u32) {
    let mut min_x = u32::MAX;
    let mut min_y = u32::MAX;
    let mut max_x = 0u32;
    let mut max_y = 0u32;
    for &t in &p.border_tiles {
        let x = map.x(t);
        let y = map.y(t);
        min_x = min_x.min(x);
        min_y = min_y.min(y);
        max_x = max_x.max(x);
        max_y = max_y.max(y);
    }
    if min_x == u32::MAX {
        return (0, 0);
    }
    (min_x + (max_x - min_x) / 2, min_y + (max_y - min_y) / 2)
}

fn closest_two_tiles(
    map: &GameMap,
    xs: &[TileRef],
    ys: &[TileRef],
) -> Option<(TileRef, TileRef)> {
    let mut x_sorted = xs.to_vec();
    let mut y_sorted = ys.to_vec();
    x_sorted.sort_by_key(|t| map.x(*t));
    y_sorted.sort_by_key(|t| map.x(*t));
    if x_sorted.is_empty() || y_sorted.is_empty() {
        return None;
    }
    let mut i = 0usize;
    let mut j = 0usize;
    let mut min_d = i32::MAX;
    let mut result = (x_sorted[0], y_sorted[0]);
    while i < x_sorted.len() && j < y_sorted.len() {
        let cx = x_sorted[i];
        let cy = y_sorted[j];
        let d = (map.x(cx) as i32 - map.x(cy) as i32).abs()
            + (map.y(cx) as i32 - map.y(cy) as i32).abs();
        if d < min_d {
            min_d = d;
            result = (cx, cy);
        }
        if i + 1 == x_sorted.len() {
            j += 1;
        } else if j + 1 == y_sorted.len() {
            i += 1;
        } else if map.x(cx) < map.x(cy) {
            i += 1;
        } else {
            j += 1;
        }
    }
    Some(result)
}

fn spawn_tiles(map: &GameMap, center: TileRef, require_all: bool) -> Vec<TileRef> {
    let all = map.bfs_euclid_center(center, 4.0);
    let ok = |t: TileRef| map.is_land(t) && !map.has_owner(t) && !map.is_impassable(t);
    if require_all {
        if all.iter().any(|&t| !ok(t)) {
            return Vec::new();
        }
        all
    } else {
        all.into_iter().filter(|&t| ok(t)).collect()
    }
}

fn is_active(e: &Exec) -> bool {
    match e {
        Exec::Spawn(s) => s.active,
        Exec::Player(p) => p.active,
        Exec::Attack(a) => a.active,
        Exec::Retreat(r) => r.active,
        Exec::WinCheck(w) => w.active,
        Exec::Boat(b) => b.active,
        Exec::Construction(c) => c.active,
        Exec::AllianceReq(a) => a.active,
        Exec::BreakAlly(b) => b.active,
        Exec::Embargo(e) => e.active,
        Exec::Tribe(t) => t.active,
        Exec::Nation(n) => n.active,
        Exec::Later(e) => e.is_active(),
    }
}

fn during_spawn(e: &Exec) -> bool {
    matches!(e, Exec::Spawn(_) | Exec::Nation(_))
}

fn keep_exec(e: &Exec, in_spawn: bool) -> bool {
    if in_spawn {
        !during_spawn(e) || is_active(e)
    } else {
        is_active(e)
    }
}

struct FlatHeap {
    pri: Vec<f32>,
    tiles: Vec<TileRef>,
}

impl FlatHeap {
    fn new() -> Self {
        Self {
            pri: Vec::new(),
            tiles: Vec::new(),
        }
    }
    fn clear(&mut self) {
        self.pri.clear();
        self.tiles.clear();
    }
    fn is_empty(&self) -> bool {
        self.pri.is_empty()
    }
    fn enqueue(&mut self, tile: TileRef, priority: f32) {
        let mut i = self.pri.len();
        self.pri.push(priority);
        self.tiles.push(tile);
        while i > 0 {
            let parent = (i - 1) >> 1;
            if priority >= self.pri[parent] {
                break;
            }
            self.pri[i] = self.pri[parent];
            self.tiles[i] = self.tiles[parent];
            i = parent;
        }
        self.pri[i] = priority;
        self.tiles[i] = tile;
    }
    fn dequeue(&mut self) -> TileRef {
        let top = self.tiles[0];
        let last_pri = self.pri.pop().unwrap();
        let last_tile = self.tiles.pop().unwrap();
        if self.pri.is_empty() {
            return top;
        }
        let mut i = 0;
        loop {
            let left = (i << 1) + 1;
            if left >= self.pri.len() {
                break;
            }
            let right = left + 1;
            let child = if right < self.pri.len() && self.pri[right] < self.pri[left] {
                right
            } else {
                left
            };
            if last_pri <= self.pri[child] {
                break;
            }
            self.pri[i] = self.pri[child];
            self.tiles[i] = self.tiles[child];
            i = child;
        }
        self.pri[i] = last_pri;
        self.tiles[i] = last_tile;
        top
    }
}

/// Boat dest existence via official mini-map water components (SpatialQuery).
pub fn boat_dest_possible(game: &Game, player_id: &str, tile: TileRef) -> bool {
    let Some(player) = game.player(player_id) else {
        return false;
    };
    target_transport_tile(
        &game.map,
        &game.mini_map,
        &game.mini_water,
        game.map.owner_id(tile),
        player.border_tiles.iter().copied(),
        tile,
    )
    .is_some()
}

fn closest_reachable_shore(
    game: &Game,
    target_sid: u16,
    from: TileRef,
    reachable: &HashSet<i32>,
) -> Option<TileRef> {
    let max_dist = 50u32;
    let mut best = None;
    let mut best_dist = u32::MAX;
    let mut seen = vec![false; game.map.len()];
    let mut stack = Vec::new();
    if game.map.manhattan(from, from) <= max_dist {
        seen[from as usize] = true;
        stack.push(from);
    }
    let mut nbuf = [0u32; 4];
    while let Some(cur) = stack.pop() {
        if game.map.is_shore(cur) && game.map.owner_id(cur) == target_sid {
            if let Some(c) = game.water.of_tile(&game.map, cur) {
                if reachable.contains(&c) {
                    let d = game.map.manhattan(from, cur);
                    if d < best_dist {
                        best_dist = d;
                        best = Some(cur);
                    }
                }
            }
        }
        let n = game.map.neighbors4(cur, &mut nbuf);
        for i in 0..n {
            let nb = nbuf[i];
            if !seen[nb as usize] && game.map.manhattan(from, nb) <= max_dist {
                seen[nb as usize] = true;
                stack.push(nb);
            }
        }
    }
    best
}

pub fn find_spawn_tile(map: &GameMap, region: i32) -> Option<TileRef> {
    let (x0, y0, x1, y1) = region_bounds(map, region);
    let cx = (x0 + x1) >> 1;
    let cy = (y0 + y1) >> 1;
    let mut candidates = vec![map.ref_xy(cx, cy).ok()?];
    for i in 0..64u32 {
        let w = (x1 - x0).max(1);
        let h = (y1 - y0).max(1);
        let x = x0 + (i * 37) % w;
        let y = y0 + (i * 53) % h;
        if let Ok(t) = map.ref_xy(x, y) {
            candidates.push(t);
        }
    }
    for tile in candidates {
        if map.is_land(tile) && !map.has_owner(tile) && !map.is_impassable(tile) {
            return Some(tile);
        }
    }
    None
}

pub fn region_of_tile(map: &GameMap, tile: TileRef) -> usize {
    let w = map.width() as usize;
    let h = map.height() as usize;
    let x = (tile as usize) % w;
    let y = (tile as usize) / w;
    let rx = ((x * crate::spec::REGION_GRID) / w).min(crate::spec::REGION_GRID - 1);
    let ry = ((y * crate::spec::REGION_GRID) / h).min(crate::spec::REGION_GRID - 1);
    ry * crate::spec::REGION_GRID + rx
}

pub fn region_bounds(map: &GameMap, region: i32) -> (u32, u32, u32, u32) {
    let rg = crate::spec::REGION_GRID as u32;
    let rx = (region as u32) % rg;
    let ry = (region as u32) / rg;
    let w = map.width();
    let h = map.height();
    let x0 = (rx * w) / rg;
    let y0 = (ry * h) / rg;
    let x1 = (((rx + 1) * w) / rg).min(w);
    let y1 = (((ry + 1) * h) / rg).min(h);
    (x0, y0, x1, y1)
}
