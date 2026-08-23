//! Player, attack, and insertion-ordered tile sets.

use indexmap::IndexSet;

use std::collections::HashMap;

use crate::diplomacy::{AllianceRequest, Embargo, Relation};
use crate::map::TileRef;
use crate::rng::PseudoRandom;
use crate::spec::PlayerType;
use crate::units::{Unit, UnitType};

#[derive(Clone, Debug)]
pub struct PlayerInfo {
    pub name: String,
    pub player_type: PlayerType,
    pub client_id: Option<String>,
    pub id: String,
}

#[derive(Clone, Debug)]
pub struct Attack {
    pub id: String,
    pub target_small_id: u16,
    pub target_player_id: Option<String>,
    pub attacker_small_id: u16,
    pub troops: f64,
    pub retreating: bool,
    pub retreated: bool,
    pub active: bool,
    pub source_tile: Option<TileRef>,
    pub(crate) border: IndexSet<TileRef>,
}

impl Attack {
    pub fn border_size(&self) -> usize {
        self.border.len()
    }
    pub fn clear_border(&mut self) {
        self.border.clear();
    }
    pub fn add_border_tile(&mut self, tile: TileRef) {
        self.border.insert(tile);
    }
    pub fn remove_border_tile(&mut self, tile: TileRef) {
        self.border.shift_remove(&tile);
    }
}

pub struct Player {
    pub small_id: u16,
    pub info: PlayerInfo,
    pub troops: i64,
    pub gold: i64,
    pub tiles: IndexSet<TileRef>,
    pub border_tiles: IndexSet<TileRef>,
    pub spawn_tile: Option<TileRef>,
    pub last_tile_change: u32,
    pub rng: PseudoRandom,
    pub outgoing: Vec<Attack>,
    pub incoming: Vec<String>, // attack ids, resolved via game
    pub traitor: bool,
    pub marked_traitor_tick: i32,
    pub units: Vec<u32>,
    pub constructed: HashMap<UnitType, u32>,
    pub relations: HashMap<String, f64>,
    pub embargoes: HashMap<String, Embargo>,
    pub outgoing_requests: Vec<AllianceRequest>,
    pub past_outgoing_requests: Vec<AllianceRequest>,
    pub last_delete_unit_tick: i32,
    pub attacks_sent: u64,
    pub disconnected: bool,
    pub in_doomsday: bool,
    pub doomsday_ticks: u32,
}

impl Player {
    pub fn new(small_id: u16, info: PlayerInfo, start_troops: f64, id_hash_seed: i32) -> Self {
        Self {
            small_id,
            info,
            troops: start_troops.floor() as i64,
            gold: 0,
            tiles: IndexSet::new(),
            border_tiles: IndexSet::new(),
            spawn_tile: None,
            last_tile_change: 0,
            rng: PseudoRandom::new(id_hash_seed),
            outgoing: Vec::new(),
            incoming: Vec::new(),
            traitor: false,
            marked_traitor_tick: -1,
            units: Vec::new(),
            constructed: HashMap::new(),
            relations: HashMap::new(),
            embargoes: HashMap::new(),
            outgoing_requests: Vec::new(),
            past_outgoing_requests: Vec::new(),
            last_delete_unit_tick: -1,
            attacks_sent: 0,
            disconnected: false,
            in_doomsday: false,
            doomsday_ticks: 0,
        }
    }

    pub fn id(&self) -> &str {
        &self.info.id
    }
    pub fn player_type(&self) -> PlayerType {
        self.info.player_type
    }
    pub fn is_alive(&self) -> bool {
        !self.tiles.is_empty()
    }
    pub fn has_spawned(&self) -> bool {
        self.spawn_tile.is_some()
    }
    pub fn num_tiles(&self) -> u32 {
        self.tiles.len() as u32
    }
    pub fn troops_f64(&self) -> f64 {
        self.troops as f64
    }

    pub fn add_troops(&mut self, n: f64) {
        if n < 0.0 {
            self.remove_troops((-n).floor() as i64);
            return;
        }
        self.troops += n.floor() as i64;
    }
    pub fn remove_troops(&mut self, n: i64) -> i64 {
        if n <= 0 {
            return 0;
        }
        let take = n.min(self.troops);
        self.troops -= take;
        take
    }
    pub fn add_gold(&mut self, n: i64) {
        self.gold += n;
    }
    pub fn remove_gold(&mut self, n: i64) -> i64 {
        let take = n.min(self.gold).max(0);
        self.gold -= take;
        take
    }

    pub fn core_hash(&self) -> f64 {
        crate::rng::simple_hash_abs_f64(self.id()) * (self.troops_f64() + self.num_tiles() as f64)
    }

    pub fn record_constructed(&mut self, ty: UnitType) {
        *self.constructed.entry(ty).or_insert(0) += 1;
    }

    pub fn units_constructed(&self, ty: UnitType) -> u32 {
        self.constructed.get(&ty).copied().unwrap_or(0)
    }

    pub fn relation_value(&self, other: &str) -> f64 {
        self.relations.get(other).copied().unwrap_or(0.0)
    }

    pub fn relation(&self, other: &str) -> Relation {
        Relation::from_value(self.relation_value(other))
    }

    pub fn update_relation(&mut self, other: &str, delta: f64) {
        let cur = self.relation_value(other);
        self.relations
            .insert(other.to_string(), crate::diplomacy::within_rel(cur + delta));
    }

    pub fn decay_relations(&mut self) {
        for v in self.relations.values_mut() {
            *v = crate::diplomacy::decay_relation(*v);
        }
    }

    pub fn has_embargo_against(&self, other: &str) -> bool {
        self.embargoes.contains_key(other)
    }

    pub fn add_embargo(&mut self, other: &str, ticks: u32, temporary: bool) {
        if let Some(e) = self.embargoes.get(other) {
            if !e.temporary {
                return;
            }
        }
        self.embargoes.insert(
            other.to_string(),
            Embargo {
                target: other.to_string(),
                created_at: ticks,
                temporary,
            },
        );
    }

    pub fn stop_embargo(&mut self, other: &str) {
        self.embargoes.remove(other);
    }

    pub fn end_temporary_embargo(&mut self, other: &str) {
        if let Some(e) = self.embargoes.get(other) {
            if !e.temporary {
                return;
            }
        }
        self.stop_embargo(other);
    }

    pub fn shore_tiles(&self, map: &crate::map::GameMap) -> Vec<TileRef> {
        self.border_tiles
            .iter()
            .copied()
            .filter(|&t| map.is_shore(t))
            .collect()
    }
}
