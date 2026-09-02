//! Official unit types, costs, and construction durations.

use crate::config::{
    WARSHIP_MAX_HEALTH, WARSHIP_MAX_VETERANCY, WARSHIP_VETERANCY_HEALTH_BONUS,
    WARSHIP_VETERANCY_TRADE_CAPTURES, WARSHIP_VETERANCY_TRANSPORT_KILLS,
};
use crate::map::TileRef;
use crate::spec::PlayerType;

pub const UNIT_HEAD_ORDER: [UnitType; 10] = [
    UnitType::City,
    UnitType::DefensePost,
    UnitType::SamLauncher,
    UnitType::MissileSilo,
    UnitType::Port,
    UnitType::Factory,
    UnitType::AtomBomb,
    UnitType::HydrogenBomb,
    UnitType::Mirv,
    UnitType::Warship,
];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnitType {
    TransportShip,
    Warship,
    Shell,
    SamMissile,
    Port,
    AtomBomb,
    HydrogenBomb,
    TradeShip,
    MissileSilo,
    DefensePost,
    SamLauncher,
    City,
    Mirv,
    MirvWarhead,
    Train,
    Factory,
}

impl UnitType {
    pub fn as_digest_number(self) -> i32 {
        // Official digest writes `u.type() as number` on a string enum, which
        // ToInt32s to 0. Keep the same so tileDigest stays comparable.
        0
    }

    pub fn official_name(self) -> &'static str {
        match self {
            Self::TransportShip => "Transport",
            Self::Warship => "Warship",
            Self::Shell => "Shell",
            Self::SamMissile => "SAMMissile",
            Self::Port => "Port",
            Self::AtomBomb => "Atom Bomb",
            Self::HydrogenBomb => "Hydrogen Bomb",
            Self::TradeShip => "Trade Ship",
            Self::MissileSilo => "Missile Silo",
            Self::DefensePost => "Defense Post",
            Self::SamLauncher => "SAM Launcher",
            Self::City => "City",
            Self::Mirv => "MIRV",
            Self::MirvWarhead => "MIRV Warhead",
            Self::Train => "Train",
            Self::Factory => "Factory",
        }
    }

    pub fn is_structure(self) -> bool {
        matches!(
            self,
            Self::City
                | Self::DefensePost
                | Self::SamLauncher
                | Self::MissileSilo
                | Self::Port
                | Self::Factory
        )
    }

    pub fn is_nuke(self) -> bool {
        matches!(self, Self::AtomBomb | Self::HydrogenBomb | Self::Mirv)
    }

    pub fn from_official_name(name: &str) -> Option<Self> {
        Some(match name {
            "Transport" | "Transport Ship" => Self::TransportShip,
            "Warship" => Self::Warship,
            "Shell" => Self::Shell,
            "SAMMissile" | "SAM Missile" => Self::SamMissile,
            "Port" => Self::Port,
            "Atom Bomb" => Self::AtomBomb,
            "Hydrogen Bomb" => Self::HydrogenBomb,
            "Trade Ship" => Self::TradeShip,
            "Missile Silo" => Self::MissileSilo,
            "Defense Post" => Self::DefensePost,
            "SAM Launcher" => Self::SamLauncher,
            "City" => Self::City,
            "MIRV" => Self::Mirv,
            "MIRV Warhead" => Self::MirvWarhead,
            "Train" => Self::Train,
            "Factory" => Self::Factory,
            _ => return None,
        })
    }

    pub fn from_head(unit: i32) -> Self {
        UNIT_HEAD_ORDER
            .get(unit as usize)
            .copied()
            .unwrap_or(Self::City)
    }
}

pub fn unit_cost(ty: UnitType, constructed: u32) -> i64 {
    match ty {
        UnitType::TransportShip | UnitType::TradeShip | UnitType::Train | UnitType::Shell
        | UnitType::SamMissile | UnitType::MirvWarhead => 0,
        UnitType::City | UnitType::Port | UnitType::Factory => {
            (2i64.pow(constructed.min(20)) * 125_000).min(1_000_000)
        }
        UnitType::DefensePost => ((constructed as i64 + 1) * 50_000).min(250_000),
        UnitType::SamLauncher => ((constructed as i64 + 1) * 1_500_000).min(3_000_000),
        UnitType::MissileSilo => 1_000_000,
        UnitType::Warship => ((constructed as i64 + 1) * 250_000).min(1_000_000),
        UnitType::AtomBomb => 750_000,
        UnitType::HydrogenBomb => 5_000_000,
        UnitType::Mirv => 25_000_000,
    }
}

pub fn construction_duration(ty: UnitType) -> u32 {
    match ty {
        UnitType::City | UnitType::Factory => 2 * 10,
        UnitType::DefensePost | UnitType::Port => 5 * 10,
        UnitType::MissileSilo => 10 * 10,
        UnitType::SamLauncher => 30 * 10,
        _ => 0,
    }
}

pub const BOAT_MAX_NUMBER: u32 = 3;
pub const STRUCTURE_MIN_DIST: u32 = 15;
pub const DEFENSE_POST_RANGE: u32 = 30;
pub const CITY_TROOP_INCREASE: f64 = 250_000.0;
pub const ALLIANCE_REQUEST_DURATION: u32 = 20 * 10;
pub const ALLIANCE_REQUEST_COOLDOWN: u32 = 30 * 10;
pub const ALLIANCE_DURATION: u32 = 300 * 10;
pub const TEMPORARY_EMBARGO_DURATION: u32 = 300 * 10;
pub const DELETE_UNIT_COOLDOWN: u32 = 30 * 10;
pub const TRAITOR_DURATION: u32 = 30 * 10;
pub const RADIUS_PORT_SPAWN: u32 = 20;

#[derive(Clone, Debug)]
pub struct Unit {
    pub id: u32,
    pub ty: UnitType,
    pub owner_small: u16,
    pub owner_id: String,
    pub tile: TileRef,
    pub last_tile: TileRef,
    pub active: bool,
    pub under_construction: bool,
    pub troops: f64,
    pub target_tile: Option<TileRef>,
    pub retreating: bool,
    pub level: u32,
    pub marked_for_deletion: bool,
    pub train_station: bool,
    pub health: i32,
    pub max_health: i32,
    pub target_unit: Option<u32>,
    pub missile_queue: Vec<u32>,
    pub targetable: bool,
    pub targeted_by_sam: bool,
    pub trajectory: Vec<TileRef>,
    pub trajectory_index: usize,
    pub patrol_tile: Option<TileRef>,
    pub warship_state: WarshipMode,
    pub retreat_port: Option<TileRef>,
    pub last_safe_pirates: u32,
    /// Official `Unit.destroyer()` after `delete(true, attacker)`.
    pub destroyer_id: Option<String>,
    /// Official `warshipState.veterancy` (0 for non-warships).
    pub veterancy: u32,
    /// Shared transport-kill / trade-capture progress toward the next level.
    pub veterancy_progress: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WarshipMode {
    None,
    Patrolling,
    Retreating,
    Docked,
}

impl Unit {
    pub fn new(
        id: u32,
        ty: UnitType,
        owner_small: u16,
        owner_id: String,
        tile: TileRef,
        troops: f64,
        target_tile: Option<TileRef>,
    ) -> Self {
        Self {
            id,
            ty,
            owner_small,
            owner_id,
            tile,
            last_tile: tile,
            active: true,
            under_construction: false,
            troops,
            target_tile,
            retreating: false,
            level: 1,
            marked_for_deletion: false,
            train_station: false,
            health: if ty == UnitType::Warship { 1000 } else { 0 },
            max_health: if ty == UnitType::Warship { 1000 } else { 0 },
            target_unit: None,
            missile_queue: Vec::new(),
            targetable: false,
            targeted_by_sam: false,
            trajectory: Vec::new(),
            trajectory_index: 0,
            patrol_tile: None,
            warship_state: if ty == UnitType::Warship {
                WarshipMode::Patrolling
            } else {
                WarshipMode::None
            },
            retreat_port: None,
            last_safe_pirates: 0,
            destroyer_id: None,
            veterancy: 0,
            veterancy_progress: 0,
        }
    }

    pub fn increase_veterancy(&mut self) {
        if self.ty != UnitType::Warship || self.veterancy >= WARSHIP_MAX_VETERANCY {
            return;
        }
        self.veterancy += 1;
        self.max_health = warship_max_health_with_veterancy(self.veterancy);
    }

    pub fn record_kill(&mut self, target_ty: UnitType) {
        if self.ty != UnitType::Warship {
            return;
        }
        if target_ty == UnitType::Warship {
            self.veterancy_progress = 0;
            self.increase_veterancy();
        } else if target_ty == UnitType::TransportShip {
            self.add_veterancy_progress(UnitType::TransportShip);
        }
    }

    pub fn record_trade_capture(&mut self) {
        self.add_veterancy_progress(UnitType::TradeShip);
    }

    fn add_veterancy_progress(&mut self, source: UnitType) {
        if self.ty != UnitType::Warship || self.veterancy >= WARSHIP_MAX_VETERANCY {
            return;
        }
        let points_per_level = WARSHIP_VETERANCY_TRANSPORT_KILLS * WARSHIP_VETERANCY_TRADE_CAPTURES;
        self.veterancy_progress += if source == UnitType::TransportShip {
            WARSHIP_VETERANCY_TRADE_CAPTURES
        } else {
            WARSHIP_VETERANCY_TRANSPORT_KILLS
        };
        while self.veterancy_progress >= points_per_level
            && self.veterancy < WARSHIP_MAX_VETERANCY
        {
            self.veterancy_progress -= points_per_level;
            self.increase_veterancy();
        }
        if self.veterancy >= WARSHIP_MAX_VETERANCY {
            self.veterancy_progress = 0;
        }
    }

    pub fn hash(&self) -> i32 {
        (self.tile as i32).wrapping_add(
            crate::rng::simple_hash(self.ty.official_name()).wrapping_mul(self.id as i32),
        )
    }

    pub fn hash_f64(&self) -> f64 {
        self.tile as f64
            + crate::rng::simple_hash_abs_f64(self.ty.official_name()) * self.id as f64
    }
}

/// Official `maxHealthWithVeterancy` for warships.
pub fn warship_max_health_with_veterancy(veterancy: u32) -> i32 {
    if veterancy == 0 {
        return WARSHIP_MAX_HEALTH;
    }
    WARSHIP_MAX_HEALTH
        + (WARSHIP_MAX_HEALTH * veterancy as i32 * WARSHIP_VETERANCY_HEALTH_BONUS) / 100
}

pub fn city_levels(units: &[Unit], player_id: &str) -> f64 {
    units
        .iter()
        .filter(|u| {
            u.active
                && u.ty == UnitType::City
                && !u.under_construction
                && u.owner_id == player_id
        })
        .map(|u| u.level as f64)
        .sum()
}

pub fn structure_count(units: &[Unit], player_id: &str, ty: UnitType) -> u32 {
    units
        .iter()
        .filter(|u| u.active && u.ty == ty && u.owner_id == player_id)
        .map(|u| {
            if u.under_construction {
                1
            } else {
                u.level
            }
        })
        .sum()
}

pub fn cheapest_buildable_cost(player_type: PlayerType, gold: i64, constructed: &[u32]) -> i64 {
    let _ = player_type;
    let _ = gold;
    let _ = constructed;
    unit_cost(UnitType::DefensePost, 0)
}

#[cfg(test)]
mod tests {
    use super::warship_max_health_with_veterancy;
    use crate::config::WARSHIP_RETREAT_HEALTH_PERCENT;

    #[test]
    fn veteran_max_health_and_retreat_threshold_match_official() {
        assert_eq!(warship_max_health_with_veterancy(0), 1000);
        assert_eq!(warship_max_health_with_veterancy(1), 1200);
        assert_eq!(warship_max_health_with_veterancy(2), 1400);
        assert_eq!(warship_max_health_with_veterancy(3), 1600);
        assert_eq!(WARSHIP_RETREAT_HEALTH_PERCENT, 75);
        assert_eq!(1400 * WARSHIP_RETREAT_HEALTH_PERCENT / 100, 1050);
        assert_eq!(crate::config::WARSHIP_PASSIVE_HEALING, 1);
        assert_eq!(crate::config::WARSHIP_PORT_HEALING_BONUS, 5);
    }

    #[test]
    fn temporary_embargo_expires_after_duration_not_at() {
        // Official: `ticks - createdAt > temporaryEmbargoDuration()`.
        let created = 6114u32;
        let dur = super::TEMPORARY_EMBARGO_DURATION;
        assert_eq!(dur, 3000);
        let keep = |ticks: u32| ticks.saturating_sub(created) <= dur;
        assert!(keep(9114));
        assert!(!keep(9115));
    }
}
