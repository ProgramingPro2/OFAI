//! Later-system executions: port/factory/rails/trade/warship/nuke/SAM.

use std::collections::HashMap;

use crate::config::*;
use crate::game::Game;
use crate::map::TileRef;
use crate::parabola::ParabolaPath;
use crate::path::PathStepper;
use crate::rail::{is_trade_type, RailNetwork};
use crate::rng::PseudoRandom;
use crate::unit_grid::UnitGrid;
use crate::units::{UnitType, WarshipMode};

pub struct LaterState {
    pub grid: UnitGrid,
    pub rail: RailNetwork,
    pub num_mirvs: u32,
    pub recent_mirv: HashMap<String, u32>,
    pub disabled: Vec<UnitType>,
    pub starting_gold: i64,
    pub max_timer: Option<u32>,
    pub doomsday_enabled: bool,
    pub doomsday_speed: String,
    pub nations: HashMap<String, crate::nation_later::NationLater>,
    pub shared_water_tick: Option<u32>,
    pub shared_water: HashMap<String, Option<std::collections::HashSet<i32>>>,
    pub sam_intercept: HashMap<u32, HashMap<u32, Option<(u32, TileRef)>>>,
    pub water_nukes: bool,
}

impl LaterState {
    pub fn new(grid: UnitGrid) -> Self {
        Self {
            grid,
            rail: RailNetwork::new(),
            num_mirvs: 0,
            recent_mirv: HashMap::new(),
            disabled: Vec::new(),
            starting_gold: 0,
            max_timer: None,
            doomsday_enabled: false,
            doomsday_speed: "normal".into(),
            nations: HashMap::new(),
            shared_water_tick: None,
            shared_water: HashMap::new(),
            sam_intercept: HashMap::new(),
            water_nukes: false,
        }
    }

    pub fn unit_disabled(&self, ty: UnitType) -> bool {
        self.disabled.contains(&ty)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PathKind {
    Next,
    Complete,
    Missing,
}

pub enum LaterExec {
    Port {
        active: bool,
        unit_id: u32,
        rng: Option<PseudoRandom>,
        check_offset: u32,
        rejections: u32,
        inited: bool,
    },
    Factory {
        active: bool,
        unit_id: u32,
        station_created: bool,
    },
    City {
        active: bool,
        unit_id: u32,
        station_created: bool,
    },
    TrainStation {
        active: bool,
        unit_id: u32,
        spawn_trains: bool,
        station_id: Option<u32>,
        rng: Option<PseudoRandom>,
        last_spawn: u32,
        inited: bool,
    },
    Train {
        active: bool,
        owner: String,
        source: u32,
        dest: u32,
        engine: Option<u32>,
        cars: Vec<u32>,
        stations: Vec<u32>,
        tiles: Vec<TileRef>,
        used: Vec<TileRef>,
        current: i32,
        trade_stops: u32,
        inited: bool,
    },
    TradeShip {
        active: bool,
        owner: String,
        src_port: u32,
        dst_port: u32,
        unit_id: Option<u32>,
        tiles_traveled: u32,
        last_move: u32,
        stepper: PathStepper,
        inited: bool,
        was_captured: bool,
    },
    Warship {
        active: bool,
        owner: String,
        patrol: TileRef,
        unit_id: Option<u32>,
        rng: Option<PseudoRandom>,
        last_shell: u32,
        last_move: u32,
        stepper: PathStepper,
        already_sent: Vec<u32>,
        inited: bool,
        heal_rem: f64,
        last_manual: u32,
    },
    Nuke {
        active: bool,
        owner: String,
        ty: UnitType,
        dst: TileRef,
        src: Option<TileRef>,
        unit_id: Option<u32>,
        speed: u32,
        wait: u32,
        up: bool,
        path: Option<ParabolaPath>,
        inited: bool,
    },
    Mirv {
        active: bool,
        owner: String,
        dst: TileRef,
        unit_id: Option<u32>,
        rng: Option<PseudoRandom>,
        path: Option<ParabolaPath>,
        spawned: bool,
        inited: bool,
    },
    Silo {
        active: bool,
        unit_id: u32,
    },
    Sam {
        active: bool,
        owner: String,
        unit_id: u32,
        rng: Option<PseudoRandom>,
        last_fire: u32,
        inited: bool,
    },
    SamMissile {
        active: bool,
        unit_id: Option<u32>,
        owner: String,
        sam: u32,
        target: u32,
        dst: TileRef,
        last_move: u32,
        air_seed: i32,
        air_path: Option<Vec<TileRef>>,
        air_index: usize,
        air_last_to: Option<TileRef>,
        inited: bool,
    },
    Shell {
        active: bool,
        owner: String,
        src: TileRef,
        shooter: u32,
        target: u32,
        unit_id: Option<u32>,
        last_move: u32,
        air_seed: i32,
        air_path: Option<Vec<TileRef>>,
        air_index: usize,
        air_last_to: Option<TileRef>,
        destroy_at: i32,
        inited: bool,
    },
    RailRecompute {
        active: bool,
    },
    Doomsday {
        active: bool,
    },
    Upgrade {
        active: bool,
        owner: String,
        unit_id: u32,
        inited: bool,
    },
}

impl LaterExec {
    pub fn is_active(&self) -> bool {
        match self {
            Self::Port { active, .. }
            | Self::Factory { active, .. }
            | Self::City { active, .. }
            | Self::TrainStation { active, .. }
            | Self::Train { active, .. }
            | Self::TradeShip { active, .. }
            | Self::Warship { active, .. }
            | Self::Nuke { active, .. }
            | Self::Mirv { active, .. }
            | Self::Silo { active, .. }
            | Self::Sam { active, .. }
            | Self::SamMissile { active, .. }
            | Self::Shell { active, .. }
            | Self::RailRecompute { active }
            | Self::Doomsday { active }
            | Self::Upgrade { active, .. } => *active,
        }
    }

    pub fn during_spawn(&self) -> bool {
        false
    }
}

impl Game {
    pub(crate) fn enqueue_later(&mut self, e: LaterExec) {
        self.push_later(e);
    }

    pub(crate) fn complete_later_structure(&mut self, uid: u32, ty: UnitType) {
        match ty {
            UnitType::Port => {
                self.enqueue_later(LaterExec::Port {
                    active: true,
                    unit_id: uid,
                    rng: None,
                    check_offset: 0,
                    rejections: 0,
                    inited: false,
                });
            }
            UnitType::Factory => {
                self.enqueue_later(LaterExec::Factory {
                    active: true,
                    unit_id: uid,
                    station_created: false,
                });
            }
            UnitType::City => {
                self.enqueue_later(LaterExec::City {
                    active: true,
                    unit_id: uid,
                    station_created: false,
                });
            }
            UnitType::MissileSilo => {
                self.enqueue_later(LaterExec::Silo {
                    active: true,
                    unit_id: uid,
                });
            }
            UnitType::SamLauncher => {
                let owner = self
                    .units()
                    .iter()
                    .find(|u| u.id == uid)
                    .map(|u| u.owner_id.clone())
                    .unwrap_or_default();
                self.enqueue_later(LaterExec::Sam {
                    active: true,
                    owner,
                    unit_id: uid,
                    rng: None,
                    last_fire: 0,
                    inited: false,
                });
            }
            _ => {}
        }
    }

    pub(crate) fn launch_warship(&mut self, owner: String, patrol: TileRef) {
        self.enqueue_later(LaterExec::Warship {
            active: true,
            owner,
            patrol,
            unit_id: None,
            rng: None,
            last_shell: 0,
            last_move: 0,
            stepper: PathStepper::new(),
            already_sent: Vec::new(),
            inited: false,
            heal_rem: 0.0,
            last_manual: 0,
        });
    }

    pub(crate) fn launch_nuke(
        &mut self,
        owner: String,
        ty: UnitType,
        dst: TileRef,
        up: bool,
    ) {
        self.launch_nuke_wait(owner, ty, dst, up, 0);
    }

    pub(crate) fn launch_nuke_wait(
        &mut self,
        owner: String,
        ty: UnitType,
        dst: TileRef,
        up: bool,
        wait: u32,
    ) {
        self.enqueue_later(LaterExec::Nuke {
            active: true,
            owner,
            ty,
            dst,
            src: None,
            unit_id: None,
            speed: nuke_speed(ty),
            wait,
            up,
            path: None,
            inited: false,
        });
    }

    pub(crate) fn launch_mirv(&mut self, owner: String, dst: TileRef) {
        self.enqueue_later(LaterExec::Mirv {
            active: true,
            owner,
            dst,
            unit_id: None,
            rng: None,
            path: None,
            spawned: false,
            inited: false,
        });
    }
}

pub fn nearby_of_types(
    game: &Game,
    tile: TileRef,
    range: u32,
    types: &[UnitType],
) -> Vec<(u32, i64)> {
    game.later_grid_nearby(tile, range, types)
}

pub fn has_factory_nearby(game: &Game, tile: TileRef) -> bool {
    game.later_has_nearby(tile, TRAIN_STATION_MAX_RANGE, UnitType::Factory)
}
