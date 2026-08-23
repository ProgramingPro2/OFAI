//! Schema-v2 OpenFront training simulator foundation.
//!
//! Schema-v2 OpenFront training simulator.
//!
//! Boats use official HPA water paths. Later systems (rails/trade/warship/
//! nukes/SAM/doomsday) are ported and fail-closed only when untestable.

pub mod actions;
pub mod ai;
pub mod capabilities;
pub mod config;
pub mod digest;
pub mod diplomacy;
pub mod doomsday;
pub mod env;
pub mod game;
pub mod hpa;
pub mod js_math;
pub mod later;
pub mod later_tick;
pub mod nation_later;
pub mod map;
pub mod obs;
pub mod parabola;
pub mod rail;
pub mod path;
pub mod player;
pub mod protocol;
pub mod rng;
pub mod spatial;
pub mod spec;
pub mod unit_grid;
pub mod terrain;
pub mod units;

pub use capabilities::Capabilities;
pub use env::{replay_golden, AgentEnv};
pub use protocol::{decode_frame, encode_frame, serve, Server};
pub use spec::EnvConfig;
pub use terrain::TerrainCache;
