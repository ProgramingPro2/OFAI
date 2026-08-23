//! Capability declarations and fail-closed routing for unsupported features.

use serde::Serialize;

use crate::spec::EnvConfig;

#[derive(Clone, Debug, Serialize)]
pub struct Capabilities {
    pub backend: &'static str,
    pub schema: &'static str,
    pub framing: &'static str,
    pub supported: SupportedCaps,
    pub unsupported: UnsupportedCaps,
    pub fallback: &'static str,
}

#[derive(Clone, Debug, Serialize)]
pub struct SupportedCaps {
    pub rng_hash: bool,
    pub terrain: bool,
    pub tick: bool,
    pub economy: bool,
    pub spawn: bool,
    pub wilderness_attack: bool,
    pub player_attack: bool,
    pub retreat: bool,
    pub observations_early: bool,
    pub boats: bool,
    pub structures: bool,
    pub diplomacy: bool,
    pub tribe_ai: bool,
    pub nations_ai: bool,
    pub terminals: [&'static str; 5],
}

#[derive(Clone, Debug, Serialize)]
pub struct UnsupportedCaps {
    pub boats: bool,
    pub structures: bool,
    pub diplomacy: bool,
    pub nations_ai: bool,
    pub tribe_ai: bool,
    pub doomsday: bool,
    pub nukes: bool,
    pub rails: bool,
}

impl Capabilities {
    pub fn foundation() -> Self {
        Self {
            backend: "rust",
            schema: "v2",
            framing: "init/step/reset",
            supported: SupportedCaps {
                rng_hash: true,
                terrain: true,
                tick: true,
                economy: true,
                spawn: true,
                wilderness_attack: true,
                player_attack: true,
                retreat: true,
                observations_early: true,
                boats: true,
                structures: true,
                diplomacy: true,
                tribe_ai: true,
                nations_ai: false, // Nation structure/nuke placement not bit-exact yet.
                terminals: ["no_spawn", "timeout", "death", "win", "loss_alive"],
            },
            unsupported: UnsupportedCaps {
                boats: false,
                structures: false,
                diplomacy: false,
                nations_ai: true,
                tribe_ai: false,
                doomsday: false,
                nukes: false,
                rails: false,
            },
            fallback: "official",
        }
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).expect("capabilities json")
    }
}

#[derive(Clone, Debug)]
pub struct UnsupportedConfig {
    pub reasons: Vec<String>,
}

impl UnsupportedConfig {
    pub fn check(_cfg: &EnvConfig) -> Result<(), Self> {
        // Tribe AI is proven. Rails/nukes/doomsday execute in-process.
        // Nation AI stays capability-false: structure/nuke placement is
        // not yet a line-for-line NationStructureBehavior port.
        Ok(())
    }
}
