//! Factorized action translation. Later systems (nukes/warships/rails) fail closed.

use crate::game::{find_spawn_tile, Intent};
use crate::obs::SlotInfo;
use crate::spec::{
    ACTION_ALLY, ACTION_ATTACK, ACTION_BOAT, ACTION_BREAK_ALLY, ACTION_BUILD, ACTION_EMBARGO,
    ACTION_NOOP, ACTION_RETREAT_ALL, ACTION_SPAWN, NUM_REGIONS, TROOP_FRACTIONS,
};
use crate::units::{UnitType, UNIT_HEAD_ORDER};

#[derive(Clone, Copy, Debug, Default)]
pub struct ActionVec {
    pub action_type: i32,
    pub target: i32,
    pub region: i32,
    pub quantity: i32,
    pub unit: i32,
}

pub fn translate(
    game: &mut crate::game::Game,
    me_id: &str,
    action: ActionVec,
    slots: &[SlotInfo],
    borders_wilderness: bool,
) -> (Vec<Intent>, Option<&'static str>) {
    let intents = match action.action_type {
        ACTION_SPAWN => spawn(game, me_id, action.region),
        ACTION_ATTACK => attack(game, me_id, action, slots, borders_wilderness),
        ACTION_RETREAT_ALL => retreat_all(game, me_id),
        ACTION_BUILD => build(game, me_id, action),
        ACTION_BOAT => boat(game, me_id, action),
        ACTION_ALLY => ally(game, me_id, action, slots),
        ACTION_BREAK_ALLY => break_ally(game, me_id, action, slots),
        ACTION_EMBARGO => embargo(game, me_id, action, slots),
        ACTION_NOOP => Vec::new(),
        _ => Vec::new(),
    };
    (intents, None)
}

fn spawn(game: &crate::game::Game, me_id: &str, region: i32) -> Vec<Intent> {
    let Some(me) = game.player(me_id) else {
        return Vec::new();
    };
    if me.has_spawned() && !game.in_spawn_phase() {
        return Vec::new();
    }
    if let Some(tile) = find_spawn_tile(&game.map, region) {
        return vec![Intent::Spawn { tile }];
    }
    for r in 0..NUM_REGIONS as i32 {
        if r == region {
            continue;
        }
        if let Some(tile) = find_spawn_tile(&game.map, r) {
            return vec![Intent::Spawn { tile }];
        }
    }
    Vec::new()
}

fn attack(
    game: &crate::game::Game,
    me_id: &str,
    action: ActionVec,
    slots: &[SlotInfo],
    borders_wilderness: bool,
) -> Vec<Intent> {
    let Some(me) = game.player(me_id) else {
        return Vec::new();
    };
    let frac = TROOP_FRACTIONS
        .get(action.quantity as usize)
        .copied()
        .unwrap_or(0.15);
    let troops = (me.troops_f64() * frac).floor();
    if troops < 1.0 {
        return Vec::new();
    }
    if action.target == 0 {
        if !borders_wilderness {
            return Vec::new();
        }
        return vec![Intent::Attack {
            target_id: None,
            troops,
            source_tile: None,
            remove_troops: true,
        }];
    }
    let Some(slot) = slots.get(action.target as usize) else {
        return Vec::new();
    };
    let Some(tid) = &slot.player_id else {
        return Vec::new();
    };
    if tid == me_id {
        return Vec::new();
    }
    let Some(target) = game.player(tid) else {
        return Vec::new();
    };
    if !game.can_attack_player(me, target) || !slot.shares_border {
        return Vec::new();
    }
    vec![Intent::Attack {
        target_id: Some(tid.clone()),
        troops,
        source_tile: None,
        remove_troops: true,
    }]
}

fn retreat_all(game: &crate::game::Game, me_id: &str) -> Vec<Intent> {
    let Some(me) = game.player(me_id) else {
        return Vec::new();
    };
    me.outgoing
        .iter()
        .filter(|a| a.active && !a.retreating)
        .map(|a| Intent::Retreat {
            attack_id: a.id.clone(),
        })
        .collect()
}

fn build(game: &mut crate::game::Game, me_id: &str, action: ActionVec) -> Vec<Intent> {
    let ty = UNIT_HEAD_ORDER
        .get(action.unit as usize)
        .copied()
        .unwrap_or(UnitType::City);
    if game.is_unit_disabled(ty) {
        return Vec::new();
    }
    if let Some(tile) = game.find_build_in_region(me_id, ty, action.region) {
        return vec![Intent::Build { unit: ty, tile }];
    }
    if let Some(tile) = game.find_build_anywhere(me_id, ty) {
        return vec![Intent::Build { unit: ty, tile }];
    }
    Vec::new()
}

fn boat(game: &mut crate::game::Game, me_id: &str, action: ActionVec) -> Vec<Intent> {
    let Some(me) = game.player(me_id) else {
        return Vec::new();
    };
    let frac = TROOP_FRACTIONS
        .get(action.quantity as usize)
        .copied()
        .unwrap_or(0.15);
    let troops = (me.troops_f64() * frac).floor();
    if troops < 1.0 {
        return Vec::new();
    }
    if let Some(dst) = game.find_boat_in_region(me_id, action.region) {
        return vec![Intent::Boat { dst, troops }];
    }
    for r in 0..NUM_REGIONS as i32 {
        if r == action.region {
            continue;
        }
        if let Some(dst) = game.find_boat_in_region(me_id, r) {
            return vec![Intent::Boat { dst, troops }];
        }
    }
    Vec::new()
}

fn ally(
    game: &crate::game::Game,
    me_id: &str,
    action: ActionVec,
    slots: &[SlotInfo],
) -> Vec<Intent> {
    let Some(slot) = slots.get(action.target as usize) else {
        return Vec::new();
    };
    let Some(tid) = &slot.player_id else {
        return Vec::new();
    };
    if tid == me_id {
        return Vec::new();
    }
    if !game.can_send_ally(me_id, tid) {
        return Vec::new();
    }
    vec![Intent::AllianceRequest {
        recipient: tid.clone(),
    }]
}

fn break_ally(
    game: &crate::game::Game,
    me_id: &str,
    action: ActionVec,
    slots: &[SlotInfo],
) -> Vec<Intent> {
    let Some(slot) = slots.get(action.target as usize) else {
        return Vec::new();
    };
    let Some(tid) = &slot.player_id else {
        return Vec::new();
    };
    if !game.is_allied(me_id, tid) {
        return Vec::new();
    }
    vec![Intent::BreakAlliance {
        recipient: tid.clone(),
    }]
}

fn embargo(
    game: &crate::game::Game,
    me_id: &str,
    action: ActionVec,
    slots: &[SlotInfo],
) -> Vec<Intent> {
    let Some(me) = game.player(me_id) else {
        return Vec::new();
    };
    let Some(slot) = slots.get(action.target as usize) else {
        return Vec::new();
    };
    let Some(tid) = &slot.player_id else {
        return Vec::new();
    };
    if tid == me_id || game.is_allied(me_id, tid) {
        return Vec::new();
    }
    let start = !me.has_embargo_against(tid);
    vec![Intent::Embargo {
        target: tid.clone(),
        start,
    }]
}
