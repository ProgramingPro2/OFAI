//! Official-compatible strong decision-boundary digest (stateDigest.ts).

use sha2::{Digest, Sha256};

use crate::game::Game;
use crate::obs::ObsBuffers;
use crate::spec::{quantize_reward, TerminalCause};

#[derive(Clone, Debug)]
pub struct StepDigestInput {
    pub reward: f64,
    pub intent_count: u32,
    pub action_accepted: bool,
    pub terminal_cause: TerminalCause,
    pub win: bool,
    pub stage_success: bool,
    pub dead: bool,
    pub tiles_frac: f64,
    pub peak_tiles_frac: f64,
    pub kills: u32,
}

impl Default for StepDigestInput {
    fn default() -> Self {
        Self {
            reward: 0.0,
            intent_count: 0,
            action_accepted: true,
            terminal_cause: TerminalCause::None,
            win: false,
            stage_success: false,
            dead: false,
            tiles_frac: 0.0,
            peak_tiles_frac: 0.0,
            kills: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct DecisionBoundary {
    pub tick: u32,
    pub core_hash: Option<f64>,
    pub tile_digest: String,
    pub obs_digest: String,
    pub step_digest: String,
    pub digest: String,
    pub reward: f64,
    pub intent_count: u32,
    pub action_accepted: bool,
    pub terminal_cause: TerminalCause,
    pub win: bool,
    pub stage_success: bool,
    pub tiles_frac: f64,
}

pub fn hash_obs_buffers(obs: &ObsBuffers) -> String {
    let mut h = Sha256::new();
    h.update(f32_bytes(&obs.spatial));
    h.update(f32_bytes(&obs.players));
    h.update(f32_bytes(&obs.global));
    h.update(&obs.action_mask);
    h.update(&obs.target_masks);
    h.update(&obs.quantity_mask);
    h.update(&obs.unit_mask);
    h.update(&obs.spawn_regions);
    h.update(&obs.build_regions);
    h.update(&obs.boat_regions);
    hex(h)
}

pub fn hash_tile_state(game: &Game) -> String {
    let mut h = Sha256::new();
    for s in game.map.tile_state_buffer() {
        h.update(s.to_le_bytes());
    }
    write_u32(&mut h, game.ticks() as i32);
    write_u32(&mut h, if game.in_spawn_phase() { 1 } else { 0 });
    write_u32(&mut h, game.map.num_land_tiles() as i32);

    let mut players: Vec<_> = game.all_players().collect();
    players.sort_by_key(|p| p.small_id);
    write_u32(&mut h, players.len() as i32);
    for p in &players {
        write_u32(&mut h, p.small_id as i32);
        write_str(&mut h, p.id());
        write_u32(&mut h, if p.is_alive() { 1 } else { 0 });
        write_u32(&mut h, if p.has_spawned() { 1 } else { 0 });
        write_u32(&mut h, p.num_tiles() as i32);
        write_f64(&mut h, p.troops_f64());
        write_str(&mut h, &p.gold.to_string());
        let mut allies: Vec<u16> = game
            .alliances()
            .iter()
            .filter(|a| a.requestor == p.id() || a.recipient == p.id())
            .filter_map(|a| {
                let other = a.other(p.id());
                game.player(other).map(|o| o.small_id)
            })
            .collect();
        allies.sort_unstable();
        write_u32(&mut h, allies.len() as i32);
        for id in allies {
            write_u32(&mut h, id as i32);
        }
        let mut embargoes: Vec<u16> = players
            .iter()
            .filter(|o| o.id() != p.id() && p.has_embargo_against(o.id()))
            .map(|o| o.small_id)
            .collect();
        embargoes.sort_unstable();
        write_u32(&mut h, embargoes.len() as i32);
        for id in embargoes {
            write_u32(&mut h, id as i32);
        }
        write_u32(&mut h, p.outgoing.len() as i32);
        for atk in &p.outgoing {
            write_str(&mut h, &atk.id);
            write_u32(&mut h, atk.target_small_id as i32);
            write_f64(&mut h, atk.troops);
            write_u32(&mut h, if atk.retreating { 1 } else { 0 });
            write_u32(&mut h, if atk.active { 1 } else { 0 });
        }
    }
    let mut units: Vec<_> = game.units().iter().filter(|u| u.active).collect();
    units.sort_by_key(|u| u.id);
    write_u32(&mut h, units.len() as i32);
    for u in units {
        write_u32(&mut h, u.id as i32);
        write_u32(&mut h, u.ty.as_digest_number());
        write_u32(&mut h, u.owner_small as i32);
        write_u32(&mut h, u.tile as i32);
        write_u32(&mut h, u.hash());
    }
    hex(h)
}

pub fn hash_step_meta(step: &StepDigestInput) -> String {
    let mut h = Sha256::new();
    write_f64(&mut h, quantize_reward(step.reward));
    write_u32(&mut h, step.intent_count as i32);
    write_u32(&mut h, if step.action_accepted { 1 } else { 0 });
    write_str(&mut h, step.terminal_cause.as_str());
    write_u32(&mut h, if step.win { 1 } else { 0 });
    write_u32(&mut h, if step.stage_success { 1 } else { 0 });
    write_u32(&mut h, if step.dead { 1 } else { 0 });
    write_f64(&mut h, quantize_reward(step.tiles_frac));
    write_f64(&mut h, quantize_reward(step.peak_tiles_frac));
    write_u32(&mut h, step.kills as i32);
    hex(h)
}

pub fn compute_decision_boundary(
    game: &Game,
    obs: &ObsBuffers,
    core_hash: Option<f64>,
    step: &StepDigestInput,
) -> DecisionBoundary {
    let tile_digest = hash_tile_state(game);
    let obs_digest = hash_obs_buffers(obs);
    let step_digest = hash_step_meta(step);
    let mut h = Sha256::new();
    write_u32(&mut h, game.ticks() as i32);
    // JS `writeInt32LE(n | 0)` is ToInt32, not a saturating f64→i32 cast.
    write_u32(&mut h, core_hash.map(to_int32).unwrap_or(0));
    write_u32(&mut h, if core_hash.is_some() { 1 } else { 0 });
    write_str(&mut h, &tile_digest);
    write_str(&mut h, &obs_digest);
    write_str(&mut h, &step_digest);
    DecisionBoundary {
        tick: game.ticks(),
        core_hash,
        tile_digest,
        obs_digest,
        step_digest,
        digest: hex(h),
        reward: quantize_reward(step.reward),
        intent_count: step.intent_count,
        action_accepted: step.action_accepted,
        terminal_cause: step.terminal_cause,
        win: step.win,
        stage_success: step.stage_success,
        tiles_frac: quantize_reward(step.tiles_frac),
    }
}

pub fn boundary_json(b: &DecisionBoundary) -> serde_json::Value {
    serde_json::json!({
        "tick": b.tick,
        "coreHash": b.core_hash,
        "digest": b.digest,
        "tileDigest": b.tile_digest,
        "obsDigest": b.obs_digest,
        "stepDigest": b.step_digest,
        "reward": b.reward,
        "intentCount": b.intent_count,
        "actionAccepted": b.action_accepted,
        "terminalCause": b.terminal_cause.as_str(),
        "win": b.win,
        "stageSuccess": b.stage_success,
        "tilesFrac": b.tiles_frac,
    })
}

fn to_int32(n: f64) -> i32 {
    if !n.is_finite() {
        return 0;
    }
    (n.trunc() as i64) as i32
}

fn write_u32(h: &mut Sha256, n: i32) {
    h.update(n.to_le_bytes());
}
fn write_f64(h: &mut Sha256, n: f64) {
    h.update(n.to_le_bytes());
}
fn write_str(h: &mut Sha256, s: &str) {
    h.update(s.as_bytes());
    h.update([0u8]);
}
fn hex(h: Sha256) -> String {
    format!("{:x}", h.finalize())
}
fn f32_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
    out
}
