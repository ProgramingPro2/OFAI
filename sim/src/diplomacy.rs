//! Alliance, embargo, and relation state matching official PlayerImpl.

use crate::spec::PlayerType;
use crate::units::{ALLIANCE_DURATION, ALLIANCE_REQUEST_COOLDOWN, ALLIANCE_REQUEST_DURATION};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Relation {
    Hostile = 0,
    Distrustful = 1,
    Neutral = 2,
    Friendly = 3,
}

impl Relation {
    pub fn from_value(v: f64) -> Self {
        if v < -50.0 {
            Self::Hostile
        } else if v < 0.0 {
            Self::Distrustful
        } else if v < 50.0 {
            Self::Neutral
        } else {
            Self::Friendly
        }
    }
}

#[derive(Clone, Debug)]
pub struct Alliance {
    pub id: u32,
    pub requestor: String,
    pub recipient: String,
    pub created_at: u32,
    pub expires_at: u32,
    pub ext_requestor: bool,
    pub ext_recipient: bool,
}

impl Alliance {
    pub fn other(&self, id: &str) -> &str {
        if self.requestor == id {
            &self.recipient
        } else {
            &self.requestor
        }
    }
    pub fn involves(&self, a: &str, b: &str) -> bool {
        (self.requestor == a && self.recipient == b)
            || (self.requestor == b && self.recipient == a)
    }
    pub fn only_one_agreed_to_extend(&self) -> bool {
        self.ext_requestor != self.ext_recipient
    }
}

#[derive(Clone, Debug)]
pub struct AllianceRequest {
    pub requestor: String,
    pub recipient: String,
    pub created_at: u32,
    pub status: RequestStatus,
}

impl AllianceRequest {
    pub fn involves_ids(&self, a: &str, b: &str) -> bool {
        (self.requestor == a && self.recipient == b)
            || (self.requestor == b && self.recipient == a)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RequestStatus {
    Pending,
    Accepted,
    Rejected,
}

#[derive(Clone, Debug)]
pub struct Embargo {
    pub target: String,
    pub created_at: u32,
    pub temporary: bool,
}

pub fn can_send_alliance_request(
    ticks: u32,
    alive: bool,
    disconnected: bool,
    other_disconnected: bool,
    friendly: bool,
    has_pending_out: bool,
    has_incoming: bool,
    last_out_created: Option<u32>,
) -> bool {
    if disconnected || other_disconnected {
        return false;
    }
    if friendly || !alive {
        return false;
    }
    if has_pending_out {
        return false;
    }
    if has_incoming {
        return true;
    }
    match last_out_created {
        None => true,
        Some(at) => ticks.saturating_sub(at) >= ALLIANCE_REQUEST_COOLDOWN,
    }
}

pub fn alliance_expires_at(created: u32) -> u32 {
    created + ALLIANCE_DURATION
}

pub fn request_expired(ticks: u32, created: u32) -> bool {
    ticks.saturating_sub(created) > ALLIANCE_REQUEST_DURATION
}

pub fn attack_relation_delta(difficulty: crate::spec::Difficulty) -> f64 {
    match difficulty {
        crate::spec::Difficulty::Easy => -60.0,
        crate::spec::Difficulty::Medium => -70.0,
        crate::spec::Difficulty::Hard => -80.0,
        crate::spec::Difficulty::Impossible => -100.0,
    }
}

pub fn decay_relation(r: f64) -> f64 {
    let sign = -1.0 * r.signum();
    let delta = 0.05;
    let mut n = r + sign * delta;
    if n.abs() < delta * 2.0 {
        n = 0.0;
    }
    n
}

pub fn within_rel(v: f64) -> f64 {
    v.max(-100.0).min(100.0)
}

pub fn is_human_like(t: PlayerType) -> bool {
    t == PlayerType::Human
}
