//! Gym-style AgentEnv: reset/step over the foundation game.

use crate::actions::{translate, ActionVec};
use crate::ai::{create_nations, tribe_name};
use crate::capabilities::UnsupportedConfig;
use crate::digest::{
    compute_decision_boundary, DecisionBoundary, StepDigestInput,
};
use crate::game::Game;
use crate::obs::{fill_masks, ObsBuffers, ObsExtractor, SlotInfo};
use crate::player::PlayerInfo;
use crate::rng::{simple_hash, PseudoRandom};
use crate::spec::{
    EnvConfig, PlayerType, TerminalCause, ACTION_NOOP, ELIMINATION_SHAPING_WEIGHT, REWARD_DEATH,
    REWARD_LOSS_ALIVE, REWARD_NO_SPAWN, REWARD_SPAWN, REWARD_TIMEOUT, REWARD_WIN,
};
use crate::terrain::{MapAssets, TerrainCache};

pub struct RewardTerms {
    pub terminal: f64,
    pub spawn: f64,
    pub territory: f64,
    pub elimination: f64,
    pub total: f64,
}

pub struct StepInfo {
    pub tick: u32,
    pub win: bool,
    pub stage_success: bool,
    pub dead: bool,
    pub kills: u32,
    pub tiles_frac: f64,
    pub spawned: bool,
    pub hash: Option<f64>,
    pub terminal_cause: TerminalCause,
    pub peak_tiles_frac: f64,
    pub reward_terms: RewardTerms,
    pub intent_count: u32,
    pub action_accepted: bool,
    pub map: String,
    pub map_size: String,
    pub seed: String,
    pub digest: Option<String>,
    pub tile_digest: Option<String>,
    pub obs_digest: Option<String>,
    pub step_digest: Option<String>,
    pub fallback: Option<String>,
}

pub struct StepResult {
    pub obs: ObsBuffers,
    pub reward: f64,
    pub done: bool,
    pub info: StepInfo,
}

pub struct AgentEnv {
    pub cfg: EnvConfig,
    game: Game,
    assets: MapAssets,
    me_id: String,
    extractor: ObsExtractor,
    obs: ObsBuffers,
    slots: Vec<SlotInfo>,
    turn: u32,
    kills: u32,
    spawned_once: bool,
    was_spawned: bool,
    peak_tiles: f64,
    prev_tiles: f64,
    prev_elim: f64,
    initial_opponents: f64,
    last_step: StepDigestInput,
    pub reset_count: u32,
    pub last_sim_ms: f64,
    pub last_mask_obs_ms: f64,
}

impl AgentEnv {
    pub fn create(cfg: EnvConfig, terrain: &mut TerrainCache) -> Result<Self, String> {
        UnsupportedConfig::check(&cfg).map_err(|e| {
            format!(
                "unsupported config (fallback=official): {}",
                e.reasons.join("; ")
            )
        })?;
        let fresh = terrain.fresh_terrain(&cfg.map, cfg.map_size)?;
        let assets = fresh.assets.clone();
        let mut env = Self {
            cfg: cfg.clone(),
            game: Game::from_assets(
                fresh.game_map,
                fresh.mini_map,
                vec![],
                cfg.difficulty,
                cfg.seed.clone(),
                Some(&assets),
            ),
            assets,
            me_id: String::new(),
            extractor: ObsExtractor::new(),
            obs: ObsBuffers::new(),
            slots: Vec::new(),
            turn: 0,
            kills: 0,
            spawned_once: false,
            was_spawned: false,
            peak_tiles: 0.0,
            prev_tiles: 0.0,
            prev_elim: 0.0,
            initial_opponents: 1.0,
            last_step: StepDigestInput::default(),
            reset_count: 0,
            last_sim_ms: 0.0,
            last_mask_obs_ms: 0.0,
        };
        env.reset_inner()?;
        Ok(env)
    }

    pub fn seed(&self) -> &str {
        &self.cfg.seed
    }

    pub fn peek_obs(&self) -> &ObsBuffers {
        &self.obs
    }

    pub fn game(&self) -> &Game {
        &self.game
    }

    pub fn reset(
        &mut self,
        terrain: &mut TerrainCache,
        next: Option<&serde_json::Value>,
    ) -> Result<&ObsBuffers, String> {
        self.apply_next_config(next, Some(terrain))?;
        self.reset_inner()?;
        Ok(&self.obs)
    }

    pub fn reset_local(&mut self, next: Option<&serde_json::Value>) -> Result<&ObsBuffers, String> {
        self.apply_next_config(next, None)?;
        self.reset_inner()?;
        Ok(&self.obs)
    }

    fn apply_next_config(
        &mut self,
        next: Option<&serde_json::Value>,
        terrain: Option<&mut TerrainCache>,
    ) -> Result<(), String> {
        let old_map = self.cfg.map.clone();
        let old_size = self.cfg.map_size;
        if let Some(v) = next {
            if let Some(s) = v.as_str() {
                self.cfg.seed = s.to_string();
            } else if v.is_object() {
                self.cfg.apply_patch(v);
                UnsupportedConfig::check(&self.cfg).map_err(|e| {
                    format!(
                        "unsupported config (fallback=official): {}",
                        e.reasons.join("; ")
                    )
                })?;
            }
        }
        if self.cfg.map != old_map || self.cfg.map_size != old_size {
            let cache = terrain.ok_or_else(|| {
                "map change requires terrain cache; refusing mid-batch map switch".to_string()
            })?;
            let fresh = cache.fresh_terrain(&self.cfg.map, self.cfg.map_size)?;
            self.assets = fresh.assets;
        }
        Ok(())
    }

    fn reset_inner(&mut self) -> Result<(), String> {
        let (game_map, mini_map) = self.assets.fresh_maps()?;
        let mut rng = PseudoRandom::new(simple_hash(&self.cfg.seed));
        let me = PlayerInfo {
            name: "OFAI".into(),
            player_type: PlayerType::Human,
            client_id: Some(crate::spec::AGENT_CLIENT_ID.into()),
            id: rng.next_id(),
        };
        self.me_id = me.id.clone();
        let mut game = Game::from_assets(
            game_map,
            mini_map,
            vec![me],
            self.cfg.difficulty,
            self.cfg.seed.clone(),
            Some(&self.assets),
        );
        let nations = create_nations(
            &self.cfg.nations,
            &self.assets.nations,
            &self.assets.additional_nations,
            self.cfg.map_size,
            &mut rng,
        );
        for n in nations {
            game.queue_nation(n);
        }
        if self.cfg.bots > 0 {
            let mut tribe_rng = PseudoRandom::new(simple_hash(&self.cfg.seed).wrapping_add(2));
            for _ in 0..self.cfg.bots {
                let name = tribe_name(&self.cfg.map, &mut tribe_rng);
                game.queue_bot_spawn(PlayerInfo {
                    name,
                    player_type: PlayerType::Bot,
                    client_id: None,
                    id: tribe_rng.next_id(),
                });
            }
        }
        game.apply_env(&self.cfg);
        game.finish_setup();
        self.game = game;
        self.turn = 0;
        self.kills = 0;
        self.spawned_once = false;
        self.was_spawned = false;
        self.peak_tiles = 0.0;
        self.prev_tiles = 0.0;
        self.prev_elim = 0.0;
        self.extractor = ObsExtractor::new();
        self.extractor.build_static(&self.game, &self.me_id);
        let _ = self.game.drain_packed_tile_updates();
        self.run_ticks(5);
        self.initial_opponents = self
            .game
            .alive_players()
            .iter()
            .filter(|p| p.id() != self.me_id)
            .count()
            .max(1) as f64;
        self.last_step = StepDigestInput::default();
        self.extract_obs();
        Ok(())
    }

    fn tick_once(&mut self) {
        self.game.execute_next_tick();
        self.turn += 1;
        let packed = self.game.drain_packed_tile_updates();
        self.extractor.apply_tile_updates(&self.game, &packed);
    }

    fn run_ticks(&mut self, n: u32) {
        for _ in 0..n {
            self.tick_once();
        }
    }

    fn extract_obs(&mut self) {
        self.slots = self.extractor.extract(
            &self.game,
            &self.me_id,
            &mut self.obs,
            self.cfg.max_ticks,
            self.cfg.difficulty,
        );
        let wild = self.extractor.has_adjacent_wilderness();
        fill_masks(
            &mut self.game,
            &self.me_id,
            &self.slots,
            &mut self.obs,
            wild,
            self.cfg.allowed_actions.as_deref(),
        );
    }

    pub fn strong_state_digest(&self) -> DecisionBoundary {
        compute_decision_boundary(
            &self.game,
            &self.obs,
            self.game.last_hash(),
            &self.last_step,
        )
    }

    pub fn step(&mut self, action: ActionVec) -> StepResult {
        let (reward, done, info) = self.step_apply(action);
        StepResult {
            obs: self.obs.clone(),
            reward,
            done,
            info,
        }
    }

    /// Step without cloning observation tensors. Server encodes from `peek_obs`.
    pub fn step_apply(&mut self, action: ActionVec) -> (f64, bool, StepInfo) {
        let t0 = std::time::Instant::now();
        let (intents, unsupported) = translate(
            &mut self.game,
            &self.me_id,
            action,
            &self.slots,
            self.extractor.has_adjacent_wilderness(),
        );
        let intent_count = intents.len() as u32;
        let action_accepted = action.action_type == ACTION_NOOP || intent_count > 0;
        self.game
            .add_intents(&self.cfg.seed, &self.me_id, &intents);
        self.tick_once();
        let rest = self.cfg.decision_interval.saturating_sub(1);
        self.run_ticks(rest);
        self.last_sim_ms = t0.elapsed().as_secs_f64() * 1000.0;

        if self.game.player(&self.me_id).map(|p| p.has_spawned()).unwrap_or(false) {
            self.spawned_once = true;
        }
        self.kills += self.game.take_kills(&self.me_id);

        let land = self.game.map.num_land_tiles().max(1) as f64;
        let tiles = self.game.player(&self.me_id).map(|p| p.num_tiles() as f64).unwrap_or(0.0);
        let tiles_frac = tiles / land;
        self.peak_tiles = self.peak_tiles.max(tiles_frac);

        let mut terms = RewardTerms {
            terminal: 0.0,
            spawn: 0.0,
            territory: 0.0,
            elimination: 0.0,
            total: 0.0,
        };
        let mut done = false;
        let mut win = false;
        let mut stage_success = false;
        let mut dead = false;
        let mut terminal = TerminalCause::None;
        let alive = self.game.player(&self.me_id).map(|p| p.is_alive()).unwrap_or(false);

        if !self.spawned_once
            && (!self.game.in_spawn_phase() || self.game.ticks() >= self.cfg.max_ticks)
        {
            done = true;
            terminal = TerminalCause::NoSpawn;
            terms.terminal = REWARD_NO_SPAWN;
        } else {
            if let Some(wf) = self.cfg.win_tiles_frac {
                if self.spawned_once && wf > 0.0 && self.peak_tiles >= wf {
                    stage_success = true;
                }
            }
            if let Some(w) = self.game.winner() {
                if w == self.me_id {
                    done = true;
                    win = true;
                    stage_success = true;
                    terminal = TerminalCause::Win;
                    terms.terminal = REWARD_WIN;
                } else if !alive {
                    done = true;
                    dead = true;
                    terminal = TerminalCause::Death;
                    terms.terminal = REWARD_DEATH;
                } else {
                    done = true;
                    terminal = TerminalCause::LossAlive;
                    terms.terminal = REWARD_LOSS_ALIVE;
                }
            } else if !alive && self.spawned_once {
                done = true;
                dead = true;
                terminal = TerminalCause::Death;
                terms.terminal = REWARD_DEATH;
            } else if self.game.ticks() >= self.cfg.max_ticks {
                done = true;
                if !alive {
                    dead = true;
                    terminal = TerminalCause::Death;
                    terms.terminal = REWARD_DEATH;
                } else {
                    terminal = TerminalCause::Timeout;
                    terms.terminal = REWARD_TIMEOUT;
                }
            }
        }

        if !self.was_spawned && self.spawned_once {
            terms.spawn = REWARD_SPAWN;
        }
        self.was_spawned = self.spawned_once;

        let elim = self.elim_progress();
        if self.cfg.shaping > 0.0 {
            terms.territory = self.cfg.shaping * (tiles_frac - self.prev_tiles);
            terms.elimination =
                self.cfg.shaping * ELIMINATION_SHAPING_WEIGHT * (elim - self.prev_elim);
        }
        self.prev_tiles = tiles_frac;
        self.prev_elim = elim;
        terms.total = terms.terminal + terms.spawn + terms.territory + terms.elimination;

        self.last_step = StepDigestInput {
            reward: terms.total,
            intent_count,
            action_accepted,
            terminal_cause: terminal,
            win,
            stage_success,
            dead,
            tiles_frac,
            peak_tiles_frac: self.peak_tiles,
            kills: self.kills,
        };
        let t_obs = std::time::Instant::now();
        self.extract_obs();
        self.last_mask_obs_ms = t_obs.elapsed().as_secs_f64() * 1000.0;

        let mut info = StepInfo {
            tick: self.game.ticks(),
            win,
            stage_success,
            dead,
            kills: self.kills,
            tiles_frac,
            spawned: self.spawned_once,
            hash: self.game.last_hash(),
            terminal_cause: terminal,
            peak_tiles_frac: self.peak_tiles,
            reward_terms: terms,
            intent_count,
            action_accepted,
            map: self.cfg.map.clone(),
            map_size: match self.cfg.map_size {
                crate::spec::MapSize::Compact => "Compact".into(),
                crate::spec::MapSize::Normal => "Normal".into(),
            },
            seed: self.cfg.seed.clone(),
            digest: None,
            tile_digest: None,
            obs_digest: None,
            step_digest: None,
            fallback: unsupported
                .map(|s| format!("official:{s}"))
                .or_else(|| self.game.fallback_reason().map(|s| format!("official:{s}"))),
        };
        if self.cfg.enable_digest {
            let b = self.strong_state_digest();
            info.digest = Some(b.digest);
            info.tile_digest = Some(b.tile_digest);
            info.obs_digest = Some(b.obs_digest);
            info.step_digest = Some(b.step_digest);
        }
        (self.last_step.reward, done, info)
    }

    fn elim_progress(&self) -> f64 {
        let alive = self
            .game
            .alive_players()
            .iter()
            .filter(|p| p.id() != self.me_id)
            .count() as f64;
        ((self.initial_opponents - alive) / self.initial_opponents).clamp(0.0, 1.0)
    }

    pub fn render_frame(&self) -> (u32, u32, Vec<u8>, serde_json::Value) {
        let w = self.game.map.width();
        let h = self.game.map.height();
        let mut cells = vec![0u8; (w * h) as usize];
        let mut roster: Vec<_> = self.game.alive_players();
        roster.sort_by(|a, b| b.num_tiles().cmp(&a.num_tiles()));
        let mut idx_by = std::collections::HashMap::new();
        for (i, p) in roster.iter().enumerate() {
            idx_by.insert(p.small_id, i);
        }
        for r in 0..(w * h) {
            if !self.game.map.is_land(r) {
                cells[r as usize] = 0;
            } else if self.game.map.is_impassable(r) {
                cells[r as usize] = 1;
            } else {
                let owner = self.game.map.owner_id(r);
                cells[r as usize] = if owner == 0 {
                    2
                } else {
                    idx_by.get(&owner).map(|i| 3 + *i as u8).unwrap_or(2)
                };
            }
        }
        let players: Vec<serde_json::Value> = roster
            .iter()
            .enumerate()
            .map(|(i, p)| {
                serde_json::json!({
                    "idx": i,
                    "name": p.info.name,
                    "tiles": p.num_tiles(),
                    "alive": p.is_alive(),
                    "isAgent": p.id() == self.me_id,
                })
            })
            .collect();
        (
            w,
            h,
            cells,
            serde_json::json!({
                "players": players,
                "units": [],
                "tick": self.game.ticks(),
            }),
        )
    }
}

pub fn replay_golden(
    terrain: &mut TerrainCache,
    config: EnvConfig,
    actions: &[[i32; 5]],
) -> Result<(DecisionBoundary, Vec<DecisionBoundary>), String> {
    let mut env = AgentEnv::create(config, terrain)?;
    let init = env.strong_state_digest();
    let mut bounds = Vec::new();
    for a in actions {
        env.step(ActionVec {
            action_type: a[0],
            target: a[1],
            region: a[2],
            quantity: a[3],
            unit: a[4],
        });
        bounds.push(env.strong_state_digest());
    }
    Ok((init, bounds))
}
