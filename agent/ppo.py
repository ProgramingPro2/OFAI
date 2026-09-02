"""PPO trainer with GAE, clipping, active-head KL/entropy, and truthful
episode metrics that persist across collect() calls.
"""

from __future__ import annotations

import copy
import os
import tempfile
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from . import spaces
from .model import Policy, obs_to_tensors

CHECKPOINT_SCHEMA_VERSION = 2

# Bounded rolling histories. Stage transitions reset these; lifetime
# counters (episodes, wins, env_steps, …) are never cleared here.
WINDOW_MAXLEN: dict[str, int] = {
    "ret_20": 20,
    "ret_100": 100,
    "ret_500": 500,
    "wins_20": 20,
    "wins_100": 100,
    "wins_500": 500,
    "stage_20": 20,
    "stage_100": 100,
    "stage_500": 500,
    "kills_20": 20,
    "kills_100": 100,
    "kills_500": 500,
    "len_20": 20,
    "len_100": 100,
    "death_20": 20,
    "timeout_20": 20,
    "accept_20": 20,
}


def atomic_torch_save(payload: dict, path: str | Path) -> None:
    """Write a checkpoint via same-dir tempfile + os.replace."""
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=dest.name + ".", suffix=".tmp", dir=dest.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(payload, tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


@dataclass
class PPOConfig:
    rollout_len: int = 128
    num_minibatches: int = 8
    update_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    vf_clip: float = 1.0
    # Active-joint KL (action + conditional heads only). Old 5-head summed
    # target of 0.3 maps to ~0.03 here.
    target_kl: float = 0.03
    hard_kl_mult: float = 4.0
    # Target is mask-normalized action-type entropy in [0, 1]:
    # H(action) / log(n_legal), averaged over samples with >1 legal
    # actions. Raw nats are unattainable on land stages (2-4 legal =>
    # max 0.69-1.39); a 1.5 nat target races ent_coef to the ceiling.
    # Conditional region entropy is logged only — no bonus.
    ent_target: float = 0.6
    # "uniform" = current legal-set entropy (fresh-run default).
    # "effective" = renormalized onto explore-worthy types so always-legal
    # boat/build/diplomacy/NOOP are not rewarded for being diverse.
    ent_weighting: str = "uniform"
    # Mask-normalized quantity-head bonus on ATTACK/BOAT only (0=off).
    qty_ent_coef: float = 0.0
    ent_coef_min: float = 0.01
    ent_coef_max: float = 0.5
    ent_kp: float = 0.05
    lr: float = 3e-4
    max_grad_norm: float = 0.5
    device: str = "cuda"
    # Frozen stage-entry reference. coef=0 disables the loss (fresh runs).
    # Loss is coef * relu(KL(π || π_ref) - target). hard triggers rollback.
    ref_kl_coef: float = 0.0
    ref_kl_target: float = 0.02
    ref_kl_hard: float = 0.50


class RolloutBuffer:
    """Rollout storage. Spatial planes are float16 (values in [0,1])."""

    def __init__(self, rollout_len: int, k: int):
        self.t = rollout_len
        self.k = k
        s = spaces
        self.obs = {
            "spatial": np.zeros(
                (rollout_len, k, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE),
                np.float16,
            ),
            "players": np.zeros(
                (rollout_len, k, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), np.float32
            ),
            "global": np.zeros((rollout_len, k, s.GLOBAL_FEATURES), np.float32),
            "action_mask": np.zeros((rollout_len, k, s.NUM_ACTION_TYPES), np.uint8),
            "target_masks": np.zeros(
                (rollout_len, k, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS), np.uint8
            ),
            "quantity_mask": np.zeros((rollout_len, k, s.NUM_QUANTITIES), np.uint8),
            "unit_mask": np.zeros((rollout_len, k, s.NUM_UNIT_TYPES), np.uint8),
            "spawn_regions": np.zeros((rollout_len, k, s.NUM_REGIONS), np.uint8),
            "build_regions": np.zeros((rollout_len, k, s.NUM_REGIONS), np.uint8),
            "boat_regions": np.zeros((rollout_len, k, s.NUM_REGIONS), np.uint8),
        }
        self.actions = np.zeros((rollout_len, k, 5), np.int64)
        self.logprobs = np.zeros((rollout_len, k), np.float32)
        self.rewards = np.zeros((rollout_len, k), np.float32)
        self.dones = np.zeros((rollout_len, k), np.float32)
        self.values = np.zeros((rollout_len, k), np.float32)


@dataclass
class EpisodeRecord:
    """One completed episode (spans any number of collect() calls)."""

    env_index: int
    ret: float
    decisions: int
    ticks: int
    terminal_cause: str
    win: bool
    stage_success: bool
    kills: int
    final_tiles_frac: float
    peak_tiles_frac: float
    spawned: bool
    reward_terms: dict
    action_hist: dict
    accepted: int
    fizzled: int
    intent_count: int
    map: str = ""
    seed: str = ""


@dataclass
class _EnvTracker:
    ret: float = 0.0
    decisions: int = 0
    reward_terms: dict = field(
        default_factory=lambda: {
            "terminal": 0.0,
            "spawn": 0.0,
            "territory": 0.0,
            "elimination": 0.0,
            "total": 0.0,
        }
    )
    action_hist: Counter = field(default_factory=Counter)
    accepted: int = 0
    fizzled: int = 0
    intent_count: int = 0


def _empty_tracker() -> _EnvTracker:
    return _EnvTracker()


@dataclass
class TrainStats:
    iterations: int = 0
    env_steps: int = 0
    episodes: int = 0
    wins: int = 0
    stage_successes: int = 0
    deaths: int = 0
    timeouts: int = 0
    kills: int = 0
    spawns: int = 0
    actions_accepted: int = 0
    actions_total: int = 0
    # Bounded rolling histories (exact windows, not unbounded lists)
    ret_20: deque = field(default_factory=lambda: deque(maxlen=20))
    ret_100: deque = field(default_factory=lambda: deque(maxlen=100))
    ret_500: deque = field(default_factory=lambda: deque(maxlen=500))
    wins_20: deque = field(default_factory=lambda: deque(maxlen=20))
    wins_100: deque = field(default_factory=lambda: deque(maxlen=100))
    wins_500: deque = field(default_factory=lambda: deque(maxlen=500))
    stage_20: deque = field(default_factory=lambda: deque(maxlen=20))
    stage_100: deque = field(default_factory=lambda: deque(maxlen=100))
    stage_500: deque = field(default_factory=lambda: deque(maxlen=500))
    kills_20: deque = field(default_factory=lambda: deque(maxlen=20))
    kills_100: deque = field(default_factory=lambda: deque(maxlen=100))
    kills_500: deque = field(default_factory=lambda: deque(maxlen=500))
    len_20: deque = field(default_factory=lambda: deque(maxlen=20))
    len_100: deque = field(default_factory=lambda: deque(maxlen=100))
    death_20: deque = field(default_factory=lambda: deque(maxlen=20))
    timeout_20: deque = field(default_factory=lambda: deque(maxlen=20))
    accept_20: deque = field(default_factory=lambda: deque(maxlen=20))
    sps: float = 0.0

    def _mean(self, d: deque) -> float:
        if not d:
            return 0.0
        return float(np.mean(d))

    @property
    def win_rate(self) -> float:
        return self._mean(self.wins_100)

    @property
    def win_rate_20(self) -> float:
        return self._mean(self.wins_20)

    @property
    def stage_success_rate(self) -> float:
        return self._mean(self.stage_100)

    @property
    def stage_success_rate_20(self) -> float:
        return self._mean(self.stage_20)

    @property
    def mean_return(self) -> float:
        return self._mean(self.ret_100)

    @property
    def mean_return_20(self) -> float:
        return self._mean(self.ret_20)

    @property
    def kills_per_ep_20(self) -> float:
        return self._mean(self.kills_20)

    @property
    def mean_ep_len_20(self) -> float:
        return self._mean(self.len_20)

    @property
    def death_rate_20(self) -> float:
        return self._mean(self.death_20)

    @property
    def timeout_rate_20(self) -> float:
        return self._mean(self.timeout_20)

    @property
    def accept_rate_20(self) -> float:
        return self._mean(self.accept_20)

    @property
    def spawn_rate(self) -> float:
        # Approximate from recent returns window size vs spawn counter rate
        if self.episodes == 0:
            return 0.0
        return self.spawns / self.episodes

    def record_episode(self, ep: EpisodeRecord) -> None:
        self.episodes += 1
        cause = ep.terminal_cause
        won = cause == "win" or (bool(ep.win) and cause != "curriculum_success")
        stage_ok = bool(ep.stage_success) or cause in ("win", "curriculum_success")
        if won:
            stage_ok = True
        dead = cause == "death"
        timeout = cause == "timeout"
        if won:
            self.wins += 1
        if stage_ok:
            self.stage_successes += 1
        if dead:
            self.deaths += 1
        if timeout:
            self.timeouts += 1
        self.kills += int(ep.kills)
        if ep.spawned:
            self.spawns += 1
        self.actions_accepted += int(ep.accepted)
        self.actions_total += int(ep.accepted + ep.fizzled)

        self.ret_20.append(ep.ret)
        self.ret_100.append(ep.ret)
        self.ret_500.append(ep.ret)
        self.wins_20.append(1.0 if won else 0.0)
        self.wins_100.append(1.0 if won else 0.0)
        self.wins_500.append(1.0 if won else 0.0)
        self.stage_20.append(1.0 if stage_ok else 0.0)
        self.stage_100.append(1.0 if stage_ok else 0.0)
        self.stage_500.append(1.0 if stage_ok else 0.0)
        self.kills_20.append(float(ep.kills))
        self.kills_100.append(float(ep.kills))
        self.kills_500.append(float(ep.kills))
        self.len_20.append(float(ep.decisions))
        self.len_100.append(float(ep.decisions))
        self.death_20.append(1.0 if dead else 0.0)
        self.timeout_20.append(1.0 if timeout else 0.0)
        total_a = ep.accepted + ep.fizzled
        self.accept_20.append(ep.accepted / total_a if total_a else 1.0)

    def rolling_snapshot(self, stage: str | None = None) -> dict:
        snap = {
            "return_20": round(self.mean_return_20, 4),
            "return_100": round(self.mean_return, 4),
            "return_500": round(self._mean(self.ret_500), 4),
            "kills_ep_20": round(self.kills_per_ep_20, 3),
            "kills_ep_100": round(self._mean(self.kills_100), 3),
            "win_20": round(self.win_rate_20, 4),
            "win_100": round(self.win_rate, 4),
            "win_500": round(self._mean(self.wins_500), 4),
            "stage_success_20": round(self.stage_success_rate_20, 4),
            "stage_success_100": round(self.stage_success_rate, 4),
            "stage_success_500": round(self._mean(self.stage_500), 4),
            "death_20": round(self.death_rate_20, 4),
            "timeout_20": round(self.timeout_rate_20, 4),
            "ep_len_20": round(self.mean_ep_len_20, 1),
            "accept_20": round(self.accept_rate_20, 4),
        }
        if stage is not None:
            snap["window_stage"] = stage
        return snap

    def reset_rolling_windows(self) -> None:
        """Clear 20/100/500 windows. Lifetime counters stay intact."""
        for name, ml in WINDOW_MAXLEN.items():
            setattr(self, name, deque(maxlen=ml))

    def state_dict(self) -> dict:
        d = {
            "iterations": self.iterations,
            "env_steps": self.env_steps,
            "episodes": self.episodes,
            "wins": self.wins,
            "stage_successes": self.stage_successes,
            "deaths": self.deaths,
            "timeouts": self.timeouts,
            "kills": self.kills,
            "spawns": self.spawns,
            "actions_accepted": self.actions_accepted,
            "actions_total": self.actions_total,
            "sps": self.sps,
        }
        for name in WINDOW_MAXLEN:
            d[name] = list(getattr(self, name))
        return d

    def load_state_dict(self, d: dict) -> None:
        for key in (
            "iterations",
            "env_steps",
            "episodes",
            "wins",
            "stage_successes",
            "deaths",
            "timeouts",
            "kills",
            "spawns",
            "actions_accepted",
            "actions_total",
            "sps",
        ):
            if key in d:
                setattr(self, key, d[key])
        for name, ml in WINDOW_MAXLEN.items():
            if name in d:
                setattr(self, name, deque(d[name], maxlen=ml))


def _normalize_advantages(adv: torch.Tensor) -> torch.Tensor:
    adv = torch.nan_to_num(adv, nan=0.0, posinf=0.0, neginf=0.0)
    std = adv.std()
    if not torch.isfinite(std) or float(std) < 1e-6:
        return adv - adv.mean()
    return (adv - adv.mean()) / (std + 1e-8)


def _terminal_cause(info: dict) -> str:
    cause = info.get("terminalCause") or info.get("terminal_cause")
    if cause in spaces.TERMINAL_CAUSES:
        return str(cause)
    if info.get("stageSuccess") or info.get("stage_success"):
        # Legacy servers that labeled a milestone as win=true.
        if not info.get("win"):
            return "curriculum_success"
    if info.get("win"):
        return "win"
    if info.get("dead"):
        return "death"
    if info.get("spawned") is False or info.get("noSpawn"):
        return "no_spawn"
    return "timeout"


def classify_outcome(
    info: dict, stage_tiles_frac: float = 0.0
) -> tuple[str, bool, bool]:
    """Return (cause, core_win, stage_success).

    `core_win` is a real FFA win only. A curriculum tile milestone is
    `curriculum_success` / stage_success and never counts as a win.
    When eval omits milestone termination, stage_success is also true if
    peak tiles reached `stage_tiles_frac`.
    """
    cause = _terminal_cause(info)
    win = cause == "win"
    if cause == "curriculum_success":
        win = False
    elif bool(info.get("win")) and cause not in (
        "curriculum_success",
        "death",
        "loss_alive",
        "timeout",
        "no_spawn",
    ):
        win = True
        cause = "win"
    peak = float(info.get("peakTilesFrac", info.get("peak_tiles_frac", 0.0)))
    stage = (
        bool(info.get("stageSuccess") or info.get("stage_success"))
        or cause in ("win", "curriculum_success")
        or (stage_tiles_frac > 0.0 and peak >= stage_tiles_frac)
        or win
    )
    return cause, win, stage


def _reward_terms(info: dict, step_reward: float) -> dict:
    rt = info.get("rewardTerms") or info.get("reward_terms") or {}
    return {
        "terminal": float(rt.get("terminal", 0.0)),
        "spawn": float(rt.get("spawn", 0.0)),
        "territory": float(rt.get("territory", 0.0)),
        "elimination": float(rt.get("elimination", 0.0)),
        "total": float(rt.get("total", step_reward)),
    }


def make_adam(policy: nn.Module, lr: float) -> torch.optim.Adam:
    """Fresh Adam with the trainer's transfer-safe defaults."""
    return torch.optim.Adam(policy.parameters(), lr=lr, betas=(0.9, 0.95), eps=1e-5)


def optimizer_is_fresh(opt: torch.optim.Optimizer) -> bool:
    """True when Adam has no inherited moments or step counters."""
    state = opt.state_dict().get("state") or {}
    return len(state) == 0


def apply_optimizer_lr(opt: torch.optim.Optimizer, lr: float) -> None:
    for group in opt.param_groups:
        group["lr"] = float(lr)


class PPOTrainer:
    def __init__(self, policy: Policy, cfg: PPOConfig):
        self.policy = policy.to(cfg.device)
        self.cfg = cfg
        self.opt = make_adam(self.policy, cfg.lr)
        self.stats = TrainStats()
        self.ent_coef = cfg.ent_coef
        self._trackers: list[_EnvTracker] = []
        self._k = 0
        self._legal_counts = np.zeros(spaces.NUM_ACTION_TYPES, np.int64)
        self._legal_n = 0
        self.reference: Policy | None = None
        self.reference_source: str = ""
        self.elite = None

    def _ensure_trackers(self, k: int) -> None:
        if self._k != k or len(self._trackers) != k:
            self._trackers = [_empty_tracker() for _ in range(k)]
            self._k = k

    def reset_episode_trackers(self) -> None:
        """Call after env reinit or RolloutInterrupted."""
        self._trackers = [_empty_tracker() for _ in range(self._k)] if self._k else []
        elite = getattr(self, "elite", None)
        if elite is not None:
            elite.discard_open()

    def set_reference(
        self,
        state_dict: dict | None = None,
        *,
        source: str = "",
    ) -> None:
        """Freeze a copy of policy (or `state_dict`) as the stage-entry ref.

        The reference is not an optimizer parameter. Ordinary fresh training
        leaves this unset / coef=0.
        """
        ref = Policy()
        ref.load_state_dict(state_dict or self.policy.state_dict())
        ref.to(self.cfg.device)
        ref.eval()
        for p in ref.parameters():
            p.requires_grad_(False)
        self.reference = ref
        self.reference_source = source

    def clear_reference(self) -> None:
        self.reference = None
        self.reference_source = ""

    def reference_enabled(self) -> bool:
        return self.reference is not None and float(self.cfg.ref_kl_coef) > 0.0

    def reference_payload(self) -> dict | None:
        """Explicit resume metadata. Never mixed into model/optimizer."""
        if self.reference is None:
            return None
        return {
            "enabled": True,
            "source": self.reference_source,
            "coef": float(self.cfg.ref_kl_coef),
            "target": float(self.cfg.ref_kl_target),
            "hard": float(self.cfg.ref_kl_hard),
            "model": {k: v.detach().cpu() for k, v in self.reference.state_dict().items()},
        }

    def load_reference_payload(self, payload: dict | None) -> None:
        if not payload or not payload.get("model"):
            return
        self.set_reference(payload["model"], source=str(payload.get("source") or ""))
        if "coef" in payload:
            self.cfg.ref_kl_coef = float(payload["coef"])
        if "target" in payload:
            self.cfg.ref_kl_target = float(payload["target"])
        if "hard" in payload:
            self.cfg.ref_kl_hard = float(payload["hard"])

    def take_legal_frac(self) -> dict[str, float]:
        """Per-iteration fraction of decisions where each action type was legal."""
        n = max(1, int(self._legal_n))
        out = {
            spaces.ACTION_NAMES[i]: round(float(self._legal_counts[i]) / n, 4)
            for i in range(spaces.NUM_ACTION_TYPES)
        }
        self._legal_counts[:] = 0
        self._legal_n = 0
        return out

    def collect(self, env, obs: dict, buf: RolloutBuffer) -> tuple[dict, list[EpisodeRecord]]:
        """Run rollout_len decision steps. Returns (latest_obs, completed episodes).

        Raises RolloutInterrupted from the env; caller must discard the partial
        buffer and reset episode trackers.
        """
        cfg = self.cfg
        device = torch.device(cfg.device)
        self._ensure_trackers(env.k)
        completed: list[EpisodeRecord] = []
        elite = getattr(self, "elite", None)
        record_elite = (
            elite is not None
            and bool(getattr(elite, "enabled", False))
            and elite.accepts_backend(getattr(env, "backend", None))
        )
        if record_elite:
            elite.configure(env.k)

        for step in range(cfg.rollout_len):
            # Store f16 spatial (same quantization evaluate() sees). Pass the
            # f16 view to obs_to_tensors so GPU does the f16->f32 cast once;
            # do not round-trip through a CPU float32 copy.
            spatial_f16 = np.asarray(obs["spatial"], dtype=np.float16)
            buf.obs["spatial"][step] = spatial_f16
            for key in buf.obs:
                if key != "spatial":
                    buf.obs[key][step] = obs[key]
            with torch.no_grad():
                if obs["spatial"].dtype == np.float16:
                    obs_t = obs_to_tensors(obs, device)
                else:
                    obs_q = dict(obs)
                    obs_q["spatial"] = spatial_f16
                    obs_t = obs_to_tensors(obs_q, device)
                actions_t, logp_t, value_t = self.policy.act(obs_t, deterministic=False)
            actions = actions_t.cpu().numpy().astype(np.int32)
            if record_elite:
                for i in range(env.k):
                    elite.on_step(i, obs, actions[i])
            batch = env.step(actions)
            buf.actions[step] = actions
            buf.logprobs[step] = logp_t.cpu().numpy()
            buf.values[step] = value_t.cpu().numpy()
            buf.rewards[step] = batch.rewards
            buf.dones[step] = batch.dones.astype(np.float32)
            mask = np.asarray(obs["action_mask"], dtype=np.int64).reshape(
                env.k, spaces.NUM_ACTION_TYPES
            )
            self._legal_n += env.k
            self._legal_counts += mask.sum(axis=0)

            for i in range(env.k):
                tr = self._trackers[i]
                info = batch.infos[i]
                r = float(batch.rewards[i])
                tr.ret += r
                tr.decisions += 1
                terms = _reward_terms(info, r)
                for k_, v_ in terms.items():
                    tr.reward_terms[k_] = tr.reward_terms.get(k_, 0.0) + v_
                at = int(actions[i, 0])
                tr.action_hist[spaces.ACTION_NAMES[at]] += 1
                accepted = info.get("actionAccepted")
                if accepted is None:
                    accepted = info.get("action_accepted", True)
                if accepted:
                    tr.accepted += 1
                else:
                    tr.fizzled += 1
                tr.intent_count += int(info.get("intentCount", info.get("intent_count", 1)))

                if batch.dones[i]:
                    cause, won, stage_ok = classify_outcome(info)
                    ep = EpisodeRecord(
                        env_index=i,
                        ret=float(tr.ret),
                        decisions=int(tr.decisions),
                        ticks=int(info.get("tick", 0)),
                        terminal_cause=cause,
                        win=won,
                        stage_success=stage_ok,
                        kills=int(info.get("kills", 0)),
                        final_tiles_frac=float(info.get("tilesFrac", info.get("tiles_frac", 0.0))),
                        peak_tiles_frac=float(
                            info.get("peakTilesFrac", info.get("peak_tiles_frac", 0.0))
                        ),
                        spawned=bool(info.get("spawned", True)),
                        reward_terms=dict(tr.reward_terms),
                        action_hist=dict(tr.action_hist),
                        accepted=tr.accepted,
                        fizzled=tr.fizzled,
                        intent_count=tr.intent_count,
                        map=str(info.get("map", "")),
                        seed=str(info.get("seed", "")),
                    )
                    self.stats.record_episode(ep)
                    completed.append(ep)
                    self._trackers[i] = _empty_tracker()
                    if record_elite:
                        elite.on_done(i, info)
            obs = batch.obs

        self.stats.env_steps += cfg.rollout_len * env.k
        return obs, completed

    def update(self, buf: RolloutBuffer, last_obs: dict) -> dict:
        cfg = self.cfg
        device = torch.device(cfg.device)
        t, k = buf.actions.shape[:2]
        n = t * k

        with torch.no_grad():
            last_obs_t = obs_to_tensors(last_obs, device)
            last_value = self.policy.logits(last_obs_t)["value"].cpu().numpy()

        advantages = np.zeros((t, k), np.float32)
        lastgaelam = np.zeros(k, np.float32)
        for step in reversed(range(t)):
            nextnonterminal = 1.0 - buf.dones[step]
            nextvalues = last_value if step == t - 1 else buf.values[step + 1]
            delta = (
                buf.rewards[step]
                + cfg.gamma * nextvalues * nextnonterminal
                - buf.values[step]
            )
            lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            advantages[step] = lastgaelam
        returns = advantages + buf.values

        flat_obs = {
            key: buf.obs[key].reshape((n,) + buf.obs[key].shape[2:]) for key in buf.obs
        }
        flat_spatial_f16 = flat_obs.pop("spatial")
        b_actions = buf.actions.reshape(n, 5)
        b_logprobs = buf.logprobs.reshape(n)
        b_advantages = advantages.reshape(n)
        b_returns = returns.reshape(n)
        b_values = buf.values.reshape(n)

        idxs = np.arange(n)
        clipfracs: list[float] = []
        approx_kl = 0.0
        running_kl_sum = 0.0
        running_kl_count = 0
        pg_loss = ent = v_loss = None
        action_ent = cond_ent = action_ent_max = measured_norm = None
        qty_measured = None
        ref_kl_mean = None
        ref_kl_parts: dict[str, float] = {}
        n_updates = 0
        stop = False
        early_stopped = False
        hard_reverted = False
        ref_kl_reverted = False
        total_minibatches = cfg.update_epochs * max(1, cfg.num_minibatches)

        pre_model = {k2: v2.detach().clone() for k2, v2 in self.policy.state_dict().items()}
        pre_opt = copy.deepcopy(self.opt.state_dict())
        pre_ent_coef = self.ent_coef

        for _ in range(cfg.update_epochs):
            if stop:
                break
            np.random.shuffle(idxs)
            # Exact requested partition (covers all samples even when n % mb != 0)
            mb_splits = np.array_split(idxs, cfg.num_minibatches)
            epoch_kl_sum = 0.0
            epoch_kl_count = 0
            for mb in mb_splits:
                if stop:
                    break
                if len(mb) == 0:
                    continue
                mb_obs = {
                    key: torch.as_tensor(flat_obs[key][mb], dtype=torch.float32, device=device)
                    for key in flat_obs
                }
                mb_obs["spatial"] = torch.as_tensor(
                    flat_spatial_f16[mb], dtype=torch.float32, device=device
                )
                mb_actions = torch.as_tensor(b_actions[mb], device=device)
                mb_adv = _normalize_advantages(
                    torch.as_tensor(b_advantages[mb], device=device)
                )
                mb_ret = torch.as_tensor(b_returns[mb], device=device)
                mb_old_logp = torch.as_tensor(b_logprobs[mb], device=device)
                mb_old_value = torch.as_tensor(b_values[mb], device=device)

                logp, entropy, value, extras = self.policy.evaluate(mb_obs, mb_actions)
                ratio = (logp - mb_old_logp).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - mb_old_logp)).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )
                    epoch_kl_sum += approx_kl * len(mb)
                    epoch_kl_count += len(mb)
                    running_kl_sum += approx_kl * len(mb)
                    running_kl_count += len(mb)

                if approx_kl > cfg.target_kl * cfg.hard_kl_mult:
                    self.policy.load_state_dict(pre_model)
                    self.opt.load_state_dict(pre_opt)
                    self.ent_coef = pre_ent_coef
                    hard_reverted = True
                    n_updates = 0
                    stop = True
                    break

                agg_kl = (
                    running_kl_sum / max(1, running_kl_count)
                    if running_kl_count
                    else approx_kl
                )
                if agg_kl > cfg.target_kl:
                    early_stopped = True
                    stop = True
                    break

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg1, pg2).mean()
                v_clipped = mb_old_value + torch.clamp(
                    value - mb_old_value, -cfg.vf_clip, cfg.vf_clip
                )
                v_loss = 0.5 * torch.max(
                    (value - mb_ret) ** 2, (v_clipped - mb_ret) ** 2
                ).mean()
                ent = entropy.mean()
                action_ent = extras["action_entropy"].mean()
                cond_ent = extras["cond_entropy"].mean()
                use_eff = cfg.ent_weighting == "effective"
                if use_eff and "action_entropy_eff_norm" in extras:
                    multi = extras["action_entropy_eff_multi"]
                    norm_key, max_key = "action_entropy_eff_norm", "action_entropy_eff_max"
                else:
                    multi = extras["action_entropy_multi"]
                    norm_key, max_key = "action_entropy_norm", "action_entropy_max"
                if bool(multi.any()):
                    measured_norm = extras[norm_key][multi].mean()
                    action_ent_max = extras[max_key][multi].mean()
                else:
                    # No multi-choice samples: fully satisfied, do not demand
                    # more exploration.
                    measured_norm = extras[norm_key].new_tensor(1.0)
                    action_ent_max = extras[max_key].new_tensor(0.0)
                # Bonus the multi-sample normalized action-type head only.
                # Raw nats and 1024-way region entropy must not drive this.
                loss = pg_loss - self.ent_coef * measured_norm + cfg.vf_coef * v_loss
                if float(cfg.qty_ent_coef) > 0.0:
                    qty_multi = extras["quantity_entropy_multi"]
                    if bool(qty_multi.any()):
                        qty_measured = extras["quantity_entropy_norm"][qty_multi].mean()
                    else:
                        qty_measured = extras["quantity_entropy_norm"].new_tensor(0.0)
                    loss = loss - float(cfg.qty_ent_coef) * qty_measured
                if self.reference is not None and float(cfg.ref_kl_coef) > 0.0:
                    ref_kl_b, ref_ex = self.policy.joint_policy_kl(
                        mb_obs, self.reference
                    )
                    ref_kl_mean = ref_kl_b.mean()
                    excess = (ref_kl_mean - float(cfg.ref_kl_target)).clamp(min=0.0)
                    loss = loss + float(cfg.ref_kl_coef) * excess
                    with torch.no_grad():
                        ref_kl_parts = {
                            k: float(v.mean().item())
                            for k, v in ref_ex.items()
                            if k.startswith("ref_kl")
                        }
                    if float(ref_kl_mean.detach().item()) > float(cfg.ref_kl_hard):
                        self.policy.load_state_dict(pre_model)
                        self.opt.load_state_dict(pre_opt)
                        self.ent_coef = pre_ent_coef
                        hard_reverted = True
                        ref_kl_reverted = True
                        n_updates = 0
                        stop = True
                        break

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                n_updates += 1

        if hard_reverted:
            n_updates = 0
        elif action_ent is not None and not hard_reverted:
            measured = float(measured_norm.item())
            err = cfg.ent_target - measured
            self.ent_coef = float(
                np.clip(
                    self.ent_coef * (1.0 + cfg.ent_kp * err),
                    cfg.ent_coef_min,
                    cfg.ent_coef_max,
                )
            )

        self.stats.iterations += 1
        var_ret = float(np.var(b_returns))
        explained_var = (
            float(1.0 - np.var(b_returns - b_values) / (var_ret + 1e-8))
            if var_ret > 0
            else 0.0
        )
        final_kl = (
            running_kl_sum / max(1, running_kl_count) if running_kl_count else float(approx_kl)
        )
        return {
            "pg_loss": float(pg_loss.item()) if pg_loss is not None else 0.0,
            "value_loss": float(v_loss.item()) if v_loss is not None else 0.0,
            "entropy": float(ent.item()) if ent is not None else 0.0,
            "action_entropy": float(action_ent.item()) if action_ent is not None else 0.0,
            "action_entropy_norm": float(measured_norm.item()) if measured_norm is not None else 0.0,
            "action_entropy_max": float(action_ent_max.item()) if action_ent_max is not None else 0.0,
            "cond_entropy": float(cond_ent.item()) if cond_ent is not None else 0.0,
            "approx_kl": float(final_kl),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "n_updates": n_updates,
            "frac_kept": round(n_updates / max(1, total_minibatches), 4),
            "ent_coef": round(self.ent_coef, 5),
            "early_stopped": bool(early_stopped),
            "hard_reverted": bool(hard_reverted),
            "explained_var": round(explained_var, 4),
            "mean_value": float(b_values.mean()),
            "mean_return_target": float(b_returns.mean()),
            "ref_kl": float(ref_kl_mean.item()) if ref_kl_mean is not None else 0.0,
            "ref_kl_reverted": bool(ref_kl_reverted),
            "ref_kl_coef": float(cfg.ref_kl_coef),
            "ent_weighting": cfg.ent_weighting,
            "qty_ent_coef": float(cfg.qty_ent_coef),
            "quantity_entropy_norm": (
                float(qty_measured.item()) if qty_measured is not None else 0.0
            ),
            **{k: round(v, 6) for k, v in ref_kl_parts.items() if k != "ref_kl"},
        }

    def reset_optimizer(self, lr: float | None = None) -> None:
        """Replace Adam with an empty optimizer at `lr` (or cfg.lr)."""
        use_lr = self.cfg.lr if lr is None else float(lr)
        self.cfg.lr = use_lr
        self.opt = make_adam(self.policy, use_lr)

    def load_weights_and_optimizer(self, path: str) -> dict:
        """Restore model, optimizer, and ent_coef. Stats/counters stay put."""
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        if "ent_coef" in ckpt:
            self.ent_coef = float(ckpt["ent_coef"])
        return ckpt

    def save(self, path: str, extra: dict | None = None):
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model": self.policy.state_dict(),
            "optimizer": self.opt.state_dict(),
            "stats": self.stats.state_dict(),
            "ent_coef": self.ent_coef,
            "hyperparameters": {
                "lr": self.cfg.lr,
                "gamma": self.cfg.gamma,
                "gae_lambda": self.cfg.gae_lambda,
                "clip_coef": self.cfg.clip_coef,
                "target_kl": self.cfg.target_kl,
                "ent_target": self.cfg.ent_target,
                "ent_weighting": self.cfg.ent_weighting,
                "qty_ent_coef": self.cfg.qty_ent_coef,
                "ref_kl_coef": self.cfg.ref_kl_coef,
                "ref_kl_target": self.cfg.ref_kl_target,
                "ref_kl_hard": self.cfg.ref_kl_hard,
                "vf_coef": self.cfg.vf_coef,
                "update_epochs": self.cfg.update_epochs,
                "num_minibatches": self.cfg.num_minibatches,
                "rollout_len": self.cfg.rollout_len,
            },
            "metrics": self.stats.rolling_snapshot(),
        }
        if extra:
            payload.update(extra)
        ref_payload = self.reference_payload()
        if ref_payload is not None and "reference" not in payload:
            payload["reference"] = ref_payload
        atomic_torch_save(payload, path)

    def load(self, path: str, *, restore_optimizer: bool = True) -> dict:
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        ver = int(ckpt.get("schema_version", 1))
        if ver < CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(
                f"checkpoint {path!r} is schema v{ver}; this trainer requires "
                f"schema v{CHECKPOINT_SCHEMA_VERSION} (pointer target head + "
                "active-head policy). Start a fresh run."
            )
        try:
            self.policy.load_state_dict(ckpt["model"])
        except RuntimeError as exc:
            raise RuntimeError(
                f"checkpoint {path!r} does not match the current policy "
                f"architecture (pointer target + active-head). {exc}"
            ) from exc
        if restore_optimizer and "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])
        if "stats" in ckpt:
            stats = ckpt["stats"]
            if isinstance(stats, dict) and "ret_20" in stats:
                self.stats.load_state_dict(stats)
            else:
                # Legacy flat TrainStats.__dict__
                for k_, v in stats.items():
                    if hasattr(self.stats, k_) and not isinstance(
                        getattr(self.stats, k_), deque
                    ):
                        setattr(self.stats, k_, v)
        if "ent_coef" in ckpt:
            self.ent_coef = float(ckpt["ent_coef"])
        raw_ref = ckpt.get("reference")
        if isinstance(raw_ref, dict) and raw_ref.get("model"):
            self.load_reference_payload(raw_ref)
        return ckpt
