"""PPO trainer with GAE, clipping, and entropy bonus.

Rollouts are collected on the vectorized env (K games), stored as numpy, and
consumed in minibatches on GPU. Action logprobs are summed across the five
factorized heads (standard for factorized discrete policies).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import spaces
from .model import Policy, obs_to_tensors


@dataclass
class PPOConfig:
    rollout_len: int = 128  # decision steps per env per iteration
    minibatch_size: int = 256
    update_epochs: int = 4
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    ent_coef: float = 0.05  # base; adaptively scaled to hold ent_target
    vf_coef: float = 0.5
    vf_clip: float = 1.0  # value-clipping range around rollout values
    target_kl: float = 0.05  # early-stop an update when exceeded
    # Entropy controller: if measured entropy drifts below ent_target we ramp
    # ent_coef up (and back down when high). This is what stops the run2-style
    # exploration collapse (entropy 6.9 -> 2.3 -> dead policy).
    ent_target: float = 3.0
    ent_coef_min: float = 0.01
    ent_coef_max: float = 0.5
    ent_kp: float = 0.05  # proportional gain per iteration
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    device: str = "cuda"


class RolloutBuffer:
    """Rollout storage. Spatial planes are quantized to uint8 (values are all
    in [0, 1]) so a (T, K, 10, 64, 64) buffer costs T*K*40KiB instead of 4x
    that; update() dequantizes back to f32."""

    def __init__(self, rollout_len: int, k: int):
        self.t = rollout_len
        self.k = k
        s = spaces
        self.obs = {
            "spatial": np.zeros((rollout_len, k, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE), np.uint8),
            "players": np.zeros((rollout_len, k, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), np.float32),
            "global": np.zeros((rollout_len, k, s.GLOBAL_FEATURES), np.float32),
            "action_mask": np.zeros((rollout_len, k, s.NUM_ACTION_TYPES), np.uint8),
            "target_mask": np.zeros((rollout_len, k, s.NUM_PLAYER_SLOTS), np.uint8),
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
class TrainStats:
    iterations: int = 0
    env_steps: int = 0
    episodes: int = 0
    wins: int = 0
    deaths: int = 0
    kills: int = 0
    recent_returns: list = field(default_factory=list)
    recent_wins: list = field(default_factory=list)
    sps: float = 0.0

    @property
    def win_rate(self) -> float:
        if not self.recent_wins:
            return 0.0
        return float(np.mean(self.recent_wins[-100:]))

    @property
    def mean_return(self) -> float:
        if not self.recent_returns:
            return 0.0
        return float(np.mean(self.recent_returns[-100:]))


class PPOTrainer:
    def __init__(self, policy: Policy, cfg: PPOConfig, rollout_policy=None):
        self.policy = policy.to(cfg.device)
        self.cfg = cfg
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
        self.stats = TrainStats()
        self.ent_coef = cfg.ent_coef  # adaptive; starts at the base value
        self._rollout_fn = None
        if rollout_policy is not None:
            # Sanity: compiled rollout logprobs must match eager evaluate()
            # logprobs for identical weights/obs/actions, else KL/clip read
            # garbage. Checked once at startup with a fixed obs batch.
            probe = self._probe_obs(policy, torch.device(cfg.device))
            with torch.no_grad():
                a, lp_c, _ = rollout_policy(probe)
                lp_e, _, _ = policy.evaluate(probe, a)
            gap = (lp_c - lp_e).abs().max().item()
            if gap < 0.02:
                self._rollout_fn = rollout_policy
            else:
                print(
                    f"[ppo] compiled policy logp gap {gap:.4f} >= 0.02; "
                    "falling back to eager rollout inference"
                )
        if self._rollout_fn is None:
            self._rollout_fn = self.policy

    @staticmethod
    def _probe_obs(policy: Policy, device: torch.device) -> dict:
        import numpy as np

        rng = np.random.default_rng(0)
        s = spaces
        obs = {
            "spatial": rng.random((4, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE), dtype=np.float32),
            "players": rng.random((4, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), dtype=np.float32),
            "global": rng.random((4, s.GLOBAL_FEATURES), dtype=np.float32),
            "action_mask": np.ones((4, s.NUM_ACTION_TYPES), np.uint8),
            "target_mask": np.ones((4, s.NUM_PLAYER_SLOTS), np.uint8),
            "unit_mask": np.ones((4, s.NUM_UNIT_TYPES), np.uint8),
            "spawn_regions": np.ones((4, s.NUM_REGIONS), np.uint8),
            "build_regions": np.ones((4, s.NUM_REGIONS), np.uint8),
            "boat_regions": np.ones((4, s.NUM_REGIONS), np.uint8),
        }
        from .model import obs_to_tensors

        return obs_to_tensors(obs, device)

    def collect(self, env, obs: dict, buf: RolloutBuffer) -> dict:
        """Run rollout_len decision steps on the vec env. Returns latest obs."""
        cfg = self.cfg
        device = torch.device(cfg.device)
        ep_returns = np.zeros(env.k, np.float32)
        for step in range(cfg.rollout_len):
            # Quantize spatial once and feed the *same* values to the policy
            # and the buffer; otherwise the rollout logprobs (computed on raw
            # f32) systematically mismatch update-time logprobs (computed on
            # dequantized obs) and the KL/clip machinery misreads.
            spatial_u8 = (obs["spatial"] * 255).astype(np.uint8)
            obs_q = dict(obs)
            obs_q["spatial"] = spatial_u8.astype(np.float32) / 255.0
            buf.obs["spatial"][step] = spatial_u8
            for key in buf.obs:
                if key != "spatial":
                    buf.obs[key][step] = obs[key]
            with torch.no_grad():
                obs_t = obs_to_tensors(obs_q, device)
                torch.compiler.cudagraph_mark_step_begin()
                actions_t, logp_t, value_t = self._rollout_fn(obs_t)
            actions = actions_t.cpu().numpy().astype(np.int32)
            batch = env.step(actions)
            buf.actions[step] = actions
            buf.logprobs[step] = logp_t.cpu().numpy()
            buf.values[step] = value_t.cpu().numpy()
            buf.rewards[step] = batch.rewards
            buf.dones[step] = batch.dones.astype(np.float32)

            ep_returns += batch.rewards
            for i, done in enumerate(batch.dones):
                if done:
                    info = batch.infos[i]
                    self.stats.episodes += 1
                    self.stats.recent_returns.append(float(ep_returns[i]))
                    self.stats.recent_wins.append(1.0 if info.get("win") else 0.0)
                    self.stats.kills += int(info.get("kills", 0))
                    if info.get("dead"):
                        self.stats.deaths += 1
                    ep_returns[i] = 0.0
            obs = batch.obs
        self.stats.env_steps += cfg.rollout_len * env.k
        return obs

    def update(self, buf: RolloutBuffer, last_obs: dict) -> dict:
        cfg = self.cfg
        device = torch.device(cfg.device)
        t, k = buf.actions.shape[:2]

        with torch.no_grad():
            last_obs_t = obs_to_tensors(last_obs, device)
            last_value = self.policy.logits(last_obs_t)["value"].cpu().numpy()

        # GAE (buf.dones[s] marks the transition after step s's action)
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

        # Flatten (T, K, ...) -> (T*K, ...)
        flat_obs = {
            key: buf.obs[key].reshape((t * k,) + buf.obs[key].shape[2:])
            for key in buf.obs
        }
        flat_spatial_u8 = flat_obs.pop("spatial")
        b_actions = buf.actions.reshape(t * k, 5)
        b_logprobs = buf.logprobs.reshape(t * k)
        b_advantages = advantages.reshape(t * k)
        b_returns = returns.reshape(t * k)
        b_values = buf.values.reshape(t * k)

        idxs = np.arange(t * k)
        clipfracs = []
        approx_kl = 0.0
        pg_loss = ent = v_loss = None
        n_updates = 0
        stop = False
        # Snapshot so a KL-blowing update can be reverted instead of just
        # stopped; the damage then never reaches the live policy.
        pre_model = {k2: v2.detach().clone() for k2, v2 in self.policy.state_dict().items()}
        pre_opt = self.opt.state_dict()
        for _ in range(cfg.update_epochs):
            if stop:
                break
            np.random.shuffle(idxs)
            for start in range(0, t * k, cfg.minibatch_size):
                mb = idxs[start : start + cfg.minibatch_size]
                mb_obs = {
                    key: torch.as_tensor(flat_obs[key][mb], dtype=torch.float32, device=device)
                    for key in flat_obs
                }
                mb_obs["spatial"] = torch.as_tensor(
                    flat_spatial_u8[mb], dtype=torch.float32, device=device
                ).div_(255.0)
                mb_actions = torch.as_tensor(b_actions[mb], device=device)
                mb_adv = torch.as_tensor(b_advantages[mb], device=device)
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                mb_ret = torch.as_tensor(b_returns[mb], device=device)
                mb_old_logp = torch.as_tensor(b_logprobs[mb], device=device)
                mb_old_value = torch.as_tensor(b_values[mb], device=device)

                logp, entropy, value = self.policy.evaluate(mb_obs, mb_actions)
                ratio = (logp - mb_old_logp).exp()
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (logp - mb_old_logp)).mean().item()
                    clipfracs.append(
                        ((ratio - 1.0).abs() > cfg.clip_coef).float().mean().item()
                    )
                if approx_kl > cfg.target_kl:
                    # Revert to the pre-update weights; a step this far off
                    # the rollout policy corrupts more than it teaches.
                    self.policy.load_state_dict(pre_model)
                    self.opt.load_state_dict(pre_opt)
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
                loss = pg_loss - self.ent_coef * ent + cfg.vf_coef * v_loss

                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.opt.step()
                n_updates += 1

        # Adaptive entropy control (SAC-style proportional controller).
        # Measured entropy below target -> push exploration harder next iter.
        if ent is not None:
            err = cfg.ent_target - float(ent.item())
            self.ent_coef = float(
                np.clip(
                    self.ent_coef * (1.0 + cfg.ent_kp * err),
                    cfg.ent_coef_min,
                    cfg.ent_coef_max,
                )
            )

        self.stats.iterations += 1
        return {
            "pg_loss": float(pg_loss.item()) if pg_loss is not None else 0.0,
            "value_loss": float(v_loss.item()) if v_loss is not None else 0.0,
            "entropy": float(ent.item()) if ent is not None else 0.0,
            "approx_kl": float(approx_kl),
            "clipfrac": float(np.mean(clipfracs)) if clipfracs else 0.0,
            "n_updates": n_updates,
            "ent_coef": round(self.ent_coef, 5),
            "reverted": bool(stop),
            "mean_value": float(b_values.mean()),
            "mean_return_target": float(b_returns.mean()),
        }

    def save(self, path: str):
        torch.save(
            {
                "model": self.policy.state_dict(),
                "optimizer": self.opt.state_dict(),
                "stats": self.stats.__dict__,
                "ent_coef": self.ent_coef,
            },
            path,
        )

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.cfg.device, weights_only=False)
        self.policy.load_state_dict(ckpt["model"])
        self.opt.load_state_dict(ckpt["optimizer"])
        for k_, v in ckpt.get("stats", {}).items():
            setattr(self.stats, k_, v)
        if "ent_coef" in ckpt:
            self.ent_coef = float(ckpt["ent_coef"])
