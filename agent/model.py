"""Policy/value network for the OpenFront agent.

Architecture: CNN over spatial planes, per-slot player MLP (pointer target
scoring + masked mean pool for the trunk), MLP over globals; shared trunk
with an autoregressive factorized policy. Only semantically active heads
contribute to log-probability and entropy for a chosen action type.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import spaces

NEG = 1e9
PLAYER_EMB = 64


def normalized_action_entropy(
    action_ent: torch.Tensor,
    action_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mask-aware action-head entropy in [0, 1].

    per-sample H / log(n_legal). Samples with <=1 legal action are marked
    un-multi (`multi=False`) and given norm=1.0 (fully satisfied / excluded
    from the exploration demand).
    """
    n_legal = action_mask.to(dtype=action_ent.dtype).sum(dim=-1)
    multi = n_legal > 1
    max_ent = torch.log(torch.clamp(n_legal, min=2.0))
    norm = torch.where(
        multi,
        (action_ent / max_ent.clamp(min=1e-6)).clamp(0.0, 1.0),
        torch.ones_like(action_ent),
    )
    max_out = torch.where(multi, max_ent, torch.zeros_like(max_ent))
    return norm, max_out, multi


def explore_action_mask(
    action_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Legal actions that participate in effectiveness-aware entropy."""
    w = (
        torch.as_tensor(spaces.ACTION_ENTROPY_WEIGHTS, device=action_mask.device)
        if weights is None
        else weights.to(device=action_mask.device)
    )
    return (action_mask.to(dtype=w.dtype) * w) > 0


def effective_action_entropy(
    action_logits: torch.Tensor,
    action_mask: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Entropy of the policy renormalized onto explore-worthy legal actions.

    Always-legal utilities (NOOP, and low-weight boat/build/diplomacy) are
    dropped from the bonus so diversity among them is not rewarded.
    Returns (entropy, norm in [0,1], max_ent, multi).
    """
    explore = explore_action_mask(action_mask, weights)
    n = explore.to(dtype=action_logits.dtype).sum(dim=-1)
    multi = n > 1
    masked = action_logits + (explore.to(dtype=action_logits.dtype) - 1.0) * NEG
    empty = n <= 0
    if bool(empty.any()):
        masked = torch.where(
            empty.unsqueeze(-1), torch.zeros_like(masked), masked
        )
    ent = torch.distributions.Categorical(logits=masked).entropy()
    ent = torch.where(empty | ~multi, torch.zeros_like(ent), ent)
    max_ent = torch.log(torch.clamp(n, min=2.0))
    norm = torch.where(
        multi,
        (ent / max_ent.clamp(min=1e-6)).clamp(0.0, 1.0),
        torch.ones_like(ent),
    )
    max_out = torch.where(multi, max_ent, torch.zeros_like(max_ent))
    return ent, norm, max_out, multi


def masked_categorical_kl(
    p_logits: torch.Tensor,
    q_logits: torch.Tensor,
) -> torch.Tensor:
    """KL(p || q) for already-masked categorical logits. [B]"""
    p = torch.distributions.Categorical(logits=p_logits)
    q = torch.distributions.Categorical(logits=q_logits)
    # Numerics can yield tiny negatives; KL is not negative.
    return torch.distributions.kl.kl_divergence(p, q).clamp(min=0.0)


class SpatialEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        c = spaces.SPATIAL_CHANNELS
        s = spaces.SPATIAL_SIZE
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (s // 8) ** 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlayersEncoder(nn.Module):
    """Per-slot embeddings + masked mean pool for trunk context."""

    def __init__(self, out_dim: int = PLAYER_EMB):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spaces.PLAYER_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, players: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # players: [B, S, F]; feature 0 is "exists"
        exists = players[..., 0:1]
        emb = self.mlp(players) * exists  # [B, S, D]
        denom = exists.sum(dim=1).clamp(min=1.0)
        pooled = emb.sum(dim=1) / denom
        return emb, pooled


class GlobalEncoder(nn.Module):
    def __init__(self, out_dim: int = 32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spaces.GLOBAL_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class Policy(nn.Module):
    def __init__(self, trunk_dim: int = 512):
        super().__init__()
        self.trunk_dim = trunk_dim
        self.spatial = SpatialEncoder(256)
        self.players = PlayersEncoder(PLAYER_EMB)
        self.global_enc = GlobalEncoder(32)
        self.trunk = nn.Sequential(
            nn.Linear(256 + PLAYER_EMB + 32, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )
        self.head_action = nn.Linear(trunk_dim, spaces.NUM_ACTION_TYPES)
        # Pointer: score (trunk, candidate slot emb) -> logit per slot
        self.target_scorer = nn.Sequential(
            nn.Linear(trunk_dim + PLAYER_EMB, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        self.head_region = nn.Linear(trunk_dim, spaces.NUM_REGIONS)
        self.head_quantity = nn.Linear(trunk_dim, spaces.NUM_QUANTITIES)
        self.head_unit = nn.Linear(trunk_dim, spaces.NUM_UNIT_TYPES)
        self.head_value = nn.Linear(trunk_dim, 1)

        for head in (
            self.head_action,
            self.head_region,
            self.head_quantity,
            self.head_unit,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        for m in self.target_scorer.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.head_value.weight, gain=1.0)
        nn.init.zeros_(self.head_value.bias)

    def _embed(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        player_emb, pooled = self.players(obs["players"])
        h = self.trunk(
            torch.cat(
                [
                    self.spatial(obs["spatial"]),
                    pooled,
                    self.global_enc(obs["global"]),
                ],
                dim=-1,
            )
        )
        # Bound embedding so head logits stay well-behaved for PPO ratios.
        return torch.tanh(h), player_emb

    @staticmethod
    def _safe_mask(mask: torch.Tensor) -> torch.Tensor:
        """Float 0/1 mask; all-zero rows become all-ones so Categorical is valid."""
        m = mask.float()
        empty = m.sum(dim=-1, keepdim=True) <= 0
        return torch.where(empty, torch.ones_like(m), m)

    @staticmethod
    def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return logits + (Policy._safe_mask(mask) - 1.0) * NEG

    @staticmethod
    def _region_mask(obs: dict[str, torch.Tensor], action_type: torch.Tensor) -> torch.Tensor:
        masks = torch.stack(
            [obs["spawn_regions"], obs["build_regions"], obs["boat_regions"]],
            dim=1,
        ).float()
        idx = torch.zeros_like(action_type)
        idx = torch.where(action_type == spaces.ACTION_BUILD, torch.ones_like(idx), idx)
        idx = torch.where(
            action_type == spaces.ACTION_BOAT, torch.full_like(idx, 2), idx
        )
        return masks.gather(1, idx.view(-1, 1, 1)).squeeze(1)

    @staticmethod
    def _target_mask(
        obs: dict[str, torch.Tensor], action_type: torch.Tensor
    ) -> torch.Tensor:
        """Action-conditioned target mask [B, S]."""
        tm = obs["target_masks"]  # [B, A, S]
        b = action_type.shape[0]
        return tm[torch.arange(b, device=action_type.device), action_type.long()].float()

    def _pointer_target_logits(
        self,
        h: torch.Tensor,
        player_emb: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _d = player_emb.shape
        h_exp = h.unsqueeze(1).expand(-1, s, -1)
        logits = self.target_scorer(torch.cat([h_exp, player_emb], dim=-1)).squeeze(-1)
        return self._masked_logits(logits, target_mask)

    def _head_logits(
        self,
        h: torch.Tensor,
        player_emb: torch.Tensor,
        obs: dict[str, torch.Tensor],
        action_type: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        return {
            "action": self._masked_logits(self.head_action(h), obs["action_mask"]),
            "target": self._pointer_target_logits(
                h, player_emb, self._target_mask(obs, action_type)
            ),
            "region": self._masked_logits(
                self.head_region(h), self._region_mask(obs, action_type)
            ),
            "quantity": self._masked_logits(self.head_quantity(h), obs["quantity_mask"]),
            "unit": self._masked_logits(self.head_unit(h), obs["unit_mask"]),
            "value": self.head_value(h).squeeze(-1),
        }

    @staticmethod
    def _active_masks(action_type: torch.Tensor) -> dict[str, torch.Tensor]:
        """Boolean [B] masks for which conditional heads are active."""
        at = action_type.long()
        return {
            "target": (at == spaces.ACTION_ATTACK)
            | (at == spaces.ACTION_ALLY)
            | (at == spaces.ACTION_BREAK_ALLY)
            | (at == spaces.ACTION_EMBARGO),
            "region": (at == spaces.ACTION_SPAWN)
            | (at == spaces.ACTION_BUILD)
            | (at == spaces.ACTION_BOAT),
            "quantity": (at == spaces.ACTION_ATTACK) | (at == spaces.ACTION_BOAT),
            "unit": at == spaces.ACTION_BUILD,
        }

    def act(
        self,
        obs: dict[str, torch.Tensor],
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Autoregressive action. Always includes action-head logprob; only
        semantically active conditional heads are added.

        Returns (actions [B,5], total_logprob [B], value [B]).
        """
        h, player_emb = self._embed(obs)
        action_logits = self._masked_logits(self.head_action(h), obs["action_mask"])
        dist_a = torch.distributions.Categorical(logits=action_logits)
        if deterministic:
            a_type = action_logits.argmax(dim=-1)
        else:
            a_type = dist_a.sample()
        logp = dist_a.log_prob(a_type)

        logits = self._head_logits(h, player_emb, obs, a_type)
        active = self._active_masks(a_type)

        def _pick(name: str) -> torch.Tensor:
            lg = logits[name]
            if deterministic:
                return lg.argmax(dim=-1)
            return torch.distributions.Categorical(logits=lg).sample()

        target = _pick("target")
        region = _pick("region")
        quantity = _pick("quantity")
        unit = _pick("unit")

        # Only accumulate logprobs for active heads (inactive stay 0 contrib).
        for name, choice in (
            ("target", target),
            ("region", region),
            ("quantity", quantity),
            ("unit", unit),
        ):
            dist = torch.distributions.Categorical(logits=logits[name])
            lp = dist.log_prob(choice)
            logp = logp + torch.where(active[name], lp, torch.zeros_like(lp))

        actions = torch.stack([a_type, target, region, quantity, unit], dim=1)
        return actions, logp, logits["value"]

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sampled act() — kept for call sites that expect nn.Module forward."""
        return self.act(obs, deterministic=False)

    def evaluate(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Active-head logprob/entropy matching act().

        Returns (total_logprob, total_entropy, value, extras). extras has
        raw action_entropy, cond_entropy (active conditional heads only),
        and mask-normalized action_entropy_norm / action_entropy_max /
        action_entropy_multi. Conditional heads are not used as a bonus.
        """
        h, player_emb = self._embed(obs)
        a_type = actions[:, 0].long()
        logits = self._head_logits(h, player_emb, obs, a_type)
        active = self._active_masks(a_type)

        dist_a = torch.distributions.Categorical(logits=logits["action"])
        total_logp = dist_a.log_prob(a_type)
        action_ent = dist_a.entropy()
        total_ent = action_ent.clone()
        cond_ent = torch.zeros_like(action_ent)

        for i, name in enumerate(("target", "region", "quantity", "unit"), start=1):
            dist = torch.distributions.Categorical(logits=logits[name])
            lp = dist.log_prob(actions[:, i].long())
            ent = dist.entropy()
            m = active[name]
            total_logp = total_logp + torch.where(m, lp, torch.zeros_like(lp))
            total_ent = total_ent + torch.where(m, ent, torch.zeros_like(ent))
            cond_ent = cond_ent + torch.where(m, ent, torch.zeros_like(ent))

        norm, max_ent, multi = normalized_action_entropy(
            action_ent, obs["action_mask"]
        )
        eff_ent, eff_norm, eff_max, eff_multi = effective_action_entropy(
            logits["action"], obs["action_mask"]
        )
        extras = {
            "action_entropy": action_ent,
            "cond_entropy": cond_ent,
            "action_entropy_norm": norm,
            "action_entropy_max": max_ent,
            "action_entropy_multi": multi,
            "action_entropy_eff": eff_ent,
            "action_entropy_eff_norm": eff_norm,
            "action_entropy_eff_max": eff_max,
            "action_entropy_eff_multi": eff_multi,
        }
        return total_logp, total_ent, logits["value"], extras

    def joint_policy_kl(
        self,
        obs: dict[str, torch.Tensor],
        other: Policy,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Exact mask-aware KL(self || other) over the factorized policy.

        KL = KL(π_a || ρ_a) + Σ_t π_a(t) Σ_{h ∈ H(t)} KL(π_h|t || ρ_h|t)
        Only active heads under legal masks enter the sum. Inactive-head
        logits (and all-zero masks turned into safe-all-ones) contribute 0
        because those heads are skipped.
        """
        h, player_emb = self._embed(obs)
        with torch.no_grad():
            h_ref, emb_ref = other._embed(obs)
        action_logits = self._masked_logits(self.head_action(h), obs["action_mask"])
        with torch.no_grad():
            action_logits_ref = other._masked_logits(
                other.head_action(h_ref), obs["action_mask"]
            )
        kl_action = masked_categorical_kl(action_logits, action_logits_ref)
        probs = torch.softmax(action_logits, dim=-1)
        per_head = {
            "action": kl_action,
            "target": torch.zeros_like(kl_action),
            "region": torch.zeros_like(kl_action),
            "quantity": torch.zeros_like(kl_action),
            "unit": torch.zeros_like(kl_action),
        }
        legal = obs["action_mask"]
        for a in range(spaces.NUM_ACTION_TYPES):
            if not spaces.active_conditional_heads(a):
                continue
            type_legal = legal[:, a] > 0
            if not bool(type_legal.any()):
                continue
            a_t = torch.full(
                (h.shape[0],), a, device=h.device, dtype=torch.long
            )
            logits = self._head_logits(h, player_emb, obs, a_t)
            with torch.no_grad():
                logits_ref = other._head_logits(h_ref, emb_ref, obs, a_t)
            active = self._active_masks(a_t)
            w = probs[:, a]
            for name in ("target", "region", "quantity", "unit"):
                m = active[name] & type_legal
                if not bool(m.any()):
                    continue
                kl_h = masked_categorical_kl(logits[name], logits_ref[name])
                contrib = torch.where(m, w * kl_h, torch.zeros_like(kl_h))
                per_head[name] = per_head[name] + contrib
        kl = kl_action
        for name in ("target", "region", "quantity", "unit"):
            kl = kl + per_head[name]
        extras = {f"ref_kl_{k}": v for k, v in per_head.items()}
        extras["ref_kl"] = kl
        return kl, extras

    def logits(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Masked logits. Prefer act()/evaluate() for training and inference."""
        h, player_emb = self._embed(obs)
        if actions is None:
            # Union masks for display / value bootstrap (value ignores masks).
            return {
                "action": self._masked_logits(self.head_action(h), obs["action_mask"]),
                "target": self._pointer_target_logits(
                    h,
                    player_emb,
                    obs["target_masks"].float().amax(dim=1),
                ),
                "quantity": self._masked_logits(
                    self.head_quantity(h), obs["quantity_mask"]
                ),
                "unit": self._masked_logits(self.head_unit(h), obs["unit_mask"]),
                "region": self._masked_logits(
                    self.head_region(h),
                    torch.clamp(
                        obs["spawn_regions"]
                        + obs["build_regions"]
                        + obs["boat_regions"],
                        max=1,
                    ),
                ),
                "value": self.head_value(h).squeeze(-1),
            }
        return self._head_logits(h, player_emb, obs, actions[:, 0].long())


def obs_to_tensors(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Numpy obs batch -> float32 tensors on device.

    Uses as_tensor (no np.array always-copy). Read-only frombuffer views are
    copied once on transfer. float16 spatial (rollout storage) is cast to
    float32 so collect logprobs match PPO evaluate().
    """
    import numpy as np

    device = torch.device(device)
    pin = device.type == "cuda"
    out: dict[str, torch.Tensor] = {}
    for key, val in obs.items():
        arr = np.asarray(val)
        if not arr.flags["WRITEABLE"]:
            arr = np.array(arr, copy=True)
        tensor = torch.as_tensor(arr)
        if (
            pin
            and tensor.device.type == "cpu"
            and tensor.dtype == torch.float32
            and key == "spatial"
            and tensor.numel() > 0
        ):
            tensor = tensor.pin_memory()
        out[key] = tensor.to(
            device=device, dtype=torch.float32, non_blocking=pin
        )
    return out


def make_dummy_obs(batch: int = 4, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    """Fully-legal synthetic obs for unit tests."""
    import numpy as np

    s = spaces
    rng = np.random.default_rng(0)
    obs = {
        "spatial": rng.random(
            (batch, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE), dtype=np.float32
        ),
        "players": rng.random((batch, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), dtype=np.float32),
        "global": rng.random((batch, s.GLOBAL_FEATURES), dtype=np.float32),
        "action_mask": np.ones((batch, s.NUM_ACTION_TYPES), np.float32),
        "target_masks": np.ones(
            (batch, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS), np.float32
        ),
        "quantity_mask": np.ones((batch, s.NUM_QUANTITIES), np.float32),
        "unit_mask": np.ones((batch, s.NUM_UNIT_TYPES), np.float32),
        "spawn_regions": np.ones((batch, s.NUM_REGIONS), np.float32),
        "build_regions": np.ones((batch, s.NUM_REGIONS), np.float32),
        "boat_regions": np.ones((batch, s.NUM_REGIONS), np.float32),
    }
    obs["players"][:, :, 0] = 1.0  # all slots exist
    return obs_to_tensors(obs, torch.device(device))
