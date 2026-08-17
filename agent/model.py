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
        extras = {
            "action_entropy": action_ent,
            "cond_entropy": cond_ent,
            "action_entropy_norm": norm,
            "action_entropy_max": max_ent,
            "action_entropy_multi": multi,
        }
        return total_logp, total_ent, logits["value"], extras

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
    """Numpy obs batch -> float tensors on device (copies: wire buffers are
    read-only views into the socket payload)."""
    import numpy as np

    return {
        k: torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        for k, v in obs.items()
    }


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
