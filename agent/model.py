"""Policy/value network for the OpenFront agent.

Architecture: a small CNN over the spatial planes, per-slot MLP + masked mean
pool over player vectors, and an MLP over globals; the concatenated embedding
feeds a shared trunk with factorized categorical heads (action type, target
player, region, quantity, unit) plus a scalar value head.

All heads support legality masks; the region head mask is selected per-sample
according to the chosen action type (spawn/build/boat each have their own
region mask).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from . import spaces

NEG_INF = -1e9


class SpatialEncoder(nn.Module):
    def __init__(self, out_dim: int = 256):
        super().__init__()
        c = spaces.SPATIAL_CHANNELS
        s = spaces.SPATIAL_SIZE
        self.net = nn.Sequential(
            nn.Conv2d(c, 32, 5, stride=2, padding=2),  # s -> s/2
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # s/2 -> s/4
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),  # s/4 -> s/8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (s // 8) ** 2, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PlayersEncoder(nn.Module):
    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(spaces.PLAYER_FEATURES, 64),
            nn.ReLU(),
            nn.Linear(64, out_dim),
            nn.ReLU(),
        )

    def forward(self, players: torch.Tensor) -> torch.Tensor:
        # players: [B, S, F]; feature 0 is "exists" - mask with it.
        exists = players[..., 0:1]
        h = self.mlp(players) * exists
        denom = exists.sum(dim=1).clamp(min=1.0)
        return h.sum(dim=1) / denom


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
        self.spatial = SpatialEncoder(256)
        self.players = PlayersEncoder(64)
        self.global_enc = GlobalEncoder(32)
        self.trunk = nn.Sequential(
            nn.Linear(256 + 64 + 32, trunk_dim),
            nn.ReLU(),
            nn.Linear(trunk_dim, trunk_dim),
            nn.ReLU(),
        )
        self.head_action = nn.Linear(trunk_dim, spaces.NUM_ACTION_TYPES)
        self.head_target = nn.Linear(trunk_dim, spaces.NUM_PLAYER_SLOTS)
        self.head_region = nn.Linear(trunk_dim, spaces.NUM_REGIONS)
        self.head_quantity = nn.Linear(trunk_dim, spaces.NUM_QUANTITIES)
        self.head_unit = nn.Linear(trunk_dim, spaces.NUM_UNIT_TYPES)
        self.head_value = nn.Linear(trunk_dim, 1)

        # Small init for policy heads keeps early exploration broad.
        for head in (
            self.head_action,
            self.head_target,
            self.head_region,
            self.head_quantity,
            self.head_unit,
        ):
            nn.init.orthogonal_(head.weight, gain=0.01)
            nn.init.zeros_(head.bias)
        nn.init.orthogonal_(self.head_value.weight, gain=1.0)
        nn.init.zeros_(self.head_value.bias)

    def _embed(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.trunk(
            torch.cat(
                [
                    self.spatial(obs["spatial"]),
                    self.players(obs["players"]),
                    self.global_enc(obs["global"]),
                ],
                dim=-1,
            )
        )

    @staticmethod
    def _region_mask(obs: dict[str, torch.Tensor], action_type: torch.Tensor) -> torch.Tensor:
        """Pick per-sample region mask by action type."""
        masks = torch.stack(
            [obs["spawn_regions"], obs["build_regions"], obs["boat_regions"]],
            dim=1,
        ).float()  # [B, 3, R]
        idx = torch.full_like(action_type, 0)
        idx = torch.where(action_type == spaces.ACTION_BUILD, torch.ones_like(idx), idx)
        idx = torch.where(
            action_type == spaces.ACTION_BOAT, torch.full_like(idx, 2), idx
        )
        return masks.gather(1, idx.view(-1, 1, 1)).squeeze(1)

    def logits(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """Returns masked logits per head. If actions given ([B,5]), the region
        mask follows the chosen action; otherwise the union of region masks."""
        h = self._embed(obs)
        out = {
            "action": self.head_action(h) + (obs["action_mask"] - 1) * 1e9,
            "target": self.head_target(h) + (obs["target_mask"] - 1) * 1e9,
            "quantity": self.head_quantity(h),
            "unit": self.head_unit(h) + (obs["unit_mask"] - 1) * 1e9,
        }
        if actions is not None:
            region_mask = self._region_mask(obs, actions[:, 0].long())
        else:
            region_mask = torch.clamp(
                obs["spawn_regions"] + obs["build_regions"] + obs["boat_regions"],
                max=1,
            ).float()
        out["region"] = self.head_region(h) + (region_mask - 1) * 1e9
        out["value"] = self.head_value(h).squeeze(-1)
        return out

    def forward(
        self, obs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample an action. Returns (actions [B,5], total logprob [B], value [B]).

        The region head is sampled under the mask for the *sampled* action
        type, so rollout logprobs are consistent with evaluate().
        """
        h = self._embed(obs)
        neg = 1e9
        dist_action = torch.distributions.Categorical(
            logits=self.head_action(h) + (obs["action_mask"] - 1) * neg
        )
        a_type = dist_action.sample()
        region_mask = self._region_mask(obs, a_type)
        dists = [
            torch.distributions.Categorical(
                logits=self.head_target(h) + (obs["target_mask"] - 1) * neg
            ),
            torch.distributions.Categorical(
                logits=self.head_region(h) + (region_mask - 1) * neg
            ),
            torch.distributions.Categorical(logits=self.head_quantity(h)),
            torch.distributions.Categorical(
                logits=self.head_unit(h) + (obs["unit_mask"] - 1) * neg
            ),
        ]
        actions = [a_type]
        total_logp = dist_action.log_prob(a_type)
        for dist in dists:
            a = dist.sample()
            actions.append(a)
            total_logp = total_logp + dist.log_prob(a)
        return torch.stack(actions, dim=1), total_logp, self.head_value(h).squeeze(-1)

    def evaluate(
        self, obs: dict[str, torch.Tensor], actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (total logprob, entropy, value) for stored actions [B,5]."""
        logits = self.logits(obs, actions)
        heads = ["action", "target", "region", "quantity", "unit"]
        total_logp = torch.zeros_like(logits["value"])
        total_ent = torch.zeros_like(logits["value"])
        for i, name in enumerate(heads):
            dist = torch.distributions.Categorical(logits=logits[name])
            total_logp = total_logp + dist.log_prob(actions[:, i])
            total_ent = total_ent + dist.entropy()
        return total_logp, total_ent, logits["value"]


def obs_to_tensors(obs: dict, device: torch.device) -> dict[str, torch.Tensor]:
    """Numpy obs batch -> float tensors on device (copies: wire buffers are
    read-only views into the socket payload)."""
    import numpy as np

    return {
        k: torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        for k, v in obs.items()
    }
