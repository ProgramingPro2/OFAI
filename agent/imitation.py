"""Behavior-cloning warm-start for schema-v2 Policy active heads.

Trains only the factorized policy heads that Policy.evaluate already
treats as active. Writes a candidate checkpoint; never replaces
production artifacts (best.pt, latest.pt, stage-entry.pt, best-*.pt).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .checkpoint import BEST_SCORE_SENTINEL, STAGE_ENTRY_FILENAME
from .model import Policy, obs_to_tensors
from .ppo import (
    CHECKPOINT_SCHEMA_VERSION,
    TrainStats,
    atomic_torch_save,
    make_adam,
)
from .trajectories import (
    OBS_KEYS,
    EliteDataset,
    TrajectoryValidationError,
    load_dataset,
)

PRODUCTION_CHECKPOINT_NAMES = frozenset(
    {"best.pt", "latest.pt", STAGE_ENTRY_FILENAME}
)


class ProductionCheckpointError(ValueError):
    """Refusing to overwrite a production ranking / resume artifact."""


class ImitationSchemaError(ValueError):
    """Fail-closed checkpoint or dataset schema rejection."""


def is_bc_checkpoint(ckpt: dict | None) -> bool:
    """True when this file is a behavior-clone candidate, not a scored run."""
    if not isinstance(ckpt, dict):
        return False
    if str(ckpt.get("save_reason") or "") == "behavior_clone":
        return True
    bc = ckpt.get("bc")
    return isinstance(bc, dict) and str(bc.get("kind") or "") == (
        "behavior_clone_warm_start"
    )


@dataclass
class BCConfig:
    epochs: int = 4
    batch_size: int = 32
    lr: float = 3e-4
    seed: int = 0
    device: str = "cpu"


@dataclass
class BCResult:
    path: Path
    train_nll: float
    final_nll: float
    steps: int
    episodes: int
    checksum: str
    epochs: int


def is_production_checkpoint_path(path: str | Path) -> bool:
    name = Path(path).name
    if name in PRODUCTION_CHECKPOINT_NAMES:
        return True
    return name.startswith("best-") and name.endswith(".pt")


def assert_safe_output_path(path: str | Path) -> Path:
    dest = Path(path)
    if is_production_checkpoint_path(dest):
        raise ProductionCheckpointError(
            f"refusing to write production checkpoint {dest.name!r}; "
            "use a candidate path such as bc-warmstart.pt"
        )
    return dest


def resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def load_schema_v2_policy(
    path: str | Path,
    device: torch.device,
) -> tuple[Policy, dict]:
    ckpt = torch.load(Path(path), map_location=device, weights_only=False)
    ver = int(ckpt.get("schema_version", 1))
    if ver < CHECKPOINT_SCHEMA_VERSION:
        raise ImitationSchemaError(
            f"checkpoint {path} is schema v{ver}; require "
            f"schema v{CHECKPOINT_SCHEMA_VERSION}"
        )
    if "model" not in ckpt:
        raise ImitationSchemaError(f"checkpoint {path} has no model weights")
    policy = Policy().to(device)
    try:
        policy.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise ImitationSchemaError(
            f"checkpoint {path} does not match Policy active-head tensors: {exc}"
        ) from exc
    return policy, ckpt


def _steps_to_obs(
    dataset: EliteDataset,
    indices: np.ndarray,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    raw = {key: dataset.steps[key][indices] for key in OBS_KEYS}
    return obs_to_tensors(raw, device)


def active_head_nll(
    policy: Policy,
    obs: dict[str, torch.Tensor],
    actions: torch.Tensor,
) -> torch.Tensor:
    """Mean NLL of semantically active heads only."""
    logp, _ent, _value, _extras = policy.evaluate(obs, actions)
    return -logp.mean()


@torch.no_grad()
def dataset_mean_nll(
    policy: Policy,
    dataset: EliteDataset,
    device: torch.device,
    *,
    batch_size: int = 64,
) -> float:
    policy.eval()
    n = dataset.n_steps
    if n <= 0:
        raise TrajectoryValidationError("LENGTH", "dataset has no steps")
    total = 0.0
    seen = 0
    for start in range(0, n, int(batch_size)):
        idx = np.arange(start, min(start + int(batch_size), n))
        obs = _steps_to_obs(dataset, idx, device)
        actions = torch.as_tensor(
            dataset.steps["actions"][idx], device=device, dtype=torch.long
        )
        nll = float(active_head_nll(policy, obs, actions).item())
        total += nll * len(idx)
        seen += len(idx)
    return total / max(1, seen)


def _fresh_policy(
    init_ckpt: str | Path | None,
    device: torch.device,
) -> tuple[Policy, dict | None]:
    if not init_ckpt:
        policy = Policy().to(device)
        return policy, None
    return load_schema_v2_policy(init_ckpt, device)


def write_bc_checkpoint(
    policy: Policy,
    path: str | Path,
    *,
    cfg: BCConfig,
    dataset: EliteDataset,
    train_nll: float,
    init_ckpt: dict | None = None,
    init_path: str | None = None,
) -> Path:
    dest = assert_safe_output_path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    opt = make_adam(policy, cfg.lr)
    stats = TrainStats()
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": {k: v.detach().cpu() for k, v in policy.state_dict().items()},
        "optimizer": opt.state_dict(),
        "stats": stats.state_dict(),
        "ent_coef": 0.05,
        "hyperparameters": {
            "lr": float(cfg.lr),
            "origin": "behavior_clone",
        },
        "metrics": stats.rolling_snapshot(),
        "save_reason": "behavior_clone",
        "pending_stage_baseline": True,
        "best_score": list(BEST_SCORE_SENTINEL),
        "global_best_score": list(BEST_SCORE_SENTINEL),
        "stage_best_score": list(BEST_SCORE_SENTINEL),
        "bc": {
            "kind": "behavior_clone_warm_start",
            "dataset_checksum": dataset.checksum,
            "dataset_steps": dataset.n_steps,
            "dataset_episodes": dataset.n_episodes,
            "init_checkpoint": init_path,
            "train_nll": float(train_nll),
            "epochs": int(cfg.epochs),
            "lr": float(cfg.lr),
            "seed": int(cfg.seed),
            "accepted": False,
            "promoted": False,
        },
    }
    if init_ckpt and isinstance(init_ckpt.get("curriculum"), dict):
        payload["curriculum"] = dict(init_ckpt["curriculum"])
    if init_ckpt and isinstance(init_ckpt.get("hyperparameters"), dict):
        hyps = dict(init_ckpt["hyperparameters"])
        hyps["origin"] = "behavior_clone"
        hyps["lr"] = float(cfg.lr)
        payload["hyperparameters"] = hyps
    atomic_torch_save(payload, dest)
    return dest


def behavior_clone(
    dataset: EliteDataset,
    *,
    out_path: str | Path,
    cfg: BCConfig | None = None,
    init_ckpt: str | Path | None = None,
) -> BCResult:
    """Fit active heads and write a non-production candidate checkpoint."""
    cfg = cfg or BCConfig()
    dest = assert_safe_output_path(out_path)
    device = resolve_device(cfg.device)
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    policy, source = _fresh_policy(init_ckpt, device)
    policy.train()
    opt = make_adam(policy, cfg.lr)
    n = dataset.n_steps
    if n <= 0:
        raise TrajectoryValidationError("LENGTH", "cannot clone an empty dataset")
    batch = max(1, min(int(cfg.batch_size), n))
    last_nll = dataset_mean_nll(policy, dataset, device, batch_size=batch)
    start_nll = last_nll
    for epoch in range(int(cfg.epochs)):
        g = torch.Generator()
        g.manual_seed(int(cfg.seed) + epoch)
        perm = torch.randperm(n, generator=g).cpu().numpy()
        policy.train()
        for start in range(0, n, batch):
            idx = perm[start : start + batch]
            obs = _steps_to_obs(dataset, idx, device)
            actions = torch.as_tensor(
                dataset.steps["actions"][idx], device=device, dtype=torch.long
            )
            loss = active_head_nll(policy, obs, actions)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            last_nll = float(loss.item())
    final_nll = dataset_mean_nll(policy, dataset, device, batch_size=batch)
    write_bc_checkpoint(
        policy,
        dest,
        cfg=cfg,
        dataset=dataset,
        train_nll=final_nll,
        init_ckpt=source,
        init_path=str(init_ckpt) if init_ckpt else None,
    )
    return BCResult(
        path=Path(dest),
        train_nll=float(start_nll),
        final_nll=float(final_nll),
        steps=n,
        episodes=dataset.n_episodes,
        checksum=dataset.checksum,
        epochs=int(cfg.epochs),
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavior-clone a schema-v2 candidate from elite trajectories"
    )
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--out", type=str, required=True, help="candidate .pt path")
    p.add_argument("--init", type=str, default="", help="optional schema-v2 init")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    dataset = load_dataset(args.dataset, validate=True)
    result = behavior_clone(
        dataset,
        out_path=args.out,
        init_ckpt=args.init or None,
        cfg=BCConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            device=args.device,
        ),
    )
    print(
        f"bc candidate={result.path} start_nll={result.train_nll:.4f} "
        f"final_nll={result.final_nll:.4f} steps={result.steps} "
        f"episodes={result.episodes} accepted=false promoted=false"
    )


if __name__ == "__main__":
    main()
