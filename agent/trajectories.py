"""Optional schema-v2 elite trajectory recording and validation.

Records official-core winning episodes (masks, actions, outcomes) for
behavior cloning. Disabled by default. Validation is deterministic and
fails closed on schema or effective-mask mismatch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import spaces
from .ppo import classify_outcome
from .seeds import holdout_seed

TRAJECTORY_SCHEMA_VERSION = 2
DATASET_KIND = "elite_official_core"
SOURCE_OFFICIAL_CORE = "official-core"
OFFICIAL_BACKENDS = frozenset(
    {"official", "official-core", "sequential", "parallel"}
)
NON_OFFICIAL_BACKENDS = frozenset({"rust", "mixed"})
DEFAULT_ELITE_MAX_EPISODES = 64
DEFAULT_ELITE_MAX_STEPS = 100_000

OBS_SPEC: dict[str, tuple[int, ...]] = {
    "spatial": (spaces.SPATIAL_CHANNELS, spaces.SPATIAL_SIZE, spaces.SPATIAL_SIZE),
    "players": (spaces.NUM_PLAYER_SLOTS, spaces.PLAYER_FEATURES),
    "global": (spaces.GLOBAL_FEATURES,),
    "action_mask": (spaces.NUM_ACTION_TYPES,),
    "target_masks": (spaces.NUM_ACTION_TYPES, spaces.NUM_PLAYER_SLOTS),
    "quantity_mask": (spaces.NUM_QUANTITIES,),
    "unit_mask": (spaces.NUM_UNIT_TYPES,),
    "spawn_regions": (spaces.NUM_REGIONS,),
    "build_regions": (spaces.NUM_REGIONS,),
    "boat_regions": (spaces.NUM_REGIONS,),
}
OBS_KEYS: tuple[str, ...] = tuple(OBS_SPEC)
MASK_KEYS = frozenset(
    {
        "action_mask",
        "target_masks",
        "quantity_mask",
        "unit_mask",
        "spawn_regions",
        "build_regions",
        "boat_regions",
    }
)
_STEP_DTYPES: dict[str, np.dtype] = {
    "spatial": np.dtype(np.float16),
    "players": np.dtype(np.float32),
    "global": np.dtype(np.float32),
    "action_mask": np.dtype(np.uint8),
    "target_masks": np.dtype(np.uint8),
    "quantity_mask": np.dtype(np.uint8),
    "unit_mask": np.dtype(np.uint8),
    "spawn_regions": np.dtype(np.uint8),
    "build_regions": np.dtype(np.uint8),
    "boat_regions": np.dtype(np.uint8),
    "actions": np.dtype(np.int16),
    "episode_id": np.dtype(np.int32),
}
_ACTION_BOUNDS = (
    spaces.NUM_ACTION_TYPES,
    spaces.NUM_PLAYER_SLOTS,
    spaces.NUM_REGIONS,
    spaces.NUM_QUANTITIES,
    spaces.NUM_UNIT_TYPES,
)


class TrajectoryValidationError(ValueError):
    """Fail-closed dataset rejection. `code` is a stable machine token."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"{code}: {message}")


@dataclass
class EliteDataset:
    schema_version: int
    kind: str
    source: str
    steps: dict[str, np.ndarray]
    episodes: list[dict]
    checksum: str
    extra: dict = field(default_factory=dict)

    @property
    def n_steps(self) -> int:
        return int(self.steps["actions"].shape[0])

    @property
    def n_episodes(self) -> int:
        return len(self.episodes)

    def to_payload(self) -> dict:
        payload = {
            "schema_version": int(self.schema_version),
            "kind": self.kind,
            "source": self.source,
            "steps": dict(self.steps),
            "episodes": list(self.episodes),
            "checksum": self.checksum,
        }
        if self.extra:
            payload["extra"] = dict(self.extra)
        return payload


@dataclass
class _OpenEpisode:
    seed: str
    map_name: str
    stage: str
    obs_rows: list[dict] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)


def slice_obs(obs: dict, index: int) -> dict[str, np.ndarray]:
    """Extract one env's observation row as numpy arrays."""
    row: dict[str, np.ndarray] = {}
    for key in OBS_KEYS:
        if key not in obs:
            raise TrajectoryValidationError(
                "SCHEMA", f"observation missing {key!r}"
            )
        arr = np.asarray(obs[key])
        if arr.ndim < 1 or index >= arr.shape[0]:
            raise TrajectoryValidationError(
                "SCHEMA", f"{key} cannot be sliced at index {index}"
            )
        row[key] = np.array(arr[index], copy=True)
    return row


def region_mask_for_action(obs_row: dict, action_type: int) -> np.ndarray:
    if action_type == spaces.ACTION_BUILD:
        return np.asarray(obs_row["build_regions"])
    if action_type == spaces.ACTION_BOAT:
        return np.asarray(obs_row["boat_regions"])
    return np.asarray(obs_row["spawn_regions"])


def target_mask_for_action(obs_row: dict, action_type: int) -> np.ndarray:
    tm = np.asarray(obs_row["target_masks"])
    if tm.shape != (spaces.NUM_ACTION_TYPES, spaces.NUM_PLAYER_SLOTS):
        raise TrajectoryValidationError(
            "SCHEMA", f"target_masks shape {tm.shape} != expected"
        )
    if not 0 <= int(action_type) < spaces.NUM_ACTION_TYPES:
        raise TrajectoryValidationError(
            "SCHEMA", f"action type {action_type} out of range"
        )
    return tm[int(action_type)]


def _legal(mask: np.ndarray, index: int) -> bool:
    if index < 0 or index >= mask.shape[-1]:
        return False
    return bool(np.asarray(mask).reshape(-1)[int(index)] > 0)


def validate_action_against_masks(
    action: np.ndarray | list[int],
    obs_row: dict,
) -> None:
    """Fail closed if the action is illegal under recorded effective masks."""
    vec = np.asarray(action, dtype=np.int64).reshape(-1)
    if vec.shape != (5,):
        raise TrajectoryValidationError("SCHEMA", f"action shape {vec.shape} != (5,)")
    for i, bound in enumerate(_ACTION_BOUNDS):
        if not 0 <= int(vec[i]) < bound:
            raise TrajectoryValidationError(
                "SCHEMA", f"action[{i}]={int(vec[i])} out of range [0, {bound})"
            )
    a_type = int(vec[0])
    action_mask = np.asarray(obs_row["action_mask"]).reshape(-1)
    if action_mask.shape != (spaces.NUM_ACTION_TYPES,):
        raise TrajectoryValidationError(
            "SCHEMA", f"action_mask shape {action_mask.shape} != expected"
        )
    if not _legal(action_mask, a_type):
        raise TrajectoryValidationError(
            "MASK",
            f"action type {a_type} ({spaces.ACTION_NAMES[a_type]}) "
            "is illegal under action_mask",
        )
    active = spaces.active_conditional_heads(a_type)
    if "target" in active:
        tmask = target_mask_for_action(obs_row, a_type)
        if int(np.asarray(tmask).reshape(-1).sum()) <= 0:
            raise TrajectoryValidationError(
                "MASK", f"empty effective target mask for action {a_type}"
            )
        if not _legal(tmask, int(vec[1])):
            raise TrajectoryValidationError(
                "MASK", f"target {int(vec[1])} illegal for action {a_type}"
            )
    if "region" in active:
        rmask = region_mask_for_action(obs_row, a_type)
        if rmask.shape != (spaces.NUM_REGIONS,):
            raise TrajectoryValidationError(
                "SCHEMA", f"region mask shape {rmask.shape} != expected"
            )
        if int(np.asarray(rmask).reshape(-1).sum()) <= 0:
            raise TrajectoryValidationError(
                "MASK", f"empty effective region mask for action {a_type}"
            )
        if not _legal(rmask, int(vec[2])):
            raise TrajectoryValidationError(
                "MASK", f"region {int(vec[2])} illegal for action {a_type}"
            )
    if "quantity" in active:
        qmask = np.asarray(obs_row["quantity_mask"]).reshape(-1)
        if qmask.shape != (spaces.NUM_QUANTITIES,):
            raise TrajectoryValidationError(
                "SCHEMA", f"quantity_mask shape {qmask.shape} != expected"
            )
        if int(qmask.sum()) <= 0:
            raise TrajectoryValidationError(
                "MASK", "empty effective quantity mask"
            )
        if not _legal(qmask, int(vec[3])):
            raise TrajectoryValidationError(
                "MASK", f"quantity {int(vec[3])} illegal under quantity_mask"
            )
    if "unit" in active:
        umask = np.asarray(obs_row["unit_mask"]).reshape(-1)
        if umask.shape != (spaces.NUM_UNIT_TYPES,):
            raise TrajectoryValidationError(
                "SCHEMA", f"unit_mask shape {umask.shape} != expected"
            )
        if int(umask.sum()) <= 0:
            raise TrajectoryValidationError(
                "MASK", "empty effective unit mask"
            )
        if not _legal(umask, int(vec[4])):
            raise TrajectoryValidationError(
                "MASK", f"unit {int(vec[4])} illegal under unit_mask"
            )


def is_official_backend(
    name: str | None,
    *,
    allow_unlabeled: bool = False,
) -> bool:
    """True only for confirmed official-core actor backends.

    Fail closed on rust/mixed/unknown. Unlabeled (None) is official only
    when the caller explicitly allows legacy servers.
    """
    if name is None or str(name).strip() == "":
        return bool(allow_unlabeled)
    key = str(name).strip().lower()
    if key in NON_OFFICIAL_BACKENDS:
        return False
    return key in OFFICIAL_BACKENDS


def is_official_core_win(info: dict, *, source: str) -> bool:
    if source != SOURCE_OFFICIAL_CORE:
        return False
    _cause, won, _stage = classify_outcome(info)
    return bool(won)


def _check_obs_row_schema(obs_row: dict) -> None:
    for key, spec in OBS_SPEC.items():
        if key not in obs_row:
            raise TrajectoryValidationError("SCHEMA", f"missing observation {key!r}")
        arr = np.asarray(obs_row[key])
        if tuple(arr.shape) != spec:
            raise TrajectoryValidationError(
                "SCHEMA", f"{key} shape {arr.shape} != {spec}"
            )


def canonicalize_steps(steps: dict) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    n = int(np.asarray(steps["actions"]).shape[0])
    for key in (*OBS_KEYS, "actions", "episode_id"):
        if key not in steps:
            raise TrajectoryValidationError("SCHEMA", f"steps missing {key!r}")
        dt = _STEP_DTYPES[key]
        arr = np.ascontiguousarray(np.asarray(steps[key]), dtype=dt)
        if arr.shape[0] != n:
            raise TrajectoryValidationError(
                "LENGTH", f"{key} length {arr.shape[0]} != {n}"
            )
        if key in OBS_SPEC and tuple(arr.shape[1:]) != OBS_SPEC[key]:
            raise TrajectoryValidationError(
                "SCHEMA", f"{key} shape {arr.shape} != {(n, *OBS_SPEC[key])}"
            )
        if key == "actions" and arr.shape != (n, 5):
            raise TrajectoryValidationError(
                "SCHEMA", f"actions shape {arr.shape} != {(n, 5)}"
            )
        if key == "episode_id" and arr.shape != (n,):
            raise TrajectoryValidationError(
                "SCHEMA", f"episode_id shape {arr.shape} != {(n,)}"
            )
        out[key] = arr
    return out


def dataset_checksum(payload: dict) -> str:
    """Deterministic blake2b over canonical arrays and episode metadata."""
    steps = canonicalize_steps(payload["steps"])
    h = hashlib.blake2b(digest_size=16)
    h.update(f"schema={int(payload.get('schema_version', 0))}\0".encode())
    h.update(f"kind={payload.get('kind', '')}\0".encode())
    h.update(f"source={payload.get('source', '')}\0".encode())
    for key in (*OBS_KEYS, "actions", "episode_id"):
        h.update(key.encode())
        h.update(b"\0")
        h.update(steps[key].tobytes())
    episodes = json.dumps(
        payload.get("episodes") or [],
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    h.update(episodes.encode())
    return h.hexdigest()


def _episode_length_map(episodes: list[dict], n_steps: int) -> None:
    total = 0
    for i, ep in enumerate(episodes):
        n = int(ep.get("n_steps") or 0)
        if n <= 0:
            raise TrajectoryValidationError(
                "LENGTH", f"episode {i} has n_steps={n}"
            )
        total += n
    if total != n_steps:
        raise TrajectoryValidationError(
            "LENGTH", f"episode n_steps sum {total} != {n_steps}"
        )


def validate_dataset(payload: dict) -> str:
    """Return the canonical checksum or raise TrajectoryValidationError."""
    if not isinstance(payload, dict):
        raise TrajectoryValidationError("SCHEMA", "payload is not a dict")
    ver = int(payload.get("schema_version", 0))
    if ver != TRAJECTORY_SCHEMA_VERSION:
        raise TrajectoryValidationError(
            "SCHEMA_VERSION",
            f"dataset schema_version={ver}; require {TRAJECTORY_SCHEMA_VERSION}",
        )
    if payload.get("kind") != DATASET_KIND:
        raise TrajectoryValidationError(
            "DATASET_KIND",
            f"kind={payload.get('kind')!r}; require {DATASET_KIND!r}",
        )
    if payload.get("source") != SOURCE_OFFICIAL_CORE:
        raise TrajectoryValidationError(
            "SOURCE",
            f"source={payload.get('source')!r}; elite data must be "
            f"{SOURCE_OFFICIAL_CORE!r}",
        )
    steps = canonicalize_steps(payload.get("steps") or {})
    episodes = list(payload.get("episodes") or [])
    if not episodes:
        raise TrajectoryValidationError("LENGTH", "elite dataset has no episodes")
    n = int(steps["actions"].shape[0])
    _episode_length_map(episodes, n)
    cursor = 0
    for i, ep in enumerate(episodes):
        if not bool(ep.get("win")):
            raise TrajectoryValidationError(
                "OUTCOME", f"episode {i} is not an official-core win"
            )
        if str(ep.get("terminal_cause") or "") != "win":
            raise TrajectoryValidationError(
                "OUTCOME",
                f"episode {i} terminal_cause={ep.get('terminal_cause')!r} "
                "is not a core win",
            )
        if str(ep.get("source") or SOURCE_OFFICIAL_CORE) != SOURCE_OFFICIAL_CORE:
            raise TrajectoryValidationError(
                "SOURCE", f"episode {i} source is not official-core"
            )
        n_ep = int(ep["n_steps"])
        ids = steps["episode_id"][cursor : cursor + n_ep]
        if not np.all(ids == i):
            raise TrajectoryValidationError(
                "LENGTH", f"episode_id mismatch for episode {i}"
            )
        cursor += n_ep
    for t in range(n):
        row = {key: steps[key][t] for key in OBS_KEYS}
        validate_action_against_masks(steps["actions"][t], row)
    checksum = dataset_checksum(
        {
            "schema_version": ver,
            "kind": payload["kind"],
            "source": payload["source"],
            "steps": steps,
            "episodes": episodes,
        }
    )
    stored = str(payload.get("checksum") or "")
    if stored and stored != checksum:
        raise TrajectoryValidationError(
            "CHECKSUM", "stored checksum does not match canonical bytes"
        )
    return checksum


def assemble_dataset(
    *,
    obs_rows: list[dict],
    actions: list[np.ndarray] | np.ndarray,
    episode_ids: list[int] | np.ndarray,
    episodes: list[dict],
    extra: dict | None = None,
) -> dict:
    """Stack validated rows into a checksummed schema-v2 payload."""
    if not obs_rows:
        raise TrajectoryValidationError("LENGTH", "no steps to assemble")
    stacked: dict[str, list[np.ndarray]] = {key: [] for key in OBS_KEYS}
    for row in obs_rows:
        _check_obs_row_schema(row)
        for key in OBS_KEYS:
            stacked[key].append(np.asarray(row[key]))
    steps = {key: np.stack(stacked[key], axis=0) for key in OBS_KEYS}
    steps["actions"] = np.asarray(actions)
    steps["episode_id"] = np.asarray(episode_ids)
    payload = {
        "schema_version": TRAJECTORY_SCHEMA_VERSION,
        "kind": DATASET_KIND,
        "source": SOURCE_OFFICIAL_CORE,
        "steps": canonicalize_steps(steps),
        "episodes": list(episodes),
    }
    if extra:
        payload["extra"] = dict(extra)
    payload["checksum"] = dataset_checksum(payload)
    validate_dataset(payload)
    return payload


def save_dataset(payload: dict, path: str | Path) -> Path:
    checksum = validate_dataset(payload)
    dest = Path(path)
    name = dest.name
    if name in {"best.pt", "latest.pt", "stage-entry.pt"} or (
        name.startswith("best-") and name.endswith(".pt")
    ):
        raise TrajectoryValidationError(
            "PRODUCTION",
            f"refusing to write elite dataset to production path {name}",
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out["checksum"] = checksum
    out["steps"] = canonicalize_steps(payload["steps"])
    fd, tmp = tempfile.mkstemp(
        prefix=dest.name + ".", suffix=".tmp", dir=dest.parent
    )
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        torch.save(out, tmp_path)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    return dest


def load_dataset(path: str | Path, *, validate: bool = True) -> EliteDataset:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    checksum = validate_dataset(payload) if validate else str(
        payload.get("checksum") or ""
    )
    if not validate:
        payload["steps"] = canonicalize_steps(payload["steps"])
    return EliteDataset(
        schema_version=int(payload["schema_version"]),
        kind=str(payload["kind"]),
        source=str(payload["source"]),
        steps=payload["steps"],
        episodes=list(payload["episodes"]),
        checksum=checksum,
        extra=dict(payload.get("extra") or {}),
    )


def split_dataset(payload: dict, holdout_seeds: list[str]) -> tuple[dict, dict]:
    """Deterministic train/holdout split by frozen episode seeds.

    Fails closed if a requested holdout seed is missing or duplicated.
    """
    validate_dataset(payload)
    wanted = [str(s) for s in holdout_seeds]
    if not wanted:
        raise TrajectoryValidationError("HOLDOUT", "holdout seed list is empty")
    if len(set(wanted)) != len(wanted):
        raise TrajectoryValidationError("HOLDOUT", "holdout seeds are not unique")
    episodes = list(payload["episodes"])
    by_seed = {str(ep.get("seed") or ""): i for i, ep in enumerate(episodes)}
    if len(by_seed) != len(episodes):
        raise TrajectoryValidationError("HOLDOUT", "episode seeds are not unique")
    missing = [s for s in wanted if s not in by_seed]
    if missing:
        raise TrajectoryValidationError(
            "HOLDOUT", f"missing holdout seeds: {missing[:4]}"
        )
    holdout_ids = {by_seed[s] for s in wanted}
    return (
        _subset_dataset(payload, [i for i in range(len(episodes)) if i not in holdout_ids]),
        _subset_dataset(payload, [by_seed[s] for s in wanted]),
    )


def _subset_dataset(payload: dict, episode_indices: list[int]) -> dict:
    if not episode_indices:
        raise TrajectoryValidationError("LENGTH", "split produced an empty subset")
    steps = payload["steps"]
    obs_rows: list[dict] = []
    actions: list[np.ndarray] = []
    episode_ids: list[int] = []
    new_episodes: list[dict] = []
    for new_i, old_i in enumerate(episode_indices):
        ep = dict(episodes_at(payload, old_i))
        mask = steps["episode_id"] == old_i
        idxs = np.flatnonzero(mask)
        for t in idxs:
            obs_rows.append({key: steps[key][t] for key in OBS_KEYS})
            actions.append(steps["actions"][t])
            episode_ids.append(new_i)
        ep["n_steps"] = int(len(idxs))
        new_episodes.append(ep)
    return assemble_dataset(
        obs_rows=obs_rows,
        actions=actions,
        episode_ids=episode_ids,
        episodes=new_episodes,
        extra=dict(payload.get("extra") or {}),
    )


def episodes_at(payload: dict, index: int) -> dict:
    return payload["episodes"][int(index)]


def frozen_stage_holdout_seeds(stage: str, n: int) -> list[str]:
    return [holdout_seed(stage, i) for i in range(int(n))]


class EliteTrajectoryRecorder:
    """Opt-in collector. Keeps an episode only after an official-core win."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        n_envs: int = 1,
        source: str = SOURCE_OFFICIAL_CORE,
        max_episodes: int = DEFAULT_ELITE_MAX_EPISODES,
        max_steps: int = DEFAULT_ELITE_MAX_STEPS,
        allow_unlabeled_backend: bool = False,
        path: str | Path | None = None,
    ):
        self.enabled = bool(enabled)
        self.source = str(source)
        self.max_episodes = max(1, int(max_episodes))
        self.max_steps = max(1, int(max_steps))
        self.allow_unlabeled_backend = bool(allow_unlabeled_backend)
        self.path = Path(path) if path else None
        self._open: list[_OpenEpisode | None] = [None] * max(1, int(n_envs))
        self._obs_rows: list[dict] = []
        self._actions: list[np.ndarray] = []
        self._episode_ids: list[int] = []
        self._episodes: list[dict] = []
        self._dirty = False

    @property
    def n_episodes(self) -> int:
        return len(self._episodes)

    @property
    def n_steps(self) -> int:
        return len(self._actions)

    def at_bound(self) -> bool:
        return (
            self.n_episodes >= self.max_episodes or self.n_steps >= self.max_steps
        )

    def accepts_backend(self, name: str | None) -> bool:
        return is_official_backend(
            name, allow_unlabeled=self.allow_unlabeled_backend
        )

    def configure(self, n_envs: int) -> None:
        n = max(1, int(n_envs))
        if len(self._open) != n:
            self._open = [None] * n

    def discard_env(self, env_i: int) -> None:
        if 0 <= env_i < len(self._open):
            self._open[env_i] = None

    def discard_open(self) -> None:
        """Drop incomplete episodes (reinit / RolloutInterrupted)."""
        self._open = [None] * len(self._open)

    def on_reset(
        self,
        env_i: int = 0,
        *,
        seed: str = "",
        map_name: str = "",
        stage: str = "",
    ) -> None:
        if not self.enabled or self.at_bound():
            return
        self.configure(max(env_i + 1, len(self._open)))
        self._open[env_i] = _OpenEpisode(
            seed=str(seed), map_name=str(map_name), stage=str(stage)
        )

    def on_step(
        self,
        env_i: int,
        obs: dict,
        action: np.ndarray | list[int],
        *,
        obs_index: int | None = None,
    ) -> None:
        if not self.enabled or self.at_bound():
            return
        self.configure(max(env_i + 1, len(self._open)))
        if self._open[env_i] is None:
            self.on_reset(env_i)
        if self._open[env_i] is None:
            return
        try:
            row = slice_obs(obs, env_i if obs_index is None else int(obs_index))
            _check_obs_row_schema(row)
            vec = np.asarray(action, dtype=np.int64).reshape(-1)
            validate_action_against_masks(vec, row)
        except TrajectoryValidationError:
            self.discard_env(env_i)
            return
        self._open[env_i].obs_rows.append(row)
        self._open[env_i].actions.append(np.array(vec, dtype=np.int16, copy=True))

    def on_done(self, env_i: int, info: dict) -> bool:
        """Keep the open episode iff it is an official-core win. Return kept."""
        if not self.enabled:
            return False
        if env_i >= len(self._open) or self._open[env_i] is None:
            return False
        opened = self._open[env_i]
        self._open[env_i] = None
        assert opened is not None
        if not opened.actions:
            return False
        if self.n_episodes >= self.max_episodes:
            return False
        if not is_official_core_win(info, source=self.source):
            return False
        cause, won, _stage = classify_outcome(info)
        if not won:
            return False
        ep_id = len(self._episodes)
        n = len(opened.actions)
        if self.n_steps + n > self.max_steps:
            return False
        self._obs_rows.extend(opened.obs_rows)
        self._actions.extend(opened.actions)
        self._episode_ids.extend([ep_id] * n)
        self._episodes.append(
            {
                "seed": opened.seed or str(info.get("seed") or ""),
                "map": opened.map_name or str(info.get("map") or ""),
                "stage": opened.stage,
                "n_steps": n,
                "terminal_cause": cause,
                "win": True,
                "peak_tiles_frac": float(
                    info.get("peakTilesFrac", info.get("peak_tiles_frac", 0.0))
                ),
                "ticks": int(info.get("tick", 0)),
                "source": SOURCE_OFFICIAL_CORE,
            }
        )
        self._dirty = True
        return True

    def to_payload(self) -> dict:
        if not self.enabled:
            raise TrajectoryValidationError(
                "DISABLED", "recorder is disabled; elite dataset was not requested"
            )
        return assemble_dataset(
            obs_rows=self._obs_rows,
            actions=self._actions,
            episode_ids=self._episode_ids,
            episodes=self._episodes,
        )

    def save(self, path: str | Path | None = None) -> Path:
        dest = Path(path) if path is not None else self.path
        if dest is None:
            raise TrajectoryValidationError("PATH", "elite save path is not set")
        out = save_dataset(self.to_payload(), dest)
        self._dirty = False
        return out

    def flush(self, path: str | Path | None = None) -> Path | None:
        """Atomic write if a new official-core win is pending."""
        if not self.enabled or not self._dirty or self.n_episodes <= 0:
            return None
        return self.save(path)


def _legal_obs_row(rng: np.random.Generator) -> dict[str, np.ndarray]:
    s = spaces
    row = {
        "spatial": rng.random(
            (s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE), dtype=np.float32
        ),
        "players": rng.random((s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES), dtype=np.float32),
        "global": rng.random((s.GLOBAL_FEATURES,), dtype=np.float32),
        "action_mask": np.ones((s.NUM_ACTION_TYPES,), np.uint8),
        "target_masks": np.ones((s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS), np.uint8),
        "quantity_mask": np.ones((s.NUM_QUANTITIES,), np.uint8),
        "unit_mask": np.ones((s.NUM_UNIT_TYPES,), np.uint8),
        "spawn_regions": np.ones((s.NUM_REGIONS,), np.uint8),
        "build_regions": np.ones((s.NUM_REGIONS,), np.uint8),
        "boat_regions": np.ones((s.NUM_REGIONS,), np.uint8),
    }
    row["players"][:, 0] = 1.0
    return row


def synthetic_elite_dataset(
    *,
    n_episodes: int = 4,
    steps_per_episode: int = 8,
    seed: int = 0,
    stage: str = "duel",
    policy=None,
    device: str | torch.device = "cpu",
    scripted: bool = False,
) -> dict:
    """Fully-legal official-core wins for tests and CPU smokes."""
    from .model import obs_to_tensors

    rng = np.random.default_rng(int(seed))
    if policy is not None:
        torch.manual_seed(int(seed))
        policy.eval()
    rec = EliteTrajectoryRecorder(
        enabled=True, n_envs=1, source=SOURCE_OFFICIAL_CORE
    )
    for ep_i in range(int(n_episodes)):
        rec.on_reset(
            0,
            seed=holdout_seed(stage, ep_i),
            map_name="Halkidiki",
            stage=stage,
        )
        for t in range(int(steps_per_episode)):
            row = _legal_obs_row(rng)
            batch = {k: v[None, ...] for k, v in row.items()}
            if policy is not None and not scripted:
                with torch.no_grad():
                    actions_t, _, _ = policy.act(
                        obs_to_tensors(batch, torch.device(device)),
                        deterministic=True,
                    )
                action = actions_t[0].detach().cpu().numpy()
            elif t == 0:
                action = np.array(
                    [spaces.ACTION_SPAWN, 0, 0, 0, 0], dtype=np.int64
                )
            else:
                action = np.array(
                    [spaces.ACTION_ATTACK, 0, 0, 3, 0], dtype=np.int64
                )
            rec.on_step(0, batch, action, obs_index=0)
        rec.on_done(
            0,
            {
                "win": True,
                "terminalCause": "win",
                "tick": 100 + ep_i,
                "peakTilesFrac": 0.85,
            },
        )
    return rec.to_payload()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a schema-v2 elite dataset")
    p.add_argument("--validate", type=str, required=True, help="dataset path")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    ds = load_dataset(args.validate, validate=True)
    print(
        f"ok schema={ds.schema_version} kind={ds.kind} source={ds.source} "
        f"episodes={ds.n_episodes} steps={ds.n_steps} checksum={ds.checksum}"
    )


if __name__ == "__main__":
    main()
