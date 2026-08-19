"""TCP client for the OpenFrontIO env server (ofai/env/EnvServer.ts).

Frames: [4B LE totalLen][4B LE headerLen][headerJson][blobs...]; the header
carries a `tensors` table (dtype/shape/offset/length) mapping names onto the
concatenated blob region. Zero-copy decode into numpy via frombuffer.

Wire contract (observation tensors):
  spatial, players, global, action_mask, target_masks [K,A,S], quantity_mask [K,Q],
  unit_mask, spawn_regions, build_regions, boat_regions.

Step infos may include terminalCause, peakTilesFrac, rewardTerms,
intentCount, actionAccepted, stageSuccess
(plus legacy win/dead/kills/tick/tilesFrac).
`win` is a core FFA win only; a tile milestone is curriculum_success.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import subprocess
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent / "OpenFrontIO"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

_DTYPES = {"f32": np.float32, "u8": np.uint8, "i32": np.int32}


class RolloutInterrupted(Exception):
    """Raised after a shard failure/restart. The partial rollout must be
    discarded and per-env episode trackers reset; healthy shards were NOT
    re-stepped (retry-all would double-step them).
    """

    def __init__(self, failed_shards: list[int], message: str = ""):
        self.failed_shards = list(failed_shards)
        super().__init__(
            message
            or f"rollout interrupted after restarting shards {self.failed_shards}"
        )


@dataclass
class StepBatch:
    """Batched results for K envs."""

    obs: dict[str, np.ndarray]
    rewards: np.ndarray  # [K] f32
    dones: np.ndarray  # [K] bool
    infos: list[dict]


def _recv_exactly(sock: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(min(n - len(buf), 1 << 20))
        if not chunk:
            raise ConnectionError("env server closed the connection")
        buf.extend(chunk)
    return bytes(buf)


def _recv_into(sock: socket.socket, buf: bytearray, n: int) -> None:
    view = memoryview(buf)
    got = 0
    while got < n:
        nread = sock.recv_into(view[got:], min(n - got, 1 << 20))
        if not nread:
            raise ConnectionError("env server closed the connection")
        got += nread


def _read_frame(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    (total_len,) = struct.unpack("<I", _recv_exactly(sock, 4))
    # Own the payload in a bytearray so frombuffer views stay valid without
    # an extra bytes() copy of the blob region.
    payload = bytearray(total_len)
    _recv_into(sock, payload, total_len)
    (header_len,) = struct.unpack_from("<I", payload, 0)
    header = json.loads(bytes(payload[4 : 4 + header_len]))
    blob_off = 4 + header_len
    tensors: dict[str, np.ndarray] = {}
    for name, spec in header.pop("tensors", {}).items():
        itemsize = np.dtype(_DTYPES[spec["dtype"]]).itemsize
        arr = np.frombuffer(
            payload,
            dtype=_DTYPES[spec["dtype"]],
            count=spec["length"] // itemsize,
            offset=blob_off + spec["offset"],
        )
        tensors[name] = arr
    return header, tensors


def _encode_frame(header: dict, tensors: dict[str, np.ndarray] | None = None) -> bytes:
    header = dict(header)
    specs = {}
    blob = bytearray()
    for name, arr in (tensors or {}).items():
        arr = np.ascontiguousarray(arr)
        if arr.dtype == np.float32:
            dtype = "f32"
        elif arr.dtype == np.uint8:
            dtype = "u8"
        elif arr.dtype in (np.int32, np.int64):
            arr = arr.astype(np.int32)
            dtype = "i32"
        else:
            raise ValueError(f"unsupported dtype {arr.dtype} for tensor {name}")
        data = arr.tobytes()
        specs[name] = {
            "dtype": dtype,
            "shape": [int(arr.size)],
            "offset": len(blob),
            "length": len(data),
        }
        blob.extend(data)
    header["tensors"] = specs
    header_bytes = json.dumps(header).encode()
    total = 4 + len(header_bytes) + len(blob)
    return struct.pack("<II", total, len(header_bytes)) + header_bytes + bytes(blob)


def _obs_dict(tensors: dict[str, np.ndarray], k: int) -> dict[str, np.ndarray]:
    s = spaces
    out = {
        "spatial": tensors["spatial"].reshape(
            k, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE
        ),
        "players": tensors["players"].reshape(k, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES),
        "global": tensors["global"].reshape(k, s.GLOBAL_FEATURES),
        "action_mask": tensors["action_mask"].reshape(k, s.NUM_ACTION_TYPES),
        "unit_mask": tensors["unit_mask"].reshape(k, s.NUM_UNIT_TYPES),
        "spawn_regions": tensors["spawn_regions"].reshape(k, s.NUM_REGIONS),
        "build_regions": tensors["build_regions"].reshape(k, s.NUM_REGIONS),
        "boat_regions": tensors["boat_regions"].reshape(k, s.NUM_REGIONS),
    }
    # Preferred: action-conditioned target_masks [K, A, S]
    if "target_masks" in tensors:
        out["target_masks"] = tensors["target_masks"].reshape(
            k, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS
        )
    elif "target_mask" in tensors:
        # Optional compat for older traces that still emit a flat union mask.
        legacy = tensors["target_mask"].reshape(k, s.NUM_PLAYER_SLOTS)
        out["target_masks"] = np.broadcast_to(
            legacy[:, None, :], (k, s.NUM_ACTION_TYPES, s.NUM_PLAYER_SLOTS)
        ).copy()
    else:
        raise KeyError("observation missing target_masks/target_mask")

    if "quantity_mask" in tensors:
        out["quantity_mask"] = tensors["quantity_mask"].reshape(k, s.NUM_QUANTITIES)
    else:
        out["quantity_mask"] = np.ones((k, s.NUM_QUANTITIES), dtype=np.uint8)
    return out


class EnvServerProcess:
    """Spawns and owns an EnvServer node subprocess.

    stdout/stderr are drained on a background thread into a log file. The
    OpenFront core (and tsx) write to console during play; if those pipes
    fill (~64KiB) the Node process blocks forever and the Python client
    times out.
    """

    def __init__(self, port: int, cwd: Path = REPO_ROOT, timeout_s: float = 120.0):
        self.port = port
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"env-server-{port}.log"
        self._log = open(self.log_path, "ab", buffering=0)
        cmd = ["npx", "tsx", "ofai/env/EnvServer.ts", "--port", str(port)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        self._ready = threading.Event()
        self._drain = threading.Thread(target=self._drain_stdout, daemon=True)
        self._drain.start()
        if not self._ready.wait(timeout_s):
            self.stop()
            raise TimeoutError(
                f"env server on {port} did not become ready; see {self.log_path}"
            )
        if self.proc.poll() is not None:
            raise RuntimeError(
                f"env server on {port} exited early; see {self.log_path}"
            )

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        for raw in iter(self.proc.stdout.readline, b""):
            self._log.write(raw)
            if b"ENV_SERVER_READY" in raw:
                self._ready.set()
        self._ready.set()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        try:
            self._log.close()
        except Exception:
            pass


class VecEnv:
    """Vectorized client: K parallel games behind one request/reply socket."""

    def __init__(self, host: str, port: int, timeout_s: float = 120.0):
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.sock = socket.create_connection((host, port), timeout=timeout_s)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.k = 0
        self._configs: list[dict] = []

    def init(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        self._configs = list(env_configs)
        self.sock.sendall(_encode_frame({"cmd": "init", "envs": env_configs}))
        header, tensors = _read_frame(self.sock)
        if header.get("type") == "error":
            raise RuntimeError(f"env server error: {header.get('error')}")
        assert header["type"] == "inited", header
        self.k = int(header["k"])
        return _obs_dict(tensors, self.k)

    def step(
        self,
        actions: np.ndarray,
        next_configs: list[dict | None] | None = None,
    ) -> StepBatch:
        """actions: int32 [K, 5] in (action_type, target, region, quantity, unit).

        When provided, `next_configs[i]` is applied by the server if env i
        ends this step (Python-owned seed / map stream).
        """
        assert actions.shape == (self.k, 5)
        acts = np.ascontiguousarray(actions, dtype=np.int32)
        header: dict = {"cmd": "step"}
        if next_configs is not None:
            header["nextConfigs"] = next_configs
        self.sock.sendall(_encode_frame(header, {"actions": acts}))
        header, tensors = _read_frame(self.sock)
        if header.get("type") == "error":
            raise RuntimeError(f"env server error: {header.get('error')}")
        assert header["type"] == "step", header
        return StepBatch(
            obs=_obs_dict(tensors, self.k),
            rewards=np.asarray(header["rewards"], dtype=np.float32),
            dones=np.asarray(header["dones"], dtype=bool),
            infos=header["infos"],
        )

    def reset(self, index: int, seed: str | None = None) -> dict[str, np.ndarray]:
        header_req: dict = {"cmd": "reset", "index": index}
        if seed is not None:
            header_req["seed"] = seed
        self.sock.sendall(_encode_frame(header_req))
        header, tensors = _read_frame(self.sock)
        assert header["type"] == "reset", header
        return _obs_dict(tensors, 1)

    def frame(self, index: int = 0) -> dict:
        """Full-resolution render of the real game map for the watch viewer."""
        self.sock.sendall(_encode_frame({"cmd": "frame", "index": index}))
        header, tensors = _read_frame(self.sock)
        assert header["type"] == "frame", header
        w, h = int(header["width"]), int(header["height"])
        return {
            "width": w,
            "height": h,
            "cells": tensors["cells"].reshape(h, w),
            "players": header["players"],
            "units": header["units"],
            "tick": int(header["tick"]),
        }

    def close(self):
        try:
            self.sock.sendall(_encode_frame({"cmd": "close"}))
            time.sleep(0.2)
        finally:
            self.sock.close()


def distribute_active_lanes(
    n: int, num_shards: int, envs_per_server: int
) -> list[list[int]]:
    """Assign episode indices ``0..n-1`` across shards.

    Uses as many shards as possible (up to ``min(n, num_shards)``) without
    placing more than ``envs_per_server`` lanes on one shard. Assignment is
    round-robin so gathering by shard and scattering by index restores
    episode order. Raises if ``n`` exceeds total capacity.
    """
    if n < 0:
        raise ValueError("n must be non-negative")
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    if envs_per_server < 1:
        raise ValueError("envs_per_server must be positive")
    assignment: list[list[int]] = [[] for _ in range(num_shards)]
    if n == 0:
        return assignment
    capacity = num_shards * envs_per_server
    if n > capacity:
        raise ValueError(
            f"cannot place {n} active lanes on {num_shards}x{envs_per_server}"
        )
    used = min(num_shards, n)
    while math.ceil(n / used) > envs_per_server:
        used += 1
    for i in range(n):
        assignment[i % used].append(i)
    return assignment


def _scatter_obs(
    shard_obs: dict[int, dict[str, np.ndarray]],
    assignment: list[list[int]],
    n: int,
    out: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Place per-shard obs back into episode-index order."""
    first = next(iter(shard_obs.values()))
    if out is None or any(
        key not in out
        or out[key].shape[0] != n
        or out[key].shape[1:] != first[key].shape[1:]
        for key in first
    ):
        out = {
            key: np.empty((n,) + first[key].shape[1:], dtype=first[key].dtype)
            for key in first
        }
    for shard_i, obs in shard_obs.items():
        lanes = assignment[shard_i]
        for key in out:
            out[key][lanes] = obs[key]
    return out


def _concat_obs(
    obses: list[dict[str, np.ndarray]],
    out: dict[str, np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    """Concatenate shard obs along env axis. Reuses `out` when shapes match."""
    k = sum(o[next(iter(o))].shape[0] for o in obses)
    first = obses[0]
    if out is None or any(
        key not in out or out[key].shape[0] != k or out[key].shape[1:] != first[key].shape[1:]
        for key in first
    ):
        out = {
            key: np.empty((k,) + first[key].shape[1:], dtype=first[key].dtype)
            for key in first
        }
    offset = 0
    for o in obses:
        n = o[next(iter(first))].shape[0]
        for key in out:
            out[key][offset : offset + n] = o[key]
        offset += n
    return out


class ShardedVecEnv:
    """Several EnvServer processes stepped in parallel behind one VecEnv-like
    interface. Training env index i lives on server i // envs_per_server.

    Eval uses ``init_active`` / ``step_active`` to spread a subset of
    episodes across shards (round-robin) and skip idle lanes. Training
    ``init()`` / ``step()`` always cover the full width.

    On shard failure: restart the dead shard(s) and raise RolloutInterrupted.
    Healthy shards are never re-stepped (retry-all would double-step them).
    """

    def __init__(
        self,
        num_servers: int,
        base_port: int,
        envs_per_server: int,
        connect: bool = False,
    ):
        self.envs_per_server = envs_per_server
        self.k = num_servers * envs_per_server
        self.ports = [base_port + i for i in range(num_servers)]
        self.connect = connect
        self.servers: list[EnvServerProcess] = []
        if not connect:
            self.servers = [EnvServerProcess(p) for p in self.ports]
        self.clients = [VecEnv("127.0.0.1", p) for p in self.ports]
        self.pool = ThreadPoolExecutor(max_workers=num_servers)
        self._configs: list[dict] = []
        self._obs_buf: dict[str, np.ndarray] | None = None
        self._leading_shards = 0
        self._leading_n = 0
        self._active_assignment: list[list[int]] = []
        self._active_shards: list[int] = []
        self._active_n = 0
        self._next_config_fn: Callable[[int], dict] | None = None
        self._on_episode_reset: Callable[[int, dict], None] | None = None

    def set_episode_hooks(
        self,
        next_config_fn: Callable[[int], dict] | None,
        on_episode_reset: Callable[[int, dict], None] | None = None,
    ) -> None:
        """Python-owned auto-reset: next train config per env, and a callback
        after a done env is re-seeded. Eval should clear hooks.
        """
        self._next_config_fn = next_config_fn
        self._on_episode_reset = on_episode_reset

    def init(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        assert len(env_configs) == self.k
        self._configs = list(env_configs)
        return self._init_all()

    def _init_all(self) -> dict[str, np.ndarray]:
        eps = self.envs_per_server
        chunks = [
            self._configs[i * eps : (i + 1) * eps] for i in range(len(self.clients))
        ]
        obses = list(self.pool.map(lambda t: t[0].init(t[1]), zip(self.clients, chunks)))
        self._obs_buf = _concat_obs(obses, self._obs_buf)
        self._leading_shards = len(self.clients)
        self._leading_n = self.k
        self._clear_active()
        return self._obs_buf

    def _clear_active(self) -> None:
        self._active_assignment = []
        self._active_shards = []
        self._active_n = 0

    def _set_active(self, assignment: list[list[int]], n: int) -> None:
        self._active_assignment = assignment
        self._active_shards = [i for i, lanes in enumerate(assignment) if lanes]
        self._active_n = n
        self._leading_shards = len(self._active_shards)
        self._leading_n = n

    def init_active(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        """Init only the shards that receive these configs.

        Episodes are spread across as many shards as possible (round-robin)
        so a 32-game holdout uses idle training shards instead of packing
        into the first few processes. Each participating shard is inited
        with only its assigned configs — idle lanes are not created or
        stepped. Training ``_configs`` are left unchanged so a later
        ``init()`` restores the train layout.

        Returns obs in the original config / episode order.
        """
        n = len(env_configs)
        if n <= 0:
            raise ValueError("init_active requires at least one env config")
        # Never call init() here: that would overwrite training `_configs`
        # and clear the active assignment. A 32-game holdout on a 4x8 farm
        # is n==k and must still scatter + step_active.
        assignment = distribute_active_lanes(
            n, len(self.clients), self.envs_per_server
        )
        self._set_active(assignment, n)

        def _init_one(shard_i: int) -> tuple[int, dict[str, np.ndarray]]:
            cfgs = [env_configs[j] for j in assignment[shard_i]]
            return shard_i, self.clients[shard_i].init(cfgs)

        pairs = list(self.pool.map(_init_one, self._active_shards))
        shard_obs = {i: obs for i, obs in pairs}
        return _scatter_obs(shard_obs, assignment, n)

    def init_leading(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        """Alias for ``init_active`` (distributed, not packed leading shards)."""
        return self.init_active(env_configs)

    def _restart_shard(self, i: int) -> None:
        port = self.ports[i]
        print(f"[env] restarting shard {i} on port {port}")
        try:
            self.clients[i].close()
        except Exception:
            pass
        if not self.connect:
            self.servers[i].stop()
            self.servers[i] = EnvServerProcess(port)
        self.clients[i] = VecEnv("127.0.0.1", port)
        eps = self.envs_per_server
        # Current per-env configs (updated on each done), not the original init.
        self.clients[i].init(self._configs[i * eps : (i + 1) * eps])

    def _next_configs_for(self, n: int) -> list[dict | None] | None:
        fn = getattr(self, "_next_config_fn", None)
        if fn is None:
            return None
        return [fn(i) for i in range(n)]

    def _note_resets(self, dones: np.ndarray, next_configs: list[dict | None] | None) -> None:
        if next_configs is None:
            return
        for i, done in enumerate(dones):
            if not done:
                continue
            cfg = next_configs[i]
            if cfg is None:
                continue
            if i < len(self._configs):
                self._configs[i] = dict(cfg)
            on_reset = getattr(self, "_on_episode_reset", None)
            if on_reset is not None:
                on_reset(i, cfg)

    def step(self, actions: np.ndarray) -> StepBatch:
        assert actions.shape == (self.k, 5)
        eps = self.envs_per_server
        next_configs = self._next_configs_for(self.k)
        chunks = [actions[i * eps : (i + 1) * eps] for i in range(len(self.clients))]
        cfg_chunks: list[list[dict | None] | None]
        if next_configs is None:
            cfg_chunks = [None] * len(self.clients)
        else:
            cfg_chunks = [
                next_configs[i * eps : (i + 1) * eps] for i in range(len(self.clients))
            ]
        futures = [
            self.pool.submit(client.step, chunk, cfgs)
            for client, chunk, cfgs in zip(self.clients, chunks, cfg_chunks)
        ]
        batches: list[StepBatch | None] = [None] * len(self.clients)
        errors: list[tuple[int, BaseException]] = []
        for i, fut in enumerate(futures):
            try:
                batches[i] = fut.result()
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                errors.append((i, exc))
        if errors:
            if self.connect:
                raise errors[0][1]
            failed = [i for i, _ in errors]
            # Also restart any process that is already dead but whose future
            # may have raised a different client.
            for i, s in enumerate(self.servers):
                if not s.alive() and i not in failed:
                    failed.append(i)
            for i in failed:
                self._restart_shard(i)
            raise RolloutInterrupted(failed)

        assert all(b is not None for b in batches)
        ready: list[StepBatch] = batches  # type: ignore[assignment]
        self._obs_buf = _concat_obs([b.obs for b in ready], self._obs_buf)
        dones = np.concatenate([b.dones for b in ready])
        self._note_resets(dones, next_configs)
        return StepBatch(
            obs=self._obs_buf,
            rewards=np.concatenate([b.rewards for b in ready]),
            dones=dones,
            infos=[info for b in ready for info in b.infos],
        )

    def step_active(self, actions: np.ndarray) -> StepBatch:
        """Step only shards that have active lanes from ``init_active``.

        ``actions`` are in episode-index order (same as ``init_active``
        configs). Idle shards are not submitted. Results are scattered
        back into that order.
        """
        n = self._active_n
        assignment = self._active_assignment
        if n <= 0 or not self._active_shards:
            raise RuntimeError("step_active requires a prior init_active")
        if actions.shape != (n, 5):
            raise ValueError(
                f"step_active expected actions ({n}, 5), got {actions.shape}"
            )
        acts = np.ascontiguousarray(actions, dtype=np.int32)
        futures: dict[int, object] = {}
        for i in self._active_shards:
            chunk = acts[assignment[i]]
            futures[i] = self.pool.submit(self.clients[i].step, chunk)
        batches: dict[int, StepBatch] = {}
        errors: list[tuple[int, BaseException]] = []
        for i, fut in futures.items():
            try:
                batches[i] = fut.result()  # type: ignore[union-attr]
            except (TimeoutError, ConnectionError, OSError, RuntimeError) as exc:
                errors.append((i, exc))
        if errors:
            if self.connect:
                raise errors[0][1]
            failed = [i for i, _ in errors]
            for i, s in enumerate(self.servers):
                if not s.alive() and i not in failed:
                    failed.append(i)
            for i in failed:
                self._restart_shard(i)
            raise RolloutInterrupted(failed)
        rewards = np.empty(n, dtype=np.float32)
        dones = np.empty(n, dtype=bool)
        infos: list[dict | None] = [None] * n
        shard_obs: dict[int, dict[str, np.ndarray]] = {}
        for i, batch in batches.items():
            lanes = assignment[i]
            shard_obs[i] = batch.obs
            rewards[lanes] = batch.rewards
            dones[lanes] = batch.dones
            for local, g in enumerate(lanes):
                infos[g] = batch.infos[local]
        return StepBatch(
            obs=_scatter_obs(shard_obs, assignment, n),
            rewards=rewards,
            dones=dones,
            infos=infos,  # type: ignore[arg-type]
        )

    def step_leading(self, actions: np.ndarray) -> StepBatch:
        """Alias for ``step_active``."""
        return self.step_active(actions)

    def reset(self, index: int, seed: str | None = None) -> dict[str, np.ndarray]:
        shard = index // self.envs_per_server
        return self.clients[shard].reset(index % self.envs_per_server, seed)

    def close(self):
        for c in self.clients:
            try:
                c.close()
            except Exception:
                pass
        self.pool.shutdown(wait=False)
        for s in self.servers:
            s.stop()
