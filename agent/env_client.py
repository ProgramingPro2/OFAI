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
import socket
import struct
import subprocess
import threading
import time
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


def _read_frame(sock: socket.socket) -> tuple[dict, dict[str, np.ndarray]]:
    (total_len,) = struct.unpack("<I", _recv_exactly(sock, 4))
    payload = _recv_exactly(sock, total_len)
    (header_len,) = struct.unpack("<I", payload[:4])
    header = json.loads(payload[4 : 4 + header_len])
    blobs = payload[4 + header_len :]
    tensors: dict[str, np.ndarray] = {}
    for name, spec in header.pop("tensors", {}).items():
        arr = np.frombuffer(
            blobs,
            dtype=_DTYPES[spec["dtype"]],
            count=spec["length"] // np.dtype(_DTYPES[spec["dtype"]]).itemsize,
            offset=spec["offset"],
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

    def step(self, actions: np.ndarray) -> StepBatch:
        """actions: int32 [K, 5] in (action_type, target, region, quantity, unit)."""
        assert actions.shape == (self.k, 5)
        self.sock.sendall(
            _encode_frame({"cmd": "step"}, {"actions": actions.astype(np.int32)})
        )
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


def _concat_obs(obses: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return {key: np.concatenate([o[key] for o in obses], axis=0) for key in obses[0]}


class ShardedVecEnv:
    """Several EnvServer processes stepped in parallel behind one VecEnv-like
    interface. Env index i lives on server i // envs_per_server.

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
        return _concat_obs(obses)

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
        self.clients[i].init(self._configs[i * eps : (i + 1) * eps])

    def step(self, actions: np.ndarray) -> StepBatch:
        assert actions.shape == (self.k, 5)
        eps = self.envs_per_server
        chunks = [actions[i * eps : (i + 1) * eps] for i in range(len(self.clients))]
        futures = [
            self.pool.submit(client.step, chunk)
            for client, chunk in zip(self.clients, chunks)
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
        return StepBatch(
            obs=_concat_obs([b.obs for b in batches]),  # type: ignore[union-attr]
            rewards=np.concatenate([b.rewards for b in batches]),  # type: ignore[union-attr]
            dones=np.concatenate([b.dones for b in batches]),  # type: ignore[union-attr]
            infos=[info for b in batches for info in b.infos],  # type: ignore[union-attr]
        )

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
