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
import os
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
from .backend import BACKEND_OFFICIAL, BACKEND_RUST, rust_sim_cmd

REPO_ROOT = Path(__file__).resolve().parent.parent / "OpenFrontIO"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

# Match OpenFrontIO resolveWorkerCount. Auto targets the measured K=8 / 4
# worker farm; resource checks may fall back to the sequential oracle.
MAX_ENV_WORKERS = 32
AUTO_ENV_WORKERS = 4
LOW_MEM_GB = 1.5
MIN_MEM_GB_HEADROOM = 2.0
MIN_MEM_GB_PER_SLOT = 0.5
OFAI_ENV_WORKERS = "OFAI_ENV_WORKERS"

_DTYPES = {"f32": np.float32, "u8": np.uint8, "i32": np.int32}

# Dead/truncated shards can raise JSONDecodeError on a zero-length header
# instead of ConnectionError. Treat those as the same recoverable interrupt.
_SHARD_IO_ERRORS = (
    TimeoutError,
    ConnectionError,
    OSError,
    RuntimeError,
    json.JSONDecodeError,
)


class RolloutInterrupted(Exception):
    """Raised after a shard failure/restart. The partial rollout must be
    discarded and per-env episode trackers reset; healthy shards were NOT
    re-stepped (retry-all would double-step them).

    ``failed_lanes`` are the episode / env indices that lived on the dead
    shards. Eval keeps completed seeds and retries only these plus any
    still-pending seeds.
    """

    def __init__(
        self,
        failed_shards: list[int],
        message: str = "",
        failed_lanes: list[int] | None = None,
    ):
        self.failed_shards = list(failed_shards)
        self.failed_lanes = list(failed_lanes or [])
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
    profile: dict | None = None


@dataclass(frozen=True)
class WorkerResolution:
    """Resolved actor-thread count passed to each EnvServer process."""

    requested: int | str
    workers: int
    reason: str
    fallback: bool
    source: str
    cpu_count: int | None = None
    mem_available_gb: float | None = None

    def log_line(self) -> str:
        return (
            f"[env] workers={self.workers} requested={self.requested} "
            f"source={self.source} fallback={str(self.fallback).lower()} "
            f"{self.reason}"
        )


def detect_cpu_count() -> int:
    try:
        n = len(os.sched_getaffinity(0))
        if n > 0:
            return n
    except (AttributeError, OSError):
        pass
    return int(os.cpu_count() or 1)


def detect_mem_available_gb() -> float | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024 * 1024)
    except (OSError, ValueError, IndexError):
        return None
    return None


# Concurrent Turbofan compilation on Node 22 hit a fatal
# RepresentationChangerError (FrameState kMachNone -> kRepTagged) on the
# official EnvServer. NODE_OPTIONS rejects this V8 flag; tsx's re-exec
# also drops it. Spawn the inner node + tsx loader directly.
NODE_NO_CONCURRENT_RECOMPILATION = "--no-concurrent-recompilation"
NODE_REPORT_ON_FATALERROR = "--report-on-fatalerror"


def _tsx_loader_args() -> tuple[str, str]:
    """preflight path and file:// loader URI for the tsx register hooks."""
    dist = REPO_ROOT / "node_modules" / "tsx" / "dist"
    loader = dist / "loader.mjs"
    return str(dist / "preflight.cjs"), loader.resolve().as_uri()


def env_server_cmd(
    port: int,
    workers: int = 0,
    backend: str = BACKEND_OFFICIAL,
) -> list[str]:
    """EnvServer argv. Official always passes --workers. Rust is the
    schema-v2 mirror and is only spawned after fail-closed selection."""
    if str(backend).strip().lower() == BACKEND_RUST:
        return rust_sim_cmd(port, workers)
    preflight, loader = _tsx_loader_args()
    return [
        "node",
        NODE_NO_CONCURRENT_RECOMPILATION,
        NODE_REPORT_ON_FATALERROR,
        f"--require={preflight}",
        f"--import={loader}",
        "ofai/env/EnvServer.ts",
        "--port",
        str(int(port)),
        "--workers",
        str(max(0, int(workers))),
    ]


def _describe_shard_failure(
    index: int,
    exc: BaseException | None,
    server: object | None,
) -> str:
    """One-line restart cause: exception type plus whether Node is still up."""
    alive = False
    rc = None
    if server is not None:
        alive_fn = getattr(server, "alive", None)
        if callable(alive_fn):
            try:
                alive = bool(alive_fn())
            except Exception:
                alive = False
        proc = getattr(server, "proc", None)
        poll = getattr(proc, "poll", None)
        if callable(poll):
            try:
                rc = poll()
            except Exception:
                rc = None
    if exc is None:
        return (
            f"[env] shard {index} process-dead "
            f"alive={str(alive).lower()} rc={rc}"
        )
    return (
        f"[env] shard {index} {type(exc).__name__}: {exc} "
        f"alive={str(alive).lower()} rc={rc}"
    )


def _parse_env_workers(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    try:
        n = int(text)
    except ValueError:
        return None
    if n < 0:
        return None
    return n


def resolve_env_workers(
    requested: int | str,
    *,
    envs_per_server: int,
    num_servers: int,
    cpu_count: int | None = None,
    mem_available_gb: float | None = None,
    env_var: str | None | object = ...,
) -> WorkerResolution:
    """Pick a per-server worker count that is safe for this farm.

    ``auto`` uses OFAI_ENV_WORKERS when that env var is a valid integer,
    otherwise targets the measured 4-worker farm (capped by K). Explicit
    N is honored after hard caps (K, 32) and a critically-low-memory
    sequential fallback. Auto also shrinks to the CPU/memory budget.
    """
    if envs_per_server < 1:
        raise ValueError("envs_per_server must be positive")
    if num_servers < 1:
        raise ValueError("num_servers must be positive")
    if cpu_count is None:
        cpu_count = detect_cpu_count()
    if mem_available_gb is None:
        mem_available_gb = detect_mem_available_gb()
    if env_var is ...:
        env_var = os.environ.get(OFAI_ENV_WORKERS)

    source = "cli"
    notes: list[str] = []
    if isinstance(requested, str) and requested.strip().lower() in ("auto", "safe"):
        env_n = _parse_env_workers(env_var if isinstance(env_var, str) else None)
        if env_n is not None:
            target = env_n
            source = "env"
            notes.append(f"{OFAI_ENV_WORKERS}={env_n}")
        else:
            target = min(envs_per_server, AUTO_ENV_WORKERS) if envs_per_server > 1 else 1
            source = "auto"
            notes.append(f"auto-safe target={target}")
    else:
        try:
            target = int(requested)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid workers {requested!r}") from exc
        if target < 0:
            raise ValueError("workers must be >= 0")
        notes.append(f"explicit={target}")

    workers = target
    fallback = False
    if workers <= 1:
        workers = 1
        notes.append("sequential oracle")
    if workers > envs_per_server:
        workers = envs_per_server
        fallback = True
        notes.append(f"capped at envs_per_server={envs_per_server}")
    if workers > MAX_ENV_WORKERS:
        workers = MAX_ENV_WORKERS
        fallback = True
        notes.append(f"capped at {MAX_ENV_WORKERS}")

    if mem_available_gb is not None and mem_available_gb < LOW_MEM_GB:
        if workers > 1:
            fallback = True
        workers = 1
        notes.append(f"low memory {mem_available_gb:.2f}GiB -> sequential")
    elif source == "auto" and mem_available_gb is not None:
        while (
            workers > 1
            and num_servers * workers * MIN_MEM_GB_PER_SLOT + MIN_MEM_GB_HEADROOM
            > mem_available_gb
        ):
            workers -= 1
            fallback = True
        if fallback and "memory budget" not in "".join(notes):
            notes.append(f"memory budget {mem_available_gb:.1f}GiB")

    if source == "auto" and cpu_count is not None and cpu_count > 0:
        budget = max(1, int(cpu_count) - 1)
        cap = max(1, budget // num_servers)
        if workers > cap:
            workers = cap
            fallback = True
            notes.append(f"cpu budget {cpu_count} -> {cap}/server")
    elif (
        source != "auto"
        and cpu_count is not None
        and workers > 1
        and num_servers * workers > int(cpu_count) * 2
    ):
        notes.append(
            f"warning: {num_servers}x{workers} oversubscribed vs cpu={cpu_count}"
        )

    if workers <= 1:
        workers = 1

    reason = (
        f"servers={num_servers} envs={envs_per_server} "
        f"cpu={cpu_count} mem="
        f"{'?' if mem_available_gb is None else f'{mem_available_gb:.1f}GiB'}; "
        + "; ".join(notes)
    )
    return WorkerResolution(
        requested=requested,
        workers=workers,
        reason=reason,
        fallback=fallback,
        source=source,
        cpu_count=cpu_count,
        mem_available_gb=mem_available_gb,
    )


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
    try:
        header = json.loads(bytes(payload[4 : 4 + header_len]))
    except json.JSONDecodeError as exc:
        raise ConnectionError(f"corrupt env frame header: {exc}") from exc
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

    def __init__(
        self,
        port: int,
        cwd: Path | None = None,
        timeout_s: float = 120.0,
        workers: int = 0,
        backend: str = BACKEND_OFFICIAL,
    ):
        self.port = port
        self.workers = max(0, int(workers))
        self.backend_kind = str(backend).strip().lower() or BACKEND_OFFICIAL
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"env-server-{port}.log"
        self._log = open(self.log_path, "ab", buffering=0)
        cmd = env_server_cmd(port, self.workers, self.backend_kind)
        if cwd is None:
            cwd = (
                Path(__file__).resolve().parent.parent
                if self.backend_kind == BACKEND_RUST
                else REPO_ROOT
            )
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

    def rss_mb(self) -> float:
        pid = getattr(self.proc, "pid", None)
        if not pid:
            return 0.0
        try:
            with open(f"/proc/{int(pid)}/status", encoding="utf-8") as fh:
                for line in fh:
                    if line.startswith("VmRSS:"):
                        return int(line.split()[1]) / 1024.0
        except (OSError, ValueError, IndexError):
            return 0.0
        return 0.0

    def _drain_stdout(self) -> None:
        assert self.proc.stdout is not None
        try:
            for raw in iter(self.proc.stdout.readline, b""):
                try:
                    if self._log.closed:
                        break
                    self._log.write(raw)
                except ValueError:
                    # stop() may close the log while this thread is draining
                    break
                if b"ENV_SERVER_READY" in raw:
                    self._ready.set()
        except ValueError:
            pass
        finally:
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
        if self.proc.stdout is not None:
            try:
                self.proc.stdout.close()
            except Exception:
                pass
        if self._drain.is_alive():
            self._drain.join(timeout=2)
        try:
            if not self._log.closed:
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
        self.backend: str | None = None
        self.workers: int | None = None

    def init(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        self._configs = list(env_configs)
        self.sock.sendall(_encode_frame({"cmd": "init", "envs": env_configs}))
        header, tensors = _read_frame(self.sock)
        if header.get("type") == "error":
            raise RuntimeError(f"env server error: {header.get('error')}")
        assert header["type"] == "inited", header
        self.k = int(header["k"])
        backend = header.get("backend")
        self.backend = str(backend) if backend is not None else None
        raw_workers = header.get("workers")
        try:
            self.workers = int(raw_workers) if raw_workers is not None else None
        except (TypeError, ValueError):
            self.workers = None
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
            profile=header.get("profile") if isinstance(header.get("profile"), dict) else None,
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


def _merge_step_profiles(profiles: list[dict | None]) -> dict | None:
    """Sum per-shard server timing counters from step headers."""
    keys = ("simMs", "maskObsMs", "serializeMs", "resetMs", "resets")
    merged = {key: 0.0 for key in keys}
    any_profile = False
    for profile in profiles:
        if not profile:
            continue
        any_profile = True
        for key in keys:
            merged[key] += float(profile.get(key) or 0)
    return merged if any_profile else None


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
    ``init()`` / ``step()`` always cover the full width. ``step_active``
    can also skip shards whose assigned lanes are already done.

    On shard failure: restart the dead shard(s) and raise RolloutInterrupted
    with ``failed_lanes`` so eval retries only those seeds. Healthy shards
    are never re-stepped (retry-all would double-step them).
    """

    def __init__(
        self,
        num_servers: int,
        base_port: int,
        envs_per_server: int,
        connect: bool = False,
        workers: int = 0,
        backend: str = BACKEND_OFFICIAL,
        timeout_s: float = 180.0,
    ):
        self.envs_per_server = envs_per_server
        self.k = num_servers * envs_per_server
        self.ports = [base_port + i for i in range(num_servers)]
        self.connect = connect
        self.workers = max(0, int(workers))
        self.sim_backend = str(backend).strip().lower() or BACKEND_OFFICIAL
        self.timeout_s = float(timeout_s)
        self.backend: str | None = None
        self.reported_workers: int | None = None
        self.servers: list[EnvServerProcess] = []
        if not connect:
            self.servers = [
                EnvServerProcess(
                    p,
                    workers=self.workers,
                    backend=self.sim_backend,
                    timeout_s=timeout_s,
                )
                for p in self.ports
            ]
        self.clients = [
            VecEnv("127.0.0.1", p, timeout_s=timeout_s) for p in self.ports
        ]
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

    def rss_mb(self) -> float:
        return sum(s.rss_mb() for s in self.servers)

    def backend_summary(self) -> str:
        return (
            f"backend={self.backend or 'unknown'} "
            f"sim_backend={getattr(self, 'sim_backend', BACKEND_OFFICIAL)} "
            f"workers={self.reported_workers}"
        )

    def _note_backends(self, shards: list[int] | None = None) -> None:
        """Record optional init-header backend/workers from participating shards."""
        if shards is None:
            clients = list(self.clients)
        else:
            clients = [
                self.clients[i] for i in shards if 0 <= i < len(self.clients)
            ]
        names = [getattr(c, "backend", None) for c in clients]
        counts = [getattr(c, "workers", None) for c in clients]
        known = [str(n) for n in names if n]
        if not known:
            self.backend = None
        elif all(n == known[0] for n in known):
            self.backend = known[0]
        else:
            self.backend = "mixed"
        known_w = []
        for w in counts:
            if w is None:
                continue
            try:
                known_w.append(int(w))
            except (TypeError, ValueError):
                continue
        if known_w and all(w == known_w[0] for w in known_w):
            self.reported_workers = known_w[0]
        elif known_w:
            self.reported_workers = max(known_w)
        else:
            self.reported_workers = None

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
        futures = [
            self.pool.submit(client.init, chunk)
            for client, chunk in zip(self.clients, chunks)
        ]
        obses: list[dict[str, np.ndarray] | None] = [None] * len(self.clients)
        errors: list[tuple[int, BaseException]] = []
        for i, fut in enumerate(futures):
            try:
                obses[i] = fut.result()
            except _SHARD_IO_ERRORS as exc:
                errors.append((i, exc))
        if errors:
            self._restart_failed_shards(errors)
        assert all(o is not None for o in obses)
        self._obs_buf = _concat_obs(obses, self._obs_buf)  # type: ignore[arg-type]
        self._leading_shards = len(self.clients)
        self._leading_n = self.k
        self._clear_active()
        self._note_backends()
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

    def _lanes_for_shards(self, shards: list[int]) -> list[int]:
        """Episode / env indices owned by ``shards`` under the current layout."""
        assignment = getattr(self, "_active_assignment", None) or []
        active_n = int(getattr(self, "_active_n", 0) or 0)
        if assignment and active_n > 0:
            lanes: list[int] = []
            for i in shards:
                if 0 <= i < len(assignment):
                    lanes.extend(assignment[i])
            return lanes
        eps = self.envs_per_server
        lanes = []
        for i in shards:
            start = i * eps
            lanes.extend(range(start, min(self.k, start + eps)))
        return lanes

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

        futures = {i: self.pool.submit(_init_one, i) for i in self._active_shards}
        pairs: list[tuple[int, dict[str, np.ndarray]]] = []
        errors: list[tuple[int, BaseException]] = []
        for i, fut in futures.items():
            try:
                pairs.append(fut.result())
            except _SHARD_IO_ERRORS as exc:
                errors.append((i, exc))
        if errors:
            self._restart_failed_shards(errors)
        shard_obs = {i: obs for i, obs in pairs}
        self._note_backends(self._active_shards)
        return _scatter_obs(shard_obs, assignment, n)

    def init_leading(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
        """Alias for ``init_active`` (distributed, not packed leading shards)."""
        return self.init_active(env_configs)

    def _restart_failed_shards(
        self, errors: list[tuple[int, BaseException]]
    ) -> None:
        """Restart dead shards and raise RolloutInterrupted.

        Used by init/step so a truncated JSON frame is the same recovery
        path as a closed socket. ``connect`` mode (tests / attach) re-raises
        the first raw error instead.
        """
        if self.connect:
            raise errors[0][1]
        failed = [i for i, _ in errors]
        by_exc = {i: exc for i, exc in errors}
        for i, s in enumerate(self.servers):
            if not s.alive() and i not in failed:
                failed.append(i)
        for i in failed:
            server = self.servers[i] if i < len(self.servers) else None
            print(_describe_shard_failure(i, by_exc.get(i), server))
            self._restart_shard(i)
        raise RolloutInterrupted(failed, failed_lanes=self._lanes_for_shards(failed))

    def _restart_shard(self, i: int) -> None:
        port = self.ports[i]
        print(f"[env] restarting shard {i} on port {port}")
        try:
            self.clients[i].close()
        except Exception:
            pass
        if not self.connect:
            self.servers[i].stop()
            self.servers[i] = EnvServerProcess(
                port,
                workers=int(getattr(self, "workers", 0) or 0),
                backend=str(getattr(self, "sim_backend", BACKEND_OFFICIAL)),
                timeout_s=float(getattr(self, "timeout_s", 180.0)),
            )
        self.clients[i] = VecEnv(
            "127.0.0.1",
            port,
            timeout_s=float(getattr(self, "timeout_s", 180.0)),
        )
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
            except _SHARD_IO_ERRORS as exc:
                errors.append((i, exc))
        if errors:
            self._restart_failed_shards(errors)

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
            profile=_merge_step_profiles([b.profile for b in ready]),
        )

    def step_active(
        self, actions: np.ndarray, live: np.ndarray | None = None
    ) -> StepBatch:
        """Step only shards that have live lanes from ``init_active``.

        ``actions`` are in episode-index order (same as ``init_active``
        configs). Idle shards, and shards whose assigned lanes are all
        done, are not submitted. Results are scattered back into that
        order. Done lanes on a skipped shard are filled with a zero
        reward, ``done=True``, and a dummy observation.
        """
        n = self._active_n
        assignment = self._active_assignment
        if n <= 0 or not self._active_shards:
            raise RuntimeError("step_active requires a prior init_active")
        if actions.shape != (n, 5):
            raise ValueError(
                f"step_active expected actions ({n}, 5), got {actions.shape}"
            )
        if live is None:
            live_mask = np.ones(n, dtype=bool)
        else:
            live_mask = np.asarray(live, dtype=bool)
            if live_mask.shape != (n,):
                raise ValueError(
                    f"step_active live mask expected ({n},), got {live_mask.shape}"
                )
        acts = np.ascontiguousarray(actions, dtype=np.int32)
        futures: dict[int, object] = {}
        skipped: list[int] = []
        for i in self._active_shards:
            lanes = assignment[i]
            if not any(live_mask[j] for j in lanes):
                skipped.append(i)
                continue
            chunk = acts[lanes]
            futures[i] = self.pool.submit(self.clients[i].step, chunk)
        batches: dict[int, StepBatch] = {}
        errors: list[tuple[int, BaseException]] = []
        for i, fut in futures.items():
            try:
                batches[i] = fut.result()  # type: ignore[union-attr]
            except _SHARD_IO_ERRORS as exc:
                errors.append((i, exc))
        if errors:
            self._restart_failed_shards(errors)
        rewards = np.zeros(n, dtype=np.float32)
        dones = np.zeros(n, dtype=bool)
        infos: list[dict | None] = [None] * n
        shard_obs: dict[int, dict[str, np.ndarray]] = {}
        for i, batch in batches.items():
            lanes = assignment[i]
            shard_obs[i] = batch.obs
            rewards[lanes] = batch.rewards
            dones[lanes] = batch.dones
            for local, g in enumerate(lanes):
                infos[g] = batch.infos[local]
        if skipped:
            template = next(iter(batches.values())).obs if batches else None
            for i in skipped:
                lanes = assignment[i]
                dones[lanes] = True
                for g in lanes:
                    infos[g] = {}
                if template is not None:
                    shard_obs[i] = {
                        key: np.zeros(
                            (len(lanes),) + template[key].shape[1:],
                            dtype=template[key].dtype,
                        )
                        for key in template
                    }
        for g in range(n):
            if infos[g] is None:
                infos[g] = {}
                dones[g] = True
        if not shard_obs:
            raise RuntimeError("step_active has no live shards to step")
        return StepBatch(
            obs=_scatter_obs(shard_obs, assignment, n),
            rewards=rewards,
            dones=dones,
            infos=infos,  # type: ignore[arg-type]
            profile=_merge_step_profiles([b.profile for b in batches.values()]),
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
