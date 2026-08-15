"""TCP client for the OpenFrontIO env server (ofai/env/EnvServer.ts).

Frames: [4B LE totalLen][4B LE headerLen][headerJson][blobs...]; the header
carries a `tensors` table (dtype/shape/offset/length) mapping names onto the
concatenated blob region. Zero-copy decode into numpy via frombuffer.
"""

from __future__ import annotations

import json
import socket
import struct
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import spaces

REPO_ROOT = Path(__file__).resolve().parent.parent / "OpenFrontIO"

_DTYPES = {"f32": np.float32, "u8": np.uint8, "i32": np.int32}


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
    return {
        "spatial": tensors["spatial"].reshape(k, s.SPATIAL_CHANNELS, s.SPATIAL_SIZE, s.SPATIAL_SIZE),
        "players": tensors["players"].reshape(k, s.NUM_PLAYER_SLOTS, s.PLAYER_FEATURES),
        "global": tensors["global"].reshape(k, s.GLOBAL_FEATURES),
        "action_mask": tensors["action_mask"].reshape(k, s.NUM_ACTION_TYPES),
        "target_mask": tensors["target_mask"].reshape(k, s.NUM_PLAYER_SLOTS),
        "unit_mask": tensors["unit_mask"].reshape(k, s.NUM_UNIT_TYPES),
        "spawn_regions": tensors["spawn_regions"].reshape(k, s.NUM_REGIONS),
        "build_regions": tensors["build_regions"].reshape(k, s.NUM_REGIONS),
        "boat_regions": tensors["boat_regions"].reshape(k, s.NUM_REGIONS),
    }


class EnvServerProcess:
    """Spawns and owns an EnvServer node subprocess."""

    def __init__(self, port: int, cwd: Path = REPO_ROOT, timeout_s: float = 120.0):
        self.port = port
        cmd = ["npx", "tsx", "ofai/env/EnvServer.ts", "--port", str(port)]
        self.proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        deadline = time.time() + timeout_s
        assert self.proc.stdout is not None
        while time.time() < deadline:
            line = self.proc.stdout.readline()
            if "ENV_SERVER_READY" in line:
                return
            if self.proc.poll() is not None:
                stderr = self.proc.stderr.read() if self.proc.stderr else ""
                raise RuntimeError(f"env server exited early: {stderr}")
        raise TimeoutError("env server did not become ready")

    def stop(self):
        self.proc.terminate()
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.proc.kill()


class VecEnv:
    """Vectorized client: K parallel games behind one request/reply socket."""

    def __init__(self, host: str, port: int):
        self.sock = socket.create_connection((host, port), timeout=600)
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.k = 0

    def init(self, env_configs: list[dict]) -> dict[str, np.ndarray]:
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

    def close(self):
        try:
            self.sock.sendall(_encode_frame({"cmd": "close"}))
            time.sleep(0.2)
        finally:
            self.sock.close()
