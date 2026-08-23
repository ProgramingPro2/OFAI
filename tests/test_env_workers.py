"""Official-runtime worker selection, spawn argv, and init-header backend.

Self-contained except the optional live smoke, which boots one EnvServer
with --workers 2. Does not start training or write checkpoints.

Run: .venv/bin/python -m pytest tests/test_env_workers.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import spaces
from agent.env_client import (
    AUTO_ENV_WORKERS,
    MAX_ENV_WORKERS,
    EnvServerProcess,
    ShardedVecEnv,
    VecEnv,
    env_server_cmd,
    resolve_env_workers,
)
from agent.train import parse_args, parse_workers_arg


def _flat_tensors(k: int) -> dict[str, np.ndarray]:
    s = spaces
    return {
        "spatial": np.zeros(
            k * s.SPATIAL_CHANNELS * s.SPATIAL_SIZE * s.SPATIAL_SIZE, np.float32
        ),
        "players": np.zeros(k * s.NUM_PLAYER_SLOTS * s.PLAYER_FEATURES, np.float32),
        "global": np.zeros(k * s.GLOBAL_FEATURES, np.float32),
        "action_mask": np.ones(k * s.NUM_ACTION_TYPES, np.uint8),
        "target_masks": np.ones(k * s.NUM_ACTION_TYPES * s.NUM_PLAYER_SLOTS, np.uint8),
        "quantity_mask": np.ones(k * s.NUM_QUANTITIES, np.uint8),
        "unit_mask": np.ones(k * s.NUM_UNIT_TYPES, np.uint8),
        "spawn_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "build_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
        "boat_regions": np.ones(k * s.NUM_REGIONS, np.uint8),
    }


def test_parse_workers_arg_auto_and_int():
    assert parse_workers_arg("auto") == "auto"
    assert parse_workers_arg("SAFE") == "auto"
    assert parse_workers_arg("0") == 0
    assert parse_workers_arg("4") == 4
    with pytest.raises(Exception, match="workers"):
        parse_workers_arg("nope")
    with pytest.raises(Exception, match="workers"):
        parse_workers_arg("-1")


def test_parse_args_workers_default_auto():
    args = parse_args(["--run-name", "workers-cli"])
    assert args.workers == "auto"
    assert args.servers == 4
    assert args.envs == 8
    explicit = parse_args(["--workers", "4", "--servers", "4", "--envs", "8"])
    assert explicit.workers == 4


def test_resolve_auto_matches_measured_farm():
    res = resolve_env_workers(
        "auto",
        envs_per_server=8,
        num_servers=4,
        cpu_count=32,
        mem_available_gb=80.0,
        env_var=None,
    )
    assert res.workers == AUTO_ENV_WORKERS
    assert res.source == "auto"
    assert res.fallback is False
    assert "auto-safe" in res.log_line()


def test_resolve_auto_reads_env_then_caps_at_k():
    res = resolve_env_workers(
        "auto",
        envs_per_server=3,
        num_servers=2,
        cpu_count=16,
        mem_available_gb=32.0,
        env_var="8",
    )
    assert res.workers == 3
    assert res.source == "env"
    assert res.fallback is True


def test_resolve_explicit_hard_caps_and_low_memory():
    cap32 = resolve_env_workers(
        64,
        envs_per_server=64,
        num_servers=1,
        cpu_count=64,
        mem_available_gb=80.0,
        env_var=None,
    )
    assert cap32.workers == MAX_ENV_WORKERS
    assert cap32.fallback is True
    low = resolve_env_workers(
        4,
        envs_per_server=8,
        num_servers=4,
        cpu_count=32,
        mem_available_gb=0.4,
        env_var=None,
    )
    assert low.workers == 1
    assert low.fallback is True
    assert "low memory" in low.reason


def test_resolve_auto_cpu_and_memory_budget():
    cpu = resolve_env_workers(
        "auto",
        envs_per_server=8,
        num_servers=16,
        cpu_count=32,
        mem_available_gb=80.0,
        env_var=None,
    )
    assert cpu.workers == 1
    assert cpu.fallback is True
    mem = resolve_env_workers(
        "auto",
        envs_per_server=8,
        num_servers=4,
        cpu_count=32,
        mem_available_gb=6.0,
        env_var=None,
    )
    assert mem.workers == 2
    assert mem.fallback is True


def test_resolve_sequential_requests():
    for requested in (0, 1):
        res = resolve_env_workers(
            requested,
            envs_per_server=8,
            num_servers=4,
            cpu_count=32,
            mem_available_gb=80.0,
            env_var="4",
        )
        assert res.workers == 1
        assert res.source == "cli"
        assert "sequential" in res.reason


def test_env_server_cmd_always_passes_workers():
    assert env_server_cmd(8765, 4) == [
        "npx",
        "tsx",
        "ofai/env/EnvServer.ts",
        "--port",
        "8765",
        "--workers",
        "4",
    ]
    assert env_server_cmd(9000, 0)[-1] == "0"


def test_sharded_spawns_and_restarts_with_workers(monkeypatch):
    spawned: list[tuple[int, int]] = []

    class FakeProc:
        def __init__(self, port, cwd=None, timeout_s=120.0, workers=0, backend="official"):
            spawned.append((int(port), int(workers)))
            self.port = port
            self.workers = workers
            self.backend_kind = backend

        def stop(self):
            pass

        def alive(self):
            return True

    class FakeClient:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self.backend = None
            self.workers = None

        def init(self, cfgs):
            self.backend = "parallel"
            self.workers = 3
            k = len(cfgs)
            return {
                "spatial": np.zeros(
                    (k, spaces.SPATIAL_CHANNELS, spaces.SPATIAL_SIZE, spaces.SPATIAL_SIZE),
                    np.float32,
                ),
                "players": np.zeros((k, spaces.NUM_PLAYER_SLOTS, spaces.PLAYER_FEATURES), np.float32),
                "global": np.zeros((k, spaces.GLOBAL_FEATURES), np.float32),
                "action_mask": np.ones((k, spaces.NUM_ACTION_TYPES), np.uint8),
                "target_masks": np.ones(
                    (k, spaces.NUM_ACTION_TYPES, spaces.NUM_PLAYER_SLOTS), np.uint8
                ),
                "quantity_mask": np.ones((k, spaces.NUM_QUANTITIES), np.uint8),
                "unit_mask": np.ones((k, spaces.NUM_UNIT_TYPES), np.uint8),
                "spawn_regions": np.ones((k, spaces.NUM_REGIONS), np.uint8),
                "build_regions": np.ones((k, spaces.NUM_REGIONS), np.uint8),
                "boat_regions": np.ones((k, spaces.NUM_REGIONS), np.uint8),
            }

        def close(self):
            pass

    monkeypatch.setattr("agent.env_client.EnvServerProcess", FakeProc)
    monkeypatch.setattr("agent.env_client.VecEnv", FakeClient)
    env = ShardedVecEnv(2, 8800, 4, workers=3)
    assert spawned == [(8800, 3), (8801, 3)]
    assert env.workers == 3
    obs = env.init([{"seed": str(i)} for i in range(8)])
    assert obs["spatial"].shape[0] == 8
    assert env.backend == "parallel"
    assert env.reported_workers == 3
    assert "backend=parallel" in env.backend_summary()

    spawned.clear()
    env._restart_shard(1)
    assert spawned == [(8801, 3)]
    env.close()


def test_vecenv_records_optional_backend(monkeypatch):
    env = object.__new__(VecEnv)
    env.sock = MagicMock()
    env.k = 0
    env._configs = []
    env.backend = None
    env.workers = None

    def fake_read(_sock):
        return (
            {"type": "inited", "k": 2, "backend": "parallel", "workers": 2},
            _flat_tensors(2),
        )

    monkeypatch.setattr("agent.env_client._read_frame", fake_read)
    obs = env.init([{"seed": "a"}, {"seed": "b"}])
    assert env.backend == "parallel"
    assert env.workers == 2
    assert obs["target_masks"].shape == (
        2,
        spaces.NUM_ACTION_TYPES,
        spaces.NUM_PLAYER_SLOTS,
    )


def test_dry_run_logs_resolved_workers(capsys):
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--run-name",
            "workers-dry",
            "--servers",
            "4",
            "--envs",
            "8",
            "--workers",
            "4",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert "[env] workers=4 requested=4 source=cli fallback=false" in out
    assert "sim_backend=official" in out


SMOKE_CFG = {
    "map": "Halkidiki",
    "mapSize": "Compact",
    "nations": "disabled",
    "difficulty": "Easy",
    "bots": 1,
    "seed": "workers-live",
    "maxTicks": 400,
    "decisionInterval": 10,
    "shaping": 0,
}


def test_live_parallel_workers_smoke():
    """Small official-core farm: workers=2, K=2, a few legal steps."""
    port = 9217
    server = EnvServerProcess(port, workers=2)
    env = None
    try:
        env = VecEnv("127.0.0.1", port, timeout_s=180.0)
        obs = env.init(
            [
                {**SMOKE_CFG, "seed": "workers-live-0"},
                {**SMOKE_CFG, "seed": "workers-live-1"},
            ]
        )
        assert env.backend == "parallel"
        assert env.workers == 2
        assert obs["spatial"].shape[0] == 2
        assert obs["target_masks"].shape == (
            2,
            spaces.NUM_ACTION_TYPES,
            spaces.NUM_PLAYER_SLOTS,
        )
        rng = np.random.default_rng(1)
        for _ in range(4):
            actions = np.stack(
                [
                    np.array(
                        [
                            int(rng.choice(np.flatnonzero(obs["action_mask"][i]))),
                            0,
                            int(rng.integers(0, spaces.NUM_REGIONS)),
                            2,
                            0,
                        ],
                        dtype=np.int32,
                    )
                    for i in range(env.k)
                ]
            )
            batch = env.step(actions)
            assert batch.obs["spatial"].shape[0] == 2
            assert batch.rewards.shape == (2,)
            assert len(batch.infos) == 2
            obs = batch.obs
        env.close()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        server.stop()


def test_drain_stdout_survives_log_closed_during_shard_restart(tmp_path):
    """stop() used to close the log while _drain_stdout was still writing.

    That raised ValueError in the drain thread during shard restart (v4val50
    it820) and raced the RolloutInterrupted recovery path.
    """
    proc = EnvServerProcess.__new__(EnvServerProcess)
    proc._ready = __import__("threading").Event()
    proc._log = open(tmp_path / "env.log", "ab", buffering=0)
    stdout = MagicMock()
    stdout.readline.side_effect = [b"line\n", b""]
    proc.proc = MagicMock(stdout=stdout)
    proc._log.close()
    proc._drain_stdout()
    assert proc._ready.is_set()
