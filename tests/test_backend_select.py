"""Fail-closed official/rust backend selection.

Does not start EnvServer or write checkpoints. Injects capability documents
so Rust CLI is not required except the optional live rust-binary smoke.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.backend import (
    BACKEND_OFFICIAL,
    BACKEND_RUST,
    config_allowed_on_backend,
    parse_backend_arg,
    refuse_mid_episode_switch,
    required_capabilities,
    rust_blocked_capabilities,
    rust_sim_cmd,
    select_sim_backend,
    select_training_backend,
    stage_env_config,
)
from agent.curriculum import STAGES
from agent.env_client import env_server_cmd
from agent.train import parse_args


SUPPORTED_CAPS = {
    "backend": "rust",
    "schema": "v2",
    "fallback": "official",
    "supported": {
        "rng_hash": True,
        "terrain": True,
        "tick": True,
        "economy": True,
        "spawn": True,
        "wilderness_attack": True,
        "player_attack": True,
        "retreat": True,
        "observations_early": True,
        "boats": True,
        "structures": True,
        "diplomacy": True,
        "tribe_ai": True,
        "nations_ai": True,
    },
    "unsupported": {
        "boats": False,
        "structures": False,
        "diplomacy": False,
        "nations_ai": False,
        "tribe_ai": False,
        "doomsday": False,
        "nukes": False,
        "rails": False,
    },
}

WILDERNESS = {
    "allowedActions": ["noop", "spawn", "attack", "retreat_all"],
    "nations": "disabled",
    "bots": 0,
}

DUEL = {
    "allowedActions": ["noop", "spawn", "attack", "retreat_all", "boat"],
    "nations": "disabled",
    "bots": 1,
}

BOATS_ONLY = {
    "allowedActions": ["noop", "spawn", "attack", "retreat_all", "boat"],
    "nations": "disabled",
    "bots": 0,
}


def test_parse_backend_and_cli_default_official():
    assert parse_backend_arg("AUTO") == "auto"
    assert parse_backend_arg("rust") == "rust"
    with pytest.raises(ValueError, match="backend"):
        parse_backend_arg("mixed")
    args = parse_args(["--run-name", "be-cli"])
    assert args.backend == "auto"
    official = parse_args(["--run-name", "be-off", "--backend", "official"])
    assert official.backend == "official"


def test_required_capabilities_from_actions_and_opponents():
    assert required_capabilities(WILDERNESS) == ()
    assert "boats" in required_capabilities(DUEL)
    assert "tribe_ai" in required_capabilities(DUEL)
    nations = {
        "allowedActions": list(range(9)),
        "nations": 8,
        "bots": 20,
    }
    caps = required_capabilities(nations)
    assert "boats" in caps
    assert "structures" in caps
    assert "diplomacy" in caps
    assert "nations_ai" in caps
    assert "tribe_ai" in caps
    missing = required_capabilities({})
    assert "boats" in missing


def test_official_always_selected_for_production_request():
    dec = select_sim_backend(
        "official",
        [WILDERNESS],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert dec.backend == BACKEND_OFFICIAL
    assert dec.fallback is False
    assert "production official-core" in dec.reason


def test_auto_uses_rust_only_when_every_cap_is_supported():
    rust = select_sim_backend(
        "auto",
        [WILDERNESS],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert rust.backend == BACKEND_RUST
    assert rust.fallback is False
    assert rust.schema == "v2"

    boats = select_sim_backend(
        "auto",
        [BOATS_ONLY],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert boats.backend == BACKEND_RUST
    assert boats.fallback is False

    duel = select_sim_backend(
        "auto",
        [DUEL],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert duel.backend == BACKEND_RUST
    assert duel.fallback is False
    assert "tribe_ai" in duel.required
    assert "nations_ai" not in duel.blocked


def test_rust_request_fails_closed_without_approximating():
    nations = {
        "allowedActions": list(range(9)),
        "nations": 2,
        "bots": 0,
    }
    dec = select_sim_backend(
        "rust",
        [nations],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert dec.backend == BACKEND_RUST
    assert dec.fallback is False
    assert "nations_ai" not in dec.blocked
    empty = select_sim_backend(
        "rust",
        [nations],
        capabilities={"schema": "v2", "fallback": "official", "supported": {}},
        rust_available=True,
    )
    assert empty.backend == BACKEND_OFFICIAL
    assert empty.fallback is True
    assert "nations_ai" in empty.blocked


def test_unknown_or_missing_caps_fail_closed():
    build_only = {
        "allowedActions": ["noop", "spawn", "attack", "retreat_all", "build"],
        "nations": "disabled",
        "bots": 0,
    }
    empty = select_sim_backend(
        "auto",
        [build_only],
        capabilities={"schema": "v2", "fallback": "official", "supported": {}},
        rust_available=True,
    )
    assert empty.backend == BACKEND_OFFICIAL
    assert "structures" in empty.blocked
    no_cfg = select_sim_backend(
        "rust",
        [],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert no_cfg.backend == BACKEND_OFFICIAL
    assert "unknown_requirements" in no_cfg.blocked
    bad_schema = rust_blocked_capabilities(
        {**SUPPORTED_CAPS, "schema": "v1"},
        (),
    )
    assert "schema" in bad_schema


def test_curriculum_union_selects_rust_when_nations_are_supported():
    dec = select_training_backend(
        "auto",
        STAGES,
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert dec.backend == BACKEND_RUST
    assert dec.fallback is False
    assert "boats" in dec.required
    assert "nations_ai" in dec.required
    assert "nations_ai" not in dec.blocked
    duel_cfg = stage_env_config(STAGES[0])
    assert "boat" in duel_cfg["allowedActions"]


def test_no_mid_episode_switch_to_unsupported_config():
    rust = select_sim_backend(
        "auto",
        [WILDERNESS],
        capabilities=SUPPORTED_CAPS,
        rust_available=True,
    )
    assert rust.backend == BACKEND_RUST
    assert config_allowed_on_backend(rust, WILDERNESS) is True
    assert config_allowed_on_backend(rust, DUEL) is False
    with pytest.raises(RuntimeError, match="mid-episode"):
        refuse_mid_episode_switch(rust, DUEL)
    refuse_mid_episode_switch(rust, WILDERNESS)


def test_env_server_cmd_official_unchanged_and_rust_uses_sim():
    official = env_server_cmd(8765, 4)
    assert official[0] == "node"
    assert "--no-concurrent-recompilation" in official
    assert official[official.index("--port") + 1] == "8765"
    assert official[official.index("--workers") + 1] == "4"
    rust = env_server_cmd(8765, 4, backend="rust")
    assert rust == rust_sim_cmd(8765, 4)
    assert "--port" in rust
    assert "8765" in rust
    assert rust[rust.index("--workers") + 1] == "4"


def test_dry_run_reports_rust_when_nations_are_supported(capsys):
    from agent import train as train_mod

    train_mod.main(
        [
            "--dry-run",
            "--run-name",
            "be-dry",
            "--backend",
            "auto",
            "--device",
            "cpu",
        ]
    )
    out = capsys.readouterr().out
    assert "sim_backend=rust" in out
    assert "requested=auto" in out
    assert "fallback=false" in out
