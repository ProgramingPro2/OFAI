"""Narrow harness for the Rust simulator foundation.

Does not start training or touch the official env client. Checks that the
schema-v2 capability document and Cargo package exist, and that later-stage
goldens stay fail-closed in the declared capability set.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "sim"
FIXTURES = ROOT / "OpenFrontIO" / "ofai" / "tests" / "fixtures"

SUPPORTED_GOLDENS = (
    "spawn",
    "wilderness-combat",
    "terminal-no-spawn",
    "terminal-timeout",
    "boats",
    "structures",
)
UNSUPPORTED_GOLDENS = (
    "player-combat",
    "diplomacy",
    "nations",
)


def test_rust_package_layout():
    assert (SIM / "Cargo.toml").is_file()
    assert (SIM / "src" / "main.rs").is_file()
    assert (SIM / "src" / "lib.rs").is_file()
    assert (SIM / "tests" / "foundation.rs").is_file()


def test_capabilities_cli_declares_foundation_and_fallback():
    out = subprocess.check_output(
        ["cargo", "run", "--quiet", "--manifest-path", str(SIM / "Cargo.toml"), "--", "--capabilities"],
        cwd=SIM,
        text=True,
    )
    caps = json.loads(out)
    assert caps["backend"] == "rust"
    assert caps["schema"] == "v2"
    assert caps["fallback"] == "official"
    assert caps["supported"]["spawn"] is True
    assert caps["supported"]["wilderness_attack"] is True
    assert caps["supported"]["player_attack"] is True
    assert caps["supported"]["boats"] is True
    assert caps["unsupported"]["boats"] is False
    assert caps["supported"]["structures"] is True
    assert caps["supported"]["diplomacy"] is True
    assert caps["supported"]["nations_ai"] is True
    assert caps["supported"]["tribe_ai"] is True
    assert caps["unsupported"]["nations_ai"] is False
    assert caps["unsupported"]["tribe_ai"] is False
    assert caps["unsupported"]["nukes"] is False
    assert caps["unsupported"]["rails"] is False


def test_oracle_goldens_partition_by_capability():
    for name in SUPPORTED_GOLDENS + UNSUPPORTED_GOLDENS:
        fixture = json.loads((FIXTURES / f"golden-{name}.json").read_text())
        assert fixture["name"] == name
        assert fixture["init"]["digest"]
        bots = fixture["config"].get("bots", 0)
        nations = fixture["config"].get("nations")
        if name in UNSUPPORTED_GOLDENS:
            assert bots > 0 or nations != "disabled"
        else:
            assert nations == "disabled"
            assert bots == 0
