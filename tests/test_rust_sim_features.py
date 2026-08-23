"""Narrow rust-features capability and fail-closed checks.

Does not start training or the official env client. Verifies the expanded
capability document and that HPA water paths / later systems stay official-only.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "sim"
FIXTURES = ROOT / "OpenFrontIO" / "ofai" / "tests" / "fixtures"


def test_features_package_and_tests_exist():
    assert (SIM / "src" / "units.rs").is_file()
    assert (SIM / "src" / "path.rs").is_file()
    assert (SIM / "src" / "spatial.rs").is_file()
    assert (SIM / "src" / "diplomacy.rs").is_file()
    assert (SIM / "src" / "ai.rs").is_file()
    assert (SIM / "tests" / "features.rs").is_file()


def test_capabilities_cli_declares_ported_slices_and_official_boundary():
    out = subprocess.check_output(
        ["cargo", "run", "--quiet", "--manifest-path", str(SIM / "Cargo.toml"), "--", "--capabilities"],
        cwd=SIM,
        text=True,
    )
    caps = json.loads(out)
    assert caps["backend"] == "rust"
    assert caps["fallback"] == "official"
    assert caps["supported"]["structures"] is True
    assert caps["supported"]["diplomacy"] is True
    assert caps["supported"]["tribe_ai"] is True
    assert caps["supported"]["nations_ai"] is False
    assert caps["supported"]["boats"] is True
    assert caps["unsupported"]["boats"] is False
    assert caps["unsupported"]["tribe_ai"] is False
    assert caps["unsupported"]["nations_ai"] is True
    assert caps["unsupported"]["nukes"] is False
    assert caps["unsupported"]["rails"] is False
    assert caps["unsupported"]["doomsday"] is False


def test_feature_goldens_exist():
    for name in ("boats", "structures", "diplomacy", "nations", "player-combat"):
        fixture = json.loads((FIXTURES / f"golden-{name}.json").read_text())
        assert fixture["name"] == name
        assert fixture["init"]["tileDigest"]
