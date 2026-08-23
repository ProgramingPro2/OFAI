"""Baseline gates: profiler helpers, representative topologies, golden schema.

Does not start training or write repository checkpoints. Live env timing is
exercised by tests/profile_pipeline.py; this file checks the gate machinery.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.curriculum import STAGES
from agent.env_client import StepBatch, _merge_step_profiles
from tests.profile_pipeline import (
    QUICK_LAYOUTS,
    REPRESENTATIVE_LAYOUTS,
    REPRESENTATIVE_STAGES,
    accumulate_profile,
    empty_profile_acc,
    format_layout,
    latency_stats,
    mean_profile,
    parse_layouts,
    percentile,
    pick_fastest_stable,
    stage_cfg,
)

FIXTURE_DIR = ROOT / "OpenFrontIO" / "ofai" / "tests" / "fixtures"
REQUIRED_GOLDENS = (
    "spawn",
    "wilderness-combat",
    "player-combat",
    "boats",
    "structures",
    "diplomacy",
    "nations",
    "terminal-no-spawn",
    "terminal-timeout",
)


def test_representative_stages_exist_in_curriculum():
    names = {s.name for s in STAGES}
    for name in REPRESENTATIVE_STAGES:
        assert name in names
    duel, peninsula, nations = (stage_cfg(n) for n in REPRESENTATIVE_STAGES)
    assert duel["map"] == "Halkidiki"
    assert peninsula["map"] == "Halkidiki"
    assert nations["nations"] == 8
    assert peninsula["nations"] == 8


def test_representative_layouts_keep_128_envs():
    totals = {s * e for s, e in REPRESENTATIVE_LAYOUTS}
    assert totals == {128}
    assert [format_layout(s, e) for s, e in REPRESENTATIVE_LAYOUTS] == [
        "16x8",
        "32x4",
        "64x2",
    ]
    assert parse_layouts("16x8,32x4,64x2") == list(REPRESENTATIVE_LAYOUTS)
    quick_totals = {s * e for s, e in QUICK_LAYOUTS}
    assert len(quick_totals) == 1


def test_latency_percentiles_and_profile_merge():
    samples = [0.010, 0.020, 0.040, 0.080]
    stats = latency_stats(samples)
    assert stats["n"] == 4
    assert stats["p50_ms"] == pytest.approx(percentile(samples, 50) * 1000)
    assert stats["p95_ms"] >= stats["p50_ms"]
    assert stats["p99_ms"] >= stats["p95_ms"]
    acc = empty_profile_acc()
    accumulate_profile(acc, {"simMs": 10, "maskObsMs": 4, "serializeMs": 2, "resets": 1})
    accumulate_profile(acc, {"simMs": 6, "maskObsMs": 2, "serializeMs": 2, "resets": 1})
    means = mean_profile(acc, 2)
    assert means["sim_ms"] == pytest.approx(8.0)
    assert means["mask_obs_ms"] == pytest.approx(3.0)
    assert means["resets"] == pytest.approx(2.0)
    merged = _merge_step_profiles(
        [
            {"simMs": 1, "maskObsMs": 2, "serializeMs": 3, "resetMs": 4, "resets": 1},
            {"simMs": 10, "maskObsMs": 20, "serializeMs": 30, "resetMs": 40, "resets": 2},
            None,
        ]
    )
    assert merged == {
        "simMs": 11.0,
        "maskObsMs": 22.0,
        "serializeMs": 33.0,
        "resetMs": 44.0,
        "resets": 3.0,
    }
    assert _merge_step_profiles([None, None]) is None


def test_step_batch_keeps_optional_profile():
    batch = StepBatch(obs={}, rewards=None, dones=None, infos=[], profile={"simMs": 1})
    assert batch.profile["simMs"] == 1


def test_fastest_stable_ignores_failed_layouts():
    rows = [
        {
            "topology": "16x8",
            "env_steps_per_s": 20.0,
            "latency": {"p50_ms": 10.0},
        },
        {"topology": "32x4", "error": "OOM"},
        {
            "topology": "64x2",
            "env_steps_per_s": 35.0,
            "latency": {"p50_ms": 8.0},
        },
    ]
    best = pick_fastest_stable(rows)
    assert best is not None
    assert best["topology"] == "64x2"
    assert pick_fastest_stable([{"topology": "16x8", "error": "fail"}]) is None


def test_golden_fixtures_cover_required_decision_boundaries():
    missing = [name for name in REQUIRED_GOLDENS if not (FIXTURE_DIR / f"golden-{name}.json").exists()]
    if missing:
        pytest.skip(
            "OpenFrontIO goldens not generated yet: " + ", ".join(missing)
        )
    seen_terminals: set[str] = set()
    for name in REQUIRED_GOLDENS:
        fixture = json.loads((FIXTURE_DIR / f"golden-{name}.json").read_text())
        assert fixture["name"] == name
        assert fixture["actions"]
        assert fixture["boundaries"]
        assert len(fixture["actions"]) == len(fixture["boundaries"])
        assert fixture["init"]["digest"]
        assert fixture["init"]["tileDigest"]
        assert fixture["init"]["obsDigest"]
        for boundary in fixture["boundaries"]:
            for key in (
                "tick",
                "digest",
                "tileDigest",
                "obsDigest",
                "stepDigest",
                "reward",
                "intentCount",
                "actionAccepted",
                "terminalCause",
            ):
                assert key in boundary
            seen_terminals.add(str(boundary["terminalCause"]))
    assert "no_spawn" in seen_terminals
    assert "timeout" in seen_terminals
