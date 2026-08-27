"""Fail-closed simulator backend selection.

Rust may generate experience only when every episode config is schema-v2 and
every required capability is explicitly supported. Unsupported features
select the official TypeScript core *before* episode creation. Mid-episode
switching and approximation are forbidden.

Training defaults to auto: Rust when the capability document supports
every remaining stage (boats, structures, diplomacy, tribe/nation AI,
rails, nukes, doomsday). Unknown or unsupported capabilities select
the official TypeScript core before episode creation. agent.eval is
always official-core.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from . import spaces

BACKEND_OFFICIAL = "official"
BACKEND_RUST = "rust"
BACKEND_AUTO = "auto"

SCHEMA_V2 = "v2"

CAP_BOATS = "boats"
CAP_STRUCTURES = "structures"
CAP_DIPLOMACY = "diplomacy"
CAP_NATIONS_AI = "nations_ai"
CAP_TRIBE_AI = "tribe_ai"
CAP_NUKES = "nukes"
CAP_RAILS = "rails"
CAP_DOOMSDAY = "doomsday"

ACTION_CAPABILITIES = {
    "boat": CAP_BOATS,
    "build": CAP_STRUCTURES,
    "ally": CAP_DIPLOMACY,
    "break_ally": CAP_DIPLOMACY,
    "embargo": CAP_DIPLOMACY,
}

ALWAYS_UNSUPPORTED_IF_DECLARED = ()

ROOT = Path(__file__).resolve().parent.parent
SIM_DIR = ROOT / "sim"
_CAPS_CACHE: dict | None = None


@dataclass(frozen=True)
class BackendDecision:
    """Resolved simulator backend for a farm / init batch."""

    backend: str
    requested: str
    reason: str
    required: tuple[str, ...]
    blocked: tuple[str, ...]
    fallback: bool
    schema: str = SCHEMA_V2
    source: str = "capability"

    def log_line(self) -> str:
        required = ",".join(self.required) or "-"
        blocked = ",".join(self.blocked) or "-"
        return (
            f"[env] sim_backend={self.backend} requested={self.requested} "
            f"fallback={str(self.fallback).lower()} schema={self.schema} "
            f"required={required} blocked={blocked} {self.reason}"
        )

    def allows_rust(self) -> bool:
        return self.backend == BACKEND_RUST and not self.blocked


def parse_backend_arg(value: str) -> str:
    key = str(value).strip().lower()
    if key in (BACKEND_OFFICIAL, BACKEND_RUST, BACKEND_AUTO):
        return key
    raise ValueError(
        f"unknown backend {value!r}; expected official, rust, or auto"
    )


def _action_names(raw: object) -> set[str]:
    if raw is None:
        return set(spaces.ACTION_NAMES)
    names: set[str] = set()
    if not isinstance(raw, (list, tuple)):
        return set(spaces.ACTION_NAMES)
    for item in raw:
        if isinstance(item, str):
            names.add(item)
        elif isinstance(item, int):
            if 0 <= item < len(spaces.ACTION_NAMES):
                names.add(spaces.ACTION_NAMES[item])
            else:
                names.update(spaces.ACTION_NAMES)
        else:
            names.update(spaces.ACTION_NAMES)
    return names


def required_capabilities(config: dict) -> tuple[str, ...]:
    """Capabilities an episode needs before it may be created on Rust."""
    needed: list[str] = []
    names = _action_names(
        config.get("allowedActions", config.get("allowed_actions"))
    )
    for action, cap in ACTION_CAPABILITIES.items():
        if action in names and cap not in needed:
            needed.append(cap)
    nations = config.get("nations", "disabled")
    if nations not in (None, "disabled", "", 0, "0"):
        needed.append(CAP_NATIONS_AI)
    try:
        bots = int(config.get("bots") or 0)
    except (TypeError, ValueError):
        bots = 1
    if bots > 0:
        needed.append(CAP_TRIBE_AI)
    dd = config.get("doomsdayClock") or {}
    if isinstance(dd, dict) and dd.get("enabled"):
        needed.append(CAP_DOOMSDAY)
    return tuple(needed)


def stage_env_config(stage: Any) -> dict:
    return {
        "allowedActions": list(getattr(stage, "allowed_actions", spaces.FULL_ACTIONS)),
        "nations": getattr(stage, "nations", "disabled"),
        "bots": getattr(stage, "bots", 0),
        "map": getattr(stage, "map", ""),
    }


def default_rust_capabilities() -> dict:
    """Fail-closed stub used when the Rust CLI cannot be queried."""
    return {
        "backend": BACKEND_RUST,
        "schema": SCHEMA_V2,
        "fallback": BACKEND_OFFICIAL,
        "supported": {},
        "unsupported": {
            CAP_BOATS: True,
            CAP_NUKES: True,
            CAP_RAILS: True,
            CAP_DOOMSDAY: True,
        },
    }


def load_rust_capabilities(
    *,
    loader: Any = None,
    use_cache: bool = True,
) -> dict:
    """Read the Rust capability document. Fail closed on any load error."""
    global _CAPS_CACHE
    if loader is not None:
        try:
            caps = loader()
        except Exception:
            return default_rust_capabilities()
        return caps if isinstance(caps, dict) else default_rust_capabilities()
    if use_cache and _CAPS_CACHE is not None:
        return _CAPS_CACHE
    manifest = SIM_DIR / "Cargo.toml"
    if not manifest.is_file():
        return default_rust_capabilities()
    try:
        binary = rust_sim_binary()
        if binary is not None:
            cap_cmd = [str(binary), "--capabilities"]
        else:
            cap_cmd = [
                "cargo",
                "run",
                "--quiet",
                "--release",
                "--manifest-path",
                str(manifest),
                "--",
                "--capabilities",
            ]
        out = subprocess.check_output(
            cap_cmd,
            cwd=SIM_DIR,
            text=True,
            timeout=120,
        )
        caps = json.loads(out)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError, ValueError):
        return default_rust_capabilities()
    if not isinstance(caps, dict):
        return default_rust_capabilities()
    if use_cache:
        _CAPS_CACHE = caps
    return caps


def rust_sim_binary() -> Path | None:
    """Prefer the release binary for throughput; fall back to debug."""
    release = SIM_DIR / "target" / "release" / "ofai-sim"
    if release.is_file():
        return release
    debug = SIM_DIR / "target" / "debug" / "ofai-sim"
    if debug.is_file():
        return debug
    return None


def rust_runtime_available() -> bool:
    if rust_sim_binary() is not None:
        return True
    return (SIM_DIR / "Cargo.toml").is_file()


def rust_sim_cmd(port: int, workers: int = 0) -> list[str]:
    """Argv for the schema-v2 Rust env server.

    Always passes ``--workers`` so a single process can own K games with
    deterministic per-env threads. Prefers ``target/release/ofai-sim``.
    """
    binary = rust_sim_binary()
    if binary is not None:
        cmd = [str(binary), "--port", str(int(port))]
    else:
        cmd = [
            "cargo",
            "run",
            "--quiet",
            "--release",
            "--manifest-path",
            str(SIM_DIR / "Cargo.toml"),
            "--",
            "--port",
            str(int(port)),
        ]
    cmd.extend(["--workers", str(max(0, int(workers)))])
    return cmd


def rust_blocked_capabilities(
    caps: dict | None,
    required: Iterable[str],
) -> tuple[str, ...]:
    """Caps that block Rust. Missing/unknown support fails closed."""
    caps = caps if isinstance(caps, dict) else default_rust_capabilities()
    blocked: list[str] = []
    schema = caps.get("schema")
    if schema not in (SCHEMA_V2, "2", 2):
        blocked.append("schema")
    if caps.get("fallback") != BACKEND_OFFICIAL:
        blocked.append("fallback_contract")
    supported = caps.get("supported") if isinstance(caps.get("supported"), dict) else {}
    unsupported = (
        caps.get("unsupported") if isinstance(caps.get("unsupported"), dict) else {}
    )
    for cap in required:
        if unsupported.get(cap) is True:
            if cap not in blocked:
                blocked.append(cap)
            continue
        if supported.get(cap) is not True:
            if cap not in blocked:
                blocked.append(cap)
    for cap in ALWAYS_UNSUPPORTED_IF_DECLARED:
        if unsupported.get(cap) is True and cap in required and cap not in blocked:
            blocked.append(cap)
    return tuple(blocked)


def union_required_capabilities(configs: Iterable[dict]) -> tuple[str, ...]:
    seen: list[str] = []
    for cfg in configs:
        for cap in required_capabilities(cfg):
            if cap not in seen:
                seen.append(cap)
    return tuple(seen)


def select_sim_backend(
    requested: str,
    configs: Iterable[dict] | None = None,
    *,
    capabilities: dict | None = None,
    rust_available: bool | None = None,
) -> BackendDecision:
    """Choose official or rust for this init batch.

    Decision is locked before the first episode. A later nextConfig that
    needs a blocked capability must not be created on Rust.
    """
    try:
        req = parse_backend_arg(requested)
    except ValueError:
        req = BACKEND_OFFICIAL
    cfg_list = [dict(c) for c in (configs or [])]
    required = union_required_capabilities(cfg_list)

    if req == BACKEND_OFFICIAL:
        return BackendDecision(
            backend=BACKEND_OFFICIAL,
            requested=req,
            reason="production official-core",
            required=required,
            blocked=(),
            fallback=False,
            source="cli",
        )

    if not cfg_list:
        return BackendDecision(
            backend=BACKEND_OFFICIAL,
            requested=req,
            reason="no episode configs; fail closed to official",
            required=required,
            blocked=("unknown_requirements",),
            fallback=True,
            source="capability",
        )

    caps = (
        capabilities
        if capabilities is not None
        else load_rust_capabilities()
    )
    blocked = list(rust_blocked_capabilities(caps, required))
    available = rust_runtime_available() if rust_available is None else bool(rust_available)
    if not available:
        blocked.append("runtime")
    blocked_t = tuple(blocked)
    if blocked_t:
        return BackendDecision(
            backend=BACKEND_OFFICIAL,
            requested=req,
            reason="unsupported rust capability; official before episode create",
            required=required,
            blocked=blocked_t,
            fallback=True,
            schema=str(caps.get("schema") or SCHEMA_V2),
            source="capability",
        )
    return BackendDecision(
        backend=BACKEND_RUST,
        requested=req,
        reason="schema-v2 rust supports every required capability",
        required=required,
        blocked=(),
        fallback=False,
        schema=str(caps.get("schema") or SCHEMA_V2),
        source="capability",
    )


def select_training_backend(
    requested: str,
    stages: Iterable[Any],
    **kwargs: Any,
) -> BackendDecision:
    """Union remaining curriculum stages, then select fail-closed."""
    configs = [stage_env_config(stage) for stage in stages]
    return select_sim_backend(requested, configs, **kwargs)


def config_allowed_on_backend(decision: BackendDecision, config: dict) -> bool:
    """True iff this episode may be created on the locked backend."""
    if decision.backend != BACKEND_RUST:
        return True
    if decision.blocked:
        return False
    needed = required_capabilities(config)
    return all(cap in decision.required for cap in needed)


def refuse_mid_episode_switch(
    decision: BackendDecision,
    next_config: dict | None,
) -> None:
    """Raise if a not-yet-created episode would require leaving Rust."""
    if next_config is None or decision.backend != BACKEND_RUST:
        return
    if config_allowed_on_backend(decision, next_config):
        return
    raise RuntimeError(
        "rust backend cannot create this episode (fallback=official); "
        "refusing mid-episode switch"
    )
