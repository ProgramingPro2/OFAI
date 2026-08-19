"""Watch a trained checkpoint play real OpenFront matches, rendered from the
actual game map (full-resolution territory per player) — not the model's
downsampled input planes.

Legend:
  dark blue   ocean
  gray        impassable terrain (mountains etc.)
  tan         unowned land
  bright cols players (a distinct color each; the agent is highlighted)
  magenta     radioactive fallout
  dots        units (structures, warships, transports)

Examples:
  # fullscreen pygame UI (needs a display; falls back to a message if none).
  # Sampled policy; --fps is displayed frames, --frame-skip skips full-res
  # render work between decisions. Playback speed is separate ([ ] in-window).
  .venv/bin/python -m agent.watch --ckpt checkpoints/run3/latest.pt \
      --preset tribes20 --scale 1 --fps 60 --frame-skip 5

  # write a video instead (works headless). Default is the sampled policy;
  # --greedy is a deterministic diagnostic, not the deployed policy.
  .venv/bin/python -m agent.watch --ckpt checkpoints/run3/latest.pt \
      --save match.mp4 --scale 1 --fps 60 --frame-skip 5 --episodes 2

  # specific opponents
  .venv/bin/python -m agent.watch --ckpt checkpoints/run3/latest.pt \
      --map FourIslands --nations 8 --difficulty Easy --bots 20
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from . import spaces
from .curriculum import STAGES
from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors

LEGEND_W = 230  # legend sidebar width in saved-video / native scene px
STATUS_H = 30  # status bar height in saved-video / native scene px
# Desktop APIs usually report the full monitor, not the space left after
# panels, docks, and window decorations. Leave conservative room so an
# automatically sized windowed viewer does not extend below the usable desktop.
DESKTOP_RESERVE = (32, 96)
HUD_TOP_H = 40
HUD_BOTTOM_H = 56
LEGEND_SCREEN_W = 260
DIFFICULTIES = ("Easy", "Medium", "Hard", "Impossible")
MAP_SIZES = ("Compact", "Normal")
OPPONENT_TYPES = ("tribes", "nations", "mixed")
PRESETS = {stage.name: stage for stage in STAGES}
SPEED_STEPS = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
POST_MATCH_ACTIONS = ("replay", "restart", "new", "configure", "quit")
REPLAY_MAX_BYTES = 96 * 1024 * 1024
REPLAY_MAX_FRAMES = 360
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
_MAP_ENUM_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9]*)\s*=\s*\"", re.M)
DEFAULT_CONFIG = {
    "map": "FourIslands",
    "map_size": "Normal",
    "nations": "disabled",
    "difficulty": "Easy",
    "bots": 20,
    "max_ticks": 12000,
}


def positive_int(value: str) -> int:
    """argparse type: integer >= 1."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


def non_negative_int(value: str) -> int:
    """argparse type: integer >= 0."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {parsed}")
    return parsed


def positive_float(value: str) -> float:
    """argparse type: float > 0."""
    try:
        parsed = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid float value: {value!r}") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0, got {parsed}")
    return parsed


def parse_window_size(value: str) -> tuple[int, int]:
    """Parse WIDTHxHEIGHT for an explicit interactive window size."""
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("must be WIDTHxHEIGHT, for example 1280x720")
    try:
        width, height = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "must be WIDTHxHEIGHT, for example 1280x720"
        ) from exc
    if width < 1 or height < 1:
        raise argparse.ArgumentTypeError("window width and height must be >= 1")
    return width, height


def scene_size(map_width: int, map_height: int, scale: int) -> tuple[int, int]:
    """Full rendered size, including the legend and status areas."""
    return map_width * scale + LEGEND_W, map_height * scale + STATUS_H


def fit_size(
    source_size: tuple[int, int],
    bounds: tuple[int, int],
    *,
    allow_upscale: bool = False,
) -> tuple[int, int]:
    """Fit source within bounds without changing its aspect ratio."""
    source_w, source_h = source_size
    bound_w, bound_h = bounds
    if min(source_w, source_h, bound_w, bound_h) < 1:
        raise ValueError("source and bound dimensions must be positive")
    ratio = min(bound_w / source_w, bound_h / source_h)
    if not allow_upscale:
        ratio = min(1.0, ratio)
    return (
        max(1, min(bound_w, math.floor(source_w * ratio))),
        max(1, min(bound_h, math.floor(source_h * ratio))),
    )


def viewport_rect(
    source_size: tuple[int, int], window_size: tuple[int, int]
) -> tuple[int, int, int, int]:
    """Centered aspect-preserving destination rectangle for a scene."""
    width, height = fit_size(source_size, window_size, allow_upscale=True)
    return (
        (window_size[0] - width) // 2,
        (window_size[1] - height) // 2,
        width,
        height,
    )


def window_size_for_scene(
    source_size: tuple[int, int],
    usable_size: tuple[int, int],
    requested_size: tuple[int, int] | None,
) -> tuple[int, int]:
    """Choose an auto-fitted or explicitly requested, monitor-safe window."""
    if requested_size is None:
        return fit_size(source_size, usable_size)
    return (
        min(requested_size[0], usable_size[0]),
        min(requested_size[1], usable_size[1]),
    )


def desktop_size(pygame, display: int) -> tuple[int, int]:
    """Return the selected monitor's full pixel size."""
    desktop_sizes = pygame.display.get_desktop_sizes()
    if not desktop_sizes:
        info = pygame.display.Info()
        desktop_sizes = [(int(info.current_w), int(info.current_h))]
    if not 0 <= display < len(desktop_sizes):
        raise ValueError(
            f"display {display} is unavailable; found {len(desktop_sizes)} display(s)"
        )
    width, height = desktop_sizes[display]
    if width < 1 or height < 1:
        raise RuntimeError(f"display {display} reported invalid size {width}x{height}")
    return int(width), int(height)


def usable_display_size(pygame, display: int) -> tuple[int, int]:
    """Return conservative usable dimensions for a windowed display."""
    width, height = desktop_size(pygame, display)
    reserve_w, reserve_h = DESKTOP_RESERVE
    return max(1, width - reserve_w), max(1, height - reserve_h)


def overlay_layout(window_size: tuple[int, int]) -> dict[str, tuple[int, int, int, int]]:
    """Fixed screen-space HUD regions. Independent of map zoom."""
    width, height = window_size
    if min(width, height) < 1:
        raise ValueError("window dimensions must be positive")
    legend_w = min(LEGEND_SCREEN_W, max(0, width // 4))
    top_h = min(HUD_TOP_H, max(1, height // 12))
    bottom_h = min(HUD_BOTTOM_H, max(1, height // 10))
    map_w = max(1, width - legend_w)
    map_h = max(1, height - top_h - bottom_h)
    return {
        "top": (0, 0, width, top_h),
        "bottom": (0, top_h + map_h, width, bottom_h),
        "legend": (map_w, top_h, legend_w, map_h),
        "map": (0, top_h, map_w, map_h),
    }


def clamp(value: float, lo: float, hi: float) -> float:
    if lo > hi:
        return lo
    return min(hi, max(lo, value))


@dataclass(frozen=True)
class ViewTransform:
    """Aspect-preserving map crop and destination inside a viewport."""

    scale: float
    src_x: float
    src_y: float
    src_w: float
    src_h: float
    dest_x: int
    dest_y: int
    dest_w: int
    dest_h: int

    def screen_to_map(self, sx: float, sy: float) -> tuple[float, float]:
        if self.scale <= 0:
            return self.src_x, self.src_y
        return (
            self.src_x + (sx - self.dest_x) / self.scale,
            self.src_y + (sy - self.dest_y) / self.scale,
        )

    def map_to_screen(self, mx: float, my: float) -> tuple[float, float]:
        return (
            self.dest_x + (mx - self.src_x) * self.scale,
            self.dest_y + (my - self.src_y) * self.scale,
        )

    def source_rect(self) -> tuple[int, int, int, int]:
        return (
            int(self.src_x),
            int(self.src_y),
            max(1, int(round(self.src_w))),
            max(1, int(round(self.src_h))),
        )

    def dest_rect(self) -> tuple[int, int, int, int]:
        return (self.dest_x, self.dest_y, self.dest_w, self.dest_h)


def compute_view_transform(
    map_size: tuple[int, int],
    viewport_size: tuple[int, int],
    zoom: float,
    center: tuple[float, float],
) -> ViewTransform:
    """Fit or crop the map into a viewport without stretching."""
    map_w, map_h = map_size
    view_w, view_h = viewport_size
    if min(map_w, map_h, view_w, view_h) < 1:
        raise ValueError("map and viewport dimensions must be positive")
    zoom = max(1.0, float(zoom))
    fit = min(view_w / map_w, view_h / map_h)
    scale = fit * zoom
    src_w = min(float(map_w), view_w / scale)
    src_h = min(float(map_h), view_h / scale)
    src_x = clamp(center[0] - src_w / 2.0, 0.0, map_w - src_w)
    src_y = clamp(center[1] - src_h / 2.0, 0.0, map_h - src_h)
    dest_w = max(1, min(view_w, int(round(src_w * scale))))
    dest_h = max(1, min(view_h, int(round(src_h * scale))))
    dest_x = (view_w - dest_w) // 2
    dest_y = (view_h - dest_h) // 2
    return ViewTransform(
        scale, src_x, src_y, src_w, src_h, dest_x, dest_y, dest_w, dest_h
    )


class Camera:
    """Interactive map camera. Overlays stay in screen space; only the map zooms."""

    def __init__(self, zoom: float = 1.0, max_zoom: float = 16.0):
        self.zoom = max(1.0, float(zoom))
        self.center_x = 0.0
        self.center_y = 0.0
        self.min_zoom = 1.0
        self.max_zoom = max(self.min_zoom, float(max_zoom))

    def reset(self, map_width: int, map_height: int) -> None:
        self.zoom = 1.0
        self.center_x = map_width / 2.0
        self.center_y = map_height / 2.0

    def transform(
        self, map_size: tuple[int, int], viewport_size: tuple[int, int]
    ) -> ViewTransform:
        return compute_view_transform(
            map_size, viewport_size, self.zoom, (self.center_x, self.center_y)
        )

    def set_zoom(
        self,
        zoom: float,
        map_size: tuple[int, int],
        viewport_size: tuple[int, int],
        anchor_map: tuple[float, float] | None = None,
    ) -> None:
        old = self.transform(map_size, viewport_size)
        self.zoom = clamp(zoom, self.min_zoom, self.max_zoom)
        if anchor_map is not None:
            screen = old.map_to_screen(*anchor_map)
            shifted = compute_view_transform(
                map_size, viewport_size, self.zoom, (self.center_x, self.center_y)
            )
            new_map = shifted.screen_to_map(*screen)
            self.center_x += anchor_map[0] - new_map[0]
            self.center_y += anchor_map[1] - new_map[1]
        self._reclamp(map_size, viewport_size)

    def zoom_by(
        self,
        factor: float,
        map_size: tuple[int, int],
        viewport_size: tuple[int, int],
        anchor_map: tuple[float, float] | None = None,
    ) -> None:
        self.set_zoom(self.zoom * factor, map_size, viewport_size, anchor_map)

    def pan(
        self,
        dx: float,
        dy: float,
        map_size: tuple[int, int],
        viewport_size: tuple[int, int],
    ) -> None:
        self.center_x += dx
        self.center_y += dy
        self._reclamp(map_size, viewport_size)

    def _reclamp(
        self, map_size: tuple[int, int], viewport_size: tuple[int, int]
    ) -> None:
        view = compute_view_transform(
            map_size, viewport_size, self.zoom, (self.center_x, self.center_y)
        )
        self.center_x = view.src_x + view.src_w / 2.0
        self.center_y = view.src_y + view.src_h / 2.0


def snap_speed(value: float) -> float:
    """Snap a positive multiplier onto the discrete playback ladder."""
    if value <= 0:
        raise ValueError("speed must be positive")
    return min(SPEED_STEPS, key=lambda step: abs(step - value))


def nearest_speed_index(value: float) -> int:
    return SPEED_STEPS.index(snap_speed(value))


class PlaybackControl:
    """Pause / step / speed. Changes pacing only; never the policy or env."""

    def __init__(self, speed: float = 1.0, paused: bool = False):
        self.speed_index = nearest_speed_index(speed)
        self.paused = bool(paused)
        self._step_once = False

    @property
    def speed(self) -> float:
        return SPEED_STEPS[self.speed_index]

    def slower(self) -> None:
        self.speed_index = max(0, self.speed_index - 1)

    def faster(self) -> None:
        self.speed_index = min(len(SPEED_STEPS) - 1, self.speed_index + 1)

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self._step_once = False

    def request_step(self) -> None:
        if self.paused:
            self._step_once = True

    def should_advance(self) -> bool:
        if not self.paused:
            return True
        if self._step_once:
            self._step_once = False
            return True
        return False

    def render_interval_ms(self, fps: float) -> int:
        rate = max(1.0, float(fps)) * self.speed
        return max(1, int(round(1000.0 / rate)))


def frame_snapshot_bytes(frame: dict) -> int:
    cells = frame["cells"]
    units = frame.get("units") or []
    players = frame.get("players") or []
    return int(getattr(cells, "nbytes", 0)) + 256 + 64 * len(units) + 96 * len(players)


def compact_frame(frame: dict, status: str) -> dict:
    """Copy the watch frame payload without sharing the live env arrays."""
    return {
        "width": int(frame["width"]),
        "height": int(frame["height"]),
        "cells": np.array(frame["cells"], dtype=np.uint8, copy=True),
        "players": [dict(player) for player in frame["players"]],
        "units": [dict(unit) for unit in frame["units"]],
        "tick": int(frame["tick"]),
        "status": status,
    }


class ReplayTape:
    """Bounded recording: every action, plus frames until a memory cap."""

    def __init__(
        self,
        max_bytes: int = REPLAY_MAX_BYTES,
        max_frames: int = REPLAY_MAX_FRAMES,
    ):
        self.max_bytes = int(max_bytes)
        self.max_frames = int(max_frames)
        self.actions: list[np.ndarray] = []
        self.frames: list[dict] | None = []
        self.bytes_used = 0
        self.seed = ""
        self.env_config: dict = {}
        self.result = ""
        self.result_info: dict = {}

    @property
    def has_frames(self) -> bool:
        return bool(self.frames)

    @property
    def replay_mode(self) -> str:
        return "frames" if self.has_frames else "actions"

    def record_action(self, action: np.ndarray) -> None:
        self.actions.append(np.array(action, dtype=np.int32, copy=True))

    def record_frame(self, frame: dict, status: str) -> None:
        if self.frames is None:
            return
        snapshot = compact_frame(frame, status)
        size = frame_snapshot_bytes(snapshot)
        if (
            self.bytes_used + size > self.max_bytes
            or len(self.frames) >= self.max_frames
        ):
            self.frames = None
            self.bytes_used = 0
            return
        self.frames.append(snapshot)
        self.bytes_used += size


def infer_game_type(nations: int | str, bots: int) -> str:
    if nations == "disabled":
        return "tribes"
    if int(bots) == 0:
        return "nations"
    return "mixed"


def _byte_rng(value) -> torch.Tensor:
    """torch RNG APIs require a CPU uint8 tensor."""
    if not isinstance(value, torch.Tensor):
        value = torch.as_tensor(value, dtype=torch.uint8)
    return value.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def seed_policy_rng(seed: int) -> None:
    """Seed CPU and CUDA generators used by sampled policy.act()."""
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def snapshot_policy_rng() -> dict:
    """Capture policy sampling RNG, including CUDA when present."""
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_policy_rng(state: dict) -> None:
    torch.set_rng_state(_byte_rng(state["torch"]))
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_rng(item) for item in cuda_state])


def parse_game_map_keys(source: str) -> list[str]:
    """Extract GameMapType enum keys from OpenFront TypeScript."""
    start = source.find("enum GameMapType")
    if start < 0:
        return []
    end = source.find("}", start)
    if end < 0:
        return []
    return _MAP_ENUM_RE.findall(source[start:end])


def discover_maps(openfront_root: Path | None = None) -> list[str]:
    """Curriculum maps plus locally installed OpenFront GameMapType keys."""
    maps = {stage.map for stage in STAGES}
    root = Path(openfront_root) if openfront_root is not None else REPO_ROOT / "OpenFrontIO"
    for rel in ("src/core/game/Maps.gen.ts", "src/core/game/Game.ts"):
        path = root / rel
        if path.is_file():
            maps.update(parse_game_map_keys(path.read_text(encoding="utf-8")))
            break
    return sorted(maps)


def discover_checkpoints(root: Path | None = None) -> list[str]:
    base = Path(root) if root is not None else CHECKPOINT_ROOT
    if not base.is_dir():
        return []
    found: list[str] = []
    for path in sorted(base.rglob("*.pt")):
        try:
            found.append(str(path.relative_to(REPO_ROOT)))
        except ValueError:
            found.append(str(path))
    return found


def parse_nations_value(value: str) -> int | str:
    text = str(value).strip()
    if text in ("default", "disabled"):
        return text
    try:
        count = int(text)
    except ValueError as exc:
        raise ValueError("nations must be a count, default, or disabled") from exc
    if not 0 <= count <= 400:
        raise ValueError(f"nations must be between 0 and 400, got {count}")
    return count


@dataclass
class WatchSettings:
    """Interactive + CLI watch configuration (single-player FFA only)."""

    ckpt: str
    preset: str = "custom"
    game_type: str = "tribes"
    map: str = "FourIslands"
    map_size: str = "Normal"
    nations: int | str = "disabled"
    difficulty: str = "Easy"
    bots: int = 20
    max_ticks: int = 12000
    seed: str = "watch"
    policy_seed: int = 0
    episodes: int = 1
    device: str = "cuda"
    display: int = 0
    window_size: tuple[int, int] | None = None
    windowed: bool = False
    fps: float = 15.0
    scale: int = 3
    greedy: bool = False
    frame_skip: int = 1
    speed: float = 1.0
    port: int = 8790

    @classmethod
    def from_args(cls, args: argparse.Namespace) -> WatchSettings:
        return cls(
            ckpt=args.ckpt,
            preset=args.preset,
            game_type=infer_game_type(args.nations, args.bots),
            map=args.map,
            map_size=args.map_size,
            nations=args.nations,
            difficulty=args.difficulty,
            bots=int(args.bots),
            max_ticks=int(args.max_ticks),
            seed=args.seed,
            policy_seed=int(args.policy_seed),
            episodes=int(args.episodes),
            device=args.device,
            display=int(args.display),
            window_size=args.window_size,
            windowed=bool(getattr(args, "windowed", False) or args.window_size),
            fps=float(args.fps),
            scale=int(args.scale),
            greedy=bool(args.greedy),
            frame_skip=int(args.frame_skip),
            speed=snap_speed(float(getattr(args, "speed", 1.0))),
            port=int(args.port),
        )

    def apply_preset(self, name: str) -> None:
        self.preset = name
        if name == "custom":
            return
        stage = PRESETS[name]
        self.map = stage.map
        self.map_size = stage.map_size
        self.nations = stage.nations
        self.difficulty = stage.difficulty
        self.bots = int(stage.bots)
        self.max_ticks = int(stage.max_ticks)
        self.game_type = infer_game_type(self.nations, self.bots)

    def apply_game_type(self, opponents: str) -> None:
        if opponents not in OPPONENT_TYPES:
            raise ValueError(f"unsupported game type: {opponents}")
        self.game_type = opponents
        self.preset = "custom"
        if opponents == "tribes":
            self.nations = "disabled"
        elif opponents == "nations":
            self.nations = "default"
            self.bots = 0
        else:
            if self.nations == "disabled":
                self.nations = "default"
            if int(self.bots) == 0:
                self.bots = int(DEFAULT_CONFIG["bots"])

    def env_config(self, episode: int) -> dict:
        return {
            "map": self.map,
            "mapSize": self.map_size,
            "nations": self.nations,
            "difficulty": self.difficulty,
            "bots": int(self.bots),
            "seed": f"{self.seed}-{episode}",
            "maxTicks": int(self.max_ticks),
            "decisionInterval": 10,
            "shaping": 0,
        }

    def resolved_ckpt(self) -> Path:
        path = Path(self.ckpt)
        if path.is_file():
            return path
        alt = REPO_ROOT / self.ckpt
        if alt.is_file():
            return alt
        return path

    def validate(self) -> str | None:
        if not self.ckpt:
            return "Checkpoint path is required."
        if not self.resolved_ckpt().is_file():
            return f"Checkpoint not found: {self.ckpt}"
        if not self.map:
            return "Map must not be empty."
        if not self.seed:
            return "Seed must not be empty."
        if float(self.speed) <= 0:
            return "Speed must be positive."
        if self.preset not in ("custom", *PRESETS):
            return f"Unknown preset: {self.preset}"
        if self.game_type not in OPPONENT_TYPES:
            return f"Unsupported game type: {self.game_type}"
        if self.map_size not in MAP_SIZES:
            return f"Unsupported map size: {self.map_size}"
        if self.difficulty not in DIFFICULTIES:
            return f"Unsupported difficulty: {self.difficulty}"
        if not 0 <= int(self.bots) <= 400:
            return "Bots must be between 0 and 400."
        if isinstance(self.nations, str) and self.nations not in ("default", "disabled"):
            return "Nations must be a count, default, or disabled."
        if isinstance(self.nations, int) and not 0 <= self.nations <= 400:
            return "Nations must be between 0 and 400."
        if int(self.max_ticks) < 1:
            return "Max ticks must be >= 1."
        if int(self.episodes) < 1:
            return "Episodes must be >= 1."
        if int(self.scale) < 1:
            return "Scale must be >= 1."
        if int(self.frame_skip) < 1:
            return "Frame skip must be >= 1."
        if float(self.fps) <= 0:
            return "FPS must be positive."
        if int(self.display) < 0:
            return "Display index must be >= 0."
        if int(self.port) < 1:
            return "Port must be >= 1."
        return None


def episode_result(info: dict) -> str:
    if info.get("win"):
        return "WIN"
    if info.get("stageSuccess") or info.get("terminalCause") == "curriculum_success":
        return "STAGE"
    if info.get("dead"):
        return "DEAD"
    return str(info.get("terminalCause") or "timeout")


def load_policy(ckpt_path: str | Path, device: torch.device) -> tuple[Policy, dict]:
    policy = Policy().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    return policy, ckpt


class PaletteCache:
    def __init__(self) -> None:
        self.n = -1
        self.palette: np.ndarray | None = None
        self.bright: np.ndarray | None = None

    def for_players(self, n_players: int) -> tuple[np.ndarray, np.ndarray]:
        if n_players != self.n or self.palette is None or self.bright is None:
            self.n = n_players
            self.palette = build_palette(n_players)
            self.bright = np.clip(self.palette.astype(np.int16) + 60, 0, 255).astype(
                np.uint8
            )
            self.bright[255] = (255, 255, 255)
        return self.palette, self.bright


def episode_video_path(path: str, ep: int) -> str:
    """ep 0 keeps the given name; later episodes insert -epN before .mp4."""
    return path.replace(".mp4", f"-ep{ep}.mp4") if ep > 0 else path


def should_render_frame(step: int, done: bool, frame_skip: int) -> bool:
    """Render the first useful frame, every Nth decision, and the terminal frame."""
    if step < 1:
        return False
    if done:
        return True
    return (step - 1) % frame_skip == 0


def open_video_writer(path: str, fps: float):
    """Streaming ffmpeg writer. Caller must close (including on interrupt)."""
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed; pip install 'imageio[ffmpeg]'")
        return None
    return imageio.get_writer(path, fps=max(1, int(fps)), codec="libx264")


def close_video_writer(writer) -> None:
    if writer is None:
        return
    try:
        writer.close()
    except Exception:
        pass


# Base palette for the color-indexed map. Index 0,1,2 are terrain; 250 is
# fallout; players start at 3. Colors chosen to read like the OpenFront map.
def _player_color(i: int) -> tuple[int, int, int]:
    # Golden-angle hue walk gives visually distinct colors per player.
    h = (i * 137.508) % 360.0
    s, v = 0.72, 0.95
    import colorsys

    r, g, b = colorsys.hsv_to_rgb(h / 360.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def build_palette(n_players: int) -> np.ndarray:
    # Keep the 256 color-index entries used by cells, and extend the same
    # palette for legends/units when a valid high-bot-count config exceeds it.
    pal = np.zeros((max(256, 3 + n_players), 3), np.uint8)
    pal[0] = (18, 38, 63)      # ocean
    pal[1] = (70, 70, 75)      # impassable
    pal[2] = (176, 164, 132)   # unowned land (tan)
    for i in range(n_players):
        pal[3 + i] = _player_color(i)
    pal[250] = (200, 40, 200)  # fallout magenta
    return pal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Watch a checkpoint in the supported single-player FFA environment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=positive_int, default=1)
    p.add_argument("--port", type=positive_int, default=8790)
    p.add_argument(
        "--preset",
        choices=("custom", *PRESETS),
        default="custom",
        help="curriculum match preset; explicit match flags override it",
    )
    p.add_argument(
        "--opponents",
        "--game-type",
        choices=OPPONENT_TYPES,
        default=None,
        help="tribes only, nations only, or both; --nations/--bots can refine it",
    )
    p.add_argument(
        "--map",
        type=str,
        default=None,
        help="OpenFront GameMapType enum key (for example Halkidiki, FourIslands, World)",
    )
    p.add_argument("--map-size", choices=MAP_SIZES, default=None)
    p.add_argument(
        "--nations",
        type=str,
        default=None,
        metavar="COUNT|default|disabled",
        help="exact nation count, all map nations, or no nations",
    )
    p.add_argument("--difficulty", choices=DIFFICULTIES, default=None)
    p.add_argument("--bots", type=int, default=None, help="tribe filler bots (0-400)")
    p.add_argument("--max-ticks", type=positive_int, default=None)
    p.add_argument(
        "--seed",
        type=str,
        default="watch",
        help="deterministic game seed prefix; episode number is appended",
    )
    p.add_argument(
        "--policy-seed",
        type=int,
        default=0,
        help="PyTorch sampling seed (ignored by --greedy)",
    )
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--display",
        type=non_negative_int,
        default=0,
        help="monitor index for the interactive window",
    )
    p.add_argument(
        "--window-size",
        type=parse_window_size,
        default=None,
        metavar="WIDTHxHEIGHT",
        help=(
            "start windowed at WIDTHxHEIGHT instead of true fullscreen; "
            "oversized requests are clamped to the usable display"
        ),
    )
    p.add_argument(
        "--windowed",
        action="store_true",
        help="start the interactive viewer windowed (default is true fullscreen)",
    )
    p.add_argument(
        "--fps",
        type=positive_float,
        default=15.0,
        help="displayed/written frames per second (not simulation decisions)",
    )
    p.add_argument(
        "--speed",
        type=positive_float,
        default=1.0,
        help="interactive playback speed multiplier (snapped to 0.25/0.5/1/2/4/8)",
    )
    p.add_argument("--scale", type=positive_int, default=3, help="pixels per map tile")
    p.add_argument("--save", type=str, default="", help="write mp4 instead of a window")
    p.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    p.add_argument(
        "--frame-skip",
        type=positive_int,
        default=1,
        metavar="N",
        help="render every N decisions (always includes first and final frames)",
    )
    args = p.parse_args(argv)
    return resolve_match_args(p, args)


def resolve_match_args(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> argparse.Namespace:
    """Resolve a preset, opponent shorthand, and explicit overrides."""
    config = dict(DEFAULT_CONFIG)
    if args.preset != "custom":
        stage = PRESETS[args.preset]
        config.update(
            map=stage.map,
            map_size=stage.map_size,
            nations=stage.nations,
            difficulty=stage.difficulty,
            bots=stage.bots,
            max_ticks=stage.max_ticks,
        )

    if args.opponents == "tribes":
        config["nations"] = "disabled"
    elif args.opponents == "nations":
        config.update(nations="default", bots=0)
    elif args.opponents == "mixed":
        config["nations"] = "default"

    for name in ("map", "map_size", "difficulty", "bots", "max_ticks"):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.nations is not None:
        config["nations"] = parse_nations(parser, args.nations)
    if not 0 <= int(config["bots"]) <= 400:
        parser.error(f"--bots must be between 0 and 400, got {config['bots']}")
    if not str(config["map"]).strip():
        parser.error("--map must not be empty")
    if not args.seed:
        parser.error("--seed must not be empty")

    for name, value in config.items():
        setattr(args, name, value)
    return args


def parse_nations(
    parser: argparse.ArgumentParser, value: str
) -> int | str:
    """Parse the environment's exact supported nations values."""
    if value in ("default", "disabled"):
        return value
    try:
        count = int(value)
    except ValueError:
        parser.error("--nations must be a non-negative integer, default, or disabled")
    if not 0 <= count <= 400:
        parser.error(f"--nations must be between 0 and 400, got {count}")
    return count


def action_str(a: np.ndarray) -> str:
    at = int(a[0])
    name = spaces.ACTION_NAMES[at]
    frac = [5, 15, 30, 50, 80]
    if at == spaces.ACTION_ATTACK:
        return f"ATTACK p{a[1]} {frac[int(a[3])]}%"
    if at == spaces.ACTION_BUILD:
        return f"BUILD {spaces.UNIT_NAMES[int(a[4])]}"
    if at == spaces.ACTION_BOAT:
        return f"BOAT {frac[int(a[3])]}%"
    if at == spaces.ACTION_SPAWN:
        return "SPAWN"
    if at == spaces.ACTION_ALLY:
        return f"ALLY p{a[1]}"
    if at == spaces.ACTION_BREAK_ALLY:
        return f"BREAK-ALLY p{a[1]}"
    if at == spaces.ACTION_EMBARGO:
        return f"EMBARGO p{a[1]}"
    return name.upper()


def compose_rgb(frame: dict, palette: np.ndarray) -> np.ndarray:
    """Color-index map -> RGB image [H,W,3]."""
    return palette[frame["cells"]]


def overlay_units(rgb: np.ndarray, frame: dict, bright_palette: np.ndarray) -> None:
    """Mark units as a bright pixel cluster in the owner's color."""
    h, w = frame["cells"].shape
    for u in frame["units"]:
        x, y, owner = u["x"], u["y"], u["owner"]
        if 0 <= x < w and 0 <= y < h:
            c = bright_palette[3 + owner] if owner >= 0 else bright_palette[255]
            rgb[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = c


def player_centroids(frame: dict) -> dict[int, tuple[float, float]]:
    """Centroid (x,y) of each player's territory in tile coords."""
    cells = frame["cells"]
    owned = (cells >= 3) & (cells < 250)
    ys, xs = np.nonzero(owned)
    if xs.size == 0:
        return {}
    owners = cells[ys, xs].astype(np.int32) - 3
    size = int(owners.max()) + 1
    counts = np.bincount(owners, minlength=size)
    x_sums = np.bincount(owners, weights=xs, minlength=size)
    y_sums = np.bincount(owners, weights=ys, minlength=size)
    return {
        i: (float(x_sums[i] / counts[i]), float(y_sums[i] / counts[i]))
        for i in np.flatnonzero(counts)
    }


def _load_fonts(pygame, scale: int):
    """Create fonts once; SysFont is expensive inside the render loop."""
    return {
        "name": pygame.font.SysFont("monospace", max(11, scale * 3), bold=True),
        "legend": pygame.font.SysFont("monospace", 13),
        "status": pygame.font.SysFont("monospace", 14),
    }


def _draw_map_surface(
    pygame,
    frame: dict,
    palette: np.ndarray,
    bright_palette: np.ndarray,
    scale: int,
    fonts: dict,
    dest=None,
):
    """Map tiles, units, and on-map name labels only (no HUD)."""
    rgb = compose_rgb(frame, palette)
    overlay_units(rgb, frame, bright_palette)
    centroids = player_centroids(frame)
    mw, mh = frame["width"] * scale, frame["height"] * scale
    if dest is None or dest.get_size() != (mw, mh):
        dest = pygame.Surface((mw, mh))
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    if scale != 1:
        surf = pygame.transform.scale(surf, (mw, mh))
    dest.blit(surf, (0, 0))

    name_font = fonts["name"]
    for p in frame["players"]:
        if p["idx"] not in centroids or not p["alive"]:
            continue
        cx, cy = centroids[p["idx"]]
        label = ("* " if p["isAgent"] else "") + p["name"]
        text = name_font.render(label, True, (255, 255, 255))
        shadow = name_font.render(label, True, (0, 0, 0))
        tx, ty = int(cx * scale), int(cy * scale)
        rect = text.get_rect(center=(tx, ty))
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            dest.blit(shadow, rect.move(dx, dy))
        dest.blit(text, rect)
    return dest


def _draw_legend(
    pygame,
    surface,
    origin: tuple[int, int],
    height: int,
    frame: dict,
    palette: np.ndarray,
    fonts: dict,
) -> None:
    """Screen-space player legend; does not scale with map zoom."""
    x, y0 = origin
    y = y0 + 8
    limit = y0 + height - 18
    for p in frame["players"]:
        if y > limit:
            break
        swatch = pygame.Rect(x + 8, y, 14, 14)
        pygame.draw.rect(surface, palette[3 + p["idx"]].tolist(), swatch)
        if not p["alive"]:
            pygame.draw.line(surface, (200, 200, 200), swatch.topleft, swatch.bottomright, 2)
        marker = "* " if p["isAgent"] else "  "
        txt = f"{marker}{p['name'][:16]:16} {p['tiles']:>6}"
        surface.blit(fonts["legend"].render(txt, True, (220, 220, 220)), (x + 26, y))
        y += 17


def _draw_scene(
    pygame,
    frame: dict,
    palette: np.ndarray,
    bright_palette: np.ndarray,
    status: str,
    scale: int,
    fonts: dict,
    scene=None,
):
    """Native composite used for saved video (map + legend + status)."""
    map_surf = _draw_map_surface(pygame, frame, palette, bright_palette, scale, fonts)
    mw, mh = map_surf.get_size()
    if scene is None or scene.get_size() != (mw + LEGEND_W, mh + STATUS_H):
        scene = pygame.Surface((mw + LEGEND_W, mh + STATUS_H))
    scene.fill((15, 15, 20))
    scene.blit(map_surf, (0, 0))
    _draw_legend(pygame, scene, (mw, 0), mh, frame, palette, fonts)
    scene.blit(fonts["status"].render(status, True, (235, 235, 235)), (6, mh + 6))
    return scene


def present_map(
    pygame,
    screen,
    map_surf,
    viewport: tuple[int, int, int, int],
    transform: ViewTransform,
) -> None:
    """Blit an aspect-correct zoomed map crop into a screen viewport."""
    vx, vy, vw, vh = viewport
    pygame.draw.rect(screen, (0, 0, 0), pygame.Rect(vx, vy, vw, vh))
    src = pygame.Rect(*transform.source_rect()).clip(map_surf.get_rect())
    if src.width < 1 or src.height < 1:
        return
    chunk = map_surf.subsurface(src)
    if (src.width, src.height) != (transform.dest_w, transform.dest_h):
        chunk = pygame.transform.scale(chunk, (transform.dest_w, transform.dest_h))
    screen.blit(chunk, (vx + transform.dest_x, vy + transform.dest_y))


def _obs_status(ep: int, step: int, acts: np.ndarray, obs: dict, tick: int | None = None) -> str:
    g = obs["global"][0]
    tick_part = f" tick{tick}" if tick is not None else ""
    return (
        f"ep{ep} step{step}{tick_part} | {action_str(acts)}"
        f" | myTiles={g[2]:.3f} troops={np.expm1(g[3] * 15):.0f}"
        f" gold={np.expm1(g[4] * 20):.0f}"
    )


def run_headless(args: argparse.Namespace) -> None:
    """Stream saved video. Unchanged --save behavior; no interactive window."""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy, ckpt = load_policy(args.ckpt, device)
    seed_policy_rng(args.policy_seed)
    print(f"loaded {args.ckpt} (iter {ckpt.get('stats', {}).get('iterations', '?')})")

    pygame.display.init()
    pygame.font.init()
    fonts = _load_fonts(pygame, args.scale)
    palettes = PaletteCache()
    server = EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)
    writer = None
    out_name = ""
    scene = None

    try:
        for ep in range(args.episodes):
            writer = None
            out_name = episode_video_path(args.save, ep)
            try:
                obs = env.init(
                    [
                        {
                            "map": args.map,
                            "mapSize": args.map_size,
                            "nations": args.nations,
                            "difficulty": args.difficulty,
                            "bots": args.bots,
                            "seed": f"{args.seed}-{ep}",
                            "maxTicks": args.max_ticks,
                            "decisionInterval": 10,
                            "shaping": 0,
                        }
                    ]
                )
                done = False
                step = 0
                legend_printed = False
                batch = None
                with torch.inference_mode():
                    while not done:
                        obs_t = obs_to_tensors(obs, device)
                        a, _, _ = policy.act(obs_t, deterministic=args.greedy)
                        acts = a[0].cpu().numpy()

                        batch = env.step(acts[None, :])
                        obs = batch.obs
                        done = bool(batch.dones[0])
                        step += 1

                        if not should_render_frame(step, done, args.frame_skip):
                            if step % 20 == 0:
                                print(_obs_status(ep, step, acts, obs))
                            continue

                        frame = env.frame(0)
                        if not legend_printed:
                            print(f"--- map {frame['width']}x{frame['height']} players ---")
                            for p in frame["players"]:
                                tag = " <== AGENT" if p["isAgent"] else ""
                                print(f"  color#{p['idx']}: {p['name']}{tag}")
                            legend_printed = True

                        palette, bright_palette = palettes.for_players(len(frame["players"]))
                        status = _obs_status(ep, step, acts, obs, tick=frame["tick"])
                        if step % 20 == 0 or done:
                            print(status)

                        expected_scene_size = scene_size(
                            frame["width"], frame["height"], args.scale
                        )
                        if scene is not None and scene.get_size() != expected_scene_size:
                            scene = None
                        scene = _draw_scene(
                            pygame,
                            frame,
                            palette,
                            bright_palette,
                            status,
                            args.scale,
                            fonts,
                            scene,
                        )
                        if writer is None:
                            writer = open_video_writer(out_name, args.fps)
                        if writer is not None:
                            arr = pygame.surfarray.array3d(scene)
                            writer.append_data(np.transpose(arr, (1, 0, 2)))

                if batch is not None:
                    info = batch.infos[0]
                    result = episode_result(info)
                    print(
                        f"episode {ep}: {result} tick={info.get('tick')} "
                        f"kills={info.get('kills')} tilesFrac={info.get('tilesFrac'):.3f}"
                    )
            finally:
                if writer is not None:
                    close_video_writer(writer)
                    print(f"wrote {out_name}")
                    writer = None
    except KeyboardInterrupt:
        print("\n[watch] interrupted", flush=True)
    finally:
        env.close()
        server.stop()
        try:
            pygame.quit()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.save:
        run_headless(args)
        return
    from .watch_ui import run_interactive

    run_interactive(args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch] interrupted", flush=True)
