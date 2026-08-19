"""Fullscreen pygame UI for agent.watch. No extra GUI toolkit.

Interactive only. Headless --save stays in agent.watch.run_headless.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .env_client import EnvServerProcess, VecEnv
from .model import obs_to_tensors
from .watch import (
    CHECKPOINT_ROOT,
    DIFFICULTIES,
    MAP_SIZES,
    OPPONENT_TYPES,
    POST_MATCH_ACTIONS,
    PRESETS,
    SPEED_STEPS,
    Camera,
    PaletteCache,
    PlaybackControl,
    ReplayTape,
    WatchSettings,
    _draw_legend,
    _draw_map_surface,
    _load_fonts,
    _obs_status,
    desktop_size,
    discover_checkpoints,
    discover_maps,
    episode_result,
    load_policy,
    overlay_layout,
    parse_nations_value,
    present_map,
    restore_policy_rng,
    seed_policy_rng,
    should_render_frame,
    snap_speed,
    snapshot_policy_rng,
    usable_display_size,
)

BG = (12, 14, 20)
PANEL = (22, 26, 36)
PANEL_HI = (34, 42, 58)
ACCENT = (70, 150, 230)
TEXT = (230, 232, 238)
MUTED = (148, 154, 168)
DANGER = (220, 86, 86)
OK = (86, 196, 126)
SHADOW = (0, 0, 0)

CONFIG_FIELDS = (
    ("ckpt", "Checkpoint"),
    ("preset", "Preset"),
    ("game_type", "Game type"),
    ("map", "Map"),
    ("map_size", "Map size"),
    ("difficulty", "Difficulty"),
    ("nations", "Nations"),
    ("bots", "Bots"),
    ("seed", "Seed"),
    ("policy_seed", "Policy seed"),
    ("max_ticks", "Max ticks"),
    ("episodes", "Episodes"),
    ("scale", "Scale"),
    ("fps", "FPS"),
    ("frame_skip", "Frame skip"),
    ("speed", "Speed"),
    ("greedy", "Greedy"),
    ("display", "Display"),
)

NATIONS_CHOICES = ("disabled", "default", 1, 4, 8, 12, 16, 20)
BOTS_CHOICES = (0, 1, 5, 10, 20, 40, 100, 200, 400)
INT_FIELDS = {
    "bots",
    "policy_seed",
    "max_ticks",
    "episodes",
    "scale",
    "frame_skip",
    "display",
    "port",
}
TEXT_FIELDS = {"ckpt", "seed", "nations", "map"}
EDITABLE_TEXT = TEXT_FIELDS | INT_FIELDS | {"fps", "speed"}
FIELD_MIN = {
    "bots": 0,
    "policy_seed": 0,
    "max_ticks": 1,
    "episodes": 1,
    "scale": 1,
    "frame_skip": 1,
    "display": 0,
    "port": 1,
}
FIELD_MAX = {
    "bots": 400,
}


def _load_ui_fonts(pygame) -> dict:
    return {
        "title": pygame.font.SysFont("monospace", 28, bold=True),
        "heading": pygame.font.SysFont("monospace", 18, bold=True),
        "body": pygame.font.SysFont("monospace", 16),
        "small": pygame.font.SysFont("monospace", 14),
        "tiny": pygame.font.SysFont("monospace", 13),
    }


def _blit_text(surface, font, text: str, pos: tuple[int, int], color=TEXT) -> None:
    surface.blit(font.render(text, True, color), pos)


def _fill_rect(pygame, surface, rect, color, border=0, border_color=None) -> None:
    pygame.draw.rect(surface, color, rect)
    if border and border_color is not None:
        pygame.draw.rect(surface, border_color, rect, border)


class DisplayState:
    """True fullscreen by default; F11 / Alt+Enter leave or re-enter it."""

    def __init__(self, pygame, settings: WatchSettings):
        self.pygame = pygame
        self.display = int(settings.display)
        self.fullscreen = not settings.windowed
        self.requested_window = settings.window_size
        self.screen = None
        self.open()

    def open(self) -> None:
        flags = 0
        if self.fullscreen:
            size = desktop_size(self.pygame, self.display)
            flags = self.pygame.FULLSCREEN
        elif self.requested_window:
            usable = usable_display_size(self.pygame, self.display)
            size = (
                min(self.requested_window[0], usable[0]),
                min(self.requested_window[1], usable[1]),
            )
        else:
            size = usable_display_size(self.pygame, self.display)
        self.screen = self.pygame.display.set_mode(size, flags, display=self.display)
        self.pygame.display.set_caption("OFAI watch")
        self.pygame.event.set_grab(False)
        self.pygame.mouse.set_visible(True)

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        self.open()

    def size(self) -> tuple[int, int]:
        return self.screen.get_size()


@dataclass
class Runtime:
    settings: WatchSettings
    maps: list[str]
    checkpoints: list[str]
    display_count: int
    device: torch.device
    policy: object | None = None
    ckpt_info: dict | None = None
    server: EnvServerProcess | None = None
    env: VecEnv | None = None
    error: str = ""
    mode: str = "config"
    overlay: str | None = None
    episode: int = 0
    camera: Camera | None = None
    playback: PlaybackControl | None = None
    tape: ReplayTape | None = None
    palettes: PaletteCache | None = None
    map_surf: object | None = None
    map_fonts: dict | None = None
    last_frame: dict | None = None
    last_status: str = ""
    live_obs: dict | None = None
    live_done: bool = False
    live_step: int = 0
    live_rng: object | None = None
    replay_index: int = 0
    replay_action_i: int = 0
    replay_obs: dict | None = None
    dragging: bool = False
    focus: int = 0
    draft: str | None = None
    scroll: int = 0
    did_render: bool = False
    hold_frame: bool = False
    fullscreen: bool = True
    camera_ready: bool = False
    menu_hits: list | None = None
    menu_index: int = 0


def field_value(settings: WatchSettings, key: str) -> str:
    value = getattr(settings, key)
    if key == "greedy":
        return "on" if value else "off"
    if key == "speed":
        return f"{float(value):g}x"
    if key == "nations":
        return str(value)
    return str(value)


def apply_text_field(settings: WatchSettings, key: str, text: str) -> str | None:
    raw = text.strip()
    try:
        if key == "nations":
            settings.nations = parse_nations_value(raw)
            settings.preset = "custom"
            settings.game_type = (
                "tribes"
                if settings.nations == "disabled"
                else infer_or_keep_mixed(settings)
            )
        elif key == "ckpt":
            settings.ckpt = raw
        elif key == "seed":
            settings.seed = raw
        elif key == "map":
            if not raw:
                return "Map must not be empty."
            settings.map = raw
            settings.preset = "custom"
        elif key == "bots":
            settings.bots = _parse_int(raw, 0, 400)
            settings.preset = "custom"
            settings.game_type = infer_or_keep_mixed(settings)
        elif key == "policy_seed":
            settings.policy_seed = int(raw)
        elif key == "max_ticks":
            settings.max_ticks = _parse_int(raw, 1, None)
            settings.preset = "custom"
        elif key == "episodes":
            settings.episodes = _parse_int(raw, 1, None)
        elif key == "scale":
            settings.scale = _parse_int(raw, 1, None)
        elif key == "fps":
            value = float(raw)
            if value <= 0:
                return "FPS must be positive."
            settings.fps = value
        elif key == "frame_skip":
            settings.frame_skip = _parse_int(raw, 1, None)
        elif key == "display":
            settings.display = _parse_int(raw, 0, None)
        elif key == "speed":
            settings.speed = snap_speed(float(raw))
        else:
            return f"Cannot edit {key} as text."
    except ValueError as exc:
        return str(exc)
    return None


def infer_or_keep_mixed(settings: WatchSettings) -> str:
    if settings.nations == "disabled":
        return "tribes"
    if int(settings.bots) == 0:
        return "nations"
    return "mixed"


def _parse_int(text: str, lo: int, hi: int | None) -> int:
    value = int(text)
    if value < lo or (hi is not None and value > hi):
        bound = f"{lo}..{hi}" if hi is not None else f">={lo}"
        raise ValueError(f"expected integer {bound}, got {value}")
    return value


def nudge_field(rt: Runtime, key: str, delta: int) -> None:
    settings = rt.settings
    if key == "preset":
        settings.apply_preset(cycle_from(("custom", *PRESETS), settings.preset, delta))
    elif key == "game_type":
        settings.apply_game_type(cycle_from(OPPONENT_TYPES, settings.game_type, delta))
    elif key == "map":
        options = list(rt.maps)
        if settings.map not in options:
            options.insert(0, settings.map)
        settings.map = cycle_from(options, settings.map, delta)
        settings.preset = "custom"
    elif key == "map_size":
        settings.map_size = cycle_from(MAP_SIZES, settings.map_size, delta)
        settings.preset = "custom"
    elif key == "difficulty":
        settings.difficulty = cycle_from(DIFFICULTIES, settings.difficulty, delta)
        settings.preset = "custom"
    elif key == "nations":
        options = list(NATIONS_CHOICES)
        if settings.nations not in options:
            options.append(settings.nations)
        settings.nations = cycle_from(options, settings.nations, delta)
        settings.preset = "custom"
        settings.game_type = infer_or_keep_mixed(settings)
    elif key == "bots":
        options = list(BOTS_CHOICES)
        if settings.bots not in options:
            options.append(settings.bots)
        settings.bots = cycle_from(sorted(set(int(v) for v in options)), settings.bots, delta)
        settings.preset = "custom"
        settings.game_type = infer_or_keep_mixed(settings)
    elif key == "ckpt":
        options = list(rt.checkpoints)
        if settings.ckpt not in options:
            options.insert(0, settings.ckpt)
        if options:
            settings.ckpt = cycle_from(options, settings.ckpt, delta)
    elif key == "speed":
        settings.speed = cycle_from(SPEED_STEPS, settings.speed, delta)
    elif key == "greedy":
        settings.greedy = not settings.greedy
    elif key == "display":
        count = max(1, rt.display_count)
        settings.display = (int(settings.display) + delta) % count
    elif key in INT_FIELDS:
        current = int(getattr(settings, key))
        lo = FIELD_MIN.get(key, 0)
        hi = FIELD_MAX.get(key)
        value = current + int(delta)
        value = max(lo, value if hi is None else min(hi, value))
        setattr(settings, key, value)
        if key in ("bots", "max_ticks"):
            settings.preset = "custom"
        if key == "bots":
            settings.game_type = infer_or_keep_mixed(settings)
    elif key == "fps":
        settings.fps = max(1.0, float(settings.fps) + (5.0 if delta > 0 else -5.0))


def cycle_from(choices, current, delta: int):
    values = list(choices)
    if not values:
        return current
    try:
        index = values.index(current)
    except ValueError:
        index = 0
    return values[(index + int(delta)) % len(values)]


def commit_draft(rt: Runtime) -> None:
    if rt.draft is None:
        return
    key = CONFIG_FIELDS[rt.focus][0]
    err = apply_text_field(rt.settings, key, rt.draft)
    rt.draft = None
    if err:
        rt.error = err


def start_session(rt: Runtime, display: DisplayState | None = None) -> None:
    commit_draft(rt)
    err = rt.settings.validate()
    if err:
        rt.error = err
        return
    if rt.settings.display >= max(1, rt.display_count):
        rt.error = (
            f"Display {rt.settings.display} is unavailable; "
            f"found {rt.display_count} display(s)."
        )
        return
    try:
        if display is not None and display.display != rt.settings.display:
            display.display = rt.settings.display
            display.open()
            rt.fullscreen = display.fullscreen
        path = rt.settings.resolved_ckpt()
        rt.policy, rt.ckpt_info = load_policy(str(path), rt.device)
        seed_policy_rng(rt.settings.policy_seed)
        if not _env_ready(rt):
            shutdown(rt)
            rt.server = EnvServerProcess(rt.settings.port)
            rt.env = VecEnv("127.0.0.1", rt.settings.port)
        rt.episode = 0
        rt.error = ""
        begin_live_match(rt)
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "config"


def begin_live_match(rt: Runtime) -> None:
    assert rt.env is not None and rt.policy is not None
    rt.tape = ReplayTape()
    rt.tape.env_config = rt.settings.env_config(rt.episode)
    rt.tape.seed = str(rt.tape.env_config["seed"])
    rt.live_rng = snapshot_policy_rng()
    rt.hold_frame = False
    rt.live_obs = rt.env.init([rt.tape.env_config])
    rt.live_done = False
    rt.live_step = 0
    rt.last_frame = None
    rt.last_status = "Starting match..."
    rt.map_surf = None
    rt.camera = Camera()
    rt.camera_ready = False
    rt.playback = PlaybackControl(rt.settings.speed)
    rt.palettes = PaletteCache()
    rt.mode = "playing"
    rt.overlay = None
    print(f"loaded {rt.settings.ckpt} seed={rt.tape.seed}")


def restart_same_match(rt: Runtime) -> None:
    if rt.live_rng is not None:
        restore_policy_rng(rt.live_rng)
    begin_live_match(rt)


def start_new_seeded_match(rt: Runtime) -> None:
    rt.episode += 1
    begin_live_match(rt)


def advance_live(rt: Runtime) -> None:
    rt.did_render = False
    if rt.live_done or rt.env is None or rt.policy is None or rt.tape is None:
        return
    try:
        with torch.inference_mode():
            obs_t = obs_to_tensors(rt.live_obs, rt.device)
            action, _, _ = rt.policy.act(obs_t, deterministic=rt.settings.greedy)
            acts = action[0].cpu().numpy()
        rt.tape.record_action(acts)
        batch = rt.env.step(acts[None, :])
        rt.live_obs = batch.obs
        rt.live_done = bool(batch.dones[0])
        rt.live_step += 1
        if should_render_frame(rt.live_step, rt.live_done, rt.settings.frame_skip):
            capture_live_frame(rt, acts)
            rt.did_render = True
        elif rt.live_step % 20 == 0:
            rt.last_status = _obs_status(rt.episode, rt.live_step, acts, rt.live_obs)
        if rt.live_done:
            info = batch.infos[0]
            rt.tape.result = episode_result(info)
            rt.tape.result_info = info
            if rt.last_frame is None:
                capture_live_frame(rt, acts)
            print(
                f"episode {rt.episode}: {rt.tape.result} tick={info.get('tick')} "
                f"kills={info.get('kills')} tilesFrac={info.get('tilesFrac'):.3f}"
            )
            rt.mode = "post"
            rt.overlay = None
            rt.menu_index = 0
            if rt.playback:
                rt.playback.paused = True
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"
        rt.overlay = None


def capture_live_frame(rt: Runtime, acts: np.ndarray) -> None:
    assert rt.env is not None and rt.tape is not None
    frame = rt.env.frame(0)
    status = _obs_status(rt.episode, rt.live_step, acts, rt.live_obs, tick=frame["tick"])
    rt.last_frame = frame
    rt.last_status = status
    rt.tape.record_frame(frame, status)
    rt.map_surf = None
    if rt.camera and not rt.camera_ready:
        rt.camera.reset(frame["width"] * rt.settings.scale, frame["height"] * rt.settings.scale)
        rt.camera_ready = True


def start_replay(rt: Runtime) -> None:
    if rt.tape is None or (not rt.tape.actions and not rt.tape.has_frames):
        rt.error = "Nothing recorded to replay."
        rt.mode = "post"
        return
    rt.playback = PlaybackControl(rt.settings.speed)
    rt.camera = rt.camera or Camera()
    rt.replay_index = 0
    rt.replay_action_i = 0
    rt.replay_obs = None
    rt.live_done = False
    rt.error = ""
    rt.hold_frame = True
    if rt.tape.has_frames:
        show_replay_frame(rt, 0)
        rt.mode = "replay"
        return
    if rt.env is None:
        rt.error = "Env server is not running; cannot action-replay."
        rt.mode = "post"
        rt.hold_frame = False
        return
    try:
        rt.replay_obs = rt.env.init([rt.tape.env_config])
        rt.last_status = "Replay (recorded actions, policy not run)"
        rt.last_frame = rt.env.frame(0)
        rt.map_surf = None
        rt.mode = "replay"
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"
        rt.hold_frame = False


def show_replay_frame(rt: Runtime, index: int) -> None:
    assert rt.tape is not None and rt.tape.frames
    rt.replay_index = max(0, min(index, len(rt.tape.frames) - 1))
    frame = rt.tape.frames[rt.replay_index]
    rt.last_frame = frame
    rt.last_status = frame.get("status") or f"replay {rt.replay_index + 1}/{len(rt.tape.frames)}"
    rt.map_surf = None
    rt.live_done = rt.replay_index >= len(rt.tape.frames) - 1


def advance_replay(rt: Runtime) -> None:
    rt.did_render = False
    if rt.tape is None:
        return
    if rt.tape.has_frames:
        if rt.replay_index + 1 < len(rt.tape.frames):
            show_replay_frame(rt, rt.replay_index + 1)
            rt.did_render = True
        else:
            rt.live_done = True
            rt.mode = "post"
            if rt.playback:
                rt.playback.paused = True
        return
    if rt.env is None or rt.replay_action_i >= len(rt.tape.actions):
        rt.live_done = True
        rt.mode = "post"
        if rt.playback:
            rt.playback.paused = True
        return
    try:
        acts = rt.tape.actions[rt.replay_action_i]
        batch = rt.env.step(acts[None, :])
        rt.replay_obs = batch.obs
        rt.replay_action_i += 1
        done = bool(batch.dones[0]) or rt.replay_action_i >= len(rt.tape.actions)
        if should_render_frame(rt.replay_action_i, done, rt.settings.frame_skip):
            frame = rt.env.frame(0)
            rt.last_frame = frame
            rt.last_status = _obs_status(
                rt.episode,
                rt.replay_action_i,
                acts,
                rt.replay_obs,
                tick=frame["tick"],
            )
            rt.map_surf = None
            rt.did_render = True
        if done:
            rt.live_done = True
            rt.mode = "post"
            if rt.playback:
                rt.playback.paused = True
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"


def retreat_replay(rt: Runtime) -> None:
    if rt.tape and rt.tape.has_frames:
        show_replay_frame(rt, rt.replay_index - 1)
        rt.mode = "replay"
        rt.live_done = False


def handle_play_key(rt: Runtime, pygame, event, map_viewport: tuple[int, int, int, int]) -> None:
    key = event.key
    if key in (pygame.K_SPACE,):
        if rt.playback:
            rt.playback.toggle_pause()
    elif key in (pygame.K_PERIOD, pygame.K_n):
        if rt.playback:
            rt.playback.request_step()
    elif key == pygame.K_COMMA and rt.mode == "replay":
        retreat_replay(rt)
    elif key == pygame.K_LEFTBRACKET:
        if rt.playback:
            rt.playback.slower()
            rt.settings.speed = rt.playback.speed
    elif key == pygame.K_RIGHTBRACKET:
        if rt.playback:
            rt.playback.faster()
            rt.settings.speed = rt.playback.speed
    elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
        zoom_camera(rt, 1.15, map_viewport, None)
    elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
        zoom_camera(rt, 1 / 1.15, map_viewport, None)
    elif key in (pygame.K_0, pygame.K_HOME, pygame.K_r):
        reset_camera(rt)
    elif key in (pygame.K_LEFT, pygame.K_a):
        pan_camera(rt, -80, 0, map_viewport)
    elif key in (pygame.K_RIGHT, pygame.K_d):
        pan_camera(rt, 80, 0, map_viewport)
    elif key in (pygame.K_UP, pygame.K_w):
        pan_camera(rt, 0, -80, map_viewport)
    elif key in (pygame.K_DOWN, pygame.K_s):
        pan_camera(rt, 0, 80, map_viewport)


def camera_sizes(rt: Runtime, map_viewport: tuple[int, int, int, int]):
    if rt.last_frame is None or rt.camera is None:
        return None
    map_size = (
        rt.last_frame["width"] * rt.settings.scale,
        rt.last_frame["height"] * rt.settings.scale,
    )
    viewport_size = (map_viewport[2], map_viewport[3])
    return map_size, viewport_size, rt.camera.transform(map_size, viewport_size)


def zoom_camera(rt: Runtime, factor: float, map_viewport, anchor_map) -> None:
    sizes = camera_sizes(rt, map_viewport)
    if not sizes or rt.camera is None:
        return
    map_size, viewport_size, _ = sizes
    rt.camera.zoom_by(factor, map_size, viewport_size, anchor_map)


def pan_camera(rt: Runtime, dx_screen: float, dy_screen: float, map_viewport) -> None:
    sizes = camera_sizes(rt, map_viewport)
    if not sizes or rt.camera is None:
        return
    map_size, viewport_size, view = sizes
    rt.camera.pan(dx_screen / view.scale, dy_screen / view.scale, map_size, viewport_size)


def reset_camera(rt: Runtime) -> None:
    if rt.camera and rt.last_frame is not None:
        rt.camera.reset(
            rt.last_frame["width"] * rt.settings.scale,
            rt.last_frame["height"] * rt.settings.scale,
        )


def point_in(rect: tuple[int, int, int, int], pos: tuple[int, int]) -> bool:
    x, y, w, h = rect
    return x <= pos[0] < x + w and y <= pos[1] < y + h


def ensure_map_surface(rt: Runtime, pygame) -> None:
    if rt.map_surf is not None:
        return
    if rt.last_frame is None or rt.palettes is None or rt.map_fonts is None:
        return
    palette, bright = rt.palettes.for_players(len(rt.last_frame["players"]))
    rt.map_surf = _draw_map_surface(
        pygame,
        rt.last_frame,
        palette,
        bright,
        rt.settings.scale,
        rt.map_fonts,
        rt.map_surf,
    )


def draw_hud(pygame, screen, fonts, layout, rt: Runtime) -> None:
    top = pygame.Rect(*layout["top"])
    bottom = pygame.Rect(*layout["bottom"])
    legend = pygame.Rect(*layout["legend"])
    _fill_rect(pygame, screen, top, PANEL)
    _fill_rect(pygame, screen, bottom, PANEL)
    _fill_rect(pygame, screen, legend, (16, 18, 26))

    speed = rt.playback.speed if rt.playback else rt.settings.speed
    paused = bool(rt.playback and rt.playback.paused)
    if rt.mode == "replay" and rt.tape:
        kind = "REPLAY frames" if rt.tape.has_frames else "REPLAY actions"
    elif rt.mode == "post":
        kind = f"MATCH OVER {rt.tape.result if rt.tape else ''}"
    else:
        kind = "LIVE"
    pause_txt = " PAUSED" if paused and rt.mode != "post" else ""
    title = (
        f"OFAI watch   SPEED {speed:g}x   {kind}{pause_txt}   "
        f"skip={rt.settings.frame_skip} fps={rt.settings.fps:g}   "
        f"{'fullscreen' if rt.fullscreen else 'windowed'}"
    )
    _blit_text(screen, fonts["body"], title, (12, top.y + 10), ACCENT if paused else TEXT)
    _blit_text(
        screen,
        fonts["tiny"],
        "F11/Alt+Enter windowed   Esc menu   Q quit   mouse not grabbed",
        (top.right - 520, top.y + 12),
        MUTED,
    )

    if rt.last_frame is not None and rt.palettes is not None:
        palette, _ = rt.palettes.for_players(len(rt.last_frame["players"]))
        _draw_legend(
            pygame,
            screen,
            (legend.x, legend.y),
            legend.h,
            rt.last_frame,
            palette,
            {"legend": fonts["tiny"]},
        )

    status = rt.last_status or ""
    if rt.error:
        status = rt.error
    _blit_text(
        screen,
        fonts["small"],
        status[:140],
        (12, bottom.y + 8),
        DANGER if rt.error else TEXT,
    )
    if rt.mode == "replay" and rt.tape and rt.tape.has_frames:
        hint = "Space pause  [ ] speed  . next  , prev  wheel/+- zoom  arrows pan  0 recenter"
    else:
        hint = "Space pause  [ ] speed  . step  wheel/+- zoom  arrows/WASD pan  drag pan  0 recenter"
    _blit_text(screen, fonts["tiny"], hint, (12, bottom.y + 32), MUTED)


def draw_config(pygame, screen, fonts, rt: Runtime) -> list[tuple[str, object]]:
    """Draw the configuration form. Returns hit targets (name, payload)."""
    w, h = screen.get_size()
    screen.fill(BG)
    _blit_text(screen, fonts["title"], "OFAI watch", (48, 28), TEXT)
    _blit_text(
        screen,
        fonts["small"],
        "Fullscreen match viewer. Tab fields, Left/Right cycle, type to edit, Enter Start, Esc quit.",
        (48, 68),
        MUTED,
    )
    hits: list[tuple[str, object]] = []
    row_h = 34
    col_w = min(620, (w - 120) // 2)
    left_x = 48
    right_x = 48 + col_w + 36
    start_y = 108
    for i, (key, label) in enumerate(CONFIG_FIELDS):
        col = 0 if i < 10 else 1
        row = i if i < 10 else i - 10
        x = left_x if col == 0 else right_x
        y = start_y + row * row_h
        focused = i == rt.focus
        label_surf = fonts["body"].render(label, True, ACCENT if focused else MUTED)
        screen.blit(label_surf, (x, y + 6))
        box = pygame.Rect(x + 160, y, col_w - 180, 28)
        _fill_rect(
            pygame,
            screen,
            box,
            PANEL_HI if focused else PANEL,
            1,
            ACCENT if focused else (50, 56, 70),
        )
        shown = rt.draft if focused and rt.draft is not None else field_value(rt.settings, key)
        _blit_text(screen, fonts["body"], shown[:48], (box.x + 8, box.y + 5), TEXT)
        hits.append(("field", (i, box)))

    btn_y = min(h - 90, start_y + 10 * row_h + 24)
    start_btn = pygame.Rect(48, btn_y, 160, 44)
    quit_btn = pygame.Rect(224, btn_y, 160, 44)
    _fill_rect(pygame, screen, start_btn, (28, 72, 48), 1, OK)
    _fill_rect(pygame, screen, quit_btn, (72, 28, 32), 1, DANGER)
    _blit_text(screen, fonts["heading"], "Start", (start_btn.x + 44, start_btn.y + 10), OK)
    _blit_text(screen, fonts["heading"], "Quit", (quit_btn.x + 52, quit_btn.y + 10), DANGER)
    hits.append(("start", start_btn))
    hits.append(("quit", quit_btn))

    if rt.error:
        _blit_text(screen, fonts["body"], rt.error, (48, btn_y + 56), DANGER)
    ckpt_n = len(rt.checkpoints)
    map_n = len(rt.maps)
    _blit_text(
        screen,
        fonts["tiny"],
        f"{ckpt_n} checkpoint(s) in {CHECKPOINT_ROOT.name}/   "
        f"{map_n} map key(s) from curriculum"
        + (" + local OpenFrontIO" if map_n > 3 else "")
        + "   Episodes is used by --save; in-window use New after a match",
        (400, btn_y + 16),
        MUTED,
    )
    return hits


def draw_center_menu(
    pygame,
    screen,
    fonts,
    title: str,
    items: list[tuple[str, str]],
    selected: int = 0,
    rt: Runtime | None = None,
) -> list[tuple[str, object]]:
    w, h = screen.get_size()
    box_w, box_h = 560, 72 + 44 * len(items)
    box = pygame.Rect((w - box_w) // 2, (h - box_h) // 2, box_w, box_h)
    shadow = box.move(4, 4)
    _fill_rect(pygame, screen, shadow, SHADOW)
    _fill_rect(pygame, screen, box, PANEL, 1, ACCENT)
    _blit_text(screen, fonts["heading"], title, (box.x + 24, box.y + 18), TEXT)
    hits = []
    for i, (key, label) in enumerate(items):
        row = pygame.Rect(box.x + 24, box.y + 58 + i * 40, box_w - 48, 34)
        _fill_rect(pygame, screen, row, PANEL_HI if i == selected else (18, 22, 30), 1, ACCENT)
        _blit_text(screen, fonts["body"], label, (row.x + 12, row.y + 6), TEXT)
        hits.append((key, row))
    if rt is not None:
        rt.menu_hits = hits
    return hits


def post_match_items(rt: Runtime) -> list[tuple[str, str]]:
    replay_note = "from frames" if rt.tape and rt.tape.has_frames else "from recorded actions"
    items = [
        ("replay", f"1  Replay last match ({replay_note})"),
        ("restart", "2  Restart same seeded match"),
        ("new", "3  New seeded match"),
        ("configure", "4  Return to configuration"),
        ("quit", "5  Quit"),
    ]
    return items


def play_menu_items() -> list[tuple[str, str]]:
    return [
        ("resume", "Resume"),
        ("configure", "Return to configuration"),
        ("quit", "Quit"),
    ]


def draw_play_view(pygame, screen, fonts, rt: Runtime) -> dict[str, tuple[int, int, int, int]]:
    screen.fill(BG)
    layout = overlay_layout(screen.get_size())
    if rt.last_frame is not None and rt.camera is not None:
        ensure_map_surface(rt, pygame)
        if rt.map_surf is not None:
            map_size = rt.map_surf.get_size()
            view = layout["map"]
            transform = rt.camera.transform(map_size, (view[2], view[3]))
            present_map(pygame, screen, rt.map_surf, view, transform)
    draw_hud(pygame, screen, fonts, layout, rt)
    if rt.overlay == "quit":
        draw_center_menu(
            pygame,
            screen,
            fonts,
            "Quit watch?",
            [("confirm_quit", "Enter  Yes"), ("cancel", "Esc  No")],
            selected=rt.menu_index,
            rt=rt,
        )
    elif rt.mode == "post":
        info = rt.tape.result_info if rt.tape else {}
        title = (
            f"{rt.tape.result if rt.tape else 'Done'}   "
            f"tick={info.get('tick', '?')} kills={info.get('kills', '?')}"
        )
        if rt.error:
            title = rt.error
        items = post_match_items(rt)
        draw_center_menu(
            pygame,
            screen,
            fonts,
            title,
            items,
            selected=min(rt.menu_index, max(0, len(items) - 1)),
            rt=rt,
        )
    elif rt.overlay == "menu":
        items = play_menu_items()
        draw_center_menu(
            pygame,
            screen,
            fonts,
            "Paused",
            items,
            selected=min(rt.menu_index, max(0, len(items) - 1)),
            rt=rt,
        )
    return layout


def choose_menu(rt: Runtime, action: str) -> None:
    if action == "resume":
        rt.overlay = None
        if rt.playback:
            rt.playback.paused = False
    elif action == "cancel":
        rt.overlay = None
    elif action == "configure":
        rt.mode = "config"
        rt.overlay = None
        rt.error = ""
    elif action == "quit":
        rt.overlay = "quit"
        rt.menu_index = 0
        if rt.playback:
            rt.playback.paused = True
    elif action == "confirm_quit":
        rt.mode = "quit"
    elif action == "replay":
        start_replay(rt)
    elif action == "restart":
        try:
            restart_same_match(rt)
        except Exception as exc:
            rt.error = f"{type(exc).__name__}: {exc}"
            rt.mode = "post"
            rt.menu_index = 0
    elif action == "new":
        try:
            start_new_seeded_match(rt)
        except Exception as exc:
            rt.error = f"{type(exc).__name__}: {exc}"
            rt.mode = "post"
            rt.menu_index = 0


def hit_menu(hits: list[tuple[str, object]], pos: tuple[int, int]) -> str | None:
    for key, rect in hits:
        if hasattr(rect, "collidepoint") and rect.collidepoint(pos):
            return key
    return None


def shutdown(rt: Runtime) -> None:
    if rt.env is not None:
        try:
            rt.env.close()
        except Exception:
            pass
        rt.env = None
    if rt.server is not None:
        try:
            rt.server.stop()
        except Exception:
            pass
        rt.server = None


def _env_ready(rt: Runtime) -> bool:
    return (
        rt.server is not None
        and rt.env is not None
        and rt.server.alive()
    )


def run_interactive(args) -> None:
    import pygame

    settings = WatchSettings.from_args(args)
    try:
        pygame.display.init()
        pygame.font.init()
        display = DisplayState(pygame, settings)
    except Exception as exc:
        print(
            f"[watch] no interactive display ({exc}); re-run with --save match.mp4 "
            "to render a video instead."
        )
        try:
            pygame.quit()
        except Exception:
            pass
        return

    device = torch.device(settings.device if torch.cuda.is_available() else "cpu")
    try:
        display_count = len(pygame.display.get_desktop_sizes()) or 1
    except Exception:
        display_count = 1
    rt = Runtime(
        settings=settings,
        maps=discover_maps(),
        checkpoints=discover_checkpoints(),
        display_count=display_count,
        device=device,
        playback=PlaybackControl(settings.speed),
        palettes=PaletteCache(),
        fullscreen=display.fullscreen,
    )
    ui_fonts = _load_ui_fonts(pygame)
    rt.map_fonts = _load_fonts(pygame, settings.scale)
    last_map_scale = settings.scale
    clock = pygame.time.Clock()
    running = True
    config_hits: list[tuple[str, object]] = []

    try:
        while running and rt.mode != "quit":
            layout = overlay_layout(display.size())
            map_view = layout["map"]
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    alt = bool(event.mod & pygame.KMOD_ALT)
                    if event.key == pygame.K_F11 or (event.key == pygame.K_RETURN and alt):
                        display.toggle_fullscreen()
                        rt.fullscreen = display.fullscreen
                        continue
                    if event.key == pygame.K_q and rt.mode != "config":
                        rt.overlay = "quit"
                        rt.menu_index = 0
                        if rt.playback:
                            rt.playback.paused = True
                        continue
                    if rt.overlay == "quit":
                        if event.key in (pygame.K_RETURN, pygame.K_y):
                            running = False
                        elif event.key in (pygame.K_ESCAPE, pygame.K_n):
                            rt.overlay = None
                        continue
                    if rt.mode == "config":
                        running = handle_config_event(rt, pygame, event, display)
                    elif rt.mode == "post":
                        handle_post_event(rt, pygame, event)
                    elif rt.overlay == "menu":
                        handle_play_menu_event(rt, pygame, event)
                    else:
                        if event.key == pygame.K_ESCAPE:
                            rt.overlay = "menu"
                            rt.menu_index = 0
                            if rt.playback:
                                rt.playback.paused = True
                        else:
                            handle_play_key(rt, pygame, event, map_view)
                elif event.type == pygame.MOUSEWHEEL:
                    if rt.overlay:
                        pass
                    elif rt.mode in ("playing", "replay", "post") and rt.last_frame:
                        pos = pygame.mouse.get_pos()
                        if point_in(map_view, pos):
                            sizes = camera_sizes(rt, map_view)
                            if sizes:
                                map_size, viewport_size, view = sizes
                                local = (pos[0] - map_view[0], pos[1] - map_view[1])
                                anchor = view.screen_to_map(*local)
                                zoom_camera(rt, 1.15 if event.y > 0 else 1 / 1.15, map_view, anchor)
                    elif rt.mode == "config":
                        nudge_field(rt, CONFIG_FIELDS[rt.focus][0], 1 if event.y > 0 else -1)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    pos = event.pos
                    if rt.overlay:
                        action = hit_menu(rt.menu_hits or [], pos)
                        if action:
                            choose_menu(rt, action)
                    elif rt.mode == "config":
                        handle_config_click(rt, pos, config_hits, display)
                    elif rt.mode == "post":
                        action = hit_menu(rt.menu_hits or [], pos)
                        if action:
                            choose_menu(rt, action)
                        elif event.button == 2:
                            reset_camera(rt)
                    elif event.button == 1 and point_in(map_view, pos):
                        rt.dragging = True
                    elif event.button == 2:
                        reset_camera(rt)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        rt.dragging = False
                elif event.type == pygame.MOUSEMOTION and rt.dragging:
                    if event.buttons[0]:
                        pan_camera(rt, -event.rel[0], -event.rel[1], map_view)

            if not running:
                break

            advanced = False
            if rt.hold_frame:
                rt.hold_frame = False
            elif rt.mode == "playing" and rt.overlay is None and rt.playback and rt.playback.should_advance():
                advance_live(rt)
                advanced = True
            elif rt.mode == "replay" and rt.overlay is None and rt.playback and rt.playback.should_advance():
                advance_replay(rt)
                advanced = True

            if rt.settings.scale != last_map_scale:
                rt.map_fonts = _load_fonts(pygame, rt.settings.scale)
                last_map_scale = rt.settings.scale
                rt.map_surf = None

            if rt.mode == "config":
                config_hits = draw_config(pygame, display.screen, ui_fonts, rt)
                if rt.overlay == "quit":
                    draw_center_menu(
                        pygame,
                        display.screen,
                        ui_fonts,
                        "Quit watch?",
                        [("confirm_quit", "Enter  Yes"), ("cancel", "Esc  No")],
                        selected=rt.menu_index,
                        rt=rt,
                    )
            else:
                draw_play_view(pygame, display.screen, ui_fonts, rt)

            pygame.display.flip()
            paused = bool(rt.playback and rt.playback.paused)
            if rt.mode in ("config", "post") or rt.overlay or paused:
                clock.tick(30)
            elif advanced and not rt.did_render:
                clock.tick()
            elif rt.playback:
                clock.tick(max(1, int(rt.settings.fps * rt.playback.speed)))
            else:
                clock.tick(30)
    except KeyboardInterrupt:
        print("\n[watch] interrupted", flush=True)
    finally:
        shutdown(rt)
        try:
            pygame.quit()
        except Exception:
            pass


def handle_config_event(rt: Runtime, pygame, event, display: DisplayState | None = None) -> bool:
    """Return False when the user confirmed quit."""
    key = event.key
    if key == pygame.K_ESCAPE:
        if rt.draft is not None:
            rt.draft = None
            return True
        rt.overlay = "quit"
        return True
    if rt.overlay == "quit":
        return True
    if key in (pygame.K_TAB, pygame.K_DOWN):
        commit_draft(rt)
        step = -1 if (event.mod & pygame.KMOD_SHIFT) and key == pygame.K_TAB else 1
        rt.focus = (rt.focus + step) % len(CONFIG_FIELDS)
        return True
    if key == pygame.K_UP:
        commit_draft(rt)
        rt.focus = (rt.focus - 1) % len(CONFIG_FIELDS)
        return True
    if key in (pygame.K_LEFT, pygame.K_RIGHT):
        if rt.draft is None:
            nudge_field(rt, CONFIG_FIELDS[rt.focus][0], 1 if key == pygame.K_RIGHT else -1)
        return True
    if key == pygame.K_RETURN:
        commit_draft(rt)
        start_session(rt, display)
        return True
    if key == pygame.K_BACKSPACE:
        key_name = CONFIG_FIELDS[rt.focus][0]
        if key_name not in EDITABLE_TEXT:
            return True
        if rt.draft is None:
            rt.draft = field_value(rt.settings, key_name)
        rt.draft = rt.draft[:-1]
        return True
    if event.unicode and event.unicode.isprintable():
        key_name = CONFIG_FIELDS[rt.focus][0]
        if key_name in EDITABLE_TEXT:
            if rt.draft is None:
                current = field_value(rt.settings, key_name)
                rt.draft = "" if event.unicode != current[:1] else current
                if rt.draft == current:
                    rt.draft = ""
            rt.draft += event.unicode
    return True


def handle_config_click(
    rt: Runtime,
    pos: tuple[int, int],
    hits: list[tuple[str, object]],
    display: DisplayState | None = None,
) -> None:
    for kind, payload in hits:
        if kind == "field":
            index, box = payload
            if box.collidepoint(pos):
                commit_draft(rt)
                rt.focus = index
                return
        elif kind == "start" and payload.collidepoint(pos):
            commit_draft(rt)
            start_session(rt, display)
            return
        elif kind == "quit" and payload.collidepoint(pos):
            rt.mode = "quit"


def handle_post_event(rt: Runtime, pygame, event) -> None:
    items = post_match_items(rt)
    if event.key in (pygame.K_UP, pygame.K_LEFT):
        rt.menu_index = (rt.menu_index - 1) % len(items)
        return
    if event.key in (pygame.K_DOWN, pygame.K_RIGHT):
        rt.menu_index = (rt.menu_index + 1) % len(items)
        return
    if event.key == pygame.K_RETURN:
        choose_menu(rt, items[rt.menu_index][0])
        return
    mapping = {
        pygame.K_1: "replay",
        pygame.K_2: "restart",
        pygame.K_3: "new",
        pygame.K_4: "configure",
        pygame.K_5: "quit",
        pygame.K_ESCAPE: "configure",
    }
    action = mapping.get(event.key)
    if action in POST_MATCH_ACTIONS or action == "configure":
        choose_menu(rt, action)


def handle_play_menu_event(rt: Runtime, pygame, event) -> None:
    items = play_menu_items()
    if event.key == pygame.K_ESCAPE:
        rt.overlay = None
        return
    if event.key in (pygame.K_UP, pygame.K_LEFT):
        rt.menu_index = (rt.menu_index - 1) % len(items)
        return
    if event.key in (pygame.K_DOWN, pygame.K_RIGHT):
        rt.menu_index = (rt.menu_index + 1) % len(items)
        return
    if event.key == pygame.K_RETURN:
        choose_menu(rt, items[rt.menu_index][0])
        return
    if event.key == pygame.K_c:
        choose_menu(rt, "configure")
