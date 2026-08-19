"""Fullscreen pygame UI for agent.watch. No extra GUI toolkit.

Interactive only. Headless --save stays in agent.watch.run_headless.
"""

from __future__ import annotations

from dataclasses import dataclass, field

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
    resolve_seek_target,
    restore_policy_rng,
    seed_policy_rng,
    should_render_frame,
    snap_speed,
    snapshot_policy_rng,
    timeline_frame_index,
    timeline_jump,
    timeline_thumb_frac,
    usable_display_size,
)

BG = (12, 14, 20)
PANEL = (22, 26, 36)
PANEL_HI = (34, 42, 58)
PANEL_HOVER = (44, 54, 74)
ACCENT = (70, 150, 230)
ACCENT_DIM = (40, 86, 128)
TEXT = (230, 232, 238)
MUTED = (148, 154, 168)
DANGER = (220, 86, 86)
OK = (86, 196, 126)
SHADOW = (0, 0, 0)
TRACK = (18, 20, 28)
EVICTED = (92, 72, 42)
BUFFERED = (38, 78, 112)
RERUN = (118, 88, 48)

CONFIG_FIELDS = (
    ("ckpt", "Checkpoint", "combo"),
    ("preset", "Preset", "dropdown"),
    ("game_type", "Game type", "dropdown"),
    ("map", "Map", "combo"),
    ("map_size", "Map size", "dropdown"),
    ("difficulty", "Difficulty", "dropdown"),
    ("nations", "Nations", "combo"),
    ("bots", "Bots", "stepper"),
    ("seed", "Seed", "text"),
    ("policy_seed", "Policy seed", "stepper"),
    ("max_ticks", "Max ticks", "stepper"),
    ("episodes", "Episodes", "stepper"),
    ("scale", "Scale", "stepper"),
    ("fps", "FPS", "stepper"),
    ("frame_skip", "Frame skip", "stepper"),
    ("speed", "Speed", "dropdown"),
    ("greedy", "Greedy", "dropdown"),
    ("display", "Display", "dropdown"),
)
FIELD_KINDS = {key: kind for key, _, kind in CONFIG_FIELDS}
FIELD_INDEX = {key: i for i, (key, _, _) in enumerate(CONFIG_FIELDS)}
CONFIG_SECTIONS = (
    ("Match", ("ckpt", "preset", "game_type", "map", "map_size", "difficulty", "nations", "bots")),
    ("Seeds and limits", ("seed", "policy_seed", "max_ticks", "episodes")),
    ("Playback", ("scale", "fps", "frame_skip", "speed", "greedy", "display")),
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


def field_kind(key: str) -> str:
    return FIELD_KINDS.get(key, "text")


def _load_ui_fonts(pygame, scale: float = 1.0) -> dict:
    s = float(scale)
    return {
        "title": pygame.font.SysFont("monospace", max(18, int(30 * s)), bold=True),
        "heading": pygame.font.SysFont("monospace", max(14, int(18 * s)), bold=True),
        "body": pygame.font.SysFont("monospace", max(13, int(16 * s))),
        "small": pygame.font.SysFont("monospace", max(12, int(14 * s))),
        "tiny": pygame.font.SysFont("monospace", max(11, int(13 * s))),
    }


def ui_metrics(size: tuple[int, int]) -> dict:
    width, height = size
    scale = max(0.72, min(1.25, min(width / 1600.0, height / 900.0)))
    return {
        "scale": scale,
        "margin": int(40 * scale),
        "gutter": int(28 * scale),
        "row_h": int(58 * scale),
        "field_h": int(34 * scale),
        "label_h": int(18 * scale),
        "btn_h": int(44 * scale),
        "btn_w": int(168 * scale),
        "section_gap": int(20 * scale),
        "two_col": width >= 1100,
    }


def _blit_text(surface, font, text: str, pos: tuple[int, int], color=TEXT) -> None:
    surface.blit(font.render(text, True, color), pos)


def _fill_rect(pygame, surface, rect, color, border=0, border_color=None) -> None:
    pygame.draw.rect(surface, color, rect)
    if border and border_color is not None:
        pygame.draw.rect(surface, border_color, rect, border)


def _ellipsize(font, text: str, max_w: int, *, tail: bool = False) -> str:
    if max_w < 8 or font.size(text)[0] <= max_w:
        return text
    ell = "..."
    if tail:
        for i in range(len(text)):
            candidate = ell + text[i:]
            if font.size(candidate)[0] <= max_w:
                return candidate
        return ell
    for i in range(len(text), 0, -1):
        candidate = text[:i] + ell
        if font.size(candidate)[0] <= max_w:
            return candidate
    return ell


def _draw_button(pygame, screen, font, rect, label, *, kind="default", hover=False):
    if kind == "ok":
        bg, border, fg = ((24, 64, 42), OK, OK)
    elif kind == "danger":
        bg, border, fg = ((64, 26, 30), DANGER, DANGER)
    elif kind == "accent":
        bg, border, fg = (ACCENT_DIM, ACCENT, TEXT)
    else:
        bg, border, fg = (PANEL_HI, (60, 68, 84), TEXT)
    if hover:
        bg = tuple(min(255, c + 18) for c in bg)
        if kind == "default":
            border = ACCENT
    _fill_rect(pygame, screen, rect, bg, 1, border)
    text = font.render(label, True, fg)
    screen.blit(text, text.get_rect(center=rect.center))


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
    replay_kind: str = "buffer"
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
    dropdown_key: str | None = None
    dropdown_index: int = 0
    dropdown_scroll: int = 0
    dropdown_menu: tuple[int, int, int, int] | None = None
    dropdown_options: list = field(default_factory=list)
    field_rects: dict = field(default_factory=dict)
    timeline_drag: bool = False
    timeline_hits: list | None = None
    pending_seek: int | None = None
    hover: str | None = None


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


def field_options(rt: Runtime, key: str) -> list[tuple[object, str]]:
    settings = rt.settings
    if key == "preset":
        return [(name, name) for name in ("custom", *PRESETS)]
    if key == "game_type":
        return [(name, name) for name in OPPONENT_TYPES]
    if key == "map":
        options = list(rt.maps)
        if settings.map not in options:
            options.insert(0, settings.map)
        return [(name, name) for name in options]
    if key == "map_size":
        return [(name, name) for name in MAP_SIZES]
    if key == "difficulty":
        return [(name, name) for name in DIFFICULTIES]
    if key == "nations":
        options = list(NATIONS_CHOICES)
        if settings.nations not in options:
            options.insert(0, settings.nations)
        return [(value, str(value)) for value in options]
    if key == "ckpt":
        options = list(rt.checkpoints)
        if settings.ckpt and settings.ckpt not in options:
            options.insert(0, settings.ckpt)
        return [(path, path) for path in options]
    if key == "greedy":
        return [(False, "off"), (True, "on")]
    if key == "speed":
        return [(step, f"{step:g}x") for step in SPEED_STEPS]
    if key == "display":
        count = max(1, rt.display_count)
        return [(i, f"{i}") for i in range(count)]
    return []


def apply_choice(rt: Runtime, key: str, value) -> str | None:
    settings = rt.settings
    if key == "preset":
        settings.apply_preset(str(value))
    elif key == "game_type":
        settings.apply_game_type(str(value))
    elif key == "map":
        if not str(value).strip():
            return "Map must not be empty."
        settings.map = str(value)
        settings.preset = "custom"
    elif key == "map_size":
        settings.map_size = str(value)
        settings.preset = "custom"
    elif key == "difficulty":
        settings.difficulty = str(value)
        settings.preset = "custom"
    elif key == "nations":
        settings.nations = value
        settings.preset = "custom"
        settings.game_type = infer_or_keep_mixed(settings)
    elif key == "ckpt":
        settings.ckpt = str(value)
    elif key == "greedy":
        settings.greedy = bool(value)
    elif key == "speed":
        settings.speed = snap_speed(float(value))
    elif key == "display":
        settings.display = int(value)
    else:
        return f"Cannot apply a list choice to {key}."
    return None


def current_option_index(rt: Runtime, key: str, options: list[tuple[object, str]]) -> int:
    current = getattr(rt.settings, key)
    for i, (value, _) in enumerate(options):
        if value == current:
            return i
    shown = field_value(rt.settings, key)
    for i, (_, label) in enumerate(options):
        if label == shown:
            return i
    return 0


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


def dropdown_menu_geometry(
    anchor: tuple[int, int, int, int],
    n_options: int,
    row_h: int,
    screen_size: tuple[int, int],
    max_ratio: float = 0.45,
) -> tuple[int, int, int, int]:
    """Place a menu below the field, flipping or clamping so it stays onscreen."""
    ax, ay, aw, ah = (int(v) for v in anchor)
    sw, sh = screen_size
    pad = 8
    row_h = max(1, int(row_h))
    menu_w = max(aw, 180)
    max_h = max(row_h, min(int(sh * max_ratio), max(row_h, sh - 2 * pad)))
    menu_h = min(max(n_options, 1) * row_h, max_h)
    x = ax
    y = ay + ah + 2
    if y + menu_h > sh - pad:
        above = ay - menu_h - 2
        if above >= pad:
            y = above
        else:
            y = max(pad, sh - menu_h - pad)
            menu_h = min(menu_h, max(row_h, sh - y - pad))
    if x + menu_w > sw - pad:
        x = max(pad, sw - menu_w - pad)
    if x < pad:
        x = pad
        menu_w = min(menu_w, max(1, sw - 2 * pad))
    if y < pad:
        y = pad
        menu_h = min(menu_h, max(row_h, sh - 2 * pad))
    return (int(x), int(y), int(menu_w), int(max(row_h, menu_h)))


def close_dropdown(rt: Runtime) -> None:
    rt.dropdown_key = None
    rt.dropdown_options = []
    rt.dropdown_menu = None
    rt.dropdown_index = 0
    rt.dropdown_scroll = 0


def open_dropdown(
    rt: Runtime,
    key: str,
    anchor: tuple[int, int, int, int],
    screen_size: tuple[int, int],
    row_h: int = 28,
) -> None:
    options = field_options(rt, key)
    if not options:
        close_dropdown(rt)
        rt.error = f"No choices available for {key}."
        return
    rt.dropdown_key = key
    rt.dropdown_options = options
    rt.dropdown_index = current_option_index(rt, key, options)
    rt.dropdown_menu = dropdown_menu_geometry(anchor, len(options), row_h, screen_size)
    visible = max(1, rt.dropdown_menu[3] // row_h)
    rt.dropdown_scroll = max(0, min(rt.dropdown_index, max(0, len(options) - visible)))
    keep_dropdown_highlight_visible(rt, row_h)


def keep_dropdown_highlight_visible(rt: Runtime, row_h: int = 28) -> None:
    if rt.dropdown_menu is None:
        return
    visible = max(1, rt.dropdown_menu[3] // max(1, row_h))
    n = len(rt.dropdown_options)
    rt.dropdown_scroll = max(0, min(rt.dropdown_scroll, max(0, n - visible)))
    if rt.dropdown_index < rt.dropdown_scroll:
        rt.dropdown_scroll = rt.dropdown_index
    elif rt.dropdown_index >= rt.dropdown_scroll + visible:
        rt.dropdown_scroll = rt.dropdown_index - visible + 1


def select_dropdown_index(rt: Runtime, index: int) -> None:
    if rt.dropdown_key is None or not rt.dropdown_options:
        return
    index = max(0, min(index, len(rt.dropdown_options) - 1))
    value, _ = rt.dropdown_options[index]
    err = apply_choice(rt, rt.dropdown_key, value)
    close_dropdown(rt)
    if err:
        rt.error = err
    else:
        rt.error = ""


def start_session(rt: Runtime, display: DisplayState | None = None) -> None:
    close_dropdown(rt)
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
    rt.replay_kind = "buffer"
    rt.pending_seek = None
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


def replay_timeline_count(rt: Runtime) -> int:
    if rt.tape is None:
        return 0
    if rt.tape.has_frames:
        return rt.tape.recorded_count
    return max(1, len(rt.tape.actions))


def replay_playhead(rt: Runtime) -> int:
    if rt.pending_seek is not None:
        return rt.pending_seek
    if rt.tape and rt.tape.has_frames and rt.replay_kind == "buffer":
        return rt.replay_index
    if rt.tape and not rt.tape.has_frames:
        return min(rt.replay_action_i, max(0, len(rt.tape.actions) - 1))
    return rt.replay_index


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
    rt.pending_seek = None
    if rt.tape.has_frames:
        show_replay_frame(rt, rt.tape.first_kept)
        if rt.tape.dropped_count:
            rt.last_status = (
                f"{rt.last_status}  (earliest buffered frame; "
                f"{rt.tape.dropped_count} earlier frame(s) evicted)"
            )
        rt.mode = "replay"
        return
    if rt.env is None:
        rt.error = "Env server is not running; cannot action-replay."
        rt.mode = "post"
        rt.hold_frame = False
        return
    try:
        rt.replay_kind = "rerun"
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
    assert rt.tape is not None
    frame = rt.tape.frame_at(index)
    if frame is None:
        return
    rt.replay_index = int(index)
    rt.replay_kind = "buffer"
    rt.last_frame = frame
    total = rt.tape.recorded_count
    rt.last_status = frame.get("status") or f"replay {rt.replay_index + 1}/{total}"
    rt.map_surf = None
    rt.live_done = total > 0 and rt.replay_index >= total - 1


def seek_replay(rt: Runtime, requested: int, *, commit_rebuild: bool = True) -> None:
    if rt.tape is None:
        return
    index, mode = resolve_seek_target(rt.tape, requested, rt.env is not None)
    if mode == "buffer":
        show_replay_frame(rt, index)
        rt.mode = "replay"
        rt.live_done = False
        rt.hold_frame = True
        rt.pending_seek = None
        if rt.playback:
            rt.playback.paused = True
    elif mode == "clamp":
        show_replay_frame(rt, index)
        rt.mode = "replay"
        rt.live_done = False
        rt.hold_frame = True
        rt.pending_seek = None
        rt.last_status = (
            f"Earlier frames were evicted ({rt.tape.dropped_count} frames). "
            f"Instant seek starts at frame {rt.tape.first_kept + 1}/"
            f"{rt.tape.recorded_count}."
        )
        if rt.playback:
            rt.playback.paused = True
    elif mode in ("rebuild", "actions") and not commit_rebuild:
        rt.pending_seek = index
    elif mode == "rebuild":
        rebuild_replay_to(rt, index)
    elif mode == "actions":
        rebuild_action_replay_to(rt, index)


def rebuild_replay_to(rt: Runtime, target_index: int) -> None:
    if rt.env is None or rt.tape is None:
        seek_replay_clamp_only(rt, target_index)
        return
    try:
        rt.replay_obs = rt.env.init([rt.tape.env_config])
        rt.replay_action_i = 0
        rt.replay_kind = "rerun"
        rt.last_frame = rt.env.frame(0)
        rt.last_status = "Rebuilding replay from recorded actions..."
        rt.map_surf = None
        rt.mode = "replay"
        rt.error = ""
        rendered = 0
        skip = rt.settings.frame_skip
        while rt.replay_action_i < len(rt.tape.actions):
            acts = rt.tape.actions[rt.replay_action_i]
            batch = rt.env.step(acts[None, :])
            rt.replay_obs = batch.obs
            rt.replay_action_i += 1
            done = bool(batch.dones[0]) or rt.replay_action_i >= len(rt.tape.actions)
            if should_render_frame(rt.replay_action_i, done, skip):
                frame = rt.env.frame(0)
                rt.last_frame = frame
                rt.last_status = (
                    _obs_status(
                        rt.episode,
                        rt.replay_action_i,
                        acts,
                        rt.replay_obs,
                        tick=frame["tick"],
                    )
                    + "  (rebuilt from actions)"
                )
                rt.map_surf = None
                if rendered >= target_index:
                    rt.replay_index = rendered
                    rt.live_done = done
                    rt.hold_frame = True
                    rt.pending_seek = None
                    if rt.playback:
                        rt.playback.paused = True
                    return
                rendered += 1
        rt.replay_index = max(0, rendered - 1) if rendered else 0
        rt.hold_frame = True
        rt.pending_seek = None
        if rt.playback:
            rt.playback.paused = True
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"


def rebuild_action_replay_to(rt: Runtime, action_index: int) -> None:
    if rt.env is None or rt.tape is None:
        rt.error = "Env server is not running; cannot rebuild the action tape."
        return
    try:
        rt.replay_obs = rt.env.init([rt.tape.env_config])
        rt.replay_action_i = 0
        rt.replay_kind = "rerun"
        rt.last_frame = rt.env.frame(0)
        rt.last_status = "Replay (recorded actions, policy not run)"
        rt.map_surf = None
        rt.mode = "replay"
        target = max(0, min(int(action_index), len(rt.tape.actions)))
        skip = rt.settings.frame_skip
        while rt.replay_action_i < target:
            acts = rt.tape.actions[rt.replay_action_i]
            batch = rt.env.step(acts[None, :])
            rt.replay_obs = batch.obs
            rt.replay_action_i += 1
            done = bool(batch.dones[0]) or rt.replay_action_i >= len(rt.tape.actions)
            if should_render_frame(rt.replay_action_i, done, skip):
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
        rt.replay_index = rt.replay_action_i
        rt.hold_frame = True
        rt.pending_seek = None
        rt.live_done = rt.replay_action_i >= len(rt.tape.actions)
        if rt.playback:
            rt.playback.paused = True
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"


def seek_replay_clamp_only(rt: Runtime, requested: int) -> None:
    if rt.tape and rt.tape.has_frames:
        show_replay_frame(rt, max(rt.tape.first_kept, min(requested, rt.tape.last_kept)))
        rt.last_status = (
            f"Earlier frames were evicted ({rt.tape.dropped_count} frames). "
            "Rebuild needs the env server."
        )


def advance_replay(rt: Runtime) -> None:
    rt.did_render = False
    if rt.tape is None:
        return
    if rt.replay_kind == "buffer" and rt.tape.has_frames:
        next_i = rt.replay_index + 1
        if rt.tape.can_seek_instant(next_i):
            show_replay_frame(rt, next_i)
            rt.did_render = True
        elif next_i >= rt.tape.recorded_count:
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
            if rt.tape.has_frames:
                rt.replay_index += 1
                if rt.tape.can_seek_instant(rt.replay_index):
                    show_replay_frame(rt, rt.replay_index)
                    rt.replay_kind = "buffer"
            else:
                rt.replay_index = rt.replay_action_i
        if done:
            rt.live_done = True
            rt.mode = "post"
            if rt.playback:
                rt.playback.paused = True
    except Exception as exc:
        rt.error = f"{type(exc).__name__}: {exc}"
        rt.mode = "post"


def retreat_replay(rt: Runtime) -> None:
    if rt.tape and rt.tape.can_seek_instant(rt.replay_index - 1):
        show_replay_frame(rt, rt.replay_index - 1)
        rt.mode = "replay"
        rt.live_done = False
        rt.hold_frame = True
        if rt.playback:
            rt.playback.paused = True


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
    elif key == pygame.K_PAGEUP and rt.mode == "replay":
        count = replay_timeline_count(rt)
        seek_replay(rt, timeline_jump(replay_playhead(rt), count, -max(1, count // 10)))
    elif key == pygame.K_PAGEDOWN and rt.mode == "replay":
        count = replay_timeline_count(rt)
        seek_replay(rt, timeline_jump(replay_playhead(rt), count, max(1, count // 10)))
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
    elif key == pygame.K_END and rt.mode == "replay":
        seek_replay(rt, max(0, replay_timeline_count(rt) - 1))
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
        if rt.tape.replay_mode == "window":
            kind = "REPLAY window"
        elif rt.tape.has_frames and rt.replay_kind == "buffer":
            kind = "REPLAY frames"
        else:
            kind = "REPLAY actions"
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
    help_w = fonts["tiny"].size("F11/Alt+Enter windowed   Esc menu   Q quit")[0]
    _blit_text(
        screen,
        fonts["tiny"],
        "F11/Alt+Enter windowed   Esc menu   Q quit",
        (max(12, top.right - help_w - 16), top.y + 12),
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
        status[:160],
        (12, bottom.y + 8),
        DANGER if rt.error else TEXT,
    )
    if rt.mode != "replay":
        hint = "Space pause  [ ] speed  . step  wheel/+- zoom  arrows/WASD pan  drag pan  0 recenter"
        _blit_text(screen, fonts["tiny"], hint, (12, bottom.y + 32), MUTED)


def draw_replay_controls(pygame, screen, fonts, layout, rt: Runtime, mouse) -> list:
    hits = []
    bottom = pygame.Rect(*layout["bottom"])
    track = pygame.Rect(*layout["timeline"])
    if track.h < 8:
        track.h = 16
        track.y = bottom.bottom - 26
    play = pygame.Rect(16, bottom.y + 30, 78, 28)
    paused = bool(rt.playback and rt.playback.paused)
    play_hover = play.collidepoint(mouse)
    _draw_button(
        pygame,
        screen,
        fonts["small"],
        play,
        "Play" if paused else "Pause",
        kind="accent",
        hover=play_hover,
    )
    hits.append(("replay_play", play))

    speed = rt.playback.speed if rt.playback else rt.settings.speed
    chip_x = play.right + 12
    for step in SPEED_STEPS:
        chip = pygame.Rect(chip_x, play.y, 54, 28)
        selected = abs(step - speed) < 1e-9
        _draw_button(
            pygame,
            screen,
            fonts["tiny"],
            chip,
            f"{step:g}x",
            kind="accent" if selected else "default",
            hover=chip.collidepoint(mouse),
        )
        hits.append(("replay_speed", (step, chip)))
        chip_x = chip.right + 6

    count = replay_timeline_count(rt)
    index = replay_playhead(rt)
    if rt.tape and rt.tape.has_frames:
        label = f"frame {index + 1}/{count}"
        if rt.tape.dropped_count:
            label += (
                f"   buffered {rt.tape.first_kept + 1}-{rt.tape.last_kept + 1}"
                f"   evicted {rt.tape.dropped_count}"
            )
        if rt.replay_kind == "rerun":
            label += "   rebuilt"
    elif rt.tape:
        label = f"action {rt.replay_action_i}/{len(rt.tape.actions)}   rerun only"
    else:
        label = "no tape"
    _blit_text(screen, fonts["tiny"], label, (chip_x + 8, play.y + 6), MUTED)

    _fill_rect(pygame, screen, track, TRACK, 1, (50, 56, 70))
    if count > 0 and track.w > 2:
        if rt.tape and rt.tape.has_frames:
            evicted_w = int(track.w * (rt.tape.dropped_count / count)) if count else 0
            if evicted_w:
                evicted = pygame.Rect(track.x, track.y, evicted_w, track.h)
                _fill_rect(pygame, screen, evicted, EVICTED)
            kept_x = track.x + evicted_w
            kept_w = max(0, track.w - evicted_w)
            if kept_w:
                _fill_rect(pygame, screen, pygame.Rect(kept_x, track.y, kept_w, track.h), BUFFERED)
            played = timeline_thumb_frac(index, count)
            played_w = max(0, min(track.w, int(track.w * played)))
            if played_w:
                color = ACCENT if rt.tape.can_seek_instant(index) else RERUN
                _fill_rect(pygame, screen, pygame.Rect(track.x, track.y, played_w, track.h), color)
        else:
            played_w = int(track.w * timeline_thumb_frac(index, count))
            _fill_rect(pygame, screen, pygame.Rect(track.x, track.y, max(4, played_w), track.h), RERUN)
        thumb_x = track.x + int(track.w * timeline_thumb_frac(index, count))
        pygame.draw.circle(screen, TEXT, (thumb_x, track.centery), 7)
        pygame.draw.circle(screen, ACCENT, (thumb_x, track.centery), 5)
    hits.append(("replay_timeline", track))
    hint = (
        "Drag timeline to seek   click evicted region to rebuild   "
        "Space play/pause   [ ] speed   , . step   PgUp/PgDn jump   End last"
    )
    _blit_text(screen, fonts["tiny"], hint, (16, track.bottom + 4), MUTED)
    return hits


def draw_config(pygame, screen, fonts, rt: Runtime) -> list[tuple[str, object]]:
    """Draw the configuration form. Returns hit targets (name, payload)."""
    w, h = screen.get_size()
    metrics = ui_metrics((w, h))
    mouse = pygame.mouse.get_pos()
    screen.fill(BG)
    margin = metrics["margin"]
    _blit_text(screen, fonts["title"], "OFAI watch", (margin, int(22 * metrics["scale"])), TEXT)
    _blit_text(
        screen,
        fonts["small"],
        "Fullscreen match viewer. Click a field to open its dropdown, type to edit "
        "free-form values, then Start.",
        (margin, int(58 * metrics["scale"])),
        MUTED,
    )
    hits: list[tuple[str, object]] = []
    rt.field_rects = {}
    footer_h = metrics["btn_h"] + 70
    start_y = int(92 * metrics["scale"]) - rt.scroll
    col_gutter = metrics["gutter"]
    content_w = w - 2 * margin
    two = metrics["two_col"]
    col_w = (content_w - col_gutter) // 2 if two else content_w
    row_h = metrics["row_h"]
    field_h = metrics["field_h"]
    y = start_y
    max_y = start_y

    for title, keys in CONFIG_SECTIONS:
        _blit_text(screen, fonts["heading"], title, (margin, y), ACCENT)
        y += int(28 * metrics["scale"])
        for i, key in enumerate(keys):
            label = next(lbl for k, lbl, _ in CONFIG_FIELDS if k == key)
            index = FIELD_INDEX[key]
            col = i % 2 if two else 0
            row_y = y
            x = margin + col * (col_w + col_gutter)
            focused = index == rt.focus
            box = pygame.Rect(x, row_y + metrics["label_h"], col_w, field_h)
            hovered = box.collidepoint(mouse)
            label_color = ACCENT if focused else (TEXT if hovered else MUTED)
            _blit_text(screen, fonts["tiny"], label, (x, row_y), label_color)
            kind = field_kind(key)
            bg = PANEL_HI if focused or hovered else PANEL
            border = ACCENT if focused else ((80, 96, 120) if hovered else (50, 56, 70))
            _fill_rect(pygame, screen, box, bg, 1, border)
            shown = rt.draft if focused and rt.draft is not None else field_value(rt.settings, key)
            text_w = box.w - (64 if kind in ("dropdown", "combo") else 78 if kind == "stepper" else 16)
            tail = key in ("ckpt", "map")
            _blit_text(
                screen,
                fonts["body"],
                _ellipsize(fonts["body"], shown, max(20, text_w), tail=tail),
                (box.x + 10, box.y + max(6, (field_h - 18) // 2)),
                TEXT,
            )
            rt.field_rects[key] = (box.x, box.y, box.w, box.h)
            hits.append(("field", (index, box)))
            if kind in ("dropdown", "combo"):
                chevron = pygame.Rect(box.right - 34, box.y + 4, 30, field_h - 8)
                _fill_rect(
                    pygame,
                    screen,
                    chevron,
                    PANEL_HOVER if chevron.collidepoint(mouse) else PANEL_HI,
                    1,
                    ACCENT if rt.dropdown_key == key else (60, 68, 84),
                )
                _blit_text(screen, fonts["small"], "v", (chevron.x + 10, chevron.y + 4), TEXT)
                hits.append(("chevron", (index, chevron)))
            elif kind == "stepper":
                plus = pygame.Rect(box.right - 32, box.y + 4, 28, field_h - 8)
                minus = pygame.Rect(plus.x - 30, box.y + 4, 28, field_h - 8)
                _draw_button(pygame, screen, fonts["heading"], minus, "-", hover=minus.collidepoint(mouse))
                _draw_button(pygame, screen, fonts["heading"], plus, "+", hover=plus.collidepoint(mouse))
                hits.append(("step", (key, -1, minus)))
                hits.append(("step", (key, 1, plus)))
            if not two or col == 1 or i == len(keys) - 1:
                y += row_h
        y += metrics["section_gap"]
        max_y = y

    btn_y = h - footer_h + 8
    start_btn = pygame.Rect(margin, btn_y, metrics["btn_w"], metrics["btn_h"])
    quit_btn = pygame.Rect(start_btn.right + 16, btn_y, metrics["btn_w"], metrics["btn_h"])
    _draw_button(
        pygame,
        screen,
        fonts["heading"],
        start_btn,
        "Start",
        kind="ok",
        hover=start_btn.collidepoint(mouse),
    )
    _draw_button(
        pygame,
        screen,
        fonts["heading"],
        quit_btn,
        "Quit",
        kind="danger",
        hover=quit_btn.collidepoint(mouse),
    )
    hits.append(("start", start_btn))
    hits.append(("quit", quit_btn))

    if rt.error:
        _blit_text(screen, fonts["body"], rt.error, (margin, btn_y + metrics["btn_h"] + 10), DANGER)
    _blit_text(
        screen,
        fonts["tiny"],
        f"{len(rt.checkpoints)} checkpoint(s) in {CHECKPOINT_ROOT.name}/   "
        f"{len(rt.maps)} map key(s)   Episodes is used by --save; in-window use New after a match",
        (quit_btn.right + 24, btn_y + 14),
        MUTED,
    )

    content_bottom = max_y + footer_h
    overflow = max(0, content_bottom - h)
    rt.scroll = max(0, min(rt.scroll, overflow))

    if rt.dropdown_key and rt.dropdown_menu:
        hits.extend(draw_dropdown_menu(pygame, screen, fonts, rt, mouse, metrics["field_h"] - 6))
    return hits


def draw_dropdown_menu(pygame, screen, fonts, rt: Runtime, mouse, row_h: int) -> list:
    hits = []
    mx, my, mw, mh = rt.dropdown_menu
    menu = pygame.Rect(mx, my, mw, mh)
    shadow = menu.move(3, 3)
    _fill_rect(pygame, screen, shadow, SHADOW)
    _fill_rect(pygame, screen, menu, PANEL, 1, ACCENT)
    row_h = max(22, int(row_h))
    visible = max(1, mh // row_h)
    keep_dropdown_highlight_visible(rt, row_h)
    options = rt.dropdown_options
    start = rt.dropdown_scroll
    end = min(len(options), start + visible)
    for draw_i, opt_i in enumerate(range(start, end)):
        _, label = options[opt_i]
        row = pygame.Rect(mx + 2, my + draw_i * row_h, mw - 4, row_h - 1)
        hovered = row.collidepoint(mouse)
        selected = opt_i == rt.dropdown_index
        bg = ACCENT_DIM if selected else (PANEL_HOVER if hovered else PANEL)
        _fill_rect(pygame, screen, row, bg)
        _blit_text(
            screen,
            fonts["small"],
            _ellipsize(fonts["small"], label, row.w - 16, tail=True),
            (row.x + 8, row.y + max(2, (row_h - 16) // 2)),
            TEXT,
        )
        hits.append(("option", (opt_i, row)))
        if hovered:
            rt.dropdown_index = opt_i
    if len(options) > visible:
        track = pygame.Rect(menu.right - 8, menu.y + 2, 6, menu.h - 4)
        _fill_rect(pygame, screen, track, TRACK)
        thumb_h = max(16, int(track.h * (visible / len(options))))
        max_scroll = max(1, len(options) - visible)
        thumb_y = track.y + int((track.h - thumb_h) * (rt.dropdown_scroll / max_scroll))
        _fill_rect(pygame, screen, pygame.Rect(track.x, thumb_y, track.w, thumb_h), ACCENT)
    hits.append(("dropdown_menu", menu))
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
    mouse = pygame.mouse.get_pos()
    box_w, box_h = 580, 80 + 46 * len(items)
    box = pygame.Rect((w - box_w) // 2, (h - box_h) // 2, box_w, box_h)
    shadow = box.move(4, 4)
    _fill_rect(pygame, screen, shadow, SHADOW)
    _fill_rect(pygame, screen, box, PANEL, 1, ACCENT)
    _blit_text(screen, fonts["heading"], title, (box.x + 24, box.y + 18), TEXT)
    hits = []
    for i, (key, label) in enumerate(items):
        row = pygame.Rect(box.x + 24, box.y + 62 + i * 42, box_w - 48, 36)
        hover = row.collidepoint(mouse)
        _fill_rect(
            pygame,
            screen,
            row,
            PANEL_HOVER if hover or i == selected else (18, 22, 30),
            1,
            ACCENT,
        )
        _blit_text(screen, fonts["body"], label, (row.x + 12, row.y + 8), TEXT)
        hits.append((key, row))
    if rt is not None:
        rt.menu_hits = hits
    return hits


def post_match_items(rt: Runtime) -> list[tuple[str, str]]:
    if rt.tape and rt.tape.replay_mode == "window":
        replay_note = (
            f"buffered {rt.tape.first_kept + 1}-{rt.tape.recorded_count}; "
            f"{rt.tape.dropped_count} evicted"
        )
    elif rt.tape and rt.tape.has_frames:
        replay_note = "from frames"
    else:
        replay_note = "from recorded actions"
    return [
        ("replay", f"1  Replay last match ({replay_note})"),
        ("restart", "2  Restart same seeded match"),
        ("new", "3  New seeded match"),
        ("configure", "4  Return to configuration"),
        ("quit", "5  Quit"),
    ]


def play_menu_items() -> list[tuple[str, str]]:
    return [
        ("resume", "Resume"),
        ("configure", "Return to configuration"),
        ("quit", "Quit"),
    ]


def draw_play_view(pygame, screen, fonts, rt: Runtime) -> dict[str, tuple[int, int, int, int]]:
    screen.fill(BG)
    layout = overlay_layout(screen.get_size(), replay=(rt.mode == "replay"))
    if rt.last_frame is not None and rt.camera is not None:
        ensure_map_surface(rt, pygame)
        if rt.map_surf is not None:
            map_size = rt.map_surf.get_size()
            view = layout["map"]
            transform = rt.camera.transform(map_size, (view[2], view[3]))
            present_map(pygame, screen, rt.map_surf, view, transform)
    draw_hud(pygame, screen, fonts, layout, rt)
    if rt.mode == "replay":
        rt.timeline_hits = draw_replay_controls(
            pygame, screen, fonts, layout, rt, pygame.mouse.get_pos()
        )
    else:
        rt.timeline_hits = None
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
        close_dropdown(rt)
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


def hit_replay_control(hits: list | None, pos: tuple[int, int]):
    if not hits:
        return None
    for kind, payload in reversed(hits):
        rect = payload[1] if kind == "replay_speed" else payload
        if hasattr(rect, "collidepoint") and rect.collidepoint(pos):
            return kind, payload
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
    metrics = ui_metrics(display.size())
    ui_fonts = _load_ui_fonts(pygame, metrics["scale"])
    last_ui_scale = metrics["scale"]
    rt.map_fonts = _load_fonts(pygame, settings.scale)
    last_map_scale = settings.scale
    clock = pygame.time.Clock()
    running = True
    config_hits: list[tuple[str, object]] = []

    try:
        while running and rt.mode != "quit":
            layout = overlay_layout(display.size(), replay=(rt.mode == "replay"))
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
                    elif rt.mode == "config":
                        handle_config_wheel(rt, event, display.size())
                    elif rt.mode in ("playing", "replay", "post") and rt.last_frame:
                        pos = pygame.mouse.get_pos()
                        if rt.mode == "replay" and layout["timeline"][3] and point_in(layout["timeline"], pos):
                            count = replay_timeline_count(rt)
                            seek_replay(rt, timeline_jump(replay_playhead(rt), count, -1 if event.y > 0 else 1))
                        elif point_in(map_view, pos):
                            sizes = camera_sizes(rt, map_view)
                            if sizes:
                                map_size, viewport_size, view = sizes
                                local = (pos[0] - map_view[0], pos[1] - map_view[1])
                                anchor = view.screen_to_map(*local)
                                zoom_camera(rt, 1.15 if event.y > 0 else 1 / 1.15, map_view, anchor)
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
                    elif rt.mode == "replay" and event.button == 1:
                        hit = hit_replay_control(rt.timeline_hits, pos)
                        if hit:
                            kind, payload = hit
                            if kind == "replay_play" and rt.playback:
                                rt.playback.toggle_pause()
                            elif kind == "replay_speed":
                                step, _ = payload
                                if rt.playback:
                                    rt.playback.speed_index = SPEED_STEPS.index(step)
                                    rt.settings.speed = step
                            elif kind == "replay_timeline":
                                rt.timeline_drag = True
                                if rt.playback:
                                    rt.playback.paused = True
                                apply_timeline_click(rt, payload, pos, dragging=True)
                        elif point_in(map_view, pos):
                            rt.dragging = True
                        elif event.button == 2:
                            reset_camera(rt)
                    elif event.button == 1 and point_in(map_view, pos):
                        rt.dragging = True
                    elif event.button == 2:
                        reset_camera(rt)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        if rt.timeline_drag:
                            track = None
                            if rt.timeline_hits:
                                for kind, payload in rt.timeline_hits:
                                    if kind == "replay_timeline":
                                        track = payload
                            if track is not None:
                                apply_timeline_click(rt, track, event.pos, dragging=False)
                            elif rt.pending_seek is not None:
                                seek_replay(rt, rt.pending_seek, commit_rebuild=True)
                            rt.timeline_drag = False
                        rt.dragging = False
                elif event.type == pygame.MOUSEMOTION and rt.timeline_drag:
                    track = None
                    if rt.timeline_hits:
                        for kind, payload in rt.timeline_hits:
                            if kind == "replay_timeline":
                                track = payload
                    if track is not None:
                        apply_timeline_click(rt, track, event.pos, dragging=True)
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
            metrics = ui_metrics(display.size())
            if metrics["scale"] != last_ui_scale:
                ui_fonts = _load_ui_fonts(pygame, metrics["scale"])
                last_ui_scale = metrics["scale"]

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


def apply_timeline_click(rt: Runtime, track, pos: tuple[int, int], *, dragging: bool) -> None:
    x, _y, w, _h = (track.x, track.y, track.w, track.h) if hasattr(track, "x") else track
    if w <= 0:
        return
    frac = (pos[0] - x) / float(w)
    count = replay_timeline_count(rt)
    requested = timeline_frame_index(frac, count)
    seek_replay(rt, requested, commit_rebuild=not dragging)


def handle_config_wheel(rt: Runtime, event, screen_size: tuple[int, int]) -> None:
    if rt.dropdown_key and rt.dropdown_menu:
        row_h = max(22, rt.dropdown_menu[3] // max(1, min(8, len(rt.dropdown_options) or 1)))
        visible = max(1, rt.dropdown_menu[3] // row_h)
        rt.dropdown_scroll = max(
            0,
            min(
                rt.dropdown_scroll - int(event.y),
                max(0, len(rt.dropdown_options) - visible),
            ),
        )
        keep_dropdown_highlight_visible(rt, row_h)
        return
    key = CONFIG_FIELDS[rt.focus][0]
    if field_kind(key) == "stepper":
        nudge_field(rt, key, 1 if event.y > 0 else -1)
        return
    rt.scroll = max(0, rt.scroll - int(event.y) * 36)


def handle_config_event(rt: Runtime, pygame, event, display: DisplayState | None = None) -> bool:
    """Return False when the user confirmed quit."""
    key = event.key
    screen_size = display.size() if display is not None else (1280, 720)
    row_h = 28
    if rt.dropdown_menu:
        row_h = max(22, rt.dropdown_menu[3] // max(1, min(12, len(rt.dropdown_options) or 1)))
    if key == pygame.K_ESCAPE:
        if rt.dropdown_key is not None:
            close_dropdown(rt)
            return True
        if rt.draft is not None:
            rt.draft = None
            return True
        rt.overlay = "quit"
        return True
    if rt.overlay == "quit":
        return True
    if rt.dropdown_key is not None:
        if key in (pygame.K_UP, pygame.K_LEFT):
            rt.dropdown_index = (rt.dropdown_index - 1) % len(rt.dropdown_options)
            keep_dropdown_highlight_visible(rt, row_h)
            return True
        if key in (pygame.K_DOWN, pygame.K_RIGHT):
            rt.dropdown_index = (rt.dropdown_index + 1) % len(rt.dropdown_options)
            keep_dropdown_highlight_visible(rt, row_h)
            return True
        if key == pygame.K_HOME:
            rt.dropdown_index = 0
            keep_dropdown_highlight_visible(rt, row_h)
            return True
        if key == pygame.K_END:
            rt.dropdown_index = len(rt.dropdown_options) - 1
            keep_dropdown_highlight_visible(rt, row_h)
            return True
        if key in (pygame.K_PAGEUP, pygame.K_PAGEDOWN):
            visible = max(1, (rt.dropdown_menu or (0, 0, 0, 28))[3] // row_h)
            step = -visible if key == pygame.K_PAGEUP else visible
            rt.dropdown_index = max(0, min(len(rt.dropdown_options) - 1, rt.dropdown_index + step))
            keep_dropdown_highlight_visible(rt, row_h)
            return True
        if key == pygame.K_RETURN:
            select_dropdown_index(rt, rt.dropdown_index)
            return True
        if key == pygame.K_TAB:
            close_dropdown(rt)
        else:
            return True
    if key in (pygame.K_TAB, pygame.K_DOWN):
        commit_draft(rt)
        key_name = CONFIG_FIELDS[rt.focus][0]
        if (
            key == pygame.K_DOWN
            and field_kind(key_name) in ("dropdown", "combo")
            and rt.draft is None
            and not (event.mod & pygame.KMOD_SHIFT)
        ):
            anchor = rt.field_rects.get(key_name)
            if anchor:
                open_dropdown(rt, key_name, anchor, screen_size)
                return True
        step = -1 if (event.mod & pygame.KMOD_SHIFT) and key == pygame.K_TAB else 1
        rt.focus = (rt.focus + step) % len(CONFIG_FIELDS)
        return True
    if key == pygame.K_UP:
        commit_draft(rt)
        rt.focus = (rt.focus - 1) % len(CONFIG_FIELDS)
        return True
    if key in (pygame.K_LEFT, pygame.K_RIGHT):
        if rt.draft is None and field_kind(CONFIG_FIELDS[rt.focus][0]) == "stepper":
            nudge_field(rt, CONFIG_FIELDS[rt.focus][0], 1 if key == pygame.K_RIGHT else -1)
        return True
    if key == pygame.K_SPACE and rt.draft is None:
        key_name = CONFIG_FIELDS[rt.focus][0]
        if field_kind(key_name) in ("dropdown", "combo"):
            if rt.dropdown_key == key_name:
                close_dropdown(rt)
            else:
                anchor = rt.field_rects.get(key_name)
                if anchor:
                    open_dropdown(rt, key_name, anchor, screen_size)
            return True
    if key == pygame.K_RETURN:
        commit_draft(rt)
        start_session(rt, display)
        return True
    if key == pygame.K_BACKSPACE:
        key_name = CONFIG_FIELDS[rt.focus][0]
        if key_name not in EDITABLE_TEXT:
            return True
        close_dropdown(rt)
        if rt.draft is None:
            rt.draft = field_value(rt.settings, key_name)
        rt.draft = rt.draft[:-1]
        return True
    if event.unicode and event.unicode.isprintable():
        key_name = CONFIG_FIELDS[rt.focus][0]
        if key_name in EDITABLE_TEXT:
            close_dropdown(rt)
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
    screen_size = display.size() if display is not None else (1280, 720)
    if rt.dropdown_key is not None:
        for kind, payload in reversed(hits):
            if kind == "option":
                index, box = payload
                if box.collidepoint(pos):
                    select_dropdown_index(rt, index)
                    return
            if kind == "dropdown_menu" and payload.collidepoint(pos):
                return
        close_dropdown(rt)
    for kind, payload in hits:
        if kind == "field":
            index, box = payload
            if box.collidepoint(pos):
                commit_draft(rt)
                rt.focus = index
                key_name = CONFIG_FIELDS[index][0]
                if field_kind(key_name) in ("dropdown", "combo"):
                    anchor = rt.field_rects.get(key_name) or (box.x, box.y, box.w, box.h)
                    open_dropdown(rt, key_name, anchor, screen_size)
                return
        elif kind == "chevron":
            index, box = payload
            if box.collidepoint(pos):
                commit_draft(rt)
                rt.focus = index
                key_name = CONFIG_FIELDS[index][0]
                if rt.dropdown_key == key_name:
                    close_dropdown(rt)
                else:
                    anchor = rt.field_rects.get(key_name) or (box.x, box.y, box.w, box.h)
                    open_dropdown(rt, key_name, anchor, screen_size)
                return
        elif kind == "step":
            key_name, delta, box = payload
            if box.collidepoint(pos):
                commit_draft(rt)
                rt.focus = FIELD_INDEX[key_name]
                nudge_field(rt, key_name, delta)
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
