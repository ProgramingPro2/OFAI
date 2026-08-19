"""Focused tests for agent.watch helpers (no display, no live env).

Run: .venv/bin/python -m pytest tests/test_watch.py -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.watch import (
    Camera,
    PlaybackControl,
    ReplayTape,
    SPEED_STEPS,
    WatchSettings,
    build_palette,
    close_video_writer,
    compute_view_transform,
    desktop_size,
    discover_checkpoints,
    discover_maps,
    episode_result,
    episode_video_path,
    fit_size,
    infer_game_type,
    open_video_writer,
    overlay_layout,
    parse_args,
    parse_game_map_keys,
    parse_nations_value,
    player_centroids,
    present_map,
    positive_float,
    positive_int,
    restore_policy_rng,
    scene_size,
    seed_policy_rng,
    should_render_frame,
    snap_speed,
    snapshot_policy_rng,
    usable_display_size,
    viewport_rect,
    window_size_for_scene,
    _draw_scene,
    _load_fonts,
)


def test_positive_int_accepts_one_and_up():
    assert positive_int("1") == 1
    assert positive_int("5") == 5


def test_positive_int_rejects_non_positive():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_int("-2")


def test_parse_args_frame_skip_default():
    args = parse_args(["--ckpt", "x.pt"])
    assert args.frame_skip == 1
    assert args.greedy is False
    assert args.map == "FourIslands"
    assert args.map_size == "Normal"
    assert args.nations == "disabled"
    assert args.difficulty == "Easy"
    assert args.bots == 20
    assert args.seed == "watch"
    assert args.policy_seed == 0
    assert args.display == 0
    assert args.window_size is None
    assert args.windowed is False
    assert args.speed == 1.0


def test_parse_args_frame_skip_custom():
    args = parse_args(
        [
            "--ckpt",
            "x.pt",
            "--frame-skip",
            "5",
            "--fps",
            "60",
            "--display",
            "1",
            "--window-size",
            "1280x720",
        ]
    )
    assert args.frame_skip == 5
    assert args.fps == 60.0
    assert args.display == 1
    assert args.window_size == (1280, 720)


def test_parse_args_frame_skip_rejects_zero():
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "x.pt", "--frame-skip", "0"])


def test_parse_args_curriculum_preset_and_explicit_override():
    args = parse_args(
        [
            "--ckpt",
            "x.pt",
            "--preset",
            "nations-medium-mid",
            "--difficulty",
            "Hard",
            "--bots",
            "12",
        ]
    )
    assert args.map == "Halkidiki"
    assert args.map_size == "Normal"
    assert args.nations == 12
    assert args.difficulty == "Hard"
    assert args.bots == 12
    assert args.max_ticks == 14000


@pytest.mark.parametrize(
    ("flag", "opponents", "nations", "bots"),
    [
        ("--opponents", "tribes", "disabled", 20),
        ("--game-type", "nations", "default", 0),
        ("--opponents", "mixed", "default", 20),
    ],
)
def test_parse_args_opponent_types(flag, opponents, nations, bots):
    args = parse_args(["--ckpt", "x.pt", flag, opponents])
    assert args.nations == nations
    assert args.bots == bots


def test_parse_args_explicit_opponent_counts_override_shorthand():
    args = parse_args(
        [
            "--ckpt",
            "x.pt",
            "--opponents",
            "nations",
            "--nations",
            "8",
            "--bots",
            "4",
        ]
    )
    assert args.nations == 8
    assert args.bots == 4


@pytest.mark.parametrize(
    "extra",
    [
        ["--difficulty", "Extreme"],
        ["--map-size", "Tiny"],
        ["--nations", "-1"],
        ["--nations", "many"],
        ["--bots", "401"],
        ["--episodes", "0"],
        ["--display", "-1"],
        ["--window-size", "1280"],
        ["--window-size", "0x720"],
        ["--speed", "0"],
        ["--speed", "-1"],
        ["--fps", "0"],
        ["--fps", "-1"],
        ["--port", "0"],
        ["--map", ""],
    ],
)
def test_parse_args_rejects_unsupported_or_out_of_range_values(extra):
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "x.pt", *extra])


def test_episode_video_path_preserves_multi_episode_names():
    assert episode_video_path("match.mp4", 0) == "match.mp4"
    assert episode_video_path("match.mp4", 1) == "match-ep1.mp4"
    assert episode_video_path("out.mp4", 2) == "out-ep2.mp4"


def test_should_render_every_decision_when_skip_is_one():
    assert should_render_frame(1, False, 1)
    assert should_render_frame(2, False, 1)
    assert should_render_frame(3, True, 1)


def test_should_render_first_interval_and_terminal():
    assert should_render_frame(1, False, 5)
    assert not should_render_frame(2, False, 5)
    assert not should_render_frame(5, False, 5)
    assert should_render_frame(6, False, 5)
    assert should_render_frame(3, True, 5)
    assert not should_render_frame(0, False, 5)


def test_player_centroids_vectorized_and_ignores_fallout():
    frame = {
        "cells": np.array(
            [
                [0, 3, 3, 250],
                [2, 4, 4, 250],
                [2, 2, 4, 250],
            ],
            dtype=np.uint8,
        )
    }
    assert player_centroids(frame) == {
        0: (1.5, 0.0),
        1: (5.0 / 3.0, 4.0 / 3.0),
    }


def test_palette_supports_valid_high_player_counts():
    palette = build_palette(401)
    assert palette.shape == (404, 3)
    assert tuple(palette[250]) == (200, 40, 200)


def test_scene_size_accounts_for_scale_legend_and_status():
    assert scene_size(600, 400, 2) == (1430, 830)


def test_auto_window_fits_both_dimensions_without_upscaling():
    assert window_size_for_scene((2000, 1000), (1200, 700), None) == (1200, 600)
    assert window_size_for_scene((400, 300), (1200, 700), None) == (400, 300)


def test_explicit_window_is_respected_but_clamped_to_usable_display():
    assert window_size_for_scene((2000, 1000), (1200, 700), (900, 650)) == (
        900,
        650,
    )
    assert window_size_for_scene((2000, 1000), (1200, 700), (1600, 900)) == (
        1200,
        700,
    )


def test_viewport_letterboxes_without_distorting_scene():
    assert viewport_rect((2000, 1000), (900, 650)) == (0, 100, 900, 450)
    assert viewport_rect((1000, 2000), (900, 650)) == (287, 0, 325, 650)


def test_fit_size_rejects_invalid_dimensions():
    with pytest.raises(ValueError):
        fit_size((0, 100), (800, 600))


def test_usable_display_size_reserves_desktop_chrome():
    class FakeDisplay:
        @staticmethod
        def get_desktop_sizes():
            return [(1920, 1080), (2560, 1440)]

    class FakePygame:
        display = FakeDisplay()

    assert usable_display_size(FakePygame(), 0) == (1888, 984)
    assert usable_display_size(FakePygame(), 1) == (2528, 1344)
    with pytest.raises(ValueError, match="display 2 is unavailable"):
        usable_display_size(FakePygame(), 2)


def test_close_video_writer_none_is_safe():
    close_video_writer(None)


def test_video_writer_lifecycle_streams_and_closes(tmp_path):
    path = tmp_path / "tiny.mp4"
    writer = open_video_writer(str(path), fps=8)
    assert writer is not None
    try:
        frame = np.zeros((64, 64, 3), dtype=np.uint8)
        frame[:, :] = (20, 40, 80)
        writer.append_data(frame)
        frame2 = np.zeros((64, 64, 3), dtype=np.uint8)
        frame2[:, :] = (80, 40, 20)
        writer.append_data(frame2)
    finally:
        close_video_writer(writer)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_parse_args_windowed_and_speed():
    args = parse_args(["--ckpt", "x.pt", "--windowed", "--speed", "3.6"])
    assert args.windowed is True
    assert args.speed == 3.6
    settings = WatchSettings.from_args(args)
    assert settings.windowed is True
    assert settings.speed == 4.0


def test_parse_args_window_size_implies_windowed_settings():
    args = parse_args(["--ckpt", "x.pt", "--window-size", "1280x720"])
    settings = WatchSettings.from_args(args)
    assert settings.windowed is True
    assert settings.window_size == (1280, 720)


def test_positive_float_rejects_non_positive():
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("0")
    with pytest.raises(argparse.ArgumentTypeError):
        positive_float("-2.5")


def test_snap_speed_ladder():
    assert snap_speed(1) == 1.0
    assert snap_speed(3) == 2.0
    assert snap_speed(3.1) == 4.0
    assert snap_speed(0.3) == 0.25
    with pytest.raises(ValueError):
        snap_speed(0)


def test_overlay_layout_is_independent_of_zoom_and_covers_window():
    first = overlay_layout((1920, 1080))
    second = overlay_layout((1920, 1080))
    assert first == second
    top, bottom, legend, mapped = first["top"], first["bottom"], first["legend"], first["map"]
    assert mapped[0] == 0
    assert mapped[1] == top[3]
    assert legend[0] == mapped[2]
    assert mapped[1] + mapped[3] == bottom[1]
    assert top[2] == 1920
    assert mapped[2] + legend[2] == 1920
    assert top[3] + mapped[3] + bottom[3] == 1080


def test_view_transform_preserves_aspect_and_letterboxes_at_fit_zoom():
    view = compute_view_transform((2000, 1000), (800, 800), 1.0, (1000, 500))
    assert view.dest_w == 800
    assert view.dest_h == 400
    assert view.dest_y == 200
    assert abs(view.dest_w / view.dest_h - 2.0) < 1e-6
    zoomed = compute_view_transform((2000, 1000), (800, 800), 2.0, (1000, 500))
    assert abs(zoomed.src_w / zoomed.src_h - zoomed.dest_w / zoomed.dest_h) < 1e-6


def test_camera_zoom_keeps_anchor_under_the_same_screen_point():
    camera = Camera()
    camera.reset(2000, 1000)
    map_size = (2000, 1000)
    viewport = (800, 400)
    before = camera.transform(map_size, viewport)
    anchor = (1500.0, 250.0)
    screen = before.map_to_screen(*anchor)
    camera.set_zoom(2.0, map_size, viewport, anchor)
    after = camera.transform(map_size, viewport)
    moved = after.map_to_screen(*anchor)
    assert camera.zoom == 2.0
    assert abs(screen[0] - moved[0]) < 2.0
    assert abs(screen[1] - moved[1]) < 2.0


def test_camera_pan_and_recenter():
    camera = Camera()
    camera.reset(400, 300)
    camera.set_zoom(4.0, (400, 300), (200, 150))
    camera.pan(50, -20, (400, 300), (200, 150))
    assert camera.zoom == 4.0
    camera.reset(400, 300)
    assert camera.zoom == 1.0
    assert camera.center_x == 200
    assert camera.center_y == 150


def test_playback_pause_step_and_speed_do_not_invent_actions():
    control = PlaybackControl(1.0)
    assert control.speed == 1.0
    assert control.should_advance() is True
    control.toggle_pause()
    assert control.paused is True
    assert control.should_advance() is False
    control.request_step()
    assert control.should_advance() is True
    assert control.should_advance() is False
    control.faster()
    control.faster()
    assert control.speed == 4.0
    control.slower()
    assert control.speed == 2.0
    assert control.render_interval_ms(10) == 50
    assert PlaybackControl(1.0).render_interval_ms(10) == 100


def test_replay_tape_keeps_actions_and_drops_frames_at_the_cap():
    tape = ReplayTape(max_bytes=8_000, max_frames=4)
    frame = {
        "width": 40,
        "height": 30,
        "cells": np.zeros((30, 40), dtype=np.uint8),
        "players": [{"idx": 0, "name": "A", "tiles": 1, "alive": True, "isAgent": True}],
        "units": [],
        "tick": 0,
    }
    for step in range(12):
        tape.record_action(np.array([0, 0, 0, 0, 0], dtype=np.int32))
        tape.record_frame(frame, f"step {step}")
    assert len(tape.actions) == 12
    assert tape.frames is None
    assert tape.has_frames is False
    assert tape.replay_mode == "actions"
    assert tape.bytes_used == 0


def test_replay_tape_keeps_compact_frames_under_the_cap():
    tape = ReplayTape(max_bytes=1_000_000, max_frames=8)
    frame = {
        "width": 8,
        "height": 6,
        "cells": np.arange(48, dtype=np.uint8).reshape(6, 8),
        "players": [{"idx": 0, "name": "A", "tiles": 3, "alive": True, "isAgent": True}],
        "units": [{"x": 1, "y": 2, "owner": 0}],
        "tick": 7,
    }
    tape.record_frame(frame, "ok")
    assert tape.has_frames
    stored = tape.frames[0]
    stored["cells"][0, 0] = 99
    assert frame["cells"][0, 0] != 99
    assert stored["status"] == "ok"
    assert stored["tick"] == 7


def test_watch_settings_from_cli_and_preset_override():
    args = parse_args(["--ckpt", "x.pt", "--preset", "duel", "--seed", "alpha"])
    settings = WatchSettings.from_args(args)
    assert settings.map == "Halkidiki"
    assert settings.map_size == "Compact"
    assert settings.bots == 1
    assert settings.seed == "alpha"
    assert settings.env_config(2)["seed"] == "alpha-2"
    settings.apply_preset("tribes20")
    assert settings.map == "FourIslands"
    assert settings.bots == 20
    settings.apply_game_type("nations")
    assert settings.nations == "default"
    assert settings.bots == 0
    assert settings.preset == "custom"
    assert infer_game_type("disabled", 20) == "tribes"


def test_watch_settings_validate_missing_checkpoint(tmp_path):
    settings = WatchSettings(ckpt=str(tmp_path / "missing.pt"))
    assert "not found" in (settings.validate() or "")
    settings.ckpt = ""
    assert settings.validate() == "Checkpoint path is required."


def test_watch_settings_validate_accepts_existing_checkpoint(tmp_path):
    path = tmp_path / "ok.pt"
    path.write_bytes(b"pt")
    settings = WatchSettings(ckpt=str(path), seed="watch")
    assert settings.validate() is None


def test_parse_nations_value_and_episode_result():
    assert parse_nations_value("disabled") == "disabled"
    assert parse_nations_value("12") == 12
    with pytest.raises(ValueError):
        parse_nations_value("many")
    assert episode_result({"win": True}) == "WIN"
    assert episode_result({"dead": True}) == "DEAD"
    assert episode_result({"terminalCause": "timeout"}) == "timeout"


def test_parse_game_map_keys_reads_only_the_enum():
    source = """
export enum GameMapType {
  Halkidiki = "Halkidiki",
  FourIslands = "Four Islands",
  World = "World",
}
export const later = {
  NotAMap = "nope",
};
"""
    assert parse_game_map_keys(source) == ["Halkidiki", "FourIslands", "World"]
    assert parse_game_map_keys("no enum here") == []


def test_discover_maps_includes_curriculum_and_local_enum(tmp_path):
    maps = discover_maps(tmp_path)
    assert set(maps) >= {"Halkidiki", "FourIslands", "World"}
    gen = tmp_path / "src" / "core" / "game"
    gen.mkdir(parents=True)
    (gen / "Maps.gen.ts").write_text(
        'export enum GameMapType {\n  Africa = "Africa",\n  Halkidiki = "Halkidiki",\n}\n',
        encoding="utf-8",
    )
    discovered = discover_maps(tmp_path)
    assert "Africa" in discovered
    assert "Halkidiki" in discovered


def test_discover_checkpoints_lists_pt_files(tmp_path, monkeypatch):
    run = tmp_path / "run1"
    run.mkdir()
    (run / "latest.pt").write_bytes(b"a")
    (run / "notes.txt").write_text("no")
    found = discover_checkpoints(tmp_path)
    assert any(path.endswith("latest.pt") for path in found)
    assert all(path.endswith(".pt") for path in found)


def test_desktop_size_uses_full_monitor_without_reserve():
    class FakeDisplay:
        @staticmethod
        def get_desktop_sizes():
            return [(1920, 1080)]

    class FakePygame:
        display = FakeDisplay()

    assert desktop_size(FakePygame(), 0) == (1920, 1080)
    assert usable_display_size(FakePygame(), 0) == (1888, 984)


def test_speed_steps_cover_pause_adjacent_playback_range():
    assert SPEED_STEPS[0] == 0.25
    assert SPEED_STEPS[-1] == 8.0
    assert 1.0 in SPEED_STEPS


def test_config_field_edits_stay_within_supported_options():
    from agent.watch_ui import Runtime, apply_text_field, field_value, nudge_field

    settings = WatchSettings(ckpt="x.pt")
    assert field_value(settings, "greedy") == "off"
    assert apply_text_field(settings, "nations", "8") is None
    assert settings.nations == 8
    assert apply_text_field(settings, "bots", "401") is not None
    rt = Runtime(
        settings=settings,
        maps=["Halkidiki", "World"],
        checkpoints=["checkpoints/a.pt"],
        display_count=2,
        device=None,
    )
    settings.game_type = "tribes"
    nudge_field(rt, "game_type", 1)
    assert settings.game_type == "nations"
    assert settings.nations == "default"
    assert settings.bots == 0
    nudge_field(rt, "map", 1)
    assert settings.map in ("Halkidiki", "World")
    nudge_field(rt, "speed", 1)
    assert settings.speed in SPEED_STEPS


def _runtime(settings=None, **kwargs):
    from agent.watch_ui import Runtime

    settings = settings or WatchSettings(ckpt="x.pt")
    defaults = dict(
        settings=settings,
        maps=["Halkidiki", "World"],
        checkpoints=["checkpoints/a.pt"],
        display_count=1,
        device=None,
    )
    defaults.update(kwargs)
    return Runtime(**defaults)


def test_apply_text_field_bots_updates_game_type_and_rejects_invalid_fps():
    from agent.watch_ui import apply_text_field

    settings = WatchSettings(ckpt="x.pt", nations="default", bots=0, game_type="nations")
    settings.fps = 15.0
    assert apply_text_field(settings, "bots", "20") is None
    assert settings.game_type == "mixed"
    assert apply_text_field(settings, "fps", "0") == "FPS must be positive."
    assert settings.fps == 15.0


def test_nudge_int_fields_stay_within_valid_bounds():
    from agent.watch_ui import nudge_field

    settings = WatchSettings(ckpt="x.pt", frame_skip=1, scale=1, episodes=1)
    rt = _runtime(settings)
    nudge_field(rt, "frame_skip", -1)
    nudge_field(rt, "scale", -1)
    nudge_field(rt, "episodes", -1)
    assert settings.frame_skip == 1
    assert settings.scale == 1
    assert settings.episodes == 1
    nudge_field(rt, "frame_skip", 1)
    nudge_field(rt, "scale", 1)
    assert settings.frame_skip == 2
    assert settings.scale == 2
    nudge_field(rt, "frame_skip", -1)
    assert settings.frame_skip == 1


def test_apply_game_type_mixed_is_actually_mixed():
    settings = WatchSettings(ckpt="x.pt")
    settings.apply_game_type("nations")
    assert settings.bots == 0
    settings.apply_game_type("mixed")
    assert settings.nations == "default"
    assert settings.bots == 20
    assert infer_game_type(settings.nations, settings.bots) == "mixed"


def test_watch_settings_validate_rejects_empty_map(tmp_path):
    path = tmp_path / "ok.pt"
    path.write_bytes(b"pt")
    settings = WatchSettings(ckpt=str(path), map="")
    assert settings.validate() == "Map must not be empty."


def test_start_replay_holds_first_frame():
    from agent.watch_ui import advance_replay, start_replay

    tape = ReplayTape(max_bytes=1_000_000, max_frames=8)
    for tick in range(3):
        tape.record_frame(
            {
                "width": 4,
                "height": 3,
                "cells": np.zeros((3, 4), dtype=np.uint8),
                "players": [{"idx": 0, "name": "A", "tiles": 1, "alive": True, "isAgent": True}],
                "units": [],
                "tick": tick,
            },
            f"frame {tick}",
        )
    rt = _runtime(tape=tape)
    start_replay(rt)
    assert rt.mode == "replay"
    assert rt.hold_frame is True
    assert rt.replay_index == 0
    assert rt.last_status == "frame 0"
    if rt.hold_frame:
        rt.hold_frame = False
    else:
        advance_replay(rt)
    assert rt.replay_index == 0
    advance_replay(rt)
    assert rt.replay_index == 1


def test_choose_menu_quit_confirms_instead_of_exiting():
    from agent.watch_ui import choose_menu, handle_play_menu_event

    rt = _runtime(mode="playing")
    rt.playback = PlaybackControl(1.0)
    choose_menu(rt, "quit")
    assert rt.mode == "playing"
    assert rt.overlay == "quit"
    assert rt.playback.paused is True
    choose_menu(rt, "confirm_quit")
    assert rt.mode == "quit"

    class Event:
        def __init__(self, key):
            self.key = key

    class Keys:
        K_DOWN = 2
        K_UP = 1
        K_LEFT = 3
        K_RIGHT = 4
        K_RETURN = 5
        K_ESCAPE = 6
        K_c = 7

    rt = _runtime(mode="playing", overlay="menu")
    rt.playback = PlaybackControl(1.0, paused=True)
    handle_play_menu_event(rt, Keys, Event(Keys.K_DOWN))
    handle_play_menu_event(rt, Keys, Event(Keys.K_DOWN))
    handle_play_menu_event(rt, Keys, Event(Keys.K_RETURN))
    assert rt.overlay == "quit"


def test_shutdown_closes_env_and_server():
    from agent.watch_ui import shutdown

    class FakeEnv:
        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

    class FakeServer:
        def __init__(self):
            self.stopped = False

        def stop(self):
            self.stopped = True

        def alive(self):
            return not self.stopped

    rt = _runtime()
    env, server = FakeEnv(), FakeServer()
    rt.env, rt.server = env, server
    shutdown(rt)
    assert env.closed is True
    assert server.stopped is True
    assert rt.env is None
    assert rt.server is None


def test_policy_rng_snapshot_restores_sampled_actions():
    from agent.model import Policy, make_dummy_obs

    policy = Policy().eval()
    obs = make_dummy_obs(1, "cpu")
    seed_policy_rng(7)
    snap = snapshot_policy_rng()
    with torch.inference_mode():
        first, _, _ = policy.act(obs, deterministic=False)
        second, _, _ = policy.act(obs, deterministic=False)
    restore_policy_rng(snap)
    with torch.inference_mode():
        first_again, _, _ = policy.act(obs, deterministic=False)
        second_again, _, _ = policy.act(obs, deterministic=False)
    assert torch.equal(first, first_again)
    assert torch.equal(second, second_again)


def test_main_save_stays_headless(monkeypatch):
    import agent.watch as watch
    import agent.watch_ui as watch_ui

    seen = {}
    monkeypatch.setattr(watch, "run_headless", lambda args: seen.setdefault("headless", args.save))
    monkeypatch.setattr(watch_ui, "run_interactive", lambda args: seen.setdefault("ui", True))
    watch.main(["--ckpt", "x.pt", "--save", "out.mp4"])
    assert seen == {"headless": "out.mp4"}


def test_main_without_save_opens_interactive(monkeypatch):
    import agent.watch as watch
    import agent.watch_ui as watch_ui

    seen = {}
    monkeypatch.setattr(watch, "run_headless", lambda args: seen.setdefault("headless", True))
    monkeypatch.setattr(watch_ui, "run_interactive", lambda args: seen.setdefault("ui", True))
    watch.main(["--ckpt", "x.pt", "--windowed"])
    assert seen == {"ui": True}


def _dummy_pygame(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    import pygame

    pygame.display.init()
    pygame.font.init()
    return pygame


def test_dummy_sdl_scene_and_present_map(monkeypatch):
    pygame = _dummy_pygame(monkeypatch)
    try:
        frame = {
            "width": 8,
            "height": 6,
            "cells": np.arange(48, dtype=np.uint8).reshape(6, 8),
            "players": [{"idx": 0, "name": "Agent", "tiles": 4, "alive": True, "isAgent": True}],
            "units": [{"x": 2, "y": 1, "owner": 0}],
            "tick": 3,
        }
        palette = build_palette(1)
        bright = np.clip(palette.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        fonts = _load_fonts(pygame, 2)
        scene = _draw_scene(pygame, frame, palette, bright, "ok", 2, fonts)
        assert scene.get_size() == scene_size(8, 6, 2)
        screen = pygame.display.set_mode((320, 240))
        layout = overlay_layout(screen.get_size())
        camera = Camera()
        camera.reset(16, 12)
        present_map(
            pygame,
            screen,
            scene,
            layout["map"],
            camera.transform((16, 12), (layout["map"][2], layout["map"][3])),
        )
        camera.set_zoom(3.0, (16, 12), (layout["map"][2], layout["map"][3]), (8.0, 6.0))
        present_map(
            pygame,
            screen,
            scene,
            layout["map"],
            camera.transform((16, 12), (layout["map"][2], layout["map"][3])),
        )
        arr = pygame.surfarray.array3d(scene)
        assert arr.shape[2] == 3
        assert arr.max() > 0
    finally:
        pygame.quit()


def test_dummy_sdl_windowed_display_and_config(monkeypatch, tmp_path):
    pygame = _dummy_pygame(monkeypatch)
    try:
        from agent.watch_ui import DisplayState, draw_config

        ckpt = tmp_path / "ok.pt"
        ckpt.write_bytes(b"pt")
        settings = WatchSettings(
            ckpt=str(ckpt),
            windowed=True,
            window_size=(640, 360),
        )
        display = DisplayState(pygame, settings)
        assert display.fullscreen is False
        width, height = display.size()
        assert width <= 640 and height <= 360
        assert width >= 1 and height >= 1
        fonts = {
            "title": pygame.font.SysFont("monospace", 18, bold=True),
            "heading": pygame.font.SysFont("monospace", 14, bold=True),
            "body": pygame.font.SysFont("monospace", 13),
            "small": pygame.font.SysFont("monospace", 12),
            "tiny": pygame.font.SysFont("monospace", 11),
        }
        rt = _runtime(settings)
        hits = draw_config(pygame, display.screen, fonts, rt)
        kinds = [kind for kind, _ in hits]
        assert "start" in kinds and "quit" in kinds

        full = DisplayState(pygame, WatchSettings(ckpt=str(ckpt)))
        assert full.fullscreen is True
        assert min(full.size()) >= 1
        full.toggle_fullscreen()
        assert full.fullscreen is False
        full.toggle_fullscreen()
        assert full.fullscreen is True
    finally:
        pygame.quit()


def test_headless_dummy_video_frame_compose(monkeypatch, tmp_path):
    pygame = _dummy_pygame(monkeypatch)
    try:
        frame = {
            "width": 6,
            "height": 4,
            "cells": np.zeros((4, 6), dtype=np.uint8),
            "players": [{"idx": 0, "name": "A", "tiles": 1, "alive": True, "isAgent": True}],
            "units": [],
            "tick": 1,
        }
        palette = build_palette(1)
        bright = np.clip(palette.astype(np.int16) + 60, 0, 255).astype(np.uint8)
        fonts = _load_fonts(pygame, 1)
        scene = _draw_scene(pygame, frame, palette, bright, "headless", 1, fonts)
        path = tmp_path / "dummy.mp4"
        writer = open_video_writer(str(path), fps=8)
        assert writer is not None
        try:
            arr = pygame.surfarray.array3d(scene)
            writer.append_data(np.transpose(arr, (1, 0, 2)))
        finally:
            close_video_writer(writer)
        assert path.is_file()
        assert path.stat().st_size > 0
    finally:
        pygame.quit()
