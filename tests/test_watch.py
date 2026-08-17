"""Focused tests for agent.watch helpers (no display, no live env).

Run: .venv/bin/python -m pytest tests/test_watch.py -q
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.watch import (
    close_video_writer,
    episode_video_path,
    open_video_writer,
    parse_args,
    positive_int,
    should_render_frame,
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


def test_parse_args_frame_skip_custom():
    args = parse_args(["--ckpt", "x.pt", "--frame-skip", "5", "--fps", "60"])
    assert args.frame_skip == 5
    assert args.fps == 60.0


def test_parse_args_frame_skip_rejects_zero():
    with pytest.raises(SystemExit):
        parse_args(["--ckpt", "x.pt", "--frame-skip", "0"])


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
