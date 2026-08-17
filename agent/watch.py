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
  # interactive window (needs a display; falls back to a message if none).
  # Sampled policy; --fps is displayed frames, --frame-skip skips full-res
  # render work between decisions.
  .venv/bin/python -m agent.watch --ckpt checkpoints/run3/latest.pt \
      --scale 1 --fps 60 --frame-skip 5

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
import time

import numpy as np
import torch

from . import spaces
from .env_client import EnvServerProcess, VecEnv
from .model import Policy, obs_to_tensors

LEGEND_W = 230  # legend sidebar width in screen px
STATUS_H = 30  # status bar height in screen px


def positive_int(value: str) -> int:
    """argparse type: integer >= 1."""
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid int value: {value!r}") from exc
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {parsed}")
    return parsed


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
    pal = np.zeros((256, 3), np.uint8)
    pal[0] = (18, 38, 63)      # ocean
    pal[1] = (70, 70, 75)      # impassable
    pal[2] = (176, 164, 132)   # unowned land (tan)
    pal[250] = (200, 40, 200)  # fallout magenta
    for i in range(n_players):
        pal[3 + i] = _player_color(i)
    return pal


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--map", type=str, default="FourIslands")
    p.add_argument("--map-size", type=str, default="Normal")
    p.add_argument("--nations", type=str, default="disabled")
    p.add_argument("--difficulty", type=str, default="Easy")
    p.add_argument("--bots", type=int, default=20)
    p.add_argument("--max-ticks", type=int, default=12000)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--fps",
        type=float,
        default=15.0,
        help="displayed/written frames per second (not simulation decisions)",
    )
    p.add_argument("--scale", type=int, default=3, help="pixels per map tile")
    p.add_argument("--save", type=str, default="", help="write mp4 instead of a window")
    p.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    p.add_argument(
        "--frame-skip",
        type=positive_int,
        default=1,
        metavar="N",
        help="render every N decisions (always includes first and final frames)",
    )
    return p.parse_args(argv)


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


def overlay_units(rgb: np.ndarray, frame: dict, palette: np.ndarray) -> None:
    """Mark units as a bright pixel cluster in the owner's color."""
    h, w = frame["cells"].shape
    for u in frame["units"]:
        x, y, owner = u["x"], u["y"], u["owner"]
        if 0 <= x < w and 0 <= y < h:
            color = palette[3 + owner] if owner >= 0 else np.array([255, 255, 255], np.uint8)
            # brighten
            c = np.clip(color.astype(int) + 60, 0, 255).astype(np.uint8)
            rgb[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2] = c


def player_centroids(frame: dict) -> dict[int, tuple[float, float]]:
    """Centroid (x,y) of each player's territory in tile coords."""
    cells = frame["cells"]
    acc: dict[int, list[float]] = {}
    ys, xs = np.nonzero(cells >= 3)
    for x, y in zip(xs, ys):
        idx = int(cells[y, x]) - 3
        a = acc.setdefault(idx, [0.0, 0.0, 0.0])
        a[0] += x
        a[1] += y
        a[2] += 1
    return {i: (a[0] / a[2], a[1] / a[2]) for i, a in acc.items() if a[2] > 0}


def _load_fonts(pygame, scale: int):
    """Create fonts once; SysFont is expensive inside the render loop."""
    return {
        "name": pygame.font.SysFont("monospace", max(11, scale * 3), bold=True),
        "legend": pygame.font.SysFont("monospace", 13),
        "status": pygame.font.SysFont("monospace", 14),
    }


def _draw_scene(pygame, frame: dict, palette: np.ndarray, status: str, scale: int, fonts: dict):
    """Map + on-map labels + legend sidebar + status bar. Visually unchanged."""
    rgb = compose_rgb(frame, palette)
    overlay_units(rgb, frame, palette)
    centroids = player_centroids(frame)

    mw, mh = frame["width"] * scale, frame["height"] * scale
    scene = pygame.Surface((mw + LEGEND_W, mh + STATUS_H))
    scene.fill((15, 15, 20))
    surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
    surf = pygame.transform.scale(surf, (mw, mh))
    scene.blit(surf, (0, 0))

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
            scene.blit(shadow, rect.move(dx, dy))
        scene.blit(text, rect)

    leg_font = fonts["legend"]
    y = 8
    for p in frame["players"]:
        if y > mh - 18:
            break
        swatch = pygame.Rect(mw + 8, y, 14, 14)
        pygame.draw.rect(scene, palette[3 + p["idx"]].tolist(), swatch)
        if not p["alive"]:
            pygame.draw.line(scene, (200, 200, 200), swatch.topleft, swatch.bottomright, 2)
        marker = "* " if p["isAgent"] else "  "
        txt = f"{marker}{p['name'][:16]:16} {p['tiles']:>6}"
        scene.blit(leg_font.render(txt, True, (220, 220, 220)), (mw + 26, y))
        y += 17

    scene.blit(fonts["status"].render(status, True, (235, 235, 235)), (6, mh + 6))
    return scene


def _obs_status(ep: int, step: int, acts: np.ndarray, obs: dict, tick: int | None = None) -> str:
    g = obs["global"][0]
    tick_part = f" tick{tick}" if tick is not None else ""
    return (
        f"ep{ep} step{step}{tick_part} | {action_str(acts)}"
        f" | myTiles={g[2]:.3f} troops={np.expm1(g[3] * 15):.0f}"
        f" gold={np.expm1(g[4] * 20):.0f}"
    )


def _pump_quit(pygame) -> bool:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
    return False


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"loaded {args.ckpt} (iter {ckpt.get('stats', {}).get('iterations', '?')})")

    nations: int | str = args.nations
    if nations not in ("default", "disabled"):
        nations = int(nations)

    # We always render through pygame (so text + legend appear in both the
    # window and saved video). For headless video we use a hidden surface.
    import os

    import pygame

    interactive = not args.save
    if interactive:
        try:
            if not os.environ.get("DISPLAY") and os.name != "nt":
                raise RuntimeError("no DISPLAY")
            pygame.display.init()
            screen = None
        except Exception as e:  # pragma: no cover
            print(
                f"[watch] no interactive display ({e}); re-run with --save match.mp4 "
                "to render a video instead."
            )
            interactive = False
    if not interactive:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    pygame.init()
    pygame.font.init()
    fonts = _load_fonts(pygame, args.scale)

    server = EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)
    writer = None
    out_name = ""

    try:
        for ep in range(args.episodes):
            writer = None
            out_name = episode_video_path(args.save, ep) if args.save else ""
            try:
                obs = env.init(
                    [
                        {
                            "map": args.map,
                            "mapSize": args.map_size,
                            "nations": nations,
                            "difficulty": args.difficulty,
                            "bots": args.bots,
                            "seed": f"watch-{ep}-{int(time.time())}",
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
                while not done:
                    with torch.no_grad():
                        obs_t = obs_to_tensors(obs, device)
                        a, _, _ = policy.act(obs_t, deterministic=args.greedy)
                        acts = a[0].cpu().numpy().astype(np.int32)

                    batch = env.step(acts[None, :])
                    obs = batch.obs
                    done = bool(batch.dones[0])
                    step += 1

                    if not should_render_frame(step, done, args.frame_skip):
                        if step % 20 == 0:
                            print(_obs_status(ep, step, acts, obs))
                        if interactive and _pump_quit(pygame):
                            done = True
                        continue

                    frame = env.frame(0)
                    if not legend_printed:
                        print(f"--- map {frame['width']}x{frame['height']} players ---")
                        for p in frame["players"]:
                            tag = " <== AGENT" if p["isAgent"] else ""
                            print(f"  color#{p['idx']}: {p['name']}{tag}")
                        legend_printed = True

                    palette = build_palette(len(frame["players"]))
                    status = _obs_status(ep, step, acts, obs, tick=frame["tick"])
                    if step % 20 == 0 or done:
                        print(status)

                    scene = _draw_scene(pygame, frame, palette, status, args.scale, fonts)

                    if args.save:
                        if writer is None:
                            writer = open_video_writer(out_name, args.fps)
                        if writer is not None:
                            arr = pygame.surfarray.array3d(scene)  # [W,H,3]
                            writer.append_data(np.transpose(arr, (1, 0, 2)))
                    if interactive:
                        if screen is None:
                            mw = frame["width"] * args.scale
                            mh = frame["height"] * args.scale
                            screen = pygame.display.set_mode((mw + LEGEND_W, mh + STATUS_H))
                            pygame.display.set_caption("OFAI watch")
                        screen.blit(scene, (0, 0))
                        pygame.display.flip()
                        if _pump_quit(pygame):
                            done = True
                        time.sleep(1.0 / max(1e-6, args.fps))

                if batch is not None:
                    info = batch.infos[0]
                    if info.get("win"):
                        result = "WIN"
                    elif info.get("stageSuccess") or info.get("terminalCause") == "curriculum_success":
                        result = "STAGE"
                    elif info.get("dead"):
                        result = "DEAD"
                    else:
                        result = str(info.get("terminalCause") or "timeout")
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


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[watch] interrupted", flush=True)
