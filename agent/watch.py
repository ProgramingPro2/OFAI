"""Watch a trained checkpoint play matches, rendered live from the same
observations the policy sees.

Shows the agent's territory / enemies / structures / border as a color map,
plus a per-decision action log (what it chose and why it's legal). This is a
debug view of the model's own inputs, not a pixel replay of the game.

Examples:
  # interactive window
  .venv/bin/python -m agent.watch --ckpt checkpoints/run2/it700.pt

  # write a video instead (no display needed)
  .venv/bin/python -m agent.watch --ckpt checkpoints/run2/it700.pt \
      --save match.mp4 --episodes 3

  # watch a specific stage's opponents
  .venv/bin/python -m agent.watch --ckpt checkpoints/run2/it700.pt \
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

try:
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
except ImportError:  # pragma: no cover
    plt = None

CH = {
    "land": 0,
    "mountain": 1,
    "mine": 2,
    "enemy": 3,
    "ally": 4,
    "border": 5,
    "fallout": 6,
    "defense": 7,
    "my_struct": 8,
    "enemy_struct": 9,
}


def parse_args() -> argparse.Namespace:
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
    p.add_argument("--fps", type=float, default=10.0, help="render speed (decisions/s)")
    p.add_argument("--save", type=str, default="", help="write to mp4 instead of a window")
    p.add_argument("--greedy", action="store_true", help="argmax instead of sampling")
    return p.parse_args()


def render_frame(spatial: np.ndarray) -> np.ndarray:
    """Compose the obs planes into an RGB image [S,S,3] in [0,1]."""
    s = spatial
    img = np.zeros((spaces.SPATIAL_SIZE, spaces.SPATIAL_SIZE, 3), np.float32)
    # water dark blue, land dark green-gray
    land = s[CH["land"]]
    img[..., 0] = 0.10 + 0.10 * land
    img[..., 1] = 0.14 + 0.12 * land
    img[..., 2] = 0.18 - 0.06 * land
    # mountains lighter
    img += (s[CH["mountain"]] * 0.25)[..., None]
    # mine: teal
    img[..., 1] += s[CH["mine"]] * 0.9
    img[..., 2] += s[CH["mine"]] * 0.7
    # enemy: red, ally: yellow-green, mine-territory: blue-cyan
    enemy = s[CH["enemy"]]
    ally = s[CH["ally"]]
    img[..., 0] += enemy * 0.85 + ally * 0.55
    img[..., 1] += ally * 0.55 - enemy * 0.05
    img[..., 2] += s[CH["mine"]] * 0.0  # mine already colored
    img[..., 2] += (s[CH["mine"]] > 0).astype(np.float32) * 0.0
    # own territory tint (blue)
    img[..., 2] += s[CH["mine"]] * 0.6
    img[..., 0] += s[CH["mine"]] * 0.1
    # border bright white
    img += (s[CH["border"]] * 0.7)[..., None]
    # fallout magenta
    img[..., 0] += s[CH["fallout"]] * 0.8
    img[..., 2] += s[CH["fallout"]] * 0.8
    # my structures cyan-ish, enemy structures orange (already partly red)
    img[..., 1] += s[CH["my_struct"]] * 0.9
    img[..., 2] += s[CH["my_struct"]] * 0.9
    img[..., 0] += s[CH["enemy_struct"]] * 0.5
    return np.clip(img, 0, 1)


def action_str(a: np.ndarray) -> str:
    at = int(a[0])
    name = spaces.ACTION_NAMES[at]
    if at == spaces.ACTION_ATTACK:
        return f"ATTACK target={a[1]} troops={spaces and [5,15,30,50,80][int(a[3])]}%"
    if at == spaces.ACTION_BUILD:
        return f"BUILD {spaces.UNIT_NAMES[int(a[4])]} region={a[2]}"
    if at == spaces.ACTION_BOAT:
        return f"BOAT region={a[2]} troops={[5,15,30,50,80][int(a[3])]}%"
    if at == spaces.ACTION_SPAWN:
        return f"SPAWN region={a[2]}"
    if at == spaces.ACTION_ALLY:
        return f"ALLY target={a[1]}"
    return name.upper()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    policy = Policy().to(device)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    policy.load_state_dict(ckpt["model"])
    policy.eval()
    print(f"loaded {args.ckpt} (iter {ckpt.get('stats', {}).get('iterations', '?')})")

    nations: int | str = args.nations
    if nations not in ("default", "disabled"):
        nations = int(nations)

    server = EnvServerProcess(args.port)
    env = VecEnv("127.0.0.1", args.port)

    writer = None
    try:
        for ep in range(args.episodes):
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
            fig = None
            im = None
            ax = None
            if not args.save and plt is not None:
                plt.ion()
                fig, ax = plt.subplots(figsize=(7, 7))
            frames = []
            done = False
            step = 0
            while not done:
                with torch.no_grad():
                    obs_t = obs_to_tensors(obs, device)
                    logits = policy.logits(obs_t)
                    if args.greedy:
                        heads = ["action", "target", "region", "quantity", "unit"]
                        acts = torch.stack(
                            [logits[h].argmax(dim=-1) for h in heads], dim=1
                        )[0].cpu().numpy().astype(np.int32)
                    else:
                        a, _, _ = policy(obs_t)
                        acts = a[0].cpu().numpy().astype(np.int32)
                batch = env.step(acts[None, :])
                obs = batch.obs
                done = bool(batch.dones[0])
                step += 1

                img = render_frame(obs["spatial"][0])
                g = obs["global"][0]
                status = (
                    f"ep{ep} step{step} tick~{step * 10} | {action_str(acts)}"
                    f" | myTiles={g[2]:.3f} troops={np.expm1(g[3] * 15):.0f}"
                    f" gold={np.expm1(g[4] * 20):.0f} alive={g[6] * spaces.NUM_PLAYER_SLOTS:.0f}"
                )
                if args.save:
                    frames.append((img * 255).astype(np.uint8))
                elif plt is not None and ax is not None:
                    if im is None:
                        im = ax.imshow(img, interpolation="nearest")
                        ax.set_title(status, fontsize=8)
                        plt.axis("off")
                        plt.show(block=False)
                    else:
                        im.set_data(img)
                        ax.set_title(status, fontsize=8)
                    fig.canvas.draw()
                    fig.canvas.flush_events()
                    time.sleep(1.0 / max(1e-6, args.fps))
                if step % 20 == 0:
                    print(status)

            info = batch.infos[0]
            print(
                f"episode {ep}: {'WIN' if info.get('win') else 'DEAD' if info.get('dead') else 'timeout'}"
                f" tick={info.get('tick')} kills={info.get('kills')} tilesFrac={info.get('tilesFrac'):.3f}"
            )
            if args.save and frames:
                _write_video(args.save, frames, args.fps, ep)
                frames = []
            if plt is not None and fig is not None:
                plt.close(fig)
    finally:
        env.close()
        server.stop()


def _write_video(path: str, frames: list[np.ndarray], fps: float, ep: int) -> None:
    try:
        import imageio.v2 as imageio
    except ImportError:
        print("imageio not installed; cannot save video. pip install imageio[ffmpeg]")
        return
    out = path if args_episode_free(path) else path
    if len(frames) and (path.endswith(".mp4")):
        # scale up nearest for visibility
        big = [np.kron(f, np.ones((8, 8, 1))).astype(np.uint8) for f in frames]
        name = out.replace(".mp4", f"-ep{ep}.mp4") if ep > 0 else out
        imageio.mimsave(name, big, fps=max(1, int(fps)), codec="libx264")
        print(f"wrote {name}")


def args_episode_free(p: str) -> bool:
    return True


if __name__ == "__main__":
    main()
