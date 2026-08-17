"""PPO training entry point.

Spawns (or connects to) env servers, then alternates rollout collection and
PPO updates. Curriculum stages advance on min_episodes plus consecutive
fixed-seed mastery passes: deterministic competency or sampled true-win
(Wilson-guarded). Deterministic argmax is a robustness diagnostic.

Example:
  .venv/bin/python -m agent.train --servers 4 --envs 8 --rollout-len 128 \
      --run-name dev --total-steps 1000000
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from .curriculum import Curriculum
from .env_client import RolloutInterrupted, ShardedVecEnv
from .model import Policy, obs_to_tensors
from . import spaces
from .ppo import PPOConfig, PPOTrainer, RolloutBuffer, classify_outcome

ROOT = Path(__file__).resolve().parent.parent
CKPT_DIR = ROOT / "checkpoints"
LOG_DIR = ROOT / "logs"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--envs", type=int, default=8, help="parallel games per server")
    p.add_argument("--servers", type=int, default=4, help="env-server processes")
    p.add_argument("--port", type=int, default=8765, help="base port (servers use port+i)")
    p.add_argument("--connect", action="store_true", help="connect to already-running servers")
    p.add_argument("--rollout-len", type=int, default=128)
    p.add_argument(
        "--total-steps",
        type=int,
        default=10_000_000,
        help="absolute env decision-step target (from 0, or from checkpoint if resumed)",
    )
    p.add_argument(
        "--additional-steps",
        type=int,
        default=0,
        help="with --resume: train this many more steps beyond checkpoint env_steps "
        "(overrides --total-steps as an absolute ceiling)",
    )
    p.add_argument("--run-name", type=str, default="run")
    p.add_argument("--resume", type=str, default="", help="checkpoint path to resume")
    p.add_argument(
        "--stage",
        type=int,
        default=None,
        help="explicit curriculum stage index for a fresh run (ignored on resume "
        "unless also passed to force override)",
    )
    p.add_argument("--force-stage", action="store_true", help="apply --stage even when resuming")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument(
        "--target-kl",
        type=float,
        default=0.03,
        help="PPO early-stop KL on active-head joint (default ~0.03)",
    )
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--update-epochs", type=int, default=4)
    p.add_argument(
        "--ent-target",
        type=float,
        default=0.6,
        help="mask-normalized action-type entropy target in [0, 1] "
        "(H / log(n_legal), averaged over samples with >1 legal actions)",
    )
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="iterations between fixed-seed inline evals (0=off)",
    )
    p.add_argument("--eval-episodes", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    return p.parse_args()


def _rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _byte_rng(t) -> torch.Tensor:
    """torch RNG APIs require a CPU uint8 tensor; checkpoints may load to CUDA."""
    if not isinstance(t, torch.Tensor):
        t = torch.as_tensor(t, dtype=torch.uint8)
    return t.detach().to(device="cpu", dtype=torch.uint8).contiguous()


def _set_rng_state(state: dict) -> None:
    if "python" in state:
        random.setstate(state["python"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if "torch" in state and state["torch"] is not None:
        torch.set_rng_state(_byte_rng(state["torch"]))
    if "cuda" in state and state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([_byte_rng(s) for s in state["cuda"]])


def run_fixed_seed_eval(
    policy: Policy,
    env: ShardedVecEnv,
    curriculum: Curriculum,
    device: torch.device,
    n_episodes: int,
    seed_prefix: str,
) -> dict:
    """Inline eval matching the active stage. Runs deterministic then sampled.

    Milestone termination is omitted (`objective=True`) so `win_rate` is the
    core 80% FFA outcome. Stage success is inferred from peak tiles.
    """
    stage = curriculum.stage
    k = min(env.k, n_episodes)
    results = {"deterministic": {}, "sampled": {}}
    stage_frac = float(stage.win_tiles_frac)

    for mode in ("deterministic", "sampled"):
        wins = 0
        stage_hits = 0
        deaths = 0
        kills = 0
        rets = []
        peaks = []
        finals = []
        ticks = []
        causes: dict[str, int] = {}
        accepted = 0
        total_a = 0
        hist: dict[str, int] = {}
        remaining = n_episodes
        while remaining > 0:
            batch_n = min(k, remaining)
            cfgs = [
                {
                    **curriculum.make_env_config(
                        f"{seed_prefix}-{mode}-{n_episodes - remaining + i}",
                        objective=True,
                    ),
                    "shaping": 0.0,  # eval without shaping
                }
                for i in range(batch_n)
            ]
            # Temporarily re-init only first batch_n via full init of size env.k
            # is awkward; use shard 0's VecEnv through a dedicated init of env.k
            # with padding configs when batch_n < env.k.
            pad = [
                curriculum.make_env_config(f"{seed_prefix}-pad-{i}", objective=True)
                for i in range(env.k - batch_n)
            ]
            for c in pad:
                c["shaping"] = 0.0
            obs = env.init(cfgs + pad)
            ep_ret = np.zeros(env.k, np.float32)
            active = np.zeros(env.k, dtype=bool)
            active[:batch_n] = True
            finished = 0
            while active[:batch_n].any():
                with torch.no_grad():
                    obs_t = obs_to_tensors(obs, device)
                    acts, _, _ = policy.act(obs_t, deterministic=(mode == "deterministic"))
                actions = acts.cpu().numpy().astype(np.int32)
                step = env.step(actions)
                ep_ret += step.rewards
                for i in range(batch_n):
                    if not active[i]:
                        continue
                    info = step.infos[i]
                    at = int(actions[i, 0])
                    name = spaces.ACTION_NAMES[at]
                    hist[name] = hist.get(name, 0) + 1
                    acc = info.get("actionAccepted", info.get("action_accepted", True))
                    total_a += 1
                    if acc:
                        accepted += 1
                    if step.dones[i]:
                        active[i] = False
                        finished += 1
                        cause, won, stage_ok = classify_outcome(info, stage_frac)
                        causes[cause] = causes.get(cause, 0) + 1
                        if won:
                            wins += 1
                        if stage_ok:
                            stage_hits += 1
                        if cause == "death":
                            deaths += 1
                        kills += int(info.get("kills", 0))
                        rets.append(float(ep_ret[i]))
                        peaks.append(float(info.get("peakTilesFrac", info.get("tilesFrac", 0))))
                        finals.append(float(info.get("tilesFrac", 0)))
                        ticks.append(int(info.get("tick", 0)))
                        ep_ret[i] = 0.0
                obs = step.obs
            remaining -= batch_n

        n = max(1, len(rets))
        results[mode] = {
            "episodes": n,
            "win_rate": wins / n,
            "stage_success_rate": stage_hits / n,
            "death_rate": deaths / n,
            "kills_per_ep": kills / n,
            "mean_return": float(np.mean(rets)) if rets else 0.0,
            "mean_peak_tiles": float(np.mean(peaks)) if peaks else 0.0,
            "mean_final_tiles": float(np.mean(finals)) if finals else 0.0,
            "mean_ticks": float(np.mean(ticks)) if ticks else 0.0,
            "terminal_causes": causes,
            "action_hist": hist,
            "accept_rate": accepted / max(1, total_a),
            "stage": stage.name,
        }
    return results


def _mode_collapsed(mode: dict) -> bool:
    """True when a mode is never-spawn / spawn-then-NOOP / no territory."""
    peak = float(mode.get("mean_peak_tiles", 0.0))
    causes = mode.get("terminal_causes") or {}
    n = max(1, int(mode.get("episodes", 1)))
    no_spawn = float(causes.get("no_spawn", 0)) / n
    hist = mode.get("action_hist") or {}
    total_a = float(sum(hist.values()) or 1.0)
    noop_frac = float(hist.get("noop", 0)) / total_a
    return (
        peak < 0.01
        or no_spawn >= 0.5
        or (noop_frac >= 0.85 and peak < 0.02)
    )


def _eval_score(ev: dict) -> tuple:
    """best.pt ranking for the deployed (sampled) policy.

    Order: sampled true win, det true win, sampled peak, det peak.
    Stage-success / milestone rates never enter this score. Collapse
    zeros that mode's contribution.
    """
    if "sampled" in ev or "deterministic" in ev:
        samp = ev.get("sampled") or {}
        det = ev.get("deterministic") or {}
    else:
        samp = ev
        det = ev
    samp_wr = float(samp.get("win_rate", 0.0))
    det_wr = float(det.get("win_rate", 0.0))
    samp_peak = float(samp.get("mean_peak_tiles", 0.0))
    det_peak = float(det.get("mean_peak_tiles", 0.0))
    if _mode_collapsed(samp):
        samp_wr = 0.0
        samp_peak = 0.0
    if _mode_collapsed(det):
        det_wr = 0.0
        det_peak = 0.0
    return (samp_wr, det_wr, samp_peak, det_peak)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")

    run_dir = CKPT_DIR / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{args.run_name}.jsonl"
    ep_log_path = LOG_DIR / f"{args.run_name}.episodes.jsonl"

    # Fresh run must not silently append to a prior run's logs.
    resume_path = args.resume
    log_mode = "a" if resume_path else "w"
    log_f = open(log_path, log_mode, buffering=1)
    ep_log_f = open(ep_log_path, log_mode, buffering=1)

    start_stage = args.stage if args.stage is not None else 0
    curriculum = Curriculum(start=start_stage)
    policy = Policy()
    cfg = PPOConfig(
        rollout_len=args.rollout_len,
        lr=args.lr,
        target_kl=args.target_kl,
        update_epochs=args.update_epochs,
        num_minibatches=args.num_minibatches,
        ent_target=args.ent_target,
        gamma=args.gamma,
        device=args.device if torch.cuda.is_available() else "cpu",
    )
    trainer = PPOTrainer(policy, cfg)
    best_score: tuple = (-1.0, -1.0, -1.0, -1.0)
    stage_baseline: dict = {}
    last_mastery: dict = {}

    if resume_path:
        ckpt = trainer.load(resume_path)
        print(f"resumed from {resume_path} (schema={ckpt.get('schema_version', 1)})")
        if "curriculum" in ckpt and not args.force_stage:
            curriculum.load_state_dict(ckpt["curriculum"])
        elif args.stage is not None and args.force_stage:
            curriculum = Curriculum(start=args.stage)
        if "rng" in ckpt:
            _set_rng_state(ckpt["rng"])
        stage_baseline = dict(ckpt.get("stage_baseline", {}))
        raw_best = tuple(ckpt.get("best_score", best_score))
        if len(raw_best) < 4:
            raw_best = tuple(list(raw_best) + [0.0] * (4 - len(raw_best)))
        best_score = raw_best[:4]  # type: ignore[assignment]
        # Pre-v2 scores ranked deterministic ticks/peaks; do not mix.
        if int((ckpt.get("curriculum") or {}).get("mastery_version", 1)) < 2:
            best_score = (-1.0, -1.0, -1.0, -1.0)
            print(
                "[train] mastery_version < 2: reset best.pt ranking "
                "(sampled true-win first)"
            )
        if args.additional_steps > 0:
            args.total_steps = trainer.stats.env_steps + args.additional_steps
            print(
                f"[train] --additional-steps={args.additional_steps} -> "
                f"total_steps={args.total_steps}"
            )
    elif args.stage is not None:
        curriculum = Curriculum(start=args.stage)

    env = ShardedVecEnv(
        num_servers=args.servers,
        base_port=args.port,
        envs_per_server=args.envs,
        connect=args.connect,
    )
    total_envs = env.k
    print(
        f"[train] batch={cfg.rollout_len * total_envs} "
        f"num_minibatches={cfg.num_minibatches} epochs={cfg.update_epochs} "
        f"target_kl={cfg.target_kl} stage={curriculum.stage.name}"
    )

    def reinit_envs() -> dict:
        cfgs = [
            curriculum.make_env_config(
                f"{args.run_name}-s{args.seed}-e{i}-it{trainer.stats.iterations}"
            )
            for i in range(total_envs)
        ]
        obs_ = env.init(cfgs)
        trainer._ensure_trackers(total_envs)
        trainer.reset_episode_trackers()
        return obs_

    def save_ckpt(path: Path, **extra) -> None:
        trainer.save(
            str(path),
            extra={
                "curriculum": curriculum.state_dict(),
                "stage_baseline": stage_baseline,
                "best_score": list(best_score),
                "rng": _rng_state(),
                **extra,
            },
        )

    if not resume_path:
        save_ckpt(run_dir / "untrained.pt")

    obs = reinit_envs()
    buf = RolloutBuffer(cfg.rollout_len, total_envs)
    steps_per_iter = cfg.rollout_len * total_envs
    device = torch.device(cfg.device)

    pbar = tqdm(
        total=args.total_steps,
        initial=trainer.stats.env_steps,
        unit="step",
        dynamic_ncols=True,
        desc=curriculum.stage.name,
    )

    try:
        while trainer.stats.env_steps < args.total_steps:
            t0 = time.time()
            try:
                obs, completed = trainer.collect(env, obs, buf)
            except RolloutInterrupted as exc:
                pbar.write(
                    f"[env] {exc}; discarding partial rollout and resetting trackers"
                )
                obs = reinit_envs()
                continue
            collect_s = time.time() - t0

            for ep in completed:
                curriculum.record_episode(ep.stage_success)
                ep_log_f.write(
                    json.dumps(
                        {
                            "it": trainer.stats.iterations,
                            "env_steps": trainer.stats.env_steps,
                            "stage": curriculum.stage.name,
                            "env_index": ep.env_index,
                            "return": round(ep.ret, 6),
                            "decisions": ep.decisions,
                            "ticks": ep.ticks,
                            "terminal_cause": ep.terminal_cause,
                            "win": ep.win,
                            "stage_success": ep.stage_success,
                            "kills": ep.kills,
                            "final_tiles_frac": ep.final_tiles_frac,
                            "peak_tiles_frac": ep.peak_tiles_frac,
                            "spawned": ep.spawned,
                            "reward_terms": ep.reward_terms,
                            "action_hist": ep.action_hist,
                            "accepted": ep.accepted,
                            "fizzled": ep.fizzled,
                            "intent_count": ep.intent_count,
                        }
                    )
                    + "\n"
                )

            t0 = time.time()
            metrics = trainer.update(buf, obs)
            update_s = time.time() - t0

            sps = steps_per_iter / max(1e-9, collect_s + update_s)
            trainer.stats.sps = sps
            st = trainer.stats

            prev_stage = curriculum.stage.name
            prev_idx = curriculum.idx
            if curriculum.maybe_advance():
                stage_baseline = {
                    "win_rate": st.win_rate,
                    "mean_return": st.mean_return,
                    "env_steps": st.env_steps,
                }
                best_score = (-1.0, -1.0, -1.0, -1.0)
                pbar.write(
                    f"[curriculum] advanced {prev_stage} -> {curriculum.stage.name} "
                    f"idx={prev_idx}->{curriculum.idx} "
                    f"reason={last_mastery.get('reason', 'unknown')} "
                    f"modes={last_mastery.get('eval_modes')} "
                    f"n_det={last_mastery.get('det_episodes')} "
                    f"n_samp={last_mastery.get('sampled_episodes')} "
                    f"samp_win={last_mastery.get('sampled_win_rate')} "
                    f"wilson_lb={last_mastery.get('sampled_wilson_lower')} "
                    f"samp_thresh={last_mastery.get('sampled_win_threshold')} "
                    f"wilson_floor={last_mastery.get('sampled_wilson_floor')} "
                    f"det_win={last_mastery.get('det_win_rate')} "
                    f"det_thresh={last_mastery.get('det_threshold')} "
                    f"passes={last_mastery.get('eval_passes_required')}"
                )
                pbar.set_description(curriculum.stage.name)
                save_ckpt(run_dir / "latest.pt")
                obs = reinit_envs()

            if args.eval_every and st.iterations > 0 and st.iterations % args.eval_every == 0:
                pbar.write(
                    f"[eval] fixed-seed eval at it={st.iterations} "
                    f"stage={curriculum.stage.name} n={args.eval_episodes} "
                    f"modes=deterministic,sampled"
                )
                try:
                    ev = run_fixed_seed_eval(
                        policy,
                        env,
                        curriculum,
                        device,
                        args.eval_episodes,
                        seed_prefix=f"{args.run_name}-eval-it{st.iterations}",
                    )
                    det = ev["deterministic"]
                    samp = ev["sampled"]
                    judgment = curriculum.judge_eval(ev)
                    curriculum.note_eval(judgment["passed"])
                    judgment["consecutive_eval_passes"] = (
                        curriculum.consecutive_eval_passes
                    )
                    last_mastery = judgment
                    score = _eval_score(ev)
                    if score > best_score:
                        best_score = score
                        save_ckpt(run_dir / "best.pt", eval=ev, mastery=judgment)
                        pbar.write(
                            f"[eval] new best.pt samp_win={samp['win_rate']:.3f} "
                            f"det_win={det['win_rate']:.3f} "
                            f"samp_peak={samp['mean_peak_tiles']:.3f} "
                            f"det_peak={det['mean_peak_tiles']:.3f}"
                        )
                    pbar.write(
                        f"[eval] modes=deterministic,sampled n={args.eval_episodes} "
                        f"samp_win={samp['win_rate']:.3f} "
                        f"det_win={det['win_rate']:.3f} "
                        f"samp_stage={samp.get('stage_success_rate', 0.0):.3f} "
                        f"det_stage={det.get('stage_success_rate', 0.0):.3f} "
                        f"wilson_lb={judgment['sampled_wilson_lower']:.3f} "
                        f"samp_thresh={judgment['sampled_win_threshold']} "
                        f"wilson_floor={judgment['sampled_wilson_floor']} "
                        f"det_thresh={judgment['det_threshold']} "
                        f"passed={judgment['passed']} reason={judgment['reason']} "
                        f"passes={judgment['consecutive_eval_passes']}"
                        f"/{judgment['eval_passes_required']}"
                    )
                    log_f.write(
                        json.dumps(
                            {"it": st.iterations, "eval": ev, "mastery": judgment}
                        )
                        + "\n"
                    )
                finally:
                    # Eval re-inits the shared shards; always restore train envs.
                    obs = reinit_envs()

            record = {
                "it": st.iterations,
                "env_steps": st.env_steps,
                "episodes": st.episodes,
                "stage": curriculum.stage.name,
                "stage_episodes": curriculum.stage_episodes,
                "stage_success_rate": round(curriculum.stage_success_rate, 4),
                "stage_win_rate": round(curriculum.stage_success_rate, 4),
                "sps": round(sps, 1),
                **st.rolling_snapshot(),
                **{
                    k: round(v, 5) if isinstance(v, float) else v
                    for k, v in metrics.items()
                },
                "legal_frac": trainer.take_legal_frac(),
            }
            log_f.write(json.dumps(record) + "\n")

            pbar.update(steps_per_iter)
            pbar.set_postfix(
                sps=f"{sps:.0f}",
                win20=f"{st.win_rate_20:.3f}",
                ss20=f"{st.stage_success_rate_20:.3f}",
                kill20=f"{st.kills_per_ep_20:.2f}",
                ret20=f"{st.mean_return_20:.3f}",
                acc20=f"{st.accept_rate_20:.2f}",
                eps=st.episodes,
                kl=f"{metrics.get('approx_kl', 0.0):.3f}",
                kept=f"{metrics.get('frac_kept', 0.0) * 100:.0f}%",
                aH=f"{metrics.get('action_entropy', 0.0):.2f}",
                nH=f"{metrics.get('action_entropy_norm', 0.0):.2f}",
                e=f"{metrics.get('ent_coef', 0.0):.3f}",
            )
            if st.iterations % 5 == 0:
                save_ckpt(run_dir / "latest.pt")
            if st.iterations % 100 == 0:
                save_ckpt(run_dir / f"it{st.iterations}.pt")
    except KeyboardInterrupt:
        pbar.write("interrupted; saving checkpoint")
    except Exception as exc:
        pbar.write(f"training crashed: {exc!r}; saving checkpoint")
        raise
    finally:
        pbar.close()
        save_ckpt(run_dir / "final.pt")
        env.close()
        log_f.close()
        ep_log_f.close()

    print(f"done. checkpoints in {run_dir}, log {log_path}, episodes {ep_log_path}")


if __name__ == "__main__":
    main()
