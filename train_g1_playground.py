"""Train a Brax PPO policy for the G1 joystick task in mujoco_playground.

Saves checkpoints under ``checkpoints/g1_joystick/`` and the final params at
``checkpoints/g1_joystick/final.pkl``.

Usage::

    python train_g1_playground.py --timesteps 100_000_000
"""
from __future__ import annotations

import argparse
import os
import pickle
import time
from pathlib import Path

# Restrict JAX to a single GPU so brax's pmap-based training loop sees only
# one device and avoids the multi-device replication path entirely. On a B200,
# single-GPU is still plenty fast. Must be set before importing jax.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import jax
import numpy as np

# Compatibility shim: brax 0.14.2 still calls ``jax.device_put_replicated``
# which JAX 0.10 removed. Brax expects the replicated array to have a leading
# axis of size ``len(devices)`` (one slice per device).
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        def _add_axis(v):
            arr = np.asarray(v)
            return np.broadcast_to(arr[None], (n,) + arr.shape)
        return jax.device_put(jax.tree_util.tree_map(_add_axis, x))
    jax.device_put_replicated = _device_put_replicated

from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import train as ppo

from mujoco_playground import locomotion, wrapper
from mujoco_playground.config import locomotion_params


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="G1JoystickFlatTerrain")
    p.add_argument("--timesteps", type=int, default=100_000_000)
    p.add_argument("--num-envs", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="checkpoints/g1_joystick")
    args = p.parse_args()

    print("JAX devices:", jax.devices(), flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    env = locomotion.load(args.task, config_overrides={"impl": "jax"})
    eval_env = locomotion.load(args.task, config_overrides={"impl": "jax"})
    print("env loaded, action_size =", env.action_size, flush=True)

    cfg = locomotion_params.brax_ppo_config(args.task)
    # Adjust timesteps / parallel envs from CLI.
    cfg.num_timesteps = int(args.timesteps)
    cfg.num_envs = int(args.num_envs)

    ppo_kwargs = dict(cfg)
    network_factory_kwargs = ppo_kwargs.pop("network_factory")
    network_factory = lambda *a, **kw: ppo_networks.make_ppo_networks(
        *a, **kw, **network_factory_kwargs,
    )

    t0 = time.time()
    progress_log = []

    def progress(num_steps, metrics):
        elapsed = time.time() - t0
        ep_rew = float(metrics.get("eval/episode_reward", 0.0))
        ep_len = float(metrics.get("eval/avg_episode_length", 0.0))
        msg = (
            f"[{elapsed:6.0f}s] step={num_steps:>10d}  "
            f"ep_reward={ep_rew:+.2f}  ep_len={ep_len:.0f}"
        )
        print(msg, flush=True)
        progress_log.append({"t": elapsed, "step": num_steps, "ep_reward": ep_rew,
                             "ep_len": ep_len})

    make_inference_fn, params, _ = ppo.train(
        environment=env,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        eval_env=eval_env,
        progress_fn=progress,
        seed=args.seed,
        **ppo_kwargs,
    )

    final_path = out_dir / "final.pkl"
    with open(final_path, "wb") as f:
        pickle.dump({"params": params, "config_overrides": {"impl": "jax"},
                     "task": args.task}, f)
    print(f"Saved {final_path}", flush=True)

    with open(out_dir / "progress.pkl", "wb") as f:
        pickle.dump(progress_log, f)
    print(f"Total wall-clock: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
