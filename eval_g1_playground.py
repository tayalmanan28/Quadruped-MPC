"""Load a trained playground PPO policy and run inference in MuJoCo.

The G1 joystick environment's observation is a fixed 103-dim vector::

    state = [
        local_linvel(3), gyro(3), gravity(3), command(3),
        joint_angles - default_pose(29), joint_vel(29), last_act(29),
        cos(phase)(2), sin(phase)(2),
    ]

We construct this each control step from plain MuJoCo state and feed it to the
inference fn from ``brax.training.agents.ppo.networks``.

Use this both as a baseline tester and as the inference layer for the
comparison script.
"""
from __future__ import annotations

import os
import pickle
import time
from pathlib import Path

# Single-GPU for inference is plenty.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import jax
import jax.numpy as jp
import numpy as np

# Same shim as the trainer.
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        def _add(v):
            a = np.asarray(v)
            return np.broadcast_to(a[None], (n,) + a.shape)
        return jax.device_put(jax.tree_util.tree_map(_add, x))
    jax.device_put_replicated = _device_put_replicated

import mujoco
from brax.training.agents.ppo import networks as ppo_networks

from mujoco_playground import locomotion
from mujoco_playground._src.locomotion.g1 import g1_constants as consts


class G1PolicyRunner:
    """Wraps a trained playground policy + an MJ model in plain MuJoCo."""

    def __init__(self, checkpoint_path: str, task: str = "G1JoystickFlatTerrain",
                 gait_freq_hz: float = 1.4):
        with open(checkpoint_path, "rb") as f:
            ckpt = pickle.load(f)
        self._params = ckpt["params"]
        self.task = ckpt.get("task", task)

        # Build the playground env once just to fetch its model, default pose,
        # phase frequency and observation/action sizes.
        env = locomotion.load(self.task, config_overrides={"impl": "jax"})
        self._env = env

        # Construct the brax PPO inference network. Match the **brax default**
        # network sizes ((32,)*4 for policy and (256,)*5 for value) because
        # our training script forgot to pass ``network_factory`` to
        # ``ppo.train`` and thus used brax defaults rather than the
        # locomotion_params config.
        nets = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size,
            action_size=env.action_size,
        )
        # Brax's make_inference_fn returns a fn (params) -> policy_fn
        from brax.training.agents.ppo.networks import make_inference_fn as _make
        make_inference_fn = _make(nets)
        self._policy = make_inference_fn(self._params, deterministic=True)
        # Jit it for speed.
        self._policy_jit = jax.jit(self._policy)

        # Pull the model from the playground env so the policy runs on the
        # exact same MJCF it was trained on.
        self.model: mujoco.MjModel = env.mj_model
        self.data: mujoco.MjData = mujoco.MjData(self.model)
        # Init pose: the env's "knees_bent" keyframe.
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        if kid < 0:
            kid = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)

        # Cache indices.
        self._pelvis_imu_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "imu_in_pelvis")
        self._pelvis_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        self._default_pose = np.asarray(env.unwrapped._default_pose, dtype=float)
        self._action_scale = float(env._config.action_scale)
        # Control / sim dt as the env uses them.
        self.ctrl_dt = float(env._config.ctrl_dt)
        self.sim_dt = float(env._config.sim_dt)
        self.n_substeps = max(1, int(round(self.ctrl_dt / self.sim_dt)))
        self.model.opt.timestep = self.sim_dt

        # Gait phase: starts [0, pi], increments at 2*pi*gait_freq*ctrl_dt.
        self.gait_freq = float(gait_freq_hz)
        self._phase = np.array([0.0, np.pi])
        self._phase_dt = 2 * np.pi * self.ctrl_dt * self.gait_freq
        self._last_act = np.zeros(env.action_size, dtype=float)
        self._command = np.zeros(3, dtype=float)
        self._rng = jax.random.PRNGKey(0)

        # Foot sites for diagnostics.
        self._foot_sites = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, s)
            for s in consts.FEET_SITES
        ]

    # ------------------------------------------------------------------
    def set_command(self, vx: float, vy: float, wz: float):
        self._command = np.array([vx, vy, wz], dtype=float)

    def reset(self):
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "knees_bent")
        if kid < 0:
            kid = 0
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)
        self._phase = np.array([0.0, np.pi])
        self._last_act = np.zeros_like(self._last_act)

    # ------------------------------------------------------------------
    def _build_obs(self) -> jp.ndarray:
        d = self.data
        m = self.model
        # Pelvis local linear velocity: rotate world linvel into pelvis frame.
        R_pelvis = d.xmat[self._pelvis_body_id].reshape(3, 3)
        v_world = d.qvel[0:3]
        linvel = R_pelvis.T @ v_world
        # Gyro (pelvis local angular vel).
        w_body = d.qvel[3:6]  # already body frame for free joint
        gyro = w_body
        # Gravity in pelvis local frame.
        gravity_world = np.array([0.0, 0.0, -1.0])
        gravity = R_pelvis.T @ gravity_world
        # Joint angles / vels.
        joint_angles = d.qpos[7:]
        joint_vel = d.qvel[6:]
        # Phase encoding.
        cos = np.cos(self._phase)
        sin = np.sin(self._phase)
        phase = np.concatenate([cos, sin])

        state = np.concatenate([
            linvel, gyro, gravity, self._command,
            joint_angles - self._default_pose, joint_vel, self._last_act,
            phase,
        ]).astype(np.float32)
        return jp.asarray(state)

    # ------------------------------------------------------------------
    def step(self):
        """One control step: query policy, apply motor targets, step physics
        ``n_substeps`` times."""
        obs = {"state": self._build_obs()}
        self._rng, sub = jax.random.split(self._rng)
        action, _ = self._policy_jit(obs, sub)
        action = np.asarray(action)
        motor_targets = self._default_pose + action * self._action_scale
        # Drive the position actuators via data.ctrl.
        self.data.ctrl[:] = motor_targets
        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)
        # Advance phase.
        self._phase = np.fmod(
            self._phase + self._phase_dt + np.pi, 2 * np.pi
        ) - np.pi
        self._last_act = action

    # ------------------------------------------------------------------
    @property
    def t(self) -> float:
        return float(self.data.time)

    def base_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()

    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[3:7].copy()

    def base_rpy(self) -> np.ndarray:
        w, x, y, z = self.base_quat_wxyz()
        roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return np.array([roll, pitch, yaw])

    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[0:3].copy()

    def base_ang_vel_world(self) -> np.ndarray:
        R = self.data.xmat[self._pelvis_body_id].reshape(3, 3)
        return R @ self.data.qvel[3:6]

    def foot_z(self, leg_idx: int) -> float:
        return float(self.data.site_xpos[self._foot_sites[leg_idx]][2])


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/g1_joystick/final.pkl")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--duration", type=float, default=8.0)
    args = p.parse_args()

    pr = G1PolicyRunner(args.ctrl) if False else G1PolicyRunner(args.ckpt)
    pr.set_command(args.vx, args.vy, args.wz)
    n_steps = int(args.duration / pr.ctrl_dt)
    t0 = time.time()
    for i in range(n_steps):
        pr.step()
        if i % 25 == 0:
            print(f"t={pr.t:.2f}  pos={pr.base_pos().round(3)}  "
                  f"rpy={pr.base_rpy().round(3)}  feet_z=[{pr.foot_z(0):.3f},{pr.foot_z(1):.3f}]")
    print(f"Sim {args.duration}s in {time.time()-t0:.2f}s wall")


if __name__ == "__main__":
    main()
