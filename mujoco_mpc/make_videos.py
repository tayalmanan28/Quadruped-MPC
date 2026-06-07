"""Render a set of demo videos of the convex-MPC Go1 controller.

Run with::

    MUJOCO_GL=osmesa python make_videos.py
"""
from __future__ import annotations

import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "osmesa")

import imageio
import mujoco
import numpy as np

from mpc import Go1Robot, LocomotionController
from mpc.controller import ControllerConfig
from mpc.gait import TrotGait

OUT = Path(__file__).parent / "videos"
OUT.mkdir(exist_ok=True)


def render_episode(
    name: str,
    duration: float,
    cmd_vx: float = 0.0,
    cmd_vy: float = 0.0,
    cmd_wz: float = 0.0,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
    camera: str = "tracking",
) -> None:
    model = mujoco.MjModel.from_xml_path("models/go1/scene.xml")
    data = mujoco.MjData(model)
    robot = Go1Robot(model, data)
    robot.reset_to_home()
    model.opt.timestep = 0.001

    gait = TrotGait()
    ctrl = LocomotionController(
        robot,
        gait=gait,
        ctrl_cfg=ControllerConfig(cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_wz=cmd_wz),
    )

    renderer = mujoco.Renderer(model, height=height, width=width)
    save_every = max(1, int(round(1.0 / (model.opt.timestep * fps))))
    n_steps = int(duration / model.opt.timestep)

    frames: list[np.ndarray] = []
    t_wall = time.time()
    for k in range(n_steps):
        ctrl.update(data.time)
        mujoco.mj_step(model, data)
        if k % save_every == 0:
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    elapsed = time.time() - t_wall

    path = OUT / f"{name}.mp4"
    imageio.mimsave(path, frames, fps=fps, codec="libx264", quality=8)
    print(
        f"[{name}] cmd=(vx={cmd_vx}, vy={cmd_vy}, wz={cmd_wz})  "
        f"final pos={robot.base_pos().round(3)}  rpy={robot.base_rpy().round(3)}  "
        f"v={robot.base_lin_vel_world().round(3)}  "
        f"sim {duration:.1f}s in {elapsed:.1f}s -> {path}"
    )


if __name__ == "__main__":
    render_episode("stand_in_place", duration=3.0)
    render_episode("trot_forward_0p4", duration=6.0, cmd_vx=0.4)
    render_episode("turn_in_place_wz0p6", duration=6.0, cmd_wz=0.6)
    render_episode("diagonal_walk", duration=6.0, cmd_vx=0.25, cmd_vy=0.1)
