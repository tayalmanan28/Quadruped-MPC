"""Render a video of the Go1 trotting around obstacles using only (vx, wz).

A simple pure-pursuit navigator computes a body-frame velocity command from a
list of world-frame waypoints and feeds it to the convex-MPC locomotion
controller. Linear speed is forward-only (``vy = 0``); yaw rate steers.

Run::

    MUJOCO_GL=osmesa python make_obstacle_video.py
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


WAYPOINTS = np.array(
    [
        [1.2, -0.55],
        [2.4,  0.55],
        [3.6, -0.55],
        [4.8,  0.55],
        [6.0,  0.00],
    ]
)


class WaypointPilot:
    """Tiny pure-pursuit navigator that emits (vx, wz) only."""

    def __init__(
        self,
        waypoints: np.ndarray,
        v_max: float = 0.45,
        wz_max: float = 1.0,
        k_yaw: float = 2.0,
        wp_radius: float = 0.30,
        slow_for_turn_rad: float = 0.6,
    ):
        self.waypoints = np.asarray(waypoints, dtype=float)
        self.idx = 0
        self.v_max = float(v_max)
        self.wz_max = float(wz_max)
        self.k_yaw = float(k_yaw)
        self.wp_radius = float(wp_radius)
        self.slow_for_turn_rad = float(slow_for_turn_rad)
        self.done = False

    def step(self, base_xy: np.ndarray, yaw: float) -> tuple[float, float]:
        if self.done:
            return 0.0, 0.0
        tgt = self.waypoints[self.idx]
        dxy = tgt - base_xy
        dist = float(np.linalg.norm(dxy))
        if dist < self.wp_radius:
            self.idx += 1
            if self.idx >= len(self.waypoints):
                self.done = True
                return 0.0, 0.0
            tgt = self.waypoints[self.idx]
            dxy = tgt - base_xy
            dist = float(np.linalg.norm(dxy))

        cy, sy = np.cos(yaw), np.sin(yaw)
        dx_body =  cy * dxy[0] + sy * dxy[1]
        dy_body = -sy * dxy[0] + cy * dxy[1]
        heading_err = float(np.arctan2(dy_body, dx_body))

        wz = float(np.clip(self.k_yaw * heading_err, -self.wz_max, self.wz_max))
        slowdown = max(0.0, 1.0 - abs(heading_err) / self.slow_for_turn_rad)
        vx = float(self.v_max * slowdown)
        if dx_body < 0.05:
            vx = 0.0
        return vx, wz


def main() -> None:
    out_dir = Path(__file__).parent / "videos"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "obstacle_slalom.mp4"

    model = mujoco.MjModel.from_xml_path("models/go1/scene_obstacles.xml")
    data = mujoco.MjData(model)
    robot = Go1Robot(model, data)
    robot.reset_to_home()
    model.opt.timestep = 0.001

    ctrl = LocomotionController(robot)
    pilot = WaypointPilot(WAYPOINTS)

    width, height, fps = 720, 480, 30
    renderer = mujoco.Renderer(model, height=height, width=width)
    save_every = max(1, int(round(1.0 / (model.opt.timestep * fps))))
    duration = 36.0
    n_steps = int(duration / model.opt.timestep)

    frames: list[np.ndarray] = []
    t_wall = time.time()
    for k in range(n_steps):
        base_xy = robot.base_pos()[:2]
        yaw = robot.base_rpy()[2]
        vx, wz = pilot.step(base_xy, yaw)
        ctrl.set_command(vx=vx, wz=wz)
        ctrl.update(data.time)
        mujoco.mj_step(model, data)
        if k % save_every == 0:
            renderer.update_scene(data, camera="tracking")
            frames.append(renderer.render())
        if pilot.done and base_xy[0] > WAYPOINTS[-1, 0]:
            break
    elapsed = time.time() - t_wall

    imageio.mimsave(out_path, frames, fps=fps, codec="libx264", quality=8)
    print(
        f"Wrote {out_path} ({len(frames)} frames, sim {data.time:.1f}s "
        f"in {elapsed:.1f}s wall)."
    )
    print(
        f"Final pose: pos={robot.base_pos().round(3)}  "
        f"yaw={robot.base_rpy()[2]:+.2f}  "
        f"reached_waypoints={pilot.idx}/{len(WAYPOINTS)}"
    )


if __name__ == "__main__":
    main()
