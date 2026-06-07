"""Simulation entry point for the MuJoCo + convex-MPC Go1 controller.

Run with ``python main.py``. Use ``--no-viewer`` for a headless run.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

from mpc import Go1Robot, LocomotionController
from mpc.controller import ControllerConfig
from mpc.convex_mpc import MPCConfig
from mpc.gait import TrotGait

MODEL_PATH = Path(__file__).parent / "models" / "go1" / "scene.xml"


def build(args: argparse.Namespace):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Could not find Go1 scene at {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    robot = Go1Robot(model, data)
    robot.reset_to_home()
    model.opt.timestep = 0.001

    gait = TrotGait(
        period=args.gait_period,
        duty_factor=args.duty,
        swing_height=args.swing_height,
    )
    mpc_cfg = MPCConfig(horizon=args.horizon, dt=args.mpc_dt, mu=args.mu)
    ctrl_cfg = ControllerConfig(
        body_height=args.body_height,
        swing_height=args.swing_height,
        cmd_vx=args.vx,
        cmd_vy=args.vy,
        cmd_wz=args.wz,
    )
    controller = LocomotionController(robot, gait=gait, mpc_cfg=mpc_cfg, ctrl_cfg=ctrl_cfg)
    return model, data, robot, controller


def run_viewer(args: argparse.Namespace) -> None:
    import mujoco.viewer

    model, data, robot, controller = build(args)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        sim_start = time.time()
        t0 = data.time
        while viewer.is_running():
            t = data.time
            if t - t0 > args.duration > 0:
                break
            controller.update(t)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - sim_start
            target = data.time - t0
            if target > elapsed:
                time.sleep(target - elapsed)


def run_headless(args: argparse.Namespace) -> None:
    model, data, robot, controller = build(args)
    n_steps = int(args.duration / model.opt.timestep)
    if args.video:
        try:
            import imageio
        except ImportError as exc:
            raise SystemExit(
                "--video requires `imageio` (pip install imageio[ffmpeg])"
            ) from exc
        renderer = mujoco.Renderer(model, height=480, width=640)
        frames = []
        save_every = max(1, int(round(1.0 / (model.opt.timestep * args.video_fps))))
    else:
        renderer = None
        frames = []
        save_every = 0

    t_wall_start = time.time()
    for k in range(n_steps):
        controller.update(data.time)
        mujoco.mj_step(model, data)
        if renderer is not None and (k % save_every == 0):
            renderer.update_scene(data, camera="tracking")
            frames.append(renderer.render())
    sim_dt = time.time() - t_wall_start
    print(f"Simulated {args.duration:.2f}s in {sim_dt:.2f}s wall "
          f"({args.duration / sim_dt:.1f}× real-time).")
    if renderer is not None:
        imageio.mimsave(args.video, frames, fps=args.video_fps)
        print(f"Wrote {args.video} ({len(frames)} frames).")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unitree Go1 convex-MPC in MuJoCo")
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--gait-period", type=float, default=0.5)
    p.add_argument("--duty", type=float, default=0.6)
    p.add_argument("--swing-height", type=float, default=0.06)
    p.add_argument("--horizon", type=int, default=14)
    p.add_argument("--mpc-dt", type=float, default=0.02)
    p.add_argument("--mu", type=float, default=0.6)
    p.add_argument("--body-height", type=float, default=0.27)
    p.add_argument("--duration", type=float, default=0.0,
                   help="Sim duration in seconds; 0 = run until viewer closed.")
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--video-fps", type=int, default=30)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.no_viewer:
        if args.duration <= 0:
            args.duration = 5.0
        run_headless(args)
    else:
        run_viewer(args)
