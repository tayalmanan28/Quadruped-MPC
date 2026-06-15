"""Run the G1 balance controller in MuJoCo.

Usage::

    python main.py                 # viewer
    python main.py --no-viewer     # headless, prints state every second
    python main.py --no-viewer --video out.mp4
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco

from g1_mpc import G1Controller, G1Robot

MODEL = Path(__file__).parent / "models" / "g1" / "scene.xml"


def build():
    model = mujoco.MjModel.from_xml_path(str(MODEL))
    data = mujoco.MjData(model)
    robot = G1Robot(model, data)
    robot.reset_to_home()
    model.opt.timestep = 0.001
    ctrl = G1Controller(robot)
    return model, data, robot, ctrl


def run_viewer(duration: float):
    import mujoco.viewer
    model, data, robot, ctrl = build()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1
        sim_start = time.time(); t0 = data.time
        while viewer.is_running():
            t = data.time
            if duration > 0 and t - t0 > duration:
                break
            ctrl.update(t)
            mujoco.mj_step(model, data)
            viewer.sync()
            elapsed = time.time() - sim_start
            target = data.time - t0
            if target > elapsed:
                time.sleep(target - elapsed)


def run_headless(duration: float, video: str | None, fps: int):
    import os
    if video:
        os.environ.setdefault("MUJOCO_GL", "osmesa")
        import imageio
    model, data, robot, ctrl = build()
    n_steps = int(duration / model.opt.timestep)

    if video:
        renderer = mujoco.Renderer(model, height=480, width=640)
        save_every = max(1, int(round(1.0 / (model.opt.timestep * fps))))
        frames = []
    else:
        renderer = None; save_every = 0; frames = []

    t_wall = time.time()
    last_log = -1.0
    for k in range(n_steps):
        ctrl.update(data.time)
        mujoco.mj_step(model, data)
        if data.time - last_log >= 1.0:
            last_log = data.time
            print(f"t={data.time:.2f}  pelvis_z={robot.base_pos()[2]:.3f}  "
                  f"rpy={robot.base_rpy().round(3)}  "
                  f"com={robot.com_world().round(3)}")
        if renderer is not None and (k % save_every == 0):
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render())
    print(f"Sim {duration:.2f}s in {time.time() - t_wall:.2f}s wall.")
    if video:
        imageio.mimsave(video, frames, fps=fps, codec="libx264", quality=8)
        print(f"Wrote {video} ({len(frames)} frames)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--video", type=str, default="")
    p.add_argument("--video-fps", type=int, default=30)
    args = p.parse_args()
    if args.no_viewer:
        if args.duration <= 0: args.duration = 5.0
        run_headless(args.duration, args.video or None, args.video_fps)
    else:
        run_viewer(args.duration)


if __name__ == "__main__":
    main()
