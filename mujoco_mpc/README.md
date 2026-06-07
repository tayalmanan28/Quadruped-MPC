# MuJoCo Quadruped MPC — Unitree Go1

Convex Model Predictive Control (MPC) locomotion controller for the
**Unitree Go1** in MuJoCo. Robot model comes from
[`google-deepmind/mujoco_menagerie`](https://github.com/google-deepmind/mujoco_menagerie),
MPC formulation is the convex GRF QP from Di Carlo *et al.* (MIT Cheetah, IROS
2018) — same approach used by
[`tayalmanan28/Quadruped-MPC`](https://github.com/tayalmanan28/Quadruped-MPC).

## What is implemented

- **Robot model**: `models/go1/` (vendored from `mujoco_menagerie`, BSD-3).
- **Gait scheduler**: trot (configurable period & duty factor).
- **Footstep planner**: yaw-aware Raibert heuristic.
- **Swing controller**: Bezier vertical profile + Cartesian PD on each foot.
- **Stance controller**: convex MPC (OSQP) that outputs ground-reaction forces,
  mapped to joint torques via the foot Jacobian transpose.
- **Sim loop**: 1 kHz physics, 50 Hz MPC, optional `mujoco.viewer` GUI.
- **Demos**: tracking trials and an obstacle-slalom navigation video.

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
# Trot in place
python main.py

# Forward trot at 0.4 m/s with viewer
python main.py --vx 0.4

# Headless 5 s run with a video
python main.py --vx 0.3 --no-viewer --duration 5 --video out.mp4

# Tracking demos
MUJOCO_GL=osmesa python make_videos.py

# Slalom with obstacles, pure-pursuit nav using only (vx, wz)
MUJOCO_GL=osmesa python make_obstacle_video.py
```

CLI flags: `--vx`, `--vy`, `--wz`, `--duration`, `--no-viewer`, `--video PATH`.

## Layout

```
models/go1/                Go1 MJCF (from mujoco_menagerie)
  scene_obstacles.xml      Slalom scene for the obstacle demo
mpc/
  robot.py                 MuJoCo wrapper + leg / foot bookkeeping
  gait.py                  Periodic contact schedule (trot)
  swing.py                 Swing-foot Bezier + yaw-aware Raibert
  convex_mpc.py            Convex MPC (OSQP, condensed QP on GRFs)
  controller.py            Top-level locomotion controller
main.py                    Simulation entry point
make_videos.py             Renders the tracking demo videos
make_obstacle_video.py     Renders the obstacle-slalom video
```

## Tuning notes

- `ControllerConfig.inertia_scale_xyz` (default `(1.0, 2.0, 10.0)`): per-axis
  multiplier on the trunk inertia used by the MPC. Trunk-only values
  under-estimate effective rotational inertia — especially `I_zz` because the
  four legs spread out below and to the sides of the trunk. Without this scale,
  yaw commands barely produce any rotation.
- `ControllerConfig.vel_tau` (default `0.01` s): first-order filter time
  constant on the velocity reference inside the MPC horizon. Larger values
  smooth velocity step-commands but reduce tracking gain.

## References

- J. Di Carlo *et al.*, "Dynamic Locomotion in the MIT Cheetah 3 Through
  Convex Model-Predictive Control," IROS 2018.
- M. Tayal, "Quadruped-MPC": <https://github.com/tayalmanan28/Quadruped-MPC>
- Google DeepMind, "MuJoCo Menagerie":
  <https://github.com/google-deepmind/mujoco_menagerie>
