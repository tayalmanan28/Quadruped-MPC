"""Side-by-side tracking comparison between our convex MPC and the playground
PPO policy on the G1 joystick task.

Both controllers are tested on the same command sweep. Per trial we report the
achieved (vx, vy, wz) in the pelvis local frame, averaged after a 1.5 s
warm-up. Body height and whether the robot fell are also reported.

Note: the two controllers run on **different** MuJoCo models — the MPC uses
``models/g1/scene.xml`` (full menagerie collision), the policy uses the
``feetonly`` MJCF the playground env was trained on. Within each controller
the same model is used across all trials, so within-controller comparisons
are apples-to-apples and the cross-controller table is "best each can do
on its native model".
"""
from __future__ import annotations

import argparse
import os
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import mujoco
import numpy as np

# MPC stack
from g1_mpc import G1Robot, G1Controller
from g1_mpc.controller import G1ControllerConfig

# RL inference
from eval_g1_playground import G1PolicyRunner


TRIALS = [
    ("stand",        0.0,  0.0,  0.0),
    ("fwd 0.10",     0.10, 0.0,  0.0),
    ("fwd 0.20",     0.20, 0.0,  0.0),
    ("fwd 0.40",     0.40, 0.0,  0.0),
    ("lat +0.20",    0.0,  0.20, 0.0),
    ("lat -0.20",    0.0, -0.20, 0.0),
    ("yaw +0.50",    0.0,  0.0,  0.50),
    ("yaw -0.50",    0.0,  0.0, -0.50),
    ("fwd+yaw",      0.2,  0.0,  0.3),
    ("fwd+lat+yaw",  0.2,  0.1,  0.2),
]


def _measure(get_state, set_cmd, step, reset, ctrl_dt, T=6.0, warmup=1.5):
    """Generic measurement driver. ``get_state()`` returns dict with keys
    base_z, rpy, com_vel_world (or base lin vel), ang_vel_world."""
    reset()
    vs, ws = [], []
    fell = False
    n = int(T / ctrl_dt)
    for k in range(n):
        step()
        st = get_state()
        if st["base_z"] < 0.35:
            fell = True
            break
        if k * ctrl_dt > warmup:
            yaw = st["rpy"][2]
            cy, sy = np.cos(yaw), np.sin(yaw)
            v_w = st["v_world"]
            vs.append([cy * v_w[0] + sy * v_w[1], -sy * v_w[0] + cy * v_w[1]])
            ws.append(st["w_world"][2])
    if fell or not vs:
        return dict(fell=True, vx=np.nan, vy=np.nan, wz=np.nan,
                    base_z=float(st["base_z"]))
    vs = np.array(vs); ws = np.array(ws)
    return dict(fell=False, vx=float(vs[:, 0].mean()),
                vy=float(vs[:, 1].mean()), wz=float(ws.mean()),
                base_z=float(st["base_z"]))


def run_mpc(cmd_vx, cmd_vy, cmd_wz, T=6.0):
    m = mujoco.MjModel.from_xml_path("models/g1/scene.xml")
    d = mujoco.MjData(m)
    r = G1Robot(m, d)
    r.reset_to_walk_pose()
    m.opt.timestep = 0.001
    c = G1Controller(r, ctrl_cfg=G1ControllerConfig(
        mode="walk", cmd_vx=cmd_vx, cmd_vy=cmd_vy, cmd_wz=cmd_wz))

    def reset():
        r.reset_to_walk_pose()

    def step():
        c.update(d.time)
        mujoco.mj_step(m, d)

    def get_state():
        return dict(
            base_z=r.base_pos()[2],
            rpy=r.base_rpy(),
            v_world=r.com_vel_world(),
            w_world=r.base_ang_vel_world(),
        )

    return _measure(get_state, None, step, reset, ctrl_dt=m.opt.timestep, T=T)


def run_rl(pr: "G1PolicyRunner", cmd_vx, cmd_vy, cmd_wz, T=6.0):
    pr.set_command(cmd_vx, cmd_vy, cmd_wz)

    def reset():
        pr.reset()
        pr.set_command(cmd_vx, cmd_vy, cmd_wz)

    def step():
        pr.step()

    def get_state():
        return dict(
            base_z=pr.base_pos()[2],
            rpy=pr.base_rpy(),
            v_world=pr.base_lin_vel_world(),
            w_world=pr.base_ang_vel_world(),
        )

    return _measure(get_state, None, step, reset, ctrl_dt=pr.ctrl_dt, T=T)


def fmt(res, ref):
    if res["fell"]:
        return f"FELL  (z={res['base_z']:.2f})"
    refv = max(abs(ref), 1e-6)
    pct = 100 * (res["vx"] if abs(ref) > 0 else 1.0) / refv if False else None
    return (f"vx={res['vx']:+.3f}  vy={res['vy']:+.3f}  wz={res['wz']:+.3f}  "
            f"z={res['base_z']:.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/g1_joystick/final.pkl")
    p.add_argument("--duration", type=float, default=6.0)
    args = p.parse_args()

    print("Loading trained policy…", flush=True)
    pr = G1PolicyRunner(args.ckpt)
    print("Policy loaded.", flush=True)

    rows = []
    print(f"\n{'trial':14s}  {'cmd (vx,vy,wz)':22s}  | {'MPC':50s}  | RL", flush=True)
    print("-" * 130, flush=True)
    for name, vx, vy, wz in TRIALS:
        mpc_res = run_mpc(vx, vy, wz, T=args.duration)
        rl_res = run_rl(pr, vx, vy, wz, T=args.duration)
        rows.append((name, (vx, vy, wz), mpc_res, rl_res))
        cmd_str = f"({vx:+.2f},{vy:+.2f},{wz:+.2f})"
        print(f"{name:14s}  {cmd_str:22s}  | {fmt(mpc_res, vx):50s}  | {fmt(rl_res, vx)}",
              flush=True)

    # Summary scoring: mean |error| per axis.
    def score(side):
        errs_vx, errs_vy, errs_wz, fails = [], [], [], 0
        for _, (vx, vy, wz), mpc, rl in rows:
            r = mpc if side == "mpc" else rl
            if r["fell"]:
                fails += 1; continue
            errs_vx.append(abs(r["vx"] - vx))
            errs_vy.append(abs(r["vy"] - vy))
            errs_wz.append(abs(r["wz"] - wz))
        return (np.mean(errs_vx) if errs_vx else np.nan,
                np.mean(errs_vy) if errs_vy else np.nan,
                np.mean(errs_wz) if errs_wz else np.nan, fails)

    mpc_evx, mpc_evy, mpc_ewz, mpc_fails = score("mpc")
    rl_evx,  rl_evy,  rl_ewz,  rl_fails  = score("rl")
    print()
    print("=== Mean absolute tracking error ===")
    print(f"            |  err_vx   err_vy   err_wz   falls")
    print(f"MPC         |  {mpc_evx:.3f}    {mpc_evy:.3f}    {mpc_ewz:.3f}    {mpc_fails}/{len(TRIALS)}")
    print(f"RL (Playgr) |  {rl_evx:.3f}    {rl_evy:.3f}    {rl_ewz:.3f}    {rl_fails}/{len(TRIALS)}")


if __name__ == "__main__":
    main()
