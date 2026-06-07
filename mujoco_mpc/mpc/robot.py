"""Thin wrapper around the Go1 MuJoCo model.

Provides convenient access to:
  - per-leg joint / actuator / foot-site indices,
  - base pose / twist (with the body→world correction for free-joint angular qvel),
  - per-foot world position, velocity, Jacobian,
  - a helper to inject pure joint torques via ``data.qfrc_applied``,
  - the full composite rigid-body inertia of the robot in the home pose.

Conventions
-----------
Leg ordering everywhere in this codebase is ``["FR", "FL", "RR", "RL"]``
(front-right, front-left, rear-right, rear-left), matching the joint
order of the menagerie Go1 model.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

LEG_NAMES = ("FR", "FL", "RR", "RL")
JOINT_SUFFIXES = ("hip_joint", "thigh_joint", "calf_joint")
ACTUATOR_SUFFIXES = ("hip", "thigh", "calf")

# Nominal hip-joint offsets in the trunk frame (read from go1.xml).
# Used as a reference for footstep planning (Raibert heuristic).
HIP_OFFSETS_BODY = np.array(
    [
        [+0.1881, -0.04675, 0.0],  # FR
        [+0.1881, +0.04675, 0.0],  # FL
        [-0.1881, -0.04675, 0.0],  # RR
        [-0.1881, +0.04675, 0.0],  # RL
    ]
)


@dataclass
class LegIndices:
    """Cached MuJoCo indices for one leg."""

    name: str
    joint_ids: np.ndarray
    qpos_adr: np.ndarray
    qvel_adr: np.ndarray
    actuator_ids: np.ndarray
    foot_site_id: int
    foot_geom_id: int


class Go1Robot:
    """Convenience wrapper around an `mjModel` / `mjData` pair."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Disable the stock position servos so we can drive joint torques
        # ourselves through ``data.qfrc_applied``. The actuators remain in the
        # model (so ``ctrl`` still exists) but they output zero force.
        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, 1] = 0.0

        self.trunk_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "trunk"
        )
        if self.trunk_body_id < 0:
            raise RuntimeError("trunk body not found")

        trunk_jnt = model.body_jntadr[self.trunk_body_id]
        self.base_qpos_adr = int(model.jnt_qposadr[trunk_jnt])
        self.base_qvel_adr = int(model.jnt_dofadr[trunk_jnt])

        self.legs: list[LegIndices] = [self._build_leg(name) for name in LEG_NAMES]

        self.trunk_mass = float(model.body_mass[self.trunk_body_id])
        self.total_mass = float(np.sum(model.body_mass))
        self.trunk_inertia = np.asarray(
            model.body_inertia[self.trunk_body_id], dtype=float
        )
        self.composite_inertia = self._compute_composite_inertia()
        self.composite_inertia_diag = np.diag(self.composite_inertia).copy()

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------
    def _build_leg(self, name: str) -> LegIndices:
        m = self.model
        jids, qpos_adr, qvel_adr = [], [], []
        for suf in JOINT_SUFFIXES:
            jname = f"{name}_{suf}"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise RuntimeError(f"joint {jname!r} not found")
            jids.append(jid)
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            qvel_adr.append(int(m.jnt_dofadr[jid]))
        act_ids = []
        for suf in ACTUATOR_SUFFIXES:
            aname = f"{name}_{suf}"
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, aname)
            if aid < 0:
                raise RuntimeError(f"actuator {aname!r} not found")
            act_ids.append(aid)
        foot_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
        foot_geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if foot_site < 0 or foot_geom < 0:
            raise RuntimeError(f"foot site / geom {name!r} not found")
        return LegIndices(
            name=name,
            joint_ids=np.array(jids, dtype=int),
            qpos_adr=np.array(qpos_adr, dtype=int),
            qvel_adr=np.array(qvel_adr, dtype=int),
            actuator_ids=np.array(act_ids, dtype=int),
            foot_site_id=foot_site,
            foot_geom_id=foot_geom,
        )

    # ------------------------------------------------------------------
    # Base state
    # ------------------------------------------------------------------
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3].copy()

    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7].copy()

    def base_rotmat(self) -> np.ndarray:
        return self.data.xmat[self.trunk_body_id].reshape(3, 3).copy()

    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[self.base_qvel_adr : self.base_qvel_adr + 3].copy()

    def base_ang_vel_world(self) -> np.ndarray:
        # MuJoCo's free-joint qvel angular block is expressed in the BODY
        # frame; rotate to the world frame.
        w_body = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]
        return self.base_rotmat() @ w_body

    def base_rpy(self) -> np.ndarray:
        """Roll/Pitch/Yaw (ZYX intrinsic) from the base quaternion (rad)."""
        w, x, y, z = self.base_quat_wxyz()
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch = np.arcsin(sinp)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        return np.array([roll, pitch, yaw])

    # ------------------------------------------------------------------
    # Per-foot state
    # ------------------------------------------------------------------
    def foot_pos_world(self, leg_idx: int) -> np.ndarray:
        sid = self.legs[leg_idx].foot_site_id
        return self.data.site_xpos[sid].copy()

    def foot_vel_world(self, leg_idx: int) -> np.ndarray:
        sid = self.legs[leg_idx].foot_site_id
        mujoco.mj_jacSite(self.model, self.data, self._jacp, self._jacr, sid)
        return self._jacp @ self.data.qvel

    def foot_jacobian_leg(self, leg_idx: int) -> np.ndarray:
        """3x3 translational Jacobian of the foot site w.r.t. that leg's joints
        (in world coordinates)."""
        leg = self.legs[leg_idx]
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, leg.foot_site_id
        )
        return self._jacp[:, leg.qvel_adr].copy()

    def foot_in_contact(self, leg_idx: int) -> bool:
        gid = self.legs[leg_idx].foot_geom_id
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return True
        return False

    # ------------------------------------------------------------------
    # Joint state
    # ------------------------------------------------------------------
    def leg_qpos(self, leg_idx: int) -> np.ndarray:
        return self.data.qpos[self.legs[leg_idx].qpos_adr].copy()

    def leg_qvel(self, leg_idx: int) -> np.ndarray:
        return self.data.qvel[self.legs[leg_idx].qvel_adr].copy()

    # ------------------------------------------------------------------
    # Torque output
    # ------------------------------------------------------------------
    def apply_joint_torques(self, torques: np.ndarray) -> None:
        """Write a (12,) torque vector — order [FR, FL, RR, RL] × (hip, thigh,
        calf) — into ``data.qfrc_applied`` for the leg DOFs and zero the rest."""
        self.data.qfrc_applied[:] = 0.0
        for i, leg in enumerate(self.legs):
            self.data.qfrc_applied[leg.qvel_adr] = torques[3 * i : 3 * i + 3]

    # ------------------------------------------------------------------
    def reset_to_home(self) -> None:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if kid < 0:
            raise RuntimeError("keyframe 'home' not found")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------
    def _compute_composite_inertia(self) -> np.ndarray:
        """Composite rigid-body inertia of the whole robot about the trunk
        frame's origin, expressed in trunk body axes. Computed once in the
        ``home`` keyframe. The convex MPC uses only a scaled version of the
        trunk inertia in practice, but this is useful for reference."""
        m, d = self.model, mujoco.MjData(self.model)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        trunk_pos = d.xpos[self.trunk_body_id].copy()
        trunk_R = d.xmat[self.trunk_body_id].reshape(3, 3).copy()
        I_world = np.zeros((3, 3))
        for b in range(m.nbody):
            if b == 0:
                continue
            mass = float(m.body_mass[b])
            p = d.xipos[b]
            R = d.ximat[b].reshape(3, 3)
            I_local = np.diag(m.body_inertia[b])
            I_b_world = R @ I_local @ R.T
            r = p - trunk_pos
            I_world += I_b_world + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return trunk_R.T @ I_world @ trunk_R
