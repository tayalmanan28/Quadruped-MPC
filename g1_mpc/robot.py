"""Wrapper around the Unitree G1 MuJoCo model.

Exposes:
 - per-leg joint addresses (6 DOFs per leg: hip pitch/roll/yaw, knee,
   ankle pitch/roll),
 - upper-body joint addresses (3 waist + 7+7 arm DOFs),
 - **per-foot 4 contact corners**: small spheres at the four corners of
   the rectangular foot patch, used both for contact detection and as the
   MPC's contact-force application points,
 - per-corner Jacobian queries (3 x n_legdof) for the GRF→torque mapping,
 - the home-pose qpos (used both as initial state and as the upper-body
   PD target).
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

LEG_NAMES = ("left", "right")
LEG_JOINT_SUFFIXES = (
    "hip_pitch_joint",
    "hip_roll_joint",
    "hip_yaw_joint",
    "knee_joint",
    "ankle_pitch_joint",
    "ankle_roll_joint",
)
WAIST_JOINTS = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
ARM_JOINT_SUFFIXES = (
    "shoulder_pitch_joint",
    "shoulder_roll_joint",
    "shoulder_yaw_joint",
    "elbow_joint",
    "wrist_roll_joint",
    "wrist_pitch_joint",
    "wrist_yaw_joint",
)


@dataclass
class LegIndices:
    name: str
    joint_ids: np.ndarray       # (6,) joint ids
    qpos_adr: np.ndarray
    qvel_adr: np.ndarray
    actuator_ids: np.ndarray    # (6,)
    foot_site_id: int
    ankle_body_id: int
    corner_geom_ids: np.ndarray  # (4,) geom ids for foot corners


class G1Robot:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data

        # Disable stock position servos so we drive torques via qfrc_applied.
        # G1 uses the same affine gain / position bias type as Go1.
        self.model.actuator_gainprm[:, 0] = 0.0
        self.model.actuator_biasprm[:, 1] = 0.0

        self.pelvis_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        if self.pelvis_body_id < 0:
            raise RuntimeError("pelvis body not found")
        trunk_jnt = model.body_jntadr[self.pelvis_body_id]
        self.base_qpos_adr = int(model.jnt_qposadr[trunk_jnt])  # 7 (xyz+quat)
        self.base_qvel_adr = int(model.jnt_dofadr[trunk_jnt])   # 6

        self.legs: list[LegIndices] = [self._build_leg(name) for name in LEG_NAMES]
        self.upper_body_joint_ids, self.upper_body_qpos_adr, \
            self.upper_body_qvel_adr, self.upper_body_actuator_ids = \
            self._build_upper_body()

        self.total_mass = float(np.sum(model.body_mass))
        self.pelvis_inertia = np.asarray(model.body_inertia[self.pelvis_body_id], dtype=float)
        self.composite_inertia = self._compute_composite_inertia()

        # Stash the home-pose qpos for upper-body PD targets.
        self.home_qpos = self._fetch_home_qpos()

        self._jacp = np.zeros((3, model.nv))
        self._jacr = np.zeros((3, model.nv))

    # ------------------------------------------------------------------
    def _build_leg(self, name: str) -> LegIndices:
        m = self.model
        jids, qpos_adr, qvel_adr, act_ids = [], [], [], []
        for suf in LEG_JOINT_SUFFIXES:
            jn = f"{name}_{suf}"
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint {jn!r} not found")
            jids.append(jid)
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            qvel_adr.append(int(m.jnt_dofadr[jid]))
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if aid < 0:
                raise RuntimeError(f"actuator {jn!r} not found")
            act_ids.append(aid)
        site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{name}_foot")
        if site_id < 0:
            raise RuntimeError(f"site {name}_foot not found")
        ankle_body_id = mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, f"{name}_ankle_roll_link"
        )
        if ankle_body_id < 0:
            raise RuntimeError(f"body {name}_ankle_roll_link not found")
        # Foot corner geoms are all unnamed children of ankle_roll_link with
        # type==sphere; collect them in a consistent corner order
        # (rear-left, rear-right, front-left, front-right) based on their
        # local positions.
        corner_geom_ids = []
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == ankle_body_id and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                corner_geom_ids.append(g)
        if len(corner_geom_ids) != 4:
            raise RuntimeError(
                f"{name}: expected 4 foot-corner sphere geoms, got {len(corner_geom_ids)}"
            )
        # Sort by local pos: first by x (rear<front), then by y (right<left).
        corner_geom_ids = sorted(
            corner_geom_ids,
            key=lambda g: (m.geom_pos[g][0], m.geom_pos[g][1]),
        )
        return LegIndices(
            name=name,
            joint_ids=np.array(jids, dtype=int),
            qpos_adr=np.array(qpos_adr, dtype=int),
            qvel_adr=np.array(qvel_adr, dtype=int),
            actuator_ids=np.array(act_ids, dtype=int),
            foot_site_id=site_id,
            ankle_body_id=ankle_body_id,
            corner_geom_ids=np.array(corner_geom_ids, dtype=int),
        )

    def _build_upper_body(self):
        m = self.model
        joints = list(WAIST_JOINTS) + \
                 [f"left_{s}" for s in ARM_JOINT_SUFFIXES] + \
                 [f"right_{s}" for s in ARM_JOINT_SUFFIXES]
        jids, qpos_adr, qvel_adr, act_ids = [], [], [], []
        for jn in joints:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jn)
            if jid < 0:
                raise RuntimeError(f"joint {jn} not found")
            jids.append(jid)
            qpos_adr.append(int(m.jnt_qposadr[jid]))
            qvel_adr.append(int(m.jnt_dofadr[jid]))
            aid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, jn)
            if aid < 0:
                raise RuntimeError(f"actuator {jn} not found")
            act_ids.append(aid)
        return (np.array(jids, dtype=int), np.array(qpos_adr, dtype=int),
                np.array(qvel_adr, dtype=int), np.array(act_ids, dtype=int))

    def _fetch_home_qpos(self) -> np.ndarray:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        if kid < 0:
            raise RuntimeError("keyframe 'stand' not found")
        return np.array(self.model.key_qpos[kid], dtype=float).copy()

    # ------------------------------------------------------------------
    def reset_to_home(self) -> None:
        kid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        mujoco.mj_forward(self.model, self.data)

    def reset_to_walk_pose(
        self,
        hip_pitch: float = -0.4,
        knee: float = 0.8,
        ankle_pitch: float = -0.4,
        pelvis_z: float = 0.74,
    ) -> None:
        """Reset to a **bent-knee** walking-ready pose. The home keyframe has
        knees at 0 — a singular configuration where the foot Jacobian
        vertical column is near zero, so the swing leg cannot lift. This
        method overrides the leg joints to a moderate crouch and lowers the
        pelvis so the feet rest on the ground.

        Defaults match the deeper crouch used by ``mujoco_playground`` G1
        (their ``knees_bent`` keyframe + ``base_height_target = 0.5``-ish)
        which makes the foot Jacobian much better conditioned for swing."""
        self.reset_to_home()
        for li, leg in enumerate(self.legs):
            # qpos order matches LEG_JOINT_SUFFIXES:
            # [hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll]
            self.data.qpos[leg.qpos_adr[0]] = hip_pitch
            self.data.qpos[leg.qpos_adr[3]] = knee
            self.data.qpos[leg.qpos_adr[4]] = ankle_pitch
        self.data.qpos[self.base_qpos_adr + 2] = pelvis_z
        mujoco.mj_forward(self.model, self.data)
        # Update the cached home_qpos so the controller's posture PD uses the
        # walk pose as its target.
        self.home_qpos = self.data.qpos.copy()

    # ------------------------------------------------------------------
    def base_pos(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos_adr : self.base_qpos_adr + 3].copy()

    def base_quat_wxyz(self) -> np.ndarray:
        return self.data.qpos[self.base_qpos_adr + 3 : self.base_qpos_adr + 7].copy()

    def base_rotmat(self) -> np.ndarray:
        return self.data.xmat[self.pelvis_body_id].reshape(3, 3).copy()

    def base_lin_vel_world(self) -> np.ndarray:
        return self.data.qvel[self.base_qvel_adr : self.base_qvel_adr + 3].copy()

    def base_ang_vel_world(self) -> np.ndarray:
        # Free-joint angular qvel is body-frame; rotate.
        w_body = self.data.qvel[self.base_qvel_adr + 3 : self.base_qvel_adr + 6]
        return self.base_rotmat() @ w_body

    def base_rpy(self) -> np.ndarray:
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

    def com_world(self) -> np.ndarray:
        """Whole-body CoM position in the world frame."""
        # mj_forward populates subtree_com for body 0 (world)
        return np.asarray(self.data.subtree_com[0], dtype=float).copy()

    def com_vel_world(self) -> np.ndarray:
        """Whole-body CoM linear velocity in world frame (via subtree_linvel)."""
        # subtree_linvel may need mj_subtreeVel; use that for correctness.
        mujoco.mj_subtreeVel(self.model, self.data)
        return np.asarray(self.data.subtree_linvel[0], dtype=float).copy()

    # ------------------------------------------------------------------
    def all_corner_positions(self) -> np.ndarray:
        """World-frame positions of all 8 foot corners, shape (8, 3),
        ordered ``[left 4 corners, right 4 corners]``."""
        out = np.zeros((8, 3))
        for li, leg in enumerate(self.legs):
            for ci, gid in enumerate(leg.corner_geom_ids):
                out[li * 4 + ci] = self.data.geom_xpos[gid]
        return out

    def corner_jacobians(self) -> list[np.ndarray]:
        """For each of the 8 corners return the 3x6 translational Jacobian
        with respect to that leg's six joints (corners 0..3 -> left leg,
        4..7 -> right leg)."""
        out = []
        for li, leg in enumerate(self.legs):
            for gid in leg.corner_geom_ids:
                p = self.data.geom_xpos[gid]
                mujoco.mj_jac(
                    self.model, self.data,
                    self._jacp, self._jacr,
                    p, leg.ankle_body_id,
                )
                out.append(self._jacp[:, leg.qvel_adr].copy())
        return out

    def corner_in_contact(self, corner_global_idx: int) -> bool:
        leg = self.legs[corner_global_idx // 4]
        gid = leg.corner_geom_ids[corner_global_idx % 4]
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            if c.geom1 == gid or c.geom2 == gid:
                return True
        return False

    def foot_site_pos(self, leg_idx: int) -> np.ndarray:
        sid = self.legs[leg_idx].foot_site_id
        return self.data.site_xpos[sid].copy()

    def foot_site_jacobian(self, leg_idx: int) -> np.ndarray:
        """3x6 translational Jacobian of the foot site w.r.t. that leg's joints."""
        leg = self.legs[leg_idx]
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, leg.foot_site_id
        )
        return self._jacp[:, leg.qvel_adr].copy()

    def foot_site_vel(self, leg_idx: int) -> np.ndarray:
        leg = self.legs[leg_idx]
        mujoco.mj_jacSite(
            self.model, self.data, self._jacp, self._jacr, leg.foot_site_id
        )
        return self._jacp @ self.data.qvel

    # ------------------------------------------------------------------
    # Torque output
    # ------------------------------------------------------------------
    def apply_torques(self, leg_torques: dict[int, np.ndarray],
                      upper_torques: np.ndarray) -> None:
        """leg_torques: dict {leg_idx -> (6,) joint torques}. upper_torques:
        (17,) for waist + arms in the order defined by ``upper_body_*``."""
        self.data.qfrc_applied[:] = 0.0
        for li, tau in leg_torques.items():
            self.data.qfrc_applied[self.legs[li].qvel_adr] = tau
        self.data.qfrc_applied[self.upper_body_qvel_adr] = upper_torques

    # ------------------------------------------------------------------
    def _compute_composite_inertia(self) -> np.ndarray:
        """Composite inertia of the whole robot about the pelvis frame origin,
        expressed in pelvis body axes, in the home pose."""
        m, d = self.model, mujoco.MjData(self.model)
        kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "stand")
        mujoco.mj_resetDataKeyframe(m, d, kid)
        mujoco.mj_forward(m, d)
        p_pelvis = d.xpos[self.pelvis_body_id].copy()
        R_pelvis = d.xmat[self.pelvis_body_id].reshape(3, 3).copy()
        I_world = np.zeros((3, 3))
        for b in range(m.nbody):
            if b == 0:
                continue
            mass = float(m.body_mass[b])
            p = d.xipos[b]
            R = d.ximat[b].reshape(3, 3)
            I_local = np.diag(m.body_inertia[b])
            I_b_world = R @ I_local @ R.T
            r = p - p_pelvis
            I_world += I_b_world + mass * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
        return R_pelvis.T @ I_world @ R_pelvis
