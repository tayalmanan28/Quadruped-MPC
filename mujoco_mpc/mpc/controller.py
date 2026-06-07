"""Top-level locomotion controller wiring the gait, MPC, and swing controllers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .convex_mpc import ConvexMPC, MPCConfig, GRAVITY
from .gait import TrotGait
from .robot import HIP_OFFSETS_BODY, Go1Robot
from .swing import SwingLegController, raibert_footstep


@dataclass
class ControllerConfig:
    body_height: float = 0.27
    swing_height: float = 0.06
    raibert_k: float = 0.03
    mpc_rate_hz: float = 100.0
    # Velocity reference filter time-constant (s). Larger values smooth velocity
    # step-commands but reduce tracking gain. ~0.01 is essentially "no filter".
    vel_tau: float = 0.01
    # Per-axis multiplier on the trunk inertia used by the MPC. Trunk-only
    # values massively underestimate ``I_yy`` and especially ``I_zz`` because
    # the four legs spread out below and to the sides of the trunk. The yaw
    # multiplier in particular makes yaw commands actually track — dropping
    # it back to 1.0 gives only ~5% of commanded yaw rate.
    inertia_scale_xyz: tuple = (1.0, 2.0, 18.0)
    # Body-frame velocity command
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0
    swing_kp: tuple = (700.0, 700.0, 350.0)
    swing_kd: tuple = (12.0, 12.0, 10.0)
    swing_max_force: float = 200.0
    tau_max: float = 33.5


class LocomotionController:
    """Convex-MPC trot controller for the Unitree Go1."""

    def __init__(
        self,
        robot: Go1Robot,
        gait: TrotGait | None = None,
        mpc_cfg: MPCConfig | None = None,
        ctrl_cfg: ControllerConfig | None = None,
    ):
        self.robot = robot
        self.gait = gait or TrotGait()
        self.cfg = ctrl_cfg or ControllerConfig()
        I_body = robot.trunk_inertia * np.asarray(self.cfg.inertia_scale_xyz)
        self.mpc = ConvexMPC(
            mass=robot.total_mass,
            inertia_body=I_body,
            cfg=mpc_cfg,
        )
        self.swings = [
            SwingLegController(
                kp=self.cfg.swing_kp,
                kd=self.cfg.swing_kd,
                max_force=self.cfg.swing_max_force,
            )
            for _ in range(4)
        ]
        self._last_mpc_t = -np.inf
        self._mpc_period = 1.0 / self.cfg.mpc_rate_hz
        self._last_grfs = np.zeros(12)
        self._ground_z = 0.0
        self._was_stance = np.ones(4, dtype=bool)

    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self.cfg.cmd_vx = float(vx)
        self.cfg.cmd_vy = float(vy)
        self.cfg.cmd_wz = float(wz)

    def _build_reference(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        N = self.mpc.cfg.horizon
        dt = self.mpc.cfg.dt
        rpy = self.robot.base_rpy()
        yaw = rpy[2]
        p = self.robot.base_pos()
        v_w = self.robot.base_lin_vel_world()
        w_w = self.robot.base_ang_vel_world()

        cy, sy = np.cos(yaw), np.sin(yaw)
        v_cmd_world = np.array(
            [
                cy * self.cfg.cmd_vx - sy * self.cfg.cmd_vy,
                sy * self.cfg.cmd_vx + cy * self.cfg.cmd_vy,
                0.0,
            ]
        )
        w_cmd_world = np.array([0.0, 0.0, self.cfg.cmd_wz])

        x0 = np.zeros(13)
        x0[0:3] = rpy
        x0[3:6] = p
        x0[6:9] = w_w
        x0[9:12] = v_w
        x0[12] = GRAVITY

        tau = max(self.cfg.vel_tau, dt)
        x_refs = np.zeros((N, 13))
        yaw_acc = yaw
        px_acc, py_acc = p[0], p[1]
        v_ref_x, v_ref_y = v_w[0], v_w[1]
        w_ref_z = w_w[2]
        for k in range(N):
            a = dt / tau
            v_ref_x += a * (v_cmd_world[0] - v_ref_x)
            v_ref_y += a * (v_cmd_world[1] - v_ref_y)
            w_ref_z += a * (w_cmd_world[2] - w_ref_z)
            yaw_acc += w_ref_z * dt
            px_acc += v_ref_x * dt
            py_acc += v_ref_y * dt
            x_refs[k, 0] = 0.0
            x_refs[k, 1] = 0.0
            x_refs[k, 2] = yaw_acc
            x_refs[k, 3] = px_acc
            x_refs[k, 4] = py_acc
            x_refs[k, 5] = self.cfg.body_height
            x_refs[k, 8] = w_ref_z
            x_refs[k, 9] = v_ref_x
            x_refs[k, 10] = v_ref_y
            x_refs[k, 12] = GRAVITY
        return x0, x_refs

    def _foot_positions_rel(self) -> np.ndarray:
        p_base = self.robot.base_pos()
        out = np.zeros((4, 3))
        for i in range(4):
            out[i] = self.robot.foot_pos_world(i) - p_base
        return out

    def _update_swing_targets(self, t: float) -> None:
        p_base = self.robot.base_pos()
        v_w = self.robot.base_lin_vel_world()
        yaw = self.robot.base_rpy()[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        v_cmd_world = np.array(
            [
                cy * self.cfg.cmd_vx - sy * self.cfg.cmd_vy,
                sy * self.cfg.cmd_vx + cy * self.cfg.cmd_vy,
                0.0,
            ]
        )
        T_st = self.gait.stance_duration()
        T_sw = self.gait.swing_duration()
        for i in range(4):
            if not self.gait.in_stance(t, i):
                t_td = (1.0 - self.gait.swing_progress(t, i)) * T_sw
                target = raibert_footstep(
                    base_pos_world=p_base,
                    base_yaw=yaw,
                    hip_offset_body=HIP_OFFSETS_BODY[i],
                    base_vel_world=v_w,
                    cmd_vel_world=v_cmd_world,
                    cmd_wz=self.cfg.cmd_wz,
                    stance_duration=T_st,
                    time_to_touchdown=t_td,
                    k_feedback=self.cfg.raibert_k,
                    ground_z=self._ground_z,
                )
                self.swings[i].update_target(target)

    def _solve_mpc(self, t: float) -> np.ndarray:
        x0, x_refs = self._build_reference(t)
        yaw = x0[2]
        foot_pos_rel = self._foot_positions_rel()
        schedule = self.gait.contact_schedule(t, self.mpc.cfg.dt, self.mpc.cfg.horizon)
        return self.mpc.solve(x0, x_refs, yaw, foot_pos_rel, schedule)

    def update(self, t: float) -> None:
        """Compute and apply joint torques for the current physics step."""
        in_stance_now = np.array(
            [self.gait.in_stance(t, i) for i in range(4)], dtype=bool
        )
        for i in range(4):
            if in_stance_now[i] and not self._was_stance[i]:
                self.swings[i].on_touchdown()
        self._was_stance = in_stance_now

        if t - self._last_mpc_t >= self._mpc_period - 1e-9:
            self._update_swing_targets(t)
            self._last_grfs = self._solve_mpc(t)
            self._last_mpc_t = t

        tau = np.zeros(12)
        T_sw = self.gait.swing_duration()
        for i in range(4):
            J = self.robot.foot_jacobian_leg(i)
            if in_stance_now[i]:
                f_world = self._last_grfs[3 * i : 3 * i + 3]
                # f_world is "force from ground on foot" (positive z up).
                # By virtual work: tau = -J^T f.
                tau[3 * i : 3 * i + 3] = -J.T @ f_world
            else:
                p_foot = self.robot.foot_pos_world(i)
                v_foot = self.robot.foot_vel_world(i)
                phase = self.gait.swing_progress(t, i)
                p_des, v_des = self.swings[i].desired_state(
                    swing_phase=phase,
                    current_foot_pos=p_foot,
                    swing_height=self.gait.swing_height,
                    swing_duration=T_sw,
                )
                tau[3 * i : 3 * i + 3] = self.swings[i].torque(
                    p_des, v_des, p_foot, v_foot, J
                )

        tau = np.clip(tau, -self.cfg.tau_max, self.cfg.tau_max)
        self.robot.apply_joint_torques(tau)
