"""Top-level G1 controller.

Supports two modes set via ``mode``:

* ``"stand"``  : both feet always in stance (double support); MPC distributes
  GRFs across 8 corners; legs held near home pose by joint-space PD.
* ``"walk"``   : alternating biped gait. The MPC contact schedule disables
  swing-leg corner forces; the swing leg is driven by a Cartesian PD that
  tracks a Bezier foot trajectory to a Raibert footstep target. The stance
  leg keeps its corner-force mapping with a *reduced* joint posture PD so
  the body can translate over the stance foot.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .convex_mpc import ConvexMPC, MPCConfig, GRAVITY
from .gait import BipedGait
from .robot import G1Robot
from .swing import HIP_OFFSETS_BODY, SwingFootController, lip_capture_footstep
from .upper_body import UpperBodyPD
from .wbc import TorqueQPConfig, WholeBodyTorqueQP


@dataclass
class G1ControllerConfig:
    mode: str = "stand"            # "stand" or "walk"
    body_height: float = 0.704     # CoM height target (matches bent-knee walk pose)
    # Steady-state pitch lean (rad). Slight forward lean is natural for a
    # walking biped — set to 0 to track upright.
    pitch_ref: float = 0.0
    mpc_rate_hz: float = 100.0
    # Per-axis multiplier on the pelvis inertia for the MPC. Same idea as
    # the quadruped: pelvis-only inertia underestimates how much the whole
    # body resists rotation, so we boost it.
    inertia_scale_xyz: tuple = (3.0, 3.0, 5.0)
    cmd_vx: float = 0.0
    cmd_vy: float = 0.0
    cmd_wz: float = 0.0
    # Upper-body PD
    upper_kp: float = 150.0
    upper_kd: float = 5.0
    upper_tau_max: float = 20.0
    # Leg posture PD (stand mode and stance-leg during walking).
    leg_posture_kp_stand: float = 80.0
    leg_posture_kd_stand: float = 4.0
    # During walking, keep the **stance** leg posture compliant so the body
    # can translate over the foot. Too high here and the leg fights the
    # MPC; too low and the leg buckles. Per-joint to give more weight to
    # hip-yaw / ankle-roll (which are otherwise underactuated by MPC GRFs).
    leg_posture_kp_walk_stance: tuple = (10.0, 30.0, 50.0, 15.0, 10.0, 50.0)
    leg_posture_kd_walk_stance: tuple = (1.5, 2.0, 3.0, 2.0, 1.5, 3.0)
    # Leg torque limit
    leg_tau_max: float = 120.0
    # LIP capture-point footstep feedback gain
    raibert_k: float = 0.1
    # Swing-leg Cartesian PD
    swing_kp: tuple = (800.0, 800.0, 3000.0)
    swing_kd: tuple = (25.0, 25.0, 60.0)
    swing_max_force: float = 600.0
    # Feed-forward vertical force on each swing leg to counter the leg's own
    # weight (~5 kg × 9.81 ≈ 50 N). Without this the PD has to chase
    # gravity and the foot doesn't actually clear the ground.
    swing_gravity_ff: float = 60.0
    # When walking, the MPC's CoM-y reference is interpolated toward the
    # stance foot's y so the MPC actively shifts weight to the support side
    # before / during swing. ``com_shift_gain`` ∈ [0, 1] picks how strongly.
    com_shift_gain: float = 0.5
    # Whole-body leg torque QP config.
    wbc: TorqueQPConfig = field(default_factory=TorqueQPConfig)


class G1Controller:
    def __init__(
        self,
        robot: G1Robot,
        gait: BipedGait | None = None,
        mpc_cfg: MPCConfig | None = None,
        ctrl_cfg: G1ControllerConfig | None = None,
    ):
        self.robot = robot
        self.cfg = ctrl_cfg or G1ControllerConfig()
        self.gait = gait or BipedGait()
        scaled_inertia = (
            np.asarray(robot.pelvis_inertia) * np.asarray(self.cfg.inertia_scale_xyz)
        )
        self.mpc = ConvexMPC(
            mass=robot.total_mass,
            inertia_body=scaled_inertia,
            n_contacts=8,
            cfg=mpc_cfg,
        )
        self.upper_pd = UpperBodyPD(
            n_joints=len(robot.upper_body_qpos_adr),
            kp=self.cfg.upper_kp,
            kd=self.cfg.upper_kd,
            tau_max=self.cfg.upper_tau_max,
        )
        self.wbc = WholeBodyTorqueQP(self.cfg.wbc)
        self.swings = [
            SwingFootController(
                kp=self.cfg.swing_kp,
                kd=self.cfg.swing_kd,
                max_force=self.cfg.swing_max_force,
            )
            for _ in range(2)
        ]

        self._upper_q_des = robot.home_qpos[robot.upper_body_qpos_adr].copy()
        self._leg_q_des = [
            robot.home_qpos[leg.qpos_adr].copy() for leg in robot.legs
        ]
        self._mpc_period = 1.0 / self.cfg.mpc_rate_hz
        self._last_mpc_t = -np.inf
        self._last_grfs = np.zeros(24)
        self._ground_z = 0.0
        self._was_stance = np.ones(2, dtype=bool)

    # ------------------------------------------------------------------
    def set_command(self, vx: float = 0.0, vy: float = 0.0, wz: float = 0.0) -> None:
        self.cfg.cmd_vx = float(vx)
        self.cfg.cmd_vy = float(vy)
        self.cfg.cmd_wz = float(wz)

    def set_mode(self, mode: str) -> None:
        assert mode in ("stand", "walk")
        self.cfg.mode = mode

    # ------------------------------------------------------------------
    def _stance_com_target(self, t: float) -> np.ndarray:
        """Return the (x, y) target the CoM should hover over at time ``t``
        given the gait state. Forms a sway pattern that shifts onto the
        currently-stance foot before swing. Falls back to mid-feet during
        double support."""
        l_in = self.gait.in_stance(t, 0)
        r_in = self.gait.in_stance(t, 1)
        l_foot = self.robot.foot_site_pos(0)[:2]
        r_foot = self.robot.foot_site_pos(1)[:2]
        if l_in and r_in:
            return 0.5 * (l_foot + r_foot)
        if l_in:
            return l_foot
        return r_foot

    def _build_reference(self, t: float = 0.0):
        N = self.mpc.cfg.horizon
        dt = self.mpc.cfg.dt
        rpy = self.robot.base_rpy()
        yaw = rpy[2]
        p = self.robot.com_world()
        v = self.robot.com_vel_world()
        w = self.robot.base_ang_vel_world()

        cy, sy = np.cos(yaw), np.sin(yaw)
        v_cmd = np.array([
            cy * self.cfg.cmd_vx - sy * self.cfg.cmd_vy,
            sy * self.cfg.cmd_vx + cy * self.cfg.cmd_vy,
            0.0,
        ])

        x0 = np.zeros(13)
        x0[0:3] = rpy
        x0[3:6] = p
        x0[6:9] = w
        x0[9:12] = v
        x0[12] = GRAVITY

        # Pre-compute the sway-anchor for the upcoming horizon. We blend the
        # nominal CoM ref (centered) with the sway target weighted by
        # ``com_shift_gain``. Done only in walk mode.
        x_refs = np.zeros((N, 13))
        for k in range(N):
            tk = (k + 1) * dt
            t_eval = t + tk
            sway_xy = (
                self._stance_com_target(t_eval) if self.cfg.mode == "walk"
                else np.array([p[0], p[1]])
            )
            x_refs[k, 0] = 0.0
            x_refs[k, 1] = self.cfg.pitch_ref
            x_refs[k, 2] = yaw + self.cfg.cmd_wz * tk
            base_x = p[0] + v_cmd[0] * tk
            base_y = p[1] + v_cmd[1] * tk
            if self.cfg.mode == "walk":
                a = self.cfg.com_shift_gain
                x_refs[k, 3] = (1 - a) * base_x + a * sway_xy[0]
                x_refs[k, 4] = (1 - a) * base_y + a * sway_xy[1]
            else:
                x_refs[k, 3] = base_x
                x_refs[k, 4] = base_y
            x_refs[k, 5] = self.cfg.body_height
            x_refs[k, 8] = self.cfg.cmd_wz
            x_refs[k, 9] = v_cmd[0]
            x_refs[k, 10] = v_cmd[1]
            x_refs[k, 12] = GRAVITY
        return x0, x_refs

    def _solve_mpc(self, t: float):
        x0, x_refs = self._build_reference(t)
        yaw = x0[2]
        com = x0[3:6]
        v_com = x0[9:12]
        corners_world = self.robot.all_corner_positions()  # (8, 3)
        dt = self.mpc.cfg.dt
        N = self.mpc.cfg.horizon

        if self.cfg.mode == "stand":
            schedule = np.ones((N, 8), dtype=bool)
        else:
            schedule = self.gait.contact_schedule_corners(t, dt, N)

        # Build predicted contact positions per horizon step.
        # - CoM is propagated linearly at commanded velocity.
        # - For each corner: if it's in stance at step k, its world position
        #   stays at the current observed location (it's planted on the
        #   ground). If it's swinging, we predict it'll be near the next
        #   touchdown target by end-of-swing. We linearly interpolate from
        #   current corner pos to the swing target during the swing window.
        contact_pos_rel = np.zeros((N, 8, 3))
        # Swing targets are per-leg (4 corners share a foot); approximate
        # corner targets by shifting current per-corner positions by the
        # delta between current foot site and swing target.
        leg_swing_target = [None, None]
        if self.cfg.mode == "walk":
            for li in range(2):
                if not self.gait.in_stance(t, li):
                    # already updated by _update_swing_targets at this MPC tick
                    leg_swing_target[li] = self.swings[li]._target.copy() \
                        if self.swings[li]._target is not None else None

        for k in range(N):
            t_eval = t + (k + 1) * dt
            com_pred = com + v_com * (k + 1) * dt
            for ci in range(8):
                li = ci // 4
                # Default: world position unchanged (stance assumption).
                corner_world = corners_world[ci].copy()
                if self.cfg.mode == "walk":
                    if (not self.gait.in_stance(t_eval, li)
                            and leg_swing_target[li] is not None):
                        # Interpolate corner toward the swing target.
                        # Use foot-site delta as the offset for this corner.
                        foot_now = self.robot.foot_site_pos(li)
                        delta = leg_swing_target[li] - foot_now
                        s = self.gait.swing_progress(t_eval, li)
                        corner_world = corners_world[ci] + s * delta
                contact_pos_rel[k, ci] = corner_world - com_pred

        return self.mpc.solve(x0, x_refs, yaw, contact_pos_rel, schedule)

    # ------------------------------------------------------------------
    def _update_swing_targets(self, t: float) -> None:
        com = self.robot.com_world()
        v = self.robot.com_vel_world()
        yaw = self.robot.base_rpy()[2]
        cy, sy = np.cos(yaw), np.sin(yaw)
        v_cmd_world = np.array([
            cy * self.cfg.cmd_vx - sy * self.cfg.cmd_vy,
            sy * self.cfg.cmd_vx + cy * self.cfg.cmd_vy,
            0.0,
        ])
        T_sw = self.gait.swing_duration()
        T_st = self.gait.stance_duration()
        for li in range(2):
            if not self.gait.in_stance(t, li):
                t_td = (1.0 - self.gait.swing_progress(t, li)) * T_sw
                target = lip_capture_footstep(
                    com_pos_world=com,
                    com_vel_world=v,
                    com_height=self.cfg.body_height,
                    base_yaw=yaw,
                    hip_offset_body=HIP_OFFSETS_BODY[li],
                    cmd_vel_world=v_cmd_world,
                    cmd_wz=self.cfg.cmd_wz,
                    stance_duration=T_st,
                    time_to_touchdown=t_td,
                    k_feedback=self.cfg.raibert_k,
                    ground_z=self._ground_z,
                )
                self.swings[li].update_target(target)

    # ------------------------------------------------------------------
    def update(self, t: float) -> None:
        if self.cfg.mode == "stand":
            in_stance_now = np.array([True, True])
        else:
            in_stance_now = np.array(
                [self.gait.in_stance(t, li) for li in range(2)]
            )

        for li in range(2):
            if in_stance_now[li] and not self._was_stance[li]:
                self.swings[li].on_touchdown()
        self._was_stance = in_stance_now

        if t - self._last_mpc_t >= self._mpc_period - 1e-9:
            if self.cfg.mode == "walk":
                self._update_swing_targets(t)
            self._last_grfs = self._solve_mpc(t)
            self._last_mpc_t = t

        # Build leg torques.
        leg_torques = {0: np.zeros(6), 1: np.zeros(6)}
        tau_mpc_stance = {0: np.zeros(6), 1: np.zeros(6)}
        tau_swing = {0: np.zeros(6), 1: np.zeros(6)}
        tau_posture = {0: np.zeros(6), 1: np.zeros(6)}
        jacs = self.robot.corner_jacobians()

        if self.cfg.mode == "stand":
            for ci in range(8):
                li = ci // 4
                f_world = self._last_grfs[3 * ci : 3 * ci + 3]
                tau_mpc_stance[li] += -jacs[ci].T @ f_world
            kp_post = self.cfg.leg_posture_kp_stand
            kd_post = self.cfg.leg_posture_kd_stand
            for li, leg in enumerate(self.robot.legs):
                q = self.robot.data.qpos[leg.qpos_adr]
                qd = self.robot.data.qvel[leg.qvel_adr]
                tau_posture[li] = kp_post * (self._leg_q_des[li] - q) - kd_post * qd

            # In stand mode both legs are stance by construction.
            leg_torques = self.wbc.solve(
                in_stance=np.array([True, True]),
                tau_mpc_stance=tau_mpc_stance,
                tau_swing=tau_swing,
                tau_posture=tau_posture,
                cmd_wz=0.0,
                wz_meas=float(self.robot.base_ang_vel_world()[2]),
            )
        else:  # walk
            T_sw = self.gait.swing_duration()
            for li in range(2):
                if in_stance_now[li]:
                    # Stance leg: track MPC force-mapped torque plus posture
                    # regularization inside the whole-body QP.
                    for ci in range(4):
                        gi = li * 4 + ci
                        f_world = self._last_grfs[3 * gi : 3 * gi + 3]
                        tau_mpc_stance[li] += -jacs[gi].T @ f_world
                    q = self.robot.data.qpos[self.robot.legs[li].qpos_adr]
                    qd = self.robot.data.qvel[self.robot.legs[li].qvel_adr]
                    kp_post = np.asarray(self.cfg.leg_posture_kp_walk_stance)
                    kd_post = np.asarray(self.cfg.leg_posture_kd_walk_stance)
                    tau_posture[li] = kp_post * (self._leg_q_des[li] - q) - kd_post * qd
                else:
                    # Swing leg: Cartesian PD on foot site + gravity FF.
                    p_foot = self.robot.foot_site_pos(li)
                    v_foot = self.robot.foot_site_vel(li)
                    phase = self.gait.swing_progress(t, li)
                    p_des, v_des = self.swings[li].desired_state(
                        swing_phase=phase,
                        current_foot_pos=p_foot,
                        swing_height=self.gait.swing_height,
                        swing_duration=T_sw,
                    )
                    Jsite = self.robot.foot_site_jacobian(li)
                    f_pd = self.swings[li].torque(
                        p_des, v_des, p_foot, v_foot, np.eye(3),
                    )  # raw 3-D force
                    f_pd[2] += self.cfg.swing_gravity_ff
                    tau_swing[li] = Jsite.T @ f_pd

                    q = self.robot.data.qpos[self.robot.legs[li].qpos_adr]
                    qd = self.robot.data.qvel[self.robot.legs[li].qvel_adr]
                    # Keep swing joints near nominal while allowing motion.
                    tau_posture[li] = 0.35 * (
                        np.asarray(self.cfg.leg_posture_kp_walk_stance)
                        * (self._leg_q_des[li] - q)
                    ) - 0.35 * (
                        np.asarray(self.cfg.leg_posture_kd_walk_stance) * qd
                    )

            leg_torques = self.wbc.solve(
                in_stance=in_stance_now,
                tau_mpc_stance=tau_mpc_stance,
                tau_swing=tau_swing,
                tau_posture=tau_posture,
                cmd_wz=self.cfg.cmd_wz,
                wz_meas=float(self.robot.base_ang_vel_world()[2]),
            )

        for li in (0, 1):
            np.clip(
                leg_torques[li], -self.cfg.leg_tau_max, self.cfg.leg_tau_max,
                out=leg_torques[li],
            )

        # Upper-body PD.
        q_up = self.robot.data.qpos[self.robot.upper_body_qpos_adr]
        qd_up = self.robot.data.qvel[self.robot.upper_body_qvel_adr]
        tau_up = self.upper_pd(q_up, qd_up, self._upper_q_des)

        self.robot.apply_torques(leg_torques, tau_up)
