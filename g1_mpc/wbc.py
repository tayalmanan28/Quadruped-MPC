"""Lightweight whole-body torque allocator for G1 leg joints.

The MPC solves for stance contact forces. Swing control computes desired
foot Cartesian forces. This module fuses those objectives into a single
12-DOF leg torque QP with box limits and torque-rate regularization.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import osqp
import scipy.sparse as sp


@dataclass
class TorqueQPConfig:
    tau_max: float = 120.0
    w_stance_track: float = 14.0
    w_swing_track: float = 10.0
    w_posture_stance: float = 1.0
    w_posture_swing: float = 2.2
    w_tau_reg: float = 0.03
    w_tau_rate: float = 0.12
    yaw_rate_ff_gain: float = 10.0
    yaw_rate_ff_max: float = 20.0


class WholeBodyTorqueQP:
    """Solve a bounded QP for both legs' joint torques.

    Decision variable: ``tau_all = [tau_left(6), tau_right(6)]``.
    """

    def __init__(self, cfg: TorqueQPConfig | None = None):
        self.cfg = cfg or TorqueQPConfig()
        self._n = 12
        self._tau_prev = np.zeros(self._n)

        # Constant box-constraint matrix: -tau_max <= tau_i <= tau_max.
        self._A = sp.eye(self._n, format="csc")
        lim = np.full(self._n, self.cfg.tau_max)
        self._l = -lim
        self._u = +lim

    def solve(
        self,
        in_stance: np.ndarray,
        tau_mpc_stance: dict[int, np.ndarray],
        tau_swing: dict[int, np.ndarray],
        tau_posture: dict[int, np.ndarray],
        cmd_wz: float,
        wz_meas: float,
    ) -> dict[int, np.ndarray]:
        # Build per-leg reference and per-joint weights.
        tau_ref = np.zeros(self._n)
        w_track = np.zeros(self._n)
        w_post = np.zeros(self._n)

        for li in (0, 1):
            sl = slice(li * 6, (li + 1) * 6)
            if in_stance[li]:
                tau_ref[sl] = tau_mpc_stance[li]
                w_track[sl] = self.cfg.w_stance_track
                w_post[sl] = self.cfg.w_posture_stance
            else:
                tau_ref[sl] = tau_swing[li]
                w_track[sl] = self.cfg.w_swing_track
                w_post[sl] = self.cfg.w_posture_swing

        # Add a symmetric hip-yaw feed-forward bias to improve yaw response.
        # Index 2 in each leg is hip_yaw_joint in this codebase.
        yaw_ff = self.cfg.yaw_rate_ff_gain * (cmd_wz - wz_meas)
        yaw_ff = float(np.clip(yaw_ff, -self.cfg.yaw_rate_ff_max, self.cfg.yaw_rate_ff_max))
        tau_ref[2] += yaw_ff
        tau_ref[8] -= yaw_ff

        # Diagonal convex quadratic objective:
        #   w_track ||tau - tau_ref||^2
        # + w_post  ||tau - tau_post||^2
        # + w_reg   ||tau||^2
        # + w_rate  ||tau - tau_prev||^2
        post_vec = np.concatenate([tau_posture[0], tau_posture[1]])
        d = w_track + w_post + self.cfg.w_tau_reg + self.cfg.w_tau_rate
        d = np.maximum(d, 1e-9)
        H = sp.diags(2.0 * d, format="csc")
        q = -2.0 * (
            w_track * tau_ref
            + w_post * post_vec
            + self.cfg.w_tau_rate * self._tau_prev
        )

        solver = osqp.OSQP()
        solver.setup(
            P=H,
            q=q,
            A=self._A,
            l=self._l,
            u=self._u,
            verbose=False,
            warm_starting=True,
            eps_abs=1e-4,
            eps_rel=1e-4,
            max_iter=2000,
            polish=False,
        )
        solver.warm_start(x=self._tau_prev)
        res = solver.solve()
        if res.x is None or "solved" not in res.info.status:
            tau = np.clip(self._tau_prev, self._l, self._u)
        else:
            tau = np.asarray(res.x).copy()
        self._tau_prev = tau
        return {0: tau[:6], 1: tau[6:]}
