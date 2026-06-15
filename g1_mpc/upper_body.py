"""Joint-space PD controller that holds the upper body (waist + arms) at the
home pose. The biped MPC controls the legs only; everything above the waist
is locked so it doesn't drift / flop around."""
from __future__ import annotations

import numpy as np


class UpperBodyPD:
    """PD on each joint of the upper body, with per-joint gains and a target
    angle taken from the home keyframe."""

    def __init__(
        self,
        n_joints: int,
        kp: float | np.ndarray = 200.0,
        kd: float | np.ndarray = 5.0,
        tau_max: float | np.ndarray = 25.0,
    ):
        self.kp = np.broadcast_to(np.asarray(kp, dtype=float), (n_joints,)).copy()
        self.kd = np.broadcast_to(np.asarray(kd, dtype=float), (n_joints,)).copy()
        self.tau_max = np.broadcast_to(np.asarray(tau_max, dtype=float), (n_joints,)).copy()

    def __call__(
        self,
        q: np.ndarray,
        qd: np.ndarray,
        q_des: np.ndarray,
        qd_des: np.ndarray | None = None,
    ) -> np.ndarray:
        if qd_des is None:
            qd_des = np.zeros_like(qd)
        tau = self.kp * (q_des - q) + self.kd * (qd_des - qd)
        return np.clip(tau, -self.tau_max, self.tau_max)
