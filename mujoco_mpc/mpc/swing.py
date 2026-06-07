"""Swing-foot trajectory and Cartesian PD law.

For each swing leg we plan:

* The **touchdown target** in world coordinates via a yaw-aware **Raibert
  heuristic**: the hip pose at touchdown is predicted using both the commanded
  linear velocity and the commanded yaw rate.

* A **Bezier vertical profile** that lifts the foot to ``swing_height`` and
  brings it back to the ground.

The desired foot position is tracked with a Cartesian PD law applied to the
leg via its Jacobian:

      f_des = Kp (p_des - p_foot) + Kd (v_des - v_foot)
      tau   = J_leg^T  f_des
"""
from __future__ import annotations

import numpy as np


def _bezier_z(swing_phase: float, height: float) -> tuple[float, float]:
    """C^2 bell-shaped Bezier for the vertical swing profile.
    Returns ``(z, dz/ds)`` where s = swing_phase ∈ [0, 1]."""
    s = float(np.clip(swing_phase, 0.0, 1.0))
    z = height * 16.0 * s * s * (1.0 - s) * (1.0 - s)
    z_dot = height * 16.0 * (2.0 * s * (1.0 - s) ** 2 - 2.0 * s * s * (1.0 - s))
    return z, z_dot


def raibert_footstep(
    base_pos_world: np.ndarray,
    base_yaw: float,
    hip_offset_body: np.ndarray,
    base_vel_world: np.ndarray,
    cmd_vel_world: np.ndarray,
    cmd_wz: float,
    stance_duration: float,
    time_to_touchdown: float,
    k_feedback: float = 0.03,
    ground_z: float = 0.0,
) -> np.ndarray:
    """Yaw-aware Raibert footstep heuristic.

    Predicts where the hip will be at the end of swing using both the commanded
    linear velocity and the commanded yaw rate, then places the foot ahead of
    that predicted hip by half a stance of the predicted hip velocity, plus
    the standard velocity-error feedback term.
    """
    yaw_td = base_yaw + cmd_wz * time_to_touchdown
    c, s = np.cos(yaw_td), np.sin(yaw_td)
    Rz_td = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_hip_td = Rz_td @ hip_offset_body
    p_hip_td = base_pos_world + base_vel_world * time_to_touchdown + r_hip_td
    wz_vec = np.array([0.0, 0.0, cmd_wz])
    v_hip_td = base_vel_world + np.cross(wz_vec, r_hip_td)
    p = p_hip_td.copy()
    p[:2] += 0.5 * stance_duration * v_hip_td[:2]
    p[:2] += k_feedback * (base_vel_world[:2] - cmd_vel_world[:2])
    p[2] = ground_z
    return p


class SwingLegController:
    """Stateful per-leg swing controller."""

    def __init__(
        self,
        kp: tuple[float, float, float] = (700.0, 700.0, 350.0),
        kd: tuple[float, float, float] = (12.0, 12.0, 10.0),
        max_force: float = 200.0,
    ):
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)
        self.max_force = float(max_force)
        self._liftoff_pos: np.ndarray | None = None
        self._target_pos: np.ndarray = np.zeros(3)
        self._was_swing: bool = False

    def reset(self) -> None:
        self._liftoff_pos = None
        self._was_swing = False

    def update_target(self, target: np.ndarray) -> None:
        self._target_pos = np.asarray(target, dtype=float).copy()

    def desired_state(
        self,
        swing_phase: float,
        current_foot_pos: np.ndarray,
        swing_height: float,
        swing_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._liftoff_pos is None or not self._was_swing:
            self._liftoff_pos = current_foot_pos.copy()
        self._was_swing = True
        s = float(np.clip(swing_phase, 0.0, 1.0))
        xy_des = (1.0 - s) * self._liftoff_pos[:2] + s * self._target_pos[:2]
        xy_dot = (self._target_pos[:2] - self._liftoff_pos[:2]) / max(
            swing_duration, 1e-3
        )
        z_off, z_off_dot = _bezier_z(s, swing_height)
        z_lin = (1.0 - s) * self._liftoff_pos[2] + s * self._target_pos[2]
        z_lin_dot = (self._target_pos[2] - self._liftoff_pos[2]) / max(
            swing_duration, 1e-3
        )
        p_des = np.array([xy_des[0], xy_des[1], z_lin + z_off])
        v_des = np.array(
            [xy_dot[0], xy_dot[1], z_lin_dot + z_off_dot / max(swing_duration, 1e-3)]
        )
        return p_des, v_des

    def on_touchdown(self) -> None:
        self._was_swing = False
        self._liftoff_pos = None

    def torque(
        self,
        p_des: np.ndarray,
        v_des: np.ndarray,
        p_foot: np.ndarray,
        v_foot: np.ndarray,
        jac_leg: np.ndarray,
    ) -> np.ndarray:
        f = self.kp @ (p_des - p_foot) + self.kd @ (v_des - v_foot)
        n = np.linalg.norm(f)
        if n > self.max_force:
            f *= self.max_force / n
        return jac_leg.T @ f
