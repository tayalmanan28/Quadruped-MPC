"""Per-leg swing foot trajectory and Cartesian PD law for the G1."""
from __future__ import annotations

import numpy as np


def _bezier_z(s: float, h: float) -> tuple[float, float]:
    s = float(np.clip(s, 0.0, 1.0))
    z = h * 16.0 * s * s * (1.0 - s) * (1.0 - s)
    zd = h * 16.0 * (2.0 * s * (1.0 - s) ** 2 - 2.0 * s * s * (1.0 - s))
    return z, zd


# Nominal hip-yaw joint positions in pelvis frame (used to find footstep
# default xy under each hip, ground projected).
# left hip pitch link at (0, 0.064452, -0.1027) under pelvis; the foot in the
# home pose sits at world (≈0, ±0.12, 0) which is ~12 cm out from pelvis y=0.
HIP_OFFSETS_BODY = np.array(
    [
        [0.0, +0.12, 0.0],   # left foot nominal
        [0.0, -0.12, 0.0],   # right foot nominal
    ]
)


def raibert_footstep(
    base_pos_world: np.ndarray,
    base_yaw: float,
    hip_offset_body: np.ndarray,
    base_vel_world: np.ndarray,
    cmd_vel_world: np.ndarray,
    cmd_wz: float,
    stance_duration: float,
    time_to_touchdown: float,
    k_feedback: float = 0.04,
    ground_z: float = 0.0,
) -> np.ndarray:
    yaw_td = base_yaw + cmd_wz * time_to_touchdown
    c, s = np.cos(yaw_td), np.sin(yaw_td)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_hip = Rz @ hip_offset_body
    p_hip_td = base_pos_world + base_vel_world * time_to_touchdown + r_hip
    wz_vec = np.array([0.0, 0.0, cmd_wz])
    v_hip = base_vel_world + np.cross(wz_vec, r_hip)
    p = p_hip_td.copy()
    p[:2] += 0.5 * stance_duration * v_hip[:2]
    p[:2] += k_feedback * (base_vel_world[:2] - cmd_vel_world[:2])
    p[2] = ground_z
    return p


def lip_capture_footstep(
    com_pos_world: np.ndarray,
    com_vel_world: np.ndarray,
    com_height: float,
    base_yaw: float,
    hip_offset_body: np.ndarray,
    cmd_vel_world: np.ndarray,
    cmd_wz: float,
    stance_duration: float,
    time_to_touchdown: float,
    k_feedback: float = 0.1,
    ground_z: float = 0.0,
    g: float = 9.81,
) -> np.ndarray:
    """LIP / capture-point footstep planner for biped walking.

    The Linear Inverted Pendulum gives the divergent component of motion::

        omega = sqrt(g / z_com)
        xi    = com + v_com / omega           (capture point in xy)

    Without intervention ``xi`` grows away from the stance CoP exponentially.
    To bring the body to rest, the next foot should land **at** the capture
    point. To track a non-zero commanded velocity ``v_cmd``, bias the foot
    forward of the capture point by ``(v_cmd - v_com) / omega``.

    A lateral hip offset ``hip_offset_body`` (rotated by predicted yaw) keeps
    the stance width sane.
    """
    omega = float(np.sqrt(g / max(com_height, 1e-3)))
    yaw_td = base_yaw + cmd_wz * time_to_touchdown
    c, s = np.cos(yaw_td), np.sin(yaw_td)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    r_hip = Rz @ hip_offset_body
    # Capture point right now in world xy
    xi_now = com_pos_world[:2] + com_vel_world[:2] / omega
    # Propagate to predicted touchdown
    com_td = com_pos_world[:2] + com_vel_world[:2] * time_to_touchdown
    xi_td = com_td + (xi_now - com_td) * np.exp(omega * time_to_touchdown)
    # Lateral hip offset and a small forward bias for the commanded velocity
    p = np.zeros(3)
    p[:2] = xi_td + r_hip[:2]
    p[:2] += k_feedback * (cmd_vel_world[:2] - com_vel_world[:2]) / omega
    p[2] = ground_z
    return p


class SwingFootController:
    """Cartesian PD on the foot site, with a Bezier vertical profile.

    The output is a 6-DOF leg joint torque vector. Computed via the foot
    site's 3x6 Jacobian transpose, ``tau_leg = J^T f_des``.
    """

    def __init__(
        self,
        kp: tuple = (400.0, 400.0, 800.0),
        kd: tuple = (15.0, 15.0, 25.0),
        max_force: float = 300.0,
    ):
        self.kp = np.diag(kp)
        self.kd = np.diag(kd)
        self.max_force = float(max_force)
        self._liftoff: np.ndarray | None = None
        self._target: np.ndarray = np.zeros(3)
        self._was_swing = False

    def reset(self) -> None:
        self._liftoff = None
        self._was_swing = False

    def update_target(self, target: np.ndarray) -> None:
        self._target = np.asarray(target, dtype=float).copy()

    def desired_state(
        self,
        swing_phase: float,
        current_foot_pos: np.ndarray,
        swing_height: float,
        swing_duration: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self._liftoff is None or not self._was_swing:
            self._liftoff = current_foot_pos.copy()
        self._was_swing = True
        s = float(np.clip(swing_phase, 0.0, 1.0))
        xy = (1 - s) * self._liftoff[:2] + s * self._target[:2]
        xy_dot = (self._target[:2] - self._liftoff[:2]) / max(swing_duration, 1e-3)
        z_off, z_off_dot = _bezier_z(s, swing_height)
        z_lin = (1 - s) * self._liftoff[2] + s * self._target[2]
        z_lin_dot = (self._target[2] - self._liftoff[2]) / max(swing_duration, 1e-3)
        p_des = np.array([xy[0], xy[1], z_lin + z_off])
        v_des = np.array([xy_dot[0], xy_dot[1], z_lin_dot + z_off_dot / max(swing_duration, 1e-3)])
        return p_des, v_des

    def on_touchdown(self) -> None:
        self._was_swing = False
        self._liftoff = None

    def torque(
        self,
        p_des: np.ndarray,
        v_des: np.ndarray,
        p_foot: np.ndarray,
        v_foot: np.ndarray,
        jac: np.ndarray,    # 3 x 6 foot-site Jacobian
    ) -> np.ndarray:
        f = self.kp @ (p_des - p_foot) + self.kd @ (v_des - v_foot)
        n = np.linalg.norm(f)
        if n > self.max_force:
            f *= self.max_force / n
        return jac.T @ f
