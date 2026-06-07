"""Convex Model Predictive Control for quadruped ground-reaction forces.

Formulation from Di Carlo *et al.* "Dynamic Locomotion in the MIT Cheetah 3
Through Convex Model-Predictive Control" (IROS 2018) — the same controller
used by ``tayalmanan28/Quadruped-MPC``.

State, :math:`x \\in \\mathbb{R}^{13}`::

    x = [ roll, pitch, yaw,
          p_x, p_y, p_z,
          omega_x, omega_y, omega_z,
          v_x, v_y, v_z,
          g ]

with :math:`\\omega` and :math:`v` in the **world** frame, and :math:`g = +9.81`
an augmented gravity state used to keep the dynamics linear time-invariant.

Input, :math:`u \\in \\mathbb{R}^{12}`: world-frame GRFs for the four feet in
order ``[FR, FL, RR, RL]``.

We Euler-discretize at the MPC step ``dt`` and **condense** the QP so the
decision variables are only the stacked GRFs ``U = [u_0, ..., u_{N-1}]`` of
size ``12 N``. The cost is a dense quadratic in ``U``; constraints are:

* 4-sided friction pyramid on every (leg, step): :math:`|f_x|, |f_y| \\le \\mu f_z`
* normal-force box :math:`f_z^{\\min} \\le f_z \\le f_z^{\\max}` on stance,
  and :math:`f_z = 0` on swing (which through the cone forces :math:`f_x = f_y = 0`).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp


GRAVITY = 9.81


DEFAULT_Q = np.array(
    [
        # orientation (roll, pitch, yaw)
        100.0, 100.0, 1000.0,
        # position (px, py, pz) — xy untracked, pz strongly tracked
        0.0, 0.0, 100.0,
        # angular vel (wx, wy, wz)
        0.05, 0.05, 30.0,
        # linear vel
        10.0, 10.0, 10.0,
        # gravity (no penalty)
        0.0,
    ]
)
DEFAULT_R = 1e-5


@dataclass
class MPCConfig:
    horizon: int = 14
    dt: float = 0.02
    mu: float = 0.6
    f_min: float = 0.0
    f_max: float = 250.0
    Q: np.ndarray = field(default_factory=lambda: DEFAULT_Q.copy())
    R_force: float = DEFAULT_R


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _Rz(yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])


class ConvexMPC:
    """Convex MPC computing one MPC step (the first input ``u_0``)."""

    NX = 13
    NU = 12

    def __init__(self, mass: float, inertia_body: np.ndarray, cfg: MPCConfig | None = None):
        self.cfg = cfg or MPCConfig()
        self.mass = float(mass)
        ib = np.asarray(inertia_body, dtype=float)
        self.I_body = np.diag(ib) if ib.ndim == 1 else ib

        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        self.Qbar = np.kron(np.eye(N), np.diag(self.cfg.Q))
        self.Rbar = self.cfg.R_force * np.eye(N * nu)

        # Constant friction-cone + fz-box constraint matrix.
        rows: list[np.ndarray] = []
        for k in range(N):
            for i in range(4):
                col = k * nu + 3 * i
                mu = self.cfg.mu
                r1 = np.zeros(N * nu); r1[col] = 1.0; r1[col + 2] = -mu
                r2 = np.zeros(N * nu); r2[col] = -1.0; r2[col + 2] = -mu
                r3 = np.zeros(N * nu); r3[col + 1] = 1.0; r3[col + 2] = -mu
                r4 = np.zeros(N * nu); r4[col + 1] = -1.0; r4[col + 2] = -mu
                r5 = np.zeros(N * nu); r5[col + 2] = 1.0
                rows.extend([r1, r2, r3, r4, r5])
        self._A_const = np.vstack(rows)
        self._A_sparse = sp.csc_matrix(self._A_const)

        self._u_prev = np.zeros(N * nu)

    def _continuous(self, yaw: float, foot_pos_rel: np.ndarray):
        nx, nu = self.NX, self.NU
        A = np.zeros((nx, nx))
        Rz = _Rz(yaw)
        A[0:3, 6:9] = Rz.T
        A[3:6, 9:12] = np.eye(3)
        A[11, 12] = -1.0
        I_world = Rz @ self.I_body @ Rz.T
        I_world_inv = np.linalg.inv(I_world)
        B = np.zeros((nx, nu))
        for i in range(4):
            r = foot_pos_rel[i]
            B[6:9, 3 * i : 3 * i + 3] = I_world_inv @ _skew(r)
            B[9:12, 3 * i : 3 * i + 3] = np.eye(3) / self.mass
        return A, B

    def _propagation(self, A: np.ndarray, B: np.ndarray):
        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        S_x = np.zeros((N * nx, nx))
        S_u = np.zeros((N * nx, N * nu))
        Apow = [np.eye(nx)]
        for _ in range(N):
            Apow.append(A @ Apow[-1])
        for k in range(N):
            S_x[k * nx : (k + 1) * nx, :] = Apow[k + 1]
            for j in range(k + 1):
                S_u[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = Apow[k - j] @ B
        return S_x, S_u

    def solve(
        self,
        x0: np.ndarray,
        x_refs: np.ndarray,
        yaw: float,
        foot_pos_rel: np.ndarray,
        contact_schedule: np.ndarray,
    ) -> np.ndarray:
        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        Ac, Bc = self._continuous(yaw, foot_pos_rel)
        Ad = np.eye(nx) + Ac * self.cfg.dt
        Bd = Bc * self.cfg.dt
        S_x, S_u = self._propagation(Ad, Bd)

        H = 2.0 * (S_u.T @ self.Qbar @ S_u + self.Rbar)
        H += 1e-8 * np.eye(H.shape[0])
        H = 0.5 * (H + H.T)
        X_ref_vec = np.asarray(x_refs).reshape(-1)
        g_vec = 2.0 * S_u.T @ self.Qbar @ (S_x @ x0 - X_ref_vec)

        n_rows = 5 * 4 * N
        l = np.full(n_rows, -np.inf)
        u = np.full(n_rows, np.inf)
        row = 0
        for k in range(N):
            for i in range(4):
                for _ in range(4):
                    l[row] = -np.inf
                    u[row] = 0.0
                    row += 1
                if contact_schedule[k, i]:
                    l[row] = self.cfg.f_min
                    u[row] = self.cfg.f_max
                else:
                    l[row] = 0.0
                    u[row] = 0.0
                row += 1

        # OSQP drops exact zeros during ``setup`` so re-using the solver with
        # ``update(Px=…)`` can fail when the dense cost matrix changes its
        # implied sparsity. The QP is small (~170 vars), so a fresh setup
        # every call is fine and keeps the code robust.
        P_csc = sp.csc_matrix(H)
        solver = osqp.OSQP()
        solver.setup(
            P=P_csc,
            q=g_vec,
            A=self._A_sparse,
            l=l,
            u=u,
            verbose=False,
            warm_starting=True,
            eps_abs=5e-4,
            eps_rel=5e-4,
            max_iter=4000,
            polish=False,
        )
        if self._u_prev is not None and np.any(self._u_prev):
            warm = np.concatenate([self._u_prev[nu:], self._u_prev[-nu:]])
            solver.warm_start(x=warm)

        res = solver.solve()
        status = res.info.status
        if res.x is None or "solved" not in status:
            U_warm = np.concatenate([self._u_prev[nu:], self._u_prev[-nu:]])
            return U_warm[:nu]
        self._u_prev = res.x.copy()
        return res.x[:nu]
