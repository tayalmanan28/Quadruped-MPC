"""Generalised convex MPC for a floating-base rigid body with N contact points.

Same Di Carlo / MIT Cheetah formulation as the quadruped controller but with
an **arbitrary number of contact points** ``n_contacts`` (the quadruped uses
4, the G1 uses 8 — four corners on each foot). Each contact contributes one
3-D world-frame force; the QP decision variable is the stacked GRFs over the
horizon.

State (13)::

    x = [roll, pitch, yaw,  px, py, pz,  omega_x, omega_y, omega_z,
         vx, vy, vz,  g]
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import osqp
import scipy.sparse as sp


GRAVITY = 9.81


@dataclass
class MPCConfig:
    horizon: int = 20
    dt: float = 0.015
    mu: float = 0.6
    f_min: float = 0.0
    f_max: float = 500.0
    Q: np.ndarray | None = None     # 13-vector
    R_force: float = 1e-5


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _Rz(yaw: float) -> np.ndarray:
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])


class ConvexMPC:
    NX = 13

    def __init__(
        self,
        mass: float,
        inertia_body: np.ndarray,
        n_contacts: int,
        cfg: MPCConfig | None = None,
    ):
        self.cfg = cfg or MPCConfig()
        if self.cfg.Q is None:
            self.cfg.Q = np.array(
                [
                    # rpy (very high — attitude must track for biped balance)
                    2000.0, 2000.0, 1200.0,
                    # position xyz
                    20.0, 20.0, 500.0,
                    # angular vel
                    3.0, 3.0, 20.0,
                    # linear vel
                    30.0, 30.0, 30.0,
                    # gravity
                    0.0,
                ]
            )
        self.mass = float(mass)
        ib = np.asarray(inertia_body, dtype=float)
        self.I_body = np.diag(ib) if ib.ndim == 1 else ib
        self.n_contacts = int(n_contacts)
        self.NU = 3 * self.n_contacts

        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        self.Qbar = np.kron(np.eye(N), np.diag(self.cfg.Q))
        self.Rbar = self.cfg.R_force * np.eye(N * nu)

        # Constant constraint matrix: 5 rows per (k, contact) — 4 friction
        # pyramid + 1 fz box (l/u set per call to gate stance/swing).
        rows: list[np.ndarray] = []
        for k in range(N):
            for i in range(self.n_contacts):
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

    # ------------------------------------------------------------------
    def _A_continuous(self, yaw: float) -> np.ndarray:
        nx = self.NX
        A = np.zeros((nx, nx))
        Rz = _Rz(yaw)
        A[0:3, 6:9] = Rz.T
        A[3:6, 9:12] = np.eye(3)
        A[11, 12] = -1.0
        return A

    def _B_continuous(self, yaw: float, contact_pos_rel: np.ndarray) -> np.ndarray:
        nx, nu = self.NX, self.NU
        Rz = _Rz(yaw)
        I_world = Rz @ self.I_body @ Rz.T
        I_world_inv = np.linalg.inv(I_world)
        B = np.zeros((nx, nu))
        for i in range(self.n_contacts):
            r = contact_pos_rel[i]
            B[6:9, 3 * i : 3 * i + 3] = I_world_inv @ _skew(r)
            B[9:12, 3 * i : 3 * i + 3] = np.eye(3) / self.mass
        return B

    def _propagation(self, A: np.ndarray, B_list: list[np.ndarray]):
        """Condensed propagation with potentially **time-varying** B.

        ``B_list[j]`` is the discrete B matrix that applies to the control
        ``u_j`` (i.e. uses the contact positions predicted at step j).
        """
        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        S_x = np.zeros((N * nx, nx))
        S_u = np.zeros((N * nx, N * nu))
        Apow = [np.eye(nx)]
        for _ in range(N):
            Apow.append(A @ Apow[-1])
        for k in range(N):
            S_x[k * nx : (k + 1) * nx, :] = Apow[k + 1]
            for j in range(k + 1):
                S_u[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = (
                    Apow[k - j] @ B_list[j]
                )
        return S_x, S_u

    def solve(
        self,
        x0: np.ndarray,
        x_refs: np.ndarray,
        yaw: float,
        contact_pos_rel: np.ndarray,
        contact_schedule: np.ndarray,
    ) -> np.ndarray:
        """Solve and return the GRFs for the first MPC step (shape ``(NU,)``).

        ``contact_pos_rel`` : either ``(n_contacts, 3)`` for constant contact
        geometry over the horizon, or ``(N, n_contacts, 3)`` with one set per
        horizon step (lets the MPC anticipate where the swing foot will land
        and how stance feet move relative to the translating CoM).
        ``contact_schedule`` : ``(N, n_contacts)`` bool, True where contact is
        active (stance).
        """
        N, nx, nu = self.cfg.horizon, self.NX, self.NU
        cp = np.asarray(contact_pos_rel)
        if cp.ndim == 2:
            cp = np.broadcast_to(cp[None], (N,) + cp.shape).copy()
        Ac = self._A_continuous(yaw)
        Ad = np.eye(nx) + Ac * self.cfg.dt
        Bds = [self._B_continuous(yaw, cp[k]) * self.cfg.dt for k in range(N)]
        S_x, S_u = self._propagation(Ad, Bds)

        H = 2.0 * (S_u.T @ self.Qbar @ S_u + self.Rbar)
        H += 1e-8 * np.eye(H.shape[0])
        H = 0.5 * (H + H.T)
        X_ref_vec = np.asarray(x_refs).reshape(-1)
        g_vec = 2.0 * S_u.T @ self.Qbar @ (S_x @ x0 - X_ref_vec)

        n_rows = 5 * self.n_contacts * N
        l = np.full(n_rows, -np.inf)
        u = np.full(n_rows, np.inf)
        row = 0
        for k in range(N):
            for i in range(self.n_contacts):
                for _ in range(4):
                    u[row] = 0.0; row += 1
                if contact_schedule[k, i]:
                    l[row] = self.cfg.f_min
                    u[row] = self.cfg.f_max
                else:
                    l[row] = 0.0
                    u[row] = 0.0
                row += 1

        P_csc = sp.csc_matrix(H)
        solver = osqp.OSQP()
        solver.setup(
            P=P_csc, q=g_vec, A=self._A_sparse, l=l, u=u,
            verbose=False, warm_starting=True,
            eps_abs=5e-4, eps_rel=5e-4, max_iter=4000, polish=False,
        )
        if self._u_prev is not None and np.any(self._u_prev):
            warm = np.concatenate([self._u_prev[nu:], self._u_prev[-nu:]])
            solver.warm_start(x=warm)

        res = solver.solve()
        if res.x is None or "solved" not in res.info.status:
            warm = np.concatenate([self._u_prev[nu:], self._u_prev[-nu:]])
            return warm[:nu]
        self._u_prev = res.x.copy()
        return res.x[:nu]
