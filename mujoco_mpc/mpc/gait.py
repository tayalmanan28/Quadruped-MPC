"""Periodic contact schedule for a trotting quadruped.

A leg ``i`` is in stance during the fraction ``[offset_i, offset_i + duty)`` of
the gait period (modulo 1), and in swing otherwise. For a trot, diagonal pairs
share the same phase offset:

    FR, RL: offset 0.0
    FL, RR: offset 0.5
"""
from __future__ import annotations

import numpy as np


class TrotGait:
    _DEFAULT_OFFSETS = np.array([0.0, 0.5, 0.5, 0.0])

    def __init__(
        self,
        period: float = 0.5,
        duty_factor: float = 0.6,
        phase_offsets: np.ndarray | None = None,
        swing_height: float = 0.06,
    ):
        self.period = float(period)
        self.duty = float(duty_factor)
        self.phase_offsets = (
            np.asarray(phase_offsets, dtype=float)
            if phase_offsets is not None
            else self._DEFAULT_OFFSETS.copy()
        )
        self.swing_height = float(swing_height)

    def leg_phase(self, t: float, leg_idx: int) -> float:
        return (t / self.period + self.phase_offsets[leg_idx]) % 1.0

    def in_stance(self, t: float, leg_idx: int) -> bool:
        return self.leg_phase(t, leg_idx) < self.duty

    def stance_progress(self, t: float, leg_idx: int) -> float:
        ph = self.leg_phase(t, leg_idx)
        return ph / self.duty if ph < self.duty else 0.0

    def swing_progress(self, t: float, leg_idx: int) -> float:
        ph = self.leg_phase(t, leg_idx)
        if ph < self.duty:
            return 0.0
        return (ph - self.duty) / (1.0 - self.duty)

    def contact_schedule(self, t0: float, dt: float, horizon: int) -> np.ndarray:
        out = np.zeros((horizon, 4), dtype=bool)
        for k in range(horizon):
            for leg in range(4):
                out[k, leg] = self.in_stance(t0 + k * dt, leg)
        return out

    def swing_duration(self) -> float:
        return self.period * (1.0 - self.duty)

    def stance_duration(self) -> float:
        return self.period * self.duty
