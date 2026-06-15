"""Alternating biped gait scheduler.

Leg 0 = left, leg 1 = right. Phase offsets ``[0, 0.5]``. With ``duty_factor``
> 0.5 there are overlapping double-support windows; with < 0.5 there's a
flight phase (running). Default uses generous double-support so the convex
MPC has time to shift weight between feet.
"""
from __future__ import annotations

import numpy as np


class BipedGait:
    _OFFSETS = np.array([0.0, 0.5])

    def __init__(
        self,
        period: float = 0.82,
        duty_factor: float = 0.64,
        swing_height: float = 0.075,
        phase_offsets: np.ndarray | None = None,
    ):
        self.period = float(period)
        self.duty = float(duty_factor)
        self.swing_height = float(swing_height)
        self.phase_offsets = (
            np.asarray(phase_offsets, dtype=float)
            if phase_offsets is not None
            else self._OFFSETS.copy()
        )

    def leg_phase(self, t: float, leg_idx: int) -> float:
        return (t / self.period + self.phase_offsets[leg_idx]) % 1.0

    def in_stance(self, t: float, leg_idx: int) -> bool:
        return self.leg_phase(t, leg_idx) < self.duty

    def swing_progress(self, t: float, leg_idx: int) -> float:
        ph = self.leg_phase(t, leg_idx)
        if ph < self.duty:
            return 0.0
        return (ph - self.duty) / (1.0 - self.duty)

    def stance_progress(self, t: float, leg_idx: int) -> float:
        ph = self.leg_phase(t, leg_idx)
        return ph / self.duty if ph < self.duty else 0.0

    def swing_duration(self) -> float:
        return self.period * (1.0 - self.duty)

    def stance_duration(self) -> float:
        return self.period * self.duty

    def contact_schedule_corners(self, t0: float, dt: float, horizon: int) -> np.ndarray:
        """Return (horizon, 8) bool: True for each of the 8 foot corners when
        that corner's leg is in stance."""
        out = np.zeros((horizon, 8), dtype=bool)
        for k in range(horizon):
            for li in range(2):
                active = self.in_stance(t0 + k * dt, li)
                out[k, li * 4 : li * 4 + 4] = active
        return out
