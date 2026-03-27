"""
Coulomb counting model block (online SOC estimation).

Inputs per step: current [A], voltage [V], temperature [C] (optional), dt [s] or timestamp [s].
Outputs per step: SOC (0..1).

Algorithm (online):
- Integrate current to Q_m_new (Ah).
- Detect CV phase: voltage >= (V_max - V_tol) continuously for cv_seconds.
- If CV detected, reset Q_m_new to 0 (SOC=1).
- SOC = 1 + Q_m_new / capacity_ah, clipped to [0, 1].
- Optional: per-step capacity_ah override (e.g., from SOH estimator).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class CCModelConfig:
    capacity_ah: float = 1.8
    soc_init: float = 1.0
    current_sign: float = 1.0
    v_max: float = 3.65
    v_tol: float = 0.02
    cv_seconds: float = 300.0


class CCModel:
    def __init__(self, config: Optional[CCModelConfig] = None):
        self.config = config or CCModelConfig()
        self.reset()

    def reset(self):
        # Q_m_new is defined so that SOC = 1 + Q_m_new / capacity_ah
        self.q_m_new = (self.config.soc_init - 1.0) * self.config.capacity_ah
        self.soc = float(self.config.soc_init)
        self.cv_time_s = 0.0
        self.last_timestamp_s = None
        self._capacity_override_init_done = False

    def step(
        self,
        current_a: float,
        voltage_v: float,
        temperature_c: Optional[float] = None,
        capacity_ah: Optional[float] = None,
        dt_s: Optional[float] = None,
        timestamp_s: Optional[float] = None,
    ) -> float:
        """Process a single sample and return SOC.

        Provide either dt_s or timestamp_s. If both are None, dt_s=0 is used.
        """
        if dt_s is None:
            if timestamp_s is None:
                dt_s = 0.0
            else:
                if self.last_timestamp_s is None:
                    dt_s = 0.0
                else:
                    dt_s = float(timestamp_s) - float(self.last_timestamp_s)
                    if dt_s < 0:
                        dt_s = 0.0
                self.last_timestamp_s = float(timestamp_s)
        else:
            dt_s = float(dt_s)

        # CV detection (voltage high for cv_seconds)
        if voltage_v >= (self.config.v_max - self.config.v_tol):
            self.cv_time_s += dt_s
        else:
            self.cv_time_s = 0.0

        if self.cv_time_s >= self.config.cv_seconds:
            self.q_m_new = 0.0
        else:
            self.q_m_new += self.config.current_sign * float(current_a) * dt_s / 3600.0

        if capacity_ah is not None and not self._capacity_override_init_done:
            # Align initial SOC with provided capacity
            self.q_m_new = (self.config.soc_init - 1.0) * float(capacity_ah)
            self._capacity_override_init_done = True

        cap = float(capacity_ah) if capacity_ah is not None else self.config.capacity_ah
        self.soc = 1.0 + (self.q_m_new / cap)
        # hard clamp
        if self.soc < 0.0:
            self.soc = 0.0
        elif self.soc > 1.0:
            self.soc = 1.0
        return self.soc

    def process_dataframe(
        self,
        df,
        time_col: str = 'Testtime[s]',
        current_col: str = 'Current[A]',
        voltage_col: str = 'Voltage[V]',
        temp_col: str = 'Temperature[°C]',
        capacity_col: Optional[str] = None,
    ):
        """Process a dataframe and return SOC array.

        Uses time_col to compute dt in seconds. If time_col is missing, assumes dt=1s.
        """
        self.reset()
        n = len(df)
        soc = np.zeros(n, dtype=np.float32)

        if time_col in df.columns:
            t = df[time_col].to_numpy(dtype=np.float64)
            dt_s = np.diff(t, prepend=t[0])
            dt_s[dt_s < 0] = 0.0
        else:
            dt_s = np.ones(n, dtype=np.float64)

        i = df[current_col].to_numpy(dtype=np.float64)
        v = df[voltage_col].to_numpy(dtype=np.float64)
        if temp_col in df.columns:
            _ = df[temp_col].to_numpy(dtype=np.float64)
        cap = df[capacity_col].to_numpy(dtype=np.float64) if capacity_col and capacity_col in df.columns else None

        for k in range(n):
            cap_k = cap[k] if cap is not None else None
            soc[k] = self.step(i[k], v[k], capacity_ah=cap_k, dt_s=dt_s[k])
        return soc
