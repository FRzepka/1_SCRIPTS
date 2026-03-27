from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import scipy.io


PARAM_NAMES = ("Ri", "R1", "R2", "tau1", "tau2", "ocv", "dOCV")
MODES = ("para_discharge", "para_charge")


class FastECMTable:
    def __init__(self, mat_path: str | Path | None = None):
        if mat_path is None:
            mat_path = Path(__file__).resolve().parent.parent.parent / "2_models" / "ECM_v2_qinnan" / "ECM_param_table.mat"
        mat = scipy.io.loadmat(mat_path)
        ecm_mat = mat["ECM"]
        self.soc_grid = np.array(ecm_mat["soc"][0, 0], dtype=np.float64).flatten()
        self.soh_grid = np.array(ecm_mat["soh"][0, 0], dtype=np.float64).flatten()
        self.param_grids: Dict[str, Dict[str, np.ndarray]] = {}
        for mode in MODES:
            self.param_grids[mode] = {}
            sub_struct = ecm_mat[mode][0, 0]
            for field in sub_struct.dtype.names:
                self.param_grids[mode][field] = np.array(sub_struct[field][0, 0], dtype=np.float64)
        self._curve_cache: Dict[Tuple[str, float], Dict[str, np.ndarray]] = {}

    def _soh_key(self, soh: float) -> float:
        return round(float(np.clip(soh, self.soh_grid[0], self.soh_grid[-1])), 6)

    def _build_curves_for_soh(self, soh: float, mode: str) -> Dict[str, np.ndarray]:
        key = (mode, self._soh_key(soh))
        cached = self._curve_cache.get(key)
        if cached is not None:
            return cached

        soh_val = key[1]
        hi = int(np.searchsorted(self.soh_grid, soh_val, side="right"))
        hi = min(max(hi, 1), len(self.soh_grid) - 1)
        lo = hi - 1
        soh_lo = float(self.soh_grid[lo])
        soh_hi = float(self.soh_grid[hi])
        if soh_hi == soh_lo:
            w_hi = 0.0
        else:
            w_hi = (soh_val - soh_lo) / (soh_hi - soh_lo)
        w_lo = 1.0 - w_hi

        curves: Dict[str, np.ndarray] = {}
        for param_name in PARAM_NAMES:
            grid = self.param_grids[mode][param_name]
            curves[param_name] = w_lo * grid[lo, :] + w_hi * grid[hi, :]
        self._curve_cache[key] = curves
        return curves

    def interpolate(self, soc: float, soh: float, current: float, param_name: str) -> float:
        mode = "para_charge" if current >= 0.0 else "para_discharge"
        curves = self._build_curves_for_soh(soh, mode)
        soc_val = float(np.clip(soc, self.soc_grid[0], self.soc_grid[-1]))
        return float(np.interp(soc_val, self.soc_grid, curves[param_name]))

    def interpolate_many(self, soc: float, soh: float, current: float, param_names=PARAM_NAMES) -> Dict[str, float]:
        mode = "para_charge" if current >= 0.0 else "para_discharge"
        curves = self._build_curves_for_soh(soh, mode)
        soc_val = float(np.clip(soc, self.soc_grid[0], self.soc_grid[-1]))
        return {name: float(np.interp(soc_val, self.soc_grid, curves[name])) for name in param_names}


class FastBatteryEKF:
    def __init__(self, soh: float, table: FastECMTable | None = None):
        self.x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(3, dtype=np.float64)
        self.soh = float(soh)
        self.deltaT = 60.0
        self.I_prev = 0.0
        self.ecm = table or FastECMTable()
        self.Q = np.diag([1e-10, 2e-5, 2e-5]).astype(np.float64)
        self.R = 9e-4
        self.C0 = 1.8 * 3600.0
        self.Cb = self.C0 * self.soh

    def predict_update(self, current: float, voltage: float):
        soc = float(np.clip(self.x[0], 0.0, 1.0))
        eff = 0.999 if self.I_prev >= 0.0 else 1.0

        pred_params = self.ecm.interpolate_many(soc, self.soh, self.I_prev, ("R1", "R2", "tau1", "tau2"))
        tau1 = max(pred_params["tau1"], 1e-9)
        tau2 = max(pred_params["tau2"], 1e-9)
        ad = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.exp(-self.deltaT / tau1), 0.0],
                [0.0, 0.0, np.exp(-self.deltaT / tau2)],
            ],
            dtype=np.float64,
        )
        bd = np.array(
            [
                eff * self.deltaT / max(self.Cb, 1e-9),
                pred_params["R1"] * (1.0 - np.exp(-self.deltaT / tau1)),
                pred_params["R2"] * (1.0 - np.exp(-self.deltaT / tau2)),
            ],
            dtype=np.float64,
        )
        x_p = ad @ self.x + bd * self.I_prev
        p_p = ad @ self.P @ ad.T + self.Q

        upd_params = self.ecm.interpolate_many(float(x_p[0]), self.soh, current, ("Ri", "ocv", "dOCV"))
        ri = upd_params["Ri"]
        ocv = upd_params["ocv"]
        docv = upd_params["dOCV"]
        cd = np.array([docv, 1.0, 1.0], dtype=np.float64)
        y_p = ocv + docv * (x_p[0] - soc) + x_p[1] + x_p[2] + ri * current

        delta_y = float(voltage) - y_p
        s = float(cd @ p_p @ cd.T + self.R)
        k = (p_p @ cd) / s
        self.x = x_p + k * delta_y
        self.x[0] = np.clip(self.x[0], 0.0, 1.0)
        self.P = (np.eye(3) - np.outer(k, cd)) @ p_p
        y_k = ocv + docv * (self.x[0] - soc) + self.x[1] + self.x[2] + ri * current
        self.I_prev = float(current)
        return self.x, self.P, y_k
