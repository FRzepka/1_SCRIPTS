from __future__ import annotations

import ctypes
from pathlib import Path
from typing import Optional, Sequence, Tuple


class ECMNativeEKF:
    """Thin ctypes wrapper around the original C EKF (battery_ekf.c)."""

    def __init__(
        self,
        lib_path: Optional[str | Path] = None,
        soc_init: float = 1.0,
        p_init: Optional[Sequence[float]] = None,
    ) -> None:
        self._lib = self._load_library(lib_path)
        self._lib.battery_EKF.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # x_k1 (len 3)
            ctypes.POINTER(ctypes.c_float),  # y_k1 (len 1)
            ctypes.c_float,                  # I
            ctypes.c_float,                  # Ut
            ctypes.c_float,                  # I_prev
            ctypes.c_float,                  # SoH
            ctypes.POINTER(ctypes.c_float),  # P_k1 (len 9)
            ctypes.POINTER(ctypes.c_float),  # x (len 3)
            ctypes.POINTER(ctypes.c_float),  # P (len 9)
        ]
        self._lib.battery_EKF.restype = None

        self._x = (ctypes.c_float * 3)()
        self._P = (ctypes.c_float * 9)()
        self._x_k1 = (ctypes.c_float * 3)()
        self._P_k1 = (ctypes.c_float * 9)()
        self._y_k1 = ctypes.c_float()
        self._I_prev = ctypes.c_float(0.0)

        self.reset(soc_init=soc_init, p_init=p_init)

    @staticmethod
    def _load_library(lib_path: Optional[str | Path]) -> ctypes.CDLL:
        if lib_path is None:
            lib_path = Path(__file__).resolve().parent / "native" / "libecm_ekf.so"
        else:
            lib_path = Path(lib_path)
        if not lib_path.exists():
            raise FileNotFoundError(
                f"Native ECM library not found at {lib_path}. "
                "Build it with: python_model/native/build_native.sh"
            )
        return ctypes.CDLL(str(lib_path))

    def reset(self, soc_init: float = 1.0, p_init: Optional[Sequence[float]] = None) -> None:
        self._x[0] = ctypes.c_float(soc_init)
        self._x[1] = ctypes.c_float(0.0)
        self._x[2] = ctypes.c_float(0.0)

        if p_init is None:
            p_init = [1e-4, 0.0, 0.0, 0.0, 1e-3, 0.0, 0.0, 0.0, 1e-3]
        if len(p_init) != 9:
            raise ValueError("p_init must have 9 elements (3x3 row-major).")
        for i, val in enumerate(p_init):
            self._P[i] = ctypes.c_float(float(val))

        self._I_prev = ctypes.c_float(0.0)

    def step(self, current_a: float, voltage_v: float, soh: float = 1.0) -> Tuple[float, float]:
        self._lib.battery_EKF(
            self._x_k1,
            ctypes.byref(self._y_k1),
            ctypes.c_float(float(current_a)),
            ctypes.c_float(float(voltage_v)),
            ctypes.c_float(float(self._I_prev.value)),
            ctypes.c_float(float(soh)),
            self._P_k1,
            self._x,
            self._P,
        )

        # update state and covariance
        for i in range(3):
            self._x[i] = self._x_k1[i]
        for i in range(9):
            self._P[i] = self._P_k1[i]
        self._I_prev = ctypes.c_float(float(current_a))

        return float(self._x_k1[0]), float(self._y_k1.value)

    def state(self) -> Tuple[Tuple[float, float, float], Tuple[float, ...]]:
        x = (float(self._x[0]), float(self._x[1]), float(self._x[2]))
        p = tuple(float(self._P[i]) for i in range(9))
        return x, p
