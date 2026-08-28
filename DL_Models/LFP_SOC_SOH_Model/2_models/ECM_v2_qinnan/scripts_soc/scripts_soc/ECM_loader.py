import numpy as np
import scipy.io
from pathlib import Path
from scipy.interpolate import interpn  # 用于插值

class ECMTable:
    def __init__(self, mat_path=None):
        if mat_path is None:
            mat_path = Path(__file__).resolve().parent / "ECM_param_table.mat"
        mat = scipy.io.loadmat(mat_path)
        ECM_mat = mat['ECM']

        # 读取 SOC / SOH
        self.soc = np.array(ECM_mat['soc'][0,0], dtype=np.float32).flatten()
        self.soh = np.array(ECM_mat['soh'][0,0], dtype=np.float32).flatten()

        # 读取 para_discharge 和 para_charge
        self.para = {}
        for mode in ['para_discharge','para_charge']:
            self.para[mode] = {}
            subStruct = ECM_mat[mode][0,0]
            for field in subStruct.dtype.names:
                val = subStruct[field][0,0]
                self.para[mode][field] = np.array(val, dtype=np.float32)

    
    
    def get_param(self, soc, soh, I, param_name):
        """
        soc, soh: 0~1
        I: 电流 >0 充电, <0 放电
        param_name: 'Ri','R1','R2','tau1','tau2','ocv','dOCV'
        """
        mode = 'para_charge' if I >= 0 else 'para_discharge'
        param_grid = self.para[mode][param_name]

        # 插值，SOC 是列，SOH 是行
        val = interpn(
            (self.soh, self.soc),  # grid 顺序和 param_grid 对应
            param_grid,
            np.array([soh, soc]),
            bounds_error=False,
            fill_value=None
        )
        return float(val)


# ecm = ECMTable()

# soc = 0.65
# soh = 0.95
# I = -1.2  # 放电电流

# Ri = ecm.get_param(soc, soh, I, "Ri")
# OCV = ecm.get_param(soc, soh, I, "ocv")

# print("Ri:", Ri)
# print("OCV:", OCV)
