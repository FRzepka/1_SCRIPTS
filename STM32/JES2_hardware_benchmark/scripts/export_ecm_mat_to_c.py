#!/usr/bin/env python3
"""Export the verified JES2 ECM MAT table as float64 C arrays."""
from pathlib import Path
import numpy as np
import scipy.io

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "DL_Models/LFP_SOC_SOH_Model/2_models/ECM_v2_qinnan/scripts_soc/scripts_soc/ECM_param_table.mat"
OUT = ROOT / "STM32/JES2_hardware_benchmark/firmware/JES2_HW_HECM/Core/Src/ecm_tables.c"

def array(name, values):
    flat=np.asarray(values,dtype=np.float64).ravel()
    lines=[]
    for i in range(0,len(flat),6):
        lines.append("  "+", ".join(f"{x:.17g}" for x in flat[i:i+6]))
    return f"const double {name}[{len(flat)}] = {{\n"+",\n".join(lines)+"\n};\n"

e=scipy.io.loadmat(SOURCE)["ECM"]
parts=['#include "ecm_tables.h"\n\n',array('ecm_soc',e['soc'][0,0]),array('ecm_soh',e['soh'][0,0])]
for mode in ('discharge','charge'):
    s=e['para_'+mode][0,0]
    for parameter in ('Ri','R1','R2','tau1','tau2','ocv','dOCV'):
        parts.append(array(f'ecm_{mode}_{parameter}',s[parameter][0,0]))
OUT.write_text("".join(parts),encoding='ascii')
print(OUT)
