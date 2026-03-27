import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

from config_utils import load_config
from EKF_fcn import BatteryEKF


def mean_mm(x):
    n = len(x) // 60
    return x[:n*60].reshape(n, 60).mean(axis=1)

def downsample_last(x):
    return [x[i+59] for i in range(0, len(x)-59, 60)]

cfg=load_config()

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent

DATA_DIR=ROOT_DIR/"data"/"rawdata"/"MGFarm_18650_FE"
DATA_DIR_ZHU=ROOT_DIR/"data"/"rawdata_ZHU"/"MGFarm_18650_Dataframes"
nums=range(1,30,2)
filepaths=[]
filepaths_zhu=[]
for num in nums:
    path=DATA_DIR/"df_FE_C{num:02d}.parquet".format(num=num)
    filepaths.append(path)

for num in nums:
    path=DATA_DIR_ZHU/"MGFarm_18650_C{num:02d}".format(num=num)/"df_SOC_scaled.parquet"
    filepaths_zhu.append(path)



for n in range(1):

    # use MGFarm_18650_FE
    df=pd.read_parquet(filepaths[n])
    I_raw=df['Current[A]'].to_numpy()
    U_raw=df['Voltage[V]'].to_numpy()
    SOH_raw=df['SOH'].to_numpy()
    SOC_raw=df['SOC'].to_numpy()
    
    I_soc=mean_mm(I_raw)
    U_soc=mean_mm(U_raw)
    SOH_soc=downsample_last(SOH_raw)
    SOC_soc=downsample_last(SOC_raw)
    # I_soc=I_raw
    # U_soc=U_raw
    # SOH_soc=SOH_raw
    # SOC_soc=SOC_raw

    # df=pd.read_parquet(filepaths_zhu[n])
    # I_soc=df['Current[A]'].to_numpy()
    # U_soc=df['Voltage[V]'].to_numpy()
    # SOH_soc=df['SOH_ZHU'].to_numpy()
    # SOC_soc=df['SOC_ZHU'].to_numpy()


    SoC_est=[]
    SoC_true=[]
    ekf = BatteryEKF(SOH_soc[0]) # ekf initialization
    print("ready to run")
    for i in range(20000):
        x,P,y_pred=ekf.predict_update(I_soc[i],U_soc[i])
        SoC_est.append(x[0])
        SoC_true.append(SOC_soc[i])
        if i%1000==0:
            print(f"Battery {n+1}, Progress: {(i+1)/len(I_soc)*100:.2f}%")
    
    SoC_true = np.array(SoC_true)
    SoC_est = np.array(SoC_est)
    rmse = np.sqrt(np.mean((SoC_true-SoC_est)**2))
    mae = np.mean(np.abs(SoC_true-SoC_est))

    plt.figure(figsize=(12,6))
    plt.plot(SoC_true, label='SOC_ture', color='blue')
    plt.plot(SoC_est, label='SOC_est', color='red', linestyle='--')

    # display RMSE and MAE
    textstr = f'RMSE: {rmse:.4f}\nMAE: {mae:.4f}'
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes,
         fontsize=12, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Time step (per 60 samples)')
    plt.ylabel('SOC')
    plt.title('SOC Comparison')
    plt.legend()
    plt.grid(True)
    plt.show()



    
