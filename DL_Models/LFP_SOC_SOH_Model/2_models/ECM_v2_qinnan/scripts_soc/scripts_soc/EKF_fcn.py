import numpy as np
from scipy.interpolate import RegularGridInterpolator
from ECM_loader import ECMTable


class BatteryEKF:
    def __init__(self,soh):
        self.x=np.array([1.0, 0.0,0.0])
        self.P=np.eye(3) 
        self.soh=soh
        self.deltaT=60.0
        self.I_prev=0
        self.ecm=ECMTable()

        self.Q=np.diag([1e-10, 2e-5, 2e-5])
        self.R=9e-4

        self.C0=1.8*3600
        self.Cb=self.C0*soh
    def predict_update(self,I,Ut):

        
        soc = np.clip(self.x[0], 0, 1)
    
    
    
        eff=1.0
        if self.I_prev>= 0: eff = 0.999
    

   
        R1 = self.ecm.get_param(soc, self.soh, self.I_prev, "R1")
        R2= self.ecm.get_param(soc, self.soh, self.I_prev, "R2")
        tau1 = self.ecm.get_param(soc,self.soh, self.I_prev, "tau1")
        tau2 =self.ecm.get_param(soc, self.soh, self.I_prev, "tau2")

    
        Ad = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.exp(-self.deltaT/tau1), 0.0],
            [0.0, 0.0, np.exp(-self.deltaT/tau2)]
        ])
        Bd = np.array([
            eff*self.deltaT/self.Cb,
            R1*(1-np.exp(-self.deltaT/tau1)),
            R2*(1-np.exp(-self.deltaT/tau2))
        ])
        #predict
        x_p = Ad @ self.x + Bd * self.I_prev

    
        P_p = Ad @ self.P @ Ad.T + self.Q

   
        Ri = self.ecm.get_param(x_p[0], self.soh, I, "Ri")
        OCV = self.ecm.get_param(x_p[0], self.soh, I, "ocv")
        dOCV = self.ecm.get_param(x_p[0], self.soh, I, "dOCV")

        Cd = np.array([dOCV, 1.0, 1.0])

    
        y_p = OCV + dOCV*(x_p[0]-soc)+x_p[1] + x_p[2] + Ri*I

        #update
        delta_y = Ut - y_p
        S = Cd @ P_p @ Cd.T + self.R      
        K = (P_p @ Cd) / S            
        self.x= x_p + K*delta_y

   
        self.x[0] = np.clip(self.x[0], 0, 1)
        self.P=(np.eye(3) - np.outer(K, Cd)) @ P_p

        

        # Ri=self.ecm.get_param(self.x[0],self.soh,I,"Ri")
        # y_k1 = OCV + dOCV*(self.x[0]-soc) + self.x[1] + self.x[2] + Ri*I
        
        
        # OCV=self.ecm.get_param(self.x[0],self.soh,I,"ocv")
        # Ri=self.ecm.get_param(self.x[0],self.soh,I,"Ri")
        # dOCV = self.ecm.get_param(self.x[0], self.soh, I, "dOCV")
        
        y_k1 = OCV + dOCV*(self.x[0]-soc) + self.x[1] + self.x[2] + Ri*I


        self.I_prev=I
        return self.x,self.P,y_k1


