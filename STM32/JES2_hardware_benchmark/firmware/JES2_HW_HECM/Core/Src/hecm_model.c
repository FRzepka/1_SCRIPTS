#include "hecm_model.h"
#include "ecm_tables.h"
#include <math.h>

static double x[3], p[3][3], previous_current;

static unsigned lower(const double *grid, unsigned count, double value) {
  if (value <= grid[0]) return 0U;
  if (value >= grid[count-1U]) return count-2U;
  unsigned i=0U; while (i+1U<count && value>grid[i+1U]) ++i; return i;
}
static double interpolation(const double *table,double soc,double soh) {
  if(soc<ecm_soc[0])soc=ecm_soc[0]; if(soc>ecm_soc[ECM_SOC_COUNT-1])soc=ecm_soc[ECM_SOC_COUNT-1];
  if(soh<ecm_soh[0])soh=ecm_soh[0]; if(soh>ecm_soh[ECM_SOH_COUNT-1])soh=ecm_soh[ECM_SOH_COUNT-1];
  unsigned si=lower(ecm_soc,ECM_SOC_COUNT,soc), hi=lower(ecm_soh,ECM_SOH_COUNT,soh);
  double ws=(soc-ecm_soc[si])/(ecm_soc[si+1]-ecm_soc[si]);
  double wh=(soh-ecm_soh[hi])/(ecm_soh[hi+1]-ecm_soh[hi]);
  double a=table[hi*ECM_SOC_COUNT+si]*(1.0-ws)+table[hi*ECM_SOC_COUNT+si+1U]*ws;
  double b=table[(hi+1U)*ECM_SOC_COUNT+si]*(1.0-ws)+table[(hi+1U)*ECM_SOC_COUNT+si+1U]*ws;
  return a*(1.0-wh)+b*wh;
}
void HECM_Reset(void){x[0]=1.0;x[1]=x[2]=0.0;previous_current=0.0;for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)p[i][j]=(i==j)?1.0:0.0;}
float HECM_Step(float current_f,float voltage_f,float soh_f,float dt_f){
  double current=current_f,voltage=voltage_f,soh=soh_f,dt=dt_f,soc=fmin(1.0,fmax(0.0,x[0]));
  const double *r1t=previous_current>=0?ecm_charge_R1:ecm_discharge_R1,*r2t=previous_current>=0?ecm_charge_R2:ecm_discharge_R2;
  const double *t1t=previous_current>=0?ecm_charge_tau1:ecm_discharge_tau1,*t2t=previous_current>=0?ecm_charge_tau2:ecm_discharge_tau2;
  double r1=interpolation(r1t,soc,soh),r2=interpolation(r2t,soc,soh),tau1=interpolation(t1t,soc,soh),tau2=interpolation(t2t,soc,soh);
  double a[3]={1.0,exp(-dt/fmax(tau1,1e-9)),exp(-dt/fmax(tau2,1e-9))};
  double xp[3]={x[0]+((previous_current>=0?0.999:1.0)*dt/(1.8*3600.0*soh))*previous_current,a[1]*x[1]+r1*(1-a[1])*previous_current,a[2]*x[2]+r2*(1-a[2])*previous_current};
  double pp[3][3]; for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)pp[i][j]=a[i]*p[i][j]*a[j]+((i==j)?(i==0?1e-10:2e-5):0.0);
  const double *rit=current>=0?ecm_charge_Ri:ecm_discharge_Ri,*ot=current>=0?ecm_charge_ocv:ecm_discharge_ocv,*dtbl=current>=0?ecm_charge_dOCV:ecm_discharge_dOCV;
  double ri=interpolation(rit,xp[0],soh),ocv=interpolation(ot,xp[0],soh),docv=interpolation(dtbl,xp[0],soh),c[3]={docv,1,1};
  double yp=ocv+docv*(xp[0]-soc)+xp[1]+xp[2]+ri*current,s=9e-4,k[3];
  for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)s+=c[i]*pp[i][j]*c[j];
  for(unsigned i=0;i<3;i++){k[i]=0;for(unsigned j=0;j<3;j++)k[i]+=pp[i][j]*c[j];k[i]/=s;x[i]=xp[i]+k[i]*(voltage-yp);} x[0]=fmin(1.0,fmax(0.0,x[0]));
  for(unsigned i=0;i<3;i++)for(unsigned j=0;j<3;j++)p[i][j]=pp[i][j]-k[i]*(c[0]*pp[0][j]+c[1]*pp[1][j]+c[2]*pp[2][j]);
  previous_current=current; return (float)x[0];
}
