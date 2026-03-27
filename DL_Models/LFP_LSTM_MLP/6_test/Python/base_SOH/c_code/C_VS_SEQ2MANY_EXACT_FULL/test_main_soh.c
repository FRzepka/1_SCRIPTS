
#include <stdio.h>
#include "lstm_model_soh.h"
#include "model_weights_soh.h"
#include "scaler_params_soh.h"

int main(){
  float in[INPUT_SIZE];
  float in_scaled[INPUT_SIZE];
  LSTMModelSOH model; lstm_model_soh_init(&model);
  while (1){
    int ok = 0;
    if (INPUT_SIZE==6){
      ok = scanf("%f %f %f %f %f %f", &in[0],&in[1],&in[2],&in[3],&in[4],&in[5]);
      if (ok!=6) break;
    } else {
      for (int i=0;i<INPUT_SIZE;i++){
        if (scanf("%f", &in[i])!=1){ ok=0; break; }
        ok++;
      }
      if (ok!=INPUT_SIZE) break;
    }
    scaler_soh_transform(in, in_scaled);
    float y=0.0f; lstm_model_soh_inference(&model, in_scaled, &y);
    printf("SOH %.6f\n", y);
    fflush(stdout);
  }
  return 0;
}
