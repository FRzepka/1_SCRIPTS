
#include <stdio.h>
#include "lstm_model_lstm_int8_fp32mlp.h"
int main(){
  LSTMState st; lstm_model_init(&st);
  float in[INPUT_SIZE];
  while (scanf("%f %f %f %f %f %f", &in[0],&in[1],&in[2],&in[3],&in[4],&in[5])==6){
    float y = lstm_model_forward(in, &st);
    printf("%.8f
", y);
    fflush(stdout);
  }
  return 0;
}
