/*
ARDUINO LSTM SOC PREDICTOR - FIXED VERSION
Based on arduino_lstm_soc_full32.ino but with corrected communication
*/

#include "lstm_weights_exact.h"
#include <math.h>

// Constants
#define INPUT_SIZE 4
#define HIDDEN_SIZE 32
#define OUTPUT_SIZE 1
#define BUFFER_SIZE 256

// LSTM Hidden States
float h_state[HIDDEN_SIZE] = {0};
float c_state[HIDDEN_SIZE] = {0};

// Temporary arrays for LSTM calculations
float input_gate[HIDDEN_SIZE];
float forget_gate[HIDDEN_SIZE];
float candidate_gate[HIDDEN_SIZE];
float output_gate[HIDDEN_SIZE];
float new_c_state[HIDDEN_SIZE];
float new_h_state[HIDDEN_SIZE];

// MLP intermediate results
float mlp_layer0[32];
float mlp_layer3[32];
float mlp_output;

// Input buffer
char inputBuffer[BUFFER_SIZE];
int bufferIndex = 0;

// Activation functions
float sigmoid_activation(float x) {
  return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
  return tanh(x);
}

float relu_activation(float x) {
  return fmax(0.0f, x);
}

// LSTM Forward Pass (32 Hidden Units)
float predictSOC(float voltage, float current, float soh, float q_c) {
  float input[INPUT_SIZE] = {voltage, current, soh, q_c};
  
  // 1. Input Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_input_ih_bias[i] + lstm_input_hh_bias[i];
    
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_input_ih_weights[i][j] * input[j];
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_input_hh_weights[i][j] * h_state[j];
    }
    
    input_gate[i] = sigmoid_activation(sum);
  }
  
  // 2. Forget Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_forget_ih_bias[i] + lstm_forget_hh_bias[i];
    
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_forget_ih_weights[i][j] * input[j];
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_forget_hh_weights[i][j] * h_state[j];
    }
    
    forget_gate[i] = sigmoid_activation(sum);
  }
  
  // 3. Candidate Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_candidate_ih_bias[i] + lstm_candidate_hh_bias[i];
    
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_candidate_ih_weights[i][j] * input[j];
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_candidate_hh_weights[i][j] * h_state[j];
    }
    
    candidate_gate[i] = tanh_activation(sum);
  }
  
  // 4. Output Gate
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    float sum = lstm_output_ih_bias[i] + lstm_output_hh_bias[i];
    
    for (int j = 0; j < INPUT_SIZE; j++) {
      sum += lstm_output_ih_weights[i][j] * input[j];
    }
    
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += lstm_output_hh_weights[i][j] * h_state[j];
    }
    
    output_gate[i] = sigmoid_activation(sum);
  }
  
  // 5. Update Cell and Hidden States
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    new_c_state[i] = forget_gate[i] * c_state[i] + input_gate[i] * candidate_gate[i];
    new_h_state[i] = output_gate[i] * tanh_activation(new_c_state[i]);
  }
  
  // Update states
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    c_state[i] = new_c_state[i];
    h_state[i] = new_h_state[i];
  }
  
  // MLP Layers
  // Layer 0: Linear(32, 32) + ReLU
  for (int i = 0; i < 32; i++) {
    float sum = mlp0_bias[i];
    for (int j = 0; j < HIDDEN_SIZE; j++) {
      sum += mlp0_weights[i][j] * h_state[j];
    }
    mlp_layer0[i] = relu_activation(sum);
  }
  
  // Layer 3: Linear(32, 32) + ReLU
  for (int i = 0; i < 32; i++) {
    float sum = mlp3_bias[i];
    for (int j = 0; j < 32; j++) {
      sum += mlp3_weights[i][j] * mlp_layer0[j];
    }
    mlp_layer3[i] = relu_activation(sum);
  }
  
  // Layer 6: Linear(32, 1) + Sigmoid
  float sum = mlp6_bias[0];
  for (int j = 0; j < 32; j++) {
    sum += mlp6_weights[0][j] * mlp_layer3[j];
  }
  mlp_output = sigmoid_activation(sum);
  
  return mlp_output;
}

void resetLSTMStates() {
  for (int i = 0; i < HIDDEN_SIZE; i++) {
    h_state[i] = 0.0f;
    c_state[i] = 0.0f;
  }
}

void processData(String dataString) {
  // Parse CSV format: "voltage,current,soh,q_c"
  int commaIndex1 = dataString.indexOf(',');
  int commaIndex2 = dataString.indexOf(',', commaIndex1 + 1);
  int commaIndex3 = dataString.indexOf(',', commaIndex2 + 1);
  
  if (commaIndex1 > 0 && commaIndex2 > 0 && commaIndex3 > 0) {
    float voltage = dataString.substring(0, commaIndex1).toFloat();
    float current = dataString.substring(commaIndex1 + 1, commaIndex2).toFloat();
    float soh = dataString.substring(commaIndex2 + 1, commaIndex3).toFloat();
    float q_c = dataString.substring(commaIndex3 + 1).toFloat();
    
    // Get SOC prediction
    float soc_pred = predictSOC(voltage, current, soh, q_c);
    
    // Send ONLY the SOC value (no extra text!)
    Serial.print(soc_pred, 6);
    Serial.println();
  } else {
    Serial.println("ERROR: Invalid format");
  }
}

void setup() {
  Serial.begin(115200);
  Serial.setTimeout(100);
  
  // Initialize LSTM states
  resetLSTMStates();
  
  // Send startup info ONCE
  Serial.println("ARDUINO_LSTM_FIXED_READY");
  Serial.println("32_HIDDEN_UNITS");
  Serial.println("READY_FOR_PREDICTIONS");
}

void loop() {
  if (Serial.available()) {
    char incomingByte = Serial.read();
    
    if (incomingByte == '\n' || incomingByte == '\r') {
      if (bufferIndex > 0) {
        inputBuffer[bufferIndex] = '\0';
        String inputString = String(inputBuffer);
        
        // Handle special commands
        if (inputString.equals("RESET")) {
          resetLSTMStates();
          Serial.println("RESET_OK");
        } else if (inputString.equals("INFO")) {
          Serial.println("ARDUINO_LSTM_32_UNITS");
          Serial.println("INPUT_4_OUTPUT_1");
          Serial.println("INFO_OK");
        } else {
          // Process prediction data
          processData(inputString);
        }
        
        bufferIndex = 0;
      }
    } else if (bufferIndex < BUFFER_SIZE - 1) {
      inputBuffer[bufferIndex++] = incomingByte;
    }
  }
}
