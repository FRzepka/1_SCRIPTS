\
#include "lstm_weights.h" // Generated weights and model parameters
#include <math.h> // For exp, tanh

// Define window size for processing - REDUCED for Arduino memory constraints
// Note: This is smaller than training window size to fit in Arduino RAM
#define WINDOW_SIZE 50 // Reduced from 5000 to fit in Arduino memory

// LSTM state variables (non-stateful, reset for each window)
float h_state[HIDDEN_SIZE];
float c_state[HIDDEN_SIZE];

// MLP intermediate layer outputs
float mlp_fc1_out[MLP_HIDDEN_SIZE];

// --- Memory and Performance Monitoring --- (Placeholder, adapt from Stateful_32_32 if needed)
unsigned long last_inference_time_us = 0;

// --- Activation Functions ---
float sigmoid(float x) {
  if (x < -10.0f) return 0.0f;
  if (x > 10.0f) return 1.0f;
  return 1.0f / (1.0f + expf(-x));
}

float tanh_activation(float x) {
  if (x < -10.0f) return -1.0f;
  if (x > 10.0f) return 1.0f;
  return tanhf(x);
}

float relu(float x) {
  return (x > 0) ? x : 0;
}

// --- LSTM Cell Forward Pass (for a single time step) ---
void lstm_cell_forward(const float input[INPUT_SIZE], float h_prev[HIDDEN_SIZE], float c_prev[HIDDEN_SIZE],
                       float h_new[HIDDEN_SIZE], float c_new[HIDDEN_SIZE]) {
    float i_gate[HIDDEN_SIZE];
    float f_gate[HIDDEN_SIZE];
    float g_gate[HIDDEN_SIZE]; // Candidate gate
    float o_gate[HIDDEN_SIZE];

    // Input Gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        i_gate[i] = lstm_input_ih_bias[i] + lstm_input_hh_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            i_gate[i] += input[j] * lstm_input_ih_weights[i][j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            i_gate[i] += h_prev[j] * lstm_input_hh_weights[i][j];
        }
        i_gate[i] = sigmoid(i_gate[i]);
    }

    // Forget Gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        f_gate[i] = lstm_forget_ih_bias[i] + lstm_forget_hh_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            f_gate[i] += input[j] * lstm_forget_ih_weights[i][j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            f_gate[i] += h_prev[j] * lstm_forget_hh_weights[i][j];
        }
        f_gate[i] = sigmoid(f_gate[i]);
    }

    // Candidate Gate (Cell Gate)
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        g_gate[i] = lstm_candidate_ih_bias[i] + lstm_candidate_hh_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            g_gate[i] += input[j] * lstm_candidate_ih_weights[i][j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            g_gate[i] += h_prev[j] * lstm_candidate_hh_weights[i][j];
        }
        g_gate[i] = tanh_activation(g_gate[i]);
    }

    // Output Gate
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        o_gate[i] = lstm_output_ih_bias[i] + lstm_output_hh_bias[i];
        for (int j = 0; j < INPUT_SIZE; j++) {
            o_gate[i] += input[j] * lstm_output_ih_weights[i][j];
        }
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            o_gate[i] += h_prev[j] * lstm_output_hh_weights[i][j];
        }
        o_gate[i] = sigmoid(o_gate[i]);
    }

    // Cell State Update
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        c_new[i] = f_gate[i] * c_prev[i] + i_gate[i] * g_gate[i];
    }

    // Hidden State Update
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_new[i] = o_gate[i] * tanh_activation(c_new[i]);
    }
}

// --- MLP Forward Pass ---
// Adjusted to match the 3-layer MLP from the Python script
float mlp_forward(const float lstm_output[HIDDEN_SIZE]) {
    float mlp_layer1_out[MLP_HIDDEN_SIZE];
    float mlp_layer2_out[MLP_HIDDEN_SIZE];

    // MLP Layer 1 (Linear -> ReLU)
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        mlp_layer1_out[i] = mlp_fc1_bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            mlp_layer1_out[i] += lstm_output[j] * mlp_fc1_weights[i][j];
        }
        mlp_layer1_out[i] = relu(mlp_layer1_out[i]);
    }

    // MLP Layer 2 (Linear -> ReLU)
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        mlp_layer2_out[i] = mlp_fc2_bias[i];
        for (int j = 0; j < MLP_HIDDEN_SIZE; j++) {
            mlp_layer2_out[i] += mlp_layer1_out[j] * mlp_fc2_weights[i][j];
        }
        mlp_layer2_out[i] = relu(mlp_layer2_out[i]);
    }

    // MLP Layer 3 (Linear Output)
    float output = mlp_fc3_bias[0];
    for (int i = 0; i < MLP_HIDDEN_SIZE; i++) {
        output += mlp_layer2_out[i] * mlp_fc3_weights[0][i];
    }
    return output;
}

// --- Full Window Prediction ---
// Processes a window of data and returns the SOC prediction for the *last* time step.
float predict_soc_for_window(const float window_data[WINDOW_SIZE][INPUT_SIZE]) {
    unsigned long inference_start_micros = micros();

    // Initialize hidden and cell states to zero for each new window
    for (int i = 0; i < HIDDEN_SIZE; i++) {
        h_state[i] = 0.0f;
        c_state[i] = 0.0f;
    }

    float current_h_state[HIDDEN_SIZE];
    float current_c_state[HIDDEN_SIZE];

    // Process each time step in the window
    for (int t = 0; t < WINDOW_SIZE; t++) {
        lstm_cell_forward(window_data[t], h_state, c_state, current_h_state, current_c_state);
        // Update states for the next time step
        for (int i = 0; i < HIDDEN_SIZE; i++) {
            h_state[i] = current_h_state[i];
            c_state[i] = current_c_state[i];
        }
    }

    // MLP forward pass using the *last* hidden state of the LSTM sequence
    float soc = mlp_forward(h_state);

    last_inference_time_us = micros() - inference_start_micros;
    return soc;
}

// --- Arduino RAM/Flash Helper Functions (Placeholder - Implement based on target Arduino board) ---
int getFreeRam() {
  // extern int __heap_start, *__brkval;
  // int v;
  // return (int) &v - (__brkval == 0 ? (int) &__heap_start : (int) __brkval);
  return 0; // Placeholder
}

int getTotalRam() {
  return 2048; // Placeholder for Uno, adjust for others (e.g., 8192 for Mega)
}

int getUsedRam() {
  return getTotalRam() - getFreeRam();
}

float getRamFragmentation() { // This is a very simplified view
  return 0.0; // Placeholder
}

int getFreeFlash() {
    // This is more complex and often not directly available at runtime without specific bootloader/library support.
    return 0; // Placeholder
}
int getTotalFlash() {
    return 32256; // Placeholder for Uno (32KB), adjust for others
}
int getUsedFlash() {
    // Typically, this is the program size reported by the compiler.
    // Runtime check is hard. For estimation, you can subtract an estimated free flash.
    return getTotalFlash() - getFreeFlash(); // Placeholder
}
float getFlashUtilization() {
    if (getTotalFlash() == 0) return 0.0;
    return ((float)getUsedFlash() / getTotalFlash()) * 100.0;
}

void printRamStats() {
  int totalRam = getTotalRam();
  int usedRam = getUsedRam();
  float ramFragmentation = getRamFragmentation(); // Simplified

  int totalFlash = getTotalFlash();
  int usedFlash = getUsedFlash(); // This is an estimation
  float flashUtilization = getFlashUtilization();

  Serial.println(F("--- MEMORY STATUS (Window-based Model) ---"));
  Serial.print(F("RAM: "));
  Serial.print(usedRam / 1024.0, 1);
  Serial.print(F("/"));
  Serial.print(totalRam / 1024.0, 1);
  Serial.print(F(" KB | Frag (approx): "));
  Serial.print(ramFragmentation, 1);
  Serial.println(F("%"));

  Serial.print(F("Flash (est.): "));
  Serial.print(usedFlash / 1024.0, 1);
  Serial.print(F("/"));
  Serial.print(totalFlash / 1024.0, 1);
  Serial.print(F(" KB | Util (est.): "));
  Serial.print(flashUtilization, 1);
  Serial.println(F("%"));
}

// --- Setup & Loop ---
void setup() {
  Serial.begin(115200);
  long entryTime = millis();
  while (!Serial && (millis() - entryTime < 4000)) { // Wait up to 4 seconds
    delay(10);
  }
  Serial.println(F("--- ARDUINO WINDOW-BASED LSTM SOC (32x32) --- "));
  Serial.println(F("Model based on BMS_SOC_LSTM_windows_2.1.4.1"));
  Serial.print(F("Window Size: ")); Serial.println(WINDOW_SIZE);
  Serial.print(F("Input Size: ")); Serial.println(INPUT_SIZE);
  Serial.print(F("LSTM Hidden: ")); Serial.println(HIDDEN_SIZE);
  Serial.print(F("MLP Hidden: ")); Serial.println(MLP_HIDDEN_SIZE);
  delay(100);
  printRamStats();
  delay(100);
  Serial.println(F("Ready for window data..."));
}

// Example window data (WINDOW_SIZE x INPUT_SIZE)
// In a real application, this would be filled with actual sensor readings for a full window.
float example_window_data[WINDOW_SIZE][INPUT_SIZE]; 

void loop() {
  // 1. Fill `example_window_data` with your sensor readings for one full window.
  //    This is just a placeholder, you need to implement data acquisition.
  //    For testing, you can fill it with dummy data.
  for (int t = 0; t < WINDOW_SIZE; t++) {
    example_window_data[t][0] = 3.5f + sin(t * 0.1f) * 0.5f; // Example Voltage
    example_window_data[t][1] = 1.0f + cos(t * 0.1f) * 0.2f; // Example Current
    example_window_data[t][2] = 0.95f;                       // Example SOH
    example_window_data[t][3] = 2.0f;                        // Example Q_m
  }

  // 2. Predict SOC for the current window
  float soc_prediction = predict_soc_for_window(example_window_data);

  // 3. Print results
  Serial.print(F("Predicted SOC: "));
  Serial.print(soc_prediction, 4);
  Serial.print(F(" ("));
  Serial.print(soc_prediction * 100.0, 2);
  Serial.print(F("%) | Inference Time: "));
  Serial.print(last_inference_time_us);
  Serial.println(F(" us"));

  // 4. Optional: Print RAM/Flash stats periodically
  // printRamStats(); 

  delay(5000); // Wait before processing the next window
}

