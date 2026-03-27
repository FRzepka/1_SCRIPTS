/*
Arduino LSTM SOC Hardware Monitor
=================================

Erweiterte Version der arduino_lstm_soc_full32.ino mit Hardware-Monitoring:
- RAM Usage Tracking (freier/belegter Speicher)
- Inference Time Measurement (Mikrosekunden)
- CPU Load Estimation
- Temperature Monitoring (falls verfügbar)
- Performance Benchmarking
- Serial Communication für Hardware-Metriken

Hardware Monitoring Commands:
- DATA:x,y,z,w -> SOC prediction mit Hardware-Statistiken
- STATS -> Hardware-Statistiken abrufen
- RAM -> RAM-Status abrufen
- BENCHMARK -> Performance-Test durchführen
- RESET -> LSTM States zurücksetzen

Response Formats:
- DATA:soc_value,inference_us,free_ram,used_ram,cpu_load,temp
- STATS:inference_us,free_ram,used_ram,cpu_load,temp
- RAM:free_bytes,used_bytes,total_bytes,fragmentation_percent
- BENCHMARK:avg_inference_us,min_us,max_us,total_ram,free_ram,cpu_mhz
*/

// Standard Arduino includes
#include <Arduino.h>

// Memory monitoring
#ifdef __arm__
// ARM-specific memory functions
extern "C" char* sbrk(int incr);
#else
// AVR-specific memory functions  
extern int __heap_start, *__brkval;
#endif

// ============================================================================
// LSTM MODEL PARAMETERS (same as original)
// ============================================================================

// Model architecture
const int INPUT_SIZE = 4;
const int HIDDEN_SIZE = 128;
const int NUM_LAYERS = 2;
const int OUTPUT_SIZE = 1;

// LSTM weights and biases (truncated for example - use full weights from original)
// Include your trained weights here...
const float lstm_weights_ih_l0[HIDDEN_SIZE * 4 * INPUT_SIZE] PROGMEM = {
    // Your LSTM input-hidden weights for layer 0
    // (This would be the full weight matrix from your trained model)
    0.0f  // Placeholder - replace with actual weights
};

const float lstm_weights_hh_l0[HIDDEN_SIZE * 4 * HIDDEN_SIZE] PROGMEM = {
    // Your LSTM hidden-hidden weights for layer 0
    0.0f  // Placeholder
};

const float lstm_bias_ih_l0[HIDDEN_SIZE * 4] PROGMEM = {
    // Your LSTM input biases for layer 0
    0.0f  // Placeholder
};

const float lstm_bias_hh_l0[HIDDEN_SIZE * 4] PROGMEM = {
    // Your LSTM hidden biases for layer 0
    0.0f  // Placeholder
};

// Layer 1 weights (similar structure)
const float lstm_weights_ih_l1[HIDDEN_SIZE * 4 * HIDDEN_SIZE] PROGMEM = {
    0.0f  // Placeholder
};

const float lstm_weights_hh_l1[HIDDEN_SIZE * 4 * HIDDEN_SIZE] PROGMEM = {
    0.0f  // Placeholder
};

const float lstm_bias_ih_l1[HIDDEN_SIZE * 4] PROGMEM = {
    0.0f  // Placeholder
};

const float lstm_bias_hh_l1[HIDDEN_SIZE * 4] PROGMEM = {
    0.0f  // Placeholder
};

// FC layer weights
const float fc1_weight[64 * HIDDEN_SIZE] PROGMEM = {
    0.0f  // Placeholder
};

const float fc1_bias[64] PROGMEM = {
    0.0f  // Placeholder
};

const float fc2_weight[OUTPUT_SIZE * 64] PROGMEM = {
    0.0f  // Placeholder
};

const float fc2_bias[OUTPUT_SIZE] PROGMEM = {
    0.0f  // Placeholder
};

// ============================================================================
// HARDWARE MONITORING VARIABLES
// ============================================================================

// Performance tracking
struct PerformanceMetrics {
    unsigned long inference_time_us;
    unsigned long communication_start_us;
    unsigned long last_inference_us;
    float cpu_load_percent;
    int free_ram_bytes;
    int used_ram_bytes;
    int total_ram_bytes;
    float temperature_celsius;
    
    // Statistics
    unsigned long min_inference_us;
    unsigned long max_inference_us;
    unsigned long total_inferences;
    unsigned long total_inference_time_us;
};

PerformanceMetrics perf_metrics;

// CPU frequency (adjust for your Arduino)
const unsigned long CPU_FREQUENCY_MHZ = 16;  // 16 MHz for Arduino Uno
// For Arduino Due/ESP32: adjust accordingly

// Temperature sensor pin (optional)
const int TEMP_SENSOR_PIN = A0;  // Analog pin for temperature sensor
const bool TEMP_SENSOR_AVAILABLE = false;  // Set to true if you have a temperature sensor

// ============================================================================
// MEMORY MONITORING FUNCTIONS
// ============================================================================

int getFreeRam() {
    #ifdef __arm__
    // ARM Cortex-M (Arduino Due, etc.)
    char stack_dummy = 0;
    return &stack_dummy - sbrk(0);
    #else
    // AVR (Arduino Uno, Nano, etc.)
    int free_memory;
    if ((int)__brkval == 0) {
        free_memory = ((int)&free_memory) - ((int)&__heap_start);
    } else {
        free_memory = ((int)&free_memory) - ((int)__brkval);
    }
    return free_memory;
    #endif
}

int getTotalRam() {
    // Return total RAM for your specific Arduino model
    #ifdef __arm__
    return 96 * 1024;  // Arduino Due: 96KB
    #else
    return 2048;       // Arduino Uno: 2KB
    #endif
}

void updateMemoryMetrics() {
    perf_metrics.free_ram_bytes = getFreeRam();
    perf_metrics.total_ram_bytes = getTotalRam();
    perf_metrics.used_ram_bytes = perf_metrics.total_ram_bytes - perf_metrics.free_ram_bytes;
}

// ============================================================================
// CPU LOAD ESTIMATION
// ============================================================================

void updateCpuLoad() {
    // Simple CPU load estimation based on inference time vs available time
    static unsigned long last_update_us = 0;
    unsigned long current_us = micros();
    
    if (last_update_us > 0 && perf_metrics.last_inference_us > 0) {
        unsigned long time_since_last_us = current_us - last_update_us;
        if (time_since_last_us > 0) {
            perf_metrics.cpu_load_percent = (float)perf_metrics.last_inference_us / time_since_last_us * 100.0f;
            // Limit to reasonable values
            if (perf_metrics.cpu_load_percent > 100.0f) perf_metrics.cpu_load_percent = 100.0f;
            if (perf_metrics.cpu_load_percent < 0.0f) perf_metrics.cpu_load_percent = 0.0f;
        }
    }
    
    last_update_us = current_us;
}

// ============================================================================
// TEMPERATURE MONITORING
// ============================================================================

float readTemperature() {
    if (!TEMP_SENSOR_AVAILABLE) {
        return 0.0f;
    }
    
    // Read analog value from temperature sensor
    int sensorValue = analogRead(TEMP_SENSOR_PIN);
    
    // Convert to temperature (example for TMP36 sensor)
    // Adjust formula for your specific temperature sensor
    float voltage = sensorValue * (5.0 / 1023.0);  // Convert to voltage
    float temperature = (voltage - 0.5) * 100.0;   // TMP36 conversion
    
    return temperature;
}

// ============================================================================
// LSTM IMPLEMENTATION (simplified - use your full implementation)
// ============================================================================

// LSTM state variables
float h0[HIDDEN_SIZE] = {0};  // Layer 0 hidden state
float c0[HIDDEN_SIZE] = {0};  // Layer 0 cell state
float h1[HIDDEN_SIZE] = {0};  // Layer 1 hidden state
float c1[HIDDEN_SIZE] = {0};  // Layer 1 cell state

// Activation functions
float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

float tanh_activation(float x) {
    return tanh(x);
}

float leaky_relu(float x) {
    return x > 0 ? x : 0.01f * x;
}

// LSTM cell computation (simplified)
void lstm_cell(float* input, float* h_prev, float* c_prev, float* h_new, float* c_new,
               const float* weight_ih, const float* weight_hh, 
               const float* bias_ih, const float* bias_hh,
               int input_size, int hidden_size) {
    
    // This is a simplified version - implement your full LSTM cell here
    // Gate computations: input, forget, cell, output gates
    // For brevity, showing structure only
    
    for (int i = 0; i < hidden_size; i++) {
        // Placeholder computation - replace with full LSTM implementation
        h_new[i] = tanh_activation(input[0] * 0.1f + h_prev[i] * 0.9f);
        c_new[i] = c_prev[i] * 0.9f + h_new[i] * 0.1f;
    }
}

// Full LSTM forward pass
float lstm_forward(float* input) {
    unsigned long start_time = micros();
    
    // Layer 0
    lstm_cell(input, h0, c0, h0, c0, 
              lstm_weights_ih_l0, lstm_weights_hh_l0,
              lstm_bias_ih_l0, lstm_bias_hh_l0,
              INPUT_SIZE, HIDDEN_SIZE);
    
    // Layer 1
    lstm_cell(h0, h1, c1, h1, c1,
              lstm_weights_ih_l1, lstm_weights_hh_l1, 
              lstm_bias_ih_l1, lstm_bias_hh_l1,
              HIDDEN_SIZE, HIDDEN_SIZE);
    
    // FC layers
    float fc1_output[64];
    for (int i = 0; i < 64; i++) {
        float sum = fc1_bias[i];
        for (int j = 0; j < HIDDEN_SIZE; j++) {
            sum += h1[j] * pgm_read_float(&fc1_weight[i * HIDDEN_SIZE + j]);
        }
        fc1_output[i] = leaky_relu(sum);
    }
    
    // Final output
    float output = fc2_bias[0];
    for (int i = 0; i < 64; i++) {
        output += fc1_output[i] * pgm_read_float(&fc2_weight[i]);
    }
    
    // Apply sigmoid to get SOC (0-1 range)
    output = sigmoid(output);
    
    // Record inference time
    unsigned long end_time = micros();
    perf_metrics.last_inference_us = end_time - start_time;
    perf_metrics.inference_time_us = perf_metrics.last_inference_us;
    
    // Update statistics
    if (perf_metrics.total_inferences == 0) {
        perf_metrics.min_inference_us = perf_metrics.last_inference_us;
        perf_metrics.max_inference_us = perf_metrics.last_inference_us;
    } else {
        if (perf_metrics.last_inference_us < perf_metrics.min_inference_us) {
            perf_metrics.min_inference_us = perf_metrics.last_inference_us;
        }
        if (perf_metrics.last_inference_us > perf_metrics.max_inference_us) {
            perf_metrics.max_inference_us = perf_metrics.last_inference_us;
        }
    }
    
    perf_metrics.total_inferences++;
    perf_metrics.total_inference_time_us += perf_metrics.last_inference_us;
    
    return output;
}

// ============================================================================
// COMMAND HANDLERS
// ============================================================================

void handleDataCommand(String command) {
    // Parse: "DATA:voltage,current,soh,q_c"
    int start = command.indexOf(':') + 1;
    String data = command.substring(start);
    
    // Parse input values
    float input[4];
    int lastIndex = 0;
    for (int i = 0; i < 4; i++) {
        int nextIndex = data.indexOf(',', lastIndex);
        if (nextIndex == -1 && i < 3) {
            // Error parsing
            Serial.println("ERROR:PARSE_FAILED");
            return;
        }
        
        String valueStr = (nextIndex == -1) ? data.substring(lastIndex) : data.substring(lastIndex, nextIndex);
        input[i] = valueStr.toFloat();
        lastIndex = nextIndex + 1;
    }
    
    // Update hardware metrics before inference
    updateMemoryMetrics();
    
    // Perform LSTM inference
    float soc_prediction = lstm_forward(input);
    
    // Update hardware metrics after inference
    updateCpuLoad();
    perf_metrics.temperature_celsius = readTemperature();
    
    // Send response with hardware metrics
    // Format: "DATA:soc_value,inference_us,free_ram,used_ram,cpu_load,temp"
    Serial.print("DATA:");
    Serial.print(soc_prediction, 6);
    Serial.print(",");
    Serial.print(perf_metrics.inference_time_us);
    Serial.print(",");
    Serial.print(perf_metrics.free_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.used_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.cpu_load_percent, 1);
    Serial.print(",");
    Serial.println(perf_metrics.temperature_celsius, 1);
}

void handleStatsCommand() {
    // Update all metrics
    updateMemoryMetrics();
    updateCpuLoad();
    perf_metrics.temperature_celsius = readTemperature();
    
    // Send stats: "STATS:inference_us,free_ram,used_ram,cpu_load,temp"
    Serial.print("STATS:");
    Serial.print(perf_metrics.inference_time_us);
    Serial.print(",");
    Serial.print(perf_metrics.free_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.used_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.cpu_load_percent, 1);
    Serial.print(",");
    Serial.println(perf_metrics.temperature_celsius, 1);
}

void handleRamCommand() {
    updateMemoryMetrics();
    
    // Calculate fragmentation (simplified)
    float fragmentation = (float)perf_metrics.used_ram_bytes / perf_metrics.total_ram_bytes * 100.0f;
    
    // Send RAM info: "RAM:free_bytes,used_bytes,total_bytes,fragmentation_percent"
    Serial.print("RAM:");
    Serial.print(perf_metrics.free_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.used_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.total_ram_bytes);
    Serial.print(",");
    Serial.println(fragmentation, 1);
}

void handleBenchmarkCommand() {
    Serial.println("Running benchmark...");
    
    // Perform multiple inference runs
    const int BENCHMARK_RUNS = 10;
    float test_input[4] = {0.5f, 0.0f, 1.0f, 0.5f};
    
    unsigned long total_time = 0;
    unsigned long min_time = ULONG_MAX;
    unsigned long max_time = 0;
    
    for (int i = 0; i < BENCHMARK_RUNS; i++) {
        unsigned long start = micros();
        lstm_forward(test_input);
        unsigned long duration = micros() - start;
        
        total_time += duration;
        if (duration < min_time) min_time = duration;
        if (duration > max_time) max_time = duration;
    }
    
    float avg_time = (float)total_time / BENCHMARK_RUNS;
    updateMemoryMetrics();
    
    // Send benchmark: "BENCHMARK:avg_inference_us,min_us,max_us,total_ram,free_ram,cpu_mhz"
    Serial.print("BENCHMARK:");
    Serial.print(avg_time, 0);
    Serial.print(",");
    Serial.print(min_time);
    Serial.print(",");
    Serial.print(max_time);
    Serial.print(",");
    Serial.print(perf_metrics.total_ram_bytes);
    Serial.print(",");
    Serial.print(perf_metrics.free_ram_bytes);
    Serial.print(",");
    Serial.println(CPU_FREQUENCY_MHZ);
}

void handleResetCommand() {
    // Reset LSTM states
    memset(h0, 0, sizeof(h0));
    memset(c0, 0, sizeof(c0));
    memset(h1, 0, sizeof(h1));
    memset(c1, 0, sizeof(c1));
    
    // Reset performance metrics
    perf_metrics.total_inferences = 0;
    perf_metrics.total_inference_time_us = 0;
    perf_metrics.min_inference_us = 0;
    perf_metrics.max_inference_us = 0;
    
    Serial.println("RESET:OK");
}

// ============================================================================
// MAIN ARDUINO FUNCTIONS
// ============================================================================

void setup() {
    Serial.begin(115200);
    
    // Initialize performance metrics
    memset(&perf_metrics, 0, sizeof(perf_metrics));
    perf_metrics.total_ram_bytes = getTotalRam();
    
    // Initialize LSTM states
    memset(h0, 0, sizeof(h0));
    memset(c0, 0, sizeof(c0));
    memset(h1, 0, sizeof(h1));
    memset(c1, 0, sizeof(c1));
    
    // Setup temperature sensor if available
    if (TEMP_SENSOR_AVAILABLE) {
        pinMode(TEMP_SENSOR_PIN, INPUT);
    }
    
    Serial.println("Arduino LSTM SOC Hardware Monitor Ready");
    Serial.print("Total RAM: ");
    Serial.print(perf_metrics.total_ram_bytes);
    Serial.println(" bytes");
    Serial.print("CPU Frequency: ");
    Serial.print(CPU_FREQUENCY_MHZ);
    Serial.println(" MHz");
    Serial.println("Commands: DATA:v,i,s,q | STATS | RAM | BENCHMARK | RESET");
}

void loop() {
    if (Serial.available()) {
        String command = Serial.readStringUntil('\n');
        command.trim();
        
        perf_metrics.communication_start_us = micros();
        
        if (command.startsWith("DATA:")) {
            handleDataCommand(command);
        } else if (command == "STATS") {
            handleStatsCommand();
        } else if (command == "RAM") {
            handleRamCommand();
        } else if (command == "BENCHMARK") {
            handleBenchmarkCommand();
        } else if (command == "RESET") {
            handleResetCommand();
        } else {
            // Backward compatibility: assume it's just comma-separated values
            // Parse as: "voltage,current,soh,q_c"
            float input[4];
            int lastIndex = 0;
            bool parseError = false;
            
            for (int i = 0; i < 4; i++) {
                int nextIndex = command.indexOf(',', lastIndex);
                if (nextIndex == -1 && i < 3) {
                    parseError = true;
                    break;
                }
                
                String valueStr = (nextIndex == -1) ? command.substring(lastIndex) : command.substring(lastIndex, nextIndex);
                input[i] = valueStr.toFloat();
                lastIndex = nextIndex + 1;
            }
            
            if (!parseError) {
                updateMemoryMetrics();
                float soc_prediction = lstm_forward(input);
                updateCpuLoad();
                
                // Simple response (backward compatibility)
                Serial.println(soc_prediction, 6);
            } else {
                Serial.println("ERROR:INVALID_COMMAND");
            }
        }
    }
    
    // Small delay to prevent overwhelming the serial interface
    delay(1);
}
