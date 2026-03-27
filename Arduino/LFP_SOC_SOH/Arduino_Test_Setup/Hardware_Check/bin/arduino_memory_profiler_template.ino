
// Arduino Memory Measurement Template
extern "C" char* sbrk(int incr);

class MemoryProfiler {
private:
    struct MemorySnapshot {
        int free_ram;
        unsigned long timestamp;
        String label;
    };
    
    MemorySnapshot snapshots[10];
    int snapshot_count = 0;

public:
    int getFreeRAM() {
        char top;
        return &top - reinterpret_cast<char*>(sbrk(0));
    }
    
    void takeSnapshot(String label) {
        if (snapshot_count < 10) {
            snapshots[snapshot_count] = {getFreeRAM(), millis(), label};
            snapshot_count++;
        }
    }
    
    void printReport() {
        Serial.println("=== MEMORY PROFILING REPORT ===");
        for (int i = 0; i < snapshot_count; i++) {
            Serial.print(snapshots[i].label);
            Serial.print(": ");
            Serial.print(snapshots[i].free_ram);
            Serial.println(" bytes free");
            
            if (i > 0) {
                int diff = snapshots[i-1].free_ram - snapshots[i].free_ram;
                Serial.print("  -> Used: ");
                Serial.print(diff);
                Serial.println(" bytes");
            }
        }
    }
};

MemoryProfiler profiler;

void setup() {
    Serial.begin(115200);
    
    profiler.takeSnapshot("Startup");
    
    // LSTM Weights initialization
    // ... your LSTM setup code ...
    profiler.takeSnapshot("After LSTM Weights");
    
    // Hidden State allocation
    // ... your hidden state setup ...
    profiler.takeSnapshot("After Hidden State");
    
    // Buffers allocation
    // ... your buffer setup ...
    profiler.takeSnapshot("After Buffers");
    
    profiler.printReport();
}
