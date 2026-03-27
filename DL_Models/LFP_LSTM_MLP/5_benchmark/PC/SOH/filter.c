#include <math.h>

void apply_filter_c(const float* input, float* output, int n, float rel_cap, float abs_cap, float ema_alpha) {
    float last = input[0];
    float ema = input[0];
    output[0] = last;
    
    for (int i = 1; i < n; i++) {
        float v = input[i];
        
        // Cap
        float cap = -1.0f;
        if (rel_cap > 0) {
            float rc = fabsf(last) * rel_cap;
            cap = rc;
        }
        if (abs_cap > 0) {
            if (cap < 0 || abs_cap < cap) cap = abs_cap;
        }
        
        if (cap > 0) {
            float delta = v - last;
            if (fabsf(delta) > cap) {
                v = last + (delta > 0 ? cap : -cap);
            }
        }
        
        // EMA
        if (ema_alpha > 0) {
            ema = ema_alpha * v + (1.0f - ema_alpha) * ema;
            v = ema;
        }
        
        output[i] = v;
        last = v;
    }
}
