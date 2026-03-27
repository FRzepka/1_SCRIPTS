"""
Export RobustScaler parameters to C header
"""
import joblib
from pathlib import Path

# Load scaler
scaler_path = Path(__file__).parent.parent.parent.parent / '1_training' / '1.5.0.0' / 'outputs' / 'scaler_robust.joblib'
scaler = joblib.load(scaler_path)

# Get parameters
center = scaler.center_
scale = scaler.scale_

print("Extracted RobustScaler parameters:")
print(f"Center: {center}")
print(f"Scale: {scale}")

# Create C header
output_path = Path(__file__).parent / 'scaler_params.h'

with open(output_path, 'w') as f:
    f.write("""/*
 * RobustScaler parameters for SOC model preprocessing
 * Features: Voltage[V], Current[A], Temperature[°C], Q_c, dU_dt[V/s], dI_dt[A/s]
 * DO NOT EDIT MANUALLY
 */

#ifndef SCALER_PARAMS_H
#define SCALER_PARAMS_H

#define SCALER_NUM_FEATURES 6

/* RobustScaler center values (subtract from raw input) */
const float SCALER_CENTER[SCALER_NUM_FEATURES] = {
""")
    
    # Write center values
    for i, val in enumerate(center):
        f.write(f"    {val:.10f}f")
        if i < len(center) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    
    f.write("};\n\n")
    f.write("/* RobustScaler scale values (divide by these after centering) */\n")
    f.write("const float SCALER_SCALE[SCALER_NUM_FEATURES] = {\n")
    
    # Write scale values
    for i, val in enumerate(scale):
        f.write(f"    {val:.10f}f")
        if i < len(scale) - 1:
            f.write(",\n")
        else:
            f.write("\n")
    
    f.write("};\n\n")
    f.write("""/* Apply RobustScaler transformation: scaled = (raw - center) / scale */
static inline void scaler_transform(const float raw[SCALER_NUM_FEATURES], 
                                    float scaled[SCALER_NUM_FEATURES])
{
    for (int i = 0; i < SCALER_NUM_FEATURES; i++) {
        scaled[i] = (raw[i] - SCALER_CENTER[i]) / SCALER_SCALE[i];
    }
}

#endif /* SCALER_PARAMS_H */
""")

print(f"\n✅ Scaler parameters exported to: {output_path}")
print(f"   Center: [{', '.join(f'{v:.3f}' for v in center)}]")
print(f"   Scale:  [{', '.join(f'{v:.3f}' for v in scale)}]")
