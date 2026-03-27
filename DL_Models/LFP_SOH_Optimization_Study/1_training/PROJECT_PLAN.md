# Project Plan: Physics-Informed Linear Transformer (RWKV) for SOC/SOH

## 1. Concept
Combine a **Linear Transformer (RWKV)** with a **Differentiable Equivalent Circuit Model (ECM)**.
- **Goal:** State-of-the-art accuracy with constant memory usage (Stateful Inference) on STM32.
- **Novelty:** First implementation of a quantized, physics-informed Linear Transformer on a microcontroller for battery diagnostics.

## 2. Architecture
1.  **Neural Network (The "Brain"):**
    - Type: Linear Transformer / RWKV (Receptance Weighted Key Value).
    - Input: Current ($I$), Voltage ($U$), Temperature ($T$).
    - Output: ECM Parameters ($\hat{R}_0, \hat{R}_1, \hat{C}_1, \hat{Q}_{max}$).
    - Constraint: Stateful inference (constant RAM, no sliding window).
2.  **Physics Layer (The "Skeleton"):**
    - Type: Thevenin Model (1RC or 2RC).
    - Logic: Calculates terminal voltage $\hat{U}$ based on NN-predicted parameters and physical laws (Ohm's law, OCV curve).
    - Differentiable: Implemented in PyTorch so gradients flow back to the NN.

## 3. Implementation Steps

### Phase 1: Setup & Data
- [ ] **Data:** Reuse `LFP DoE Cycle Ageing Dataset`.
- [ ] **OCV Curve:** Extract a robust OCV vs. SOC curve from low-current discharges in the dataset. Store as Lookup Table.

### Phase 2: Modeling (Python/PyTorch)
- [ ] **Implement Differentiable ECM:** Write a PyTorch module that takes $I, SOC_{prev}, \text{Params}$ and outputs $U, SOC_{new}$.
- [ ] **Implement RWKV/Linear Attention:** Build the neural network block. Crucial: Must support "RNN-mode" (state passing) for inference.
- [ ] **Training Loop:**
    - Loss Function: $MSE(\hat{U} - U_{measured})$.
    - Optional: Add regularization to keep parameters ($R, C$) in physically plausible ranges.

### Phase 3: Compression
- [ ] **Pruning:** Apply structured pruning to the Linear Transformer weights.
- [ ] **Quantization:** Quantize weights to INT8 (Post-Training Quantization or QAT).

### Phase 4: Embedded Deployment (C/STM32)
- [ ] **C-Kernel:** Implement the RWKV state update formula in C (Matrix-Vector multiplications).
- [ ] **ECM in C:** Implement the physics equations.
- [ ] **State Management:** Ensure the "State Matrix" is stored in RAM and updated correctly at each second.
- [ ] **Integration:** Combine NN and ECM on the STM32H7.

### Phase 5: Validation
- [ ] **Accuracy:** Compare MAE (SOC & SOH) against the pure LSTM baseline.
- [ ] **Robustness:** Test on different temperatures (if data available) or unseen profiles.
- [ ] **Efficiency:** Measure Flash, RAM, and Latency on STM32.

## 4. Research Questions
- Can a Linear Transformer learn battery physics better than an LSTM?
- Does the hybrid approach improve SOH estimation without explicit SOH labels?
- Is INT8 quantization precise enough for the "Physics-Informed" feedback loop?
