# Arduino Memory Calculator - Complete Mathematical Formulas

This document contains all mathematical formulas used in the Arduino Memory Usage Calculator for LSTM neural networks, formatted for Obsidian with LaTeX support.

---

## 📋 Configuration Parameters

> [!info] Network Configuration
> - **Hidden Size**: $h = 32$ (LSTM hidden state dimension)
> - **Input Size**: $n = 4$ (number of input features)  
> - **Output Layers**: $[32, 32, 1]$ (MLP architecture)
> - **Data Type**: Float32 (4 bytes per value)
> - **Buffer Size**: $B = 256$ bytes

---

## 🧠 SRAM Memory Calculations

### Master Formula

> [!note]- Total SRAM Usage
> 
> $$\boxed{\text{SRAM}_{\text{total}} = \text{NN}_{\text{arch}} + \text{NN}_{\text{runtime}} + \text{System}_{\text{overhead}}}$$

### Neural Network Architecture Memory

> [!note]- LSTM State Memory
> 
> LSTM requires persistent storage for cell states and hidden states:
> 
> $$\text{LSTM}_{\text{states}} = h \times 4 \times 2 \times 4 \text{ bytes}$$
> 
> **Breakdown:**
> - $h = 32$ (hidden size/number of LSTM units)
> - First $4$: represents the 4 LSTM gates (forget, input, output, candidate)
> - $2$: for both cell state ($C_t$) and hidden state ($h_t$)
> - Last $4$: bytes per Float32 value
> 
> $$\text{LSTM}_{\text{states}} = 32 \times 4 \times 2 \times 4 = \boxed{1024 \text{ bytes}}$$

> [!note]- Output Layer Memory  
> 
> Memory buffers for MLP output layer computations:
> 
> $$\text{Output}_{\text{buffers}} = \left(\sum_{i=0}^{L-1} s_i\right) \times 4 \text{ bytes}$$
> 
> **For layers** $[s_0, s_1, s_2] = [32, 32, 1]$:
> $$\text{Output}_{\text{buffers}} = (32 + 32 + 1) \times 4 = \boxed{260 \text{ bytes}}$$

### Neural Network Runtime Memory

> [!note]- Input/Output Buffers
> 
> Temporary storage for data input and output:
> 
> $$\text{Buffers} = B_{\text{size}} \times 2 = 256 \times 2 = \boxed{512 \text{ bytes}}$$
> 
> Where the factor of 2 accounts for separate input and output buffers.

> [!note]- LSTM Temporary Computation Memory
> 
> Memory for intermediate LSTM gate calculations:
> 
> $$\text{LSTM}_{\text{temps}} = h \times 6 \times 2 \times 4 \text{ bytes}$$
> 
> **Breakdown:**
> - $h = 32$ (hidden size)
> - $6$: temporary variables per computation cycle (gate outputs, activations, intermediates)
> - $2$: double buffering for computation pipeline
> - $4$: Float32 byte size
> 
> $$\text{LSTM}_{\text{temps}} = 32 \times 6 \times 2 \times 4 = \boxed{1536 \text{ bytes}}$$

> [!note]- Neural Network Overhead
> 
> Fixed overhead for activation functions, loop counters, and NN framework:
> 
> $$\text{NN}_{\text{overhead}} = \boxed{512 \text{ bytes}}$$

### Arduino System Overhead Memory

> [!note]- System Component Breakdown
> 
> **Total System Overhead:**
> $$\text{System}_{\text{overhead}} = \sum_{i} \text{Component}_i$$
> 
> | Component | Formula | Value (bytes) |
> |-----------|---------|---------------|
> | Arduino Core | $\text{const}$ | $1536$ |
> | Serial Buffers | $\text{const}$ | $768$ |
> | Stack Space | $\text{const}$ | $2560$ |
> | Interrupt Vectors | $\text{const}$ | $512$ |
> | Memory Alignment | $\text{const}$ | $256$ |
> | Heap Reserve | $\text{const}$ | $512$ |
> 
> $$\text{System}_{\text{overhead}} = 1536 + 768 + 2560 + 512 + 256 + 512 = \boxed{6144 \text{ bytes}}$$

### Complete SRAM Calculation

> [!example]- Final SRAM Result
> 
> **Combining all components:**
> 
> $$\begin{align}
> \text{SRAM}_{\text{total}} &= \text{LSTM}_{\text{states}} + \text{Output}_{\text{buffers}} + \text{Buffers} \\
> &\quad + \text{LSTM}_{\text{temps}} + \text{NN}_{\text{overhead}} + \text{System}_{\text{overhead}} \\[10pt]
> &= 1024 + 260 + 512 + 1536 + 512 + 6144 \\[5pt]
> &= \boxed{9988 \text{ bytes} \approx 9.76 \text{ KB}}
> \end{align}$$

---

## 💾 Flash Memory Calculations

### Master Formula

> [!note]- Total Flash Usage
> 
> $$\boxed{\text{Flash}_{\text{total}} = \text{LSTM}_{\text{weights}} + \text{Output}_{\text{weights}} + \text{Application}_{\text{code}}}$$

### LSTM Weight Storage

> [!note]- LSTM Architecture Deep Dive
> 
> **LSTM Gate Structure:** Each LSTM cell has 4 gates with the following weight matrices:
> 
> 1. **Forget Gate:** $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$
> 2. **Input Gate:** $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$ 
> 3. **Candidate Gate:** $\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$
> 4. **Output Gate:** $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

> [!note]- Input-to-Hidden Weights
> 
> Weights connecting input features to LSTM gates:
> 
> $$W_{ih} = 4 \times n \times h \times 4 \text{ bytes}$$
> 
> **Breakdown:**
> - $4$: number of LSTM gates
> - $n = 4$: input feature size
> - $h = 32$: hidden state size
> - $4$: Float32 byte size
> 
> $$W_{ih} = 4 \times 4 \times 32 \times 4 = \boxed{2048 \text{ bytes}}$$

> [!note]- Hidden-to-Hidden Weights
> 
> Recurrent weights connecting previous hidden state to current gates:
> 
> $$W_{hh} = 4 \times h \times h \times 4 \text{ bytes}$$
> 
> $$W_{hh} = 4 \times 32 \times 32 \times 4 = \boxed{16384 \text{ bytes}}$$

> [!note]- LSTM Biases
> 
> Bias terms for each gate:
> 
> $$b = 4 \times h \times 4 \text{ bytes}$$
> 
> $$b = 4 \times 32 \times 4 = \boxed{512 \text{ bytes}}$$

> [!note]- Total LSTM Weights
> 
> $$\text{LSTM}_{\text{weights}} = W_{ih} + W_{hh} + b$$
> $$\text{LSTM}_{\text{weights}} = 2048 + 16384 + 512 = \boxed{18944 \text{ bytes}}$$

### Output Layer Weight Storage

> [!note]- MLP Layer-by-Layer Calculation
> 
> **For MLP architecture** $[32, 32, 1]$:
> 
> **Layer 0 → Layer 1:** $(32 \to 32)$
> - Weights: $W_0 = 32 \times 32 \times 4 = 4096$ bytes
> - Biases: $b_0 = 32 \times 4 = 128$ bytes
> - Subtotal: $4096 + 128 = 4224$ bytes
> 
> **Layer 1 → Layer 2:** $(32 \to 1)$ 
> - Weights: $W_1 = 32 \times 1 \times 4 = 128$ bytes
> - Biases: $b_1 = 1 \times 4 = 4$ bytes
> - Subtotal: $128 + 4 = 132$ bytes
> 
> $$\text{Output}_{\text{weights}} = 4224 + 132 = \boxed{4356 \text{ bytes}}$$

> [!note]- General MLP Weight Formula
> 
> For consecutive layer sizes $[s_0, s_1, s_2, ..., s_L]$:
> 
> $$\text{Output}_{\text{weights}} = \sum_{i=0}^{L-1} \left[(s_i \times s_{i+1} + s_{i+1}) \times 4\right]$$

### Application Code Storage

> [!note]- Code Base Components
> 
> $$\text{Application}_{\text{code}} = \text{Code}_{\text{base}} + \text{Constants}_{\text{data}}$$
> 
> **Components:**
> - $\text{Code}_{\text{base}} = 92160$ bytes (90 KB) - Compiled application logic
> - $\text{Constants}_{\text{data}} = 2048$ bytes (2 KB) - Lookup tables, strings, literals
> 
> $$\text{Application}_{\text{code}} = 92160 + 2048 = \boxed{94208 \text{ bytes}}$$

### Complete Flash Calculation

> [!example]- Final Flash Result
> 
> **Combining all Flash components:**
> 
> $$\begin{align}
> \text{Flash}_{\text{total}} &= \text{LSTM}_{\text{weights}} + \text{Output}_{\text{weights}} + \text{Application}_{\text{code}} \\[5pt]
> &= 18944 + 4356 + 94208 \\[5pt]
> &= \boxed{117508 \text{ bytes} \approx 114.8 \text{ KB}}
> \end{align}$$

---

## 📊 Accuracy & Validation Formulas

### Prediction Error Metrics

> [!success]- Error Calculation Formula
> 
> $$\text{Error}_{\%} = \frac{|\text{Predicted} - \text{Real}|}{\text{Real}} \times 100\%$$

> [!success]- SRAM Accuracy Assessment
> 
> **Given measurements:**
> - Predicted SRAM: $9988$ bytes
> - Real SRAM: $8984$ bytes
> 
> $$\text{SRAM Error} = \frac{|9988 - 8984|}{8984} \times 100\% = \frac{1004}{8984} \times 100\% = \boxed{11.2\%}$$
> 
> **Classification:** ✅ **GOOD** accuracy (< 20% error)

> [!success]- Flash Accuracy Assessment
> 
> **Given measurements:**
> - Predicted Flash: $117508$ bytes
> - Real Flash: $122880$ bytes (120 KB)
> 
> $$\text{Flash Error} = \frac{|117508 - 122880|}{122880} \times 100\% = \frac{5372}{122880} \times 100\% = \boxed{4.4\%}$$
> 
> **Classification:** ✅ **EXCELLENT** accuracy (< 10% error)

### Accuracy Categories

> [!tip]- Performance Benchmarks
> 
> | Error Range | Classification | Status |
> |-------------|----------------|--------|
> | $< 10\%$ | **EXCELLENT** | ✅ Ready for production |
> | $10\% - 20\%$ | **GOOD** | ✅ Acceptable accuracy |
> | $> 20\%$ | **NEEDS IMPROVEMENT** | ⚠️ Requires calibration |

---

## 🔧 Arduino Hardware Utilization

### Memory Utilization Formulas

> [!warning]- Arduino Uno R4 WiFi Specifications
> 
> **Hardware Limits:**
> - SRAM: 32 KB (32,768 bytes)
> - Flash: 256 KB (262,144 bytes)

> [!warning]- SRAM Utilization
> 
> $$\text{SRAM}_{\text{utilization}} = \frac{\text{SRAM}_{\text{used}}}{\text{SRAM}_{\text{total}}} \times 100\%$$
> 
> $$\text{SRAM}_{\text{utilization}} = \frac{8984}{32768} \times 100\% = \boxed{27.4\%}$$
> 
> **Status:** ✅ Safe usage level (< 70%)

> [!warning]- Flash Utilization
> 
> $$\text{Flash}_{\text{utilization}} = \frac{\text{Flash}_{\text{used}}}{\text{Flash}_{\text{total}}} \times 100\%$$
> 
> $$\text{Flash}_{\text{utilization}} = \frac{122880}{262144} \times 100\% = \boxed{46.9\%}$$
> 
> **Status:** ✅ Moderate usage level (< 80%)

### Memory Safety Thresholds

> [!danger]- Critical Memory Limits
> 
> **SRAM Safety Levels:**
> - **Green Zone:** < 70% (< 22.9 KB) - Safe operation
> - **Yellow Zone:** 70-90% (22.9-29.5 KB) - Monitor carefully  
> - **Red Zone:** > 90% (> 29.5 KB) - Risk of stack overflow
> 
> **Flash Safety Levels:**
> - **Green Zone:** < 80% (< 204.8 KB) - Comfortable headroom
> - **Yellow Zone:** 80-95% (204.8-243.2 KB) - Limited space
> - **Red Zone:** > 95% (> 243.2 KB) - Near capacity limit

---

## 🧮 Advanced Mathematical Details

### LSTM Gate Computations (Deep Dive)

> [!note]- Mathematical Memory Requirements
> 
> **Per-Gate Computation Memory:**
> 
> For each gate $g \in \{f, i, C, o\}$ (forget, input, candidate, output):
> 
> $$\text{Gate}_g = \text{activation}(W_g^{(ih)} \times x_t + W_g^{(hh)} \times h_{t-1} + b_g)$$
> 
> **Memory per gate:**
> - Input multiplication: $n \times h \times 4$ bytes
> - Hidden multiplication: $h \times h \times 4$ bytes  
> - Bias addition: $h \times 4$ bytes
> - Activation result: $h \times 4$ bytes
> 
> **Total per gate:** $(n + h + 2) \times h \times 4$ bytes
> **All 4 gates:** $4 \times (n + h + 2) \times h \times 4$ bytes

> [!note]- Cell State Update Memory
> 
> **Cell State Formula:**
> $$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
> 
> **Hidden State Formula:**  
> $$h_t = o_t \odot \tanh(C_t)$$
> 
> **Memory Requirements:**
> - Previous cell state: $h \times 4$ bytes
> - Current cell state: $h \times 4$ bytes
> - Hidden state: $h \times 4$ bytes
> - Temporary tanh result: $h \times 4$ bytes
> 
> **Total state memory:** $4 \times h \times 4$ bytes

### Weight Storage Optimization Formulas  

> [!tip]- Quantization Memory Savings
> 
> **INT8 Quantization:**
> $$\text{Memory}_{\text{INT8}} = \frac{\text{Memory}_{\text{Float32}}}{4}$$
> 
> **Potential Flash savings:**
> $$\text{Flash}_{\text{optimized}} = \frac{18944 + 4356}{4} + 94208 = 99984 \text{ bytes}$$
> 
> **Savings:** $117508 - 99984 = 17524$ bytes (14.9% reduction)

> [!tip]- Weight Pruning Impact
> 
> **With sparsity ratio** $s$ (fraction of weights set to zero):
> 
> $$\text{Sparse Memory} = \text{Dense Memory} \times (1 - s) + \text{Index Overhead}$$
> 
> **For 50% sparsity:**
> $$\text{Sparse Weights} = 23300 \times 0.5 + 2330 = 14000 \text{ bytes}$$
> 
> Where index overhead ≈ 10% of original size for sparse storage format.

---

## 📋 Summary & Reference Tables

> [!abstract]- Quick Reference - Key Formulas
> 
> | Component | Formula | Result |
> |-----------|---------|--------|
> | **LSTM States** | $h \times 4 \times 2 \times 4$ | 1,024 bytes |
> | **Output Buffers** | $(32+32+1) \times 4$ | 260 bytes |
> | **Runtime Temps** | $h \times 6 \times 2 \times 4$ | 1,536 bytes |
> | **System Overhead** | $\sum \text{Arduino Components}$ | 6,144 bytes |
> | **SRAM Total** | $\sum \text{All SRAM}$ | **9,988 bytes** |
> | **LSTM Weights** | $4 \times (n \times h + h \times h + h) \times 4$ | 18,944 bytes |
> | **Output Weights** | $\sum \text{Layer Connections}$ | 4,356 bytes |
> | **Application Code** | $92160 + 2048$ | 94,208 bytes |
> | **Flash Total** | $\sum \text{All Flash}$ | **117,508 bytes** |

> [!abstract]- Accuracy Summary
> 
> | Metric | Predicted | Real | Error | Status |
> |--------|-----------|------|-------|--------|
> | **SRAM** | 9,988 B | 8,984 B | 11.2% | ✅ Good |
> | **Flash** | 117,508 B | 122,880 B | 4.4% | ✅ Excellent |
> | **SRAM %** | 30.5% | 27.4% | - | ✅ Safe |
> | **Flash %** | 44.8% | 46.9% | - | ✅ Safe |

This comprehensive mathematical reference provides all formulas needed to understand, validate, and optimize Arduino memory usage for LSTM neural networks.
