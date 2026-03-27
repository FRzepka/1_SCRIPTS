# Arduino Memory Calculator - Mathematical Formulas

This document contains all mathematical formulas used in the Arduino Memory Calculator for LSTM neural networks.

## Configuration Parameters

> [!info] Network Configuration
> - **Hidden Size**: $h = 32$ (LSTM hidden state dimension)
> - **Input Size**: $n = 4$ (number of input features)
> - **Output Layers**: $[32, 32, 1]$ (MLP architecture)
> - **Data Type**: Float32 (4 bytes per value)

---

## SRAM Memory Calculations

> [!note]- SRAM Formula Overview
> 
> The total SRAM usage is calculated as:
> 
> $$\text{SRAM}_{\text{total}} = \text{NN}_{\text{architecture}} + \text{NN}_{\text{runtime}} + \text{System}_{\text{overhead}}$$

### Neural Network Architecture Memory

> [!note]- LSTM State Memory
> 
> LSTM requires storage for cell states and hidden states:
> 
> $$\text{LSTM}_{\text{states}} = h \times 4 \times 2 \times 4 \text{ bytes}$$
> 
> Where:
> - $h = 32$ (hidden size)
> - $4$ represents the 4 LSTM gates (forget, input, output, candidate)
> - $2$ for both cell state ($C_t$) and hidden state ($h_t$)
> - $4$ bytes per float32 value
> 
> $$\text{LSTM}_{\text{states}} = 32 \times 4 \times 2 \times 4 = 1024 \text{ bytes}$$

> [!note]- Output Layer Memory
> 
> Memory for MLP output layers:
> 
> $$\text{Output}_{\text{buffers}} = \sum_{i=0}^{L-1} s_i \times 4 \text{ bytes}$$
> 
> Where $s_i$ is the size of layer $i$:
> - Layer 0: $s_0 = 32$
> - Layer 1: $s_1 = 32$ 
> - Layer 2: $s_2 = 1$
> 
> $$\text{Output}_{\text{buffers}} = (32 + 32 + 1) \times 4 = 260 \text{ bytes}$$

### Neural Network Runtime Memory

> [!note]- Input/Output Buffers
> 
> Buffers for data input and output:
> 
> $$\text{Buffers} = \text{buffer\_size} \times 2 = 256 \times 2 = 512 \text{ bytes}$$

> [!note]- LSTM Temporary Computation
> 
> Temporary memory for LSTM gate computations:
> 
> $$\text{LSTM}_{\text{temps}} = h \times 6 \times 2 \times 4 \text{ bytes}$$
> 
> Where:
> - $h = 32$ (hidden size)
> - $6$ temporary variables per computation cycle
> - $2$ for intermediate calculations
> - $4$ bytes per float32
> 
> $$\text{LSTM}_{\text{temps}} = 32 \times 6 \times 2 \times 4 = 1536 \text{ bytes}$$

> [!note]- Neural Network Overhead
> 
> Additional memory for activation functions and neural network operations:
> 
> $$\text{NN}_{\text{overhead}} = 512 \text{ bytes (constant)}$$

### System Overhead Memory

> [!note]- Arduino System Components
> 
> The system overhead includes various Arduino framework components:
> 
> $$\text{System}_{\text{overhead}} = \sum \text{Arduino Components}$$
> 
> | Component | Formula | Value |
> |-----------|---------|-------|
> | Arduino Core | $\text{constant}$ | 1536 bytes |
> | Serial Buffers | $\text{constant}$ | 768 bytes |
> | Stack Space | $\text{constant}$ | 2560 bytes |
> | Interrupt Vectors | $\text{constant}$ | 512 bytes |
> | Memory Alignment | $\text{constant}$ | 256 bytes |
> | Heap Reserve | $\text{constant}$ | 512 bytes |
> 
> $$\text{System}_{\text{overhead}} = 1536 + 768 + 2560 + 512 + 256 + 512 = 6144 \text{ bytes}$$

### Complete SRAM Calculation

> [!example]- Final SRAM Formula
> 
> Combining all components:
> 
> $$\begin{align}
> \text{SRAM}_{\text{total}} &= \text{LSTM}_{\text{states}} + \text{Output}_{\text{buffers}} + \text{Buffers} \\
> &\quad + \text{LSTM}_{\text{temps}} + \text{NN}_{\text{overhead}} + \text{System}_{\text{overhead}}
> \end{align}$$
> 
> $$\begin{align}
> \text{SRAM}_{\text{total}} &= 1024 + 260 + 512 + 1536 + 512 + 6144 \\
> &= 9988 \text{ bytes} \approx 9.76 \text{ KB}
> \end{align}$$

---

## Flash Memory Calculations

> [!note]- Flash Formula Overview
> 
> The total Flash usage is calculated as:
> 
> $$\text{Flash}_{\text{total}} = \text{LSTM}_{\text{weights}} + \text{Output}_{\text{weights}} + \text{Application}_{\text{code}}$$

### LSTM Weight Storage

> [!note]- LSTM Weight Components
> 
> LSTM has 4 gates, each requiring input weights, hidden weights, and biases:
> 
> **Input-to-Hidden Weights:**
> $$W_{ih} = 4 \times n \times h \times 4 \text{ bytes}$$
> $$W_{ih} = 4 \times 4 \times 32 \times 4 = 2048 \text{ bytes}$$
> 
> **Hidden-to-Hidden Weights:**
> $$W_{hh} = 4 \times h \times h \times 4 \text{ bytes}$$
> $$W_{hh} = 4 \times 32 \times 32 \times 4 = 16384 \text{ bytes}$$
> 
> **Biases:**
> $$b = 4 \times h \times 4 \text{ bytes}$$
> $$b = 4 \times 32 \times 4 = 512 \text{ bytes}$$
> 
> **Total LSTM Weights:**
> $$\text{LSTM}_{\text{weights}} = W_{ih} + W_{hh} + b = 2048 + 16384 + 512 = 18944 \text{ bytes}$$

### Output Layer Weight Storage

> [!note]- MLP Weight Calculation
> 
> For each layer transition in the MLP $[32, 32, 1]$:
> 
> **Layer 0 → Layer 1:** $(32 \to 32)$
> $$W_0 = 32 \times 32 \times 4 = 4096 \text{ bytes (weights)}$$
> $$b_0 = 32 \times 4 = 128 \text{ bytes (biases)}$$
> 
> **Layer 1 → Layer 2:** $(32 \to 1)$
> $$W_1 = 32 \times 1 \times 4 = 128 \text{ bytes (weights)}$$
> $$b_1 = 1 \times 4 = 4 \text{ bytes (biases)}$$
> 
> **Total Output Weights:**
> $$\text{Output}_{\text{weights}} = (4096 + 128) + (128 + 4) = 4356 \text{ bytes}$$

### Application Code Storage

> [!note]- Code Base Components
> 
> $$\text{Application}_{\text{code}} = \text{Code}_{\text{base}} + \text{Constants}$$
> 
> Where:
> - $\text{Code}_{\text{base}} = 92160 \text{ bytes (90 KB)}$ - Base application code
> - $\text{Constants} = 2048 \text{ bytes (2 KB)}$ - Lookup tables, strings
> 
> $$\text{Application}_{\text{code}} = 92160 + 2048 = 94208 \text{ bytes}$$

### Complete Flash Calculation

> [!example]- Final Flash Formula
> 
> Combining all components:
> 
> $$\begin{align}
> \text{Flash}_{\text{total}} &= \text{LSTM}_{\text{weights}} + \text{Output}_{\text{weights}} + \text{Application}_{\text{code}} \\
> &= 18944 + 4356 + 94208 \\
> &= 117508 \text{ bytes} \approx 114.8 \text{ KB}
> \end{align}$$

---

## Accuracy Validation

> [!success]- Prediction Accuracy
> 
> **SRAM Accuracy:**
> $$\text{Error}_{\text{SRAM}} = \frac{|\text{SRAM}_{\text{predicted}} - \text{SRAM}_{\text{real}}|}{\text{SRAM}_{\text{real}}} \times 100\%$$
> 
> $$\text{Error}_{\text{SRAM}} = \frac{|9988 - 8984|}{8984} \times 100\% = 11.2\%$$
> 
> **Flash Accuracy:**
> $$\text{Error}_{\text{Flash}} = \frac{|\text{Flash}_{\text{predicted}} - \text{Flash}_{\text{real}}|}{\text{Flash}_{\text{real}}} \times 100\%$$
> 
> $$\text{Error}_{\text{Flash}} = \frac{|117508 - 122880|}{122880} \times 100\% = 4.4\%$$

---

## Memory Utilization on Arduino

> [!warning]- Arduino Uno R4 WiFi Limits
> 
> **SRAM Utilization:**
> $$\text{SRAM}_{\%} = \frac{\text{SRAM}_{\text{used}}}{\text{SRAM}_{\text{total}}} \times 100\%$$
> $$\text{SRAM}_{\%} = \frac{8984}{32768} \times 100\% = 27.4\%$$
> 
> **Flash Utilization:**
> $$\text{Flash}_{\%} = \frac{\text{Flash}_{\text{used}}}{\text{Flash}_{\text{total}}} \times 100\%$$
> $$\text{Flash}_{\%} = \frac{122880}{262144} \times 100\% = 46.9\%$$

---

## Summary

> [!abstract] Key Formulas
> 
> | Memory Type | Formula | Result |
> |-------------|---------|--------|
> | **SRAM Total** | $1024 + 260 + 512 + 1536 + 512 + 6144$ | 9,988 bytes |
> | **Flash Total** | $18944 + 4356 + 94208$ | 117,508 bytes |
> | **SRAM Error** | $\frac{|9988 - 8984|}{8984} \times 100\%$ | 11.2% |
> | **Flash Error** | $\frac{|117508 - 122880|}{122880} \times 100\%$ | 4.4% |

These formulas provide accurate memory predictions for LSTM neural networks on Arduino hardware with excellent precision for both SRAM and Flash memory usage.
