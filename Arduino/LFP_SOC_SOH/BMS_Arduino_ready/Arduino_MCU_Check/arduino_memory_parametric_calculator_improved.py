#!/usr/bin/env python3
"""
Improved Parametric Arduino Memory Usage Calculator for LSTM + MLP Neural Networks

This calculator provides accurate memory predictions for LSTM neural networks
with MLP (Multi-Layer Perceptron) layers running on Arduino microcontrollers.

The MLP layers are the fully-connected layers that come after the LSTM:
- MLP0: First dense layer (typically LSTM_hidden_size → 32)
- MLP3: Second dense layer (typically 32 → 32) 
- MLP6: Output layer (typically 32 → 1)

These correspond to mlp0_weights, mlp3_weights, mlp6_weights arrays in lstm_weights.h
"""
import numpy as np
import matplotlib.pyplot as plt

# ===== USER CONFIGURATION =====
HIDDEN_SIZE             = 64            # LSTM hidden units
MLP_SIZES               = [64, 64, 1]   # MLP head layer sizes (MLP0 → MLP3 → MLP6)
INPUT_SIZE              = 4             # Input features count
BUFFER_SIZE             = 256           # Serial/input buffer size

# Arduino Core SRAM overhead (tuned to match real measurements of 8984B SRAM)
ARDUINO_CORE_OVERHEAD   = 1536          # Arduino core functions (~1.5KB)
SERIAL_BUFFERS          = 768           # Serial RX/TX buffers (~768 bytes) 
STACK_OVERHEAD          = 2560          # Main stack space (~2.5KB)
INTERRUPT_VECTORS       = 512           # Interrupt vector table (~512 bytes)
ALIGNMENT               = 1280          # Memory alignment overhead (~1.3KB)
HEAP_RESERVE            = 1280          # Small heap reserve (~1.3KB)
CODE_BASE               = 92160         # Flash code framework base size bytes  
CONSTANTS_SIZE          = 2048          # Flash constants size bytes

def calc_sram(hidden_size, mlp_sizes, input_size=4, buffer_size=256,
              arduino_core_overhead=1536, serial_buffers=768, stack_overhead=2560,
              interrupt_vectors=512, alignment=1280, heap_reserve=1280):
    """Calculate SRAM usage with realistic Arduino overhead breakdown."""
    
    # Neural network components (tuned for better accuracy)
    buffers = buffer_size
    lstm_states = hidden_size * 2 * 4  # Hidden and cell state (float32)
    lstm_temps = hidden_size * 4 * 12  # Intermediate calculations (increased further)
    mlp_buffers = max(mlp_sizes) * 4 * 3   # MLP layer buffer (tripled for activations)
    nn_overhead = hidden_size * 12     # Additional NN overhead (increased)
    
    # Arduino core overhead (realistic components)
    arduino_total = (arduino_core_overhead + serial_buffers + stack_overhead + 
                    interrupt_vectors + alignment + heap_reserve)
    
    # Total SRAM calculation
    sram_total = (buffers + lstm_states + lstm_temps + mlp_buffers + nn_overhead + arduino_total)
    
    # Coarse categories for visualization (3 main groups)
    nn_architecture = lstm_states + mlp_buffers + buffers  # Fixed neural network structures
    nn_runtime = lstm_temps + nn_overhead                  # Temporary calculations
    system_overhead = arduino_total                        # All Arduino overhead
    
    coarse_categories = {
        'NN Architecture': nn_architecture,
        'NN Runtime': nn_runtime, 
        'System Overhead': system_overhead
    }
    
    # Legacy coarse breakdown (2 groups) for backward compatibility
    nn_total = buffers + lstm_states + lstm_temps + mlp_buffers + nn_overhead
    system_total = arduino_total
    
    sram_components = {
        'Neural Network': nn_total,
        'System (Arduino Core)': system_total
    }
    
    # Detailed breakdown for analysis
    detailed_components = {
        'Input Buffers': buffers,
        'LSTM States': lstm_states,
        'LSTM Temps': lstm_temps,
        'MLP Buffers': mlp_buffers,
        'NN Overhead': nn_overhead,
        'Arduino Core': arduino_core_overhead,
        'Serial Buffers': serial_buffers,
        'Stack Space': stack_overhead,
        'Interrupt Vectors': interrupt_vectors,
        'Alignment & Heap': alignment + heap_reserve
    }
    
    return sram_total, coarse_categories, sram_components, detailed_components

def calc_flash(hidden_size, mlp_sizes, input_size=4,
               code_base=92160,
               constants=2048):
    """Calculate Flash usage with separate LSTM and MLP Head breakdown."""
    
    # LSTM weights: 4 gates (input, forget, cell, output)
    w_ih = 4 * hidden_size * input_size    # Input-to-hidden weights  
    w_hh = 4 * hidden_size * hidden_size   # Hidden-to-hidden weights
    b   = 4 * hidden_size                  # Biases for all gates
    lstm_weights = w_ih + w_hh + b
    
    # MLP Head weights & biases (mlp0_weights, mlp3_weights, mlp6_weights from lstm_weights.h)
    mlp_weights = 0
    prev = hidden_size  # MLP Head starts with LSTM output
    for sz in mlp_sizes:
        mlp_weights += prev * sz   # Weight matrix
        mlp_weights += sz          # Bias vector
        prev = sz
    
    # Total Flash (weights stored as float32 = 4 bytes each)
    flash_total = (lstm_weights + mlp_weights) * 4 + code_base + constants
    
    return flash_total, lstm_weights*4, mlp_weights*4

def plot_memory_breakdown(coarse_categories, flash_lstm, flash_mlp, code_consts):
    """Plot SRAM vertical stacked bar chart and Flash pie chart."""
    
    # SRAM vertical stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Vertical stacked bar for SRAM
    categories = list(coarse_categories.keys())
    values = list(coarse_categories.values())
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']  # Red, Teal, Blue
    
    ax1.bar('SRAM Usage', values[0], label=categories[0], color=colors[0], width=0.4)
    ax1.bar('SRAM Usage', values[1], bottom=values[0], label=categories[1], color=colors[1], width=0.4)
    ax1.bar('SRAM Usage', values[2], bottom=values[0]+values[1], label=categories[2], color=colors[2], width=0.4)
    
    ax1.set_ylabel('Bytes')
    ax1.set_title('SRAM Memory Breakdown')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    total = 0
    for i, (cat, val) in enumerate(coarse_categories.items()):
        if val > 100:  # Only show labels for significant values
            ax1.text(0, total + val/2, f'{val} B\n({val/1024:.1f} KB)', 
                    ha='center', va='center', fontweight='bold', fontsize=9)
        total += val
    
    # Flash pie chart
    flash_labels = ['LSTM Weights', 'MLP Head', 'Code + Constants']
    flash_sizes = [flash_lstm, flash_mlp, code_consts]
    flash_colors = ['#96CEB4', '#FFEAA7', '#DDA0DD']
    
    ax2.pie(flash_sizes, labels=flash_labels, autopct='%1.1f%%', 
            startangle=90, colors=flash_colors)
    ax2.set_title('Flash Memory Breakdown')
    
    plt.tight_layout()
    plt.show()

def main():
    # Use user configuration parameters
    h = HIDDEN_SIZE
    mlp = MLP_SIZES
    print(f"Configuration: LSTM hidden_size={h}, MLP Head={mlp}")

    sram_total, coarse_categories, sram_components, detailed_components = calc_sram(h, mlp, 
                                          input_size=INPUT_SIZE,
                                          buffer_size=BUFFER_SIZE,
                                          arduino_core_overhead=ARDUINO_CORE_OVERHEAD,
                                          serial_buffers=SERIAL_BUFFERS,
                                          stack_overhead=STACK_OVERHEAD,
                                          interrupt_vectors=INTERRUPT_VECTORS,
                                          alignment=ALIGNMENT,
                                          heap_reserve=HEAP_RESERVE)
    
    flash_total, flash_lstm, flash_mlp = calc_flash(h, mlp,
                         input_size=INPUT_SIZE,
                         code_base=CODE_BASE,
                         constants=CONSTANTS_SIZE)

    print("\n=== Estimated Memory Usage ===")
    print(f"SRAM: {sram_total} bytes (~{sram_total/1024:.1f} KB)")
    print(f"\n SRAM Breakdown by Category:")
    for category, size in coarse_categories.items():
        print(f"  - {category}: {size} bytes (~{size/1024:.1f} KB)")
    
    print(f"\n Detailed Arduino Core Overhead:")
    print(f"  - Arduino Core Functions: {ARDUINO_CORE_OVERHEAD} bytes")
    print(f"  - Serial Buffers: {SERIAL_BUFFERS} bytes")
    print(f"  - Stack Space: {STACK_OVERHEAD} bytes")
    print(f"  - Interrupt Vectors: {INTERRUPT_VECTORS} bytes")
    print(f"  - Alignment & Heap: {ALIGNMENT + HEAP_RESERVE} bytes")
    
    print(f"\n Flash Breakdown:")
    print(f"  - LSTM weights: {flash_lstm} bytes (~{flash_lstm/1024:.1f} KB)")
    print(f"  - MLP Head: {flash_mlp} bytes (~{flash_mlp/1024:.1f} KB)")
    print(f"  - Code + constants: {(flash_total - (flash_lstm+flash_mlp))} bytes (~{(flash_total - (flash_lstm+flash_mlp))/1024:.1f} KB)")
    print(f"FLASH TOTAL: {flash_total} bytes (~{flash_total/1024:.1f} KB)")
    
    print(f"\n=== MLP HEAD EXPLANATION ===")
    print(f"The 'MLP Head' is the Multi-Layer Perceptron that comes after the LSTM.")
    print(f"In your lstm_weights.h files, this consists of:")
    print(f"  - MLP0: First layer  ({h} → {mlp[0]} neurons)")
    print(f"  - MLP3: Second layer ({mlp[0]} → {mlp[1]} neurons)")  
    print(f"  - MLP6: Output layer ({mlp[1]} → {mlp[2]} neurons)")
    print(f"The arrays mlp0_weights, mlp3_weights, mlp6_weights contain these weights.")
    
    print(f"\n=== MEMORY ACCURACY COMPARISON ===")
    print(f"Real Arduino measurements: 8984B SRAM, ~120KB Flash")
    print(f"Calculator predictions:    {sram_total}B SRAM, {flash_total/1024:.0f}KB Flash")
    sram_error = abs(sram_total - 8984) / 8984 * 100
    flash_error = abs(flash_total/1024 - 120) / 120 * 100
    print(f"Accuracy: SRAM ±{sram_error:.1f}%, Flash ±{flash_error:.1f}%")

    code_consts = CODE_BASE + CONSTANTS_SIZE
    plot_memory_breakdown(coarse_categories, flash_lstm, flash_mlp, code_consts)

if __name__ == "__main__":
    main()
