#!/usr/bin/env python3
"""
Debug Script: Find exact parameter counts from real model
"""

import torch
import torch.nn as nn
import os

# Model definition (exact copy from training script)
HIDDEN_SIZE = 32
NUM_LAYERS = 1
MLP_HIDDEN = 32

class SOCModel(nn.Module):
    """
    LSTM SOC Model - identisch zum Training Script
    Architecture: LSTM(4→32) + MLP(32→32→32→1)
    """
    def __init__(self, input_size=4, dropout=0.05):
        super().__init__()
        self.lstm = nn.LSTM(input_size, HIDDEN_SIZE, NUM_LAYERS,
                            batch_first=True, dropout=0.0)
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(MLP_HIDDEN, 1),
            nn.Sigmoid()
        )

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.contiguous()
        h, c = hidden
        h, c = h.contiguous(), c.contiguous()
        hidden = (h, c)
        out, hidden = self.lstm(x, hidden)
        batch, seq_len, hid = out.size()
        out_flat = out.contiguous().view(batch * seq_len, hid)
        soc_flat = self.mlp(out_flat)
        soc = soc_flat.view(batch, seq_len)
        return soc, hidden

def analyze_model_parameters():
    """Analyze exact parameter counts"""
    
    # Create model
    model = SOCModel(input_size=4, dropout=0.05)
    
    print("=== EXACT MODEL PARAMETER ANALYSIS ===")
    print()
    
    total_params = 0
    lstm_params = 0
    mlp_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        
        print(f"{name:25s}: {param.shape} = {param_count:,} parameters")
        
        if 'lstm' in name:
            lstm_params += param_count
        elif 'mlp' in name:
            mlp_params += param_count
    
    print()
    print(f"📊 PARAMETER BREAKDOWN:")
    print(f"   LSTM Parameters:     {lstm_params:,}")
    print(f"   MLP Parameters:      {mlp_params:,}")
    print(f"   Total Parameters:    {total_params:,}")
    print()
    
    # Manual LSTM calculation
    input_size = 4
    hidden_size = 32
    
    # LSTM formula: 4 × [(input_size + hidden_size) × hidden_size + hidden_size]
    manual_lstm = 4 * ((input_size + hidden_size) * hidden_size + hidden_size)
    
    # MLP calculation
    mlp_manual = (32 * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
    
    print(f"📐 MANUAL CALCULATIONS:")
    print(f"   LSTM (formula):      {manual_lstm:,}")
    print(f"   MLP (manual):        {mlp_manual:,}")
    print(f"   Total (manual):      {manual_lstm + mlp_manual:,}")
    print()
    
    # Compare with PyTorch
    print(f"✅ VERIFICATION:")
    print(f"   PyTorch LSTM:        {lstm_params:,} {'✓' if lstm_params == manual_lstm else '✗'}")
    print(f"   PyTorch MLP:         {mlp_params:,} {'✓' if mlp_params == mlp_manual else '✗'}")
    print(f"   PyTorch Total:       {total_params:,} {'✓' if total_params == manual_lstm + mlp_manual else '✗'}")
    
    return total_params, lstm_params, mlp_params

def test_different_sizes():
    """Test parameter calculations for different hidden sizes"""
    
    print("\n=== PARAMETER SCALING ANALYSIS ===")
    
    sizes = [8, 16, 32, 64]
    
    for hidden_size in sizes:
        # Manual calculation
        input_size = 4
        lstm_params = 4 * ((input_size + hidden_size) * hidden_size + hidden_size)
        mlp_params = (hidden_size * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
        total_params = lstm_params + mlp_params
        
        print(f"Hidden Size {hidden_size:2d}: LSTM={lstm_params:,}, MLP={mlp_params:,}, Total={total_params:,}")

if __name__ == "__main__":
    analyze_model_parameters()
    test_different_sizes()
