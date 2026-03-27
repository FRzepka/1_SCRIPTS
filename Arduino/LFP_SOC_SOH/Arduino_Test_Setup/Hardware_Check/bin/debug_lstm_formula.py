#!/usr/bin/env python3
"""
Debug LSTM Parameter Formula - Find the EXACT correct formula
"""

import torch
import torch.nn as nn

def test_lstm_formulas():
    """Test different LSTM parameter formulas against PyTorch"""
    
    print("=== LSTM PARAMETER FORMULA DEBUG ===")
    print()
    
    test_cases = [
        (4, 8),   # 4→8  
        (4, 16),  # 4→16
        (4, 32),  # 4→32
        (4, 64),  # 4→64
    ]
    
    for input_size, hidden_size in test_cases:
        print(f"Testing {input_size}→{hidden_size}:")
        
        # Create actual LSTM
        lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0)
        
        # Count actual parameters
        actual_params = sum(p.numel() for p in lstm.parameters())
        
        print(f"  PyTorch actual: {actual_params:,} parameters")
        
        # Manual breakdown
        for name, param in lstm.named_parameters():
            print(f"    {name}: {param.shape} = {param.numel():,}")
        
        # Test different formulas
        formulas = {
            "Old wrong": 4 * ((input_size + hidden_size) * hidden_size + hidden_size),
            "My attempt": 4 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size),
            "Manual calc": (
                4 * input_size * hidden_size +    # weight_ih
                4 * hidden_size * hidden_size +   # weight_hh  
                4 * hidden_size +                 # bias_ih
                4 * hidden_size                   # bias_hh
            )
        }
        
        for name, calc in formulas.items():
            match = "✓" if calc == actual_params else "✗"
            print(f"  {name:12s}: {calc:,} {match}")
        
        print()

def analyze_lstm_structure():
    """Analyze the exact structure of PyTorch LSTM"""
    
    print("=== PYTORCH LSTM STRUCTURE ANALYSIS ===")
    
    input_size, hidden_size = 4, 32
    lstm = nn.LSTM(input_size, hidden_size, num_layers=1, batch_first=True, dropout=0.0)
    
    print(f"LSTM({input_size}, {hidden_size}, num_layers=1)")
    print()
    
    for name, param in lstm.named_parameters():
        shape = param.shape
        size = param.numel()
        
        print(f"{name:20s}: {shape} = {size:,} parameters")
        
        if "weight_ih" in name:
            print(f"  → Input-to-Hidden weights: 4 gates × {hidden_size} × {input_size} = {4 * hidden_size * input_size}")
            print(f"  → Shape interpretation: [4*hidden_size, input_size] = [{4*hidden_size}, {input_size}]")
            
        elif "weight_hh" in name:
            print(f"  → Hidden-to-Hidden weights: 4 gates × {hidden_size} × {hidden_size} = {4 * hidden_size * hidden_size}")
            print(f"  → Shape interpretation: [4*hidden_size, hidden_size] = [{4*hidden_size}, {hidden_size}]")
            
        elif "bias_ih" in name:
            print(f"  → Input-to-Hidden bias: 4 gates × {hidden_size} = {4 * hidden_size}")
            print(f"  → Shape interpretation: [4*hidden_size] = [{4*hidden_size}]")
            
        elif "bias_hh" in name:
            print(f"  → Hidden-to-Hidden bias: 4 gates × {hidden_size} = {4 * hidden_size}")
            print(f"  → Shape interpretation: [4*hidden_size] = [{4*hidden_size}]")
        
        print()
    
    total = sum(p.numel() for p in lstm.parameters())
    manual = 4*input_size*hidden_size + 4*hidden_size*hidden_size + 4*hidden_size + 4*hidden_size
    
    print(f"Total PyTorch:    {total:,}")
    print(f"Manual breakdown: {manual:,}")
    print(f"Match: {'✓' if total == manual else '✗'}")
    
    print()
    print("CORRECT FORMULA:")
    print("LSTM_params = 4×input_size×hidden_size + 4×hidden_size×hidden_size + 2×4×hidden_size")
    print("            = 4×hidden_size×(input_size + hidden_size + 2)")
    print(f"            = 4×{hidden_size}×({input_size} + {hidden_size} + 2)")
    print(f"            = 4×{hidden_size}×{input_size + hidden_size + 2}")
    print(f"            = {4 * hidden_size * (input_size + hidden_size + 2):,}")

if __name__ == "__main__":
    test_lstm_formulas()
    print()
    analyze_lstm_structure()
