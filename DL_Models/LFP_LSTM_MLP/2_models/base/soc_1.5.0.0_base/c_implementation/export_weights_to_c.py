#!/usr/bin/env python
"""
Export PyTorch model weights to C arrays.
This creates a .h file with all weights that can be used in C implementation.
"""
import os
import argparse
import torch
import torch.nn as nn
import numpy as np
import yaml


def format_array_c(arr, name, dtype='float'):
    """Format numpy array as C array declaration"""
    flat = arr.flatten()
    
    # Header
    lines = [f"const {dtype} {name}[{len(flat)}] = {{"]
    
    # Data (8 values per line)
    for i in range(0, len(flat), 8):
        chunk = flat[i:i+8]
        values = ', '.join([f'{v:.8f}f' for v in chunk])
        lines.append(f"    {values},")
    
    # Remove trailing comma and close
    lines[-1] = lines[-1].rstrip(',')
    lines.append("};")
    
    return '\n'.join(lines)


def export_lstm_weights(lstm, layer_idx=0):
    """Export LSTM weights for a specific layer"""
    weights = {}
    
    # PyTorch LSTM stores weights as: weight_ih_l[k], weight_hh_l[k], bias_ih_l[k], bias_hh_l[k]
    # Format: [4*hidden_size, input_size] for weight_ih
    #         [4*hidden_size, hidden_size] for weight_hh
    # Order: input gate, forget gate, cell gate, output gate
    
    weight_ih = getattr(lstm, f'weight_ih_l{layer_idx}').detach().cpu().numpy()  # [4H, I]
    weight_hh = getattr(lstm, f'weight_hh_l{layer_idx}').detach().cpu().numpy()  # [4H, H]
    bias_ih = getattr(lstm, f'bias_ih_l{layer_idx}').detach().cpu().numpy()      # [4H]
    bias_hh = getattr(lstm, f'bias_hh_l{layer_idx}').detach().cpu().numpy()      # [4H]
    
    # Combine biases
    bias = bias_ih + bias_hh  # [4H]
    
    weights['weight_ih'] = weight_ih
    weights['weight_hh'] = weight_hh
    weights['bias'] = bias
    
    return weights


def export_linear_weights(linear):
    """Export Linear layer weights"""
    weight = linear.weight.detach().cpu().numpy()  # [out, in]
    bias = linear.bias.detach().cpu().numpy()      # [out]
    
    return {'weight': weight, 'bias': bias}


def main():
    parser = argparse.ArgumentParser(description='Export model weights to C arrays')
    parser.add_argument('--checkpoint', type=str, default='../1.5.0.0_soc_epoch0001_rmse0.02897.pt')
    parser.add_argument('--config', type=str, default='../../../1_training/1.5.0.0/config/train_soc.yaml')
    parser.add_argument('--output', type=str, default='model_weights.h')
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.abspath(os.path.join(script_dir, args.checkpoint))
    config_path = os.path.abspath(os.path.join(script_dir, args.config))
    output_path = os.path.join(script_dir, args.output)
    
    print("="*80)
    print("Export PyTorch Weights to C Arrays")
    print("="*80)
    print(f"\n📁 Checkpoint: {checkpoint_path}")
    print(f"📁 Config:     {config_path}")
    print(f"📁 Output:     {output_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    features = cfg['model']['features']
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 1))
    
    print(f"\n📥 Model configuration:")
    print(f"   Input features: {len(features)}")
    print(f"   Hidden size: {hidden_size}")
    print(f"   MLP hidden: {mlp_hidden}")
    print(f"   Num layers: {num_layers}")
    
    # Define model class (same as in training)
    class LSTMMLP(nn.Module):
        def __init__(self, in_features, hidden_size, mlp_hidden, num_layers=1, dropout=0.05, sigmoid_head=True):
            super().__init__()
            self.lstm = nn.LSTM(in_features, hidden_size, num_layers=num_layers, batch_first=True)
            head = [nn.Linear(hidden_size, mlp_hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mlp_hidden, 1)]
            if sigmoid_head:
                head.append(nn.Sigmoid())
            self.mlp = nn.Sequential(*head)
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    model = LSTMMLP(
        in_features=len(features),
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        dropout=0.0,
        sigmoid_head=True
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\n✓ Model loaded")
    
    # Extract weights
    print(f"\n📤 Extracting weights...")
    
    # LSTM weights
    lstm_weights = export_lstm_weights(model.lstm, layer_idx=0)
    
    # MLP weights
    mlp_fc1 = export_linear_weights(model.mlp[0])  # First linear layer
    mlp_fc2 = export_linear_weights(model.mlp[3])  # Second linear layer (after ReLU and Dropout)
    
    print(f"   ✓ Weights extracted")
    
    # Generate C header file
    print(f"\n🔧 Generating C header file...")
    
    header_lines = [
        "/*",
        " * Auto-generated model weights for LSTM-MLP SOC prediction",
        f" * Model: {os.path.basename(checkpoint_path)}",
        " * DO NOT EDIT MANUALLY",
        " */",
        "",
        "#ifndef MODEL_WEIGHTS_H",
        "#define MODEL_WEIGHTS_H",
        "",
        "/* Model configuration */",
        f"#define INPUT_SIZE {len(features)}",
        f"#define HIDDEN_SIZE {hidden_size}",
        f"#define MLP_HIDDEN {mlp_hidden}",
        f"#define NUM_LAYERS {num_layers}",
        "",
        "/* Feature names (for reference) */",
        "const char* FEATURE_NAMES[] = {",
    ]
    
    for feat in features:
        header_lines.append(f'    "{feat}",')
    header_lines.append("};")
    header_lines.append("")
    
    # LSTM weights
    header_lines.append("/* ========== LSTM Layer 0 Weights ========== */")
    header_lines.append("")
    header_lines.append(f"/* LSTM input weights: [4*HIDDEN_SIZE={4*hidden_size}, INPUT_SIZE={len(features)}] */")
    header_lines.append(format_array_c(lstm_weights['weight_ih'], 'LSTM_WEIGHT_IH'))
    header_lines.append("")
    
    header_lines.append(f"/* LSTM hidden weights: [4*HIDDEN_SIZE={4*hidden_size}, HIDDEN_SIZE={hidden_size}] */")
    header_lines.append(format_array_c(lstm_weights['weight_hh'], 'LSTM_WEIGHT_HH'))
    header_lines.append("")
    
    header_lines.append(f"/* LSTM bias: [4*HIDDEN_SIZE={4*hidden_size}] */")
    header_lines.append(format_array_c(lstm_weights['bias'], 'LSTM_BIAS'))
    header_lines.append("")
    
    # MLP weights
    header_lines.append("/* ========== MLP Weights ========== */")
    header_lines.append("")
    header_lines.append(f"/* MLP FC1 weights: [{mlp_hidden}, {hidden_size}] */")
    header_lines.append(format_array_c(mlp_fc1['weight'], 'MLP_FC1_WEIGHT'))
    header_lines.append("")
    
    header_lines.append(f"/* MLP FC1 bias: [{mlp_hidden}] */")
    header_lines.append(format_array_c(mlp_fc1['bias'], 'MLP_FC1_BIAS'))
    header_lines.append("")
    
    header_lines.append(f"/* MLP FC2 weights: [1, {mlp_hidden}] */")
    header_lines.append(format_array_c(mlp_fc2['weight'], 'MLP_FC2_WEIGHT'))
    header_lines.append("")
    
    header_lines.append("/* MLP FC2 bias: [1] */")
    header_lines.append(format_array_c(mlp_fc2['bias'], 'MLP_FC2_BIAS'))
    header_lines.append("")
    
    header_lines.append("#endif /* MODEL_WEIGHTS_H */")
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(header_lines))
    
    file_size = os.path.getsize(output_path) / 1024
    
    print(f"\n{'='*80}")
    print("EXPORT COMPLETED")
    print(f"{'='*80}")
    print(f"\n✅ C header file created: {output_path}")
    print(f"   Size: {file_size:.2f} KB")
    print(f"\n📊 Weight statistics:")
    print(f"   LSTM weight_ih: {lstm_weights['weight_ih'].shape} = {lstm_weights['weight_ih'].size} values")
    print(f"   LSTM weight_hh: {lstm_weights['weight_hh'].shape} = {lstm_weights['weight_hh'].size} values")
    print(f"   LSTM bias:      {lstm_weights['bias'].shape} = {lstm_weights['bias'].size} values")
    print(f"   MLP FC1 weight: {mlp_fc1['weight'].shape} = {mlp_fc1['weight'].size} values")
    print(f"   MLP FC1 bias:   {mlp_fc1['bias'].shape} = {mlp_fc1['bias'].size} values")
    print(f"   MLP FC2 weight: {mlp_fc2['weight'].shape} = {mlp_fc2['weight'].size} values")
    print(f"   MLP FC2 bias:   {mlp_fc2['bias'].shape} = {mlp_fc2['bias'].size} values")
    
    total_params = (lstm_weights['weight_ih'].size + lstm_weights['weight_hh'].size + 
                    lstm_weights['bias'].size + mlp_fc1['weight'].size + mlp_fc1['bias'].size +
                    mlp_fc2['weight'].size + mlp_fc2['bias'].size)
    print(f"\n   Total parameters: {total_params}")
    print(f"   Memory (float32): {total_params * 4 / 1024:.2f} KB")


if __name__ == '__main__':
    main()
