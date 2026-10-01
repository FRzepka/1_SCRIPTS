import os
import sys
import argparse
from pathlib import Path
import yaml
import torch
import torch.nn as nn
from typing import List, Tuple, Optional


# ========================
# Model Definition (from train_soh.py)
# ========================

class ResidualMLPBlock(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        out = self.fc2(self.act(self.fc1(x)))
        out = self.drop(out)
        return self.norm(x + out)


class SOH_LSTM_Seq2Seq(nn.Module):
    """Stateful-ready LSTM that outputs SOH at every timestep."""
    def __init__(
        self,
        in_features: int,
        embed_size: int,
        hidden_size: int,
        mlp_hidden: int,
        num_layers: int = 2,
        res_blocks: int = 2,
        bidirectional: bool = False,
        dropout: float = 0.15,
    ):
        super().__init__()
        if bidirectional:
            print('Warning: bidirectional=True breaks true stateful inference.')
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_directions = 2 if bidirectional else 1

        self.feature_proj = nn.Sequential(
            nn.Linear(in_features, embed_size),
            nn.LayerNorm(embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(embed_size, embed_size),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out = hidden_size * self.num_directions
        self.post_norm = nn.LayerNorm(lstm_out)
        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(lstm_out, mlp_hidden, dropout) for _ in range(max(0, int(res_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(lstm_out, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, x, state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None, return_state: bool = False):
        x = self.feature_proj(x)
        out, new_state = self.lstm(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y_seq = self.head(out).squeeze(-1)
        if return_state:
            return y_seq, new_state
        return y_seq


# ========================
# Helper Functions
# ========================

def expand_env_with_defaults(path_str: str, defaults: dict = None) -> str:
    """Expand environment variables with defaults."""
    if defaults is None:
        defaults = {}

    # Simple regex to find ${VAR:-default} patterns
    import re
    pattern = r'\$\{([^:}]+)(?::-)([^}]+)\}'

    def replacer(match):
        var_name = match.group(1)
        default_val = match.group(2)
        return os.environ.get(var_name, default_val)

    return re.sub(pattern, replacer, path_str)


def expand_features_for_sampling(base_features: List[str], sampling_cfg: dict) -> List[str]:
    """Expand base features with aggregation suffixes if sampling is enabled."""
    if not sampling_cfg.get('enabled', False):
        return base_features

    feature_aggs = sampling_cfg.get('feature_aggs', ['mean', 'std', 'min', 'max'])
    expanded = []
    for feat in base_features:
        for agg in feature_aggs:
            expanded.append(f"{feat}_{agg}")
    return expanded


def load_model_from_checkpoint(
    checkpoint_path: str,
    config_path: str,
    device: str = 'cpu'
) -> Tuple[SOH_LSTM_Seq2Seq, dict, List[str]]:
    """
    Load model from checkpoint and config file.

    Returns:
        model: The loaded model in eval mode
        cfg: Configuration dictionary
        features: List of feature names
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # Get base features and expand if sampling is enabled
    base_features = cfg['model']['features']
    sampling_cfg = cfg.get('sampling', {})
    features = expand_features_for_sampling(base_features, sampling_cfg)

    # Extract model hyperparameters from config
    embed_size = int(cfg['model'].get('embed_size', 128))
    hidden_size = int(cfg['model']['hidden_size'])
    mlp_hidden = int(cfg['model']['mlp_hidden'])
    num_layers = int(cfg['model'].get('num_layers', 3))
    res_blocks = int(cfg['model'].get('res_blocks', 2))
    bidirectional = bool(cfg['model'].get('bidirectional', False))
    dropout = float(cfg['model'].get('dropout', 0.20))

    # Initialize model
    model = SOH_LSTM_Seq2Seq(
        in_features=len(features),
        embed_size=embed_size,
        hidden_size=hidden_size,
        mlp_hidden=mlp_hidden,
        num_layers=num_layers,
        res_blocks=res_blocks,
        bidirectional=bidirectional,
        dropout=dropout,
    )

    # Load checkpoint
    device_obj = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device_obj)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    model.to(device_obj)

    return model, cfg, features


def convert_to_onnx(
    model: SOH_LSTM_Seq2Seq,
    output_path: str,
    seq_len: int,
    num_features: int,
    opset_version: int = 11,
    verbose: bool = True
) -> None:
    """
    Convert PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode
        output_path: Path to save ONNX file
        seq_len: Sequence length (e.g., 168 for hourly data over 1 week)
        num_features: Number of input features
        opset_version: ONNX opset version (X-CUBE-AI works best with 9-11)
        verbose: Print conversion details
    """
    # Create dummy input
    batch_size = 1
    dummy_input = torch.randn(batch_size, seq_len, num_features)

    if verbose:
        print(f"\nConverting model to ONNX...")
        print(f"  Input shape: ({batch_size}, {seq_len}, {num_features})")
        print(f"  Output path: {output_path}")
        print(f"  ONNX opset version: {opset_version}")

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )

    if verbose:
        print(f"✓ Model successfully converted to ONNX!")

        # Verify the export
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print(f"✓ ONNX model validation passed")
        except ImportError:
            print("  (Install 'onnx' package to verify exported model)")
        except Exception as e:
            print(f"⚠ ONNX validation warning: {e}")


def verify_conversion(
    pytorch_model: SOH_LSTM_Seq2Seq,
    onnx_path: str,
    num_features: int,
    seq_len: int,
    tolerance: float = 1e-5
) -> bool:
    """
    Verify that ONNX model produces same outputs as PyTorch model.

    Returns:
        True if outputs match within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        print("⚠ Install 'onnxruntime' to verify conversion accuracy")
        return False

    # Create test input
    test_input = torch.randn(1, seq_len, num_features)

    # Get PyTorch output
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input).numpy()

    # Get ONNX output
    ort_session = ort.InferenceSession(onnx_path)
    onnx_output = ort_session.run(None, {'input': test_input.numpy()})[0]

    # Compare
    max_diff = abs(pytorch_output - onnx_output).max()
    print(f"\nVerification:")
    print(f"  Max difference: {max_diff:.2e}")
    print(f"  Tolerance: {tolerance:.2e}")

    if max_diff < tolerance:
        print(f"✓ Outputs match within tolerance!")
        return True
    else:
        print(f"⚠ Warning: Outputs differ more than tolerance")
        return False


# ========================
# Main Function
# ========================

def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch SOH models to ONNX for STM32 X-CUBE-AI'
    )
    parser.add_argument(
        '--checkpoint', '-c',
        type=str,
        required=True,
        help='Path to PyTorch checkpoint (.pt file)'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to config YAML file'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output ONNX file path (default: same name as checkpoint with .onnx extension)'
    )
    parser.add_argument(
        '--opset',
        type=int,
        default=11,
        help='ONNX opset version (default: 11, recommended for X-CUBE-AI)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify conversion by comparing outputs'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for conversion'
    )

    args = parser.parse_args()

    # Determine output path
    if args.output is None:
        checkpoint_path = Path(args.checkpoint)
        args.output = str(checkpoint_path.with_suffix('.onnx'))

    # Create output directory if needed
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("PyTorch to ONNX Conversion for X-CUBE-AI")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Config: {args.config}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print("=" * 80)

    # Load model
    print("\nLoading model...")
    model, cfg, features = load_model_from_checkpoint(
        args.checkpoint,
        args.config,
        device=args.device
    )

    print(f"✓ Model loaded successfully")
    print(f"  Features ({len(features)}): {features}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Get sequence length from config
    seq_len = int(cfg['training']['seq_chunk_size'])
    print(f"  Sequence length: {seq_len}")

    # Convert to ONNX
    convert_to_onnx(
        model=model,
        output_path=args.output,
        seq_len=seq_len,
        num_features=len(features),
        opset_version=args.opset,
        verbose=True
    )

    # Verify conversion if requested
    if args.verify:
        verify_conversion(
            pytorch_model=model,
            onnx_path=args.output,
            num_features=len(features),
            seq_len=seq_len
        )

    print("\n" + "=" * 80)
    print("Conversion Complete!")
    print("=" * 80)
    print(f"\nYou can now use this ONNX file with X-CUBE-AI:")
    print(f"  stm32ai generate -m {args.output} --target stm32h7")
    print("\n" + "=" * 80 + "\n")


if __name__ == '__main__':
    main()
