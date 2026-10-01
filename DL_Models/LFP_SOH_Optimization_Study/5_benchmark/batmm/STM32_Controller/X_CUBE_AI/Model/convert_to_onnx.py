"""
convert_to_onnx.py
------------------
Converts the trained PyTorch battery life model checkpoint (.pt)
to ONNX format for interoperability (e.g. TensorFlow, TFLite, Edge AI).
"""

import torch
import torch.nn as nn
import onnx

INPUT_SEQ = 32

# ================================================================
# 1. Define model architecture (from checkpoint inspection)
# ================================================================
class BatterySOHEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last time step output
        out = self.mlp(out)
        return out


# ================================================================
# 2. Load model checkpoint safely
# ================================================================
def load_model(checkpoint_path: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract actual weights
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = BatterySOHEstimator()
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    print("✅ Model loaded successfully")

    if "config" in checkpoint:
        cfg = checkpoint["config"]
        print("Model config:")
        print(f"  Type: {cfg['model']['type']}")
        print(f"  Hidden size: {cfg['model']['hidden_size']}")
        print(f"  Features: {cfg['model']['features']}")
        print(f"  Sequence length: {cfg['training']['seq_chunk_size']}")

    return model


# ================================================================
# 3. Export to ONNX
# ================================================================
def export_to_onnx(model, onnx_path=f"battery_lstm_{INPUT_SEQ}.onnx"):
    # Dummy input: 1 sequence of 1024 steps, 7 features
    dummy_input = torch.randn(1, INPUT_SEQ, 7, dtype=torch.float32)  # 2048

    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        verbose=True
    )

    print(f"✅ Exported model to {onnx_path}")

    # Optional: verify the exported model structure
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    print("✅ ONNX model structure verified successfully")


# ================================================================
# 4. Main
# ================================================================
if __name__ == "__main__":
    checkpoint_path = "2.1.0.0_soh_epoch0005_mae0.00369.pt"
    onnx_output_path = "battery_lstm.onnx"

    print("Loading model...")
    model = load_model(checkpoint_path)

    print("Exporting to ONNX...")
    export_to_onnx(model)

    print("\nAll done!")
