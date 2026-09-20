import torch
import torch.nn as nn
import onnx

class BatterySOHEstimator(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=7, hidden_size=128, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x, h=None, c=None):
        # Keep or initialize hidden state
        if h is None or c is None:
            out, (h, c) = self.lstm(x)
        else:
            out, (h, c) = self.lstm(x, (h, c))
        # Only the last time-step output goes through MLP
        y = self.mlp(out[:, -1, :])
        return y, h, c


def export_to_onnx(model, onnx_path="battery_lstm_stateful.onnx"):
    dummy_x = torch.randn(1, 16, 7)            # e.g. 16-step chunk
    h0 = torch.zeros(1, 1, 128)
    c0 = torch.zeros(1, 1, 128)

    torch.onnx.export(
        model,
        (dummy_x, h0, c0),
        onnx_path,
        input_names=["x", "h_in", "c_in"],
        output_names=["y", "h_out", "c_out"],
        dynamic_axes={
            "x": {1: "seq_len"},  # allow variable chunk size
        },
        opset_version=17,
        do_constant_folding=True
    )

    print(f"✅ Exported stateful ONNX model to {onnx_path}")
    onnx.checker.check_model(onnx.load(onnx_path))
    print("✅ ONNX model verified")

if __name__ == "__main__":
    ckpt = "2.1.0.0_soh_epoch0005_mae0.00369.pt"
    state_dict = torch.load(ckpt, map_location="cpu")
    model = BatterySOHEstimator()
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    export_to_onnx(model)
