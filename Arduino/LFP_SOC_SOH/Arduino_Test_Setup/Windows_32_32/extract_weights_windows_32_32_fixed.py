import torch
import torch.nn as nn
import numpy as np
import os

# Define constants based on BMS_SOC_LSTM_windows_2.1.4.1.py
INPUT_SIZE = 4
HIDDEN_SIZE = 32  # From VERSION 2.1.4.1 - Option A: 32
MLP_HIDDEN = 32   # From VERSION 2.1.4.1 - Option A: 32
NUM_LAYERS = 1    # Assuming single layer LSTM
DROPOUT_RATE = 0.1 # Matching typical dropout in BMS_SOC_LSTM_windows_2.1.4.1.py

# Define the model architecture (must match the trained model)
class SOCModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, batch_first=True, dropout=0) # Dropout is 0 for single layer LSTM in eval
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, MLP_HIDDEN),       # mlp[0]
            nn.ReLU(),                                # mlp[1]
            nn.Dropout(DROPOUT_RATE),                 # mlp[2]
            nn.Linear(MLP_HIDDEN, MLP_HIDDEN),        # mlp[3]
            nn.ReLU(),                                # mlp[4]
            nn.Dropout(DROPOUT_RATE),                 # mlp[5]
            nn.Linear(MLP_HIDDEN, 1)                  # mlp[6]
        )

    def forward(self, x, h_c_init=None):
        lstm_out, (hn, cn) = self.lstm(x, h_c_init)
        # Use the last time step's output from LSTM for the MLP
        out = self.mlp(lstm_out[:, -1, :])
        return out, (hn, cn) # Original script returned (out, (hn,cn))

def format_array_c(name, arr, is_bias=False):
    s = f"const float {name}[] = {{" if is_bias else f"const float {name}[][{arr.shape[1]}] = {{"
    if arr.ndim == 1:
        s += ", ".join([f"{x:.8f}f" for x in arr.flatten()])
    else:
        for i, row in enumerate(arr):
            s += "{" + ", ".join([f"{x:.8f}f" for x in row]) + "}"
            if i < len(arr) - 1:
                s += ", "
    s += "};\n"  # Fixed: single newline instead of double escaped
    return s

def extract_weights(model_path, output_header_path):
    device = torch.device('cpu')
    model = SOCModel()
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()

    os.makedirs(os.path.dirname(output_header_path), exist_ok=True)

    with open(output_header_path, "w") as f:
        f.write("#ifndef LSTM_WEIGHTS_H\n")
        f.write("#define LSTM_WEIGHTS_H\n\n")
        f.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
        f.write(f"#define HIDDEN_SIZE {HIDDEN_SIZE}\n")
        f.write(f"#define MLP_HIDDEN_SIZE {MLP_HIDDEN}\n\n")
        f.write("// LSTM Layer (Layer 0)\n")

        W_ih = model.lstm.weight_ih_l0.data.numpy()
        W_hh = model.lstm.weight_hh_l0.data.numpy()
        b_ih = model.lstm.bias_ih_l0.data.numpy()
        b_hh = model.lstm.bias_hh_l0.data.numpy()

        f.write("// Input Gate\n")
        f.write(format_array_c("lstm_input_ih_weights", W_ih[0*HIDDEN_SIZE : 1*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_input_hh_weights", W_hh[0*HIDDEN_SIZE : 1*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_input_ih_bias", b_ih[0*HIDDEN_SIZE : 1*HIDDEN_SIZE], is_bias=True))
        f.write(format_array_c("lstm_input_hh_bias", b_hh[0*HIDDEN_SIZE : 1*HIDDEN_SIZE], is_bias=True))
        f.write("\n")

        f.write("// Forget Gate\n")
        f.write(format_array_c("lstm_forget_ih_weights", W_ih[1*HIDDEN_SIZE : 2*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_forget_hh_weights", W_hh[1*HIDDEN_SIZE : 2*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_forget_ih_bias", b_ih[1*HIDDEN_SIZE : 2*HIDDEN_SIZE], is_bias=True))
        f.write(format_array_c("lstm_forget_hh_bias", b_hh[1*HIDDEN_SIZE : 2*HIDDEN_SIZE], is_bias=True))
        f.write("\n")

        f.write("// Candidate Gate\n")
        f.write(format_array_c("lstm_candidate_ih_weights", W_ih[2*HIDDEN_SIZE : 3*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_candidate_hh_weights", W_hh[2*HIDDEN_SIZE : 3*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_candidate_ih_bias", b_ih[2*HIDDEN_SIZE : 3*HIDDEN_SIZE], is_bias=True))
        f.write(format_array_c("lstm_candidate_hh_bias", b_hh[2*HIDDEN_SIZE : 3*HIDDEN_SIZE], is_bias=True))
        f.write("\n")

        f.write("// Output Gate\n")
        f.write(format_array_c("lstm_output_ih_weights", W_ih[3*HIDDEN_SIZE : 4*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_output_hh_weights", W_hh[3*HIDDEN_SIZE : 4*HIDDEN_SIZE, :]))
        f.write(format_array_c("lstm_output_ih_bias", b_ih[3*HIDDEN_SIZE : 4*HIDDEN_SIZE], is_bias=True))
        f.write(format_array_c("lstm_output_hh_bias", b_hh[3*HIDDEN_SIZE : 4*HIDDEN_SIZE], is_bias=True))
        f.write("\n")

        f.write("// MLP Layers (Sequential)\n")
        # MLP Layer 1 (model.mlp[0])
        mlp_fc1_w = model.mlp[0].weight.data.numpy()
        mlp_fc1_b = model.mlp[0].bias.data.numpy()
        f.write("// MLP Layer 1 (Linear -> ReLU -> Dropout)\n")
        f.write(format_array_c("mlp_fc1_weights", mlp_fc1_w)) # Shape (MLP_HIDDEN, HIDDEN_SIZE)
        f.write(format_array_c("mlp_fc1_bias", mlp_fc1_b, is_bias=True))
        f.write("\n")

        # MLP Layer 2 (model.mlp[3])
        mlp_fc2_w = model.mlp[3].weight.data.numpy()
        mlp_fc2_b = model.mlp[3].bias.data.numpy()
        f.write("// MLP Layer 2 (Linear -> ReLU -> Dropout)\n")
        f.write(format_array_c("mlp_fc2_weights", mlp_fc2_w)) # Shape (MLP_HIDDEN, MLP_HIDDEN)
        f.write(format_array_c("mlp_fc2_bias", mlp_fc2_b, is_bias=True))
        f.write("\n")

        # MLP Layer 3 (model.mlp[6])
        mlp_fc3_w = model.mlp[6].weight.data.numpy()
        mlp_fc3_b = model.mlp[6].bias.data.numpy()
        f.write("// MLP Layer 3 (Linear Output)\n")
        # mlp_fc3_w has shape (1, MLP_HIDDEN).
        f.write(f"const float mlp_fc3_weights[1][{MLP_HIDDEN}] = {{{{{', '.join([f'{x:.8f}f' for x in mlp_fc3_w.flatten()])}}}}};\n")
        f.write(format_array_c("mlp_fc3_bias", mlp_fc3_b, is_bias=True)) # mlp_fc3_b has shape (1)
        f.write("\n")

        f.write("#endif // LSTM_WEIGHTS_H\n")

    print(f"Weights extracted to {output_header_path}")

if __name__ == "__main__":
    model_file_path = "c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Windows_32_32/model/best_model.pth"
    output_h_file_path = "c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup/Windows_32_32/arduino_lstm_soc_windows_32_32/lstm_weights.h"
    
    extract_weights(model_file_path, output_h_file_path)
    print(f"Fixed weights extractor completed! Run: python {os.path.abspath(__file__)}")
