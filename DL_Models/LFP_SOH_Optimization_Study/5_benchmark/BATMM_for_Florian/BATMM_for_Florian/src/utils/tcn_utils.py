import typing as T
import torch
import torch.nn as nn
import os
import tensorflow as tf
import numpy as np
from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers

from config import MODELS_DIR, COMPRESSED_MODELS_DIR
from src.utils.data_utils import load_config, FEATURES


# =========
#   PATHS
# =========

CONFIG_PATH = os.path.join(MODELS_DIR, "TCN/train_soh.yaml")
SCALER_PATH = os.path.join(MODELS_DIR, "TCN/scaler_robust.joblib")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "TCN/best_model.pt")


# =========
#   MODEL
# =========

class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.pad  = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=self.pad)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        return out[:, :, :-self.pad] if self.pad > 0 else out


class TemporalBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1      = CausalConv1d(in_ch,  out_ch, kernel_size, dilation=dilation)
        self.relu1      = nn.ReLU()
        self.dropout1   = nn.Dropout(dropout)
        self.conv2      = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.relu2      = nn.ReLU()
        self.dropout2   = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout1(self.relu1(self.conv1(x)))
        out = self.dropout2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class SOH_TCN_Seq2Seq(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_size: int,
        mlp_hidden:  int,
        kernel_size: int,
        num_layers:  int,
        dilations:   T.List[int],
        dropout:     float,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilations   = dilations

        layers, ch_in = [], in_features
        for d in dilations:
            layers.append(TemporalBlock(ch_in, hidden_size, kernel_size, d, dropout))
            ch_in = hidden_size
        self.tcn = nn.Sequential(*layers)

        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mlp_hidden, 1, kernel_size=1),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    @property
    def receptive_field(self) -> int:
        rf = 1
        for d in self.dilations:
            rf += 2 * (self.kernel_size - 1) * d
        return rf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        y = self.head(self.tcn(x))
        return y.squeeze(1)


def create_tcn():
    config = load_config(CONFIG_PATH)
    model = SOH_TCN_Seq2Seq(
        in_features=len(FEATURES),
        **config
    )
    return model


# =====================
#   QUANTIZABLE MODEL
# =====================

def pointwise_conv1d_tflite(x, filters, name, seq_len):
    """For kernel=1 convolutions — uses Conv2D for TFLite compatibility."""
    in_ch = x.shape[-1]
    x = layers.Reshape((seq_len, 1, in_ch), name=f'{name}_expand')(x)
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='valid',
                      use_bias=True, name=name)(x)
    x = layers.Reshape((seq_len, filters), name=f'{name}_squeeze')(x)
    return x

def causal_conv1d_tflite(x, filters, kernel_size, dilation_rate, name, seq_len):
    """For causal dilated convolutions — uses Conv2D for TFLite compatibility."""
    in_ch = x.shape[-1]
    x = layers.Reshape((seq_len, 1, in_ch), name=f'{name}_expand')(x)
    x = layers.ZeroPadding2D(
            padding=(((kernel_size - 1) * dilation_rate, 0), (0, 0)),
            name=f'{name}_pad'
        )(x)
    x = layers.Conv2D(
            filters,
            kernel_size=(kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            padding='valid',
            use_bias=True,
            name=name
        )(x)
    x = layers.Reshape((seq_len, filters), name=f'{name}_squeeze')(x)
    return x

def build_keras_tcn_model(in_features, hidden_size, mlp_hidden, kernel_size,
                           num_layers, dilations, dropout, seq_len=96, tcn_shapes=None, **kwargs):
    inputs = keras.Input(shape=(seq_len, in_features), name="input")
    x = inputs

    ch_in = in_features
    for i, d in enumerate(dilations):
        res = x

        # Read exact pruned shapes if provided, else fallback to standard hidden_size
        if tcn_shapes is not None:
            c1_out = tcn_shapes[i]['conv1_out']
            c2_out = tcn_shapes[i]['conv2_out']
            down_out = tcn_shapes[i]['downsample_out']
        else:
            c1_out = hidden_size
            c2_out = hidden_size
            down_out = hidden_size if ch_in != hidden_size else None

        # Conv1
        x = causal_conv1d_tflite(x, c1_out, kernel_size, d, f'tcn_{i}_conv1', seq_len)
        x = layers.ReLU(name=f'tcn_{i}_relu1')(x)
        x = layers.Dropout(dropout, name=f'tcn_{i}_drop1')(x)

        # Conv2
        x = causal_conv1d_tflite(x, c2_out, kernel_size, d, f'tcn_{i}_conv2', seq_len)
        x = layers.ReLU(name=f'tcn_{i}_relu2')(x)
        x = layers.Dropout(dropout, name=f'tcn_{i}_drop2')(x)

        # Downsample if needed (1x1 conv)
        if down_out is not None or ch_in != c2_out:
            d_out = down_out if down_out is not None else c2_out
            res = pointwise_conv1d_tflite(res, d_out, f'tcn_{i}_downsample', seq_len)

        x = layers.Add(name=f'tcn_{i}_add')([x, res])
        ch_in = c2_out # Update channels for subsequent blocks

    # Head (All kernel=1)
    x = pointwise_conv1d_tflite(x, mlp_hidden, 'head_0', seq_len)
    x = layers.ReLU(name='head_1')(x)
    x = layers.Dropout(dropout, name='head_2')(x)
    x = pointwise_conv1d_tflite(x, 1, 'head_3', seq_len)

    outputs = layers.Reshape((seq_len,), name='squeeze')(x)

    return keras.Model(inputs, outputs, name="SOH_TCN")


def transfer_conv1d_weights(pt_conv_layer, keras_layer):
    """Helper function to transpose and transfer Conv1d weights to Conv2D"""
    w = pt_conv_layer.weight.detach().cpu().numpy()  # (out, in, kernel)

    # Handle biases safely
    if pt_conv_layer.bias is not None:
        b = pt_conv_layer.bias.detach().cpu().numpy()
    else:
        b = np.zeros(w.shape[0])

    if isinstance(keras_layer, keras.layers.Conv2D):
        # Conv2D expected: (kernel_H, kernel_W, in, out)
        # PyTorch Conv1d: (out, in, kernel_size) -> transposes to (kernel_size, 1, in, out)
        w_keras = w.transpose(2, 1, 0)[:, np.newaxis, :, :]
    else:
        # Standard Keras Conv1D expected: (kernel, in, out)
        w_keras = w.transpose(2, 1, 0)

    keras_layer.set_weights([w_keras, b])


def create_keras_tcn(model_name="tcn", seq_len=96, pruning_args: dict = None):

    from src.pruning.prune import main as prune_model   # to avoid circular import
    config = load_config(CONFIG_PATH)

    # 2. Check if we should prune or load the baseline model
    if pruning_args:
        print("Pruning the model...")

        pruning_cmd_list = []
        for key, value in pruning_args.items():
            pruning_cmd_list.append(str(key))
            pruning_cmd_list.append(str(value))

        # Run the pruning script programmatically
        pt_model, pt_model_name = prune_model(pruning_cmd_list)
        model_name = pt_model_name
        pt_model.eval()

        # DYNAMICALLY OVERRIDE CONFIG SHAPES WITH EXACT LAYER-BY-LAYER DIMENSIONS
        tcn_shapes = []
        for i in range(len(pt_model.dilations)):
            tcn_shapes.append({
                'conv1_out': pt_model.tcn[i].conv1.conv.out_channels,
                'conv2_out': pt_model.tcn[i].conv2.conv.out_channels,
                'downsample_out': pt_model.tcn[i].downsample.out_channels if pt_model.tcn[i].downsample is not None else None
            })

        config["tcn_shapes"] = tcn_shapes
        config["mlp_hidden"] = pt_model.head[0].out_channels
        print(f"--> Pruning completed. Extracted heterogeneous block shapes. mlp_hidden: {config['mlp_hidden']}")

    else:
        print("No pruning arguments provided. Loading the unpruned baseline model...")

        # Fallback path: Load standard baseline architecture and its checkpoint
        pt_model = create_tcn()
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        pt_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        pt_model.eval()
        print("Loaded baseline model")

    # 3. Instantiate Keras Model with adjusted shapes
    print(f"Building Keras model architecture...")
    keras_model = build_keras_tcn_model(
        in_features=len(FEATURES),
        seq_len=seq_len,
        **config
    )

    print("Transferring weights...")

    # 1. Temporal Blocks
    for i, d in enumerate(pt_model.dilations):
        # TCN block
        pt_block = pt_model.tcn[i]

        # Transfer CausalConv1ds (We access .conv because CausalConv1d wraps nn.Conv1d)
        transfer_conv1d_weights(pt_block.conv1.conv, keras_model.get_layer(f'tcn_{i}_conv1'))
        transfer_conv1d_weights(pt_block.conv2.conv, keras_model.get_layer(f'tcn_{i}_conv2'))

        # Transfer Downsample if it exists (It's a direct nn.Conv1d)
        if pt_block.downsample is not None:
            transfer_conv1d_weights(pt_block.downsample, keras_model.get_layer(f'tcn_{i}_downsample'))

    # 2. Head
    # Indices 0 and 3 are the Conv1d layers in nn.Sequential
    transfer_conv1d_weights(pt_model.head[0], keras_model.get_layer('head_0'))
    transfer_conv1d_weights(pt_model.head[3], keras_model.get_layer('head_3'))

    model_path = Path(COMPRESSED_MODELS_DIR).joinpath(f"{model_name}.keras")
    # 5. Save the final Keras model
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keras_model.save(save_path)
    print(f"Model successfully converted and saved to: {save_path}")
    return model_path
