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

CONFIG_PATH = os.path.join(MODELS_DIR, "CNN/train_soh.yaml")
SCALER_PATH = os.path.join(MODELS_DIR, "CNN/scaler_robust.joblib")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "CNN/best_model.pt")


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


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dropout: float, dilation: int = 1):
        super().__init__()
        self.conv1      = CausalConv1d(in_ch,  out_ch, kernel_size, dilation=dilation)
        self.relu1      = nn.ReLU()
        self.drop1      = nn.Dropout(dropout)
        self.conv2      = CausalConv1d(out_ch, out_ch, kernel_size, dilation=dilation)
        self.relu2      = nn.ReLU()
        self.drop2      = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class SOH_CNN_Seq2Seq(nn.Module):
    def __init__(
        self,
        in_features:        int,
        hidden_size:        int,
        mlp_hidden:         int,
        kernel_size:        int = 5,
        dilations:          T.Optional[T.List[int]] = None,
        num_blocks:         int = 4,
        dropout:            float = 0.15,
        output_kernel_size: int = 1,
    ):
        super().__init__()
        self.input_proj = nn.Conv1d(in_features, hidden_size, kernel_size=1)

        if dilations is None:
            dilations = [1] * max(1, int(num_blocks))
        self.dilations = [int(d) for d in dilations]

        self.blocks = nn.Sequential(
            *[ConvBlock(hidden_size, hidden_size, kernel_size, dropout, dilation=d)
              for d in self.dilations]
        )
        self.head = nn.Sequential(
            nn.Conv1d(hidden_size, mlp_hidden, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(mlp_hidden, 1, kernel_size=1),
        )
        self.output_kernel_size = int(output_kernel_size)
        self.output_smoother = (
            CausalConv1d(1, 1, self.output_kernel_size, dilation=1)
            if self.output_kernel_size > 1 else None
        )
        self.kernel_size = kernel_size
        self.num_blocks  = len(self.dilations)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x   = x.transpose(1, 2)
        out = self.blocks(self.input_proj(x))
        y   = self.head(out)
        if self.output_smoother is not None:
            y = self.output_smoother(y)
        return y.squeeze(1)


def create_cnn():
    config = load_config(CONFIG_PATH)
    model = SOH_CNN_Seq2Seq(
        in_features=len(FEATURES),
        **config
    )
    return model


# =====================
#   QUANTIZABLE MODEL
# =====================

def pointwise_conv1d_tflite(x, filters, name, seq_len):
    """For kernel=1 convolutions — also needs Conv2D treatment in TFLite."""
    in_ch = x.shape[-1]
    x = layers.Reshape((seq_len, 1, in_ch), name=f'{name}_expand')(x)
    x = layers.Conv2D(filters, kernel_size=(1, 1), padding='valid',
                      use_bias=True, name=name)(x)
    x = layers.Reshape((seq_len, filters), name=f'{name}_squeeze')(x)
    return x

def causal_conv1d_tflite(x, filters, kernel_size, dilation_rate, name, seq_len):
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


def build_keras_cnn_model(in_features, hidden_size, mlp_hidden, kernel_size=5,
                       dilations=None, num_blocks=4, dropout=0.15,
                       output_kernel_size=1, seq_len=128):
    if dilations is None:
        dilations = [1] * max(1, int(num_blocks))

    inputs = keras.Input(shape=(seq_len, in_features), name="input")

    # Every Conv1D replaced — input_proj is pointwise (kernel=1)
    x = pointwise_conv1d_tflite(inputs, hidden_size, 'input_proj', seq_len)

    for i, d in enumerate(dilations):
        res = x
        x = causal_conv1d_tflite(x, hidden_size, kernel_size, d, f'blocks_{i}_conv1', seq_len)
        x = layers.ReLU(name=f'blocks_{i}_relu1')(x)
        x = layers.Dropout(dropout, name=f'blocks_{i}_drop1')(x)
        x = causal_conv1d_tflite(x, hidden_size, kernel_size, d, f'blocks_{i}_conv2', seq_len)
        x = layers.ReLU(name=f'blocks_{i}_relu2')(x)
        x = layers.Dropout(dropout, name=f'blocks_{i}_drop2')(x)
        x = layers.Add(name=f'blocks_{i}_add')([x, res])

    # Head — all kernel=1, use pointwise helper
    x = pointwise_conv1d_tflite(x, mlp_hidden, 'head_0', seq_len)
    x = layers.ReLU(name='head_1')(x)
    x = layers.Dropout(dropout, name='head_2')(x)
    x = pointwise_conv1d_tflite(x, 1, 'head_3', seq_len)

    if output_kernel_size > 1:
        x = causal_conv1d_tflite(x, 1, output_kernel_size, 1, 'output_smoother', seq_len)

    outputs = layers.Reshape((seq_len,), name='squeeze')(x)

    return keras.Model(inputs, outputs, name="SOH_CNN")

# 1. Helper function to transpose and transfer Conv1d weights
def transfer_conv1d_weights(pt_conv_layer, keras_layer):
    w = pt_conv_layer.weight.detach().cpu().numpy()  # (out, in, kernel)
    b = pt_conv_layer.bias.detach().cpu().numpy()

    if isinstance(keras_layer, keras.layers.Conv2D):
        # (out, in, k) → (kH, kW, in, out) = (k, 1, in, out)
        w_keras = w.transpose(2, 1, 0)[:, np.newaxis, :, :]
    else:
        # Conv1D: (out, in, k) → (k, in, out)
        w_keras = w.transpose(2, 1, 0)

    keras_layer.set_weights([w_keras, b])


def create_keras_cnn(model_name="cnn", seq_len=128, pruning_args: dict = None):

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

        # DYNAMICALLY OVERRIDE CONFIG SHAPES WITH PRUNED CHANNEL DIMENSIONS
        config["hidden_size"] = pt_model.input_proj.out_channels
        config["mlp_hidden"] = pt_model.head[0].out_channels
        print(f"--> Pruning completed. Adjusted hidden_size: {config['hidden_size']}, mlp_hidden: {config['mlp_hidden']}")

    else:
        print("No pruning arguments provided. Loading the unpruned baseline model...")

        # Fallback path: Load standard baseline architecture and its checkpoint
        pt_model = create_cnn()
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        pt_model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)))
        pt_model.eval()
        print("Loaded baseline model")

    # 3. Instantiate Keras Model with adjusted shapes
    print(f"Building Keras model architecture...")
    keras_model = build_keras_cnn_model(
        in_features=len(FEATURES),
        seq_len=seq_len,
        **config
    )

    # input_proj is now a Conv2D named 'input_proj'
    transfer_conv1d_weights(pt_model.input_proj, keras_model.get_layer('input_proj'))

    for i in range(pt_model.num_blocks):
        transfer_conv1d_weights(pt_model.blocks[i].conv1.conv, keras_model.get_layer(f'blocks_{i}_conv1'))
        transfer_conv1d_weights(pt_model.blocks[i].conv2.conv, keras_model.get_layer(f'blocks_{i}_conv2'))

    # head_0 and head_3 are now Conv2D too
    transfer_conv1d_weights(pt_model.head[0], keras_model.get_layer('head_0'))
    transfer_conv1d_weights(pt_model.head[3], keras_model.get_layer('head_3'))

    if pt_model.output_smoother is not None:
        transfer_conv1d_weights(pt_model.output_smoother.conv, keras_model.get_layer('output_smoother'))

    # 4. Map the weights
    print("Transferring weights...")

    # 4a. Input Projection
    transfer_conv1d_weights(pt_model.input_proj, keras_model.get_layer('input_proj'))

    # 4b. ConvBlocks
    for i in range(pt_model.num_blocks):
        for conv_attr, layer_name in [('conv1', f'blocks_{i}_conv1'), ('conv2', f'blocks_{i}_conv2')]:
            pt_conv = getattr(pt_model.blocks[i], conv_attr).conv
            k_layer = keras_model.get_layer(layer_name)
            transfer_conv1d_weights(pt_conv, k_layer)

    # 4c. Head (nn.Sequential indices 0 and 3 are Conv1d layers)
    transfer_conv1d_weights(pt_model.head[0], keras_model.get_layer('head_0'))
    transfer_conv1d_weights(pt_model.head[3], keras_model.get_layer('head_3'))

    # 4d. Output Smoother
    if pt_model.output_smoother is not None:
        transfer_conv1d_weights(pt_model.output_smoother.conv, keras_model.get_layer('output_smoother'))

    model_path = Path(COMPRESSED_MODELS_DIR).joinpath(f"{model_name}.keras")
    # 5. Save the final Keras model
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keras_model.save(save_path)
    print(f"Model successfully converted and saved to: {save_path}")
    return model_path
