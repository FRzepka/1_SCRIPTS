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

CONFIG_PATH = os.path.join(MODELS_DIR, "GRU/train_soh.yaml")
SCALER_PATH = os.path.join(MODELS_DIR, "GRU/scaler_robust.joblib")
CHECKPOINT_PATH = os.path.join(MODELS_DIR, "GRU/best_model.pt")


# =========
#   MODEL
# =========

class ResidualMLPBlock(nn.Module):
    """Two-layer MLP residual block with LayerNorm, used by GRU and LSTM heads."""

    def __init__(self, dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.fc2(self.act(self.fc1(x)))
        return self.norm(x + self.drop(out))


class SOH_GRU_Seq2Seq(nn.Module):
    """Stateful-ready GRU that outputs SOH at every timestep."""

    def __init__(
        self,
        in_features:   int,
        embed_size:    int,
        hidden_size:   int,
        mlp_hidden:    int,
        num_layers:    int   = 2,
        res_blocks:    int   = 2,
        bidirectional: bool  = False,
        dropout:       float = 0.15,
    ):
        super().__init__()
        if bidirectional:
            print("Warning: bidirectional=True breaks true stateful inference.")
        self.hidden_size    = hidden_size
        self.num_layers     = num_layers
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

        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        gru_out = hidden_size * self.num_directions
        self.post_norm  = nn.LayerNorm(gru_out)
        self.res_blocks = nn.ModuleList(
            [ResidualMLPBlock(gru_out, mlp_hidden, dropout)
             for _ in range(max(0, int(res_blocks)))]
        )
        self.head = nn.Sequential(
            nn.Linear(gru_out,    mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                nn.init.zeros_(param.data)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        x:            torch.Tensor,
        state:        T.Optional[torch.Tensor] = None,
        return_state: bool = False,
    ):
        x = self.feature_proj(x)
        out, new_state = self.gru(x, state)
        out = self.post_norm(out)
        for blk in self.res_blocks:
            out = blk(out)
        y_seq = self.head(out).squeeze(-1)
        if return_state:
            return y_seq, new_state
        return y_seq


def create_gru():
    config = load_config(CONFIG_PATH)
    model = SOH_GRU_Seq2Seq(
        in_features=len(FEATURES),
        **config
    )
    return model

# ==========================================
#   WEIGHT TRANSFER HELPERS
# ==========================================

def transfer_linear(pt_layer: nn.Linear, keras_layer: keras.layers.Layer):
    """Transposes PyTorch Linear weights to Keras Dense weights."""
    w = pt_layer.weight.detach().cpu().numpy().T
    if pt_layer.bias is not None:
        b = pt_layer.bias.detach().cpu().numpy()
    else:
        b = np.zeros(w.shape[1])
    keras_layer.set_weights([w, b])

def transfer_layernorm(pt_layer: nn.LayerNorm, keras_layer: keras.layers.Layer):
    """Copies weights directly (same shape)."""
    g = pt_layer.weight.detach().cpu().numpy()
    b = pt_layer.bias.detach().cpu().numpy()
    keras_layer.set_weights([g, b])

def transfer_gru(pt_gru: nn.GRU, keras_model: keras.Model, num_layers: int):
    """
    Extracts PyTorch GRU weights, matches Keras' gate ordering (z, r, n),
    and loads them into Keras GRU layers expecting reset_after=True.
    """
    for i in range(num_layers):
        # 1. Extract PyTorch weights
        w_ih = getattr(pt_gru, f'weight_ih_l{i}').detach().cpu().numpy()
        w_hh = getattr(pt_gru, f'weight_hh_l{i}').detach().cpu().numpy()
        b_ih = getattr(pt_gru, f'bias_ih_l{i}').detach().cpu().numpy()
        b_hh = getattr(pt_gru, f'bias_hh_l{i}').detach().cpu().numpy()

        # 2. Split weights into individual gates
        # PyTorch order: Reset (r), Update (z), New/Hidden (n)
        w_ih_r, w_ih_z, w_ih_n = np.split(w_ih, 3, axis=0)
        w_hh_r, w_hh_z, w_hh_n = np.split(w_hh, 3, axis=0)

        b_ih_r, b_ih_z, b_ih_n = np.split(b_ih, 3, axis=0)
        b_hh_r, b_hh_z, b_hh_n = np.split(b_hh, 3, axis=0)

        # 3. Recombine matching Keras order: Update (z), Reset (r), New (n)
        kernel = np.concatenate([w_ih_z, w_ih_r, w_ih_n], axis=0).T
        recurrent_kernel = np.concatenate([w_hh_z, w_hh_r, w_hh_n], axis=0).T

        # Keras with reset_after=True expects bias shape (2, 3 * hidden)
        bias_i = np.concatenate([b_ih_z, b_ih_r, b_ih_n], axis=0)
        bias_h = np.concatenate([b_hh_z, b_hh_r, b_hh_n], axis=0)
        bias = np.stack([bias_i, bias_h])

        # 4. Apply to Keras layer
        keras_gru = keras_model.get_layer(f"gru_{i}")
        keras_gru.set_weights([kernel, recurrent_kernel, bias])

# ==========================================
#   KERAS UNROLLED GRU MODEL
# ==========================================

def build_keras_gru_model(
    in_features: int,
    embed_size: int,
    hidden_size: int,
    mlp_hidden: int,
    num_layers: int = 2,
    res_blocks: int = 2,
    gru_shapes: dict = None,
    **kwargs
) -> keras.Model:
    """Creates a Keras model predicting 1 timestep with strictly static shapes for STM32."""

    # 1. Main Input
    x_input = keras.Input(batch_shape=(1, 1, in_features), name="input_x")

    # Fallback to standard symmetrical shapes if no pruning map is provided
    if gru_shapes is None:
        gru_shapes = {
            "fp_dense_1_out": embed_size,
            "fp_dense_2_out": embed_size,
            "head_fc1_out": mlp_hidden,
            "head_fc2_out": mlp_hidden,
            "res_blocks": [{"fc1_out": mlp_hidden} for _ in range(res_blocks)]
        }

    # 2. State Inputs: GRU only requires `h` (no cell state `c` like LSTM)
    states_in = []
    for i in range(num_layers):
        h_in = keras.Input(batch_shape=(1, hidden_size), name=f"h_in_{i}")
        states_in.append(h_in)

    # Feature Projection
    x = layers.Dense(gru_shapes["fp_dense_1_out"], name="fp_dense_1")(x_input)
    x = layers.LayerNormalization(epsilon=1e-5, name="fp_ln")(x)
    x = layers.Activation('gelu', name="fp_gelu_1")(x)
    x = layers.Dense(gru_shapes["fp_dense_2_out"], name="fp_dense_2")(x)
    x = layers.Activation('gelu', name="fp_gelu_2")(x)

    # GRU Blocks
    states_out = []
    for i in range(num_layers):
        gru_layer = layers.GRU(
            hidden_size,
            return_sequences=True,
            return_state=True,
            unroll=True,
            reset_after=True,  # CRITICAL: Ensures PyTorch mathematical equivalence
            name=f"gru_{i}"
        )
        h_in_i = states_in[i]

        x, h_out = gru_layer(x, initial_state=[h_in_i])
        states_out.append(h_out)

    # Post GRU Norm
    x = layers.LayerNormalization(epsilon=1e-5, name="post_norm")(x)

    # Residual MLP Blocks
    for i in range(int(res_blocks)):
        res = x
        x = layers.Dense(gru_shapes["res_blocks"][i]["fc1_out"], name=f"res_{i}_fc1")(x)
        x = layers.Activation('gelu', name=f"res_{i}_gelu")(x)
        # res_x_fc2 returns to hidden_size, protected from pruning
        x = layers.Dense(hidden_size, name=f"res_{i}_fc2")(x)
        x = layers.Add(name=f"res_{i}_add")([res, x])
        x = layers.LayerNormalization(epsilon=1e-5, name=f"res_{i}_ln")(x)

    # Head
    x = layers.Dense(gru_shapes["head_fc1_out"], name="head_fc1")(x)
    x = layers.Activation('gelu', name="head_gelu_1")(x)
    x = layers.Dense(gru_shapes["head_fc2_out"], name="head_fc2")(x)
    x = layers.Activation('gelu', name="head_gelu_2")(x)
    x = layers.Dense(1, name="head_out")(x)

    # Clean the output shape
    x = layers.Reshape((1,), name='squeeze')(x)

    return keras.Model(
        inputs=[x_input] + states_in,
        outputs=[x] + states_out,
        name="SOH_GRU_Step"
    )

# ==========================================
#   CONVERSION SCRIPT
# ==========================================

def create_keras_gru(model_name="gru", pruning_args: dict = None):

    from src.pruning.prune import main as prune_model   # to avoid circular import
    config = load_config(CONFIG_PATH)

    # 1. Check if we should prune or load the baseline model
    if pruning_args:
        print("Pruning arguments detected. Executing structured pruning pipeline...")

        pruning_cmd_list = []
        for key, value in pruning_args.items():
            pruning_cmd_list.append(str(key))
            pruning_cmd_list.append(str(value))

        # Run the pruning script programmatically
        pt_model, pt_model_name = prune_model(pruning_cmd_list)
        model_name = pt_model_name
        pt_model.eval()

        # DYNAMICALLY OVERRIDE CONFIG SHAPES WITH EXACT LAYER-BY-LAYER DIMENSIONS
        gru_shapes = {
            "fp_dense_1_out": pt_model.feature_proj[0].out_features,
            "fp_dense_2_out": pt_model.feature_proj[4].out_features,
            "head_fc1_out": pt_model.head[0].out_features,
            "head_fc2_out": pt_model.head[3].out_features,
            "res_blocks": [{"fc1_out": blk.fc1.out_features} for blk in pt_model.res_blocks]
        }

        config["gru_shapes"] = gru_shapes
        config["embed_size"] = pt_model.feature_proj[0].out_features
        config["hidden_size"] = pt_model.gru.hidden_size
        config["mlp_hidden"] = pt_model.head[0].out_features
        print(f"--> Pruning completed. Extracted heterogeneous dense shapes.")

    else:
        print("No pruning arguments provided. Loading the unpruned baseline model...")
        pt_model = create_gru()
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
        pt_model.load_state_dict(state_dict)
        pt_model.eval()
        print(f"--> Baseline loaded successfully")

    # 2. Creating Keras unrolled GRU model
    print("Building Keras model architecture...")
    keras_model = build_keras_gru_model(
        in_features=len(FEATURES),
        **config
    )

    # 3. Transferring weights (matching exact architecture)
    print("Transferring weights (matching exact architecture)...")

    # Feature Projection
    transfer_linear(pt_model.feature_proj[0], keras_model.get_layer("fp_dense_1"))
    transfer_layernorm(pt_model.feature_proj[1], keras_model.get_layer("fp_ln"))
    transfer_linear(pt_model.feature_proj[4], keras_model.get_layer("fp_dense_2"))

    # GRU Stack
    transfer_gru(pt_model.gru, keras_model, pt_model.num_layers)

    # Post Norm
    transfer_layernorm(pt_model.post_norm, keras_model.get_layer("post_norm"))

    # Residual Blocks
    for i, blk in enumerate(pt_model.res_blocks):
        transfer_linear(blk.fc1, keras_model.get_layer(f"res_{i}_fc1"))
        transfer_linear(blk.fc2, keras_model.get_layer(f"res_{i}_fc2"))
        transfer_layernorm(blk.norm, keras_model.get_layer(f"res_{i}_ln"))

    # Head
    transfer_linear(pt_model.head[0], keras_model.get_layer("head_fc1"))
    transfer_linear(pt_model.head[3], keras_model.get_layer("head_fc2"))
    transfer_linear(pt_model.head[6], keras_model.get_layer("head_out"))

    # 4. Save the final Keras model
    model_path = Path(COMPRESSED_MODELS_DIR).joinpath(f"{model_name}.keras")
    save_path = Path(model_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keras_model.save(save_path)
    print(f"Model successfully converted and saved to: {save_path}")

    return model_path
