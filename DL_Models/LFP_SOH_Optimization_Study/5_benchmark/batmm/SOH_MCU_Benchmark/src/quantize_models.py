import numpy as np
import tensorflow as tf
import joblib
from pathlib import Path

from config import SCALER_PATH, DATA_DIR
import src.utils.data_utils as data_utils


# ----- HELPERS -----

def get_calibration_sequence(seq_len=1):
    """
    Uses data_utils to fetch a continuous real sequence of data based on model's window size.
    """
    _, val_loader, _ = data_utils.build_dataloaders(
        data_root=DATA_DIR,
        scaler=joblib.load(SCALER_PATH),
        chunk=seq_len,
        batch_size=1,
        num_workers=0,
        load_train=False,
        load_val=True,
        load_test=False
    )

    print("%%%%% Loaded data successfully.")
    return val_loader


def configure_quantization(
    model,
    representative_data_gen,
    int16_activations: bool,
    io_type: str,
    fp32_states: bool = False,
) -> None:
    """Configures TFLite converter quantization, operations, and I/O types."""

    print(f"%%%%% Configuring TFLite Converter...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # Configure Supported Ops (8x8 vs 16x8)
    if int16_activations:
        print("%%%%% Using 16x8 Quantization (16-bit activations, 8-bit weights)")
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
    else:
        print("%%%%% Using standard 8x8 INT8 Quantization")
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

    # Configure I/O Types
    if fp32_states:
        print("%%%%% fp32_states=True: Forcing I/O interface to FP32 to protect recurrent states")
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32
    else:
        io_type_clean = io_type.lower()
        if io_type_clean == "int8":
            converter.inference_input_type = tf.int8
            converter.inference_output_type = tf.int8
        elif io_type_clean == "fp32":
            converter.inference_input_type = tf.float32
            converter.inference_output_type = tf.float32
        else:
            raise ValueError("##### Invalid io_type provided. Use 'int8' or 'fp32'")
    return converter


def save_tflite_model(converter, output_path: str | Path, model_name: str = "Model") -> None:
    """Converts a TFLite converter object and saves it to disk."""
    try:
        print(f"%%%%% Converting {model_name} to TFLite...")
        tflite_model = converter.convert()

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "wb") as f:
            f.write(tflite_model)
        print(f"%%%%% Success! {model_name} saved to: {path}")

    except Exception as e:
        print(f"##### Quantization failed for {model_name}: {e}")


##########
# LSTM
##########

def quantize_lstm(
    keras_model_path="SOH_LSTM_converted_unrolled.keras",
    tflite_output_path="SOH_LSTM_converted_unrolled_int8.tflite",
    io_type="int8",
    fp32_states=False,
    int16_activations=False
):
    print("%%%%% Loading Keras LSTM model...")
    model = tf.keras.models.load_model(keras_model_path)

    val_loader = get_calibration_sequence(seq_len=1)   # LSTM feeds 1 timestep at a time

    def representative_data_gen():
        input_specs = {inp.name.split(':')[0]: tuple(inp.shape) for inp in model.inputs}

        current_states = {}
        for name, shape in input_specs.items():
            if "input_x" not in name:
                current_states[name] = np.zeros(shape, dtype=np.float32)

        print(f"\n%%%%% Calibrating...")
        for X, y in val_loader:
            x_val = X.detach().cpu().numpy().astype(np.float32)

            inputs_dict = {}
            for name, shape in input_specs.items():
                if "input_x" in name:
                    inputs_dict[name] = np.reshape(x_val, shape)
                else:
                    inputs_dict[name] = current_states[name]

            yield inputs_dict

            outputs = model(inputs_dict, training=False)

            num_layers = (len(outputs) - 1) // 2
            for i in range(num_layers):
                current_states[f"h_in_{i}"] = outputs[1 + 2*i].numpy()
                current_states[f"c_in_{i}"] = outputs[2 + 2*i].numpy()

    converter = configure_quantization(model, representative_data_gen, int16_activations, io_type, fp32_states=fp32_states)

    save_tflite_model(converter, tflite_output_path, model_name="LSTM")


##########
# GRU
##########

def quantize_gru(
    keras_model_path="SOH_GRU_converted_unrolled.keras",
    tflite_output_path="SOH_GRU_converted_unrolled_int8.tflite",
    io_type="int8",
    fp32_states=False,
    int16_activations=False
):
    print("%%%%% Loading Keras GRU model...")
    model = tf.keras.models.load_model(keras_model_path)

    val_loader = get_calibration_sequence(seq_len=1)   # GRU feeds 1 timestep at a time

    def representative_data_gen():
        input_specs = {inp.name.split(':')[0]: tuple(inp.shape) for inp in model.inputs}

        current_states = {}
        for name, shape in input_specs.items():
            if "input_x" not in name:
                current_states[name] = np.zeros(shape, dtype=np.float32)

        print(f"\n%%%%% Calibrating...")
        for X, y in val_loader:
            x_val = X.detach().cpu().numpy().astype(np.float32)

            inputs_dict = {}
            for name, shape in input_specs.items():
                if "input_x" in name:
                    inputs_dict[name] = np.reshape(x_val, shape)
                else:
                    inputs_dict[name] = current_states[name]

            yield inputs_dict

            outputs = model(inputs_dict, training=False)

            num_layers = len(outputs) - 1
            for i in range(num_layers):
                current_states[f"h_in_{i}"] = outputs[1 + i].numpy()

    converter = configure_quantization(model, representative_data_gen, int16_activations, io_type, fp32_states=fp32_states)

    save_tflite_model(converter, tflite_output_path, model_name="GRU")


##########
# CNN
##########

def quantize_cnn(
    keras_model_path="SOH_CNN_converted.keras",
    tflite_output_path="SOH_CNN_converted_int8.tflite",
    io_type="int8",
    int16_activations=False,
    seq_len=128
):
    print("%%%%% Loading Keras CNN model...")
    model = tf.keras.models.load_model(keras_model_path)

    val_loader = get_calibration_sequence(seq_len=seq_len)

    def representative_data_gen():
        input_tensor = model.inputs[0]
        input_name = input_tensor.name.split(':')[0]
        input_shape = tuple(input_tensor.shape)

        print(f"\n%%%%% Calibrating...")
        for X, y in val_loader:
            x_val = X.detach().cpu().numpy().astype(np.float32)
            target_shape = [1 if dim is None else dim for dim in input_shape]
            yield {input_name: np.reshape(x_val, target_shape)}

    converter = configure_quantization(model, representative_data_gen, int16_activations, io_type)

    save_tflite_model(converter, tflite_output_path, model_name="CNN")

##########
# TCN
##########

def quantize_tcn(
    keras_model_path="SOH_TCN_converted.keras",
    tflite_output_path="SOH_TCN_converted_int8.tflite",
    io_type="int8",
    int16_activations=False,
    seq_len=96
):
    print("%%%%% Loading Keras TCN model...")
    model = tf.keras.models.load_model(keras_model_path)

    val_loader = get_calibration_sequence(seq_len=seq_len)

    def representative_data_gen():
        input_tensor = model.inputs[0]
        input_name = input_tensor.name.split(':')[0]
        input_shape = tuple(input_tensor.shape)

        print(f"\n%%%%% Calibrating...")
        for X, y in val_loader:
            x_val = X.detach().cpu().numpy().astype(np.float32)
            target_shape = [1 if dim is None else dim for dim in input_shape]
            yield {input_name: np.reshape(x_val, target_shape)}

    converter = configure_quantization(model, representative_data_gen, int16_activations, io_type)

    save_tflite_model(converter, tflite_output_path, model_name="TCN")

