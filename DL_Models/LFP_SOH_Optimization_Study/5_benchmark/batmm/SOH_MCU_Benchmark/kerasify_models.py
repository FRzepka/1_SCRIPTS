from src.utils.cnn_utils import create_keras_cnn
from src.utils.tcn_utils import create_keras_tcn
from src.utils.lstm_utils import create_keras_lstm
from src.utils.gru_utils import create_keras_gru

from src.quantize_models import quantize_lstm, quantize_gru, quantize_cnn, quantize_tcn
from config import MODEL_ARCHS, PRUNING_RATIOS, PRUNE_SETTINGS


# ----- HELPERS -----

MODEL_OVERRIDES = {
    "cnn":  {"--model_name": "cnn"},
    "tcn":  {"--model_name": "tcn"},
    "lstm": {"--model_name": "lstm"},
    "gru":  {"--model_name": "gru"},
}


def build_prune_args(model_key: str, ratio: float, rounds: int) -> dict:
    args = dict(PRUNE_SETTINGS)
    args.update(MODEL_OVERRIDES.get(model_key, {}))   # Per-model overrides (merged on top of PRUNE_SETTINGS)
    args["--pruning_ratio"] = str(ratio)

    if rounds == 1:
        args["--pruning_mode"]    = "oneshot"
        args["--num_iterations"]  = "1"
    else:
        args["--pruning_mode"]    = "iterative"
        args["--num_iterations"]  = str(rounds)

    return args


# ----- CONVERSION -----

def kerasify(model_key: str, create_fn, quantize_fn):
    """
    For a given model:
      1. Run one baseline (no pruning) conversion + quantization.
      2. Sweep every (ratio, rounds) pair in PRUNING_RATIOS and do the same.
    """
    print(f"\n%%%%% {'='*64}")
    print(f"%%%%%   MODEL: {model_key.upper()}")
    print(f"%%%%% {'='*64}")

    # ----- Baseline -----
    print(f"\n%%%%% [{model_key}] -- baseline (no pruning) --")
    keras_path = create_fn()
    quantize_fn(
        keras_model_path=keras_path,
        tflite_output_path=str(keras_path).removesuffix(".keras") + ".tflite",
    )

    # ----- Pruning sweep -----
    for ratio, rounds in PRUNING_RATIOS.items():
        print(f"\n%%%%% [{model_key}] -- ratio={ratio:.0%}  num_iterations={rounds} --")
        prune_args = build_prune_args(model_key, ratio, rounds)
        keras_path = create_fn(pruning_args=prune_args)
        quantize_fn(
            keras_model_path=keras_path,
            tflite_output_path=str(keras_path).removesuffix(".keras") + ".tflite",
        )


def main():
    if "cnn" in MODEL_ARCHS:
        kerasify("cnn", create_keras_cnn, quantize_cnn)
    if "tcn" in MODEL_ARCHS:
        kerasify("tcn", create_keras_tcn, quantize_tcn)
    if "lstm" in MODEL_ARCHS:
        kerasify("lstm", create_keras_lstm, quantize_lstm)
    if "gru" in MODEL_ARCHS:
        kerasify("gru", create_keras_gru, quantize_gru)


if __name__ == "__main__":
    main()
