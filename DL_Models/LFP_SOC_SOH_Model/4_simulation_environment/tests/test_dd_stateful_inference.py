import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch


RUNNER_PATH = (
    Path(__file__).resolve().parents[1]
    / "SOC_SOH_1.7.0.0_0.1.2.3"
    / "run_soc_soh_scenario.py"
)
SPEC = importlib.util.spec_from_file_location("jes2_dd_runner", RUNNER_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RUNNER)


@pytest.mark.parametrize("model_class", [RUNNER.GRUMLP, RUNNER.LSTMMLP])
def test_chunked_stateful_matches_one_step_execution(model_class):
    torch.manual_seed(17)
    rng = np.random.default_rng(17)
    features = rng.normal(size=(73, 4)).astype(np.float32)
    target = rng.uniform(size=73).astype(np.float32)
    model = model_class(in_features=4, hidden_size=7, mlp_hidden=5, num_layers=1, dropout=0.0)

    one_step, _, _ = RUNNER.stateful_stream_predict_soc(
        features, target, model, torch.device("cpu"), stream_chunk=1, show_progress=False
    )
    chunked, _, _ = RUNNER.stateful_stream_predict_soc(
        features, target, model, torch.device("cpu"), stream_chunk=19, show_progress=False
    )

    np.testing.assert_allclose(chunked, one_step, rtol=1e-6, atol=1e-7)


def test_stateful_stream_chunk_must_be_positive():
    model = RUNNER.GRUMLP(in_features=2, hidden_size=3, mlp_hidden=3, dropout=0.0)
    with pytest.raises(ValueError, match="stream_chunk must be positive"):
        RUNNER.stateful_stream_predict_soc(
            np.zeros((2, 2), dtype=np.float32),
            np.zeros(2, dtype=np.float32),
            model,
            torch.device("cpu"),
            stream_chunk=0,
            show_progress=False,
        )
