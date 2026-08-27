from __future__ import annotations


# Alias, runner scenario, command-line arguments. The levels are fixed here so
# the campaign manifest and paper figures refer to one immutable protocol.
SCENARIOS = [
    ("baseline", "baseline", []),
    # Internal duration-matched reference for the 48 h burst-dropout run. It is
    # excluded from paper scenario panels and is used only for paired deltas.
    ("missing_gap_baseline_48h", "baseline", []),
    ("current_noise_low", "current_noise", ["--current_noise_std", "0.02"]),
    ("current_noise_high", "current_noise", ["--current_noise_std", "0.10"]),
    ("voltage_noise", "voltage_noise", ["--voltage_noise_std", "0.01"]),
    ("temperature_noise", "temp_noise", ["--temp_noise_std", "1.0"]),
    ("current_bias_0p5pct", "current_offset", ["--current_offset_pct", "0.005"]),
    ("current_bias_1p5pct", "current_offset", ["--current_offset_pct", "0.015"]),
    ("current_bias_3p0pct", "current_offset", ["--current_offset_pct", "0.030"]),
    ("voltage_offset", "voltage_offset", ["--voltage_offset_v", "0.02"]),
    ("temperature_offset", "temp_offset", ["--temp_offset_c", "3.0"]),
    ("adc_quantization", "adc_quantization", []),
    ("initial_soc_error", "initial_soc_error", ["--soc_init_error", "-0.10", "--warmup_seconds", "0"]),
    ("missing_samples_periodic", "missing_samples", ["--missing_samples_every", "50"]),
    ("missing_samples_random", "missing_samples", ["--missing_samples_pct", "0.02"]),
    ("irregular_sampling_0p1s", "irregular_sampling", ["--irregular_dt_jitter", "0.1"]),
    ("irregular_sampling_0p5s", "irregular_sampling", ["--irregular_dt_jitter", "0.5"]),
    ("irregular_sampling_0p9s", "irregular_sampling", ["--irregular_dt_jitter", "0.9"]),
    (
        "missing_gap_1h",
        "missing_gap",
        [
            "--missing_gap_seconds", "3600",
            "--missing_gap_placement", "max_abs_net_charge",
            "--missing_gap_min_pre_seconds", "43200",
            "--missing_gap_min_post_seconds", "86400",
        ],
    ),
    (
        "voltage_spikes",
        "spikes",
        ["--spike_channel", "Voltage[V]", "--spike_magnitude", "0.20", "--spike_period", "1000"],
    ),
]

STOCHASTIC_ALIASES = {
    "current_noise_low",
    "current_noise_high",
    "voltage_noise",
    "temperature_noise",
    "missing_samples_random",
    "irregular_sampling_0p1s",
    "irregular_sampling_0p5s",
    "irregular_sampling_0p9s",
    "voltage_spikes",
}

# Sensor-noise cases receive the larger repeat budget. The remaining random
# integrity cases are repeated with the secondary budget because each run also
# spans all available cell/SOH windows.
PRIMARY_STOCHASTIC_ALIASES = {
    "current_noise_low",
    "current_noise_high",
    "voltage_noise",
    "temperature_noise",
}

# Reference SOH is an explanatory ablation, not a second primary benchmark.
# These scenarios isolate the most plausible SOH-error propagation paths.
DEFAULT_REFERENCE_ALIASES = {
    "baseline",
    "missing_gap_baseline_48h",
    "current_noise_high",
    "voltage_noise",
    "temperature_noise",
    "current_bias_3p0pct",
    "missing_samples_random",
    "missing_gap_1h",
}

# The same -0.10 SOC initialization mismatch is mapped into each estimator's
# available online state. For DD this is the equivalent Q_c offset; its realized
# output mismatch is reported explicitly because Q_c passes through the GRU.
INITIAL_STATE_APPLICABLE_MODELS = {"DM", "HDM", "HECM", "DD"}

MODEL_ORDER = ["DM", "HDM", "HECM", "DD"]
MODEL_LABELS = {
    "DM": "Direct measurement",
    "HDM": "Hybrid direct measurement",
    "HECM": "Hybrid ECM",
    "DD": "Data-driven",
}

SCENARIO_LABELS = {
    "baseline": "Baseline",
    "missing_gap_baseline_48h": "Burst-dropout duration reference (48 h)",
    "current_noise_low": "Current noise (0.02 A)",
    "current_noise_high": "Current noise (0.10 A)",
    "voltage_noise": "Voltage noise (0.01 V)",
    "temperature_noise": "Temperature noise (1.0 degC)",
    "current_bias_0p5pct": "Current bias (0.5%)",
    "current_bias_1p5pct": "Current bias (1.5%)",
    "current_bias_3p0pct": "Current bias (3.0%)",
    "voltage_offset": "Voltage offset (0.02 V)",
    "temperature_offset": "Temperature offset (3 degC)",
    "adc_quantization": "ADC quantization",
    "initial_soc_error": "Initial SOC error (-10%)",
    "missing_samples_periodic": "Periodic missing samples (1/50)",
    "missing_samples_random": "Random missing samples (2%)",
    "irregular_sampling_0p1s": "Timing jitter (+/-0.1 s)",
    "irregular_sampling_0p5s": "Timing jitter (+/-0.5 s)",
    "irregular_sampling_0p9s": "Timing jitter (+/-0.9 s)",
    "missing_gap_1h": "Burst dropout (1 h)",
    "voltage_spikes": "Voltage spikes (+/-0.20 V)",
}
