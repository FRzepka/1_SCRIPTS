$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$pkg = Join-Path $repoRoot 'DL_Models\LFP_LSTM_MLP\2_models\quantized\manual_int8_lstm_soh'
$inc = Join-Path $PSScriptRoot 'Core\Inc'

Copy-Item -Force (Join-Path $pkg 'model_weights_lstm_int8_manual_soh.h') $inc
Copy-Item -Force (Join-Path $pkg 'mlp_weights_fp32_soh.h') $inc
Copy-Item -Force (Join-Path $pkg 'scaler_params_soh.h') $inc
Copy-Item -Force (Join-Path $pkg 'lstm_model_lstm_int8_fp32mlp_soh.h') $inc
Write-Host 'Copied SOH manual INT8 headers into Core/Inc'
