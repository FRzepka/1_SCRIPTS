$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
$pkg = Join-Path $repoRoot 'DL_Models\LFP_LSTM_MLP\2_models\quantized\manual_int8_lstm'
$inc = Join-Path $PSScriptRoot 'Core\Inc'

Copy-Item -Force (Join-Path $pkg 'model_weights_lstm_int8_manual.h') $inc
Copy-Item -Force (Join-Path $pkg 'mlp_weights_fp32.h') $inc
Write-Host 'Copied manual INT8 headers into Core/Inc'
