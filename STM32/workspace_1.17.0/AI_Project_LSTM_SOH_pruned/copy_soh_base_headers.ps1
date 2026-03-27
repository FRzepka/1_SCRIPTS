Continue = 'Stop'
 = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent ))
 = Join-Path  'DL_Models\LFP_LSTM_MLP\2_models\base\SOH_c_implementation'
 = Join-Path  'Core\Inc'
Copy-Item -Force (Join-Path  'lstm_model_soh.h') 
Copy-Item -Force (Join-Path  'model_weights_soh.h') 
Copy-Item -Force (Join-Path  'scaler_params_soh.h') 
Write-Host 'Copied SOH base headers into Core/Inc'
