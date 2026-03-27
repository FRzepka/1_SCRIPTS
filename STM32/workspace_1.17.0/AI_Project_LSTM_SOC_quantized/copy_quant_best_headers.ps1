param()
$srcDir = "DL_Models\LFP_LSTM_MLP\2_models\quantized\SOH_Quantized\c_export"
$dstDir = "STM32\workspace_1.17.0\AI_Project_LSTM_quantized\Core\Inc"
$files = @("quant_params_soh.h","quant_weights_soh.h")
if (!(Test-Path $srcDir)) { Write-Error "Source not found: $srcDir"; exit 1 }
if (!(Test-Path $dstDir)) { Write-Error "Dest not found: $dstDir"; exit 1 }
foreach ($f in $files) {
  Copy-Item (Join-Path $srcDir $f) -Destination (Join-Path $dstDir $f) -Force
}
Write-Output "Copied: $($files -join ', ') to $dstDir"
