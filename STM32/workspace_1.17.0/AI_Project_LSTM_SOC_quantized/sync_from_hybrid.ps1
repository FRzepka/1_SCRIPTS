$ErrorActionPreference = 'Stop'

# Paths
$dst = Join-Path $PSScriptRoot '.'
$src = Join-Path (Split-Path $PSScriptRoot -Parent) 'AI_Project_LSTM_hybrid_int8'

if (!(Test-Path $src)) { throw "Hybrid project not found: $src" }

# Relative paths to KEEP in dst (do not overwrite)
$keep = @(
  'Core/Src/lstm_model_int8.c',
  'Core/Src/main.c',
  'Core/Inc/model_weights_lstm_int8_manual.h',
  'Core/Inc/mlp_weights_fp32.h',
  'Core/Inc/scaler_params.h'
)

function Should-Skip($rel) {
  foreach ($k in $keep) {
    if ((($rel -replace '\\','/')) -ieq $k) { return $true }
  }
  return $false
}

Write-Host "Sync from: $src"
Write-Host "        to: $dst"

# Create directories
Get-ChildItem -Path $src -Recurse -Directory | ForEach-Object {
  $rel = $_.FullName.Substring($src.Length).TrimStart('\','/')
  $tgt = Join-Path $dst $rel
  if (!(Test-Path $tgt)) { New-Item -ItemType Directory -Force -Path $tgt | Out-Null }
}

# Copy files
Get-ChildItem -Path $src -Recurse -File | ForEach-Object {
  $rel = $_.FullName.Substring($src.Length).TrimStart('\','/')
  if (Should-Skip $rel) { return }
  $tgt = Join-Path $dst $rel
  Copy-Item -Force $_.FullName $tgt
}

Write-Host "Sync complete. Preserved custom INT8 files in:`n  $($keep -join "`n  ")"
