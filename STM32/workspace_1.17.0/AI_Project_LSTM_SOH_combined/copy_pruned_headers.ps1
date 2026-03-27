$ErrorActionPreference = 'Stop'
param(
  [Parameter(Mandatory=$true)] [string]$HeaderPath
)

$dst = Join-Path $PSScriptRoot 'Core\Inc\model_weights.h'
Copy-Item -Force $HeaderPath $dst
Write-Host "Copied pruned header to $dst"


