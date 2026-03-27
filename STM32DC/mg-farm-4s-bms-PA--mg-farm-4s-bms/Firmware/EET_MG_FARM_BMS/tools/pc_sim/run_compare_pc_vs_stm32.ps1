#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Compare PC-based SOH inference vs STM32 hardware predictions.

.DESCRIPTION
    Wrapper for compare_pc_vs_stm32_soh.py that:
    1. Activates the ml1 conda environment
    2. Runs PC inference with quantized LSTM model
    3. Compares against STM32 hardware predictions
    4. Generates comparison plots and metrics

.PARAMETER Csv
    Path to stm32_hw_c11.csv file (required)

.PARAMETER ModelDir
    Path to model directory (default: auto-detect)

.PARAMETER SeqLen
    LSTM sequence length (default: 16)

.PARAMETER OutDir
    Output directory (default: same as CSV)

.PARAMETER NoPlot
    Skip plotting

.EXAMPLE
    .\run_compare_pc_vs_stm32.ps1 --csv "C:\...\HW_C11_20260120_105233\stm32_hw_c11.csv"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Csv,
    
    [string]$ModelDir = "",
    [int]$SeqLen = 168,
    [string]$OutDir = "",
    [switch]$NoPlot
)

$ErrorActionPreference = "Stop"

# Resolve script directory
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# Find conda
$CondaExe = $null
$CondaPaths = @(
    "$env:USERPROFILE\miniconda3\Scripts\conda.exe",
    "$env:USERPROFILE\anaconda3\Scripts\conda.exe",
    "C:\ProgramData\miniconda3\Scripts\conda.exe",
    "C:\ProgramData\anaconda3\Scripts\conda.exe"
)
foreach ($p in $CondaPaths) {
    if (Test-Path $p) {
        $CondaExe = $p
        break
    }
}
if (-not $CondaExe) {
    $CondaExe = (Get-Command conda -ErrorAction SilentlyContinue).Source
}
if (-not $CondaExe) {
    Write-Host "ERROR: conda not found. Install miniconda or anaconda." -ForegroundColor Red
    exit 1
}

Write-Host "Using conda: $CondaExe" -ForegroundColor Cyan

# Build conda run command
$CondaArgs = @("run", "-n", "ml1", "--no-capture-output", "python")
$CondaArgs += Join-Path $ScriptDir "compare_pc_vs_stm32_soh.py"
$CondaArgs += "--csv", $Csv

if ($ModelDir) {
    $CondaArgs += "--model-dir", $ModelDir
}
if ($SeqLen -ne 168) {
    $CondaArgs += "--seq-len", $SeqLen
}
if ($OutDir) {
    $CondaArgs += "--out-dir", $OutDir
}
if ($NoPlot) {
    $CondaArgs += "--no-plot"
}

Write-Host ""
Write-Host "Running PC vs STM32 SOH comparison..." -ForegroundColor Green
Write-Host "CSV: $Csv" -ForegroundColor Cyan
Write-Host ""

& $CondaExe $CondaArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Comparison failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "SUCCESS! Check output directory for results." -ForegroundColor Green
