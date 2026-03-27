#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Plot complete PC vs STM32 SOH comparison (all data, 0 to end).

.PARAMETER Csv
    Path to pc_vs_stm32_soh_full.csv (required)

.EXAMPLE
    .\run_plot_complete.ps1 --csv "path\to\pc_vs_stm32_soh_full.csv"
#>

param(
    [Parameter(Mandatory=$true)]
    [string]$Csv,
    
    [string]$OutDir = ""
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
$CondaArgs += Join-Path $ScriptDir "plot_pc_vs_stm32_complete.py"
$CondaArgs += "--csv", $Csv

if ($OutDir) {
    $CondaArgs += "--out-dir", $OutDir
}

Write-Host ""
Write-Host "Plotting complete PC vs STM32 SOH comparison..." -ForegroundColor Green
Write-Host "CSV: $Csv" -ForegroundColor Cyan
Write-Host ""

& $CondaExe $CondaArgs

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "ERROR: Plotting failed with exit code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "SUCCESS! Check output directory for results." -ForegroundColor Green
