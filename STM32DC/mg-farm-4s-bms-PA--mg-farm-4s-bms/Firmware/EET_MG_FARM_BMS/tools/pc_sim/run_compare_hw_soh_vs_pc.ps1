$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts

Push-Location $repoRoot

# Use the torch-capable env for compute, and the stable matplotlib env for plotting.
$pyTorch = "C:\\Users\\Florian\\anaconda3\\envs\\ml1\\python.exe"
$pyPlot  = "C:\\Users\\Florian\\anaconda3\\envs\\ml\\python.exe"
if (!(Test-Path $pyTorch)) { throw "Missing python env (torch): $pyTorch" }
if (!(Test-Path $pyPlot))  { throw "Missing python env (plot):  $pyPlot" }

$scriptCompute = Join-Path $repoRoot "STM32DC\\mg-farm-4s-bms-PA--mg-farm-4s-bms\\Firmware\\EET_MG_FARM_BMS\\tools\\pc_sim\\compare_hw_soh_vs_pc.py"
$scriptPlot = Join-Path $repoRoot "STM32DC\\mg-farm-4s-bms-PA--mg-farm-4s-bms\\Firmware\\EET_MG_FARM_BMS\\tools\\pc_sim\\plot_hw_soh_compare.py"

# Some users copy patterns like `script.ps1 -- --hw-dir ...` from bash conventions.
# PowerShell doesn't require that; strip standalone `--` so argparse doesn't treat it as end-of-options.
$forwardArgs = @()
foreach ($a in $args) {
  if ($a -ne "--") { $forwardArgs += $a }
}

# Compute `soh_pc` into `stm32_hw_c11_with_pc.csv`
& $pyTorch $scriptCompute @forwardArgs

# Find hw-dir for plotting (either passed explicitly or latest).
$hwDir = ""
for ($i=0; $i -lt $forwardArgs.Length; $i++) {
  if ($forwardArgs[$i] -eq "--hw-dir" -and ($i+1) -lt $forwardArgs.Length) { $hwDir = $forwardArgs[$i+1] }
}
if ($hwDir -eq "") {
  $outRoot = Join-Path $repoRoot "DL_Models\\LFP_SOH_Optimization_Study\\6_test\\STM32DC\\LSTM_0.1.2.3"
  $hwDir = (Get-ChildItem -Path $outRoot -Directory -Filter "HW_C11_*" | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}

# Plot PNG (matplotlib is stable in env `ml`)
& $pyPlot $scriptPlot --hw-dir $hwDir

Pop-Location
