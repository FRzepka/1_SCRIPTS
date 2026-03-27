param(
  [Parameter(Mandatory=$true)][string]$Port,
  [int]$Baud = 115200,
  [double]$Voltage = 14.8,
  [double]$Current = 0.1,
  [double]$Temp = 25.0,
  [double]$Rate = 2.0,
  [double]$Duration = 2.0
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts
$log = Join-Path $repoRoot "DL_Models\\LFP_SOH_Optimization_Study\\6_test\\STM32DC\\LSTM_0.1.2.3\\SMOKE_LOG.txt"

Push-Location $repoRoot

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logDir = Split-Path -Parent $log
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
"$ts smoke port=$Port baud=$Baud V=$Voltage I=$Current T=$Temp rate=$Rate duration=$Duration" | Out-File -FilePath $log -Append -Encoding utf8

# Prefer direct python from the known conda env to avoid conda plugin issues.
$py = "C:\\Users\\Florian\\anaconda3\\envs\\ml1\\python.exe"
$script = Join-Path $repoRoot "STM32DC\\mg-farm-4s-bms-PA--mg-farm-4s-bms\\Firmware\\EET_MG_FARM_BMS\\tools\\pc_sim\\send_sim_measurements.py"

if (Test-Path $py) {
  $out = & $py $script --port $Port --baud $Baud --voltage $Voltage --current $Current --temp $Temp --rate $Rate --duration $Duration --poll-est --soh-only 2>&1
  $out | ForEach-Object { $_ } | Out-File -FilePath $log -Append -Encoding utf8
  $out
} else {
  $env:CONDA_NO_PLUGINS = "true"
  $env:CONDA_REPORT_ERRORS = "false"
  $out = conda --no-plugins run -n ml python $script --port $Port --baud $Baud --voltage $Voltage --current $Current --temp $Temp --rate $Rate --duration $Duration --poll-est --soh-only 2>&1
  $out | ForEach-Object { $_ } | Out-File -FilePath $log -Append -Encoding utf8
  $out
}

Pop-Location
