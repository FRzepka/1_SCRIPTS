param(
  [Parameter(Mandatory = $true)]
  [string]$Port,
  [int]$MaxPoints = 500000,
  [double]$RateHz = 50,
  [int]$Baud = 115200,
  [string]$OutDir = "",
  [switch]$NoPlot = $true
)

$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts
$script = Join-Path $here "hwtest_c11_playback.py"

Push-Location $repoRoot

$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_REPORT_ERRORS = "false"
$env:PIP_NO_INDEX = ""
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:GIT_HTTP_PROXY = ""
$env:GIT_HTTPS_PROXY = ""

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = Join-Path $repoRoot "DL_Models\\LFP_SOH_Optimization_Study\\6_test\\STM32DC\\LSTM_0.1.2.3"
$logRoot = Resolve-Path $logRoot
$stdoutLog = Join-Path $logRoot ("HW_C11_" + $ts + "_500k_stdout.log")
$stderrLog = Join-Path $logRoot ("HW_C11_" + $ts + "_500k_stderr.log")

$py = "C:\\Users\\Florian\\anaconda3\\envs\\ml\\python.exe"
if (!(Test-Path $py)) { throw "Missing python env: $py" }

$outRoot = Join-Path $logRoot "HW_C11_$ts"
if (!$OutDir) { $OutDir = $outRoot }

$pyArgs = @(
  $script,
  "--port", $Port,
  "--baud", $Baud,
  "--max-points", $MaxPoints,
  "--rate-hz", $RateHz,
  "--settle-ms", "30",
  "--ack-timeout-s", "0",
  "--poll-every", "3600",
  "--expected-ts-offset-ms", "-1000",
  "--wait-for-ts",
  "--max-wait-ms", "2000",
  "--poll-interval-ms", "5",
  "--flush-every", "500",
  "--progress-every", "5000",
  "--plot-max-points", "50000"
)

if ($NoPlot) {
  $pyArgs += "--no-plot"
}
$pyArgs += @("--out-dir", $OutDir)

Write-Host "Starting long run in background (rows=$MaxPoints). This can take a long time."
Write-Host "HW output dir: $OutDir"
Write-Host "Stdout log: $stdoutLog"
Write-Host "Stderr log: $stderrLog"

Start-Process `
  -FilePath $py `
  -ArgumentList $pyArgs `
  -WorkingDirectory $repoRoot `
  -RedirectStandardOutput $stdoutLog `
  -RedirectStandardError $stderrLog `
  -WindowStyle Minimized

Write-Host "Started. Monitor progress with:"
Write-Host ("  Get-Content -Path `"" + $stdoutLog + "`" -Wait")
Write-Host ("  Get-Content -Path `"" + $stderrLog + "`" -Wait")
Write-Host "Or watch HW progress.json:"
Write-Host ("  Get-Content -Path `"" + (Join-Path $OutDir "progress.json") + "`" -Wait")

Pop-Location
