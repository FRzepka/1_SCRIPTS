$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts
$script = Join-Path $here "hwtest_c11_playback.py"

# IMPORTANT: No automatic installs/downloads here.
# Use the existing conda env `ml` for HW playback (matplotlib works reliably there).

Push-Location $repoRoot

$env:CONDA_NO_PLUGINS = "true"
$env:CONDA_REPORT_ERRORS = "false"
$env:PIP_NO_INDEX = ""
$env:HTTP_PROXY = ""
$env:HTTPS_PROXY = ""
$env:ALL_PROXY = ""
$env:GIT_HTTP_PROXY = ""
$env:GIT_HTTPS_PROXY = ""

$py = "C:\\Users\\Florian\\anaconda3\\envs\\ml\\python.exe"
if (!(Test-Path $py)) { throw "Missing python env: $py" }
Write-Host "Running HW test via $py (pass args directly)..."

# Some users copy patterns like `script.ps1 -- --port COMx ...` from bash conventions.
# PowerShell doesn't require that; strip standalone `--` so argparse doesn't treat it as end-of-options.
$forwardArgs = @()
foreach ($a in $args) {
  if ($a -ne "--") { $forwardArgs += $a }
}

& $py $script @forwardArgs

Pop-Location
