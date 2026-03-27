$ErrorActionPreference = "Stop"

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Resolve-Path (Join-Path $here "..\\..\\..\\..\\..\\..")  # .../1_Scripts
$script = Join-Path $here "reconstruct_cycle_features.py"

Push-Location $repoRoot

# Use the stable matplotlib/pyarrow env.
$py = "C:\\Users\\Florian\\anaconda3\\envs\\ml\\python.exe"
if (!(Test-Path $py)) { throw "Missing python env: $py" }

# Strip standalone `--` (bash convention) so argparse doesn't treat it as end-of-options.
$forwardArgs = @()
foreach ($a in $args) {
  if ($a -ne "--") { $forwardArgs += $a }
}

& $py $script @forwardArgs

Pop-Location

