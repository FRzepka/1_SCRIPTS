param(
    [string]$Python = 'C:/Users/Florian/anaconda3/envs/ml1/python.exe'
)
$ErrorActionPreference = 'Stop'
$oldPath = $env:PATH
$prefix = Split-Path $Python -Parent
try {
    $env:PATH = "$prefix;$prefix/Library/bin;$prefix/Scripts;$oldPath"
    $arguments = @((Join-Path $PSScriptRoot 'build_times_new_roman.py'))
    & $Python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Plotting failed with exit code $LASTEXITCODE" }
} finally {
    $env:PATH = $oldPath
}
