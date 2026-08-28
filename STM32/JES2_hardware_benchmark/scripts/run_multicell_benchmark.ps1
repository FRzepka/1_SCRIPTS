param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Port = "COM7",
    [int]$Rounds = 3
)

$ErrorActionPreference = "Stop"
$Models = @("DM", "HDM", "HECM", "DD")
if ($Models -notcontains $Model) {
    throw "Model must be one of: $($Models -join ', ')"
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Cells = @("C09", "C13", "C15", "C25", "C27", "C29")
foreach ($Cell in $Cells) {
    $Vectors = Join-Path $Root "test_vectors/multicell/jes2_nominal_${Cell}_vectors.csv"
    $OutDir = Join-Path $Root "results/$Cell/$Model"
    Write-Host "Running $Model on $Cell ($Rounds rounds)"
    python (Join-Path $Root "scripts/collect_serial_benchmark.py") `
        --port $Port `
        --model $Model `
        --vectors $Vectors `
        --out-dir $OutDir `
        --rounds $Rounds
}
