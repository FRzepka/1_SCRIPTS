param(
    [Parameter(Mandatory=$true)][string]$Model,
    [string]$Port = "COM7",
    [int]$Rounds = 3
)

$ErrorActionPreference = "Stop"
$Models = @("DM", "HDM", "HECM", "DD", "DDS", "DDP")
if ($Models -notcontains $Model) {
    throw "Model must be one of: $($Models -join ', ')"
}

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Cells = @("C09", "C13", "C15", "C25", "C27", "C29")
$ScratchBase = [IO.Path]::GetFullPath((Join-Path $env:TEMP "JES2_hardware_benchmark"))
New-Item -ItemType Directory -Path $ScratchBase -Force | Out-Null
foreach ($Cell in $Cells) {
    $Vectors = Join-Path $Root "test_vectors/multicell/jes2_nominal_${Cell}_vectors.csv"
    $OutDir = Join-Path $Root "results/$Cell/$Model"
    $ScratchDir = Join-Path $ScratchBase "$Model-$Cell-$([guid]::NewGuid().ToString('N'))"
    New-Item -ItemType Directory -Path $ScratchDir | Out-Null
    Write-Host "Running $Model on $Cell ($Rounds rounds)"
    try {
        python (Join-Path $Root "scripts/collect_serial_benchmark.py") `
            --port $Port `
            --model $Model `
            --vectors $Vectors `
            --out-dir $ScratchDir `
            --rounds $Rounds
        if ($LASTEXITCODE -ne 0) {
            throw "Benchmark failed for $Model on $Cell"
        }
        New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
        Get-ChildItem -LiteralPath $ScratchDir -File | Copy-Item -Destination $OutDir -Force
    }
    finally {
        $ResolvedScratch = [IO.Path]::GetFullPath($ScratchDir)
        if (-not $ResolvedScratch.StartsWith($ScratchBase + [IO.Path]::DirectorySeparatorChar)) {
            throw "Refusing to remove scratch directory outside $ScratchBase"
        }
        Remove-Item -LiteralPath $ResolvedScratch -Recurse -Force
    }
}
