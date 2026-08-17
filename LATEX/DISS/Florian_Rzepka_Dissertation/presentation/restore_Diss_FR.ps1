$ErrorActionPreference = 'Stop'

$presentationDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$partsDir = Join-Path $presentationDir 'Diss_FR.pptx.parts'
$outputPath = Join-Path $presentationDir 'Diss_FR.pptx'
$temporaryPath = Join-Path $presentationDir 'Diss_FR.pptx.restoring'
$expectedHash = 'C74F18B44F992C67A80FB70E6849D1A6267711C3A24F356A973D0260FFA18360'

if (-not (Test-Path -LiteralPath $partsDir -PathType Container)) {
    throw "Missing parts directory: $partsDir"
}

if (Test-Path -LiteralPath $outputPath -PathType Leaf) {
    $existingHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $outputPath).Hash
    if ($existingHash -eq $expectedHash) {
        Write-Host 'Diss_FR.pptx already exists and matches the stored SHA-256 hash.'
        exit 0
    }
    throw 'Diss_FR.pptx already exists with different content. It was not overwritten.'
}

$parts = @(Get-ChildItem -LiteralPath $partsDir -Filter 'Diss_FR.pptx.part*' -File | Sort-Object Name)
if ($parts.Count -eq 0) {
    throw "No presentation parts found in $partsDir"
}

$output = [System.IO.File]::Open($temporaryPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write)
try {
    foreach ($part in $parts) {
        $input = [System.IO.File]::OpenRead($part.FullName)
        try { $input.CopyTo($output) } finally { $input.Dispose() }
    }
} finally {
    $output.Dispose()
}

$restoredHash = (Get-FileHash -Algorithm SHA256 -LiteralPath $temporaryPath).Hash
if ($restoredHash -ne $expectedHash) {
    throw "Restored presentation failed SHA-256 verification: $restoredHash"
}

Move-Item -LiteralPath $temporaryPath -Destination $outputPath
Write-Host "Restored and verified: $outputPath"
