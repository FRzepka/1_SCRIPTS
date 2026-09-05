param(
    [string]$ReviewRoot = ''
)

$ErrorActionPreference = 'Stop'

if (-not $ReviewRoot) {
    $ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
} else {
    $ReviewRoot = (Resolve-Path $ReviewRoot).Path
}

$SourceTex = Join-Path $ReviewRoot 'Embedded_Ai_Manuscript_Anonymized.tex'
$SourceBbl = Join-Path $ReviewRoot 'Embedded_Ai_Manuscript_Anonymized.bbl'
$SourceBib = Join-Path $ReviewRoot 'bib\paper2_socsoh_fr.bib'
$SourceBst = Join-Path $ReviewRoot 'elsarticle-num.bst'
$UploadRoot = Join-Path $ReviewRoot 'EAAI_Upload'
$SourceCls = Join-Path $UploadRoot 'LaTeX_Source\elsarticle.cls'
$OutputZip = Join-Path $UploadRoot 'LaTeX_Source.zip'
$OutputTex = Join-Path $UploadRoot 'Embedded_Ai_Manuscript_Anonymized.tex'

foreach ($required in @($SourceTex, $SourceBbl, $SourceBib, $SourceBst, $SourceCls)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Required source file is missing: $required"
    }
}

$StageRoot = Join-Path $env:TEMP ('eaai_flat_submission_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $StageRoot | Out-Null

$tex = [System.IO.File]::ReadAllText($SourceTex)
$figurePattern = '\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
$figurePaths = @(
    [regex]::Matches($tex, $figurePattern) |
        ForEach-Object { $_.Groups[1].Value } |
        Where-Object { $_ -notmatch '#' } |
        Sort-Object -Unique
)

$basenames = @{}
foreach ($relativePath in $figurePaths) {
    $sourcePath = Join-Path $ReviewRoot ($relativePath -replace '/', '\')
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Referenced figure is missing: $relativePath"
    }

    $basename = [System.IO.Path]::GetFileName($relativePath)
    if ($basenames.ContainsKey($basename) -and $basenames[$basename] -ne $sourcePath) {
        throw "Duplicate figure basename prevents flat packaging: $basename"
    }
    $basenames[$basename] = $sourcePath
    $tex = $tex.Replace($relativePath, $basename)
    Copy-Item -LiteralPath $sourcePath -Destination (Join-Path $StageRoot $basename)
}

$tex = $tex.Replace('bib/paper2_socsoh_fr', 'paper2_socsoh_fr')
$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
$flatTex = Join-Path $StageRoot 'Embedded_Ai_Manuscript_Anonymized.tex'
[System.IO.File]::WriteAllText($flatTex, $tex, $utf8NoBom)

Copy-Item -LiteralPath $SourceBbl -Destination $StageRoot
Copy-Item -LiteralPath $SourceBib -Destination $StageRoot
Copy-Item -LiteralPath $SourceBst -Destination $StageRoot
Copy-Item -LiteralPath $SourceCls -Destination $StageRoot

Push-Location $StageRoot
try {
    & pdflatex -interaction=nonstopmode -halt-on-error 'Embedded_Ai_Manuscript_Anonymized.tex' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Flat source compilation failed on pass 1.' }
    & pdflatex -interaction=nonstopmode -halt-on-error 'Embedded_Ai_Manuscript_Anonymized.tex' | Out-Null
    if ($LASTEXITCODE -ne 0) { throw 'Flat source compilation failed on pass 2.' }
} finally {
    Pop-Location
}

$logPath = Join-Path $StageRoot 'Embedded_Ai_Manuscript_Anonymized.log'
$logIssues = Select-String -LiteralPath $logPath -Pattern 'LaTeX Error|undefined references|Citation.*undefined|Reference.*undefined'
if (@($logIssues).Count -gt 0) {
    throw 'Flat source compilation contains LaTeX, citation, or reference errors.'
}

Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$temporaryZip = Join-Path $env:TEMP ('LaTeX_Source_' + [guid]::NewGuid().ToString('N') + '.zip')
$archive = [System.IO.Compression.ZipFile]::Open(
    $temporaryZip,
    [System.IO.Compression.ZipArchiveMode]::Create
)
try {
    $packageFiles = @(
        Get-ChildItem -LiteralPath $StageRoot -File |
            Where-Object { $_.Extension -in @('.tex', '.bbl', '.bib', '.bst', '.cls', '.png') }
    )
    foreach ($file in $packageFiles) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $archive,
            $file.FullName,
            $file.Name,
            [System.IO.Compression.CompressionLevel]::Optimal
        ) | Out-Null
    }
} finally {
    $archive.Dispose()
}

$checkArchive = [System.IO.Compression.ZipFile]::OpenRead($temporaryZip)
try {
    $nestedEntries = @($checkArchive.Entries | Where-Object { $_.FullName -match '[/\\]' })
    if ($nestedEntries.Count -gt 0) {
        throw 'The generated ZIP still contains a directory structure.'
    }
    $entryCount = $checkArchive.Entries.Count
} finally {
    $checkArchive.Dispose()
}

Copy-Item -LiteralPath $temporaryZip -Destination $OutputZip -Force
Copy-Item -LiteralPath $flatTex -Destination $OutputTex -Force

Write-Host "Created flat LaTeX submission ZIP with $entryCount root-level files:"
Write-Host "  $OutputZip"
Write-Host "Validated figures: $($figurePaths.Count)"
