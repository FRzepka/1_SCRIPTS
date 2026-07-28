param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot),
    [string]$OutputZip = (Join-Path (Split-Path -Parent $PSScriptRoot) 'diss_overleaf_upload_260728.zip')
)

$ErrorActionPreference = 'Stop'

$project = (Resolve-Path -LiteralPath $ProjectRoot).Path
$mainTex = Join-Path $project 'main.tex'
$bibliography = Join-Path $project 'bib\Dissertation.bib'
$stage = Join-Path ([System.IO.Path]::GetTempPath()) ('diss_overleaf_' + [guid]::NewGuid().ToString('N'))
$graphicRoots = @(
    'pictures\eaai_palette',
    'pictures'
)
$graphicExtensions = @('', '.png', '.pdf', '.jpg', '.jpeg')

try {
    New-Item -ItemType Directory -Path $stage | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $stage 'bib') | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $stage 'pictures\eaai_palette') | Out-Null
    New-Item -ItemType Directory -Path (Join-Path $stage 'pictures\schematics') | Out-Null

    Copy-Item -LiteralPath $mainTex -Destination (Join-Path $stage 'main.tex')
    Copy-Item -LiteralPath $bibliography -Destination (Join-Path $stage 'bib\Dissertation.bib')

    $tex = Get-Content -LiteralPath $mainTex -Raw
    $graphics = [regex]::Matches(
        $tex,
        '\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}'
    ) | ForEach-Object {
        $_.Groups[1].Value
    } | Sort-Object -Unique

    $copiedGraphics = 0
    foreach ($graphic in $graphics) {
        $source = $null
        $relativeSource = $null

        foreach ($root in $graphicRoots) {
            foreach ($extension in $graphicExtensions) {
                $candidateRelative = Join-Path $root ($graphic + $extension)
                $candidate = Join-Path $project $candidateRelative
                if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                    $source = $candidate
                    $relativeSource = $candidateRelative
                    break
                }
            }
            if ($source) {
                break
            }
        }

        if (!$source) {
            throw "Referenced graphic not found: $graphic"
        }

        $destination = Join-Path $stage $relativeSource
        $destinationDirectory = Split-Path -Parent $destination
        if (!(Test-Path -LiteralPath $destinationDirectory)) {
            New-Item -ItemType Directory -Path $destinationDirectory | Out-Null
        }
        Copy-Item -LiteralPath $source -Destination $destination
        $copiedGraphics++
    }

    $svgSources = Get-ChildItem -LiteralPath (Join-Path $project 'pictures\schematics') -Filter '*.svg' -File
    foreach ($svg in $svgSources) {
        Copy-Item -LiteralPath $svg.FullName -Destination (Join-Path $stage 'pictures\schematics')
    }

    if (Test-Path -LiteralPath $OutputZip) {
        [System.IO.File]::Delete([System.IO.Path]::GetFullPath($OutputZip))
    }
    Compress-Archive -Path (Join-Path $stage '*') -DestinationPath $OutputZip -CompressionLevel Optimal

    $zip = Get-Item -LiteralPath $OutputZip
    Write-Output "Created: $($zip.FullName)"
    Write-Output "Raster graphics: $copiedGraphics"
    Write-Output "SVG sources: $($svgSources.Count)"
    Write-Output "ZIP size: $([math]::Round($zip.Length / 1MB, 2)) MB"
}
finally {
    if (Test-Path -LiteralPath $stage) {
        [System.IO.Directory]::Delete($stage, $true)
    }
}
