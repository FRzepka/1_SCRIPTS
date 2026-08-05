param(
    [string]$ProjectRoot = (Split-Path -Parent $PSScriptRoot)
)

$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $edge)) {
    throw "Microsoft Edge was not found at $edge"
}

$sourceDirectory = Join-Path $ProjectRoot 'pictures\schematics'
$outputDirectory = Join-Path $ProjectRoot 'pictures\eaai_palette'
$items = @(
    'robustness_dd_architecture',
    'embedded_architecture'
)

foreach ($name in $items) {
    $source = (Join-Path $sourceDirectory "$name.svg").Replace('\', '/')
    $output = Join-Path $outputDirectory "$name.png"
    & $edge `
        --headless `
        --disable-gpu `
        --hide-scrollbars `
        --no-first-run `
        --force-device-scale-factor=1 `
        --window-size=1700,900 `
        --screenshot="$output" `
        "file:///$source" | Out-Null

    if ($LASTEXITCODE -ne 0) {
        throw "Rendering failed for $name.svg"
    }
}

Get-Item -LiteralPath ($items | ForEach-Object {
    Join-Path $outputDirectory "$_.png"
})
