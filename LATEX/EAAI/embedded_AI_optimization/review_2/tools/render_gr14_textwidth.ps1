$ErrorActionPreference = 'Stop'
$review = Split-Path $PSScriptRoot -Parent
$source = Join-Path (Split-Path $review -Parent) 'gr14.pdf'
$outputStem = Join-Path $review 'gr14_textwidth'
$sourceHash = (Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash

& pdftoppm -f 1 -singlefile -r 300 -png $source $outputStem
if ($LASTEXITCODE -ne 0) { throw 'PDF-to-PNG rendering failed.' }
if ((Get-FileHash -LiteralPath $source -Algorithm SHA256).Hash -ne $sourceHash) {
    throw 'Source PDF changed during conversion.'
}

Add-Type -AssemblyName System.Drawing
$png = [System.Drawing.Image]::FromFile("$outputStem.png")
try {
    $record = [ordered]@{
        revision_date = '2026-10-02'
        source_pdf = $source
        source_sha256 = $sourceHash.ToLowerInvariant()
        output_png = "$outputStem.png"
        output_sha256 = (Get-FileHash -LiteralPath "$outputStem.png" -Algorithm SHA256).Hash.ToLowerInvariant()
        width_px = $png.Width
        height_px = $png.Height
        rendering_dpi = 300
        conversion = 'Direct Poppler rendering of the specified parent-folder PDF. No artwork or manuscript changes.'
    }
} finally {
    $png.Dispose()
}
$record | ConvertTo-Json | Set-Content -LiteralPath "$outputStem.json" -Encoding UTF8
$record | ConvertTo-Json
