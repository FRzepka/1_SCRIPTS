$ErrorActionPreference = 'Stop'
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$inventory = Get-Content -LiteralPath (Join-Path $root 'archive/inspection_v1_5/inventory.json') -Raw | ConvertFrom-Json
$output = Join-Path $root 'Bilder_Vortrag/v1_5_Original'
New-Item -ItemType Directory -Path $output -Force | Out-Null
$seen = @{}
$rows = @()
foreach ($slide in $inventory.Slides) {
    foreach ($picture in $slide.Pictures) {
        if (!$seen.ContainsKey($picture.Hash)) {
            $name = 'Folie_{0:D2}_{1}' -f [int]$slide.Slide, (Split-Path $picture.File -Leaf)
            $target = Join-Path $output $name
            Copy-Item -LiteralPath $picture.File -Destination $target -Force
            if ((Get-FileHash -LiteralPath $target).Hash -ne $picture.Hash) { throw 'Image copy mismatch.' }
            $seen[$picture.Hash] = $name
        }
        $rows += [pscustomobject]@{
            Slide = $slide.Slide
            Title = $slide.Title
            Image = $seen[$picture.Hash]
            SHA256 = $picture.Hash
            PowerPointCrop = $picture.Crop
        }
    }
}
$rows | Export-Csv -LiteralPath (Join-Path $output 'Bildzuordnung.csv') -NoTypeInformation -Encoding UTF8
Write-Output ("Preserved {0} distinct images across {1} picture placements." -f $seen.Count, $rows.Count)
