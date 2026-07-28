$ErrorActionPreference = 'Stop'

$dissRoot = Split-Path -Parent $PSScriptRoot
$scriptsRoot = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $dissRoot))
$source = Join-Path $scriptsRoot 'DL_Models\LFP_SOC_SOH_Model\4_simulation_environment\results\paper_tables_v4\table_baseline.md'
$svgPath = Join-Path $dissRoot 'pictures\schematics\robustness_baseline.svg'
$pngPath = Join-Path $dissRoot 'pictures\eaai_palette\robustness_baseline.png'

$classOrder = @(
    'Direct measurement',
    'Hybrid direct measurement',
    'Hybrid ECM',
    'Data-driven'
)
$shortLabels = @{
    'Direct measurement' = 'DM'
    'Hybrid direct measurement' = 'HDM'
    'Hybrid ECM' = 'HECM'
    'Data-driven' = 'DD'
}
$colors = @{
    'DM' = @('#a6d7a6', '#2ca02c')
    'HDM' = @('#d2bfe3', '#9467bd')
    'HECM' = @('#a1c6e0', '#1f77b4')
    'DD' = @('#eea4a5', '#d62728')
}

$rowsByClass = @{}
Import-Csv -LiteralPath $source | ForEach-Object {
    $rowsByClass[$_.class] = $_
}

$width = 1600
$height = 600
$plotTop = 58
$plotHeight = 450
$plotWidth = 615
$panelLeft = @(108, 888)
$labelX = @(92, 872)
$metrics = @('mae', 'rmse')
$maxima = @(0.033, 0.042)
$ticks = @(
    @(0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030),
    @(0.000, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030, 0.035, 0.040)
)

$svg = [System.Collections.Generic.List[string]]::new()
$svg.Add('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="600" viewBox="0 0 1600 600">')
$svg.Add('<rect width="1600" height="600" fill="#ffffff"/>')
$svg.Add('<style>')
$svg.Add('text { font-family: Arial, Helvetica, sans-serif; fill: #000000; }')
$svg.Add('.panel { font-size: 27px; font-weight: 700; }')
$svg.Add('.axis-label { font-size: 25px; }')
$svg.Add('.tick { font-size: 18px; }')
$svg.Add('.model { font-size: 25px; }')
$svg.Add('.value { font-size: 22px; }')
$svg.Add('</style>')

for ($panel = 0; $panel -lt 2; $panel++) {
    $left = $panelLeft[$panel]
    $right = $left + $plotWidth
    $bottom = $plotTop + $plotHeight
    $metric = $metrics[$panel]
    $maximum = $maxima[$panel]
    $panelLabel = if ($panel -eq 0) { '(a)' } else { '(b)' }

    $svg.Add("<rect x=`"$left`" y=`"$plotTop`" width=`"$plotWidth`" height=`"$plotHeight`" fill=`"#fbfbfb`" stroke=`"#444444`" stroke-width=`"1.5`"/>")
    $svg.Add("<text class=`"panel`" x=`"$left`" y=`"39`">$panelLabel</text>")

    foreach ($tick in $ticks[$panel]) {
        $x = $left + $plotWidth * ($tick / $maximum)
        $tickLabel = $tick.ToString('0.000', [Globalization.CultureInfo]::InvariantCulture)
        $svg.Add("<line x1=`"$($x.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))`" y1=`"$plotTop`" x2=`"$($x.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))`" y2=`"$bottom`" stroke=`"#d5d5d5`" stroke-width=`"1.2`"/>")
        $svg.Add("<text class=`"tick`" x=`"$($x.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))`" y=`"538`" text-anchor=`"middle`">$tickLabel</text>")
    }

    for ($index = 0; $index -lt $classOrder.Count; $index++) {
        $className = $classOrder[$index]
        $short = $shortLabels[$className]
        $row = $rowsByClass[$className]
        $value = [double]::Parse($row.$metric, [Globalization.CultureInfo]::InvariantCulture)
        $barY = 82 + $index * 103
        $barHeight = 76
        $barWidth = $plotWidth * ($value / $maximum)
        $fill = $colors[$short][0]
        $edge = $colors[$short][1]
        $valueLabel = $value.ToString('0.0000', [Globalization.CultureInfo]::InvariantCulture)

        $svg.Add("<line x1=`"$left`" y1=`"$($barY + $barHeight / 2)`" x2=`"$right`" y2=`"$($barY + $barHeight / 2)`" stroke=`"#d5d5d5`" stroke-width=`"1.2`"/>")
        $svg.Add("<text class=`"model`" x=`"$($labelX[$panel])`" y=`"$($barY + 48)`" text-anchor=`"end`">$short</text>")
        $svg.Add("<rect x=`"$left`" y=`"$barY`" width=`"$($barWidth.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))`" height=`"$barHeight`" fill=`"$fill`" stroke=`"$edge`" stroke-width=`"1.7`"/>")

        if ($value -gt 0.75 * $maximum) {
            $textX = $left + $barWidth - 16
            $anchor = 'end'
        }
        else {
            $textX = $left + $barWidth + 14
            $anchor = 'start'
        }
        $svg.Add("<text class=`"value`" x=`"$($textX.ToString('0.0', [Globalization.CultureInfo]::InvariantCulture))`" y=`"$($barY + 48)`" text-anchor=`"$anchor`">$valueLabel</text>")
    }

    $axisLabel = $metric.ToUpperInvariant()
    $svg.Add("<text class=`"axis-label`" x=`"$($left + $plotWidth / 2)`" y=`"582`" text-anchor=`"middle`">$axisLabel</text>")
}

$svg.Add('</svg>')
[IO.Directory]::CreateDirectory((Split-Path -Parent $svgPath)) | Out-Null
[IO.Directory]::CreateDirectory((Split-Path -Parent $pngPath)) | Out-Null
[IO.File]::WriteAllText($svgPath, ($svg -join [Environment]::NewLine), [Text.UTF8Encoding]::new($false))

$edge = 'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
if (-not (Test-Path -LiteralPath $edge)) {
    $edge = 'C:\Program Files\Microsoft\Edge\Application\msedge.exe'
}
if (-not (Test-Path -LiteralPath $edge)) {
    throw 'Microsoft Edge is required to render the SVG to PNG.'
}

$tempProfile = Join-Path $env:TEMP 'codex-edge-baseline-render'
$svgUri = [Uri]::new($svgPath).AbsoluteUri
& $edge `
    --headless `
    --disable-gpu `
    --hide-scrollbars `
    --force-device-scale-factor=1 `
    --window-size="$width,$height" `
    --user-data-dir="$tempProfile" `
    --screenshot="$pngPath" `
    $svgUri | Out-Null

if (-not (Test-Path -LiteralPath $pngPath)) {
    throw "PNG rendering failed: $pngPath"
}
Write-Host "Saved $svgPath"
Write-Host "Saved $pngPath"
