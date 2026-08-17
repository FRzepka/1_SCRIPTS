$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$figuresDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $figuresDir '..\..\..')).Path
$archiveDir = Join-Path $repoRoot 'DL_Models\LFP_SOH_Optimization_Study\6_test\CURRENT_MODELS_BASE_VS_OPTIMIZED\base_vs_optimized'
$outputPath = Join-Path $figuresDir 'selected_baseline_soh_trajectory.png'

$models = @(
    [pscustomobject]@{ Name = 'CNN';  File = 'CNN_0.4.2.1_hp_MGFarm_18650_C11_base_vs_optimized.png'; Color = '#59C7C2' },
    [pscustomobject]@{ Name = 'GRU';  File = 'GRU_0.3.1.2_MGFarm_18650_C11_base_vs_optimized.png';    Color = '#59E83A' },
    [pscustomobject]@{ Name = 'LSTM'; File = 'LSTM_0.1.2.4_MGFarm_18650_C11_base_vs_optimized.png';   Color = '#E76B91' },
    [pscustomobject]@{ Name = 'TCN';  File = 'TCN_0.2.2.2_MGFarm_18650_C11_base_vs_optimized.png';    Color = '#294862' }
)

# Geometry of the archived 2100 x 750 Matplotlib figures.
$sourceLeft = 99.0
$sourceRight = 2078.0
$sourceTop = 52.0
$sourceBottom = 663.0
$sampleCount = 2716
$sourceXMin = -0.05 * ($sampleCount - 1)
$sourceXMax = 1.05 * ($sampleCount - 1)
$sampleStep = 3

function Get-Rgb([string]$hex) {
    $value = $hex.TrimStart('#')
    return @(
        [Convert]::ToInt32($value.Substring(0, 2), 16),
        [Convert]::ToInt32($value.Substring(2, 2), 16),
        [Convert]::ToInt32($value.Substring(4, 2), 16)
    )
}

function Get-Median([double[]]$values) {
    if ($values.Count -eq 0) { return [double]::NaN }
    $sorted = @($values | Sort-Object)
    $middle = [int][Math]::Floor($sorted.Count / 2)
    if (($sorted.Count % 2) -eq 1) { return [double]$sorted[$middle] }
    return 0.5 * ([double]$sorted[$middle - 1] + [double]$sorted[$middle])
}

function Fill-Missing([double[]]$values) {
    $valid = @()
    for ($i = 0; $i -lt $values.Count; $i++) {
        if (-not [double]::IsNaN($values[$i])) { $valid += $i }
    }
    if ($valid.Count -lt 2) { throw 'Too few trace pixels were recovered.' }

    $first = $valid[0]
    $last = $valid[-1]
    for ($i = 0; $i -lt $first; $i++) { $values[$i] = $values[$first] }
    for ($i = $last + 1; $i -lt $values.Count; $i++) { $values[$i] = $values[$last] }

    for ($segment = 0; $segment -lt $valid.Count - 1; $segment++) {
        $a = $valid[$segment]
        $b = $valid[$segment + 1]
        if (($b - $a) -le 1) { continue }
        for ($i = $a + 1; $i -lt $b; $i++) {
            $fraction = ($i - $a) / [double]($b - $a)
            $values[$i] = $values[$a] + $fraction * ($values[$b] - $values[$a])
        }
    }
    return $values
}

function Convert-YToSoh([double]$pixelY) {
    return ($sourceBottom - $pixelY) / ($sourceBottom - $sourceTop)
}

function Get-BlueTrace([System.Drawing.Bitmap]$bitmap) {
    $target = Get-Rgb '#1F77B4'
    $values = New-Object double[] ([Math]::Ceiling($sampleCount / [double]$sampleStep))
    for ($i = 0; $i -lt $values.Count; $i++) { $values[$i] = [double]::NaN }

    for ($point = 0; $point -lt $values.Count; $point++) {
        $sample = [Math]::Min($point * $sampleStep, $sampleCount - 1)
        $sourceX = $sourceLeft + (($sample - $sourceXMin) / ($sourceXMax - $sourceXMin)) * ($sourceRight - $sourceLeft)
        $candidateY = New-Object System.Collections.Generic.List[double]
        for ($x = [Math]::Max(0, [int][Math]::Round($sourceX) - 2); $x -le [Math]::Min($bitmap.Width - 1, [int][Math]::Round($sourceX) + 2); $x++) {
            for ($y = 54; $y -le 330; $y++) {
                # Ignore the blue sample line in the archived legend.
                if ($x -ge 1815 -and $y -le 175) { continue }
                $pixel = $bitmap.GetPixel($x, $y)
                $distance = [Math]::Sqrt(
                    [Math]::Pow($pixel.R - $target[0], 2) +
                    [Math]::Pow($pixel.G - $target[1], 2) +
                    [Math]::Pow($pixel.B - $target[2], 2)
                )
                if ($distance -le 72 -and $pixel.B -gt ($pixel.R + 28) -and $pixel.B -gt ($pixel.G + 12)) {
                    $candidateY.Add([double]$y)
                }
            }
        }
        if ($candidateY.Count -gt 0) {
            $values[$point] = Convert-YToSoh (Get-Median $candidateY.ToArray())
        }
    }
    return Fill-Missing $values
}

function Get-ReferenceTrace([System.Drawing.Bitmap]$bitmap) {
    $values = New-Object double[] ([Math]::Ceiling($sampleCount / [double]$sampleStep))
    for ($i = 0; $i -lt $values.Count; $i++) { $values[$i] = [double]::NaN }
    $previousY = 58.0

    for ($point = 0; $point -lt $values.Count; $point++) {
        $sample = [Math]::Min($point * $sampleStep, $sampleCount - 1)
        $sourceX = $sourceLeft + (($sample - $sourceXMin) / ($sourceXMax - $sourceXMin)) * ($sourceRight - $sourceLeft)
        $candidateY = New-Object System.Collections.Generic.List[double]
        for ($x = [Math]::Max(0, [int][Math]::Round($sourceX) - 2); $x -le [Math]::Min($bitmap.Width - 1, [int][Math]::Round($sourceX) + 2); $x++) {
            for ($y = 55; $y -le 330; $y++) {
                $pixel = $bitmap.GetPixel($x, $y)
                $neutral = [Math]::Max([Math]::Abs($pixel.R - $pixel.G), [Math]::Abs($pixel.G - $pixel.B))
                if ($pixel.R -ge 20 -and $pixel.R -le 105 -and $neutral -le 8) {
                    $candidateY.Add([double]$y)
                }
            }
        }
        if ($candidateY.Count -gt 0) {
            $nearest = $candidateY | Sort-Object { [Math]::Abs($_ - $previousY) } | Select-Object -First 1
            if ([Math]::Abs($nearest - $previousY) -le 14) {
                $previousY = [double]$nearest
                $values[$point] = Convert-YToSoh $previousY
            }
        }
    }
    return Fill-Missing $values
}

$traces = @{}
$reference = $null
foreach ($model in $models) {
    $inputPath = Join-Path $archiveDir $model.File
    if (-not (Test-Path -LiteralPath $inputPath)) { throw "Missing archived plot: $inputPath" }
    $bitmap = [System.Drawing.Bitmap]::FromFile($inputPath)
    try {
        if ($null -eq $reference) { $reference = Get-ReferenceTrace $bitmap }
        $traces[$model.Name] = Get-BlueTrace $bitmap
    } finally {
        $bitmap.Dispose()
    }
}

$width = 2400
$height = 1200
$left = 190
$right = 65
$top = 65
$bottom = 155
$plotWidth = $width - $left - $right
$plotHeight = $height - $top - $bottom
$yMin = 0.62
$yMax = 1.00

function Get-PlotX([double]$sample) {
    return [single]($left + ($sample / ($sampleCount - 1)) * $plotWidth)
}

function Get-PlotY([double]$soh) {
    return [single]($top + (($yMax - $soh) / ($yMax - $yMin)) * $plotHeight)
}

function Convert-TraceToPoints([double[]]$trace) {
    $points = New-Object System.Drawing.PointF[] $trace.Count
    for ($i = 0; $i -lt $trace.Count; $i++) {
        $sample = [Math]::Min($i * $sampleStep, $sampleCount - 1)
        $points[$i] = New-Object System.Drawing.PointF((Get-PlotX $sample), (Get-PlotY $trace[$i]))
    }
    return $points
}

$output = New-Object System.Drawing.Bitmap($width, $height)
$graphics = [System.Drawing.Graphics]::FromImage($output)
try {
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $fontTick = New-Object System.Drawing.Font('Arial', 25)
    $fontAxis = New-Object System.Drawing.Font('Arial', 31)
    $fontLegend = New-Object System.Drawing.Font('Arial', 25)
    $axisPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(34, 34, 34), 3)
    $gridPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(218, 221, 224), 2)
    $textBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(34, 34, 34))

    foreach ($tick in @(0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)) {
        $y = Get-PlotY $tick
        $graphics.DrawLine($gridPen, $left, $y, $left + $plotWidth, $y)
        $label = $tick.ToString('0.00', [Globalization.CultureInfo]::InvariantCulture)
        $size = $graphics.MeasureString($label, $fontTick)
        $graphics.DrawString($label, $fontTick, $textBrush, $left - $size.Width - 20, $y - $size.Height / 2)
    }
    foreach ($tick in @(0, 500, 1000, 1500, 2000, 2500)) {
        $x = Get-PlotX $tick
        $graphics.DrawLine($gridPen, $x, $top, $x, $top + $plotHeight)
        $label = $tick.ToString([Globalization.CultureInfo]::InvariantCulture)
        $size = $graphics.MeasureString($label, $fontTick)
        $graphics.DrawString($label, $fontTick, $textBrush, $x - $size.Width / 2, $top + $plotHeight + 18)
    }

    $graphics.DrawLine($axisPen, $left, $top, $left, $top + $plotHeight)
    $graphics.DrawLine($axisPen, $left, $top + $plotHeight, $left + $plotWidth, $top + $plotHeight)

    foreach ($model in $models) {
        $rgb = Get-Rgb $model.Color
        $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb($rgb[0], $rgb[1], $rgb[2]), 4)
        try { $graphics.DrawLines($pen, (Convert-TraceToPoints $traces[$model.Name])) } finally { $pen.Dispose() }
    }
    $referencePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(25, 25, 25), 6)
    $graphics.DrawLines($referencePen, (Convert-TraceToPoints $reference))

    $xLabel = 'Time [h]'
    $xLabelSize = $graphics.MeasureString($xLabel, $fontAxis)
    $graphics.DrawString($xLabel, $fontAxis, $textBrush, $left + ($plotWidth - $xLabelSize.Width) / 2, $height - 70)

    $state = $graphics.Save()
    $graphics.TranslateTransform(55, $top + ($plotHeight + 130) / 2)
    $graphics.RotateTransform(-90)
    $yLabel = 'SOH [0-1]'
    $yLabelSize = $graphics.MeasureString($yLabel, $fontAxis)
    $graphics.DrawString($yLabel, $fontAxis, $textBrush, -$yLabelSize.Width / 2, 0)
    $graphics.Restore($state)

    $legendX = $left + 45
    $legendY = $top + $plotHeight - 255
    $legendWidth = 310
    $legendHeight = 225
    $legendBrush = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(238, 255, 255, 255))
    $legendBorder = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(175, 175, 175), 2)
    $graphics.FillRectangle($legendBrush, $legendX, $legendY, $legendWidth, $legendHeight)
    $graphics.DrawRectangle($legendBorder, $legendX, $legendY, $legendWidth, $legendHeight)

    $entries = @([pscustomobject]@{ Name = 'Reference SOH'; Color = '#191919'; Width = 6 }) +
        @($models | ForEach-Object { [pscustomobject]@{ Name = $_.Name; Color = $_.Color; Width = 4 } })
    for ($i = 0; $i -lt $entries.Count; $i++) {
        $entry = $entries[$i]
        $entryRgb = Get-Rgb $entry.Color
        $entryPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb($entryRgb[0], $entryRgb[1], $entryRgb[2]), $entry.Width)
        $entryY = $legendY + 29 + $i * 39
        $graphics.DrawLine($entryPen, $legendX + 22, $entryY, $legendX + 92, $entryY)
        $graphics.DrawString($entry.Name, $fontLegend, $textBrush, $legendX + 110, $entryY - 18)
        $entryPen.Dispose()
    }

    $output.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $graphics.Dispose()
    $output.Dispose()
}

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length, LastWriteTime
