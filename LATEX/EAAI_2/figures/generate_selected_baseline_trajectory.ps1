$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$figuresDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = (Resolve-Path (Join-Path $figuresDir '..\..\..')).Path
$resultsDir = Join-Path $repoRoot 'DL_Models\LFP_SOH_Optimization_Study\5_benchmark\batmm\LFP_SOH_Optimization_Study\5_benchmark\Stateful_Base_Comparison\results'
$inputPath = Join-Path $resultsDir 'soh_trajectory_C11.csv'
$outputPath = Join-Path $figuresDir 'selected_baseline_soh_trajectory.png'
$culture = [Globalization.CultureInfo]::InvariantCulture

$models = @(
    [pscustomobject]@{ Name = 'CNN';  Column = 'soh_cnn';  Color = '#59C7C2' },
    [pscustomobject]@{ Name = 'GRU';  Column = 'soh_gru';  Color = '#59E83A' },
    [pscustomobject]@{ Name = 'LSTM'; Column = 'soh_lstm'; Color = '#E76B91' },
    [pscustomobject]@{ Name = 'TCN';  Column = 'soh_tcn';  Color = '#294862' }
)

function Get-Rgb([string]$hex) {
    $value = $hex.TrimStart('#')
    return @(
        [Convert]::ToInt32($value.Substring(0, 2), 16),
        [Convert]::ToInt32($value.Substring(2, 2), 16),
        [Convert]::ToInt32($value.Substring(4, 2), 16)
    )
}

function Get-CenteredRollingMedian([double[]]$values, [int]$window = 7) {
    $radius = [int][Math]::Floor($window / 2)
    $output = New-Object double[] $values.Count
    for ($index = 0; $index -lt $values.Count; $index++) {
        $start = [Math]::Max(0, $index - $radius)
        $stop = [Math]::Min($values.Count - 1, $index + $radius)
        $segment = [double[]]$values[$start..$stop]
        [Array]::Sort($segment)
        $middle = [int][Math]::Floor($segment.Count / 2)
        $output[$index] = if (($segment.Count % 2) -eq 1) {
            $segment[$middle]
        } else {
            0.5 * ($segment[$middle - 1] + $segment[$middle])
        }
    }
    return $output
}

$rows = @(Import-Csv -LiteralPath $inputPath)
$sampleCount = $rows.Count
$reference = [double[]]($rows | ForEach-Object {
    [double]::Parse($_.soh_reference, $culture)
})
$traces = @{}
foreach ($model in $models) {
    $column = $model.Column
    $raw = [double[]]($rows | ForEach-Object {
        [double]::Parse($_.$column, $culture)
    })
    $traces[$model.Name] = Get-CenteredRollingMedian $raw
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

function Convert-TraceToPoints([double[]]$trace, [int]$step = 3) {
    $pointCount = [int][Math]::Floor(($trace.Count - 1) / $step) + 1
    $points = New-Object System.Drawing.PointF[] $pointCount
    $pointIndex = 0
    for ($index = 0; $index -lt $trace.Count; $index += $step) {
        $points[$pointIndex] = New-Object System.Drawing.PointF((Get-PlotX $index), (Get-PlotY $trace[$index]))
        $pointIndex++
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
        $label = $tick.ToString('0.00', $culture)
        $size = $graphics.MeasureString($label, $fontTick)
        $graphics.DrawString($label, $fontTick, $textBrush, $left - $size.Width - 20, $y - $size.Height / 2)
    }
    foreach ($tick in @(0, 500, 1000, 1500, 2000, 2500)) {
        $x = Get-PlotX $tick
        $graphics.DrawLine($gridPen, $x, $top, $x, $top + $plotHeight)
        $label = $tick.ToString($culture)
        $size = $graphics.MeasureString($label, $fontTick)
        $graphics.DrawString($label, $fontTick, $textBrush, $x - $size.Width / 2, $top + $plotHeight + 18)
    }

    $graphics.DrawLine($axisPen, $left, $top, $left, $top + $plotHeight)
    $graphics.DrawLine($axisPen, $left, $top + $plotHeight, $left + $plotWidth, $top + $plotHeight)

    foreach ($model in $models) {
        $rgb = Get-Rgb $model.Color
        $pen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb($rgb[0], $rgb[1], $rgb[2]), 4)
        try {
            $graphics.DrawLines($pen, (Convert-TraceToPoints $traces[$model.Name]))
        } finally {
            $pen.Dispose()
        }
    }
    $referencePen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(25, 25, 25), 6)
    try {
        $graphics.DrawLines($referencePen, (Convert-TraceToPoints $reference))
    } finally {
        $referencePen.Dispose()
    }

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
    for ($index = 0; $index -lt $entries.Count; $index++) {
        $entry = $entries[$index]
        $entryRgb = Get-Rgb $entry.Color
        $entryPen = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb($entryRgb[0], $entryRgb[1], $entryRgb[2]), $entry.Width)
        $entryY = $legendY + 29 + $index * 39
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
