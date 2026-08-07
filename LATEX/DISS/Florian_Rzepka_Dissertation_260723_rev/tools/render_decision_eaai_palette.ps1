param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$scriptsRoot = [System.IO.Path]::GetFullPath((Join-Path $DissertationRoot '..\..\..'))
$tableDir = Join-Path $scriptsRoot 'DL_Models\LFP_SOC_SOH_Model\4_simulation_environment\results\paper_tables_v4'
$outDir = Join-Path $DissertationRoot 'pictures\eaai_palette'
$outFile = Join-Path $outDir 'robustness_decision.png'

$classOrder = @(
    'Direct measurement',
    'Hybrid direct measurement',
    'Hybrid ECM',
    'Data-driven'
)

$classShort = @{
    'Direct measurement' = 'DM'
    'Hybrid direct measurement' = 'HDM'
    'Hybrid ECM' = 'HECM'
    'Data-driven' = 'DD'
}

$robustnessScenarios = @(
    'Current noise (high)',
    'Current bias',
    'Irregular sampling',
    'Burst dropout',
    'Missing samples',
    'Voltage spikes',
    'Temperature noise',
    'Voltage noise'
)

$profileWeights = [ordered]@{
    'Accuracy-weighted' = @{ Accuracy = 0.60; Robustness = 0.20; Recovery = 0.20 }
    'Robustness-weighted' = @{ Accuracy = 0.20; Robustness = 0.60; Recovery = 0.20 }
    'Recovery-weighted' = @{ Accuracy = 0.20; Robustness = 0.20; Recovery = 0.60 }
}

$palette = [ordered]@{
    'DM' = @{
        Main = '#2CA02C'
        Fill = '#A6D7A6'
    }
    'HDM' = @{
        Main = '#9467BD'
        Fill = '#D2BFE3'
    }
    'HECM' = @{
        Main = '#1F77B4'
        Fill = '#A1C6E0'
    }
    'DD' = @{
        Main = '#D62728'
        Fill = '#EEA4A5'
    }
}

function Convert-ToDouble {
    param([Parameter(Mandatory)][string]$Value)

    if ($Value.Trim().ToLowerInvariant() -eq 'nan') {
        return [double]::NaN
    }

    return [double]::Parse(
        $Value,
        [System.Globalization.NumberStyles]::Float,
        [System.Globalization.CultureInfo]::InvariantCulture
    )
}

function Get-LowerBetterScores {
    param([Parameter(Mandatory)][double[]]$Values)

    $min = ($Values | Measure-Object -Minimum).Minimum
    $max = ($Values | Measure-Object -Maximum).Maximum
    if ([math]::Abs($max - $min) -lt 1e-12) {
        return @(foreach ($value in $Values) { 1.0 })
    }

    return @(foreach ($value in $Values) { ($max - $value) / ($max - $min) })
}

function Get-PenalizedLowerBetterScores {
    param([Parameter(Mandatory)][double[]]$Values)

    $finite = @($Values | Where-Object { -not [double]::IsNaN($_) -and -not [double]::IsInfinity($_) })
    if ($finite.Count -eq 0) {
        return @(foreach ($value in $Values) { 0.0 })
    }

    $min = ($finite | Measure-Object -Minimum).Minimum
    $max = ($finite | Measure-Object -Maximum).Maximum
    $penalty = [math]::Max($max * 1.25, $min + 1e-6)
    if ([math]::Abs($penalty - $min) -lt 1e-12) {
        return @(foreach ($value in $Values) { 1.0 })
    }

    return @(foreach ($value in $Values) {
        $filled = $value
        if ([double]::IsNaN($filled) -or [double]::IsInfinity($filled)) {
            $filled = $penalty
        }
        ($penalty - $filled) / ($penalty - $min)
    })
}

function Read-MarkdownTable {
    param([Parameter(Mandatory)][string]$Path)

    $headers = $null
    $rows = @()
    foreach ($line in Get-Content -LiteralPath $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed.StartsWith('|')) {
            continue
        }

        $rowText = $trimmed -replace '^\|', '' -replace '\|$', ''
        $cells = @($rowText -split '\|' | ForEach-Object { $_.Trim() })
        $separator = $true
        foreach ($cell in $cells) {
            if ($cell -notmatch '^[\s:\-]+$') {
                $separator = $false
                break
            }
        }
        if ($separator) {
            continue
        }

        if ($null -eq $headers) {
            $headers = $cells
            continue
        }

        $obj = [ordered]@{}
        for ($i = 0; $i -lt $headers.Count; $i++) {
            $obj[$headers[$i]] = $cells[$i]
        }
        $rows += [pscustomobject]$obj
    }

    return $rows
}

function New-ColorFromHex {
    param([Parameter(Mandatory)][string]$Hex)

    $clean = $Hex.TrimStart('#')
    return [System.Drawing.Color]::FromArgb(
        [Convert]::ToInt32($clean.Substring(0, 2), 16),
        [Convert]::ToInt32($clean.Substring(2, 2), 16),
        [Convert]::ToInt32($clean.Substring(4, 2), 16)
    )
}

function New-AlphaColor {
    param(
        [Parameter(Mandatory)][int]$Alpha,
        [Parameter(Mandatory)][System.Drawing.Color]$Color
    )

    return [System.Drawing.Color]::FromArgb($Alpha, $Color.R, $Color.G, $Color.B)
}

function New-PointF {
    param(
        [Parameter(Mandatory)][double]$X,
        [Parameter(Mandatory)][double]$Y
    )

    return [System.Drawing.PointF]::new([single]$X, [single]$Y)
}

function Draw-CenteredText {
    param(
        [Parameter(Mandatory)][System.Drawing.Graphics]$Graphics,
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][System.Drawing.Font]$Font,
        [Parameter(Mandatory)][System.Drawing.Brush]$Brush,
        [Parameter(Mandatory)][double]$X,
        [Parameter(Mandatory)][double]$Y
    )

    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = [System.Drawing.StringAlignment]::Center
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $Graphics.DrawString($Text, $Font, $Brush, [single]$X, [single]$Y, $format)
    $format.Dispose()
}

function Draw-RightText {
    param(
        [Parameter(Mandatory)][System.Drawing.Graphics]$Graphics,
        [Parameter(Mandatory)][string]$Text,
        [Parameter(Mandatory)][System.Drawing.Font]$Font,
        [Parameter(Mandatory)][System.Drawing.Brush]$Brush,
        [Parameter(Mandatory)][double]$X,
        [Parameter(Mandatory)][double]$Y
    )

    $format = [System.Drawing.StringFormat]::new()
    $format.Alignment = [System.Drawing.StringAlignment]::Far
    $format.LineAlignment = [System.Drawing.StringAlignment]::Center
    $Graphics.DrawString($Text, $Font, $Brush, [single]$X, [single]$Y, $format)
    $format.Dispose()
}

function Get-RadarPoint {
    param(
        [Parameter(Mandatory)][double]$CenterX,
        [Parameter(Mandatory)][double]$CenterY,
        [Parameter(Mandatory)][double]$Radius,
        [Parameter(Mandatory)][double]$Angle,
        [Parameter(Mandatory)][double]$Value
    )

    $r = $Radius * $Value
    return (New-PointF -X ($CenterX + $r * [math]::Sin($Angle)) -Y ($CenterY - $r * [math]::Cos($Angle)))
}

function Build-MetaScores {
    $baseline = Import-Csv -LiteralPath (Join-Path $tableDir 'table_baseline.md')
    $key = Import-Csv -LiteralPath (Join-Path $tableDir 'table_key_results.md')
    $localPath = Join-Path $tableDir 'table_local_behaviour.md'
    $local = Read-MarkdownTable -Path $localPath

    $accuracySums = @{}
    foreach ($class in $classOrder) {
        $accuracySums[$class] = 0.0
    }

    foreach ($metric in @('mae', 'rmse', 'p95_error')) {
        $values = @(
            foreach ($class in $classOrder) {
                $row = $baseline | Where-Object { $_.class -eq $class } | Select-Object -First 1
                Convert-ToDouble $row.$metric
            }
        )
        $scores = Get-LowerBetterScores -Values $values
        for ($i = 0; $i -lt $classOrder.Count; $i++) {
            $accuracySums[$classOrder[$i]] += $scores[$i]
        }
    }

    $robustnessSums = @{}
    foreach ($class in $classOrder) {
        $robustnessSums[$class] = 0.0
    }

    foreach ($scenario in $robustnessScenarios) {
        $values = @(
            foreach ($class in $classOrder) {
                $row = $key | Where-Object { $_.scenario_label -eq $scenario -and $_.class -eq $class } | Select-Object -First 1
                Convert-ToDouble $row.delta_mae
            }
        )
        $scores = Get-LowerBetterScores -Values $values
        for ($i = 0; $i -lt $classOrder.Count; $i++) {
            $robustnessSums[$classOrder[$i]] += $scores[$i]
        }
    }

    $recoveryValues = @(
        foreach ($class in $classOrder) {
            $row = $local | Where-Object {
                $_.class -eq $class -and $_.local_metric -eq 'recovery_time_to_baseline_band_strict_h'
            } | Select-Object -First 1
            Convert-ToDouble $row.value
        }
    )
    $recoveryScores = Get-PenalizedLowerBetterScores -Values $recoveryValues

    $metaScores = @()
    for ($i = 0; $i -lt $classOrder.Count; $i++) {
        $class = $classOrder[$i]
        $accuracy = $accuracySums[$class] / 3.0
        $robustness = $robustnessSums[$class] / [double]$robustnessScenarios.Count
        $recovery = $recoveryScores[$i]
        $model = $classShort[$class]
        $profiles = [ordered]@{}
        foreach ($profile in $profileWeights.Keys) {
            $weights = $profileWeights[$profile]
            $profiles[$profile] = (
                $accuracy * $weights.Accuracy +
                $robustness * $weights.Robustness +
                $recovery * $weights.Recovery
            )
        }

        $metaScores += [pscustomobject]@{
            Model = $model
            Class = $class
            Accuracy = $accuracy
            Robustness = $robustness
            Recovery = $recovery
            Profiles = $profiles
        }
    }

    return $metaScores
}

function Render-Figure {
    param([Parameter(Mandatory)][object[]]$MetaScores)

    New-Item -ItemType Directory -Path $outDir -Force | Out-Null

    $width = 3317
    $height = 1540
    $bmp = [System.Drawing.Bitmap]::new($width, $height, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $graphics = [System.Drawing.Graphics]::FromImage($bmp)

    $pens = New-Object System.Collections.Generic.List[System.IDisposable]
    $brushes = New-Object System.Collections.Generic.List[System.IDisposable]
    $fonts = New-Object System.Collections.Generic.List[System.IDisposable]

    try {
        $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
        $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
        $graphics.Clear([System.Drawing.Color]::White)

        $fontTitle = [System.Drawing.Font]::new('Arial', 31, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
        $fontLabel = [System.Drawing.Font]::new('Arial', 29, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        $fontSmall = [System.Drawing.Font]::new('Arial', 22, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        $fontAxis = [System.Drawing.Font]::new('Arial', 26, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        $fontLegend = [System.Drawing.Font]::new('Arial', 27, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        $fontYLabel = [System.Drawing.Font]::new('Arial', 30, [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
        foreach ($font in @($fontTitle, $fontLabel, $fontSmall, $fontAxis, $fontLegend, $fontYLabel)) {
            $fonts.Add($font) | Out-Null
        }

        $brushText = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(32, 32, 32))
        $brushMuted = [System.Drawing.SolidBrush]::new([System.Drawing.Color]::FromArgb(90, 90, 90))
        $brushes.Add($brushText) | Out-Null
        $brushes.Add($brushMuted) | Out-Null

        $penGrid = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(218, 218, 218), 2.0)
        $penAxis = [System.Drawing.Pen]::new([System.Drawing.Color]::FromArgb(96, 96, 96), 2.2)
        $pens.Add($penGrid) | Out-Null
        $pens.Add($penAxis) | Out-Null

        Draw-CenteredText -Graphics $graphics -Text '(a) Relative decision dimensions' -Font $fontTitle -Brush $brushText -X 680 -Y 118
        Draw-CenteredText -Graphics $graphics -Text '(b) Priority-weighted composite scores' -Font $fontTitle -Brush $brushText -X 2350 -Y 118

        $radarCenterX = 690.0
        $radarCenterY = 802.0
        $radarRadius = 455.0
        $radarLabels = @('Accuracy', 'Robustness', 'Recovery')
        $radarAngles = @(0.0, (2.0 * [math]::PI / 3.0), (4.0 * [math]::PI / 3.0))

        foreach ($tick in @(0.25, 0.50, 0.75, 1.00)) {
            $ringRadius = [single]($radarRadius * $tick)
            $rect = [System.Drawing.RectangleF]::new(
                [single]($radarCenterX - $ringRadius),
                [single]($radarCenterY - $ringRadius),
                [single](2.0 * $ringRadius),
                [single](2.0 * $ringRadius)
            )
            $graphics.DrawEllipse($penGrid, $rect)
            Draw-CenteredText -Graphics $graphics -Text ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.00}', $tick)) -Font $fontSmall -Brush $brushMuted -X ($radarCenterX + 30) -Y ($radarCenterY - $ringRadius)
        }

        for ($i = 0; $i -lt $radarAngles.Count; $i++) {
            $endPoint = Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[$i] -Value 1.0
            $graphics.DrawLine($penGrid, (New-PointF -X $radarCenterX -Y $radarCenterY), $endPoint)

            $labelPoint = Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius ($radarRadius + 88.0) -Angle $radarAngles[$i] -Value 1.0
            Draw-CenteredText -Graphics $graphics -Text $radarLabels[$i] -Font $fontLabel -Brush $brushText -X $labelPoint.X -Y $labelPoint.Y
        }

        foreach ($row in $MetaScores) {
            $main = New-ColorFromHex $palette[$row.Model].Main
            $fill = New-ColorFromHex $palette[$row.Model].Fill
            $brush = [System.Drawing.SolidBrush]::new((New-AlphaColor -Alpha 105 -Color $fill))
            $brushes.Add($brush) | Out-Null
            $points = @(
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[0] -Value $row.Accuracy
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[1] -Value $row.Robustness
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[2] -Value $row.Recovery
            )
            $graphics.FillPolygon($brush, [System.Drawing.PointF[]]$points)
        }

        foreach ($row in $MetaScores) {
            $main = New-ColorFromHex $palette[$row.Model].Main
            $pen = [System.Drawing.Pen]::new($main, 5.0)
            $pen.LineJoin = [System.Drawing.Drawing2D.LineJoin]::Round
            $pens.Add($pen) | Out-Null
            $brush = [System.Drawing.SolidBrush]::new($main)
            $brushes.Add($brush) | Out-Null
            $points = @(
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[0] -Value $row.Accuracy
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[1] -Value $row.Robustness
                Get-RadarPoint -CenterX $radarCenterX -CenterY $radarCenterY -Radius $radarRadius -Angle $radarAngles[2] -Value $row.Recovery
            )
            $closedPoints = @($points[0], $points[1], $points[2], $points[0])
            $graphics.DrawLines($pen, [System.Drawing.PointF[]]$closedPoints)
            foreach ($point in $points) {
                $graphics.FillEllipse($brush, [single]($point.X - 7), [single]($point.Y - 7), 14, 14)
            }
        }

        $plotLeft = 1518.0
        $plotTop = 276.0
        $plotWidth = 1638.0
        $plotHeight = 850.0
        $plotBottom = $plotTop + $plotHeight
        $plotRight = $plotLeft + $plotWidth

        for ($tick = 0.0; $tick -le 1.0001; $tick += 0.2) {
            $y = $plotBottom - $tick * $plotHeight
            $graphics.DrawLine($penGrid, (New-PointF -X $plotLeft -Y $y), (New-PointF -X $plotRight -Y $y))
            Draw-RightText -Graphics $graphics -Text ([string]::Format([System.Globalization.CultureInfo]::InvariantCulture, '{0:0.0}', $tick)) -Font $fontSmall -Brush $brushMuted -X ($plotLeft - 22) -Y $y
        }

        $graphics.DrawLine($penAxis, (New-PointF -X $plotLeft -Y $plotBottom), (New-PointF -X $plotRight -Y $plotBottom))
        $graphics.DrawLine($penAxis, (New-PointF -X $plotLeft -Y $plotTop), (New-PointF -X $plotLeft -Y $plotBottom))

        $state = $graphics.Save()
        $graphics.TranslateTransform([single]($plotLeft - 115), [single]($plotTop + $plotHeight / 2.0))
        $graphics.RotateTransform(-90.0)
        Draw-CenteredText -Graphics $graphics -Text 'Composite score' -Font $fontYLabel -Brush $brushText -X 0 -Y 0
        $graphics.Restore($state)

        $profiles = @($profileWeights.Keys)
        $groupCenters = @()
        for ($i = 0; $i -lt $profiles.Count; $i++) {
            $groupCenters += ($plotLeft + ($i + 0.5) * ($plotWidth / $profiles.Count))
        }

        $barWidth = 82.0
        $barGap = 16.0
        $clusterWidth = $MetaScores.Count * $barWidth + ($MetaScores.Count - 1) * $barGap

        for ($g = 0; $g -lt $profiles.Count; $g++) {
            $profile = $profiles[$g]
            $startX = $groupCenters[$g] - $clusterWidth / 2.0
            for ($i = 0; $i -lt $MetaScores.Count; $i++) {
                $row = $MetaScores[$i]
                $value = [double]$row.Profiles[$profile]
                $x = $startX + $i * ($barWidth + $barGap)
                $barH = $value * $plotHeight
                $y = $plotBottom - $barH
                $main = New-ColorFromHex $palette[$row.Model].Main
                $fill = New-ColorFromHex $palette[$row.Model].Fill
                $brush = [System.Drawing.SolidBrush]::new($fill)
                $pen = [System.Drawing.Pen]::new($main, 4.0)
                $brushes.Add($brush) | Out-Null
                $pens.Add($pen) | Out-Null

                $rect = [System.Drawing.RectangleF]::new([single]$x, [single]$y, [single]$barWidth, [single]$barH)
                $graphics.FillRectangle($brush, $rect)
                $graphics.DrawRectangle($pen, [single]$x, [single]$y, [single]$barWidth, [single]$barH)
            }

            Draw-CenteredText -Graphics $graphics -Text $profile -Font $fontAxis -Brush $brushText -X $groupCenters[$g] -Y ($plotBottom + 72)
        }

        $legendY = 198.0
        $legendStartX = 1910.0
        $legendSpacing = 250.0
        for ($i = 0; $i -lt $MetaScores.Count; $i++) {
            $row = $MetaScores[$i]
            $main = New-ColorFromHex $palette[$row.Model].Main
            $fill = New-ColorFromHex $palette[$row.Model].Fill
            $brush = [System.Drawing.SolidBrush]::new($fill)
            $pen = [System.Drawing.Pen]::new($main, 3.2)
            $brushes.Add($brush) | Out-Null
            $pens.Add($pen) | Out-Null

            $x = $legendStartX + $i * $legendSpacing
            $graphics.FillRectangle($brush, [single]$x, [single]($legendY - 16), 50, 30)
            $graphics.DrawRectangle($pen, [single]$x, [single]($legendY - 16), 50, 30)
            $graphics.DrawString($row.Model, $fontLegend, $brushText, [single]($x + 66), [single]($legendY - 23))
        }

        if (Test-Path -LiteralPath $outFile) {
            Remove-Item -LiteralPath $outFile -Force
        }
        $bmp.Save($outFile, [System.Drawing.Imaging.ImageFormat]::Png)
    }
    finally {
        foreach ($item in $pens) { $item.Dispose() }
        foreach ($item in $brushes) { $item.Dispose() }
        foreach ($item in $fonts) { $item.Dispose() }
        $graphics.Dispose()
        $bmp.Dispose()
    }
}

$metaScores = Build-MetaScores
Render-Figure -MetaScores $metaScores

Write-Host "Rendered $outFile"
foreach ($row in $metaScores) {
    Write-Host ([string]::Format(
        [System.Globalization.CultureInfo]::InvariantCulture,
        '{0}: Accuracy={1:0.000}, Robustness={2:0.000}, Recovery={3:0.000}',
        $row.Model,
        $row.Accuracy,
        $row.Robustness,
        $row.Recovery
    ))
}
