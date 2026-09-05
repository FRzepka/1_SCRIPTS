param(
    [string]$ReviewRoot = ''
)

$ErrorActionPreference = 'Stop'
if (-not $ReviewRoot) {
    $ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
} else {
    $ReviewRoot = (Resolve-Path $ReviewRoot).Path
}

Add-Type -AssemblyName System.Drawing

$FigureRoot = Join-Path $ReviewRoot 'figures\Schematics'
$Red = [System.Drawing.ColorTranslator]::FromHtml('#D62728')
$PaleRed = [System.Drawing.ColorTranslator]::FromHtml('#EEA4A5')
$Blue = [System.Drawing.ColorTranslator]::FromHtml('#1F77B4')
$PaleBlue = [System.Drawing.ColorTranslator]::FromHtml('#A1C6E0')
$Green = [System.Drawing.ColorTranslator]::FromHtml('#2CA02C')
$PaleGreen = [System.Drawing.ColorTranslator]::FromHtml('#A6D7A6')
$Purple = [System.Drawing.ColorTranslator]::FromHtml('#9467BD')
$PalePurple = [System.Drawing.ColorTranslator]::FromHtml('#D8C2EA')
$Dark = [System.Drawing.ColorTranslator]::FromHtml('#222222')
$Gray = [System.Drawing.ColorTranslator]::FromHtml('#67727A')
$LightGray = [System.Drawing.ColorTranslator]::FromHtml('#CBD2D6')

function New-Font([float]$size, [bool]$bold = $false) {
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    return New-Object System.Drawing.Font('Arial', $size, $style)
}

function Draw-Node($graphics, [float]$x, [float]$y, [float]$radius,
    [System.Drawing.Color]$fill, [System.Drawing.Color]$border, [float]$borderWidth = 6) {
    $brush = New-Object System.Drawing.SolidBrush($fill)
    $pen = New-Object System.Drawing.Pen($border, $borderWidth)
    try {
        $graphics.FillEllipse($brush, $x - $radius, $y - $radius, 2 * $radius, 2 * $radius)
        $graphics.DrawEllipse($pen, $x - $radius, $y - $radius, 2 * $radius, 2 * $radius)
    } finally {
        $brush.Dispose()
        $pen.Dispose()
    }
}

function Draw-CentredText($graphics, [string]$text, $font, $brush,
    [float]$x, [float]$y, [float]$width, [float]$height) {
    $format = New-Object System.Drawing.StringFormat
    try {
        $format.Alignment = [System.Drawing.StringAlignment]::Center
        $format.LineAlignment = [System.Drawing.StringAlignment]::Center
        $graphics.DrawString($text, $font, $brush,
            (New-Object System.Drawing.RectangleF($x, $y, $width, $height)), $format)
    } finally {
        $format.Dispose()
    }
}

function Draw-Arrow($graphics, [float]$x1, [float]$y1, [float]$x2, [float]$y2,
    [System.Drawing.Color]$color, [float]$width = 10) {
    $pen = New-Object System.Drawing.Pen($color, $width)
    $cap = New-Object System.Drawing.Drawing2D.AdjustableArrowCap(10, 13, $true)
    try {
        $pen.CustomEndCap = $cap
        $graphics.DrawLine($pen, $x1, $y1, $x2, $y2)
    } finally {
        $pen.Dispose()
        $cap.Dispose()
    }
}

function Draw-Box($graphics, [float]$x, [float]$y, [float]$width, [float]$height,
    [System.Drawing.Color]$fill, [System.Drawing.Color]$border, [string]$text,
    $font, $textBrush, [float]$borderWidth = 6) {
    $brush = New-Object System.Drawing.SolidBrush($fill)
    $pen = New-Object System.Drawing.Pen($border, $borderWidth)
    try {
        $graphics.FillRectangle($brush, $x, $y, $width, $height)
        $graphics.DrawRectangle($pen, $x, $y, $width, $height)
        Draw-CentredText $graphics $text $font $textBrush ($x + 12) ($y + 8) ($width - 24) ($height - 16)
    } finally {
        $brush.Dispose()
        $pen.Dispose()
    }
}

function Save-DoeFigure {
    $bitmap = New-Object System.Drawing.Bitmap(2200, 1500)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $edgePen = New-Object System.Drawing.Pen($Gray, 7)
    $axisPen = New-Object System.Drawing.Pen($Blue, 10)
    $font = New-Font 42
    $smallFont = New-Font 36
    $legendFont = New-Font 38 $true
    $textBrush = New-Object System.Drawing.SolidBrush($Dark)

    try {
        $points = @(
            (New-Object System.Drawing.PointF(430, 1120)),
            (New-Object System.Drawing.PointF(1370, 1120)),
            (New-Object System.Drawing.PointF(1370, 430)),
            (New-Object System.Drawing.PointF(430, 430)),
            (New-Object System.Drawing.PointF(840, 850)),
            (New-Object System.Drawing.PointF(1780, 850)),
            (New-Object System.Drawing.PointF(1780, 160)),
            (New-Object System.Drawing.PointF(840, 160))
        )
        $edges = @(
            @(0,1), @(1,2), @(2,3), @(3,0),
            @(4,5), @(5,6), @(6,7), @(7,4),
            @(0,4), @(1,5), @(2,6), @(3,7)
        )
        foreach ($edge in $edges) {
            $graphics.DrawLine($edgePen, $points[$edge[0]], $points[$edge[1]])
        }

        # Emphasise the three factor axes from the low/low/low corner.
        $graphics.DrawLine($axisPen, $points[0], $points[1])
        $graphics.DrawLine($axisPen, $points[0], $points[3])
        $graphics.DrawLine($axisPen, $points[0], $points[4])

        foreach ($point in $points) {
            Draw-Node $graphics $point.X $point.Y 31 $PaleRed $Red 8
        }

        Draw-CentredText $graphics 'Charge C-rate' $font $textBrush 570 1320 650 70
        Draw-CentredText $graphics '0.5 C' $smallFont $textBrush 325 1170 210 60
        Draw-CentredText $graphics '0.9 C' $smallFont $textBrush 1270 1170 210 60

        $state = $graphics.Save()
        $graphics.TranslateTransform(150, 910)
        $graphics.RotateTransform(-90)
        Draw-CentredText $graphics 'Depth of discharge' $font $textBrush 0 0 650 70
        $graphics.Restore($state)
        Draw-CentredText $graphics '45%' $smallFont $textBrush 235 1050 150 60
        Draw-CentredText $graphics '65%' $smallFont $textBrush 235 380 150 60

        Draw-CentredText $graphics 'Discharge C-rate' $font $textBrush 820 590 650 70
        Draw-CentredText $graphics '1.0 C' $smallFont $textBrush 610 890 190 60
        Draw-CentredText $graphics '2.5 C' $smallFont $textBrush 790 770 190 60

        $bitmap.Save((Join-Path $FigureRoot 'Figure_2_DoE_Cube.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $edgePen.Dispose()
        $axisPen.Dispose()
        $font.Dispose()
        $smallFont.Dispose()
        $legendFont.Dispose()
        $textBrush.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Draw-NetworkPanel($graphics, [float]$offsetX, [string]$title,
    [float[]]$hiddenYs, [int]$highlightIndex = -1) {
    $inputX = $offsetX + 90
    $hiddenX = $offsetX + 410
    $mlpX = $offsetX + 700
    $outputX = $offsetX + 910
    $inputYs = @(350, 550, 750)
    $mlpYs = @(390, 550, 710)
    $outputYs = @(550)
    $edgePen = New-Object System.Drawing.Pen($LightGray, 4)
    $highlightPen = New-Object System.Drawing.Pen($Red, 7)
    $font = New-Font 36
    $titleFont = New-Font 48 $true
    $textBrush = New-Object System.Drawing.SolidBrush($Dark)

    try {
        Draw-CentredText $graphics $title $titleFont $textBrush $offsetX 30 1000 80
        Draw-CentredText $graphics 'Inputs' $font $textBrush ($inputX - 100) 160 200 60
        Draw-CentredText $graphics 'LSTM hidden' $font $textBrush ($hiddenX - 170) 160 340 60
        Draw-CentredText $graphics 'MLP' $font $textBrush ($mlpX - 100) 160 200 60
        Draw-CentredText $graphics 'Output' $font $textBrush ($outputX - 100) 160 200 60

        foreach ($iy in $inputYs) {
            for ($h = 0; $h -lt $hiddenYs.Count; $h++) {
                $pen = if ($h -eq $highlightIndex) { $highlightPen } else { $edgePen }
                $graphics.DrawLine($pen, $inputX + 31, $iy, $hiddenX - 31, $hiddenYs[$h])
            }
        }
        for ($h = 0; $h -lt $hiddenYs.Count; $h++) {
            foreach ($my in $mlpYs) {
                $pen = if ($h -eq $highlightIndex) { $highlightPen } else { $edgePen }
                $graphics.DrawLine($pen, $hiddenX + 31, $hiddenYs[$h], $mlpX - 31, $my)
            }
        }
        foreach ($my in $mlpYs) {
            $graphics.DrawLine($edgePen, $mlpX + 31, $my, $outputX - 31, $outputYs[0])
        }

        foreach ($iy in $inputYs) { Draw-Node $graphics $inputX $iy 31 $PaleBlue $Blue 7 }
        for ($h = 0; $h -lt $hiddenYs.Count; $h++) {
            if ($h -eq $highlightIndex) {
                Draw-Node $graphics $hiddenX $hiddenYs[$h] 34 $PaleRed $Red 9
            } else {
                Draw-Node $graphics $hiddenX $hiddenYs[$h] 31 $PalePurple $Purple 7
            }
        }
        foreach ($my in $mlpYs) { Draw-Node $graphics $mlpX $my 31 $PaleGreen $Green 7 }
        Draw-Node $graphics $outputX $outputYs[0] 31 $PaleBlue $Blue 7
    } finally {
        $edgePen.Dispose()
        $highlightPen.Dispose()
        $font.Dispose()
        $titleFont.Dispose()
        $textBrush.Dispose()
    }
}

function Save-PruningFigure {
    $bitmap = New-Object System.Drawing.Bitmap(2400, 1100)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $font = New-Font 36
    $smallFont = New-Font 28
    $boldFont = New-Font 30 $true
    $textBrush = New-Object System.Drawing.SolidBrush($Dark)
    try {
        Draw-NetworkPanel $graphics 40 '(a) Base FP32 model' @(270, 410, 550, 690, 830) 2
        Draw-NetworkPanel $graphics 1360 '(b) Pruned FP32 model' @(290, 460, 640, 810) -1

        Draw-Arrow $graphics 1090 535 1290 535 $Red 10
        Draw-CentredText $graphics "Remove one low-saliency`nLSTM channel" $boldFont $textBrush 1020 300 340 150
        Draw-CentredText $graphics "SOC: 64 to 45`nSOH: 128 to 90" $smallFont $textBrush 1020 705 340 110

        $bitmap.Save((Join-Path $FigureRoot 'Figure_6_Pruning_Schematic.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $font.Dispose()
        $smallFont.Dispose()
        $boldFont.Dispose()
        $textBrush.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

function Save-QuantizationFigure {
    $bitmap = New-Object System.Drawing.Bitmap(2400, 1150)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
    $graphics.Clear([System.Drawing.Color]::White)

    $titleFont = New-Font 32 $true
    $font = New-Font 31
    $smallFont = New-Font 27
    $textBrush = New-Object System.Drawing.SolidBrush($Dark)
    $panelPen = New-Object System.Drawing.Pen($LightGray, 5)
    try {
        $panelXs = @(40, 840, 1640)
        foreach ($panelX in $panelXs) {
            $graphics.DrawRectangle($panelPen, $panelX, 35, 720, 1040)
        }

        Draw-CentredText $graphics '(a) Base model' $titleFont $textBrush 40 65 720 75
        Draw-Box $graphics 130 205 540 180 $PalePurple $Purple "Recurrent matrices`nFP32" $font $textBrush
        Draw-Box $graphics 130 470 540 150 $PaleBlue $Blue "Biases and states`nFP32" $font $textBrush
        Draw-Box $graphics 130 705 540 150 $PaleGreen $Green "MLP weights`nFP32" $font $textBrush

        Draw-CentredText $graphics '(b) Stored model constants' $titleFont $textBrush 840 65 720 75
        Draw-Box $graphics 930 190 540 180 $PaleRed $Red "Recurrent codes`nINT8: -127 to 127" $font $textBrush
        Draw-Box $graphics 930 450 540 150 $PalePurple $Purple "One row scale`nFP32" $font $textBrush
        Draw-Box $graphics 930 680 540 180 $PaleBlue $Blue "Biases, states, and`nMLP weights: FP32" $font $textBrush
        Draw-CentredText $graphics 'Round and clip once after training' $smallFont $textBrush 930 900 540 70

        Draw-CentredText $graphics '(c) STM32 gate computation' $titleFont $textBrush 1640 65 720 75
        Draw-Box $graphics 1730 180 540 145 $PaleRed $Red "Load INT8 code" $font $textBrush
        Draw-Arrow $graphics 2000 335 2000 420 $Gray 8
        Draw-Box $graphics 1730 430 540 145 $PalePurple $Purple "Apply FP32 row scale" $font $textBrush
        Draw-Arrow $graphics 2000 585 2000 670 $Gray 8
        Draw-Box $graphics 1730 680 540 145 $PaleBlue $Blue "Accumulate gate`nin FP32" $font $textBrush
        Draw-Arrow $graphics 2000 835 2000 900 $Gray 8
        Draw-Box $graphics 1730 910 540 115 $PaleGreen $Green "FP32 state and MLP" $font $textBrush

        Draw-Arrow $graphics 775 555 825 555 $Red 9
        Draw-Arrow $graphics 1575 555 1625 555 $Red 9

        $bitmap.Save((Join-Path $FigureRoot 'Figure_7_Quantization_Schematic.png'), [System.Drawing.Imaging.ImageFormat]::Png)
    } finally {
        $titleFont.Dispose()
        $font.Dispose()
        $smallFont.Dispose()
        $textBrush.Dispose()
        $panelPen.Dispose()
        $graphics.Dispose()
        $bitmap.Dispose()
    }
}

Save-DoeFigure
Save-PruningFigure
Save-QuantizationFigure

Write-Output 'Reviewer 4 figure clarity updates generated.'
