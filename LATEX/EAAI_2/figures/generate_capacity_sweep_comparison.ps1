$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$figuresDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$workspaceRoot = (Resolve-Path (Join-Path $figuresDir '..\..\..')).Path
$sourceDir = Join-Path $workspaceRoot 'DL_Models\LFP_SOH_Optimization_Study\5_benchmark\SOH_Comparison_Base\results\RESULTS_20260323_084721\plots'
$outputPath = Join-Path $figuresDir 'baseline_capacity_sensitivity.png'

$panels = @(
    @{ Family = 'CNN';  File = 'cnn_tradeoff.png';  Color = '#59C7C2'; Label = '(a)' },
    @{ Family = 'GRU';  File = 'gru_tradeoff.png';  Color = '#59E83A'; Label = '(b)' },
    @{ Family = 'LSTM'; File = 'lstm_tradeoff.png'; Color = '#E76B91'; Label = '(c)' },
    @{ Family = 'TCN';  File = 'tcn_tradeoff.png';  Color = '#294862'; Label = '(d)' }
)

function Convert-HexColor([string]$hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($hex)
}

function Recolor-MatplotlibBlue([System.Drawing.Bitmap]$bitmap, [System.Drawing.Color]$target) {
    $source = [System.Drawing.Color]::FromArgb(31, 119, 180)
    for ($y = 0; $y -lt $bitmap.Height; $y++) {
        for ($x = 0; $x -lt $bitmap.Width; $x++) {
            $pixel = $bitmap.GetPixel($x, $y)
            $distance = [Math]::Sqrt(
                [Math]::Pow($pixel.R - $source.R, 2) +
                [Math]::Pow($pixel.G - $source.G, 2) +
                [Math]::Pow($pixel.B - $source.B, 2)
            )
            if ($distance -lt 72) {
                $bitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($pixel.A, $target.R, $target.G, $target.B))
            }
        }
    }
}

$panelWidth = 825
$panelHeight = 630
$gap = 18
$canvasWidth = 2 * $panelWidth + $gap
$canvasHeight = 2 * $panelHeight + $gap
$canvas = New-Object System.Drawing.Bitmap($canvasWidth, $canvasHeight)
$canvas.SetResolution(300, 300)
$graphics = [System.Drawing.Graphics]::FromImage($canvas)
$graphics.Clear([System.Drawing.Color]::White)
$graphics.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::HighQuality
$graphics.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$font = New-Object System.Drawing.Font('Arial', 7.5, [System.Drawing.FontStyle]::Bold)
$brush = [System.Drawing.Brushes]::Black

try {
    for ($i = 0; $i -lt $panels.Count; $i++) {
        $panel = $panels[$i]
        $sourcePath = Join-Path $sourceDir $panel.File
        if (-not (Test-Path -LiteralPath $sourcePath)) {
            throw "Missing source plot: $sourcePath"
        }

        $sourceImage = [System.Drawing.Image]::FromFile($sourcePath)
        try {
            $cropped = New-Object System.Drawing.Bitmap($panelWidth, $panelHeight)
            $cropped.SetResolution(300, 300)
            $cropGraphics = [System.Drawing.Graphics]::FromImage($cropped)
            try {
                $cropGraphics.Clear([System.Drawing.Color]::White)
                $sourceRectangle = New-Object System.Drawing.Rectangle(0, 0, $panelWidth, $panelHeight)
                $destinationRectangle = New-Object System.Drawing.Rectangle(0, 0, $panelWidth, $panelHeight)
                $cropGraphics.DrawImage($sourceImage, $destinationRectangle, $sourceRectangle, [System.Drawing.GraphicsUnit]::Pixel)
                $cropGraphics.FillRectangle([System.Drawing.Brushes]::White, 0, 0, $panelWidth, 48)
            } finally {
                $cropGraphics.Dispose()
            }

            Recolor-MatplotlibBlue $cropped (Convert-HexColor $panel.Color)

            $column = $i % 2
            $row = [Math]::Floor($i / 2)
            $offsetX = $column * ($panelWidth + $gap)
            $offsetY = $row * ($panelHeight + $gap)
            $graphics.DrawImageUnscaled($cropped, $offsetX, $offsetY)
            $graphics.DrawString($panel.Label, $font, $brush, $offsetX + 18, $offsetY + 10)
            $cropped.Dispose()
        } finally {
            $sourceImage.Dispose()
        }
    }

    $canvas.Save($outputPath, [System.Drawing.Imaging.ImageFormat]::Png)
} finally {
    $font.Dispose()
    $graphics.Dispose()
    $canvas.Dispose()
}

Get-Item -LiteralPath $outputPath | Select-Object FullName, Length, LastWriteTime
