$ErrorActionPreference = 'Stop'

function Get-OfficeColor([string]$hex) {
    $value = $hex.TrimStart('#')
    $r = [Convert]::ToInt32($value.Substring(0, 2), 16)
    $g = [Convert]::ToInt32($value.Substring(2, 2), 16)
    $b = [Convert]::ToInt32($value.Substring(4, 2), 16)
    return $r + 256 * $g + 65536 * $b
}

function Add-Text($slide, [double]$x, [double]$y, [double]$w, [double]$h, [string]$text,
                  [double]$size = 12, [bool]$bold = $false, [int]$align = 2,
                  [string]$color = '#222222') {
    $shape = $slide.Shapes.AddTextbox(1, $x, $y, $w, $h)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $range = $shape.TextFrame.TextRange
    $range.Text = $text
    $range.Font.Name = 'Arial'
    $range.Font.Size = $size
    $range.Font.Bold = $(if ($bold) { -1 } else { 0 })
    $range.Font.Color.RGB = Get-OfficeColor $color
    $range.ParagraphFormat.Alignment = $align
    return $shape
}

function Add-Line($slide, [double]$x1, [double]$y1, [double]$x2, [double]$y2,
                  [string]$color = '#222222', [double]$weight = 1.0, [bool]$dashed = $false) {
    $line = $slide.Shapes.AddLine($x1, $y1, $x2, $y2)
    $line.Line.ForeColor.RGB = Get-OfficeColor $color
    $line.Line.Weight = $weight
    if ($dashed) { $line.Line.DashStyle = 4 }
    return $line
}

function Add-Bar($slide, [double]$x, [double]$y, [double]$w, [double]$h,
                 [string]$color, [double]$transparency = 0) {
    $bar = $slide.Shapes.AddShape(1, $x, $y, $w, $h)
    $bar.Fill.ForeColor.RGB = Get-OfficeColor $color
    $bar.Fill.Transparency = $transparency
    $bar.Line.ForeColor.RGB = Get-OfficeColor $color
    $bar.Line.Weight = 1.4
    return $bar
}

$figuresDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$sourceDir = Join-Path $figuresDir 'source'
New-Item -ItemType Directory -Path $sourceDir -Force | Out-Null
$repoRoot = (Resolve-Path (Join-Path $figuresDir '..\..\..')).Path
$resultsDir = Join-Path $repoRoot 'DL_Models\LFP_SOH_Optimization_Study\5_benchmark\batmm\LFP_SOH_Optimization_Study\5_benchmark\Stateful_Base_Comparison\results'
$metrics = Import-Csv -LiteralPath (Join-Path $resultsDir 'metrics_summary.csv') |
    Where-Object { $_.aggregation -eq 'cell_macro' }
$inventory = Import-Csv -LiteralPath (Join-Path $resultsDir 'model_inventory.csv')

$pngPath = Join-Path $figuresDir 'baseline_model_comparison.png'
$pptxPath = Join-Path $sourceDir 'baseline_model_comparison.pptx'

$colors = @{ CNN = '#59C7C2'; GRU = '#59E83A'; LSTM = '#E76B91'; TCN = '#294862' }
$models = foreach ($name in @('CNN', 'GRU', 'LSTM', 'TCN')) {
    $metric = $metrics | Where-Object { $_.model -eq $name.ToLowerInvariant() }
    $modelInfo = $inventory | Where-Object { $_.model -eq $name.ToLowerInvariant() }
    [pscustomobject]@{
        Name = $name
        Mae = [double]::Parse($metric.mae, [Globalization.CultureInfo]::InvariantCulture)
        Rmse = [double]::Parse($metric.rmse, [Globalization.CultureInfo]::InvariantCulture)
        Size = ([double]::Parse($modelInfo.parameters, [Globalization.CultureInfo]::InvariantCulture) * 4 / [Math]::Pow(1024, 2))
        Color = $colors[$name]
    }
}
$culture = [Globalization.CultureInfo]::InvariantCulture

$ppt = New-Object -ComObject PowerPoint.Application
$ppt.Visible = -1
try {
    $presentation = $ppt.Presentations.Add()
    $presentation.PageSetup.SlideWidth = 1000
    $presentation.PageSetup.SlideHeight = 460
    $slide = $presentation.Slides.Add(1, 12)

    $white = Get-OfficeColor '#FFFFFF'
    $slide.Background.Fill.ForeColor.RGB = $white
    $slide.FollowMasterBackground = 0

    $plotTop = 48
    $plotBottom = 348
    $plotHeight = $plotBottom - $plotTop
    $grid = '#D5D8DC'
    $axis = '#222222'

    # Panel (a): baseline prediction errors.
    $leftA = 72
    $rightA = 482
    $centersA = @(130, 225, 320, 415)
    $maxError = 0.025
    $null = Add-Text $slide 48 12 35 25 '(a)' 16 $true 1

    foreach ($tick in @(0.000, 0.005, 0.010, 0.015, 0.020, 0.025)) {
        $y = $plotBottom - ($tick / $maxError) * $plotHeight
        $null = Add-Line $slide $leftA $y $rightA $y $grid 0.8 ($tick -gt 0)
        $null = Add-Text $slide 25 ($y - 9) 41 20 ($tick.ToString('0.000', $culture)) 11 $false 3
    }
    $null = Add-Line $slide $leftA $plotTop $leftA $plotBottom $axis 1.2
    $null = Add-Line $slide $leftA $plotBottom $rightA $plotBottom $axis 1.2

    for ($i = 0; $i -lt $models.Count; $i++) {
        $m = $models[$i]
        $center = $centersA[$i]
        $maeHeight = ($m.Mae / $maxError) * $plotHeight
        $rmseHeight = ($m.Rmse / $maxError) * $plotHeight
        $null = Add-Bar $slide ($center - 27) ($plotBottom - $maeHeight) 23 $maeHeight $m.Color 0.52
        $null = Add-Bar $slide ($center + 4) ($plotBottom - $rmseHeight) 23 $rmseHeight $m.Color 0.04
        $null = Add-Text $slide ($center - 42) ($plotBottom - $maeHeight - 19) 54 18 ($m.Mae.ToString('0.0000', $culture)) 10 $false 2
        $null = Add-Text $slide ($center - 11) ($plotBottom - $rmseHeight - 19) 54 18 ($m.Rmse.ToString('0.0000', $culture)) 10 $false 2
        $null = Add-Text $slide ($center - 34) 354 68 22 $m.Name 13 $true 2
    }

    $legendMae = Add-Bar $slide 225 15 16 12 '#777777' 0.52
    $legendRmse = Add-Bar $slide 310 15 16 12 '#777777' 0.04
    $null = Add-Text $slide 247 12 52 20 'MAE' 12 $false 1
    $null = Add-Text $slide 332 12 58 20 'RMSE' 12 $false 1
    $yLabelA = Add-Text $slide -55 188 160 27 'SOH error [0-1]' 14 $false 2
    $yLabelA.Rotation = 270
    $null = Add-Text $slide 190 398 175 24 'Architecture' 14 $false 2

    # Panel (b): parameter-only FP32 footprint.
    $leftB = 576
    $rightB = 972
    $centersB = @(630, 725, 820, 915)
    $maxSize = 4.0
    $null = Add-Text $slide 550 12 35 25 '(b)' 16 $true 1

    foreach ($tick in @(0, 1, 2, 3, 4)) {
        $y = $plotBottom - ($tick / $maxSize) * $plotHeight
        $null = Add-Line $slide $leftB $y $rightB $y $grid 0.8 ($tick -gt 0)
        $null = Add-Text $slide 535 ($y - 9) 34 20 ($tick.ToString('0', $culture)) 11 $false 3
    }
    $null = Add-Line $slide $leftB $plotTop $leftB $plotBottom $axis 1.2
    $null = Add-Line $slide $leftB $plotBottom $rightB $plotBottom $axis 1.2

    for ($i = 0; $i -lt $models.Count; $i++) {
        $m = $models[$i]
        $center = $centersB[$i]
        $height = ($m.Size / $maxSize) * $plotHeight
        $null = Add-Bar $slide ($center - 25) ($plotBottom - $height) 50 $height $m.Color 0.28
        $null = Add-Text $slide ($center - 35) ($plotBottom - $height - 21) 70 20 ($m.Size.ToString('0.000', $culture)) 11 $false 2
        $null = Add-Text $slide ($center - 34) 354 68 22 $m.Name 13 $true 2
    }

    $yLabelB = Add-Text $slide 450 188 160 27 'FP32 weights [MiB]' 14 $false 2
    $yLabelB.Rotation = 270
    $null = Add-Text $slide 690 398 175 24 'Architecture' 14 $false 2

    $presentation.SaveAs($pptxPath, 24)
    $slide.Export($pngPath, 'PNG', 2400, 1104)
    $presentation.Close()
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}

Get-Item -LiteralPath $pngPath, $pptxPath | Select-Object FullName, Length, LastWriteTime
