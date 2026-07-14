param(
    [string]$ReviewRoot = ''
)

$ErrorActionPreference = 'Stop'
$Inv = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $Inv
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $Inv
if (-not $ReviewRoot) {
    $ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
} else {
    $ReviewRoot = (Resolve-Path $ReviewRoot).Path
}

$ResultsRoot = Join-Path $ReviewRoot 'review_analysis\results'
$Reviewer3Results = Join-Path $ResultsRoot 'reviewer3'
$FiguresRoot = Join-Path $ReviewRoot 'figures\Review_1_Additional'
New-Item -ItemType Directory -Force -Path $Reviewer3Results | Out-Null
New-Item -ItemType Directory -Force -Path $FiguresRoot | Out-Null

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$Green = '#2CA02C'
$Red = '#D62728'
$Blue = '#1F77B4'
$Purple = '#9467BD'
$Gray = '#67727A'
$Dark = '#222222'
$Grid = '#D9DEE1'
$PaleGreen = '#A6D7A6'
$PaleRed = '#EEA4A5'
$PaleBlue = '#A1C6E0'
$PalePurple = '#D8C2EA'
$PaleGray = '#E5E9EC'
$VeryPaleBlue = '#EDF4F8'
$VeryPaleRed = '#FBEDEE'
$VeryPaleGreen = '#EEF7EE'

function C([string]$hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($hex)
}

function D([object]$value) {
    return [double]::Parse([string]$value, [System.Globalization.NumberStyles]::Float, $Inv)
}

function Font([float]$size, [bool]$bold=$false) {
    $style = if ($bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    return New-Object System.Drawing.Font('Arial', $size, $style)
}

function New-Chart([int]$width=2200, [int]$height=1450) {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = $width
    $chart.Height = $height
    $chart.BackColor = [System.Drawing.Color]::White
    $chart.AntiAliasing = 'All'
    $chart.TextAntiAliasingQuality = 'High'
    return $chart
}

function Add-Area($chart, [string]$name, [float]$x, [float]$y, [float]$w, [float]$h,
    [string]$xTitle, [string]$yTitle) {
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea $name
    $area.BackColor = [System.Drawing.Color]::White
    $area.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition($x,$y,$w,$h)
    $area.AxisX.Title = $xTitle
    $area.AxisY.Title = $yTitle
    $area.AxisX.TitleFont = Font 15
    $area.AxisY.TitleFont = Font 15
    $area.AxisX.LabelStyle.Font = Font 15
    $area.AxisY.LabelStyle.Font = Font 15
    $area.AxisX.MajorGrid.LineColor = C '#E4E8EA'
    $area.AxisY.MajorGrid.LineColor = C $Grid
    $area.AxisX.LineColor = C '#4E5960'
    $area.AxisY.LineColor = C '#4E5960'
    $area.AxisX.MajorTickMark.LineColor = C '#4E5960'
    $area.AxisY.MajorTickMark.LineColor = C '#4E5960'
    $area.AxisX.IsMarginVisible = $false
    $chart.ChartAreas.Add($area)
    return $area
}

function Add-PanelTitle($chart, [string]$areaName, [string]$text, [float]$fontSize=18) {
    $title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $title.Text = $text
    $title.DockedToChartArea = $areaName
    $title.IsDockedInsideChartArea = $false
    $title.Alignment = 'MiddleLeft'
    $title.Font = Font $fontSize $true
    $title.ForeColor = C '#172026'
    $chart.Titles.Add($title)
}

function Add-AreaLegend($chart, [string]$name, [string]$areaName, [string]$alignment='Near', [float]$fontSize=12) {
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend $name
    $legend.DockedToChartArea = $areaName
    $legend.IsDockedInsideChartArea = $true
    $legend.Docking = 'Top'
    $legend.Alignment = $alignment
    $legend.LegendStyle = 'Column'
    $legend.Font = Font $fontSize
    $legend.BackColor = [System.Drawing.Color]::FromArgb(232,255,255,255)
    $chart.Legends.Add($legend)
    return $legend
}

function Add-CustomLineLegendItem($legend, [string]$name, [string]$color, [string]$dash='Solid') {
    $item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
    $item.Name = $name
    $item.Color = C $color
    $item.BorderColor = C $color
    $item.BorderWidth = 4
    $item.ImageStyle = 'Line'
    $item.BorderDashStyle = $dash
    $legend.CustomItems.Add($item)
}

function Add-CustomBoxLegendItem($legend, [string]$name, [string]$fill, [string]$border) {
    $item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
    $item.Name = $name
    $item.Color = C $fill
    $item.BorderColor = C $border
    $item.BorderWidth = 2
    $item.ImageStyle = 'Rectangle'
    $legend.CustomItems.Add($item)
}

function Add-LineSeries($chart, [string]$area, [string]$name, [string]$color,
    [object[]]$rows, [string]$xProperty, [string]$yProperty, [string]$dash='Solid',
    [bool]$markers=$false, [int]$width=4) {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series "$area-$name-$($chart.Series.Count)"
    $series.ChartArea = $area
    $series.ChartType = 'Line'
    $series.Color = C $color
    $series.BorderWidth = $width
    $series.BorderDashStyle = $dash
    $series.IsVisibleInLegend = $false
    if ($markers) {
        $series.MarkerStyle = 'Circle'
        $series.MarkerSize = 7
        $series.MarkerColor = [System.Drawing.Color]::White
        $series.MarkerBorderColor = C $color
        $series.MarkerBorderWidth = 2
    }
    foreach ($row in $rows) {
        [void]$series.Points.AddXY((D $row.$xProperty), (D $row.$yProperty))
    }
    $chart.Series.Add($series)
}

function Add-PointSeries($chart, [string]$area, [string]$name, [string]$color,
    [double]$x, [double]$y, [string]$label, [string]$marker='Circle', [float]$fontSize=12,
    [string]$labelStyle='Auto') {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series "$area-$name-$($chart.Series.Count)"
    $series.ChartArea = $area
    $series.ChartType = 'Point'
    $series.Color = C $color
    $series.MarkerStyle = $marker
    $series.MarkerSize = 11
    $series.MarkerColor = [System.Drawing.Color]::White
    $series.MarkerBorderColor = C $color
    $series.MarkerBorderWidth = 3
    $series.IsVisibleInLegend = $false
    $index = $series.Points.AddXY($x,$y)
    $series.Points[$index].Label = $label
    $series.Points[$index].Font = Font $fontSize $true
    $series.Points[$index].LabelForeColor = C $color
    if ($labelStyle -ne 'Auto') {
        $series.Points[$index].SetCustomProperty('LabelStyle',$labelStyle)
    }
    $chart.Series.Add($series)
}

function Save-Chart($chart, [string]$name) {
    $path = Join-Path $FiguresRoot $name
    $chart.SaveImage($path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
    Write-Host "Saved $path"
}

function New-Canvas([int]$width=2400, [int]$height=1400) {
    $bitmap = New-Object System.Drawing.Bitmap($width,$height)
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    $graphics.Clear([System.Drawing.Color]::White)
    $graphics.SmoothingMode = 'AntiAlias'
    $graphics.InterpolationMode = 'HighQualityBicubic'
    $graphics.TextRenderingHint = 'AntiAliasGridFit'
    return [pscustomobject]@{ Bitmap=$bitmap; Graphics=$graphics; Width=$width; Height=$height }
}

function Draw-Text($g, [string]$text, [float]$x, [float]$y, [float]$w, [float]$h,
    [float]$size=24, [bool]$bold=$false, [string]$color=$Dark,
    [string]$align='Near', [string]$valign='Near') {
    $format = New-Object System.Drawing.StringFormat
    $format.Alignment = [System.Drawing.StringAlignment]::$align
    $format.LineAlignment = [System.Drawing.StringAlignment]::$valign
    $format.Trimming = [System.Drawing.StringTrimming]::Word
    $format.FormatFlags = [System.Drawing.StringFormatFlags]::LineLimit
    $brush = New-Object System.Drawing.SolidBrush (C $color)
    $font = Font $size $bold
    $rect = New-Object System.Drawing.RectangleF($x,$y,$w,$h)
    $g.DrawString($text,$font,$brush,$rect,$format)
    $font.Dispose(); $brush.Dispose(); $format.Dispose()
}

function Draw-Box($g, [float]$x, [float]$y, [float]$w, [float]$h,
    [string]$fill, [string]$border, [string]$text='', [float]$fontSize=22,
    [bool]$bold=$false) {
    $brush = New-Object System.Drawing.SolidBrush (C $fill)
    $pen = New-Object System.Drawing.Pen((C $border),3)
    $g.FillRectangle($brush,$x,$y,$w,$h)
    $g.DrawRectangle($pen,$x,$y,$w,$h)
    $brush.Dispose(); $pen.Dispose()
    if ($text) { Draw-Text $g $text $x $y $w $h $fontSize $bold $Dark 'Center' 'Center' }
}

function Draw-Arrow($g, [float]$x1, [float]$y1, [float]$x2, [float]$y2,
    [string]$color=$Blue, [float]$width=5) {
    $pen = New-Object System.Drawing.Pen((C $color),$width)
    $pen.EndCap = [System.Drawing.Drawing2D.LineCap]::ArrowAnchor
    $g.DrawLine($pen,$x1,$y1,$x2,$y2)
    $pen.Dispose()
}

function Save-Canvas($canvas, [string]$name) {
    $path = Join-Path $FiguresRoot $name
    $canvas.Bitmap.Save($path,[System.Drawing.Imaging.ImageFormat]::Png)
    $canvas.Graphics.Dispose(); $canvas.Bitmap.Dispose()
    Write-Host "Saved $path"
}

function Draw-StackedStorageBars($g, [float]$x, [float]$y, [float]$w, [float]$h,
    [object[]]$rows, [double]$maxValue, [string]$panelTitle) {
    Draw-Text $g $panelTitle $x ($y-72) $w 55 30 $true $Dark 'Near' 'Center'
    $plotLeft = $x + 88
    $plotTop = $y + 25
    $plotWidth = $w - 115
    $plotHeight = $h - 125
    $axisPen = New-Object System.Drawing.Pen((C '#4E5960'),3)
    $gridPen = New-Object System.Drawing.Pen((C $Grid),2)
    for ($i=0; $i -le 4; $i++) {
        $value = $maxValue * $i / 4
        $yy = $plotTop + $plotHeight - ($plotHeight * $i / 4)
        $g.DrawLine($gridPen,$plotLeft,$yy,$plotLeft+$plotWidth,$yy)
        Draw-Text $g ('{0:0}' -f $value) ($x+2) ($yy-22) 76 44 19 $false $Gray 'Far' 'Center'
    }
    $g.DrawLine($axisPen,$plotLeft,$plotTop,$plotLeft,$plotTop+$plotHeight)
    $g.DrawLine($axisPen,$plotLeft,$plotTop+$plotHeight,$plotLeft+$plotWidth,$plotTop+$plotHeight)
    Draw-Text $g 'KiB' ($x+8) ($plotTop-38) 65 36 18 $false $Gray 'Center' 'Center'

    $components = @(
        @('FP32RecurrentWeightsBytes',$PaleGreen,$Green),
        @('Int8RecurrentWeightsBytes',$PaleBlue,$Blue),
        @('FP32MLPBytes',$PalePurple,$Purple),
        @('FP32ScalesAndBiasBytes',$PaleRed,$Red),
        @('PersistentStateBytes',$PaleGray,$Gray)
    )
    $barWidth = [math]::Min(150,$plotWidth/4)
    $centers = @(($plotLeft + $plotWidth*0.30),($plotLeft + $plotWidth*0.70))
    for ($r=0; $r -lt $rows.Count; $r++) {
        $baseY = $plotTop + $plotHeight
        foreach ($component in $components) {
            $value = (D $rows[$r].($component[0])) / 1024.0
            if ($value -le 0) { continue }
            $segmentHeight = $plotHeight * $value / $maxValue
            $topY = $baseY - $segmentHeight
            $brush = New-Object System.Drawing.SolidBrush (C $component[1])
            $pen = New-Object System.Drawing.Pen((C $component[2]),2)
            $g.FillRectangle($brush,$centers[$r]-$barWidth/2,$topY,$barWidth,$segmentHeight)
            $g.DrawRectangle($pen,$centers[$r]-$barWidth/2,$topY,$barWidth,$segmentHeight)
            $brush.Dispose(); $pen.Dispose()
            $baseY = $topY
        }
        $total = (D $rows[$r].TotalModelBytes) / 1024.0
        Draw-Text $g ('{0:0.0}' -f $total) ($centers[$r]-90) ($baseY-48) 180 38 20 $true $Dark 'Center' 'Center'
        Draw-Text $g ([string]$rows[$r].Variant) ($centers[$r]-110) ($plotTop+$plotHeight+15) 220 45 21 $true $Dark 'Center' 'Center'
    }
    $axisPen.Dispose(); $gridPen.Dispose()
}

# -----------------------------------------------------------------------------
# rev_6: analytical hidden-size scaling and the two implemented architecture points.
# This figure reports operation counts only and does not extrapolate latency, flash,
# or energy to another MCU family.
# -----------------------------------------------------------------------------
$scalingRows = @()
foreach ($taskSpec in @(
    [pscustomobject]@{Task='SOC'; MlpWidth=64; BaseH=64; PrunedH=45},
    [pscustomobject]@{Task='SOH'; MlpWidth=128; BaseH=128; PrunedH=90}
)) {
    for ($h=30; $h -le 1000; $h+=10) {
        $hp = [int][math]::Round(0.7*$h,[System.MidpointRounding]::AwayFromZero)
        $baseMac = 4*$h*(6+$h) + $h*$taskSpec.MlpWidth + $taskSpec.MlpWidth
        $prunedMac = 4*$hp*(6+$hp) + $hp*$taskSpec.MlpWidth + $taskSpec.MlpWidth
        $scalingRows += [pscustomobject]@{
            Task=$taskSpec.Task; HiddenSize=$h; PrunedHiddenSize=$hp; TotalMACs=$baseMac
            MACReductionPct=100.0*(1.0-$prunedMac/$baseMac)
        }
    }
}
$scalingRows | Export-Csv -LiteralPath (Join-Path $Reviewer3Results 'model_complexity_scaling.csv') -NoTypeInformation -Encoding UTF8

$chart = New-Chart 2500 2050
$left = Add-Area $chart 'Complexity' 6 3 89 42 'LSTM hidden size H' 'Analytical MACs per inference'
$right = Add-Area $chart 'Reduction' 6 52 89 42 'Original LSTM hidden size H' 'MAC reduction after 30% hidden-unit pruning [%]'
Add-PanelTitle $chart 'Complexity' '(a) Architecture-level operation scaling' 21
Add-PanelTitle $chart 'Reduction' '(b) Why 30% hidden-unit pruning approaches a 51% MAC reduction' 21
$left.AxisX.TitleFont=Font 18; $left.AxisY.TitleFont=Font 18
$left.AxisX.LabelStyle.Font=Font 17; $left.AxisY.LabelStyle.Font=Font 17
$right.AxisX.TitleFont=Font 18; $right.AxisY.TitleFont=Font 18
$right.AxisX.LabelStyle.Font=Font 17; $right.AxisY.LabelStyle.Font=Font 17
$left.AxisX.Minimum=30; $left.AxisX.Maximum=160; $left.AxisX.Interval=20
$left.AxisY.Minimum=0; $left.AxisY.Maximum=180000; $left.AxisY.Interval=30000
$left.AxisY.LabelStyle.Format='N0'
$right.AxisX.Minimum=30; $right.AxisX.Maximum=1000; $right.AxisX.Interval=100
$right.AxisY.Minimum=38; $right.AxisY.Maximum=52; $right.AxisY.Interval=2

foreach ($task in @('SOC','SOH')) {
    $color = if ($task -eq 'SOC') { $Red } else { $Blue }
    $rows = @($scalingRows | Where-Object Task -eq $task)
    $complexityRows = @($rows | Where-Object { [int]$_.HiddenSize -le 160 })
    Add-LineSeries $chart 'Complexity' $task $color $complexityRows 'HiddenSize' 'TotalMACs' 'Solid' $false 5
    Add-LineSeries $chart 'Reduction' $task $color $rows 'HiddenSize' 'MACReductionPct' 'Solid' $false 5
}

$limitRows = @(
    [pscustomobject]@{HiddenSize=30; MACReductionPct=51},
    [pscustomobject]@{HiddenSize=1000; MACReductionPct=51}
)
Add-LineSeries $chart 'Reduction' 'Quadratic limit' $Gray $limitRows 'HiddenSize' 'MACReductionPct' 'Dash' $false 4

$socBaseMac = 4*64*(6+64)+64*64+64
$socPrunedMac = 4*45*(6+45)+45*64+64
$sohBaseMac = 4*128*(6+128)+128*128+128
$sohPrunedMac = 4*90*(6+90)+90*128+128
$socActualReduction = 100.0*(1.0-$socPrunedMac/$socBaseMac)
$sohActualReduction = 100.0*(1.0-$sohPrunedMac/$sohBaseMac)
Add-PointSeries $chart 'Complexity' 'SOC base' $Red 64 $socBaseMac 'SOC Base' 'Circle' 14
Add-PointSeries $chart 'Complexity' 'SOC pruned' $Red 45 $socPrunedMac 'SOC Pruned' 'Diamond' 14
Add-PointSeries $chart 'Complexity' 'SOH base' $Blue 128 $sohBaseMac 'SOH Base' 'Circle' 14
Add-PointSeries $chart 'Complexity' 'SOH pruned' $Blue 90 $sohPrunedMac 'SOH Pruned' 'Diamond' 14
Add-PointSeries $chart 'Reduction' 'SOC implemented architecture' $Red 64 $socActualReduction 'SOC 64 -> 45: 45.1%' 'Circle' 15 'Right'
Add-PointSeries $chart 'Reduction' 'SOH implemented architecture' $Blue 128 $sohActualReduction 'SOH 128 -> 90: 45.7%' 'Circle' 15 'Right'

$legendLeft = Add-AreaLegend $chart 'ComplexityLegend' 'Complexity' 'Near' 15
Add-CustomLineLegendItem $legendLeft 'SOC analytical MACs' $Red
Add-CustomLineLegendItem $legendLeft 'SOH analytical MACs' $Blue
$legendRight = Add-AreaLegend $chart 'ReductionLegend' 'Reduction' 'Near' 15
Add-CustomLineLegendItem $legendRight 'SOC analytical MAC reduction' $Red
Add-CustomLineLegendItem $legendRight 'SOH analytical MAC reduction' $Blue
Add-CustomLineLegendItem $legendRight 'H^2 limit: 1 - 0.70^2 = 51%' $Gray 'Dash'
Save-Chart $chart 'rev_6_model_complexity_scaling.png'

# -----------------------------------------------------------------------------
# rev_7: audited derivative boundary. The code uses timestamp-aware backward
# differences; the current paper wording "centred" is not reproduced here.
# -----------------------------------------------------------------------------
$canvas = New-Canvas 2400 1320
$g = $canvas.Graphics
Draw-Text $g '(a) Audited benchmark feature path' 80 55 1060 70 34 $true
Draw-Text $g '(b) Causal option for non-ideal sampling' 1260 55 1060 70 34 $true

Draw-Box $g 90 190 300 150 $VeryPaleBlue $Blue "Timestamped samples`nx[k-1], x[k]" 25 $true
Draw-Arrow $g 390 265 485 265
Draw-Box $g 490 170 555 190 $VeryPaleRed $Red "Backward difference`ndx/dt[k] = (x[k] - x[k-1]) /`nmax(t[k] - t[k-1], 1e-6 s)" 23 $false
Draw-Arrow $g 765 360 765 455
Draw-Box $g 535 460 460 130 $VeryPaleGreen $Green "Robust scaling with`nembedded median and IQR" 23 $false
Draw-Arrow $g 765 590 765 685
Draw-Box $g 535 690 460 125 $VeryPaleBlue $Blue "Prepared six-feature`nSOC vector" 24 $true
Draw-Arrow $g 765 815 765 910
Draw-Box $g 535 915 460 125 $VeryPaleGreen $Green "UART replay to`nSTM32 estimator" 24 $true
Draw-Box $g 165 1090 900 115 '#FFF7E8' $Red "Derivative features were prepared before UART replay.`nTheir computation was not timed or stress-tested on the MCU." 23 $true

Draw-Box $g 1275 185 320 135 $VeryPaleBlue $Blue "New timestamped`nmeasurement" 24 $true
Draw-Arrow $g 1595 252 1680 252
Draw-Box $g 1685 175 555 155 $VeryPaleGreen $Green "Validity and timing checks`nfinite values, Delta t bounds,`nsequence and stale-sample flags" 22 $false
Draw-Arrow $g 1960 330 1960 420
Draw-Box $g 1685 425 555 150 $VeryPaleBlue $Blue "Timestamp-aware backward`ndifference using accepted`ncurrent and previous samples" 22 $false
Draw-Arrow $g 1960 575 1960 665
Draw-Box $g 1685 670 555 150 $VeryPaleGreen $Green "Causal low-pass filtering or`nderivative limiting before`nfeature normalization" 22 $false
Draw-Arrow $g 1960 820 1960 910
Draw-Box $g 1685 915 555 125 $VeryPaleBlue $Blue "State estimator and`ndiagnostic flag" 24 $true

Draw-Arrow $g 1685 252 1480 500 $Red 4
Draw-Box $g 1280 500 320 170 '#FFF7E8' $Red "Invalid or missing sample:`nhold/flag, skip derivative,`nresynchronise state" 21 $false
Draw-Arrow $g 1440 670 1685 745 $Red 4
Draw-Box $g 1315 1090 945 115 $VeryPaleRed $Red "Proposed mitigation path for deployment; not evaluated in the`ncurrent compression benchmark." 23 $true
Save-Canvas $canvas 'rev_7_derivative_deployment_boundary.png'

# -----------------------------------------------------------------------------
# rev_8: cross-task L2 saliency plus a method-property comparison. The right panel
# is a rationale, not an empirical comparison of pruning accuracy.
# -----------------------------------------------------------------------------
$saliency = Import-Csv (Join-Path $ResultsRoot 'weights\lstm_unit_saliency.csv')
$canvas = New-Canvas 2400 1400
$g = $canvas.Graphics
Draw-Text $g '(a) Gate-group L2 ranking for complete hidden units' 80 55 1100 70 30 $true
Draw-Text $g '(b) Criterion scope and deployment implications' 1280 55 1040 70 30 $true

$plotX=135; $plotY=185; $plotW=980; $plotH=900
$removedBrush = New-Object System.Drawing.SolidBrush ([System.Drawing.Color]::FromArgb(125,(C $PaleRed)))
$g.FillRectangle($removedBrush,$plotX,$plotY,$plotW*0.30,$plotH)
$removedBrush.Dispose()
$gridPen = New-Object System.Drawing.Pen((C $Grid),2)
$axisPen = New-Object System.Drawing.Pen((C '#4E5960'),3)
for ($i=0;$i -le 5;$i++) {
    $xx=$plotX+$plotW*$i/5; $yy=$plotY+$plotH-$plotH*$i/5
    $g.DrawLine($gridPen,$xx,$plotY,$xx,$plotY+$plotH)
    $g.DrawLine($gridPen,$plotX,$yy,$plotX+$plotW,$yy)
    Draw-Text $g ('{0}' -f ($i*20)) ($xx-40) ($plotY+$plotH+12) 80 36 18 $false $Gray 'Center' 'Center'
    Draw-Text $g ('{0:0.0}' -f ($i/5.0)) ($plotX-78) ($yy-18) 65 36 18 $false $Gray 'Far' 'Center'
}
$g.DrawLine($axisPen,$plotX,$plotY,$plotX,$plotY+$plotH)
$g.DrawLine($axisPen,$plotX,$plotY+$plotH,$plotX+$plotW,$plotY+$plotH)
Draw-Text $g 'Unit rank [% of task units]' ($plotX+250) ($plotY+$plotH+55) 500 45 22 $false $Dark 'Center' 'Center'
Draw-Text $g 'Normalized L2 score' $plotX 132 300 36 19 $false $Gray 'Near' 'Center'
Draw-Text $g 'Lowest 30% removed' ($plotX+15) ($plotY+18) ($plotW*0.30-30) 40 20 $true $Red 'Center' 'Center'

foreach ($task in @('SOC','SOH')) {
    $rows = @($saliency | Where-Object Task -eq $task | Sort-Object { [int]$_.RankAscending })
    $scores = @($rows | ForEach-Object { D $_.Score })
    $minScore = ($scores | Measure-Object -Minimum).Minimum
    $maxScore = ($scores | Measure-Object -Maximum).Maximum
    $color = if ($task -eq 'SOC') { $Red } else { $Blue }
    $pen = New-Object System.Drawing.Pen((C $color),5)
    $previous=$null
    for ($i=0;$i -lt $rows.Count;$i++) {
        $rankPct=100.0*$i/[math]::Max(1,$rows.Count-1)
        $norm=((D $rows[$i].Score)-$minScore)/[math]::Max(1e-12,$maxScore-$minScore)
        $point=New-Object System.Drawing.PointF(($plotX+$plotW*$rankPct/100.0),($plotY+$plotH*(1.0-$norm)))
        if ($null -ne $previous) { $g.DrawLine($pen,$previous,$point) }
        $previous=$point
    }
    $pen.Dispose()
}
Draw-Text $g 'SOC' ($plotX+700) ($plotY+90) 90 35 22 $true $Red 'Near' 'Center'
Draw-Text $g 'SOH' ($plotX+830) ($plotY+90) 90 35 22 $true $Blue 'Near' 'Center'
$gridPen.Dispose(); $axisPen.Dispose()

Draw-Box $g 1300 170 300 115 $VeryPaleBlue $Blue 'Four gate rows per hidden unit' 21 $true
Draw-Arrow $g 1600 228 1680 228
Draw-Box $g 1685 170 280 115 $VeryPaleRed $Red 'Aggregate row L2 norms' 21 $true
Draw-Arrow $g 1965 228 2045 228
Draw-Box $g 2050 170 270 115 $VeryPaleGreen $Green 'Remove one complete unit' 21 $true
Draw-Arrow $g 2185 285 2185 355
Draw-Box $g 1690 360 495 105 $VeryPaleGreen $Green 'Slice recurrent columns and MLP input' 21 $false

$tableX=1280; $tableY=540; $tableW=1040; $rowH=125
$colWidths=@(270,245,245,280)
$headers=@('Criterion','Required evidence','Unit grouping','Dense-kernel effect')
$rows=@(
    @('Gate-group L2`n(used)','Saved weights only','All four gates`ncombined','Direct dimension`nreduction'),
    @('Gradient sensitivity','Training data and`nbackpropagation','Must be designed`nas structured','Only if complete`nunits are removed'),
    @('Activation statistics','Calibration stream','Must aggregate`nunit activations','Only if complete`nunits are removed')
)
$xx=$tableX
for($c=0;$c -lt $headers.Count;$c++) {
    Draw-Box $g $xx $tableY $colWidths[$c] 85 '#E9EEF1' $Gray $headers[$c] 19 $true
    $xx += $colWidths[$c]
}
for($r=0;$r -lt $rows.Count;$r++) {
    $xx=$tableX; $fill=if($r -eq 0){$VeryPaleRed}else{'#F8FAFB'}; $border=if($r -eq 0){$Red}else{$Gray}
    for($c=0;$c -lt 4;$c++) {
        Draw-Box $g $xx ($tableY+85+$r*$rowH) $colWidths[$c] $rowH $fill $border ($rows[$r][$c] -replace '`n',"`n") 19 ($r -eq 0)
        $xx += $colWidths[$c]
    }
}
Draw-Box $g 1325 1075 950 125 '#FFF7E8' $Red "Method-property rationale only. Gradient- and activation-based`ncriteria were not experimentally compared in this study." 22 $true
Save-Canvas $canvas 'rev_8_pruning_criterion_scope.png'

# -----------------------------------------------------------------------------
# rev_9: exact mixed-precision boundary and persistent storage composition.
# -----------------------------------------------------------------------------
$memory = Import-Csv (Join-Path $ResultsRoot 'weights\quantization_memory_accounting.csv')
$canvas = New-Canvas 2500 1450
$g = $canvas.Graphics
Draw-Text $g '(a) Precision path in the exported quantized kernel' 70 45 1080 70 32 $true

Draw-Box $g 90 170 310 145 $VeryPaleGreen $Green "Input x and previous`nhidden/cell states`nFP32" 23 $true
Draw-Arrow $g 400 242 500 242
Draw-Box $g 505 150 460 185 $VeryPaleBlue $Blue "Recurrent matrices INT8`n+ per-row FP32 scales`nWeights reconstructed during`nmultiply-accumulate" 22 $true
Draw-Arrow $g 735 335 735 430
Draw-Box $g 505 435 460 145 $VeryPaleGreen $Green "Biases, gate activations,`nhidden state and cell state`nremain FP32" 22 $true
Draw-Arrow $g 735 580 735 675
Draw-Box $g 505 680 460 135 $VeryPaleGreen $Green "MLP weights and`nactivations remain FP32" 23 $true
Draw-Arrow $g 735 815 735 910
Draw-Box $g 505 915 460 120 $VeryPaleGreen $Green "Estimator output`nFP32" 24 $true
Draw-Box $g 110 1090 1030 145 '#FFF7E8' $Red "The export is weight-only mixed precision, not a fully integer path.`nTransient activation buffers were not separately profiled." 23 $true

$socRows = @($memory | Where-Object Task -eq 'SOC' | Sort-Object { if($_.Variant -eq 'Base'){0}else{1} })
$sohRows = @($memory | Where-Object Task -eq 'SOH' | Sort-Object { if($_.Variant -eq 'Base'){0}else{1} })
Draw-StackedStorageBars $g 1240 160 560 980 $socRows 100 '(b) SOC persistent storage'
Draw-StackedStorageBars $g 1870 160 560 980 $sohRows 360 '(c) SOH persistent storage'

$legendItems=@(
    @('FP32 recurrent weights',$PaleGreen,$Green),
    @('INT8 recurrent weights',$PaleBlue,$Blue),
    @('FP32 MLP',$PalePurple,$Purple),
    @('FP32 scales and bias',$PaleRed,$Red),
    @('FP32 persistent h+c state',$PaleGray,$Gray)
)
$lx=1240; $ly=1210
for($i=0;$i -lt $legendItems.Count;$i++) {
    $row=[math]::Floor($i/3); $col=$i%3; $xx=$lx+$col*390; $yy=$ly+$row*72
    Draw-Box $g $xx $yy 48 34 $legendItems[$i][1] $legendItems[$i][2]
    Draw-Text $g $legendItems[$i][0] ($xx+60) ($yy-5) 315 46 18 $false $Dark 'Near' 'Center'
}
Save-Canvas $canvas 'rev_9_mixed_precision_quantization.png'

# -----------------------------------------------------------------------------
# rev_10: static operation accounting beside the observed STM32 kernel times.
# It does not claim cycle-level attribution or measured memory bandwidth.
# -----------------------------------------------------------------------------
$operations = Import-Csv (Join-Path $ResultsRoot 'operations\static_kernel_operation_counts.csv')
$chart = New-Chart 2300 1500
$panels=@(
    @('SocOps',5,4,43,41,'Model variant','Operations per inference','(a) SOC static operation accounting'),
    @('SocTime',53,4,43,41,'Model variant','Observed inference time [ms]','(b) SOC measured kernel time'),
    @('SohOps',5,51,43,41,'Model variant','Operations per inference','(c) SOH static operation accounting'),
    @('SohTime',53,51,43,41,'Model variant','Observed inference time [ms]','(d) SOH measured kernel time')
)
foreach($p in $panels) {
    $area=Add-Area $chart $p[0] $p[1] $p[2] $p[3] $p[4] $p[5] $p[6]
    Add-PanelTitle $chart $p[0] $p[7]
    $area.AxisX.IsMarginVisible=$true
}
$chart.ChartAreas['SocOps'].AxisY.Maximum=45000; $chart.ChartAreas['SocOps'].AxisY.Interval=10000
$chart.ChartAreas['SohOps'].AxisY.Maximum=170000; $chart.ChartAreas['SohOps'].AxisY.Interval=40000
$chart.ChartAreas['SocTime'].AxisY.Maximum=8; $chart.ChartAreas['SocTime'].AxisY.Interval=2
$chart.ChartAreas['SohTime'].AxisY.Maximum=35; $chart.ChartAreas['SohTime'].AxisY.Interval=5

foreach($task in @('SOC','SOH')) {
    $opsArea=if($task -eq 'SOC'){'SocOps'}else{'SohOps'}
    $timeArea=if($task -eq 'SOC'){'SocTime'}else{'SohTime'}
    $rows=@($operations | Where-Object Task -eq $task)
    foreach($component in @('TotalMACs','AdditionalDequantScaleMultiplications')) {
        $series=New-Object System.Windows.Forms.DataVisualization.Charting.Series "$opsArea-$component"
        $series.ChartArea=$opsArea; $series.ChartType='StackedColumn'; $series.IsVisibleInLegend=$false
        $series['PointWidth']='0.58'
        $series.Color=C $(if($component -eq 'TotalMACs'){$PaleGray}else{$PaleBlue})
        $series.BorderColor=C $(if($component -eq 'TotalMACs'){$Gray}else{$Blue})
        $series.BorderWidth=2
        foreach($variant in @('Base','Pruned','Quantized')) {
            $row=$rows | Where-Object Variant -eq $variant | Select-Object -First 1
            $idx=$series.Points.AddXY($variant,(D $row.$component))
            if($component -eq 'AdditionalDequantScaleMultiplications' -and (D $row.$component) -gt 0) {
                $series.Points[$idx].Label=('{0:N0}' -f (D $row.$component))
                $series.Points[$idx].Font=Font 13 $true
                $series.Points[$idx].LabelForeColor=C $Blue
            }
        }
        $chart.Series.Add($series)
    }
    $timeSeries=New-Object System.Windows.Forms.DataVisualization.Charting.Series "$timeArea-Time"
    $timeSeries.ChartArea=$timeArea; $timeSeries.ChartType='Column'; $timeSeries.IsVisibleInLegend=$false
    $timeSeries['PointWidth']='0.58'
    foreach($variant in @('Base','Pruned','Quantized')) {
        $row=$rows | Where-Object Variant -eq $variant | Select-Object -First 1
        $fill=if($variant -eq 'Base'){$PaleGreen}elseif($variant -eq 'Pruned'){$PaleRed}else{$PaleBlue}
        $border=if($variant -eq 'Base'){$Green}elseif($variant -eq 'Pruned'){$Red}else{$Blue}
        $idx=$timeSeries.Points.AddXY($variant,(D $row.ObservedInferenceMs))
        $timeSeries.Points[$idx].Color=C $fill; $timeSeries.Points[$idx].BorderColor=C $border; $timeSeries.Points[$idx].BorderWidth=3
        $timeSeries.Points[$idx].Label=('{0:0.00}' -f (D $row.ObservedInferenceMs))
        $timeSeries.Points[$idx].Font=Font 14 $true; $timeSeries.Points[$idx].LabelForeColor=C $border
    }
    $chart.Series.Add($timeSeries)
}

foreach($areaName in @('SocOps','SohOps')) {
    $legend=Add-AreaLegend $chart "$areaName-Legend" $areaName 'Near'
    Add-CustomBoxLegendItem $legend 'Model MACs' $PaleGray $Gray
    Add-CustomBoxLegendItem $legend 'Additional FP32 scale multiplications' $PaleBlue $Blue
}
Save-Chart $chart 'rev_10_quantized_runtime_accounting.png'

Write-Host 'Reviewer 3 figure generation complete.'
