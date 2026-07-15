param(
    [string]$ReviewRoot = ''
)

$ErrorActionPreference = 'Stop'
$Inv = [System.Globalization.CultureInfo]::InvariantCulture
if (-not $ReviewRoot) {
    $ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
} else {
    $ReviewRoot = (Resolve-Path $ReviewRoot).Path
}

$ResultsRoot = Join-Path $ReviewRoot 'review_analysis\results'
$FiguresRoot = Join-Path $ReviewRoot 'figures\Review_1_Additional'
New-Item -ItemType Directory -Force -Path $FiguresRoot | Out-Null

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$BaseColor = '#2CA02C'
$PrunedColor = '#D62728'
$QuantColor = '#1F77B4'
$PurpleColor = '#9467BD'
$GrayColor = '#67727A'
$LightGray = '#D8DEE2'
$BlackColor = '#222222'
$ModelColors = @{ Base=$BaseColor; Pruned=$PrunedColor; Quantized=$QuantColor }
$FillColors = @{
    Base='#A6D7A6'; Pruned='#EEA4A5'; Quantized='#A1C6E0';
    Corrupted='#EEA4A5'; Mitigated='#A6D7A6'; Removed='#E5E9EC'; Retained='#F2B4B5'
}

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

function New-Figure([string]$title, [int]$width=2200, [int]$height=1450, [string]$subtitle='') {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = $width
    $chart.Height = $height
    $chart.BackColor = [System.Drawing.Color]::White
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
    $area.AxisY.MajorGrid.LineColor = C '#D9DEE1'
    $area.AxisX.LineColor = C '#4E5960'
    $area.AxisY.LineColor = C '#4E5960'
    $area.AxisX.MajorTickMark.LineColor = C '#4E5960'
    $area.AxisY.MajorTickMark.LineColor = C '#4E5960'
    $area.AxisX.IsMarginVisible = $false
    $chart.ChartAreas.Add($area)
    return $area
}

function Add-PanelTitle($chart, [string]$areaName, [string]$text) {
    $title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $title.Text = $text
    $title.DockedToChartArea = $areaName
    $title.IsDockedInsideChartArea = $false
    $title.Alignment = 'MiddleLeft'
    $title.Font = Font 18 $true
    $title.ForeColor = C '#172026'
    $chart.Titles.Add($title)
}

function Add-Legend($chart, [string]$name='Legend', [string]$dock='Bottom') {
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend $name
    $legend.Docking = $dock
    $legend.Alignment = 'Center'
    $legend.Font = Font 15
    $legend.BackColor = [System.Drawing.Color]::White
    $chart.Legends.Add($legend)
    return $legend
}

function Add-AreaLegend($chart, [string]$name, [string]$areaName, [string]$alignment='Near') {
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend $name
    $legend.DockedToChartArea = $areaName
    $legend.IsDockedInsideChartArea = $true
    $legend.Docking = 'Top'
    $legend.Alignment = $alignment
    $legend.LegendStyle = 'Row'
    $legend.Font = Font 13
    $legend.BackColor = [System.Drawing.Color]::FromArgb(230,255,255,255)
    $chart.Legends.Add($legend)
    return $legend
}

function Add-AreaBottomLegend($chart, [string]$name, [string]$areaName) {
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend $name
    $legend.DockedToChartArea = $areaName
    $legend.IsDockedInsideChartArea = $false
    $legend.Docking = 'Bottom'
    $legend.Alignment = 'Center'
    $legend.LegendStyle = 'Row'
    $legend.Font = Font 14
    $legend.BackColor = [System.Drawing.Color]::White
    $chart.Legends.Add($legend)
    return $legend
}

function Add-CustomLegendItem($legend, [string]$name, [string]$fill, [string]$border) {
    $item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
    $item.Name = $name
    $item.Color = C $fill
    $item.BorderColor = C $border
    $item.BorderWidth = 2
    $item.ImageStyle = 'Rectangle'
    $legend.CustomItems.Add($item)
}

function Add-CustomLineLegendItem($legend, [string]$name, [string]$color) {
    $item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
    $item.Name = $name
    $item.Color = C $color
    $item.BorderColor = C $color
    $item.BorderWidth = 4
    $item.ImageStyle = 'Line'
    $legend.CustomItems.Add($item)
}

function Add-Line($chart, [string]$area, [string]$name, [string]$color, [object[]]$rows,
    [string]$xProperty, [string]$yProperty, [bool]$legend=$false, [string]$dash='Solid',
    [bool]$markers=$false, [int]$width=3) {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series "$area-$name-$($chart.Series.Count)"
    $series.LegendText = $name
    $series.ChartArea = $area
    $series.ChartType = 'Line'
    $series.Color = C $color
    $series.BorderWidth = $width
    $series.BorderDashStyle = $dash
    $series.IsVisibleInLegend = $legend
    if ($markers) {
        $series.MarkerStyle = 'Circle'
        $series.MarkerSize = 7
        $series.MarkerColor = C '#FFFFFF'
        $series.MarkerBorderColor = C $color
        $series.MarkerBorderWidth = 2
    }
    foreach ($row in $rows) {
        [void]$series.Points.AddXY((D $row.$xProperty), (D $row.$yProperty))
    }
    $chart.Series.Add($series)
}

function Add-GroupedBars($chart, [string]$area, [object[]]$rows, [string]$categoryProperty,
    [string]$seriesProperty, [string]$valueProperty, [hashtable]$outlineColors,
    [hashtable]$fillColors, [bool]$labels=$false, [bool]$legend=$false) {
    $categories = @($rows | Select-Object -ExpandProperty $categoryProperty -Unique)
    $seriesNames = @($rows | Select-Object -ExpandProperty $seriesProperty -Unique)
    foreach ($seriesName in $seriesNames) {
        $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series ([string]$seriesName)
        $series.ChartArea = $area
        $series.ChartType = 'Column'
        $series.Color = C ([string]$fillColors[[string]$seriesName])
        $series.BorderColor = C ([string]$outlineColors[[string]$seriesName])
        $series.BorderWidth = 2
        $series.IsVisibleInLegend = $legend
        $series['PointWidth'] = '0.62'
        foreach ($category in $categories) {
            $row = $rows | Where-Object { $_.$categoryProperty -eq $category -and $_.$seriesProperty -eq $seriesName } | Select-Object -First 1
            $value = if ($row) { D $row.$valueProperty } else { 0.0 }
            $index = $series.Points.AddXY([string]$category, $value)
            if ($labels -and $row) {
                $series.Points[$index].Label = ('{0:0.0}' -f $value)
                $series.Points[$index].Font = Font 14 $true
                $series.Points[$index].LabelForeColor = C ([string]$outlineColors[[string]$seriesName])
            }
        }
        $chart.Series.Add($series)
    }
}

function Save-Figure($chart, [string]$name) {
    $path = Join-Path $FiguresRoot $name
    $chart.SaveImage($path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $chart.Dispose()
    Write-Host "Saved $path"
}

# rev_1: SOC pruning evidence
$saliency = Import-Csv (Join-Path $ResultsRoot 'weights\lstm_unit_saliency.csv') |
    Where-Object Task -eq 'SOC' | Sort-Object { [int]$_.RankAscending }
$weights = Import-Csv (Join-Path $ResultsRoot 'weights\lstm_weight_histograms.csv') |
    Where-Object { $_.Task -eq 'SOC' -and [math]::Abs((D $_.BinCenter)) -le 0.8 }

$chart = New-Figure 'Structured SOC pruning diagnostics'
$left = Add-Area $chart 'Saliency' 6 5 43 88 'Unit rank (ascending saliency)' 'L2 saliency score'
$right = Add-Area $chart 'Weights' 54 5 43 88 'Recurrent-weight value' 'Density'
Add-PanelTitle $chart 'Saliency' '(a) Unit saliency before one-shot pruning'
Add-PanelTitle $chart 'Weights' '(b) Central recurrent-weight distribution'
$salSeries = New-Object System.Windows.Forms.DataVisualization.Charting.Series 'Saliency'
$salSeries.ChartArea = 'Saliency'; $salSeries.ChartType = 'Column'; $salSeries.IsVisibleInLegend = $false
$salSeries['PointWidth'] = '0.78'
foreach ($row in $saliency) {
    $selected = [System.Convert]::ToBoolean($row.SelectedByMagnitude)
    $idx = $salSeries.Points.AddXY([int]$row.RankAscending, (D $row.Score))
    $salSeries.Points[$idx].Color = C $(if ($selected) { $FillColors.Retained } else { $FillColors.Removed })
    $salSeries.Points[$idx].BorderColor = C $(if ($selected) { $PrunedColor } else { $GrayColor })
    $salSeries.Points[$idx].BorderWidth = 1
}
$chart.Series.Add($salSeries)
$left.AxisX.Minimum = 0; $left.AxisX.Maximum = 65; $left.AxisX.Interval = 10
foreach ($model in @('Base','Pruned')) {
    $subset = @($weights | Where-Object Variant -eq $model | Sort-Object { D $_.BinCenter })
    Add-Line $chart 'Weights' $model $ModelColors[$model] $subset 'BinCenter' 'Density' $false 'Solid' $false 4
}
$right.AxisX.Minimum = -0.8; $right.AxisX.Maximum = 0.8; $right.AxisX.Interval = 0.2
$legendA = Add-AreaLegend $chart 'SaliencyLegend' 'Saliency' 'Near'
Add-CustomLegendItem $legendA 'Removed units (19)' $FillColors.Removed $GrayColor
Add-CustomLegendItem $legendA 'Retained units (45)' $FillColors.Retained $PrunedColor
$legendB = Add-AreaLegend $chart 'WeightLegend' 'Weights' 'Far'
Add-CustomLineLegendItem $legendB 'Base weights' $BaseColor
Add-CustomLineLegendItem $legendB 'Pruned weights' $PrunedColor
Save-Figure $chart 'rev_1_pruning_evidence.png'

# rev_2: long-horizon stability
$chart = New-Figure 'Long-horizon stability of compressed estimators'
$positions = @(
    @('SocMae',5,4,43,41,'Sequence segment','MAE [% of full scale]','(a) SOC error by sequence segment'),
    @('SocDev',53,4,43,41,'Sequence segment','Deviation from Base [percentage points]','(b) SOC compressed-to-Base deviation'),
    @('SohMae',5,50,43,41,'Sequence segment','MAE [% of full scale]','(c) SOH error by sequence segment'),
    @('SohDev',53,50,43,41,'Sequence segment','Deviation from Base [percentage points]','(d) SOH compressed-to-Base deviation')
)
foreach ($p in $positions) {
    $a = Add-Area $chart $p[0] $p[1] $p[2] $p[3] $p[4] $p[5] $p[6]
    $a.AxisX.Minimum = 1; $a.AxisX.Maximum = 10; $a.AxisX.Interval = 1
    Add-PanelTitle $chart $p[0] $p[7]
}
foreach ($task in @('soc','soh')) {
    $areaMae = if ($task -eq 'soc') { 'SocMae' } else { 'SohMae' }
    $areaDev = if ($task -eq 'soc') { 'SocDev' } else { 'SohDev' }
    $windowRows = Import-Csv (Join-Path $ResultsRoot "long_horizon\${task}_windowed_stability.csv")
    foreach ($model in @('Base','Pruned','Quantized')) {
        $subset = @($windowRows | Where-Object Model -eq $model | Sort-Object { [int]$_.Window })
        Add-Line $chart $areaMae $model $ModelColors[$model] $subset 'Window' 'MAE_pct' $false 'Solid' $true 3
    }
    $deviationRows = Import-Csv (Join-Path $ResultsRoot "long_horizon\${task}_compression_deviation.csv")
    foreach ($model in @('Pruned','Quantized')) {
        $subset = @($deviationRows | Where-Object Variant -eq $model | Sort-Object { [int]$_.Window })
        Add-Line $chart $areaDev $model $ModelColors[$model] $subset 'Window' 'MeanAbsDeviation_pp' $false 'Solid' $true 3
    }
}
$legend = Add-Legend $chart
foreach ($model in @('Base','Pruned','Quantized')) { Add-CustomLineLegendItem $legend $model $ModelColors[$model] }
Save-Figure $chart 'rev_2_long_horizon_stability.png'

# rev_3: verified sequential SOH filter pipeline
$trajectoryPath = Join-Path $ResultsRoot 'filter\soh_filter_compression_local_trajectory.csv'
$metricPath = Join-Path $ResultsRoot 'filter\soh_filter_compression_local_windows.csv'
$trajectory = Import-Csv $trajectoryPath
$metrics = Import-Csv $metricPath
if (-not ($trajectory[0].PSObject.Properties.Name -contains 'BaseSequential')) {
    throw 'Sequential SOH filter columns are missing. Re-run RunLocalWindowsFilterAnalysis first.'
}
$chart = New-Figure 'SOH filtering and compression interaction' 2200 1500 'Local C re-execution; EMA-only response at 1 Hz: T90 = 114 s (alpha=0.02) and 26.65 d (alpha=1e-6)'
$filterPanels = @(
    @('Raw',5,4,43,39,'Raw','(a) First-point calibration only'),
    @('Stage1',53,4,43,39,'Benchmark','(b) Stage 1: alpha=0.02 + symmetric limiter'),
    @('Final',5,50,43,39,'Sequential','(c) Final: Stage 1 + alpha=1e-6 + downward limiter')
)
foreach ($p in $filterPanels) {
    $a = Add-Area $chart $p[0] $p[1] $p[2] $p[3] $p[4] 'Sequence progress [%]' 'SOH [-]'
    $a.AxisX.Minimum = 0; $a.AxisX.Maximum = 100; $a.AxisX.Interval = 20
    $a.AxisY.Minimum = 0.84; $a.AxisY.Maximum = 1.04; $a.AxisY.Interval = 0.04
    Add-PanelTitle $chart $p[0] $p[6]
    Add-Line $chart $p[0] 'Reference' $BlackColor $trajectory 'ProgressPercent' 'Reference' $false 'Solid' $false 3
    foreach ($model in @('Base','Pruned','Quantized')) {
        Add-Line $chart $p[0] $model $ModelColors[$model] $trajectory 'ProgressPercent' "$model$($p[5])" $false 'Solid' $false 3
    }
}
$barArea = Add-Area $chart 'FilterMae' 53 50 43 39 '' 'MAE [% of full scale]'
$barArea.AxisX.Interval = 1
Add-PanelTitle $chart 'FilterMae' '(d) Accuracy after each processing stage'
$barRows = @()
$filterMap = [ordered]@{
    'Raw'='Raw_first-point-scaled'
    'Stage 1'='BenchmarkCode_alpha0.02_symmetric-cap'
    'Final'='FinalSequential_alpha0.02_then_alpha1e-6'
}
foreach ($category in $filterMap.Keys) {
    foreach ($model in @('Base','Pruned','Quantized')) {
        $row = $metrics | Where-Object { $_.Model -eq $model -and $_.Filter -eq $filterMap[$category] } | Select-Object -First 1
        $barRows += [pscustomobject]@{Stage=$category;Model=$model;Value=$row.MAE_pct}
    }
}
Add-GroupedBars $chart 'FilterMae' $barRows 'Stage' 'Model' 'Value' $ModelColors $FillColors $true $false
$legend = Add-Legend $chart
Add-CustomLineLegendItem $legend 'Reference' $BlackColor
foreach ($model in @('Base','Pruned','Quantized')) { Add-CustomLineLegendItem $legend $model $ModelColors[$model] }
Save-Figure $chart 'rev_3_soh_filter_pipeline.png'

# rev_4: utility sensitivity
$sweep = Import-Csv (Join-Path $ResultsRoot 'utility\utility_priority_sweep.csv')
$ranking = Import-Csv (Join-Path $ResultsRoot 'utility\utility_ranking_summary.csv')
$chart = New-Figure 'Utility-ranking sensitivity to application priorities' 2200 1500 'At 25%, all four objectives are weighted equally; the remaining weight is shared equally among the other objectives'
$socArea = Add-Area $chart 'UtilitySOC' 5 4 43 37 'Weight of highlighted objective [%]' ''
$sohArea = Add-Area $chart 'UtilitySOH' 53 4 43 37 'Weight of highlighted objective [%]' ''
$rankArea = Add-Area $chart 'UtilityRank' 18 51 64 36 '' 'Weight combinations ranked best [%]'
Add-PanelTitle $chart 'UtilitySOC' '(a) SOC winning model in focused-priority sweeps'
Add-PanelTitle $chart 'UtilitySOH' '(b) SOH winning model in focused-priority sweeps'
Add-PanelTitle $chart 'UtilityRank' '(c) Ranking across all 1,771 weight combinations'
$metricY = @{ Accuracy=4; Flash=3; RAM=2; Energy=1 }
foreach ($task in @('SOC','SOH')) {
    $areaName = if ($task -eq 'SOC') { 'UtilitySOC' } else { 'UtilitySOH' }
    $area = $chart.ChartAreas[$areaName]
    $area.AxisX.Minimum = 22.5; $area.AxisX.Maximum = 87.5; $area.AxisX.Interval = 10
    foreach ($tick in 25,35,45,55,65,75,85) {
        $area.AxisX.CustomLabels.Add($tick-2.5,$tick+2.5,[string]$tick) | Out-Null
    }
    $area.AxisY.Minimum = 0.5; $area.AxisY.Maximum = 4.5; $area.AxisY.Interval = 1
    $area.AxisY.MajorGrid.Enabled = $false
    $area.AxisY.CustomLabels.Add(0.5,1.5,'Energy') | Out-Null
    $area.AxisY.CustomLabels.Add(1.5,2.5,'RAM') | Out-Null
    $area.AxisY.CustomLabels.Add(2.5,3.5,'Flash') | Out-Null
    $area.AxisY.CustomLabels.Add(3.5,4.5,'Accuracy') | Out-Null
    $taskRows = @($sweep | Where-Object Task -eq $task)
    foreach ($winner in @('Base','Pruned','Quantized')) {
        $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series "$task-$winner"
        $series.ChartArea = $areaName; $series.ChartType = 'Point'; $series.IsVisibleInLegend = $false
        $series.MarkerStyle = 'Square'; $series.MarkerSize = 38
        $series.MarkerColor = C $FillColors[$winner]; $series.MarkerBorderColor = C $ModelColors[$winner]; $series.MarkerBorderWidth = 2
        foreach ($row in ($taskRows | Where-Object Winner -eq $winner)) {
            [void]$series.Points.AddXY(100*(D $row.FocalWeight), $metricY[$row.FocalMetric])
        }
        $chart.Series.Add($series)
    }
}
$rankingRows = @($ranking | ForEach-Object {
    [pscustomobject]@{Task=$_.Task;Model=$_.Variant;Value=100*(D $_.WinningShare)}
})
Add-GroupedBars $chart 'UtilityRank' $rankingRows 'Task' 'Model' 'Value' $ModelColors $FillColors $true $false
$rankArea.AxisX.Interval = 1; $rankArea.AxisY.Minimum = 0; $rankArea.AxisY.Maximum = 100; $rankArea.AxisY.Interval = 20
$legend = Add-Legend $chart
foreach ($model in @('Base','Pruned','Quantized')) { Add-CustomLegendItem $legend $model $FillColors[$model] $ModelColors[$model] }
Save-Figure $chart 'rev_4_utility_sensitivity.png'

# rev_5: limited stored-output fault sensitivity
$faultRows = @()
foreach ($task in @('SOC','SOH')) {
    $path = Join-Path $ResultsRoot "faults\$($task.ToLowerInvariant())_output_bitflip_robustness.csv"
    $faultRows += Import-Csv $path | Where-Object BitClass -eq 'AnyBit'
}
$chart = New-Figure 'Limited software-level output fault sensitivity' 2200 1400 'Random single-bit corruption of the FP32 estimator output; 20,000 trials per model (not internal MCU, weight, activation, or recurrent-state injection)'
$catArea = Add-Area $chart 'Catastrophic' 5 5 43 82 '' 'Large-error outcomes (>10 percentage points) [% of trials]'
$mitArea = Add-Area $chart 'Mitigation' 53 5 43 82 '' '95th-percentile absolute error [percentage points]'
Add-PanelTitle $chart 'Catastrophic' '(a) Unmitigated corruption of stored FP32 outputs'
Add-PanelTitle $chart 'Mitigation' '(b) Effect of range check and hold-last'
$catRows = @($faultRows | ForEach-Object {
    [pscustomobject]@{Task=$_.Task;Model=$_.Model;Value=100*(D $_.CatastrophicRate_gt10pp)}
})
Add-GroupedBars $chart 'Catastrophic' $catRows 'Task' 'Model' 'Value' $ModelColors $FillColors $true $false
$catArea.AxisX.Interval = 1; $catArea.AxisY.Minimum = 0; $catArea.AxisY.Maximum = 40; $catArea.AxisY.Interval = 10
$mitRows = @()
foreach ($row in $faultRows) {
    $category = "$($row.Task) $($row.Model)"
    $mitRows += [pscustomobject]@{Category=$category;Condition='Corrupted';Value=$row.P95CorruptedError_pct}
    $mitRows += [pscustomobject]@{Category=$category;Condition='Mitigated';Value=$row.P95MitigatedError_pct}
}
$conditionOutline = @{Corrupted=$PrunedColor;Mitigated=$BaseColor}
$conditionFill = @{Corrupted=$FillColors.Corrupted;Mitigated=$FillColors.Mitigated}
Add-GroupedBars $chart 'Mitigation' $mitRows 'Category' 'Condition' 'Value' $conditionOutline $conditionFill $true $false
$mitArea.AxisX.Interval = 1; $mitArea.AxisX.LabelStyle.Angle = -25; $mitArea.AxisY.Minimum = 0; $mitArea.AxisY.Maximum = 200; $mitArea.AxisY.Interval = 50
$modelLegend = Add-AreaBottomLegend $chart 'ModelLegend' 'Catastrophic'
foreach ($model in @('Base','Pruned','Quantized')) { Add-CustomLegendItem $modelLegend $model $FillColors[$model] $ModelColors[$model] }
$mitigationLegend = Add-AreaBottomLegend $chart 'MitigationLegend' 'Mitigation'
Add-CustomLegendItem $mitigationLegend 'Corrupted output' $FillColors.Corrupted $PrunedColor
Add-CustomLegendItem $mitigationLegend 'Range check + hold-last' $FillColors.Mitigated $BaseColor
Save-Figure $chart 'rev_5_limited_fault_sensitivity.png'

Write-Host 'Reviewer 1 figures completed.'
