param(
    [string]$WorkspaceRoot = '',
    [switch]$SkipSequenceAnalyses
)

$ErrorActionPreference = 'Stop'
$Inv = [System.Globalization.CultureInfo]::InvariantCulture
$ReviewRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
if (-not $WorkspaceRoot) {
    $WorkspaceRoot = (Resolve-Path (Join-Path $ReviewRoot '..\..\..\..\..')).Path
} else {
    $WorkspaceRoot = (Resolve-Path $WorkspaceRoot).Path
}

$ResultsRoot = Join-Path $ReviewRoot 'review_analysis\results'
$FiguresRoot = Join-Path $ReviewRoot 'figures\Review_1_Additional'
New-Item -ItemType Directory -Force -Path $ResultsRoot,$FiguresRoot | Out-Null

Add-Type -AssemblyName System.IO.Compression.FileSystem
Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing
Add-Type -Path (Join-Path $PSScriptRoot 'ReviewAnalysisCore.cs') -ReferencedAssemblies @(
    'System.IO.Compression.FileSystem.dll',
    'System.IO.Compression.dll',
    'System.dll',
    'System.Core.dll'
)

$BaseColor = '#2CA02C'
$PrunedColor = '#D62728'
$QuantColor = '#1F77B4'
$PurpleColor = '#9467BD'
$GrayColor = '#67727A'
$LightGray = '#D8DEE2'
$ModelColors = @{ Base=$BaseColor; Pruned=$PrunedColor; Quantized=$QuantColor }
$FillColors = @{
    $BaseColor='#A6D7A6'; $PrunedColor='#EEA4A5'; $QuantColor='#A1C6E0';
    $PurpleColor='#D2BFE3'; $GrayColor='#D8DEE2'
}

function Resolve-SourcePath([string]$RelativePath) {
    $path = Join-Path $WorkspaceRoot $RelativePath
    if (-not (Test-Path -LiteralPath $path)) { throw "Required source not found: $path" }
    return (Resolve-Path $path).Path
}

function To-Double([object]$Value) {
    return [double]::Parse([string]$Value, [System.Globalization.NumberStyles]::Float, $Inv)
}

function Write-CsvRows([object[]]$Rows, [string]$Path) {
    $parent = Split-Path -Parent $Path
    New-Item -ItemType Directory -Force -Path $parent | Out-Null
    $normalized = foreach ($row in $Rows) {
        $values = [ordered]@{}
        foreach ($property in $row.PSObject.Properties) {
            $value = $property.Value
            if ($value -is [double] -or $value -is [single] -or $value -is [decimal]) {
                $values[$property.Name] = ([IFormattable]$value).ToString('G17', $Inv)
            } else {
                $values[$property.Name] = $value
            }
        }
        [pscustomobject]$values
    }
    $normalized | Export-Csv -LiteralPath $Path -NoTypeInformation -Encoding UTF8
}

function Get-HexColor([string]$Hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($Hex)
}

function New-Chart([string]$Title, [string]$XTitle, [string]$YTitle, [int]$Width=1900, [int]$Height=1050) {
    $chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
    $chart.Width = $Width
    $chart.Height = $Height
    $chart.BackColor = [System.Drawing.Color]::White
    $area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea 'Main'
    $area.BackColor = [System.Drawing.Color]::White
    $area.AxisX.Title = $XTitle
    $area.AxisY.Title = $YTitle
    $area.AxisX.TitleFont = New-Object System.Drawing.Font('Arial',13)
    $area.AxisY.TitleFont = New-Object System.Drawing.Font('Arial',13)
    $area.AxisX.LabelStyle.Font = New-Object System.Drawing.Font('Arial',11)
    $area.AxisY.LabelStyle.Font = New-Object System.Drawing.Font('Arial',11)
    $area.AxisX.MajorGrid.LineColor = Get-HexColor '#E3E7E9'
    $area.AxisY.MajorGrid.LineColor = Get-HexColor '#D8DEE2'
    $area.AxisX.LineColor = Get-HexColor '#4E5960'
    $area.AxisY.LineColor = Get-HexColor '#4E5960'
    $area.AxisX.MajorTickMark.LineColor = Get-HexColor '#4E5960'
    $area.AxisY.MajorTickMark.LineColor = Get-HexColor '#4E5960'
    $chart.ChartAreas.Add($area)

    $titleObj = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $titleObj.Text = $Title
    $titleObj.Font = New-Object System.Drawing.Font('Arial',17,[System.Drawing.FontStyle]::Bold)
    $chart.Titles.Add($titleObj)
    $legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend
    $legend.Docking = 'Bottom'
    $legend.Font = New-Object System.Drawing.Font('Arial',11)
    $chart.Legends.Add($legend)
    return $chart
}

function Add-LineSeries($Chart, [string]$Name, [string]$Color, [object[]]$Rows,
    [string]$XProperty, [string]$YProperty, [string]$ChartArea='Main', [int]$Width=3,
    [string]$Dash='Solid') {
    $series = New-Object System.Windows.Forms.DataVisualization.Charting.Series $Name
    $series.ChartType = 'Line'
    $series.ChartArea = $ChartArea
    $series.BorderWidth = $Width
    $series.Color = Get-HexColor $Color
    $series.BorderDashStyle = $Dash
    foreach ($row in $Rows) {
        [void]$series.Points.AddXY((To-Double $row.$XProperty), (To-Double $row.$YProperty))
    }
    $Chart.Series.Add($series)
}

function Save-Chart($Chart, [string]$Path) {
    $Chart.SaveImage($Path, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
    $Chart.Dispose()
}

function New-ModelLineChart([object[]]$Rows, [string]$Title, [string]$XProperty,
    [string]$YProperty, [string]$XTitle, [string]$YTitle, [string]$Path) {
    $chart = New-Chart $Title $XTitle $YTitle
    foreach ($model in @('Base','Pruned','Quantized')) {
        $subset = @($Rows | Where-Object Model -eq $model | Sort-Object { To-Double $_.$XProperty })
        if ($subset.Count) { Add-LineSeries $chart $model $ModelColors[$model] $subset $XProperty $YProperty }
    }
    Save-Chart $chart $Path
}

function Get-CArray([string]$Path, [string]$Name) {
    $text = Get-Content -LiteralPath $Path -Raw
    $pattern = '(?s)\b' + [regex]::Escape($Name) + '(?:\s*\[[^\]]*\])+\s*=\s*\{(?<body>.*?)\};'
    $match = [regex]::Match($text, $pattern)
    if (-not $match.Success) { throw "C array $Name not found in $Path" }
    $numberPattern = '[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?'
    $matches = [regex]::Matches($match.Groups['body'].Value, $numberPattern)
    $values = New-Object double[] $matches.Count
    for ($i=0; $i -lt $matches.Count; $i++) {
        $values[$i] = [double]::Parse($matches[$i].Value, [System.Globalization.NumberStyles]::Float, $Inv)
    }
    return ,$values
}

function Get-CDefine([string]$Path, [string]$Name) {
    $text = Get-Content -LiteralPath $Path -Raw
    $m = [regex]::Match($text, '(?m)^\s*#define\s+' + [regex]::Escape($Name) + '\s+(\d+)')
    if (-not $m.Success) { throw "C define $Name not found in $Path" }
    return [int]$m.Groups[1].Value
}

function Join-DoubleArrays([double[]]$A, [double[]]$B) {
    $result = New-Object double[] ($A.Length + $B.Length)
    [Array]::Copy($A,0,$result,0,$A.Length)
    [Array]::Copy($B,0,$result,$A.Length,$B.Length)
    return ,$result
}

function Get-ArrayStats([double[]]$Values) {
    $sum=0.0; $sumSq=0.0; $sumAbs=0.0; $maxAbs=0.0; $below=0
    $abs = New-Object double[] $Values.Length
    for ($i=0; $i -lt $Values.Length; $i++) {
        $v=$Values[$i]; $a=[math]::Abs($v)
        $sum += $v; $sumSq += $v*$v; $sumAbs += $a
        if ($a -gt $maxAbs) { $maxAbs=$a }
        if ($a -lt 0.01) { $below++ }
        $abs[$i]=$a
    }
    [Array]::Sort($abs)
    $mean=$sum/$Values.Length
    $variance=[math]::Max(0.0,$sumSq/$Values.Length-$mean*$mean)
    return [pscustomobject]@{
        Count=$Values.Length; Mean=$mean; Std=[math]::Sqrt($variance); MeanAbs=$sumAbs/$Values.Length
        P95Abs=$abs[[math]::Floor(0.95*($abs.Length-1))]
        P99Abs=$abs[[math]::Floor(0.99*($abs.Length-1))]
        MaxAbs=$maxAbs; FractionAbsBelow0p01=$below/$Values.Length
    }
}

function Get-HistogramRows([double[]]$Values, [int]$Bins, [double]$Min, [double]$Max,
    [string]$Task, [string]$Variant, [string]$ValueName='Weight') {
    $counts = New-Object int[] $Bins
    $width=($Max-$Min)/$Bins
    foreach ($v in $Values) {
        $idx=[math]::Floor(($v-$Min)/$width)
        if ($idx -lt 0) { $idx=0 }
        if ($idx -ge $Bins) { $idx=$Bins-1 }
        $counts[$idx]++
    }
    $rows=@()
    for ($i=0; $i -lt $Bins; $i++) {
        $rows += [pscustomobject]@{Task=$Task;Variant=$Variant;ValueName=$ValueName;BinCenter=$Min+($i+0.5)*$width;Density=$counts[$i]/($Values.Length*$width)}
    }
    return $rows
}

function Get-UnitSaliency([double[]]$IH, [double[]]$HH, [int]$Hidden, [int]$InputSize, [string]$Task) {
    $scores = New-Object double[] $Hidden
    for ($h=0; $h -lt $Hidden; $h++) {
        $score=0.0
        for ($g=0; $g -lt 4; $g++) {
            $row=$g*$Hidden+$h
            $sumIH=0.0; $sumHH=0.0
            for ($j=0; $j -lt $InputSize; $j++) { $v=$IH[$row*$InputSize+$j]; $sumIH += $v*$v }
            for ($j=0; $j -lt $Hidden; $j++) { $v=$HH[$row*$Hidden+$j]; $sumHH += $v*$v }
            $score += [math]::Sqrt($sumIH)+[math]::Sqrt($sumHH)
        }
        $scores[$h]=$score
    }
    $indices=0..($Hidden-1) | Sort-Object { $scores[$_] }
    $remove=[math]::Round(0.30*$Hidden)
    $rows=@()
    for ($rank=0; $rank -lt $indices.Count; $rank++) {
        $unit=$indices[$rank]
        $rows += [pscustomobject]@{Task=$Task;Unit=$unit;RankAscending=$rank+1;Score=$scores[$unit];SelectedByMagnitude=($rank -ge $remove)}
    }
    return $rows
}

function Get-QuantizationStats([double[]]$Base, [double[]]$Codes, [double[]]$Scales,
    [int]$Rows, [int]$Cols, [string]$Task, [string]$Matrix) {
    $errors=New-Object double[] $Base.Length
    $sumSq=0.0; $sumAbs=0.0; $weightSq=0.0; $max=0.0; $sat=0
    for ($r=0; $r -lt $Rows; $r++) {
        for ($c=0; $c -lt $Cols; $c++) {
            $idx=$r*$Cols+$c
            $recon=$Codes[$idx]*$Scales[$r]
            $e=$recon-$Base[$idx]; $a=[math]::Abs($e)
            $errors[$idx]=$a; $sumSq += $e*$e; $sumAbs += $a; $weightSq += $Base[$idx]*$Base[$idx]
            if ($a -gt $max) { $max=$a }
            if ([math]::Abs($Codes[$idx]) -ge 127) { $sat++ }
        }
    }
    $sorted=[double[]]$errors.Clone(); [Array]::Sort($sorted)
    $rmse=[math]::Sqrt($sumSq/$Base.Length)
    $stats=[pscustomobject]@{
        Task=$Task;Matrix=$Matrix;Count=$Base.Length;MAE=$sumAbs/$Base.Length;RMSE=$rmse
        RelativeRMSE=$rmse/[math]::Sqrt($weightSq/$Base.Length)
        P95Abs=$sorted[[math]::Floor(.95*($sorted.Length-1))];P99Abs=$sorted[[math]::Floor(.99*($sorted.Length-1))]
        MaxAbs=$max;SaturationFraction=$sat/$Base.Length
    }
    return [pscustomobject]@{Stats=$stats;Errors=$errors;P99=$stats.P99Abs}
}

function New-SaliencyChart([object[]]$Rows, [string]$Task, [string]$Path) {
    $chart=New-Chart "$Task LSTM unit saliency before pruning" 'Unit rank (ascending saliency)' 'L2 saliency score'
    $series=New-Object System.Windows.Forms.DataVisualization.Charting.Series 'Units'
    $series.ChartType='Column'; $series.Color=Get-HexColor $LightGray; $series.BorderColor=Get-HexColor $GrayColor; $series.BorderWidth=1
    foreach($r in ($Rows | Sort-Object {[int]$_.RankAscending})) {
        $idx=$series.Points.AddXY([int]$r.RankAscending, [double]$r.Score)
        $series.Points[$idx].Color=Get-HexColor ($(if([bool]$r.SelectedByMagnitude){$PrunedColor}else{$LightGray}))
        $series.Points[$idx].BorderColor=Get-HexColor ($(if([bool]$r.SelectedByMagnitude){'#8F1E1F'}else{$GrayColor}))
    }
    $chart.Series.Add($series)
    Save-Chart $chart $Path
}

function New-GroupedBarChart([object[]]$Rows,[string]$Title,[string]$CategoryProperty,[string]$SeriesProperty,
    [string]$ValueProperty,[string]$YTitle,[hashtable]$Colors,[string]$Path,[bool]$ShowLabels=$false) {
    $chart=New-Chart $Title '' $YTitle
    $categories=@($Rows | Select-Object -ExpandProperty $CategoryProperty -Unique)
    $seriesNames=@($Rows | Select-Object -ExpandProperty $SeriesProperty -Unique)
    foreach($name in $seriesNames) {
        $series=New-Object System.Windows.Forms.DataVisualization.Charting.Series ([string]$name)
        $outline=[string]$Colors[[string]$name]
        $fill=if($FillColors.ContainsKey($outline)){$FillColors[$outline]}else{$outline}
        $series.ChartType='Column'; $series.Color=Get-HexColor $fill; $series.BorderColor=Get-HexColor $outline; $series.BorderWidth=2
        $series['PointWidth']='0.62'
        foreach($cat in $categories) {
            $row=$Rows | Where-Object { $_.$CategoryProperty -eq $cat -and $_.$SeriesProperty -eq $name } | Select-Object -First 1
            if($row){
                $value=To-Double $row.$ValueProperty
                $index=$series.Points.AddXY([string]$cat,$value)
                if($ShowLabels) {
                    $series.Points[$index].Label=('{0:0.0} %' -f $value)
                    $series.Points[$index].Font=New-Object System.Drawing.Font('Arial',11,[System.Drawing.FontStyle]::Bold)
                    $series.Points[$index].LabelForeColor=Get-HexColor $outline
                    if([math]::Abs($value) -lt 1e-12) {
                        $series.Points[$index].MarkerStyle='Square'
                        $series.Points[$index].MarkerSize=10
                        $series.Points[$index].MarkerColor=Get-HexColor $fill
                        $series.Points[$index].MarkerBorderColor=Get-HexColor $outline
                        $series.Points[$index].MarkerBorderWidth=2
                    }
                }
            }
        }
        $chart.Series.Add($series)
    }
    $chart.ChartAreas['Main'].AxisX.Interval=1
    Save-Chart $chart $Path
}

Write-Host "Review root: $ReviewRoot"
Write-Host "Workspace root: $WorkspaceRoot"

$socNpz = Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\5_benchmark\PC\SOC\bench_v_soc_full\soc_streaming_base_quant_pruned_data.npz'
$sohNpz = Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\5_benchmark\PC\SOH\BENCH_SOH_FULL_FINAL_20251124\benchmark_results.npz'
$rawSohNpz = Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\6_test\Python\pruned_SOH\archive\COMPARE_SEQ2MANY_VS_PRUNED_20251122_093717\MGFarm_18650_C07\arrays.npz'
$localBasePrunedNpz=Join-Path $ResultsRoot 'filter\base_pruned_soh_raw_windows_stored.npz'
$localQuantizedNpz=Join-Path $ResultsRoot 'filter\quantized_soh_raw_predictions_stored.npz'

if (-not $SkipSequenceAnalyses) {
    Write-Host 'Running full-sequence, filter, and software fault analyses...'
    [ReviewOne.AnalysisCore]::RunSequenceAnalyses($socNpz,$sohNpz,$rawSohNpz,$ResultsRoot)
}
$savedFilteredTrajectory=Join-Path $ResultsRoot 'filter\soh_saved_filtered_trajectory_downsampled.csv'
if (-not (Test-Path -LiteralPath $savedFilteredTrajectory)) {
    [ReviewOne.AnalysisCore]::WriteDownsampledSohFilteredTrajectory($sohNpz,$savedFilteredTrajectory,2500)
}
if ((Test-Path -LiteralPath $localBasePrunedNpz) -and (Test-Path -LiteralPath $localQuantizedNpz)) {
    [ReviewOne.AnalysisCore]::RunLocalWindowsFilterAnalysis($localBasePrunedNpz,$localQuantizedNpz,$ResultsRoot)
}

# Long-horizon and software fault figures
foreach($task in @('soc','soh')) {
    $pretty=$task.ToUpperInvariant()
    $windows=Import-Csv (Join-Path $ResultsRoot "long_horizon\${task}_windowed_stability.csv")
    New-ModelLineChart $windows "$pretty error stability over the full sequence" 'Window' 'MAE_pct' 'Sequence segment' 'MAE [% of full scale]' (Join-Path $FiguresRoot "review_${task}_windowed_mae.png")
    New-ModelLineChart $windows "$pretty tail-error stability over the full sequence" 'Window' 'P95_pct' 'Sequence segment' '95th percentile absolute error [%]' (Join-Path $FiguresRoot "review_${task}_windowed_p95.png")

    $cum=Import-Csv (Join-Path $ResultsRoot "long_horizon\${task}_cumulative_mae.csv")
    New-ModelLineChart $cum "$pretty cumulative error convergence" 'ProgressPercent' 'CumulativeMAE_pct' 'Processed sequence [%]' 'Cumulative MAE [%]' (Join-Path $FiguresRoot "review_${task}_cumulative_mae.png")

    $dev=Import-Csv (Join-Path $ResultsRoot "long_horizon\${task}_compression_deviation.csv")
    $devRows=@($dev | ForEach-Object { [pscustomobject]@{Model=$_.Variant;Window=$_.Window;MeanAbsDeviation_pp=$_.MeanAbsDeviation_pp} })
    New-ModelLineChart $devRows "$pretty compressed-to-base deviation" 'Window' 'MeanAbsDeviation_pp' 'Sequence segment' 'Mean absolute deviation from Base [percentage points]' (Join-Path $FiguresRoot "review_${task}_compression_deviation.png")

    $missing=Import-Csv (Join-Path $ResultsRoot "faults\${task}_missing_update_robustness.csv") | Where-Object { $_.Scenario -like 'RandomDrop*' -or $_.Scenario -eq 'Original' }
    $missingRows=@($missing | ForEach-Object { [pscustomobject]@{Model=$_.Model;DroppedPercent=100*(To-Double $_.DroppedFraction);DeltaMAE_pp=$_.DeltaMAE_pp} })
    New-ModelLineChart $missingRows "$pretty robustness to lost estimator updates (hold-last)" 'DroppedPercent' 'DeltaMAE_pp' 'Randomly lost updates [%]' 'Increase in MAE [percentage points]' (Join-Path $FiguresRoot "review_${task}_missing_updates.png")
}

$bitRows=@()
foreach($task in @('soc','soh')) {
    $bitRows += Import-Csv (Join-Path $ResultsRoot "faults\${task}_output_bitflip_robustness.csv") |
        Where-Object BitClass -eq 'AnyBit' | ForEach-Object {
            [pscustomobject]@{Category="$($task.ToUpperInvariant()) $($_.Model)";Series='Unmitigated';Value=100*(To-Double $_.CatastrophicRate_gt10pp)}
        }
}
New-GroupedBarChart $bitRows 'Random output-register bit flips' 'Category' 'Series' 'Value' 'Catastrophic outcomes (>10 percentage points) [%]' @{Unmitigated=$PrunedColor} (Join-Path $FiguresRoot 'review_output_bitflip_catastrophic_rate.png')

$mitRows=@()
foreach($task in @('soc','soh')) {
    $mitRows += Import-Csv (Join-Path $ResultsRoot "faults\${task}_output_bitflip_robustness.csv") |
        Where-Object BitClass -eq 'AnyBit' | ForEach-Object {
            [pscustomobject]@{Category="$($task.ToUpperInvariant()) $($_.Model)";Series='Corrupted';Value=$_.P95CorruptedError_pct}
            [pscustomobject]@{Category="$($task.ToUpperInvariant()) $($_.Model)";Series='Range check + hold-last';Value=$_.P95MitigatedError_pct}
        }
}
New-GroupedBarChart $mitRows 'Effect of a simple output range check' 'Category' 'Series' 'Value' '95th percentile absolute error [%]' @{Corrupted=$PrunedColor;'Range check + hold-last'=$BaseColor} (Join-Path $FiguresRoot 'review_output_bitflip_mitigation.png')

# SOH filter figures
$filterRows=Import-Csv (Join-Path $ResultsRoot 'filter\soh_filter_comparison.csv') | ForEach-Object {
    $short = if($_.Filter -like 'Raw*'){'Raw'}elseif($_.Filter -like 'Benchmark*'){'Code: alpha=0.02'}else{'Text: alpha=1e-6'}
    [pscustomobject]@{Filter=$short;Model=$_.Model;MAE_pct=$_.MAE_pct}
}
New-GroupedBarChart $filterRows 'SOH post-processing variants on the same raw predictions' 'Filter' 'Model' 'MAE_pct' 'MAE [% of full scale]' @{Base=$BaseColor;Pruned=$PrunedColor} (Join-Path $FiguresRoot 'review_soh_filter_mae_comparison.png')

$filterPenaltyRows=@()
foreach($filterName in @('Raw','Code: alpha=0.02','Text: alpha=1e-6')) {
    $baseMae=To-Double (($filterRows | Where-Object {$_.Filter -eq $filterName -and $_.Model -eq 'Base'} | Select-Object -First 1).MAE_pct)
    $prunedMae=To-Double (($filterRows | Where-Object {$_.Filter -eq $filterName -and $_.Model -eq 'Pruned'} | Select-Object -First 1).MAE_pct)
    $filterPenaltyRows += [pscustomobject]@{Filter=$filterName;Series='Pruned minus Base';DeltaMAE_pp=$prunedMae-$baseMae}
}
New-GroupedBarChart $filterPenaltyRows 'How SOH filtering changes the apparent pruning penalty' 'Filter' 'Series' 'DeltaMAE_pp' 'Pruned MAE - Base MAE [percentage points]' @{'Pruned minus Base'=$PurpleColor} (Join-Path $FiguresRoot 'review_soh_filter_pruning_interaction.png')

$traj=Import-Csv (Join-Path $ResultsRoot 'filter\soh_filter_trajectory_downsampled.csv')
$chart=New-Chart 'SOH filter definitions applied to the Base predictions' 'Sequence progress [%]' 'SOH [-]'
$chart.ChartAreas['Main'].AxisX.Minimum=0; $chart.ChartAreas['Main'].AxisX.Maximum=100; $chart.ChartAreas['Main'].AxisX.Interval=20; $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0'
Add-LineSeries $chart 'Reference' '#222222' $traj 'ProgressPercent' 'Reference' 'Main' 3
Add-LineSeries $chart 'Raw' $GrayColor $traj 'ProgressPercent' 'BaseRaw' 'Main' 2 'Dash'
Add-LineSeries $chart 'Code: alpha=0.02' $BaseColor $traj 'ProgressPercent' 'BaseBenchmark' 'Main' 3
Add-LineSeries $chart 'Text: alpha=1e-6' $PurpleColor $traj 'ProgressPercent' 'BaseManuscript' 'Main' 3
Save-Chart $chart (Join-Path $FiguresRoot 'review_soh_filter_trajectory.png')

$chart=New-Chart 'SOH filter definitions applied to the Pruned predictions' 'Sequence progress [%]' 'SOH [-]'
$chart.ChartAreas['Main'].AxisX.Minimum=0; $chart.ChartAreas['Main'].AxisX.Maximum=100; $chart.ChartAreas['Main'].AxisX.Interval=20; $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0'
Add-LineSeries $chart 'Reference' '#222222' $traj 'ProgressPercent' 'Reference' 'Main' 3
Add-LineSeries $chart 'Pruned raw' $GrayColor $traj 'ProgressPercent' 'PrunedRaw' 'Main' 2 'Dash'
Add-LineSeries $chart 'Pruned: alpha=0.02' $PrunedColor $traj 'ProgressPercent' 'PrunedBenchmark' 'Main' 3
Add-LineSeries $chart 'Pruned: alpha=1e-6' $PurpleColor $traj 'ProgressPercent' 'PrunedManuscript' 'Main' 3
Save-Chart $chart (Join-Path $FiguresRoot 'review_soh_filter_trajectory_pruned.png')

$savedFiltered=Import-Csv $savedFilteredTrajectory
$chart=New-Chart 'Stored SOH model outputs after benchmark filtering' 'Sequence progress [%]' 'SOH [-]'
$chart.ChartAreas['Main'].AxisX.Minimum=0; $chart.ChartAreas['Main'].AxisX.Maximum=100; $chart.ChartAreas['Main'].AxisX.Interval=20; $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0'
Add-LineSeries $chart 'Reference' '#222222' $savedFiltered 'ProgressPercent' 'Reference' 'Main' 3
Add-LineSeries $chart 'Base filtered' $BaseColor $savedFiltered 'ProgressPercent' 'BaseFiltered' 'Main' 3
Add-LineSeries $chart 'Pruned filtered' $PrunedColor $savedFiltered 'ProgressPercent' 'PrunedFiltered' 'Main' 3
Add-LineSeries $chart 'Quantized filtered' $QuantColor $savedFiltered 'ProgressPercent' 'QuantizedFiltered' 'Main' 3
Save-Chart $chart (Join-Path $FiguresRoot 'review_soh_saved_filtered_compressed_trajectories.png')

$filteredMaeRows=@()
$sohCumulative=Import-Csv (Join-Path $ResultsRoot 'long_horizon\soh_cumulative_mae.csv')
foreach($model in @('Base','Pruned','Quantized')) {
    $last=$sohCumulative | Where-Object Model -eq $model | Select-Object -Last 1
    $filteredMaeRows += [pscustomobject]@{Pipeline='Saved filtered output';Model=$model;MAE_pct=$last.CumulativeMAE_pct}
}
New-GroupedBarChart $filteredMaeRows 'Accuracy of the stored filtered SOH model outputs' 'Pipeline' 'Model' 'MAE_pct' 'MAE [% of full scale]' $ModelColors (Join-Path $FiguresRoot 'review_soh_saved_filtered_compression_accuracy.png')

$localFilterMetricsPath=Join-Path $ResultsRoot 'filter\soh_filter_compression_local_windows.csv'
if (Test-Path -LiteralPath $localFilterMetricsPath) {
    $localFilterRows=Import-Csv $localFilterMetricsPath | ForEach-Object {
        $short=if($_.Filter -like 'Raw*'){'Raw'}elseif($_.Filter -like 'Benchmark*'){'Code: alpha=0.02'}else{'Text: alpha=1e-6'}
        [pscustomobject]@{Filter=$short;Model=$_.Model;MAE_pct=$_.MAE_pct;LimiterActivation_pct=100*(To-Double $_.LimiterActivationFraction);PostprocessChange_pp=$_.MeanAbsPostprocessChange_pp}
    }
    New-GroupedBarChart $localFilterRows 'Local C re-execution: compression and SOH filtering' 'Filter' 'Model' 'MAE_pct' 'MAE [% of full scale]' $ModelColors (Join-Path $FiguresRoot 'review_soh_filter_compression_all_models_mae.png')

    $localPenaltyRows=@()
    foreach($filterName in @('Raw','Code: alpha=0.02','Text: alpha=1e-6')) {
        $baseMae=To-Double (($localFilterRows|Where-Object {$_.Filter -eq $filterName -and $_.Model -eq 'Base'}|Select-Object -First 1).MAE_pct)
        foreach($model in @('Pruned','Quantized')) {
            $mae=To-Double (($localFilterRows|Where-Object {$_.Filter -eq $filterName -and $_.Model -eq $model}|Select-Object -First 1).MAE_pct)
            $localPenaltyRows += [pscustomobject]@{Filter=$filterName;Model="$model minus Base";DeltaMAE_pp=$mae-$baseMae}
        }
    }
    New-GroupedBarChart $localPenaltyRows 'Local C re-execution: compression penalty after filtering' 'Filter' 'Model' 'DeltaMAE_pp' 'Compressed MAE - Base MAE [percentage points]' @{'Pruned minus Base'=$PrunedColor;'Quantized minus Base'=$QuantColor} (Join-Path $FiguresRoot 'review_soh_filter_compression_penalty_all_models.png')

    $activationRows=@($localFilterRows|Where-Object Filter -ne 'Raw'|ForEach-Object{[pscustomobject]@{Filter=$_.Filter;Model=$_.Model;Value=$_.LimiterActivation_pct}})
    New-GroupedBarChart $activationRows 'How often SOH rate limiting is active' 'Filter' 'Model' 'Value' 'Limiter activation [% of samples]' $ModelColors (Join-Path $FiguresRoot 'review_soh_filter_limiter_activation_all_models.png')
    $changeRows=@($localFilterRows|Where-Object Filter -ne 'Raw'|ForEach-Object{[pscustomobject]@{Filter=$_.Filter;Model=$_.Model;Value=$_.PostprocessChange_pp}})
    New-GroupedBarChart $changeRows 'Magnitude of SOH post-processing by model' 'Filter' 'Model' 'Value' 'Mean absolute raw-to-filter change [percentage points]' $ModelColors (Join-Path $FiguresRoot 'review_soh_filter_postprocess_change_all_models.png')

    $localTrajectory=Import-Csv (Join-Path $ResultsRoot 'filter\soh_filter_compression_local_trajectory.csv')
    foreach($definition in @('Raw','Benchmark','Manuscript')) {
        $label=if($definition -eq 'Raw'){'Unfiltered'}elseif($definition -eq 'Benchmark'){'Code filter: alpha=0.02'}else{'Text filter: alpha=1e-6'}
        $chart=New-Chart "Local C re-execution - $label" 'Sequence progress [%]' 'SOH [-]'
        $chart.ChartAreas['Main'].AxisX.Minimum=0; $chart.ChartAreas['Main'].AxisX.Maximum=100; $chart.ChartAreas['Main'].AxisX.Interval=20; $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0'
        Add-LineSeries $chart 'Reference' '#222222' $localTrajectory 'ProgressPercent' 'Reference' 'Main' 3
        foreach($model in @('Base','Pruned','Quantized')) { Add-LineSeries $chart $model $ModelColors[$model] $localTrajectory 'ProgressPercent' "${model}${definition}" 'Main' 3 }
        Save-Chart $chart (Join-Path $FiguresRoot "review_soh_filter_local_$($definition.ToLowerInvariant())_all_models.png")
    }

    $chart=New-Chart 'Quantized SOH: direct filter interaction' 'Sequence progress [%]' 'SOH [-]'
    $chart.ChartAreas['Main'].AxisX.Minimum=0; $chart.ChartAreas['Main'].AxisX.Maximum=100; $chart.ChartAreas['Main'].AxisX.Interval=20; $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0'
    Add-LineSeries $chart 'Reference' '#222222' $localTrajectory 'ProgressPercent' 'Reference' 'Main' 3
    Add-LineSeries $chart 'Quantized, unfiltered' $QuantColor $localTrajectory 'ProgressPercent' 'QuantizedRaw' 'Main' 2
    Add-LineSeries $chart 'Quantized, code filter' $PurpleColor $localTrajectory 'ProgressPercent' 'QuantizedBenchmark' 'Main' 3
    Save-Chart $chart (Join-Path $FiguresRoot 'review_soh_filter_quantized_raw_vs_filtered.png')
}

$step=Import-Csv (Join-Path $ResultsRoot 'filter\soh_filter_step_response.csv') | Where-Object { (To-Double $_.Samples) -gt 0 }
$chart=New-Chart 'Theoretical EMA step response at 1 Hz' 'Time [days, logarithmic]' 'Normalized response [-]'
$chart.ChartAreas['Main'].AxisX.IsLogarithmic=$true
$chart.ChartAreas['Main'].AxisX.LogarithmBase=10
Add-LineSeries $chart 'Code: alpha=0.02' $BaseColor $step 'TimeDays' 'Benchmark_alpha0.02' 'Main' 3
Add-LineSeries $chart 'Text: alpha=1e-6' $PurpleColor $step 'TimeDays' 'Manuscript_alpha1e-6' 'Main' 3
Save-Chart $chart (Join-Path $FiguresRoot 'review_soh_filter_step_response.png')

# Utility sensitivity
$utilityDir=Join-Path $ResultsRoot 'utility'
New-Item -ItemType Directory -Force -Path $utilityDir | Out-Null
$kpis=@{
    SOC=@{
        Base=@{MAE=2.6845146;Flash=105.32;RAM=4.93;Energy=700.83}
        Pruned=@{MAE=2.3379620;Flash=62.27;RAM=4.03;Energy=400.38}
        Quantized=@{MAE=2.7911590;Flash=52.48;RAM=3.96;Energy=3494.66}
    }
    SOH=@{
        Base=@{MAE=0.8523505;Flash=335.00;RAM=8.69;Energy=11366.93}
        Pruned=@{MAE=1.4573121;Flash=182.41;RAM=6.96;Energy=6362.23}
        Quantized=@{MAE=1.4103794;Flash=138.00;RAM=6.70;Energy=14604.27}
    }
}
$gridRows=@(); $sweepRows=@(); $summaryRows=@()
foreach($task in @('SOC','SOH')) {
    $counts=@{Base=0;Pruned=0;Quantized=0}
    for($a=0;$a -le 20;$a++) { for($f=0;$f -le (20-$a);$f++) { for($r=0;$r -le (20-$a-$f);$r++) {
        $e=20-$a-$f-$r; $weights=@(($a / 20.0), ($f / 20.0), ($r / 20.0), ($e / 20.0))
        $u=@{Base=1.0}
        foreach($variant in @('Pruned','Quantized')) {
            $u[$variant]=$weights[0]*$kpis[$task][$variant].MAE/$kpis[$task].Base.MAE +
                $weights[1]*$kpis[$task][$variant].Flash/$kpis[$task].Base.Flash +
                $weights[2]*$kpis[$task][$variant].RAM/$kpis[$task].Base.RAM +
                $weights[3]*$kpis[$task][$variant].Energy/$kpis[$task].Base.Energy
        }
        $winner=@('Base','Pruned','Quantized') | Sort-Object {$u[$_]} | Select-Object -First 1; $counts[$winner]++
        $gridRows += [pscustomobject]@{Task=$task;WeightAccuracy=$weights[0];WeightFlash=$weights[1];WeightRAM=$weights[2];WeightEnergy=$weights[3];UBase=1.0;UPruned=$u.Pruned;UQuantized=$u.Quantized;Winner=$winner}
    }}}
    $total=($counts.Values | Measure-Object -Sum).Sum
    foreach($variant in @('Base','Pruned','Quantized')) { $summaryRows += [pscustomobject]@{Task=$task;Variant=$variant;WinningCombinations=$counts[$variant];WinningShare=$counts[$variant]/$total} }

    foreach($metric in @('Accuracy','Flash','RAM','Energy')) {
        for($w=0.25;$w -le 0.85001;$w+=0.05) {
            $weights=@{Accuracy=(1-$w)/3;Flash=(1-$w)/3;RAM=(1-$w)/3;Energy=(1-$w)/3}; $weights[$metric]=$w
            $uP=$weights.Accuracy*$kpis[$task].Pruned.MAE/$kpis[$task].Base.MAE + $weights.Flash*$kpis[$task].Pruned.Flash/$kpis[$task].Base.Flash + $weights.RAM*$kpis[$task].Pruned.RAM/$kpis[$task].Base.RAM + $weights.Energy*$kpis[$task].Pruned.Energy/$kpis[$task].Base.Energy
            $uQ=$weights.Accuracy*$kpis[$task].Quantized.MAE/$kpis[$task].Base.MAE + $weights.Flash*$kpis[$task].Quantized.Flash/$kpis[$task].Base.Flash + $weights.RAM*$kpis[$task].Quantized.RAM/$kpis[$task].Base.RAM + $weights.Energy*$kpis[$task].Quantized.Energy/$kpis[$task].Base.Energy
            $winner=@([pscustomobject]@{N='Base';U=1.0},[pscustomobject]@{N='Pruned';U=$uP},[pscustomobject]@{N='Quantized';U=$uQ}) | Sort-Object U | Select-Object -First 1
            $sweepRows += [pscustomobject]@{Task=$task;FocalMetric=$metric;FocalWeight=$w;RemainingEach=(1-$w)/3;UBase=1.0;UPruned=$uP;UQuantized=$uQ;QuantMinusPruned=$uQ-$uP;Winner=$winner.N}
        }
    }
}
Write-CsvRows $gridRows (Join-Path $utilityDir 'utility_weight_grid.csv')
Write-CsvRows $sweepRows (Join-Path $utilityDir 'utility_priority_sweep.csv')
Write-CsvRows $summaryRows (Join-Path $utilityDir 'utility_ranking_summary.csv')

$winnerBars=@($summaryRows | ForEach-Object {[pscustomobject]@{Category=$_.Task;Series=$_.Variant;Value=100*$_.WinningShare}})
New-GroupedBarChart $winnerBars 'Best-ranked model across all 5%-spaced utility weights' 'Category' 'Series' 'Value' 'Share of weight combinations ranked best [%]' $ModelColors (Join-Path $FiguresRoot 'review_utility_ranking_robustness.png') $true
foreach($task in @('SOC','SOH')) {
    $rows=@($sweepRows | Where-Object Task -eq $task)
    $chart=New-Chart "$task`: Quantized versus Pruned under changing priorities" 'Weight of highlighted metric; remaining weight shared equally' 'U(Quantized) - U(Pruned)  [> 0: Pruned better; < 0: Quantized better]'
    $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0.00'
    if($task -eq 'SOC') { $chart.ChartAreas['Main'].AxisY.Minimum=-0.25 }
    $priorityColors=@{Accuracy=$PrunedColor;Flash=$QuantColor;RAM=$PurpleColor;Energy=$BaseColor}
    foreach($metric in @('Accuracy','Flash','RAM','Energy')) { Add-LineSeries $chart $metric $priorityColors[$metric] @($rows|Where-Object FocalMetric -eq $metric) 'FocalWeight' 'QuantMinusPruned' 'Main' 3 }
    $zero=New-Object System.Windows.Forms.DataVisualization.Charting.StripLine; $zero.IntervalOffset=0; $zero.StripWidth=0; $zero.BorderColor=Get-HexColor '#333333'; $zero.BorderWidth=2; $chart.ChartAreas['Main'].AxisY.StripLines.Add($zero)
    Save-Chart $chart (Join-Path $FiguresRoot "review_utility_priority_${task}.png")
}

# Weight, saliency, quantization, memory, and static operation analyses
$weightsDir=Join-Path $ResultsRoot 'weights'; $operationsDir=Join-Path $ResultsRoot 'operations'
New-Item -ItemType Directory -Force -Path $weightsDir,$operationsDir | Out-Null
$paths=@{
    SOCBase=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\base\soc_1.5.0.0_base\c_implementation\model_weights.h'
    SOCPruned=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\pruned\soc_1.5.0.0_pruned\prune_30pct_20250916_140404\c_implementation\model_weights.h'
    SOCQuant=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\quantized\soc_1.5.0.0_quantized\model_weights_lstm_int8_manual.h'
    SOHBase=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\base\soh_2.1.0.0_base\c_implementation\model_weights_soh.h'
    SOHPruned=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\pruned\soh_2.1.0.0\prune_30pct_20251122_010142\c_implementation\model_weights_soh.h'
    SOHQuant=Resolve-SourcePath 'DL_Models\LFP_LSTM_MLP\2_models\quantized\soh_2.1.0.0_quantized\c_implementation\model_weights_lstm_int8_manual_soh.h'
}
$weightStats=@(); $histRows=@(); $saliencyRows=@(); $quantStats=@(); $quantHist=@(); $memoryRows=@(); $operationRows=@()
$taskData=@{}
foreach($task in @('SOC','SOH')) {
    $basePath=$paths["${task}Base"]; $prunedPath=$paths["${task}Pruned"]; $quantPath=$paths["${task}Quant"]
    $H=Get-CDefine $basePath 'HIDDEN_SIZE'; $In=Get-CDefine $basePath 'INPUT_SIZE'; $M=Get-CDefine $basePath 'MLP_HIDDEN'
    $Hp=Get-CDefine $prunedPath 'HIDDEN_SIZE'
    $baseIH=Get-CArray $basePath 'LSTM_WEIGHT_IH'; $baseHH=Get-CArray $basePath 'LSTM_WEIGHT_HH'; $baseAll=Join-DoubleArrays $baseIH $baseHH
    $prunedIH=Get-CArray $prunedPath 'LSTM_WEIGHT_IH'; $prunedHH=Get-CArray $prunedPath 'LSTM_WEIGHT_HH'; $prunedAll=Join-DoubleArrays $prunedIH $prunedHH
    $taskData[$task]=@{H=$H;Hp=$Hp;In=$In;M=$M;BaseIH=$baseIH;BaseHH=$baseHH;BaseAll=$baseAll;PrunedAll=$prunedAll;QuantPath=$quantPath;BasePath=$basePath}
    foreach($variant in @('Base','Pruned')) {
        $values=if($variant -eq 'Base'){$baseAll}else{$prunedAll}; $s=Get-ArrayStats $values
        $weightStats += [pscustomobject]@{Task=$task;Variant=$variant;Count=$s.Count;Mean=$s.Mean;Std=$s.Std;MeanAbs=$s.MeanAbs;P95Abs=$s.P95Abs;P99Abs=$s.P99Abs;MaxAbs=$s.MaxAbs;FractionAbsBelow0p01=$s.FractionAbsBelow0p01}
    }
    $limit=[math]::Max(($weightStats|Where-Object Task -eq $task|Measure-Object P99Abs -Maximum).Maximum,1e-6)
    $histRows += Get-HistogramRows $baseAll 90 (-$limit) $limit $task 'Base'
    $histRows += Get-HistogramRows $prunedAll 90 (-$limit) $limit $task 'Pruned'
    $saliencyRows += Get-UnitSaliency $baseIH $baseHH $H $In $task

    $qIH=Get-CArray $quantPath 'LSTM_W_IH'; $qHH=Get-CArray $quantPath 'LSTM_W_HH'; $sIH=Get-CArray $quantPath 'LSTM_W_IH_SCALE'; $sHH=Get-CArray $quantPath 'LSTM_W_HH_SCALE'
    $qIHResult=Get-QuantizationStats $baseIH $qIH $sIH (4*$H) $In $task 'W_ih'
    $qHHResult=Get-QuantizationStats $baseHH $qHH $sHH (4*$H) $H $task 'W_hh'
    $quantStats += $qIHResult.Stats,$qHHResult.Stats
    $quantHist += Get-HistogramRows $qIHResult.Errors 80 0 ([math]::Max($qIHResult.P99,1e-10)) $task 'W_ih' 'Absolute reconstruction error'
    $quantHist += Get-HistogramRows $qHHResult.Errors 80 0 ([math]::Max($qHHResult.P99,1e-10)) $task 'W_hh' 'Absolute reconstruction error'

    $baseBias=(Get-CArray $basePath 'LSTM_BIAS').Length; $baseMlp=(Get-CArray $basePath 'MLP_FC1_WEIGHT').Length+(Get-CArray $basePath 'MLP_FC1_BIAS').Length+(Get-CArray $basePath 'MLP_FC2_WEIGHT').Length+(Get-CArray $basePath 'MLP_FC2_BIAS').Length
    $memoryRows += [pscustomobject]@{Task=$task;Variant='Base';Int8RecurrentWeightsBytes=0;FP32RecurrentWeightsBytes=4*($baseIH.Length+$baseHH.Length);FP32ScalesAndBiasBytes=4*$baseBias;FP32MLPBytes=4*$baseMlp;PersistentStateBytes=2*$H*4;TotalModelBytes=4*($baseIH.Length+$baseHH.Length+$baseBias+$baseMlp)}
    $memoryRows += [pscustomobject]@{Task=$task;Variant='Quantized';Int8RecurrentWeightsBytes=$qIH.Length+$qHH.Length;FP32RecurrentWeightsBytes=0;FP32ScalesAndBiasBytes=4*($sIH.Length+$sHH.Length+$baseBias);FP32MLPBytes=4*$baseMlp;PersistentStateBytes=2*$H*4;TotalModelBytes=$qIH.Length+$qHH.Length+4*($sIH.Length+$sHH.Length+$baseBias+$baseMlp)}

    $times=if($task -eq 'SOC'){@{Base=1.40;Pruned=.80;Quantized=6.99}}else{@{Base=22.73;Pruned=12.72;Quantized=29.21}}
    foreach($variant in @('Base','Pruned','Quantized')) {
        $hv=if($variant -eq 'Pruned'){$Hp}else{$H}; $recurrent=4*$hv*($In+$hv); $mlp=$hv*$M+$M; $extra=if($variant -eq 'Quantized'){$recurrent}else{0}
        $operationRows += [pscustomobject]@{Task=$task;Variant=$variant;HiddenSize=$hv;RecurrentMACs=$recurrent;MLPMACs=$mlp;TotalMACs=$recurrent+$mlp;AdditionalDequantScaleMultiplications=$extra;ObservedInferenceMs=$times[$variant];MicrosecondsPer1000MACs=1000*$times[$variant]/($recurrent+$mlp)}
    }
}
Write-CsvRows $weightStats (Join-Path $weightsDir 'lstm_weight_statistics.csv')
Write-CsvRows $histRows (Join-Path $weightsDir 'lstm_weight_histograms.csv')
Write-CsvRows $saliencyRows (Join-Path $weightsDir 'lstm_unit_saliency.csv')
Write-CsvRows $quantStats (Join-Path $weightsDir 'quantization_reconstruction_statistics.csv')
Write-CsvRows $quantHist (Join-Path $weightsDir 'quantization_error_histograms.csv')
Write-CsvRows $memoryRows (Join-Path $weightsDir 'quantization_memory_accounting.csv')
Write-CsvRows $operationRows (Join-Path $operationsDir 'static_kernel_operation_counts.csv')

foreach($task in @('SOC','SOH')) {
    $rows=@($histRows|Where-Object Task -eq $task)
    $chart=New-Chart "$task recurrent-weight distributions" 'Weight value' 'Density'
    $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0.00'
    foreach($variant in @('Base','Pruned')) { Add-LineSeries $chart $variant $ModelColors[$variant] @($rows|Where-Object Variant -eq $variant) 'BinCenter' 'Density' 'Main' 3 }
    Save-Chart $chart (Join-Path $FiguresRoot "review_${task}_weight_distribution.png")
    New-SaliencyChart @($saliencyRows|Where-Object Task -eq $task) $task (Join-Path $FiguresRoot "review_${task}_unit_saliency.png")
    $qrows=@($quantHist|Where-Object Task -eq $task)
    $chart=New-Chart "$task INT8 weight-reconstruction error" 'Absolute weight error' 'Density'
    $chart.ChartAreas['Main'].AxisX.LabelStyle.Format='0.0000'
    Add-LineSeries $chart 'Input-to-hidden' $PrunedColor @($qrows|Where-Object Variant -eq 'W_ih') 'BinCenter' 'Density' 'Main' 3
    Add-LineSeries $chart 'Hidden-to-hidden' $QuantColor @($qrows|Where-Object Variant -eq 'W_hh') 'BinCenter' 'Density' 'Main' 3
    Save-Chart $chart (Join-Path $FiguresRoot "review_${task}_quantization_error.png")
}

$memoryPlot=@($memoryRows|ForEach-Object{[pscustomobject]@{Category="$($_.Task) $($_.Variant)";Series='Model storage';Value=(To-Double $_.TotalModelBytes)/1024}})
New-GroupedBarChart $memoryPlot 'Exported model storage (activations remain FP32)' 'Category' 'Series' 'Value' 'Model storage [KiB]' @{'Model storage'=$QuantColor} (Join-Path $FiguresRoot 'review_quantization_model_storage.png')
$statePlot=@($memoryRows|ForEach-Object{[pscustomobject]@{Category="$($_.Task) $($_.Variant)";Series='Persistent h+c state';Value=(To-Double $_.PersistentStateBytes)}})
New-GroupedBarChart $statePlot 'Persistent recurrent state is not quantized' 'Category' 'Series' 'Value' 'Persistent hidden + cell state [bytes]' @{'Persistent h+c state'=$PurpleColor} (Join-Path $FiguresRoot 'review_quantization_fp32_state_memory.png')

$opPlot=@()
foreach($r in $operationRows){$opPlot += [pscustomobject]@{Category="$($r.Task) $($r.Variant)";Series='MACs';Value=$r.TotalMACs};$opPlot += [pscustomobject]@{Category="$($r.Task) $($r.Variant)";Series='Additional scale multiplications';Value=$r.AdditionalDequantScaleMultiplications}}
New-GroupedBarChart $opPlot 'Static operation-count explanation of runtime trends' 'Category' 'Series' 'Value' 'Operations per inference' @{MACs=$GrayColor;'Additional scale multiplications'=$QuantColor} (Join-Path $FiguresRoot 'review_static_operation_counts.png')

# Machine-readable provenance
$provenance=[ordered]@{
    GeneratedAt=(Get-Date).ToString('s')
    ReviewRoot=$ReviewRoot
    Sources=[ordered]@{SOCStreaming=$socNpz;SOHFilteredStreaming=$sohNpz;SOHRawBasePruned=$rawSohNpz}
    Constraints=@('No QAT','No new HPC training','No new STM32 execution','No modification of the original elsarticle directory')
    Notes=@('SOH benchmark_results.npz already contains first-point calibration and the alpha=0.02 symmetric-cap filter.','The manuscript text instead specifies alpha=1e-6 with a downward-only limiter; both definitions are analysed separately.','A separate local Windows C re-execution provides raw Base, Pruned, and Quantized trajectories under one common numerical environment.','Bit-flip analysis targets the floating-point estimator output register, not internal weights or recurrent states.')
}
$provenance | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $ResultsRoot 'analysis_provenance.json') -Encoding UTF8

Write-Host "Generated figures: $((Get-ChildItem -LiteralPath $FiguresRoot -Filter 'review_*.png').Count)"
Write-Host "Generated result files: $((Get-ChildItem -LiteralPath $ResultsRoot -Recurse -File).Count)"
