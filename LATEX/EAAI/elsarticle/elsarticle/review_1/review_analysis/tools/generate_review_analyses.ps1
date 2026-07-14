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
    New-ModelLineChart $windows "$pretty error stability over the fußÎ9¶‰ËkºwµçI¥•Ì€‘¡…ÉĞ€EÕ…¹Ñ¥é•°Õ¹™¥±Ñ•É•œ€‘EÕ…¹Ñ½±½È€‘±½…±QÉ…©•Ñ½Éä€AÉ½É•ÍÍA•É•¹Ğœ€EÕ…¹Ñ¥é•‘I…Üœ€5…¥¸œ€È(€€€‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€EÕ…¹Ñ¥é•°½‘”™¥±Ñ•Èœ€‘AÕÉÁ±•½±½È€‘±½…±QÉ…©•Ñ½Éä€AÉ½É•ÍÍA•É•¹Ğœ€EÕ…¹Ñ¥é•‘	•¹¡µ…É¬œ€5…¥¸œ€Ì(€€€M…Ù”µ¡…ÉĞ€‘¡…ÉĞ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}Í½¡}™¥±Ñ•É}ÅÕ…¹Ñ¥é•‘}É…İ}ÙÍ}™¥±Ñ•É•¹Á¹œœ¤)ô((‘ÍÑ•Àõ%µÁ½ÉĞµÍØ€¡)½¥¸µA…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€™¥±Ñ•ÉqÍ½¡}™¥±Ñ•É}ÍÑ•Á}É•ÍÁ½¹Í”¹ÍØœ¤ğ]¡•É”µ=‰©•Ğì€¡Q¼µ½Õ‰±”€‘|¹M…µÁ±•Ì¤€µĞ€Àô(‘¡…ÉĞõ9•Üµ¡…ÉĞ€Q¡•½É•Ñ¥…°5ÍÑ•ÀÉ•ÍÁ½¹Í”…Ğ€Ä!èœ€Q¥µ”m‘…åÌ°±½…É¥Ñ¡µ¥tœ€9½Éµ…±¥é•É•ÍÁ½¹Í”lµtœ(‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Í`¹%Í1½…É¥Ñ¡µ¥Œô‘ÑÉÕ”(‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Í`¹1½…É¥Ñ¡µ	…Í”ôÄÀ)‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€½‘”è…±Á¡„ôÀ¸ÀÈœ€‘	…Í•½±½È€‘ÍÑ•À€Q¥µ•…åÌœ€	•¹¡µ…É­}…±Á¡„À¸ÀÈœ€5…¥¸œ€Ì)‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€Q•áĞè…±Á¡„ôÅ”´Øœ€‘AÕÉÁ±•½±½È€‘ÍÑ•À€Q¥µ•…åÌœ€5…¹ÕÍÉ¥ÁÑ}…±Á¡„Å”´Øœ€5…¥¸œ€Ì)M…Ù”µ¡…ÉĞ€‘¡…ÉĞ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}Í½¡}™¥±Ñ•É}ÍÑ•Á}É•ÍÁ½¹Í”¹Á¹œœ¤((ŒUÑ¥±¥ÑäÍ•¹Í¥Ñ¥Ù¥Ñä(‘ÕÑ¥±¥Ñå¥Èõ)½¥¸µA…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€ÕÑ¥±¥Ñäœ)9•Üµ%Ñ•´€µ%Ñ•µQåÁ”¥É•Ñ½Éä€µ½É”€µA…Ñ €‘ÕÑ¥±¥Ñå¥Èğ=ÕĞµ9Õ±°(‘­Á¥Ìõì(€€€M=õì(€€€€€€€	…Í”õí5ôÈ¸ØàĞÔÄĞØí±…Í ôÄÀÔ¸ÌÈíI4ôĞ¸äÌí¹•ÉäôÜÀÀ¸àÍô(€€€€€€€AÉÕ¹•õí5ôÈ¸ÌÌÜäØÈÀí±…Í ôØÈ¸ÈÜíI4ôĞ¸ÀÌí¹•ÉäôĞÀÀ¸Ìáô(€€€€€€€EÕ…¹Ñ¥é•õí5ôÈ¸ÜäÄÄÔäÀí±…Í ôÔÈ¸ĞàíI4ôÌ¸äØí¹•ÉäôÌĞäĞ¸ØÙô(€€€ô(€€€M= õì(€€€€€€€	…Í”õí5ôÀ¸àÔÈÌÔÀÔí±…Í ôÌÌÔ¸ÀÀíI4ôà¸Øäí¹•ÉäôÄÄÌØØ¸äÍô(€€€€€€€AÉÕ¹•õí5ôÄ¸ĞÔÜÌÄÈÄí±…Í ôÄàÈ¸ĞÄíI4ôØ¸äØí¹•ÉäôØÌØÈ¸ÈÍô(€€€€€€€EÕ…¹Ñ¥é•õí5ôÄ¸ĞÄÀÌÜäĞí±…Í ôÄÌà¸ÀÀíI4ôØ¸ÜÀí¹•ÉäôÄĞØÀĞ¸Èİô(€€€ô)ô(‘É¥‘I½İÌõ  ¤ì€‘Íİ••ÁI½İÌõ  ¤ì€‘ÍÕµµ…ÉåI½İÌõ  ¤)™½É•…  ‘Ñ…Í¬¥¸  M=œ°M= œ¤¤ì(€€€€‘½Õ¹ÑÌõí	…Í”ôÀíAÉÕ¹•ôÀíEÕ…¹Ñ¥é•ôÁô(€€€™½È ‘„ôÀì‘„€µ±”€ÈÀì‘„¬¬¤ì™½È ‘˜ôÀì‘˜€µ±”€ ÈÀ´‘„¤ì‘˜¬¬¤ì™½È ‘ÈôÀì‘È€µ±”€ ÈÀ´‘„´‘˜¤ì‘È¬¬¤ì(€€€€€€€€‘”ôÈÀ´‘„´‘˜´‘Èì€‘İ•¥¡ÑÌõ   ‘„€¼€ÈÀ¸À¤°€ ‘˜€¼€ÈÀ¸À¤°€ ‘È€¼€ÈÀ¸À¤°€ ‘”€¼€ÈÀ¸À¤¤(€€€€€€€€‘Ôõí	…Í”ôÄ¸Áô(€€€€€€€™½É•…  ‘Ù…É¥…¹Ğ¥¸  AÉÕ¹•œ°EÕ…¹Ñ¥é•œ¤¤ì(€€€€€€€€€€€€‘Õl‘Ù…É¥…¹Ñtô‘İ•¥¡ÑÍlÁt¨‘­Á¥Íl‘Ñ…Í­ul‘Ù…É¥…¹Ñt¹5¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹5€¬(€€€€€€€€€€€€€€€€‘İ•¥¡ÑÍlÅt¨‘­Á¥Íl‘Ñ…Í­ul‘Ù…É¥…¹Ñt¹±…Í ¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹±…Í €¬(€€€€€€€€€€€€€€€€‘İ•¥¡ÑÍlÉt¨‘­Á¥Íl‘Ñ…Í­ul‘Ù…É¥…¹Ñt¹I4¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹I4€¬(€€€€€€€€€€€€€€€€‘İ•¥¡ÑÍlÍt¨‘­Á¥Íl‘Ñ…Í­ul‘Ù…É¥…¹Ñt¹¹•Éä¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹¹•Éä(€€€€€€€ô(€€€€€€€€‘İ¥¹¹•Èõ  	…Í”œ°AÉÕ¹•œ°EÕ…¹Ñ¥é•œ¤ğM½ÉĞµ=‰©•Ğì‘Õl‘}uôğM•±•Ğµ=‰©•Ğ€µ¥ÉÍĞ€Äì€‘½Õ¹ÑÍl‘İ¥¹¹•Ét¬¬(€€€€€€€€‘É¥‘I½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬í]•¥¡ÑÕÉ…äô‘İ•¥¡ÑÍlÁtí]•¥¡Ñ±…Í ô‘İ•¥¡ÑÍlÅtí]•¥¡ÑI4ô‘İ•¥¡ÑÍlÉtí]•¥¡Ñ¹•Éäô‘İ•¥¡ÑÍlÍtíU	…Í”ôÄ¸ÀíUAÉÕ¹•ô‘Ô¹AÉÕ¹•íUEÕ…¹Ñ¥é•ô‘Ô¹EÕ…¹Ñ¥é•í]¥¹¹•Èô‘İ¥¹¹•Éô(€€€õõô(€€€€‘Ñ½Ñ…°ô ‘½Õ¹ÑÌ¹Y…±Õ•Ìğ5•…ÍÕÉ”µ=‰©•Ğ€µMÕ´¤¹MÕ´(€€€™½É•…  ‘Ù…É¥…¹Ğ¥¸  	…Í”œ°AÉÕ¹•œ°EÕ…¹Ñ¥é•œ¤¤ì€‘ÍÕµµ…ÉåI½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬íY…É¥…¹Ğô‘Ù…É¥…¹Ğí]¥¹¹¥¹½µ‰¥¹…Ñ¥½¹Ìô‘½Õ¹ÑÍl‘Ù…É¥…¹Ñtí]¥¹¹¥¹M¡…É”ô‘½Õ¹ÑÍl‘Ù…É¥…¹Ñt¼‘Ñ½Ñ…±ôô((€€€™½É•…  ‘µ•ÑÉ¥Œ¥¸  ÕÉ…äœ°±…Í œ°I4œ°¹•Éäœ¤¤ì(€€€€€€€™½È ‘ÜôÀ¸ÈÔì‘Ü€µ±”€À¸àÔÀÀÄì‘Ü¬ôÀ¸ÀÔ¤ì(€€€€€€€€€€€€‘İ•¥¡ÑÌõíÕÉ…äô Ä´‘Ü¤¼Ìí±…Í ô Ä´‘Ü¤¼ÌíI4ô Ä´‘Ü¤¼Ìí¹•Éäô Ä´‘Ü¤¼Íôì€‘İ•¥¡ÑÍl‘µ•ÑÉ¥tô‘Ü(€€€€€€€€€€€€‘Õ@ô‘İ•¥¡ÑÌ¹ÕÉ…ä¨‘­Á¥Íl‘Ñ…Í­t¹AÉÕ¹•¹5¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹5€¬€‘İ•¥¡ÑÌ¹±…Í ¨‘­Á¥Íl‘Ñ…Í­t¹AÉÕ¹•¹±…Í ¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹±…Í €¬€‘İ•¥¡ÑÌ¹I4¨‘­Á¥Íl‘Ñ…Í­t¹AÉÕ¹•¹I4¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹I4€¬€‘İ•¥¡ÑÌ¹¹•Éä¨‘­Á¥Íl‘Ñ…Í­t¹AÉÕ¹•¹¹•Éä¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹¹•Éä(€€€€€€€€€€€€‘ÕDô‘İ•¥¡ÑÌ¹ÕÉ…ä¨‘­Á¥Íl‘Ñ…Í­t¹EÕ…¹Ñ¥é•¹5¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹5€¬€‘İ•¥¡ÑÌ¹±…Í ¨‘­Á¥Íl‘Ñ…Í­t¹EÕ…¹Ñ¥é•¹±…Í ¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹±…Í €¬€‘İ•¥¡ÑÌ¹I4¨‘­Á¥Íl‘Ñ…Í­t¹EÕ…¹Ñ¥é•¹I4¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹I4€¬€‘İ•¥¡ÑÌ¹¹•Éä¨‘­Á¥Íl‘Ñ…Í­t¹EÕ…¹Ñ¥é•¹¹•Éä¼‘­Á¥Íl‘Ñ…Í­t¹	…Í”¹¹•Éä(€€€€€€€€€€€€‘İ¥¹¹•Èõ ¡mÁÍÕÍÑ½µ½‰©•Ñuí8ô	…Í”œíTôÄ¸Áô±mÁÍÕÍÑ½µ½‰©•Ñuí8ôAÉÕ¹•œíTô‘ÕAô±mÁÍÕÍÑ½µ½‰©•Ñuí8ôEÕ…¹Ñ¥é•œíTô‘ÕEô¤ğM½ÉĞµ=‰©•ĞTğM•±•Ğµ=‰©•Ğ€µ¥ÉÍĞ€Ä(€€€€€€€€€€€€‘Íİ••ÁI½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬í½…±5•ÑÉ¥Œô‘µ•ÑÉ¥Œí½…±]•¥¡Ğô‘ÜíI•µ…¥¹¥¹… ô Ä´‘Ü¤¼ÌíU	…Í”ôÄ¸ÀíUAÉÕ¹•ô‘Õ@íUEÕ…¹Ñ¥é•ô‘ÕDíEÕ…¹Ñ5¥¹ÕÍAÉÕ¹•ô‘ÕD´‘Õ@í]¥¹¹•Èô‘İ¥¹¹•È¹9ô(€€€€€€€ô(€€€ô)ô)]É¥Ñ”µÍÙI½İÌ€‘É¥‘I½İÌ€¡)½¥¸µA…Ñ €‘ÕÑ¥±¥Ñå¥È€ÕÑ¥±¥Ñå}İ•¥¡Ñ}É¥¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘Íİ••ÁI½İÌ€¡)½¥¸µA…Ñ €‘ÕÑ¥±¥Ñå¥È€ÕÑ¥±¥Ñå}ÁÉ¥½É¥Ñå}Íİ••À¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘ÍÕµµ…ÉåI½İÌ€¡)½¥¸µA…Ñ €‘ÕÑ¥±¥Ñå¥È€ÕÑ¥±¥Ñå}É…¹­¥¹}ÍÕµµ…Éä¹ÍØœ¤((‘İ¥¹¹•É	…ÉÌõ  ‘ÍÕµµ…ÉåI½İÌğ½É… µ=‰©•ĞímÁÍÕÍÑ½µ½‰©•Ñuí…Ñ•½Éäô‘|¹Q…Í¬íM•É¥•Ìô‘|¹Y…É¥…¹ĞíY…±Õ”ôÄÀÀ¨‘|¹]¥¹¹¥¹M¡…É•õô¤)9•ÜµÉ½ÕÁ•‘	…É¡…ÉĞ€‘İ¥¹¹•É	…ÉÌ€	•ÍĞµÉ…¹­•µ½‘•°…É½ÍÌ…±°€Ô”µÍÁ…•ÕÑ¥±¥Ñäİ•¥¡ÑÌœ€…Ñ•½Éäœ€M•É¥•Ìœ€Y…±Õ”œ€M¡…É”½˜İ•¥¡Ğ½µ‰¥¹…Ñ¥½¹ÌÉ…¹­•‰•ÍĞl•tœ€‘5½‘•±½±½ÉÌ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}ÕÑ¥±¥Ñå}É…¹­¥¹}É½‰ÕÍÑ¹•ÍÌ¹Á¹œœ¤€‘ÑÉÕ”)™½É•…  ‘Ñ…Í¬¥¸  M=œ°M= œ¤¤ì(€€€€‘É½İÌõ  ‘Íİ••ÁI½İÌğ]¡•É”µ=‰©•ĞQ…Í¬€µ•Ä€‘Ñ…Í¬¤(€€€€‘¡…ÉĞõ9•Üµ¡…ÉĞ€ˆ‘Ñ…Í­€èEÕ…¹Ñ¥é•Ù•ÉÍÕÌAÉÕ¹•Õ¹‘•È¡…¹¥¹œÁÉ¥½É¥Ñ¥•Ìˆ€]•¥¡Ğ½˜¡¥¡±¥¡Ñ•µ•ÑÉ¥ŒìÉ•µ…¥¹¥¹œİ•¥¡ĞÍ¡…É••ÅÕ…±±äœ€T¡EÕ…¹Ñ¥é•¤€´T¡AÉÕ¹•¤€lø€ÀèAÉÕ¹•‰•ÑÑ•Èì€ğ€ÀèEÕ…¹Ñ¥é•‰•ÑÑ•Étœ(€€€€‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Í`¹1…‰•±MÑå±”¹½Éµ…ĞôœÀ¸ÀÀœ(€€€¥˜ ‘Ñ…Í¬€µ•Ä€M=œ¤ì€‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Íd¹5¥¹¥µÕ´ô´À¸ÈÔô(€€€€‘ÁÉ¥½É¥Ñå½±½ÉÌõíÕÉ…äô‘AÉÕ¹•‘½±½Èí±…Í ô‘EÕ…¹Ñ½±½ÈíI4ô‘AÕÉÁ±•½±½Èí¹•Éäô‘	…Í•½±½Éô(€€€™½É•…  ‘µ•ÑÉ¥Œ¥¸  ÕÉ…äœ°±…Í œ°I4œ°¹•Éäœ¤¤ì‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€‘µ•ÑÉ¥Œ€‘ÁÉ¥½É¥Ñå½±½ÉÍl‘µ•ÑÉ¥t  ‘É½İÍñ]¡•É”µ=‰©•Ğ½…±5•ÑÉ¥Œ€µ•Ä€‘µ•ÑÉ¥Œ¤€½…±]•¥¡Ğœ€EÕ…¹Ñ5¥¹ÕÍAÉÕ¹•œ€5…¥¸œ€Ìô(€€€€‘é•É¼õ9•Üµ=‰©•ĞMåÍÑ•´¹]¥¹‘½İÌ¹½ÉµÌ¹…Ñ…Y¥ÍÕ…±¥é…Ñ¥½¸¹¡…ÉÑ¥¹œ¹MÑÉ¥Á1¥¹”ì€‘é•É¼¹%¹Ñ•ÉÙ…±=™™Í•ĞôÀì€‘é•É¼¹MÑÉ¥Á]¥‘Ñ ôÀì€‘é•É¼¹	½É‘•É½±½Èõ•Ğµ!•á½±½È€œŒÌÌÌÌÌÌœì€‘é•É¼¹	½É‘•É]¥‘Ñ ôÈì€‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Íd¹MÑÉ¥Á1¥¹•Ì¹‘ ‘é•É¼¤(€€€M…Ù”µ¡…ÉĞ€‘¡…ÉĞ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€‰É•Ù¥•İ}ÕÑ¥±¥Ñå}ÁÉ¥½É¥Ñå|‘íÑ…Í­ô¹Á¹œˆ¤)ô((Œ]•¥¡Ğ°Í…±¥•¹ä°ÅÕ…¹Ñ¥é…Ñ¥½¸°µ•µ½Éä°…¹ÍÑ…Ñ¥Œ½Á•É…Ñ¥½¸…¹…±åÍ•Ì(‘İ•¥¡ÑÍ¥Èõ)½¥¸µA…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€İ•¥¡ÑÌœì€‘½Á•É…Ñ¥½¹Í¥Èõ)½¥¸µA…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€½Á•É…Ñ¥½¹Ìœ)9•Üµ%Ñ•´€µ%Ñ•µQåÁ”¥É•Ñ½Éä€µ½É”€µA…Ñ €‘İ•¥¡ÑÍ¥È°‘½Á•É…Ñ¥½¹Í¥Èğ=ÕĞµ9Õ±°(‘Á…Ñ¡Ìõì(€€€M=	…Í”õI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±Íq‰…Í•qÍ½|Ä¸Ô¸À¸Á}‰…Í•q}¥µÁ±•µ•¹Ñ…Ñ¥½¹qµ½‘•±}İ•¥¡ÑÌ¹ œ(€€€M=AÉÕ¹•õI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±ÍqÁÉÕ¹•‘qÍ½|Ä¸Ô¸À¸Á}ÁÉÕ¹•‘qÁÉÕ¹•|ÌÁÁÑ|ÈÀÈÔÀäÄÙ|ÄĞÀĞÀÑq}¥µÁ±•µ•¹Ñ…Ñ¥½¹qµ½‘•±}İ•¥¡ÑÌ¹ œ(€€€M=EÕ…¹ĞõI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±ÍqÅÕ…¹Ñ¥é•‘qÍ½|Ä¸Ô¸À¸Á}ÅÕ…¹Ñ¥é•‘qµ½‘•±}İ•¥¡ÑÍ}±ÍÑµ}¥¹Ğá}µ…¹Õ…°¹ œ(€€€M=!	…Í”õI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±Íq‰…Í•qÍ½¡|È¸Ä¸À¸Á}‰…Í•q}¥µÁ±•µ•¹Ñ…Ñ¥½¹qµ½‘•±}İ•¥¡ÑÍ}Í½ ¹ œ(€€€M=!AÉÕ¹•õI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±ÍqÁÉÕ¹•‘qÍ½¡|È¸Ä¸À¸ÁqÁÉÕ¹•|ÌÁÁÑ|ÈÀÈÔÄÄÈÉ|ÀÄÀÄĞÉq}¥µÁ±•µ•¹Ñ…Ñ¥½¹qµ½‘•±}İ•¥¡ÑÍ}Í½ ¹ œ(€€€M=!EÕ…¹ĞõI•Í½±Ù”µM½ÕÉ•A…Ñ €1}5½‘•±Íq1A}1MQ5}51ApÉ}µ½‘•±ÍqÅÕ…¹Ñ¥é•‘qÍ½¡|È¸Ä¸À¸Á}ÅÕ…¹Ñ¥é•‘q}¥µÁ±•µ•¹Ñ…Ñ¥½¹qµ½‘•±}İ•¥¡ÑÍ}±ÍÑµ}¥¹Ğá}µ…¹Õ…±}Í½ ¹ œ)ô(‘İ•¥¡ÑMÑ…ÑÌõ  ¤ì€‘¡¥ÍÑI½İÌõ  ¤ì€‘Í…±¥•¹åI½İÌõ  ¤ì€‘ÅÕ…¹ÑMÑ…ÑÌõ  ¤ì€‘ÅÕ…¹Ñ!¥ÍĞõ  ¤ì€‘µ•µ½ÉåI½İÌõ  ¤ì€‘½Á•É…Ñ¥½¹I½İÌõ  ¤(‘Ñ…Í­…Ñ„õíô)™½É•…  ‘Ñ…Í¬¥¸  M=œ°M= œ¤¤ì(€€€€‘‰…Í•A…Ñ ô‘Á…Ñ¡Ílˆ‘íÑ…Í­õ	…Í”‰tì€‘ÁÉÕ¹•‘A…Ñ ô‘Á…Ñ¡Ílˆ‘íÑ…Í­õAÉÕ¹•‰tì€‘ÅÕ…¹ÑA…Ñ ô‘Á…Ñ¡Ílˆ‘íÑ…Í­õEÕ…¹Ğ‰t(€€€€‘ õ•Ğµ•™¥¹”€‘‰…Í•A…Ñ €!%9}M%iœì€‘%¸õ•Ğµ•™¥¹”€‘‰…Í•A…Ñ €%9AUQ}M%iœì€‘4õ•Ğµ•™¥¹”€‘‰…Í•A…Ñ €51A}!%8œ(€€€€‘!Àõ•Ğµ•™¥¹”€‘ÁÉÕ¹•‘A…Ñ €!%9}M%iœ(€€€€‘‰…Í•% õ•ĞµÉÉ…ä€‘‰…Í•A…Ñ €1MQ5}]%!Q}% œì€‘‰…Í•! õ•ĞµÉÉ…ä€‘‰…Í•A…Ñ €1MQ5}]%!Q}! œì€‘‰…Í•±°õ)½¥¸µ½Õ‰±•ÉÉ…åÌ€‘‰…Í•% €‘‰…Í•! (€€€€‘ÁÉÕ¹•‘% õ•ĞµÉÉ…ä€‘ÁÉÕ¹•‘A…Ñ €1MQ5}]%!Q}% œì€‘ÁÉÕ¹•‘! õ•ĞµÉÉ…ä€‘ÁÉÕ¹•‘A…Ñ €1MQ5}]%!Q}! œì€‘ÁÉÕ¹•‘±°õ)½¥¸µ½Õ‰±•ÉÉ…åÌ€‘ÁÉÕ¹•‘% €‘ÁÉÕ¹•‘! (€€€€‘Ñ…Í­…Ñ…l‘Ñ…Í­tõí ô‘ í!Àô‘!Àí%¸ô‘%¸í4ô‘4í	…Í•% ô‘‰…Í•% í	…Í•! ô‘‰…Í•! í	…Í•±°ô‘‰…Í•±°íAÉÕ¹•‘±°ô‘ÁÉÕ¹•‘±°íEÕ…¹ÑA…Ñ ô‘ÅÕ…¹ÑA…Ñ í	…Í•A…Ñ ô‘‰…Í•A…Ñ¡ô(€€€™½É•…  ‘Ù…É¥…¹Ğ¥¸  	…Í”œ°AÉÕ¹•œ¤¤ì(€€€€€€€€‘Ù…±Õ•Ìõ¥˜ ‘Ù…É¥…¹Ğ€µ•Ä€	…Í”œ¥ì‘‰…Í•±±õ•±Í•ì‘ÁÉÕ¹•‘±±ôì€‘Ìõ•ĞµÉÉ…åMÑ…ÑÌ€‘Ù…±Õ•Ì(€€€€€€€€‘İ•¥¡ÑMÑ…ÑÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬íY…É¥…¹Ğô‘Ù…É¥…¹Ğí½Õ¹Ğô‘Ì¹½Õ¹Ğí5•…¸ô‘Ì¹5•…¸íMÑô‘Ì¹MÑí5•…¹‰Ìô‘Ì¹5•…¹‰Ìí@äÕ‰Ìô‘Ì¹@äÕ‰Ìí@äå‰Ìô‘Ì¹@äå‰Ìí5…á‰Ìô‘Ì¹5…á‰ÌíÉ…Ñ¥½¹‰Í	•±½ÜÁÀÀÄô‘Ì¹É…Ñ¥½¹‰Í	•±½ÜÁÀÀÅô(€€€ô(€€€€‘±¥µ¥Ğõmµ…Ñ¡tèé5…à  ‘İ•¥¡ÑMÑ…ÑÍñ]¡•É”µ=‰©•ĞQ…Í¬€µ•Ä€‘Ñ…Í­ñ5•…ÍÕÉ”µ=‰©•Ğ@äå‰Ì€µ5…á¥µÕ´¤¹5…á¥µÕ´°Å”´Ø¤(€€€€‘¡¥ÍÑI½İÌ€¬ô•Ğµ!¥ÍÑ½É…µI½İÌ€‘‰…Í•±°€äÀ€ ´‘±¥µ¥Ğ¤€‘±¥µ¥Ğ€‘Ñ…Í¬€	…Í”œ(€€€€‘¡¥ÍÑI½İÌ€¬ô•Ğµ!¥ÍÑ½É…µI½İÌ€‘ÁÉÕ¹•‘±°€äÀ€ ´‘±¥µ¥Ğ¤€‘±¥µ¥Ğ€‘Ñ…Í¬€AÉÕ¹•œ(€€€€‘Í…±¥•¹åI½İÌ€¬ô•ĞµU¹¥ÑM…±¥•¹ä€‘‰…Í•% €‘‰…Í•! €‘ €‘%¸€‘Ñ…Í¬((€€€€‘Å% õ•ĞµÉÉ…ä€‘ÅÕ…¹ÑA…Ñ €1MQ5}]}% œì€‘Å! õ•ĞµÉÉ…ä€‘ÅÕ…¹ÑA…Ñ €1MQ5}]}! œì€‘Í% õ•ĞµÉÉ…ä€‘ÅÕ…¹ÑA…Ñ €1MQ5}]}%!}M1œì€‘Í! õ•ĞµÉÉ…ä€‘ÅÕ…¹ÑA…Ñ €1MQ5}]}!!}M1œ(€€€€‘Å%!I•ÍÕ±Ğõ•ĞµEÕ…¹Ñ¥é…Ñ¥½¹MÑ…ÑÌ€‘‰…Í•% €‘Å% €‘Í% € Ğ¨‘ ¤€‘%¸€‘Ñ…Í¬€]}¥ œ(€€€€‘Å!!I•ÍÕ±Ğõ•ĞµEÕ…¹Ñ¥é…Ñ¥½¹MÑ…ÑÌ€‘‰…Í•! €‘Å! €‘Í! € Ğ¨‘ ¤€‘ €‘Ñ…Í¬€]}¡ œ(€€€€‘ÅÕ…¹ÑMÑ…ÑÌ€¬ô€‘Å%!I•ÍÕ±Ğ¹MÑ…ÑÌ°‘Å!!I•ÍÕ±Ğ¹MÑ…ÑÌ(€€€€‘ÅÕ…¹Ñ!¥ÍĞ€¬ô•Ğµ!¥ÍÑ½É…µI½İÌ€‘Å%!I•ÍÕ±Ğ¹ÉÉ½ÉÌ€àÀ€À€¡mµ…Ñ¡tèé5…à ‘Å%!I•ÍÕ±Ğ¹@ää°Å”´ÄÀ¤¤€‘Ñ…Í¬€]}¥ œ€‰Í½±ÕÑ”É•½¹ÍÑÉÕÑ¥½¸•ÉÉ½Èœ(€€€€‘ÅÕ…¹Ñ!¥ÍĞ€¬ô•Ğµ!¥ÍÑ½É…µI½İÌ€‘Å!!I•ÍÕ±Ğ¹ÉÉ½ÉÌ€àÀ€À€¡mµ…Ñ¡tèé5…à ‘Å!!I•ÍÕ±Ğ¹@ää°Å”´ÄÀ¤¤€‘Ñ…Í¬€]}¡ œ€‰Í½±ÕÑ”É•½¹ÍÑÉÕÑ¥½¸•ÉÉ½Èœ((€€€€‘‰…Í•	¥…Ìô¡•ĞµÉÉ…ä€‘‰…Í•A…Ñ €1MQ5}	%Lœ¤¹1•¹Ñ ì€‘‰…Í•5±Àô¡•ĞµÉÉ…ä€‘‰…Í•A…Ñ €51A}Å}]%!Pœ¤¹1•¹Ñ ¬¡•ĞµÉÉ…ä€‘‰…Í•A…Ñ €51A}Å}	%Lœ¤¹1•¹Ñ ¬¡•ĞµÉÉ…ä€‘‰…Í•A…Ñ €51A}É}]%!Pœ¤¹1•¹Ñ ¬¡•ĞµÉÉ…ä€‘‰…Í•A…Ñ €51A}É}	%Lœ¤¹1•¹Ñ (€€€€‘µ•µ½ÉåI½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬íY…É¥…¹Ğô	…Í”œí%¹ĞáI•ÕÉÉ•¹Ñ]•¥¡ÑÍ	åÑ•ÌôÀí@ÌÉI•ÕÉÉ•¹Ñ]•¥¡ÑÍ	åÑ•ÌôĞ¨ ‘‰…Í•% ¹1•¹Ñ ¬‘‰…Í•! ¹1•¹Ñ ¤í@ÌÉM…±•Í¹‘	¥…Í	åÑ•ÌôĞ¨‘‰…Í•	¥…Ìí@ÌÉ51A	åÑ•ÌôĞ¨‘‰…Í•5±ÀíA•ÉÍ¥ÍÑ•¹ÑMÑ…Ñ•	åÑ•ÌôÈ¨‘ ¨ĞíQ½Ñ…±5½‘•±	åÑ•ÌôĞ¨ ‘‰…Í•% ¹1•¹Ñ ¬‘‰…Í•! ¹1•¹Ñ ¬‘‰…Í•	¥…Ì¬‘‰…Í•5±À¥ô(€€€€‘µ•µ½ÉåI½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬íY…É¥…¹ĞôEÕ…¹Ñ¥é•œí%¹ĞáI•ÕÉÉ•¹Ñ]•¥¡ÑÍ	åÑ•Ìô‘Å% ¹1•¹Ñ ¬‘Å! ¹1•¹Ñ í@ÌÉI•ÕÉÉ•¹Ñ]•¥¡ÑÍ	åÑ•ÌôÀí@ÌÉM…±•Í¹‘	¥…Í	åÑ•ÌôĞ¨ ‘Í% ¹1•¹Ñ ¬‘Í! ¹1•¹Ñ ¬‘‰…Í•	¥…Ì¤í@ÌÉ51A	åÑ•ÌôĞ¨‘‰…Í•5±ÀíA•ÉÍ¥ÍÑ•¹ÑMÑ…Ñ•	åÑ•ÌôÈ¨‘ ¨ĞíQ½Ñ…±5½‘•±	åÑ•Ìô‘Å% ¹1•¹Ñ ¬‘Å! ¹1•¹Ñ ¬Ğ¨ ‘Í% ¹1•¹Ñ ¬‘Í! ¹1•¹Ñ ¬‘‰…Í•	¥…Ì¬‘‰…Í•5±À¥ô((€€€€‘Ñ¥µ•Ìõ¥˜ ‘Ñ…Í¬€µ•Ä€M=œ¥íí	…Í”ôÄ¸ĞÀíAÉÕ¹•ô¸àÀíEÕ…¹Ñ¥é•ôØ¸äåõõ•±Í•íí	…Í”ôÈÈ¸ÜÌíAÉÕ¹•ôÄÈ¸ÜÈíEÕ…¹Ñ¥é•ôÈä¸ÈÅõô(€€€™½É•…  ‘Ù…É¥…¹Ğ¥¸  	…Í”œ°AÉÕ¹•œ°EÕ…¹Ñ¥é•œ¤¤ì(€€€€€€€€‘¡Øõ¥˜ ‘Ù…É¥…¹Ğ€µ•Ä€AÉÕ¹•œ¥ì‘!Áõ•±Í•ì‘!ôì€‘É•ÕÉÉ•¹ĞôĞ¨‘¡Ø¨ ‘%¸¬‘¡Ø¤ì€‘µ±Àô‘¡Ø¨‘4¬‘4ì€‘•áÑÉ„õ¥˜ ‘Ù…É¥…¹Ğ€µ•Ä€EÕ…¹Ñ¥é•œ¥ì‘É•ÕÉÉ•¹Ñõ•±Í•ìÁô(€€€€€€€€‘½Á•É…Ñ¥½¹I½İÌ€¬ômÁÍÕÍÑ½µ½‰©•ÑuíQ…Í¬ô‘Ñ…Í¬íY…É¥…¹Ğô‘Ù…É¥…¹Ğí!¥‘‘•¹M¥é”ô‘¡ØíI•ÕÉÉ•¹Ñ5Ìô‘É•ÕÉÉ•¹Ğí51A5Ìô‘µ±ÀíQ½Ñ…±5Ìô‘É•ÕÉÉ•¹Ğ¬‘µ±Àí‘‘¥Ñ¥½¹…±•ÅÕ…¹ÑM…±•5Õ±Ñ¥Á±¥…Ñ¥½¹Ìô‘•áÑÉ„í=‰Í•ÉÙ•‘%¹™•É•¹•5Ìô‘Ñ¥µ•Íl‘Ù…É¥…¹Ñtí5¥É½Í•½¹‘ÍA•ÈÄÀÀÁ5ÌôÄÀÀÀ¨‘Ñ¥µ•Íl‘Ù…É¥…¹Ñt¼ ‘É•ÕÉÉ•¹Ğ¬‘µ±À¥ô(€€€ô)ô)]É¥Ñ”µÍÙI½İÌ€‘İ•¥¡ÑMÑ…ÑÌ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€±ÍÑµ}İ•¥¡Ñ}ÍÑ…Ñ¥ÍÑ¥Ì¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘¡¥ÍÑI½İÌ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€±ÍÑµ}İ•¥¡Ñ}¡¥ÍÑ½É…µÌ¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘Í…±¥•¹åI½İÌ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€±ÍÑµ}Õ¹¥Ñ}Í…±¥•¹ä¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘ÅÕ…¹ÑMÑ…ÑÌ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€ÅÕ…¹Ñ¥é…Ñ¥½¹}É•½¹ÍÑÉÕÑ¥½¹}ÍÑ…Ñ¥ÍÑ¥Ì¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘ÅÕ…¹Ñ!¥ÍĞ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€ÅÕ…¹Ñ¥é…Ñ¥½¹}•ÉÉ½É}¡¥ÍÑ½É…µÌ¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘µ•µ½ÉåI½İÌ€¡)½¥¸µA…Ñ €‘İ•¥¡ÑÍ¥È€ÅÕ…¹Ñ¥é…Ñ¥½¹}µ•µ½Éå}…½Õ¹Ñ¥¹œ¹ÍØœ¤)]É¥Ñ”µÍÙI½İÌ€‘½Á•É…Ñ¥½¹I½İÌ€¡)½¥¸µA…Ñ €‘½Á•É…Ñ¥½¹Í¥È€ÍÑ…Ñ¥}­•É¹•±}½Á•É…Ñ¥½¹}½Õ¹ÑÌ¹ÍØœ¤()™½É•…  ‘Ñ…Í¬¥¸  M=œ°M= œ¤¤ì(€€€€‘É½İÌõ  ‘¡¥ÍÑI½İÍñ]¡•É”µ=‰©•ĞQ…Í¬€µ•Ä€‘Ñ…Í¬¤(€€€€‘¡…ÉĞõ9•Üµ¡…ÉĞ€ˆ‘Ñ…Í¬É•ÕÉÉ•¹Ğµİ•¥¡Ğ‘¥ÍÑÉ¥‰ÕÑ¥½¹Ìˆ€]•¥¡ĞÙ…±Õ”œ€•¹Í¥Ñäœ(€€€€‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Í`¹1…‰•±MÑå±”¹½Éµ…ĞôœÀ¸ÀÀœ(€€€™½É•…  ‘Ù…É¥…¹Ğ¥¸  	…Í”œ°AÉÕ¹•œ¤¤ì‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€‘Ù…É¥…¹Ğ€‘5½‘•±½±½ÉÍl‘Ù…É¥…¹Ñt  ‘É½İÍñ]¡•É”µ=‰©•ĞY…É¥…¹Ğ€µ•Ä€‘Ù…É¥…¹Ğ¤€	¥¹•¹Ñ•Èœ€•¹Í¥Ñäœ€5…¥¸œ€Ìô(€€€M…Ù”µ¡…ÉĞ€‘¡…ÉĞ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€‰É•Ù¥•İ|‘íÑ…Í­õ}İ•¥¡Ñ}‘¥ÍÑÉ¥‰ÕÑ¥½¸¹Á¹œˆ¤(€€€9•ÜµM…±¥•¹å¡…ÉĞ  ‘Í…±¥•¹åI½İÍñ]¡•É”µ=‰©•ĞQ…Í¬€µ•Ä€‘Ñ…Í¬¤€‘Ñ…Í¬€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€‰É•Ù¥•İ|‘íÑ…Í­õ}Õ¹¥Ñ}Í…±¥•¹ä¹Á¹œˆ¤(€€€€‘ÅÉ½İÌõ  ‘ÅÕ…¹Ñ!¥ÍÑñ]¡•É”µ=‰©•ĞQ…Í¬€µ•Ä€‘Ñ…Í¬¤(€€€€‘¡…ÉĞõ9•Üµ¡…ÉĞ€ˆ‘Ñ…Í¬%9Pàİ•¥¡ĞµÉ•½¹ÍÑÉÕÑ¥½¸•ÉÉ½Èˆ€‰Í½±ÕÑ”İ•¥¡Ğ•ÉÉ½Èœ€•¹Í¥Ñäœ(€€€€‘¡…ÉĞ¹¡…ÉÑÉ•…Íl5…¥¸t¹á¥Í`¹1…‰•±MÑå±”¹½Éµ…ĞôœÀ¸ÀÀÀÀœ(€€€‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€%¹ÁÕĞµÑ¼µ¡¥‘‘•¸œ€‘AÉÕ¹•‘½±½È  ‘ÅÉ½İÍñ]¡•É”µ=‰©•ĞY…É¥…¹Ğ€µ•Ä€]}¥ œ¤€	¥¹•¹Ñ•Èœ€•¹Í¥Ñäœ€5…¥¸œ€Ì(€€€‘µ1¥¹•M•É¥•Ì€‘¡…ÉĞ€!¥‘‘•¸µÑ¼µ¡¥‘‘•¸œ€‘EÕ…¹Ñ½±½È  ‘ÅÉ½İÍñ]¡•É”µ=‰©•ĞY…É¥…¹Ğ€µ•Ä€]}¡ œ¤€	¥¹•¹Ñ•Èœ€•¹Í¥Ñäœ€5…¥¸œ€Ì(€€€M…Ù”µ¡…ÉĞ€‘¡…ÉĞ€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€‰É•Ù¥•İ|‘íÑ…Í­õ}ÅÕ…¹Ñ¥é…Ñ¥½¹}•ÉÉ½È¹Á¹œˆ¤)ô((‘µ•µ½ÉåA±½Ğõ  ‘µ•µ½ÉåI½İÍñ½É… µ=‰©•ÑímÁÍÕÍÑ½µ½‰©•Ñuí…Ñ•½Éäôˆ ‘|¹Q…Í¬¤€ ‘|¹Y…É¥…¹Ğ¤ˆíM•É¥•Ìô5½‘•°ÍÑ½É…”œíY…±Õ”ô¡Q¼µ½Õ‰±”€‘|¹Q½Ñ…±5½‘•±	åÑ•Ì¤¼ÄÀÈÑõô¤)9•ÜµÉ½ÕÁ•‘	…É¡…ÉĞ€‘µ•µ½ÉåA±½Ğ€áÁ½ÉÑ•µ½‘•°ÍÑ½É…”€¡…Ñ¥Ù…Ñ¥½¹ÌÉ•µ…¥¸@ÌÈ¤œ€…Ñ•½Éäœ€M•É¥•Ìœ€Y…±Õ”œ€5½‘•°ÍÑ½É…”m-¥	tœì5½‘•°ÍÑ½É…”œô‘EÕ…¹Ñ½±½Éô€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}ÅÕ…¹Ñ¥é…Ñ¥½¹}µ½‘•±}ÍÑ½É…”¹Á¹œœ¤(‘ÍÑ…Ñ•A±½Ğõ  ‘µ•µ½ÉåI½İÍñ½É… µ=‰©•ÑímÁÍÕÍÑ½µ½‰©•Ñuí…Ñ•½Éäôˆ ‘|¹Q…Í¬¤€ ‘|¹Y…É¥…¹Ğ¤ˆíM•É¥•ÌôA•ÉÍ¥ÍÑ•¹Ğ ­ŒÍÑ…Ñ”œíY…±Õ”ô¡Q¼µ½Õ‰±”€‘|¹A•ÉÍ¥ÍÑ•¹ÑMÑ…Ñ•	åÑ•Ì¥õô¤)9•ÜµÉ½ÕÁ•‘	…É¡…ÉĞ€‘ÍÑ…Ñ•A±½Ğ€A•ÉÍ¥ÍÑ•¹ĞÉ•ÕÉÉ•¹ĞÍÑ…Ñ”¥Ì¹½ĞÅÕ…¹Ñ¥é•œ€…Ñ•½Éäœ€M•É¥•Ìœ€Y…±Õ”œ€A•ÉÍ¥ÍÑ•¹Ğ¡¥‘‘•¸€¬•±°ÍÑ…Ñ”m‰åÑ•ÍtœìA•ÉÍ¥ÍÑ•¹Ğ ­ŒÍÑ…Ñ”œô‘AÕÉÁ±•½±½Éô€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}ÅÕ…¹Ñ¥é…Ñ¥½¹}™ÀÌÉ}ÍÑ…Ñ•}µ•µ½Éä¹Á¹œœ¤((‘½ÁA±½Ğõ  ¤)™½É•…  ‘È¥¸€‘½Á•É…Ñ¥½¹I½İÌ¥ì‘½ÁA±½Ğ€¬ômÁÍÕÍÑ½µ½‰©•Ñuí…Ñ•½Éäôˆ ‘È¹Q…Í¬¤€ ‘È¹Y…É¥…¹Ğ¤ˆíM•É¥•Ìô5ÌœíY…±Õ”ô‘È¹Q½Ñ…±5Íôì‘½ÁA±½Ğ€¬ômÁÍÕÍÑ½µ½‰©•Ñuí…Ñ•½Éäôˆ ‘È¹Q…Í¬¤€ ‘È¹Y…É¥…¹Ğ¤ˆíM•É¥•Ìô‘‘¥Ñ¥½¹…°Í…±”µÕ±Ñ¥Á±¥…Ñ¥½¹ÌœíY…±Õ”ô‘È¹‘‘¥Ñ¥½¹…±•ÅÕ…¹ÑM…±•5Õ±Ñ¥Á±¥…Ñ¥½¹Íõô)9•ÜµÉ½ÕÁ•‘	…É¡…ÉĞ€‘½ÁA±½Ğ€MÑ…Ñ¥Œ½Á•É…Ñ¥½¸µ½Õ¹Ğ•áÁ±…¹…Ñ¥½¸½˜ÉÕ¹Ñ¥µ”ÑÉ•¹‘Ìœ€…Ñ•½Éäœ€M•É¥•Ìœ€Y…±Õ”œ€=Á•É…Ñ¥½¹ÌÁ•È¥¹™•É•¹”œí5Ìô‘É…å½±½Èì‘‘¥Ñ¥½¹…°Í…±”µÕ±Ñ¥Á±¥…Ñ¥½¹Ìœô‘EÕ…¹Ñ½±½Éô€¡)½¥¸µA…Ñ €‘¥ÕÉ•ÍI½½Ğ€É•Ù¥•İ}ÍÑ…Ñ¥}½Á•É…Ñ¥½¹}½Õ¹ÑÌ¹Á¹œœ¤((Œ5…¡¥¹”µÉ•…‘…‰±”ÁÉ½Ù•¹…¹”(‘ÁÉ½Ù•¹…¹”õm½É‘•É•‘uì(€€€•¹•É…Ñ•‘Ğô¡•Ğµ…Ñ”¤¹Q½MÑÉ¥¹œ Ìœ¤(€€€I•Ù¥•İI½½Ğô‘I•Ù¥•İI½½Ğ(€€€M½ÕÉ•Ìõm½É‘•É•‘uíM=MÑÉ•…µ¥¹œô‘Í½9ÁèíM=!¥±Ñ•É•‘MÑÉ•…µ¥¹œô‘Í½¡9ÁèíM=!I…İ	…Í•AÉÕ¹•ô‘É…İM½¡9Áéô(€€€½¹ÍÑÉ…¥¹ÑÌõ  9¼EPœ°9¼¹•Ü!AÑÉ…¥¹¥¹œœ°9¼¹•ÜMQ4ÌÈ•á•ÕÑ¥½¸œ°9¼µ½‘¥™¥…Ñ¥½¸½˜Ñ¡”½É¥¥¹…°•±Í…ÉÑ¥±”‘¥É•Ñ½Éäœ¤(€€€9½Ñ•Ìõ  M= ‰•¹¡µ…É­}É•ÍÕ±ÑÌ¹¹Áè…±É•…‘ä½¹Ñ…¥¹Ì™¥ÉÍĞµÁ½¥¹Ğ…±¥‰É…Ñ¥½¸…¹Ñ¡”…±Á¡„ôÀ¸ÀÈÍåµµ•ÑÉ¥Œµ…À™¥±Ñ•È¸œ°Q¡”µ…¹ÕÍÉ¥ÁĞÑ•áĞ¥¹ÍÑ•…ÍÁ•¥™¥•Ì…±Á¡„ôÅ”´Øİ¥Ñ „‘½İ¹İ…Éµ½¹±ä±¥µ¥Ñ•Èì‰½Ñ ‘•™¥¹¥Ñ¥½¹Ì…É”…¹…±åÍ•Í•Á…É…Ñ•±ä¸œ°Í•Á…É…Ñ”±½…°]¥¹‘½İÌÉ”µ•á•ÕÑ¥½¸ÁÉ½Ù¥‘•ÌÉ…Ü	…Í”°AÉÕ¹•°…¹EÕ…¹Ñ¥é•ÑÉ…©•Ñ½É¥•ÌÕ¹‘•È½¹”½µµ½¸¹Õµ•É¥…°•¹Ù¥É½¹µ•¹Ğ¸œ°	¥Ğµ™±¥À…¹…±åÍ¥ÌÑ…É•ÑÌÑ¡”™±½…Ñ¥¹œµÁ½¥¹Ğ•ÍÑ¥µ…Ñ½È½ÕÑÁÕĞÉ•¥ÍÑ•È°¹½Ğ¥¹Ñ•É¹…°İ•¥¡ÑÌ½ÈÉ•ÕÉÉ•¹ĞÍÑ…Ñ•Ì¸œ¤)ô(‘ÁÉ½Ù•¹…¹”ğ½¹Ù•ÉÑQ¼µ)Í½¸€µ•ÁÑ €ØğM•Ğµ½¹Ñ•¹Ğ€µ1¥Ñ•É…±A…Ñ €¡)½¥¸µA…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€…¹…±åÍ¥Í}ÁÉ½Ù•¹…¹”¹©Í½¸œ¤€µ¹½‘¥¹œUQà()]É¥Ñ”µ!½ÍĞ€‰•¹•É…Ñ•™¥ÕÉ•Ìè€ ¡•Ğµ¡¥±‘%Ñ•´€µ1¥Ñ•É…±A…Ñ €‘¥ÕÉ•ÍI½½Ğ€µ¥±Ñ•È€É•Ù¥•İ|¨¹Á¹œœ¤¹½Õ¹Ğ¤ˆ)]É¥Ñ”µ!½ÍĞ€‰•¹•É…Ñ•É•ÍÕ±Ğ™¥±•Ìè€ ¡•Ğµ¡¥±‘%Ñ•´€µ1¥Ñ•É…±A…Ñ €‘I•ÍÕ±ÑÍI½½Ğ€µI•ÕÉÍ”€µ¥±”¤¹½Õ¹Ğ¤ˆ(