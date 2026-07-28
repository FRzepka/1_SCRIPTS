param(
    [string]$RunDirectory = '',
    [string]$OutputPath = ''
)

$ErrorActionPreference = 'Stop'
$Inv = [System.Globalization.CultureInfo]::InvariantCulture
[System.Threading.Thread]::CurrentThread.CurrentCulture = $Inv
[System.Threading.Thread]::CurrentThread.CurrentUICulture = $Inv

$BenchmarkRoot = $PSScriptRoot
if (-not $RunDirectory) {
    $RunDirectory = Get-ChildItem -Directory (Join-Path $BenchmarkRoot 'results') |
        Where-Object { Test-Path (Join-Path $_.FullName 'summary.csv') } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1 -ExpandProperty FullName
}
if (-not $RunDirectory -or -not (Test-Path (Join-Path $RunDirectory 'summary.csv'))) {
    throw "No completed input-bitflip run found below $BenchmarkRoot\results"
}
$RunDirectory = (Resolve-Path $RunDirectory).Path

if (-not $OutputPath) {
    $Workspace = (Resolve-Path (Join-Path $BenchmarkRoot '../../../../..')).Path
    $OutputPath = Join-Path $Workspace (
        'LATEX/EAAI/elsarticle/elsarticle/review_1/figures/' +
        'Review_1_Additional/Figure_20_Limited_Fault_Sensitivity.png'
    )
}
$OutputParent = Split-Path -Parent $OutputPath
New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null

$Summary = @(Import-Csv (Join-Path $RunDirectory 'summary.csv'))
$Trace = @(Import-Csv (Join-Path $RunDirectory 'representative_trace.csv'))
if ($Summary.Count -ne 6) {
    throw "Expected six summary rows, found $($Summary.Count)"
}
if ($Trace.Count -eq 0) {
    throw 'Representative trace is empty.'
}

Add-Type -AssemblyName System.Windows.Forms.DataVisualization
Add-Type -AssemblyName System.Drawing

$BaseColor = '#2CA02C'
$PrunedColor = '#D62728'
$QuantColor = '#1F77B4'
$GridColor = '#D9DEE1'
$AxisColor = '#4E5960'
$TextColor = '#172026'
$FigureFontScale = 1.60
$ModelColors = @{Base=$BaseColor; Pruned=$PrunedColor; Quantized=$QuantColor}
$ModelFills = @{Base='#A6D7A6'; Pruned='#EEA4A5'; Quantized='#A1C6E0'}

function C([string]$Hex) {
    return [System.Drawing.ColorTranslator]::FromHtml($Hex)
}

function D([object]$Value) {
    return [double]::Parse([string]$Value, [System.Globalization.NumberStyles]::Float, $Inv)
}

function Font([float]$Size, [bool]$Bold=$false) {
    $Style = if ($Bold) { [System.Drawing.FontStyle]::Bold } else { [System.Drawing.FontStyle]::Regular }
    return New-Object System.Drawing.Font('Arial', ($Size * $FigureFontScale), $Style)
}

function Add-Area(
    $Chart,
    [string]$Name,
    [float]$X,
    [float]$Y,
    [float]$Width,
    [float]$Height,
    [string]$XTitle,
    [string]$YTitle
) {
    $Area = New-Object System.Windows.Forms.DataVisualization.Charting.ChartArea $Name
    $Area.BackColor = [System.Drawing.Color]::White
    $Area.Position = New-Object System.Windows.Forms.DataVisualization.Charting.ElementPosition(
        $X, $Y, $Width, $Height
    )
    $Area.AxisX.Title = $XTitle
    $Area.AxisY.Title = $YTitle
    $Area.AxisX.TitleFont = Font 17
    $Area.AxisY.TitleFont = Font 17
    $Area.AxisX.LabelStyle.Font = Font 15
    $Area.AxisY.LabelStyle.Font = Font 15
    $Area.AxisX.MajorGrid.LineColor = C '#E8EBED'
    $Area.AxisY.MajorGrid.LineColor = C $GridColor
    $Area.AxisX.LineColor = C $AxisColor
    $Area.AxisY.LineColor = C $AxisColor
    $Area.AxisX.MajorTickMark.LineColor = C $AxisColor
    $Area.AxisY.MajorTickMark.LineColor = C $AxisColor
    $Area.AxisX.IsMarginVisible = $false
    $Chart.ChartAreas.Add($Area)
    return $Area
}

function Add-PanelTitle($Chart, [string]$AreaName, [string]$Text) {
    $Title = New-Object System.Windows.Forms.DataVisualization.Charting.Title
    $Title.Text = $Text
    $Title.DockedToChartArea = $AreaName
    $Title.IsDockedInsideChartArea = $false
    $Title.Alignment = 'MiddleLeft'
    $Title.Font = Font 19 $true
    $Title.ForeColor = C $TextColor
    $Chart.Titles.Add($Title)
}

function Add-LineLegend($Chart, [string]$Name, [string]$AreaName) {
    $Legend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend $Name
    $Legend.DockedToChartArea = $AreaName
    $Legend.IsDockedInsideChartArea = $true
    $Legend.Docking = 'Top'
    $Legend.Alignment = 'Far'
    $Legend.LegendStyle = 'Row'
    $Legend.Font = Font 14
    $Legend.BackColor = [System.Drawing.Color]::FromArgb(232,255,255,255)
    foreach ($Model in @('Base','Pruned','Quantized')) {
        $Item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
        $Item.Name = $Model
        $Item.Color = C $ModelColors[$Model]
        $Item.BorderColor = C $ModelColors[$Model]
        $Item.BorderWidth = 4
        $Item.ImageStyle = 'Line'
        $Legend.CustomItems.Add($Item)
    }
    $Chart.Legends.Add($Legend)
}

function Add-TraceLines($Chart, [string]$AreaName, [string]$Task) {
    foreach ($Model in @('Base','Pruned','Quantized')) {
        $Rows = @($Trace |
            Where-Object { $_.task -eq $Task -and $_.model -eq $Model } |
            Sort-Object { [int]$_.seconds_after_fault })
        $Series = New-Object System.Windows.Forms.DataVisualization.Charting.Series "$AreaName-$Model"
        $Series.ChartArea = $AreaName
        $Series.ChartType = 'Line'
        $Series.Color = C $ModelColors[$Model]
        $Series.BorderWidth = 4
        $Series.IsVisibleInLegend = $false
        foreach ($Row in $Rows) {
            [void]$Series.Points.AddXY((D $Row.seconds_after_fault), (D $Row.abs_deviation_pp))
        }
        $Chart.Series.Add($Series)
    }
}

function Add-CustomXLabels($Area, [object[]]$Rows) {
    $Area.AxisX.CustomLabels.Clear()
    for ($Index = 0; $Index -lt $Rows.Count; $Index++) {
        $X = $Index + 1
        $ModelLabel = if ($Rows[$Index].model -eq 'Quantized') { 'Quant.' } else { $Rows[$Index].model }
        $Text = "$($Rows[$Index].task)`n$ModelLabel"
        [void]$Area.AxisX.CustomLabels.Add($X - 0.48, $X + 0.48, $Text)
    }
}

$Chart = New-Object System.Windows.Forms.DataVisualization.Charting.Chart
$Chart.Width = 2400
$Chart.Height = 1700
$Chart.BackColor = [System.Drawing.Color]::White
$Chart.AntiAliasing = 'All'
$Chart.TextAntiAliasingQuality = 'High'

$SocArea = Add-Area $Chart 'SOCTrace' 5 5 42 39 'Time after fault [s]' 'Absolute deviation [pp]'
$SohArea = Add-Area $Chart 'SOHTrace' 54 5 42 39 'Time after fault [s]' 'Absolute deviation [pp]'
$PeakArea = Add-Area $Chart 'Peak' 5 55 42 37 '' 'Peak deviation [pp]'
$RecoveryArea = Add-Area $Chart 'Recovery' 54 55 42 37 '' 'Recovery within 60 s [%]'

Add-PanelTitle $Chart 'SOCTrace' '(a)'
Add-PanelTitle $Chart 'SOHTrace' '(b)'
Add-PanelTitle $Chart 'Peak' '(c)'
Add-PanelTitle $Chart 'Recovery' '(d)'

Add-TraceLines $Chart 'SOCTrace' 'SOC'
Add-TraceLines $Chart 'SOHTrace' 'SOH'
Add-LineLegend $Chart 'SOCTraceLegend' 'SOCTrace'
Add-LineLegend $Chart 'SOHTraceLegend' 'SOHTrace'

foreach ($Area in @($SocArea,$SohArea)) {
    $Area.AxisX.Minimum = 0
    $Area.AxisX.Maximum = 60
    $Area.AxisX.Interval = 10
    $Area.AxisY.Minimum = 0
    $Area.AxisY.IsStartedFromZero = $true
}

$OrderedSummary = @()
foreach ($Task in @('SOC','SOH')) {
    foreach ($Model in @('Base','Pruned','Quantized')) {
        $OrderedSummary += $Summary | Where-Object { $_.task -eq $Task -and $_.model -eq $Model }
    }
}

$MedianBars = New-Object System.Windows.Forms.DataVisualization.Charting.Series 'Median peak'
$MedianBars.ChartArea = 'Peak'
$MedianBars.ChartType = 'Column'
$MedianBars.IsVisibleInLegend = $false
$MedianBars['PointWidth'] = '0.62'
$P95Points = New-Object System.Windows.Forms.DataVisualization.Charting.Series 'P95 peak'
$P95Points.ChartArea = 'Peak'
$P95Points.ChartType = 'Point'
$P95Points.IsVisibleInLegend = $false
$P95Points.MarkerStyle = 'Diamond'
$P95Points.MarkerSize = 15
$P95Points.MarkerBorderWidth = 2
$MaxP95 = 0.0

for ($Index = 0; $Index -lt $OrderedSummary.Count; $Index++) {
    $Row = $OrderedSummary[$Index]
    $X = $Index + 1
    $Model = [string]$Row.model
    $Median = D $Row.median_peak_deviation_pp
    $P95 = D $Row.p95_peak_deviation_pp
    $MaxP95 = [math]::Max($MaxP95, $P95)
    $BarIndex = $MedianBars.Points.AddXY($X, $Median)
    $Bar = $MedianBars.Points[$BarIndex]
    $Bar.Color = C $ModelFills[$Model]
    $Bar.BorderColor = C $ModelColors[$Model]
    $Bar.BorderWidth = 3
    $PointIndex = $P95Points.Points.AddXY($X, $P95)
    $Point = $P95Points.Points[$PointIndex]
    $Point.MarkerColor = C $ModelColors[$Model]
    $Point.MarkerBorderColor = C $ModelColors[$Model]
    $Point.Label = $P95.ToString('0.0', $Inv)
    $Point.LabelForeColor = C $TextColor
    $Point.Font = Font 13 $true
}
$Chart.Series.Add($MedianBars)
$Chart.Series.Add($P95Points)
Add-CustomXLabels $PeakArea $OrderedSummary
$PeakArea.AxisX.Minimum = 0.35
$PeakArea.AxisX.Maximum = 6.65
$PeakArea.AxisX.MajorGrid.Enabled = $false
$PeakArea.AxisX.LabelStyle.Font = Font 14
$PeakArea.AxisY.Minimum = 0
$PeakArea.AxisY.Maximum = [math]::Ceiling(($MaxP95 * 1.20) / 5.0) * 5.0
$PeakArea.AxisY.Interval = 5

$PeakLegend = New-Object System.Windows.Forms.DataVisualization.Charting.Legend 'PeakLegend'
$PeakLegend.DockedToChartArea = 'Peak'
$PeakLegend.IsDockedInsideChartArea = $true
$PeakLegend.Docking = 'Top'
$PeakLegend.Alignment = 'Near'
$PeakLegend.LegendStyle = 'Row'
$PeakLegend.Font = Font 13
$PeakLegend.BackColor = [System.Drawing.Color]::FromArgb(232,255,255,255)
$MedianItem = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
$MedianItem.Name = 'Median'
$MedianItem.Color = C '#DCE3E7'
$MedianItem.BorderColor = C $AxisColor
$MedianItem.BorderWidth = 2
$MedianItem.ImageStyle = 'Rectangle'
$PeakLegend.CustomItems.Add($MedianItem)
$P95Item = New-Object System.Windows.Forms.DataVisualization.Charting.LegendItem
$P95Item.Name = 'P95'
$P95Item.Color = C $AxisColor
$P95Item.MarkerColor = C $AxisColor
$P95Item.MarkerStyle = 'Diamond'
$P95Item.MarkerSize = 11
$P95Item.ImageStyle = 'Marker'
$PeakLegend.CustomItems.Add($P95Item)
$Chart.Legends.Add($PeakLegend)

$RecoveryBars = New-Object System.Windows.Forms.DataVisualization.Charting.Series 'Recovered by 60 s'
$RecoveryBars.ChartArea = 'Recovery'
$RecoveryBars.ChartType = 'Column'
$RecoveryBars.IsVisibleInLegend = $false
$RecoveryBars['PointWidth'] = '0.62'
for ($Index = 0; $Index -lt $OrderedSummary.Count; $Index++) {
    $Row = $OrderedSummary[$Index]
    $X = $Index + 1
    $Model = [string]$Row.model
    $Recovered = 100.0 - (D $Row.not_recovered_by_horizon_pct)
    $PointIndex = $RecoveryBars.Points.AddXY($X, $Recovered)
    $Point = $RecoveryBars.Points[$PointIndex]
    $Point.Color = C $ModelFills[$Model]
    $Point.BorderColor = C $ModelColors[$Model]
    $Point.BorderWidth = 3
    $Point.Label = $Recovered.ToString('0.0', $Inv) + '%'
    $Point.LabelForeColor = C $TextColor
    $Point.Font = Font 14 $true
}
$Chart.Series.Add($RecoveryBars)
Add-CustomXLabels $RecoveryArea $OrderedSummary
$RecoveryArea.AxisX.Minimum = 0.35
$RecoveryArea.AxisX.Maximum = 6.65
$RecoveryArea.AxisX.MajorGrid.Enabled = $false
$RecoveryArea.AxisX.LabelStyle.Font = Font 14
$RecoveryArea.AxisY.Minimum = 0
$RecoveryArea.AxisY.Maximum = 110
$RecoveryArea.AxisY.Interval = 20

$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
$ResultCopy = Join-Path $RunDirectory 'review_input_bitflip_recovery.png'
$Chart.SaveImage($OutputPath, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$Chart.SaveImage($ResultCopy, [System.Windows.Forms.DataVisualization.Charting.ChartImageFormat]::Png)
$Chart.Dispose()

Write-Host "Input-bitflip review figure written to:"
Write-Host "  $OutputPath"
Write-Host "  $ResultCopy"
