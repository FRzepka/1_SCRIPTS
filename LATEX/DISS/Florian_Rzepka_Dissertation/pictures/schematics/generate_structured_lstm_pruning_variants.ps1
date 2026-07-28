$ErrorActionPreference = 'Stop'

$culture = [System.Globalization.CultureInfo]::InvariantCulture
$red = '#d62728'
$redLight = '#f5c7c8'
$blue = '#2678b2'
$blueLight = '#d7eaf6'
$purple = '#7a4ea3'
$purpleLight = '#e3d8ee'
$gray = '#4a4a4a'
$grid = '#9b9b9b'
$paper = '#ffffff'

function New-SvgBuilder {
    param([int]$Width, [int]$Height)

    $script:svg = [System.Text.StringBuilder]::new()
    [void]$script:svg.AppendLine('<?xml version="1.0" encoding="UTF-8"?>')
    [void]$script:svg.AppendLine("<svg xmlns=`"http://www.w3.org/2000/svg`" width=`"$Width`" height=`"$Height`" viewBox=`"0 0 $Width $Height`">")
    [void]$script:svg.AppendLine(@"
  <defs>
    <style>
      .panel { font-family: Arial, Helvetica, sans-serif; fill: #222222; font-size: 42px; font-weight: 700; }
      .heading { font-family: Arial, Helvetica, sans-serif; fill: #222222; font-size: 31px; font-weight: 700; }
      .label { font-family: Arial, Helvetica, sans-serif; fill: #2d2d2d; font-size: 27px; }
      .small { font-family: Arial, Helvetica, sans-serif; fill: #4c4c4c; font-size: 23px; }
      .tiny { font-family: Arial, Helvetica, sans-serif; fill: #4c4c4c; font-size: 18px; }
      .math { font-family: "Times New Roman", Times, serif; fill: #222222; font-size: 31px; font-style: italic; }
      .mathsmall { font-family: "Times New Roman", Times, serif; fill: #222222; font-size: 23px; font-style: italic; }
      .value { font-family: Arial, Helvetica, sans-serif; fill: #222222; font-size: 17px; }
      .neutral { fill: #ffffff; stroke: $gray; stroke-width: 1.7; }
      .score { fill: $redLight; stroke: $red; stroke-width: 2.2; }
      .dependent { fill: $blueLight; stroke: $blue; stroke-width: 2.2; }
      .both { fill: $purpleLight; stroke: $purple; stroke-width: 2.2; }
      .gate { stroke: $gray; stroke-width: 4; }
      .divider { stroke: #d6d6d6; stroke-width: 3; }
      .arrow { stroke: $red; stroke-width: 6; fill: none; marker-end: url(#arrowhead); }
      .note { fill: #f7f7f7; stroke: $gray; stroke-width: 2.5; }
    </style>
    <marker id="arrowhead" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="10" markerHeight="10" orient="auto">
      <path d="M 0 0 L 10 5 L 0 10 z" fill="$red"/>
    </marker>
  </defs>
  <rect width="$Width" height="$Height" fill="$paper"/>
"@)
}

function Add-Line {
    param([string]$Line)
    [void]$script:svg.AppendLine($Line)
}

function Add-Text {
    param(
        [double]$X,
        [double]$Y,
        [string]$Text,
        [string]$Class = 'label',
        [string]$Anchor = 'start'
    )
    Add-Line "  <text x=`"$X`" y=`"$Y`" class=`"$Class`" text-anchor=`"$Anchor`">$Text</text>"
}

function Add-Cell {
    param(
        [double]$X,
        [double]$Y,
        [double]$Size,
        [string]$Class = 'neutral',
        [string]$Text = ''
    )
    Add-Line "  <rect x=`"$X`" y=`"$Y`" width=`"$Size`" height=`"$Size`" class=`"$Class`"/>"
    if ($Text -ne '') {
        $cx = $X + $Size / 2
        $cy = $Y + $Size / 2 + 6
        Add-Text $cx $cy $Text 'value' 'middle'
    }
}

function Add-GateLabels {
    param(
        [double]$X,
        [double]$Y,
        [int]$H,
        [double]$Cell
    )
    $names = @('i', 'f', 'g', 'o')
    for ($gate = 0; $gate -lt 4; $gate++) {
        $cy = $Y + ($gate * $H + $H / 2) * $Cell + 8
        Add-Text $X $cy $names[$gate] 'mathsmall' 'middle'
    }
}

function Add-GateSeparators {
    param(
        [double]$X,
        [double]$Y,
        [int]$Rows,
        [int]$Cols,
        [int]$H,
        [double]$Cell
    )
    for ($gate = 1; $gate -lt 4; $gate++) {
        $gy = $Y + $gate * $H * $Cell
        $x2 = $X + $Cols * $Cell
        Add-Line "  <line x1=`"$X`" y1=`"$gy`" x2=`"$x2`" y2=`"$gy`" class=`"gate`"/>"
    }
}

function Add-SchematicMatrix {
    param(
        [double]$X,
        [double]$Y,
        [int]$Rows,
        [int]$Cols,
        [double]$Cell,
        [int]$GateH = 0,
        [int]$RemovedChannel = -1,
        [bool]$HighlightRows = $false,
        [bool]$HighlightColumn = $false
    )

    for ($row = 0; $row -lt $Rows; $row++) {
        for ($col = 0; $col -lt $Cols; $col++) {
            $isRow = $HighlightRows -and $GateH -gt 0 -and (($row % $GateH) -eq $RemovedChannel)
            $isCol = $HighlightColumn -and ($col -eq $RemovedChannel)
            $class = if ($isRow -and $isCol) { 'both' } elseif ($isRow) { 'score' } elseif ($isCol) { 'dependent' } else { 'neutral' }
            Add-Cell ($X + $col * $Cell) ($Y + $row * $Cell) $Cell $class
        }
    }
    if ($GateH -gt 0) {
        Add-GateSeparators $X $Y $Rows $Cols $GateH $Cell
    }
}

function Add-NumericMatrix {
    param(
        [double]$X,
        [double]$Y,
        [object[]]$Values,
        [double]$Cell,
        [int]$GateH = 0,
        [int]$RemovedChannel = -1,
        [bool]$HighlightRows = $false,
        [bool]$HighlightColumn = $false
    )

    $rows = $Values.Count
    $cols = $Values[0].Count
    for ($row = 0; $row -lt $rows; $row++) {
        for ($col = 0; $col -lt $cols; $col++) {
            $isRow = $HighlightRows -and $GateH -gt 0 -and (($row % $GateH) -eq $RemovedChannel)
            $isCol = $HighlightColumn -and ($col -eq $RemovedChannel)
            $class = if ($isRow -and $isCol) { 'both' } elseif ($isRow) { 'score' } elseif ($isCol) { 'dependent' } else { 'neutral' }
            $formatted = [string]::Format($culture, '{0:0.00}', [double]$Values[$row][$col])
            Add-Cell ($X + $col * $Cell) ($Y + $row * $Cell) $Cell $class $formatted
        }
    }
    if ($GateH -gt 0) {
        Add-GateSeparators $X $Y $rows $cols $GateH $Cell
    }
}

function Complete-Svg {
    param([string]$Path)
    [void]$script:svg.AppendLine('</svg>')
    [System.IO.File]::WriteAllText($Path, $script:svg.ToString(), [System.Text.UTF8Encoding]::new($false))
}

function New-GridVariant {
    param([string]$Path)

    New-SvgBuilder 1800 900
    $cell = 24
    $h = 5
    $hp = 4
    $d = 4
    $m = 4
    $remove = 2

    Add-Line '  <line x1="900" y1="55" x2="900" y2="845" class="divider"/>'
    Add-Text 35 55 '(a)' 'panel'
    Add-Text 105 55 'Full tensors with one candidate channel' 'heading'
    Add-Text 935 55 '(b)' 'panel'
    Add-Text 1005 55 'Dense tensors after coupled removal' 'heading'

    # Full tensors
    $wihX = 105
    $whhX = 355
    $headX = 610
    $topY = 155

    Add-Text ($wihX + 2 * $cell) 120 'W<tspan baseline-shift="sub" font-size="19">ih</tspan>  (4H x D)' 'math' 'middle'
    Add-SchematicMatrix $wihX $topY (4 * $h) $d $cell $h $remove $true $false
    Add-GateLabels ($wihX - 32) $topY $h $cell

    Add-Text ($whhX + 2.5 * $cell) 120 'W<tspan baseline-shift="sub" font-size="19">hh</tspan>  (4H x H)' 'math' 'middle'
    Add-SchematicMatrix $whhX $topY (4 * $h) $h $cell $h $remove $true $true
    Add-GateLabels ($whhX - 32) $topY $h $cell

    Add-Text ($headX + 2.5 * $cell) 120 'W<tspan baseline-shift="sub" font-size="19">MLP,1</tspan>  (M x H)' 'math' 'middle'
    Add-SchematicMatrix $headX $topY $m $h $cell 0 $remove $false $true

    Add-Text ($headX + 2.5 * $cell) 315 'b<tspan baseline-shift="sub" font-size="19">ih</tspan>, b<tspan baseline-shift="sub" font-size="19">hh</tspan>' 'math' 'middle'
    for ($gate = 0; $gate -lt 4; $gate++) {
        for ($channel = 0; $channel -lt $h; $channel++) {
            $class = if ($channel -eq $remove) { 'dependent' } else { 'neutral' }
            Add-Cell ($headX + $channel * $cell) (340 + $gate * $cell) $cell $class
        }
    }

    Add-Text ($headX + 2.5 * $cell) 495 'h<tspan baseline-shift="sub" font-size="19">t</tspan>, c<tspan baseline-shift="sub" font-size="19">t</tspan>' 'math' 'middle'
    for ($row = 0; $row -lt 2; $row++) {
        for ($channel = 0; $channel -lt $h; $channel++) {
            $class = if ($channel -eq $remove) { 'dependent' } else { 'neutral' }
            Add-Cell ($headX + $channel * $cell) (520 + $row * $cell) $cell $class
        }
    }

    Add-Text 85 690 's<tspan baseline-shift="sub" font-size="18">h</tspan> uses the eight red gate rows only.' 'math'
    Add-Line '  <rect x="90" y="730" width="25" height="20" class="score"/>'
    Add-Text 130 749 'rows included in the L2 score' 'small'
    Add-Line '  <rect x="90" y="775" width="25" height="20" class="dependent"/>'
    Add-Text 130 794 'dependent dimensions removed with channel h' 'small'
    Add-Line '  <rect x="90" y="820" width="25" height="20" class="both"/>'
    Add-Text 130 839 'intersection of a scored row and dependent column' 'small'

    # Transition
    Add-Line '  <path d="M 820 430 L 975 430" class="arrow"/>'
    Add-Text 897 375 'remove h = 3' 'label' 'middle'
    Add-Text 897 405 'retain R = {1,2,4,5}' 'small' 'middle'

    # Compact tensors with the same square-cell style
    $wih2X = 1020
    $whh2X = 1270
    $head2X = 1515
    $top2Y = 190

    Add-Text ($wih2X + 2 * $cell) 145 'W&#x0303;<tspan baseline-shift="sub" font-size="19">ih</tspan>  (4H&apos; x D)' 'math' 'middle'
    Add-SchematicMatrix $wih2X $top2Y (4 * $hp) $d $cell $hp -1 $false $false
    Add-GateLabels ($wih2X - 32) $top2Y $hp $cell

    Add-Text ($whh2X + 2 * $cell) 145 'W&#x0303;<tspan baseline-shift="sub" font-size="19">hh</tspan>  (4H&apos; x H&apos;)' 'math' 'middle'
    Add-SchematicMatrix $whh2X $top2Y (4 * $hp) $hp $cell $hp -1 $false $false
    Add-GateLabels ($whh2X - 32) $top2Y $hp $cell

    Add-Text ($head2X + 2 * $cell) 145 'W&#x0303;<tspan baseline-shift="sub" font-size="19">MLP,1</tspan>  (M x H&apos;)' 'math' 'middle'
    Add-SchematicMatrix $head2X $top2Y $m $hp $cell

    Add-Text ($head2X + 2 * $cell) 350 'b&#x0303;<tspan baseline-shift="sub" font-size="19">ih</tspan>, b&#x0303;<tspan baseline-shift="sub" font-size="19">hh</tspan>' 'math' 'middle'
    Add-SchematicMatrix $head2X 375 4 $hp $cell

    Add-Text ($head2X + 2 * $cell) 535 'h&#x0303;<tspan baseline-shift="sub" font-size="19">t</tspan>, c&#x0303;<tspan baseline-shift="sub" font-size="19">t</tspan>' 'math' 'middle'
    Add-SchematicMatrix $head2X 560 2 $hp $cell

    Add-Text 1050 690 'Remaining coefficients form regular dense arrays.' 'label'
    Add-Text 1050 745 'The cell size is identical in both panels.' 'small'
    Add-Text 1050 790 'Schematic example: H = 5 becomes H&apos; = 4.' 'small'
    Add-Text 1050 835 'Evaluated models: SOC 64 to 45, SOH 128 to 90.' 'small'

    Complete-Svg $Path
}

function New-NumericVariant {
    param([string]$Path)

    $wih = @(
        @(0.42, -0.31), @(0.01, -0.02), @(-0.55, 0.28),
        @(0.38, 0.44), @(-0.02, 0.01), @(0.61, -0.35),
        @(-0.47, 0.25), @(0.00, 0.02), @(0.33, 0.58),
        @(0.29, -0.52), @(0.01, 0.00), @(-0.41, 0.36)
    )
    $whh = @(
        @(0.31, -0.22, 0.18), @(0.01, 0.00, -0.01), @(-0.43, 0.27, 0.35),
        @(0.22, 0.41, -0.30), @(0.00, -0.02, 0.01), @(0.55, -0.38, 0.24),
        @(-0.35, 0.29, 0.44), @(0.01, 0.01, 0.00), @(0.26, -0.47, 0.51),
        @(0.48, -0.33, 0.21), @(-0.01, 0.00, 0.02), @(-0.37, 0.42, -0.28)
    )
    $wmlp = @(
        @(0.72, 0.03, -0.41),
        @(-0.28, -0.02, 0.63)
    )

    $h = 3
    $remove = 1
    $keep = @(0, 2)
    $scores = @()
    for ($channel = 0; $channel -lt $h; $channel++) {
        $score = 0.0
        for ($gate = 0; $gate -lt 4; $gate++) {
            $row = $gate * $h + $channel
            $n1 = [Math]::Sqrt(($wih[$row] | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
            $n2 = [Math]::Sqrt(($whh[$row] | ForEach-Object { $_ * $_ } | Measure-Object -Sum).Sum)
            $score += $n1 + $n2
        }
        $scores += $score
    }

    $wihCompact = @()
    $whhCompact = @()
    for ($gate = 0; $gate -lt 4; $gate++) {
        foreach ($channel in $keep) {
            $row = $gate * $h + $channel
            $wihCompact += ,@($wih[$row])
            $newRow = @()
            foreach ($col in $keep) {
                $newRow += $whh[$row][$col]
            }
            $whhCompact += ,@($newRow)
        }
    }
    $wmlpCompact = @()
    foreach ($row in $wmlp) {
        $wmlpCompact += ,@($row[0], $row[2])
    }

    New-SvgBuilder 2000 1050
    Add-Line '  <line x1="1000" y1="75" x2="1000" y2="960" class="divider"/>'
    Add-Text 40 60 '(a)' 'panel'
    Add-Text 110 60 'Numerical channel score' 'heading'
    Add-Text 1040 60 '(b)' 'panel'
    Add-Text 1110 60 'Dense slicing with retained channels' 'heading'
    Add-Text 70 105 'Readable toy example: H = 3, D = 2, remove channel h2.' 'small'
    Add-Text 1070 105 'The same operation is used for SOC 64 to 45 and SOH 128 to 90 with D = 6.' 'small'

    $cell = 46
    $top = 190
    $wihX = 120
    $whhX = 355
    $sideX = 650

    Add-Text ($wihX + $cell) 155 'W<tspan baseline-shift="sub" font-size="19">ih</tspan>' 'math' 'middle'
    Add-NumericMatrix $wihX $top $wih $cell $h $remove $true $false
    Add-GateLabels ($wihX - 35) $top $h $cell

    Add-Text ($whhX + 1.5 * $cell) 155 'W<tspan baseline-shift="sub" font-size="19">hh</tspan>' 'math' 'middle'
    Add-NumericMatrix $whhX $top $whh $cell $h $remove $true $true
    Add-GateLabels ($whhX - 35) $top $h $cell

    Add-Text ($sideX + 1.5 * $cell) 155 'W<tspan baseline-shift="sub" font-size="19">MLP,1</tspan>' 'math' 'middle'
    Add-NumericMatrix $sideX $top $wmlp $cell 0 $remove $false $true

    Add-Text $sideX 360 'L2 channel scores' 'heading'
    for ($channel = 0; $channel -lt $h; $channel++) {
        $x = $sideX + $channel * 100
        $class = if ($channel -eq $remove) { 'score' } else { 'neutral' }
        Add-Line "  <rect x=`"$x`" y=`"400`" width=`"90`" height=`"75`" rx=`"4`" class=`"$class`"/>"
        Add-Text ($x + 45) 430 "h$($channel + 1)" 'small' 'middle'
        $scoreText = [string]::Format($culture, '{0:0.00}', $scores[$channel])
        Add-Text ($x + 45) 462 $scoreText 'label' 'middle'
    }
    Add-Text $sideX 530 's<tspan baseline-shift="sub" font-size="18">h</tspan> = sum of eight row norms' 'math'
    Add-Text $sideX 575 'The smallest score selects h2.' 'small'

    Add-Line '  <rect x="650" y="650" width="26" height="21" class="score"/>'
    Add-Text 690 670 'scored rows to remove' 'small'
    Add-Line '  <rect x="650" y="695" width="26" height="21" class="dependent"/>'
    Add-Text 690 715 'dependent columns to remove' 'small'
    Add-Line '  <rect x="650" y="740" width="26" height="21" class="both"/>'
    Add-Text 690 760 'row-column intersection' 'small'

    Add-Line '  <path d="M 900 500 L 1090 500" class="arrow"/>'
    Add-Text 995 445 'remove h2' 'label' 'middle'
    Add-Text 995 477 'retain R = {h1,h3}' 'small' 'middle'

    $wih2X = 1170
    $whh2X = 1430
    $head2X = 1690
    $top2 = 245

    Add-Text ($wih2X + $cell) 205 'W&#x0303;<tspan baseline-shift="sub" font-size="19">ih</tspan>' 'math' 'middle'
    Add-NumericMatrix $wih2X $top2 $wihCompact $cell 2 -1 $false $false
    Add-GateLabels ($wih2X - 35) $top2 2 $cell

    Add-Text ($whh2X + $cell) 205 'W&#x0303;<tspan baseline-shift="sub" font-size="19">hh</tspan>' 'math' 'middle'
    Add-NumericMatrix $whh2X $top2 $whhCompact $cell 2 -1 $false $false
    Add-GateLabels ($whh2X - 35) $top2 2 $cell

    Add-Text ($head2X + $cell) 205 'W&#x0303;<tspan baseline-shift="sub" font-size="19">MLP,1</tspan>' 'math' 'middle'
    Add-NumericMatrix $head2X $top2 $wmlpCompact $cell

    Add-Line '  <rect x="1620" y="430" width="320" height="150" rx="6" class="note"/>'
    Add-Text 1780 470 'Result' 'heading' 'middle'
    Add-Text 1780 515 'H = 3 becomes H&apos; = 2' 'small' 'middle'
    Add-Text 1780 555 'all remaining arrays are dense' 'small' 'middle'

    Add-Text 1120 720 'Rows h2 disappear once from every gate block.' 'label'
    Add-Text 1120 770 'Column h2 disappears from W<tspan baseline-shift="sub" font-size="16">hh</tspan> and W<tspan baseline-shift="sub" font-size="16">MLP,1</tspan>.' 'small'
    Add-Text 1120 820 'The remaining values keep their relative order.' 'small'
    Add-Text 1120 885 'Illustrative values use the exact scoring' 'small'
    Add-Text 1120 930 'and slicing rules of the implementation.' 'small'

    Complete-Svg $Path
}

$gridPath = Join-Path $PSScriptRoot 'structured_lstm_pruning.svg'
$numericPath = Join-Path $PSScriptRoot 'structured_lstm_pruning_numeric.svg'

New-GridVariant $gridPath
New-NumericVariant $numericPath

Write-Output "Generated:"
Write-Output "  $gridPath"
Write-Output "  $numericPath"
