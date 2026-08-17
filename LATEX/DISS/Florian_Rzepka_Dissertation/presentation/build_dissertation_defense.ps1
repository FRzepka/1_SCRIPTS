param(
    [string]$TemplatePath = (Join-Path $PSScriptRoot 'Diss_FR.pptx'),
    [string]$OutputPath = (Join-Path $PSScriptRoot 'Diss_FR_Dissertation_30min_v3.pptx')
)

$ErrorActionPreference = 'Stop'

$root = Split-Path $PSScriptRoot -Parent
$pictureRoot = Join-Path $root 'pictures'
$paletteRoot = Join-Path $pictureRoot 'eaai_palette'
$assetRoot = Join-Path $PSScriptRoot 'assets'

$red = 1969604       # C40D1E, template red
$dark = 4408131      # 434343
$midGray = 7434614   # 767171
$lightGray = 15132390
$white = 16777215
$blue = 11826975

function Get-ImageSize {
    param([string]$Path)
    Add-Type -AssemblyName System.Drawing
    $image = [System.Drawing.Image]::FromFile($Path)
    try {
        return @($image.Width, $image.Height)
    }
    finally {
        $image.Dispose()
    }
}

function Add-FitPicture {
    param(
        $Slide,
        [string]$Path,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [switch]$Border
    )
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Missing figure: $Path"
    }
    $size = Get-ImageSize -Path $Path
    $ratio = $size[0] / $size[1]
    $boxRatio = $Width / $Height
    if ($ratio -gt $boxRatio) {
        $drawWidth = $Width
        $drawHeight = $Width / $ratio
        $drawLeft = $Left
        $drawTop = $Top + (($Height - $drawHeight) / 2)
    }
    else {
        $drawHeight = $Height
        $drawWidth = $Height * $ratio
        $drawTop = $Top
        $drawLeft = $Left + (($Width - $drawWidth) / 2)
    }
    $shape = $Slide.Shapes.AddPicture($Path, 0, -1, $drawLeft, $drawTop, $drawWidth, $drawHeight)
    if ($Border) {
        $shape.Line.Visible = -1
        $shape.Line.ForeColor.RGB = $lightGray
        $shape.Line.Weight = 0.75
    }
    return $shape
}

function Add-Text {
    param(
        $Slide,
        [string]$Text,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$Size = 18,
        [int]$Color = $dark,
        [switch]$Bold,
        [int]$Alignment = 1
    )
    $shape = $Slide.Shapes.AddTextBox(1, $Left, $Top, $Width, $Height)
    $shape.TextFrame.MarginLeft = 0
    $shape.TextFrame.MarginRight = 0
    $shape.TextFrame.MarginTop = 0
    $shape.TextFrame.MarginBottom = 0
    $shape.TextFrame.WordWrap = -1
    $shape.TextFrame.AutoSize = 0
    $range = $shape.TextFrame.TextRange
    $range.Text = $Text
    $range.Font.Name = 'Arial'
    $range.Font.Size = $Size
    $range.Font.Color.RGB = $Color
    $range.Font.Bold = $(if ($Bold) { -1 } else { 0 })
    $range.ParagraphFormat.Alignment = $Alignment
    return $shape
}

function Add-Bullets {
    param(
        $Slide,
        [string[]]$Items,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [double]$Height,
        [double]$Size = 18,
        [double]$SpaceAfter = 8
    )
    $bullet = [char]0x2022
    $shape = Add-Text -Slide $Slide -Text (($Items | ForEach-Object { "$bullet $_" }) -join "`r") -Left $Left -Top $Top -Width $Width -Height $Height -Size $Size
    $range = $shape.TextFrame.TextRange
    for ($i = 1; $i -le $Items.Count; $i++) {
        $paragraph = $range.Paragraphs($i)
        $paragraph.ParagraphFormat.SpaceAfter = $SpaceAfter
        $paragraph.ParagraphFormat.SpaceWithin = 1.0
    }
    return $shape
}

function Add-SectionLabel {
    param($Slide, [string]$Text, [double]$Left, [double]$Top, [double]$Width)
    $line = $Slide.Shapes.AddShape(1, $Left, $Top + 24, $Width, 2)
    $line.Fill.ForeColor.RGB = $red
    $line.Line.Visible = 0
    Add-Text -Slide $Slide -Text $Text -Left $Left -Top $Top -Width $Width -Height 24 -Size 18 -Color $red -Bold | Out-Null
}

function Add-Metric {
    param(
        $Slide,
        [string]$Value,
        [string]$Label,
        [double]$Left,
        [double]$Top,
        [double]$Width,
        [int]$Accent = $red
    )
    $bar = $Slide.Shapes.AddShape(1, $Left, $Top, 4, 48)
    $bar.Fill.ForeColor.RGB = $Accent
    $bar.Line.Visible = 0
    Add-Text -Slide $Slide -Text $Value -Left ($Left + 11) -Top ($Top - 2) -Width ($Width - 11) -Height 28 -Size 22 -Color $dark -Bold | Out-Null
    Add-Text -Slide $Slide -Text $Label -Left ($Left + 11) -Top ($Top + 27) -Width ($Width - 11) -Height 34 -Size 18 -Color $midGray | Out-Null
}

function Add-FlowBox {
    param($Slide, [string]$Title, [string]$Subtitle, [double]$Left, [double]$Top, [double]$Width, [int]$Number)
    $box = $Slide.Shapes.AddShape(1, $Left, $Top, $Width, 130)
    $box.Fill.ForeColor.RGB = $white
    $box.Line.ForeColor.RGB = $lightGray
    $box.Line.Weight = 1.1
    $num = $Slide.Shapes.AddShape(9, $Left + 12, $Top + 12, 33, 33)
    $num.Fill.ForeColor.RGB = $red
    $num.Line.Visible = 0
    $num.TextFrame.TextRange.Text = [string]$Number
    $num.TextFrame.TextRange.Font.Name = 'Arial'
    $num.TextFrame.TextRange.Font.Size = 16
    $num.TextFrame.TextRange.Font.Bold = -1
    $num.TextFrame.TextRange.Font.Color.RGB = $white
    $num.TextFrame.TextRange.ParagraphFormat.Alignment = 2
    $num.TextFrame.VerticalAnchor = 3
    Add-Text -Slide $Slide -Text $Title -Left ($Left + 55) -Top ($Top + 14) -Width ($Width - 65) -Height 30 -Size 18 -Bold | Out-Null
    Add-Text -Slide $Slide -Text $Subtitle -Left ($Left + 16) -Top ($Top + 57) -Width ($Width - 32) -Height 65 -Size 18 -Color $midGray | Out-Null
}

function Set-Title {
    param($Slide, [string]$Title)
    for ($i = 1; $i -le $Slide.Shapes.Count; $i++) {
        $shape = $Slide.Shapes.Item($i)
        try {
            if ($shape.HasTextFrame -and $shape.Top -lt 95 -and $shape.Left -lt 800) {
                $shape.TextFrame.TextRange.Text = $Title
                $shape.TextFrame.TextRange.Font.Name = 'Arial'
                $shape.TextFrame.TextRange.Font.Size = 24
                $shape.TextFrame.TextRange.Font.Color.RGB = $red
                return
            }
        }
        catch {}
    }
}

function Set-Navigation {
    param($Slide, [string[]]$Labels, [int]$ActiveIndex)
    $nav = $null
    for ($i = 1; $i -le $Slide.Shapes.Count; $i++) {
        $shape = $Slide.Shapes.Item($i)
        try {
            if ($shape.HasTextFrame -and $shape.HasTextFrame -and $shape.Left -gt 800 -and $shape.TextFrame.HasText) {
                $nav = $shape
            }
        }
        catch {}
    }
    if ($null -eq $nav) { return }
    $nav.TextFrame.TextRange.Text = (($Labels | ForEach-Object -Begin {$i = 1} -Process { "$i. $_"; $i++ }) -join "`r")
    $nav.TextFrame.MarginLeft = 7.2
    $nav.TextFrame.MarginRight = 7.2
    $nav.TextFrame.MarginTop = 3
    $range = $nav.TextFrame.TextRange
    for ($i = 1; $i -le $Labels.Count; $i++) {
        $paragraph = $range.Paragraphs($i)
        $paragraph.Font.Name = 'Arial'
        $paragraph.Font.Color.RGB = $midGray
        $paragraph.Font.Size = $(if ($i -eq $ActiveIndex) { 16 } else { 12 })
        $paragraph.Font.Bold = $(if ($i -eq $ActiveIndex) { -1 } else { 0 })
        $paragraph.ParagraphFormat.SpaceAfter = 3
    }
}

function Clear-ContentArea {
    param($Slide)
    for ($i = $Slide.Shapes.Count; $i -ge 1; $i--) {
        $shape = $Slide.Shapes.Item($i)
        if ($shape.Left -lt 810 -and $shape.Top -gt 95) {
            $shape.Delete()
        }
    }
}

function Set-Divider {
    param($Slide, [string]$Title)
    for ($i = $Slide.Shapes.Count; $i -ge 1; $i--) {
        $Slide.Shapes.Item($i).Delete()
    }
    $background = $Slide.Shapes.AddShape(1, 0, 132, 960, 408)
    $background.Fill.ForeColor.RGB = $red
    $background.Line.Visible = 0
    Add-Text -Slide $Slide -Text $Title -Left 62 -Top 255 -Width 760 -Height 100 -Size 31 -Color $white -Bold | Out-Null
    try {
        for ($i = 1; $i -le $Slide.Master.Shapes.Count; $i++) {
            $shape = $Slide.Master.Shapes.Item($i)
            if ($shape.Left -gt 825 -and $shape.Top -gt 450) {
                $shape.Copy()
                $null = $Slide.Shapes.Paste()
                break
            }
        }
    }
    catch {}
}

function Add-TitleBackground {
    param($Slide)
    $builder = $Slide.Shapes.BuildFreeform(1, 0, 378)
    $builder.AddNodes(0, 1, 960, 350)
    $builder.AddNodes(0, 1, 960, 540)
    $builder.AddNodes(0, 1, 0, 540)
    $shape = $builder.ConvertToShape()
    $shape.Fill.ForeColor.RGB = $red
    $shape.Line.Visible = 0
    $shape.ZOrder(1)
    try {
        for ($i = 1; $i -le $Slide.Master.Shapes.Count; $i++) {
            $masterShape = $Slide.Master.Shapes.Item($i)
            if ($masterShape.Left -gt 825 -and $masterShape.Top -gt 450) {
                $masterShape.Copy()
                $null = $Slide.Shapes.Paste()
                break
            }
        }
    }
    catch {}
}

function Add-FooterOverlay {
    param($Slide, [int]$PageNumber)
    $cover = $Slide.Shapes.AddShape(1, 0, 501, 815, 39)
    $cover.Fill.ForeColor.RGB = $red
    $cover.Line.Visible = 0
    Add-Text -Slide $Slide -Text "Seite $PageNumber" -Left 43 -Top 507 -Width 95 -Height 18 -Size 10 -Color $white | Out-Null
    Add-Text -Slide $Slide -Text 'Florian Rzepka | Disputation | 2026' -Left 128 -Top 507 -Width 560 -Height 18 -Size 10 -Color $white | Out-Null
}

if (-not (Test-Path -LiteralPath $TemplatePath)) {
    throw "Template not found: $TemplatePath"
}

$slides = @(
    @{ Type = 'title' },
    @{ Type = 'agenda' },
    @{ Type = 'divider'; Title = 'Motivation und Forschungsfragen' },
    @{ Type = 'content'; Section = 1; Title = 'Warum deployment-orientierte Zustandsbestimmung?' },
    @{ Type = 'content'; Section = 1; Title = 'Forschungsfragen und roter Faden' },
    @{ Type = 'divider'; Title = 'State of the Art und Research Gap' },
    @{ Type = 'content'; Section = 2; Title = 'State of the Art: Battery State Estimation' },
    @{ Type = 'content'; Section = 2; Title = 'Research Gap und Beitrag dieser Arbeit' },
    @{ Type = 'divider'; Title = 'Datenbasis und BMS-Plattform' },
    @{ Type = 'content'; Section = 3; Title = 'Zwei Datenbasen für drei Untersuchungen' },
    @{ Type = 'content'; Section = 3; Title = 'Custom BMS als gemeinsame Experimental Platform' },
    @{ Type = 'divider'; Title = 'Neural SOH Estimation: Wie viel Modellkomplexität ist nötig?' },
    @{ Type = 'content'; Section = 4; Title = 'Temporal Context durch Feature Engineering' },
    @{ Type = 'content'; Section = 4; Title = 'Compact MLP für die SOH Estimation' },
    @{ Type = 'content'; Section = 4; Title = 'Ergebnis: Geeignete Features ermöglichen kleine Netze' },
    @{ Type = 'divider'; Title = 'Zuverlässigkeit und Robustheit der Batteriezustandsbestimmung' },
    @{ Type = 'content'; Section = 5; Title = 'Vier Estimator-Klassen im direkten Vergleich' },
    @{ Type = 'content'; Section = 5; Title = 'Nominale Accuracy ist nur der Ausgangspunkt' },
    @{ Type = 'content'; Section = 5; Title = 'Messstörungen verändern die Estimator-Rangfolge' },
    @{ Type = 'content'; Section = 5; Title = 'Der beste Estimator hängt vom Fehlerfall ab' },
    @{ Type = 'divider'; Title = 'Embedded AI: Vom Neural Network zum Microcontroller' },
    @{ Type = 'content'; Section = 6; Title = 'Deployment Chain: vom PyTorch Model zur STM32 Inference' },
    @{ Type = 'content'; Section = 6; Title = 'Pruning und Quantization für Embedded Deployment' },
    @{ Type = 'content'; Section = 6; Title = 'Accuracy und Stabilität im Streaming Replay' },
    @{ Type = 'content'; Section = 6; Title = 'Trade-offs bei Flash, RAM, Latency und Utility' },
    @{ Type = 'divider'; Title = 'Gesamtergebnisse, Schlussfolgerungen und Ausblick' },
    @{ Type = 'content'; Section = 7; Title = 'Was die drei Studien gemeinsam zeigen' },
    @{ Type = 'content'; Section = 7; Title = 'Design Principles und Next Steps' },
    @{ Type = 'divider'; Title = 'Vielen Dank | Fragen' }
)

$navigation = @('Motivation', 'State of Art', 'Daten & BMS', 'Neural SOH', 'Robustheit', 'Embedded AI', 'Ergebnisse')

if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}
Copy-Item -LiteralPath $TemplatePath -Destination $OutputPath

$powerPoint = New-Object -ComObject PowerPoint.Application
$powerPoint.Visible = -1
$presentation = $null

try {
    $presentation = $powerPoint.Presentations.Open($OutputPath, $false, $false, $false)
    $originalCount = $presentation.Slides.Count

    foreach ($spec in $slides) {
        $sourceIndex = switch ($spec.Type) {
            'title' { 1 }
            'agenda' { 2 }
            'divider' { 3 }
            default { 4 }
        }
        $presentation.Slides.Item($sourceIndex).Copy()
        $null = $presentation.Slides.Paste($presentation.Slides.Count + 1)
    }

    for ($i = 1; $i -le $originalCount; $i++) {
        $presentation.Slides.Item(1).Delete()
    }

    # Title slide
    $titleSlide = $presentation.Slides.Item(1)
    $university = "Technische Universit$([char]0x00E4)t Berlin"
    $titleShape = $titleSlide.Shapes.Item('TextBox 6')
    $titleShape.TextFrame.TextRange.Text = "Deployment-Oriented Neural Network State Estimation for Battery Management Systems"
    $titleShape.TextFrame.TextRange.Font.Size = 26
    $subtitleShape = $titleSlide.Shapes.Item('TextBox 8')
    $subtitleShape.TextFrame.TextRange.Text = "Florian Rzepka | $university | Disputation 2026"
    $subtitleShape.TextFrame.TextRange.Font.Size = 14
    Add-TitleBackground -Slide $titleSlide

    # Agenda slide
    $agenda = $presentation.Slides.Item(2)
    $agenda.Shapes.Item(2).TextFrame.TextRange.Text = 'Gliederung'
    $agendaItems = @(
        'Motivation und Forschungsfragen',
        'State of the Art und Research Gap',
        'Datenbasis und BMS-Plattform',
        'Neural SOH Estimation: Wie viel Modellkomplexität ist nötig?',
        'Zuverlässigkeit und Robustheit der Batteriezustandsbestimmung',
        'Embedded AI: Vom Neural Network zum Microcontroller',
        'Gesamtergebnisse, Schlussfolgerungen und Ausblick'
    )
    $agenda.Shapes.Item(1).Top = 105
    $agenda.Shapes.Item(1).Height = 330
    $agenda.Shapes.Item(1).TextFrame.TextRange.Text = ($agendaItems -join "`r")
    $agendaRange = $agenda.Shapes.Item(1).TextFrame.TextRange
    for ($i = 1; $i -le $agendaItems.Count; $i++) {
        $p = $agendaRange.Paragraphs($i)
        $p.Font.Name = 'Arial'
        $p.Font.Size = 18
        $p.Font.Color.RGB = $dark
        $p.ParagraphFormat.SpaceAfter = 8
    }

    for ($index = 3; $index -le $slides.Count; $index++) {
        $spec = $slides[$index - 1]
        $slide = $presentation.Slides.Item($index)
        if ($spec.Type -eq 'divider') {
            Set-Divider -Slide $slide -Title $spec.Title
        }
        elseif ($spec.Type -eq 'content') {
            Clear-ContentArea -Slide $slide
            Set-Title -Slide $slide -Title $spec.Title
            Set-Navigation -Slide $slide -Labels $navigation -ActiveIndex $spec.Section
        }
    }

    # 4: Motivation
    $s = $presentation.Slides.Item(4)
    Add-Bullets $s @(
        'Stationäre Microgrids und mobile Systeme mit begrenzter Kommunikation oder Wartung',
        'SOC und SOH müssen lokal, kontinuierlich und kausal verfügbar sein',
        'Embedded systems begrenzen Speicher, Rechenzeit und Energie',
        'Praktische Eignung verbindet Accuracy, Robustness und Hardwarebedarf'
    ) 50 125 300 285 18 10 | Out-Null
    Add-FitPicture $s (Join-Path $assetRoot 'mg_farm_context.png') 360 105 425 315 | Out-Null
    Add-Text $s 'Lokale Zustandsbestimmung als Voraussetzung für autonomen Betrieb' 385 420 375 36 18 $red -Bold 2 | Out-Null

    # 5: Research path
    $s = $presentation.Slides.Item(5)
    Add-FlowBox $s 'Modellkomplexität' 'Wie einfach darf das neuronale Netz sein?' 48 130 224 1
    Add-FlowBox $s 'Robustness' 'Wie stabil bleibt der Estimator bei Störungen?' 305 130 224 2
    Add-FlowBox $s 'Embedded deployment' 'Wie gelingt die Ausführung auf dem MCU?' 562 130 224 3
    foreach ($x in @(274, 531)) {
        $arrow = $s.Shapes.AddShape(33, $x, 172, 28, 24)
        $arrow.Fill.ForeColor.RGB = $red
        $arrow.Line.Visible = 0
    }
    Add-Bullets $s @(
        'RQ1: Wie komplex müssen neuronale Netze für eine genaue SOC- und SOH-Bestimmung tatsächlich sein?',
        'RQ2: Wie verändern Messstörungen und Zustandsfehler die Zuverlässigkeit verschiedener Estimator-Klassen?',
        'RQ3: Wie lassen sich neuronale Estimatoren innerhalb der Memory-, Latency- und Energy-Limits eines BMS einsetzen?'
    ) 70 285 700 165 18 8 | Out-Null

    # 7: State of the art
    $s = $presentation.Slides.Item(7)
    Add-FitPicture $s (Join-Path $paletteRoot 'estimator_families.png') 48 108 730 230 | Out-Null
    Add-Bullets $s @(
        'Coulomb Counting: geringer Rechenaufwand, aber akkumulierte Drift',
        'Kalman-filter- und ECM-based approaches: physikalische Korrektur und transparente State Propagation',
        'Neural estimators: flexible nonlinear mapping und temporal context, aber abhängig von repräsentativen Daten'
    ) 65 345 700 125 18 7 | Out-Null

    # 8: Research gap
    $s = $presentation.Slides.Item(8)
    Add-FlowBox $s 'Model Design' 'Model complexity kompensiert fehlende task-aligned features' 48 125 224 1
    Add-FlowBox $s 'Evaluation' 'Clean MAE statt Bias, Noise, Dropouts und Recovery' 305 125 224 2
    Add-FlowBox $s 'Deployment' 'Offline inference statt measured MCU behavior' 562 125 224 3
    Add-Bullets $s @(
        'Research Gap: keine durchgängige Bewertung von representation, robustness und embedded execution',
        'Beitrag: drei verknüpfte Studien entlang gemeinsamer BMS Requirements'
    ) 75 300 690 125 18 10 | Out-Null

    # 10: Data basis
    $s = $presentation.Slides.Item(10)
    Add-SectionLabel $s 'NMC Ageing Study' 45 105 350
    Add-FitPicture $s (Join-Path $paletteRoot 'paper1_soh_cycles.png') 45 142 350 190 -Border | Out-Null
    Add-Bullets $s @(
        '14 Zellen und 7 Alterungsszenarien',
        'Periodische Referenzkapazität als SOH label'
    ) 55 345 330 95 18 7 | Out-Null
    Add-SectionLabel $s 'LFP DoE Campaign' 425 105 350
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_doe_cube.png') 425 142 350 190 -Border | Out-Null
    Add-Bullets $s @(
        '15 Zellen und 8 operating points',
        '1-Hz measurements für SOC, SOH und Robustness tests'
    ) 435 345 330 95 18 7 | Out-Null

    # 11: Custom BMS platform
    $s = $presentation.Slides.Item(11)
    Add-FitPicture $s (Join-Path $assetRoot 'custom_bms_stm32_overview.png') 35 120 485 315 | Out-Null
    Add-Bullets $s @(
        'Custom 4s BMS: Zellspannungen, Strom und Temperatur über Battery-Monitoring IC',
        'STM32H755 Dual Core: Cortex-M7 mit 480 MHz und Cortex-M4 mit 240 MHz',
        'M7: State Estimation | M4: sensing, balancing und communication',
        'Open firmware für austauschbare Estimatoren und wiederholbare Hardwaretests'
    ) 535 120 245 300 18 7 | Out-Null

    # 13: Feature representation
    $s = $presentation.Slides.Item(13)
    Add-FitPicture $s (Join-Path $paletteRoot 'paper1_lag_sequence.png') 42 108 515 315 | Out-Null
    Add-SectionLabel $s 'Temporal Context' 575 118 205
    Add-Bullets $s @(
        'Die Historie wird in den Feature-Vektor verlagert',
        'Das MLP benötigt dadurch keinen recurrent state',
        'Die Kontextlänge steuert Informationsgehalt und Eingangsgröße'
    ) 575 165 205 225 18 10 | Out-Null

    # 14: MLP architecture
    $s = $presentation.Slides.Item(14)
    Add-FitPicture $s (Join-Path $paletteRoot 'paper1_architecture.png') 45 103 490 340 | Out-Null
    Add-Bullets $s @(
        'Bewusst einfache feedforward architecture als Test der Repräsentation',
        'Zeitliche Information entsteht außerhalb des Netzes durch Feature Engineering',
        'Modellkapazität und zeitliche Informationsaufbereitung bleiben getrennt bewertbar'
    ) 555 135 225 250 18 10 | Out-Null

    # 15: Ageing results
    $s = $presentation.Slides.Item(15)
    Add-FitPicture $s (Join-Path $paletteRoot 'paper1_soh_pred.png') 42 102 735 250 -Border | Out-Null
    Add-Metric $s '1,10 pp' 'mittlerer SOH MAE' 60 370 200
    Add-Metric $s '160 min' 'beste History-Konfiguration' 285 370 215 $blue
    Add-Metric $s 'Cell 9' 'Grenze bei atypischer Alterung' 535 370 220 $midGray

    # 17: Robustness methodology
    $s = $presentation.Slides.Item(17)
    Add-FitPicture $s (Join-Path $paletteRoot 'robustness_methodology.png') 45 108 735 205 | Out-Null
    Add-Bullets $s @(
        'Identischer Messstrom und gleiche Störungsverläufe isolieren das Estimator-Verhalten',
        'Vier Korrekturprinzipien statt vier unterschiedlich getesteter Modelle',
        'Getrennte Evidenz für nominale Accuracy, Recovery und Disturbance response'
    ) 70 325 690 135 18 7 | Out-Null

    # 18: Baseline results
    $s = $presentation.Slides.Item(18)
    Add-FitPicture $s (Join-Path $assetRoot 'robustness_baseline_panel_a.png') 40 105 500 350 -Border | Out-Null
    Add-Bullets $s @(
        'Unter nominalen Signalen profitiert HDM von der SOH-gestützten Kapazitätsanpassung',
        'Die clean ranking dient nur als Referenz für die Disturbance tests',
        'Der niedrigste MAE sagt noch nichts über Drift oder Recovery aus'
    ) 565 130 215 270 18 11 | Out-Null

    # 19: Disturbance results
    $s = $presentation.Slides.Item(19)
    Add-FitPicture $s (Join-Path $paletteRoot 'robustness_cross_scenario.png') 42 102 535 345 -Border | Out-Null
    Add-Bullets $s @(
        'Persistenter Bias akkumuliert in integrationsbasierten Pfaden',
        'Voltage feedback begrenzt systematischen Drift',
        'Temporal context unterstützt bei Dropouts und Recovery',
        'Die Rangfolge folgt dem dominanten Fehlermechanismus'
    ) 595 125 190 300 18 9 | Out-Null

    # 20: Robustness synthesis
    $s = $presentation.Slides.Item(20)
    Add-FitPicture $s (Join-Path $paletteRoot 'robustness_decision.png') 42 102 735 340 -Border | Out-Null
    Add-Text $s 'Estimator-Auswahl = relevantes Störprofil + geeigneter Korrekturmechanismus' 78 418 660 38 18 $dark -Bold 2 | Out-Null

    # 22: Embedded pipeline
    $s = $presentation.Slides.Item(22)
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_lstm_schematic.png') 48 100 730 135 | Out-Null
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_pipeline.png') 48 245 730 155 | Out-Null
    Add-Bullets $s @(
        'Ein identischer recurrent inference path isoliert den Effekt der Compression',
        'Flash, RAM und kernel time ergänzen die Accuracy um gemessene Hardwaremetriken'
    ) 55 405 715 48 18 2 | Out-Null

    # 23: Compression routes
    $s = $presentation.Slides.Item(23)
    Add-SectionLabel $s 'Structured pruning' 45 105 350
    Add-FitPicture $s (Join-Path $paletteRoot 'structured_lstm_pruning_numeric.png') 45 142 350 220 -Border | Out-Null
    Add-Bullets $s @('Komplette hidden channels entfernt', 'Kleinere dense tensors nach fine-tuning') 55 372 330 72 18 3 | Out-Null
    Add-SectionLabel $s 'INT8 weight quantization' 425 105 350
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_quantization_schematic.png') 425 142 350 220 -Border | Out-Null
    Add-Bullets $s @('Recurrent weights auf signed INT8 gemappt', 'States, scales und MLP bleiben FP32') 435 372 330 72 18 3 | Out-Null

    # 24: Accuracy and stability
    $s = $presentation.Slides.Item(24)
    Add-SectionLabel $s 'SOC streaming replay' 42 103 405
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_soc_dashboard.png') 42 140 405 275 -Border | Out-Null
    Add-SectionLabel $s 'SOH streaming replay' 475 103 305
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_soh_dashboard.png') 475 140 305 155 -Border | Out-Null
    Add-Bullets $s @(
        'SOC MAE [pp]: Base 2,68 | Pruned 2,34 | Quantized 2,79',
        'SOH MAE [pp]: Base 0,85 | Pruned 1,46 | Quantized 1,41',
        'Kein wachsender Quantized-to-Base drift im vollständigen replay'
    ) 485 305 295 145 18 7 | Out-Null

    # 25: Resource trade-offs
    $s = $presentation.Slides.Item(25)
    Add-FitPicture $s (Join-Path $paletteRoot 'embedded_model_sizes.png') 40 100 500 350 -Border | Out-Null
    Add-SectionLabel $s 'SOC' 560 105 220
    Add-Bullets $s @(
        'Pruned: -41% Flash | -43% kernel time',
        'Quantized: -50% Flash | +399% kernel time',
        'Utility: 0,71 vs. 1,83'
    ) 560 145 220 135 18 5 | Out-Null
    Add-SectionLabel $s 'SOH' 560 285 220
    Add-Bullets $s @(
        'Pruned: -46% Flash | -44% kernel time',
        'Quantized: -59% Flash | +29% kernel time',
        'Hardwarelimit entscheidet'
    ) 560 325 220 125 18 5 | Out-Null

    # 27: Integrated findings
    $s = $presentation.Slides.Item(27)
    Add-FlowBox $s 'Repräsentation' 'Explizite Historie ermöglicht kompakte Modelle' 48 125 224 1
    Add-FlowBox $s 'Validierung' 'Clean accuracy und Robustness liefern getrennte Evidenz' 305 125 224 2
    Add-FlowBox $s 'Implementierung' 'Compression bleibt hardware- und firmwareabhängig' 562 125 224 3
    Add-Bullets $s @(
        'Task-aligned features können notwendige Modellkomplexität begrenzen',
        'Clean MAE kann operationale Robustness nicht nachweisen',
        'Deployment-Eignung entsteht aus der gesamten signal-to-hardware chain'
    ) 75 295 690 145 18 10 | Out-Null

    # 28: Design principles and outlook
    $s = $presentation.Slides.Item(28)
    Add-SectionLabel $s 'Kernaussagen' 50 108 335
    Add-Bullets $s @(
        'Representation vor unnötiger Modellkomplexität',
        'Disturbance tests zusätzlich zu clean MAE',
        'Compression passend zum tatsächlichen Bottleneck',
        'Target-hardware validation vor dem Einsatz'
    ) 50 150 335 220 18 9 | Out-Null
    Add-SectionLabel $s 'Nächste Schritte' 440 108 335
    Add-Bullets $s @(
        'Weitere cells, chemistries und operating profiles',
        'Fault injection in sensing und firmware',
        'Joint pruning and quantization',
        'Größere Packs und autonomer Langzeitbetrieb'
    ) 440 150 335 220 18 9 | Out-Null
    Add-Text $s 'Zielbild: integrierte, testbare und resource-aware State Estimation im BMS' 75 404 690 42 17 $red -Bold 2 | Out-Null

    for ($i = 1; $i -le $presentation.Slides.Count; $i++) {
        Add-FooterOverlay -Slide $presentation.Slides.Item($i) -PageNumber $i
    }

    $presentation.Save()

    $pdfPath = [System.IO.Path]::ChangeExtension($OutputPath, '.pdf')
    if (Test-Path -LiteralPath $pdfPath) {
        Remove-Item -LiteralPath $pdfPath -Force
    }
    $presentation.SaveAs($pdfPath, 32)
}
finally {
    if ($null -ne $presentation) {
        $presentation.Close()
        [void][Runtime.InteropServices.Marshal]::ReleaseComObject($presentation)
    }
    $powerPoint.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($powerPoint)
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}

Write-Host "Created: $OutputPath"
Write-Host "Created: $([System.IO.Path]::ChangeExtension($OutputPath, '.pdf'))"
