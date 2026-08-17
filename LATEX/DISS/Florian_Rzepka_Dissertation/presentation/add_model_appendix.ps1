$ErrorActionPreference = 'Stop'

$src = (Resolve-Path (Join-Path $PSScriptRoot 'Diss_FR_Dissertation_30min_v3.pptx')).Path
$out = Join-Path $PSScriptRoot 'Diss_FR_Dissertation_30min_v4.pptx'
$preview = Join-Path $PSScriptRoot '_appendix_preview'
if (-not (Test-Path $preview)) {
    New-Item -ItemType Directory -Path $preview | Out-Null
}

$red = 1969604
$lightGray = 15856113
$midGray = 14277081
$dark = 4408131
$white = 16777215
$borderColor = 11776947
$ppLayoutBlank = 12
$msoTextOrientationHorizontal = 1
$msoShapeRectangle = 1
$ppAlignLeft = 1
$ppAlignCenter = 2
$msoAnchorMiddle = 3
$msoTrue = -1
$msoFalse = 0

function Add-TextBox {
    param($Slide, $Left, $Top, $Width, $Height, $Text, $Size, $Color, $Bold = $false, $Align = 1)

    $shape = $Slide.Shapes.AddTextbox($msoTextOrientationHorizontal, $Left, $Top, $Width, $Height)
    $shape.TextFrame.TextRange.Text = $Text
    $shape.TextFrame.MarginLeft = 2
    $shape.TextFrame.MarginRight = 2
    $shape.TextFrame.MarginTop = 1
    $shape.TextFrame.MarginBottom = 1
    $shape.TextFrame.VerticalAnchor = $msoAnchorMiddle
    $shape.TextFrame.TextRange.Font.Name = 'Arial'
    $shape.TextFrame.TextRange.Font.Size = $Size
    $shape.TextFrame.TextRange.Font.Color.RGB = $Color
    $shape.TextFrame.TextRange.Font.Bold = if ($Bold) { $msoTrue } else { $msoFalse }
    $shape.TextFrame.TextRange.ParagraphFormat.Alignment = $Align
    return $shape
}

function Add-Frame {
    param($Slide, $Title, $Page)

    $null = Add-TextBox $Slide 43.8 31.2 730 56.5 $Title 24 $red $true $ppAlignLeft

    $side = $Slide.Shapes.AddShape($msoShapeRectangle, 815, 0, 145, 465.1)
    $side.Fill.ForeColor.RGB = $midGray
    $side.Line.Visible = $msoFalse
    $null = Add-TextBox $Slide 826 205 120 40 'Anhang' 18 $red $true $ppAlignCenter

    $footer = $Slide.Shapes.AddShape($msoShapeRectangle, 0, 501, 815, 39)
    $footer.Fill.ForeColor.RGB = $red
    $footer.Line.Visible = $msoFalse
    $null = Add-TextBox $Slide 43 507 95 21.8 ("Seite {0}" -f $Page) 10 $white $false $ppAlignLeft
    $null = Add-TextBox $Slide 128 507 560 21.8 'Florian Rzepka | Disputation | 2026' 10 $white $false $ppAlignLeft
}

function Add-Table {
    param($Slide, $Left, $Top, $Width, $Height, $Headers, $Rows, $ColumnWidths, $FontSize = 12)

    $rowCount = $Rows.Count + 1
    $columnCount = $Headers.Count
    $shape = $Slide.Shapes.AddTable($rowCount, $columnCount, $Left, $Top, $Width, $Height)
    $table = $shape.Table

    for ($column = 1; $column -le $columnCount; $column++) {
        $table.Columns.Item($column).Width = $ColumnWidths[$column - 1]
    }

    for ($row = 1; $row -le $rowCount; $row++) {
        for ($column = 1; $column -le $columnCount; $column++) {
            $cell = $table.Cell($row, $column)
            $text = if ($row -eq 1) { $Headers[$column - 1] } else { $Rows[$row - 2][$column - 1] }
            $cell.Shape.TextFrame.TextRange.Text = [string]$text
            $cell.Shape.TextFrame.MarginLeft = 5
            $cell.Shape.TextFrame.MarginRight = 5
            $cell.Shape.TextFrame.MarginTop = 2
            $cell.Shape.TextFrame.MarginBottom = 2
            $cell.Shape.TextFrame.VerticalAnchor = $msoAnchorMiddle
            $cell.Shape.TextFrame.TextRange.Font.Name = 'Arial'
            $cell.Shape.TextFrame.TextRange.Font.Size = $FontSize
            $cell.Shape.TextFrame.TextRange.Font.Bold = if ($row -eq 1) { $msoTrue } else { $msoFalse }
            $cell.Shape.TextFrame.TextRange.Font.Color.RGB = if ($row -eq 1) { $white } else { $dark }
            $cell.Shape.TextFrame.TextRange.ParagraphFormat.Alignment = if ($column -ge ($columnCount - 1)) { $ppAlignCenter } else { $ppAlignLeft }
            $cell.Shape.Fill.ForeColor.RGB = if ($row -eq 1) { $red } elseif ($row % 2 -eq 0) { $white } else { $lightGray }

            foreach ($borderIndex in 1..4) {
                try {
                    $cellBorder = $cell.Borders.Item($borderIndex)
                    $cellBorder.ForeColor.RGB = $borderColor
                    $cellBorder.Weight = 0.75
                } catch {
                    # Some PowerPoint versions expose only part of the border API.
                }
            }
        }
    }

    return $shape
}

$ppt = New-Object -ComObject PowerPoint.Application
try {
    $presentation = $ppt.Presentations.Open($src, $false, $false, $false)

    $slide1 = $presentation.Slides.Add($presentation.Slides.Count + 1, $ppLayoutBlank)
    Add-Frame $slide1 'Anhang: Übersicht der verwendeten Estimatoren' 30
    $headers1 = @('Modell', 'State', 'Methode', 'MAE [pp]', 'Größe')
    $rows1 = @(
        @('MLP', 'SOH', 'Feedforward-Netz mit 16 Lag-Schritten', '1,10', 'ca. 1,66 MiB FP32'),
        @('DM', 'SOC', 'Coulomb Counting ohne gelernte Gewichte', '3,11', 'keine NN-Gewichte'),
        @('HDM', 'SOC', 'Coulomb Counting mit LSTM-SOH-Korrektur', '0,24', 'ca. 1,46 MiB FP32'),
        @('HECM', 'SOC', '2-RC-ECM, EKF und gemeinsames LSTM-SOH', '0,90', '1,46 MiB + ECM-Tabelle'),
        @('DD', 'SOC', 'GRU-MLP für SOC und gemeinsames LSTM-SOH', '1,15', 'ca. 1,78 MiB FP32')
    )
    $null = Add-Table $slide1 43 105 730 310 $headers1 $rows1 @(62, 58, 300, 90, 220) 12
    $null = Add-TextBox $slide1 45 430 725 42 'MAE in Prozentpunkten. Die FP32-Größen umfassen nur trainierbare Parameter; Datensätze und Testbedingungen unterscheiden sich.' 11 $dark $false $ppAlignLeft

    $slide2 = $presentation.Slides.Add($presentation.Slides.Count + 1, $ppLayoutBlank)
    Add-Frame $slide2 'Anhang: Embedded LSTM-MLP-Modelle' 31
    $headers2 = @('Ziel', 'Variante', 'MAE [pp]', 'Parameter', 'STM32 Flash')
    $rows2 = @(
        @('SOC', 'Base FP32', '2,68', '22.785', '105,32 KiB / 0,103 MiB'),
        @('SOC', 'Pruned FP32', '2,34', 'ca. 13.500', '62,27 KiB / 0,061 MiB'),
        @('SOC', 'Quantized INT8', '2,79', '22.785', '52,48 KiB / 0,051 MiB'),
        @('SOH', 'Base FP32', '0,85', '85.761', '335,00 KiB / 0,327 MiB'),
        @('SOH', 'Pruned FP32', '1,46', '46.697', '182,41 KiB / 0,178 MiB'),
        @('SOH', 'Quantized INT8', '1,41', '85.761', '138,00 KiB / 0,135 MiB')
    )
    $null = Add-Table $slide2 43 105 730 330 $headers2 $rows2 @(72, 175, 105, 120, 258) 12
    $null = Add-TextBox $slide2 45 449 725 30 'Flash umfasst die vollständige STM32-Firmware einschließlich Code, Konstanten und Modellgewichten.' 11 $dark $false $ppAlignLeft

    $slide3 = $presentation.Slides.Add($presentation.Slides.Count + 1, $ppLayoutBlank)
    Add-Frame $slide3 'Anhang: SOH Architecture Study' 32
    $headers3 = @('Architektur', 'Base MAE [pp]', 'Optimized MAE [pp]', 'Base-Größe', 'Optimized-Größe')
    $rows3 = @(
        @('CNN', '1,4011', '1,4672', '1,920 MiB', '0,355 MiB'),
        @('GRU', '1,2377', '1,2427', '3,221 MiB', '0,643 MiB'),
        @('LSTM', '1,4226', '1,3438', '3,418 MiB', '0,658 MiB'),
        @('TCN', '1,2102', '1,1306', '1,678 MiB', '1,188 MiB')
    )
    $null = Add-Table $slide3 43 115 730 255 $headers3 $rows3 @(110, 145, 155, 145, 175) 12
    $null = Add-TextBox $slide3 45 392 725 52 'Optimized = structured pruning + fine-tuning + INT8 quantization. Größen entsprechen dem dokumentierten Speicherbedarf der Modellparameter.' 11 $dark $false $ppAlignLeft

    $presentation.SaveAs($out, 24)
    foreach ($slideIndex in 30..32) {
        $imagePath = Join-Path $preview ("slide_{0}.png" -f $slideIndex)
        $presentation.Slides.Item($slideIndex).Export($imagePath, 'PNG', 1920, 1080)
    }
    $presentation.Close()
} finally {
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}

Get-Item -LiteralPath $out | Select-Object FullName, Length, LastWriteTime
