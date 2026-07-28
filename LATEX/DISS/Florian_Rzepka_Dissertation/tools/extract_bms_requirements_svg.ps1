param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

$sourceSvg = (Resolve-Path (Join-Path $DissertationRoot '..\..\JES\paper_robustness_benchmark\figures\Schematics\Schematics.svg')).Path
$targetDir = Join-Path $DissertationRoot 'pictures\schematics'
$blackTarget = Join-Path $targetDir 'bms_requirements_icon_black.svg'
$whiteTarget = Join-Path $targetDir 'bms_requirements_icon_white.svg'
$svgNamespace = 'http://www.w3.org/2000/svg'

function New-SvgElement {
    param(
        [System.Xml.XmlDocument]$Document,
        [string]$Name,
        [hashtable]$Attributes
    )

    $element = $Document.CreateElement($Name, $svgNamespace)
    foreach ($key in $Attributes.Keys) {
        $element.SetAttribute($key, [string]$Attributes[$key])
    }
    return $element
}

function Add-CentralBoardIcon {
    param(
        [System.Xml.XmlDocument]$Document,
        [string]$Color
    )

    $root = $Document.DocumentElement
    $group = New-SvgElement $Document 'g' @{
        id = 'central-bms-board-vector'
        fill = 'none'
        stroke = $Color
        'stroke-width' = '1.45'
        'stroke-linecap' = 'round'
        'stroke-linejoin' = 'round'
    }

    $outline = New-SvgElement $Document 'rect' @{
        x = '467.8'
        y = '340.2'
        width = '36.8'
        height = '34.2'
        rx = '1.8'
        ry = '1.8'
    }
    [void]$group.AppendChild($outline)

    $chip = New-SvgElement $Document 'rect' @{
        x = '480.7'
        y = '351.6'
        width = '10.2'
        height = '10.2'
        rx = '0.8'
        ry = '0.8'
    }
    [void]$group.AppendChild($chip)

    $paths = @(
        'M 480.7,353.3 H 476.1 V 347.0 H 471.1',
        'M 480.7,356.7 H 471.0',
        'M 480.7,360.1 H 476.0 V 367.5 H 471.0',
        'M 483.1,351.6 V 346.3 H 478.7 V 343.8',
        'M 486.0,351.6 V 343.8',
        'M 488.8,351.6 V 347.2 H 493.8 V 343.8',
        'M 490.9,353.3 H 495.5 V 347.0 H 501.3',
        'M 490.9,356.7 H 501.3',
        'M 490.9,360.1 H 495.5 V 367.5 H 501.3',
        'M 483.1,361.8 V 367.4 H 478.7 V 371.1',
        'M 486.0,361.8 V 371.1',
        'M 488.8,361.8 V 367.4 H 493.8 V 371.1'
    )

    foreach ($data in $paths) {
        $path = New-SvgElement $Document 'path' @{ d = $data }
        [void]$group.AppendChild($path)
    }

    $terminals = @(
        @(471.1, 347.0), @(471.0, 356.7), @(471.0, 367.5),
        @(478.7, 343.8), @(486.0, 343.8), @(493.8, 343.8),
        @(501.3, 347.0), @(501.3, 356.7), @(501.3, 367.5),
        @(478.7, 371.1), @(486.0, 371.1), @(493.8, 371.1)
    )

    foreach ($terminal in $terminals) {
        $circle = New-SvgElement $Document 'circle' @{
            cx = [string]$terminal[0]
            cy = [string]$terminal[1]
            r = '1.05'
            fill = $Color
            stroke = 'none'
        }
        [void]$group.AppendChild($circle)
    }

    [void]$root.AppendChild($group)
}

function Save-Svg {
    param(
        [System.Xml.XmlDocument]$Document,
        [string]$Path
    )

    $settings = [System.Xml.XmlWriterSettings]::new()
    $settings.Indent = $true
    $settings.Encoding = [System.Text.UTF8Encoding]::new($false)
    $writer = [System.Xml.XmlWriter]::Create($Path, $settings)
    try {
        $Document.Save($writer)
    }
    finally {
        $writer.Dispose()
    }
}

$source = [System.Xml.XmlDocument]::new()
$source.PreserveWhitespace = $true
$source.Load($sourceSvg)

$namespaceManager = [System.Xml.XmlNamespaceManager]::new($source.NameTable)
$namespaceManager.AddNamespace('svg', $svgNamespace)
$definitions = $source.SelectSingleNode('/svg:svg/svg:defs', $namespaceManager)
$figureGroup = $source.SelectSingleNode('//*[@id="g35"]', $namespaceManager)

if ($null -eq $definitions -or $null -eq $figureGroup) {
    throw 'Could not locate the BMS requirement figure or its SVG definitions.'
}

New-Item -ItemType Directory -Path $targetDir -Force | Out-Null

$baseDocument = [System.Xml.XmlDocument]::new()
$root = $baseDocument.CreateElement('svg', $svgNamespace)
$root.SetAttribute('width', '1685')
$root.SetAttribute('height', '1060')
$root.SetAttribute('viewBox', '78.564 -36.099 807.753 506.746')
$root.SetAttribute('version', '1.1')
[void]$baseDocument.AppendChild($root)
[void]$root.AppendChild($baseDocument.ImportNode($definitions, $true))
[void]$root.AppendChild($baseDocument.ImportNode($figureGroup, $true))

$embeddedBoard = $baseDocument.SelectSingleNode('//*[@id="image6475-2-0-7"]')
if ($null -eq $embeddedBoard) {
    throw 'Could not locate the embedded central BMS board symbol.'
}
[void]$embeddedBoard.ParentNode.RemoveChild($embeddedBoard)

$blackDocument = $baseDocument.Clone()
Add-CentralBoardIcon $blackDocument '#191919'
Save-Svg $blackDocument $blackTarget

$whiteDocument = $baseDocument.Clone()
Add-CentralBoardIcon $whiteDocument '#ffffff'
Save-Svg $whiteDocument $whiteTarget

Write-Host "Created: $blackTarget"
Write-Host "Created: $whiteTarget"
