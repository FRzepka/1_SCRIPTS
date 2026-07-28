param(
    [string]$ImagePath = (Join-Path (Resolve-Path (Join-Path $PSScriptRoot '..')).Path 'pictures\eaai_palette\bms_requirements_icon_white.png')
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$resolvedImage = (Resolve-Path -LiteralPath $ImagePath).Path
$tempImage = Join-Path (Split-Path -Parent $resolvedImage) 'bms_requirements_center_white.tmp.png'

$source = [System.Drawing.Bitmap]::new($resolvedImage)
$bitmap = [System.Drawing.Bitmap]::new(
    $source.Width,
    $source.Height,
    [System.Drawing.Imaging.PixelFormat]::Format32bppArgb
)

try {
    $graphics = [System.Drawing.Graphics]::FromImage($bitmap)
    try {
        $graphics.DrawImage($source, 0, 0, $source.Width, $source.Height)
    }
    finally {
        $graphics.Dispose()
    }
    $source.Dispose()
    $source = $null

    $centerX = [int][Math]::Round($bitmap.Width * 0.5033)
    $centerY = [int][Math]::Round($bitmap.Height * 0.7811)
    $radius = [int][Math]::Round([Math]::Min($bitmap.Width, $bitmap.Height) * 0.056)

    $baseR = 0xC5
    $baseG = 0x59
    $baseB = 0x57
    $darkR = 0x19
    $darkG = 0x19
    $darkB = 0x19

    $vectorR = $darkR - $baseR
    $vectorG = $darkG - $baseG
    $vectorB = $darkB - $baseB
    $denominator = $vectorR * $vectorR + $vectorG * $vectorG + $vectorB * $vectorB

    for ($y = $centerY - $radius; $y -le $centerY + $radius; $y++) {
        for ($x = $centerX - $radius; $x -le $centerX + $radius; $x++) {
            $dx = $x - $centerX
            $dy = $y - $centerY
            if ($dx * $dx + $dy * $dy -gt $radius * $radius) {
                continue
            }

            $pixel = $bitmap.GetPixel($x, $y)
            $projection = (
                ($pixel.R - $baseR) * $vectorR +
                ($pixel.G - $baseG) * $vectorG +
                ($pixel.B - $baseB) * $vectorB
            ) / $denominator

            if ($projection -le 0.06) {
                continue
            }

            $blend = [Math]::Min(1.0, [Math]::Max(0.0, $projection))
            $red = [int][Math]::Round($baseR + $blend * (255 - $baseR))
            $green = [int][Math]::Round($baseG + $blend * (255 - $baseG))
            $blue = [int][Math]::Round($baseB + $blend * (255 - $baseB))
            $bitmap.SetPixel($x, $y, [System.Drawing.Color]::FromArgb($pixel.A, $red, $green, $blue))
        }
    }

    $bitmap.Save($tempImage, [System.Drawing.Imaging.ImageFormat]::Png)
}
finally {
    if ($null -ne $source) {
        $source.Dispose()
    }
    $bitmap.Dispose()
}

Copy-Item -LiteralPath $tempImage -Destination $resolvedImage -Force
Remove-Item -LiteralPath $tempImage -Force
Write-Host "Updated central BMS symbol in: $resolvedImage"
