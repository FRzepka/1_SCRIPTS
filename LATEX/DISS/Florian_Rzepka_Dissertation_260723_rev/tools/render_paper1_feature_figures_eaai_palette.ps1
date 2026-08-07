param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$sourceDir = Join-Path $DissertationRoot 'pictures'
$outDir = Join-Path $sourceDir 'eaai_palette'

$converterCode = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class PaperOneEaaiFeatureRecolor
{
    static readonly Color Voltage = Color.FromArgb(0xD6, 0x27, 0x28);
    static readonly Color Temperature = Color.FromArgb(0x2C, 0xA0, 0x2C);
    static readonly Color Current = Color.FromArgb(0x1F, 0x77, 0xB4);
    static readonly Color Soh = Color.FromArgb(0x94, 0x67, 0xBD);
    static readonly Color SourceCurrentA = Color.FromArgb(0x10, 0x80, 0x70);
    static readonly Color SourceCurrentB = Color.FromArgb(0x00, 0x78, 0x68);
    static readonly Color SourceSoh = Color.FromArgb(0x00, 0x78, 0x98);

    public static void ConvertImage(string sourcePath, string targetPath)
    {
        using (var src = new Bitmap(sourcePath))
        using (var bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format32bppArgb))
        {
            using (var g = Graphics.FromImage(bmp))
            {
                g.DrawImage(src, 0, 0, src.Width, src.Height);
            }

            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var data = bmp.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                int stride = Math.Abs(data.Stride);
                byte[] bytes = new byte[stride * bmp.Height];
                Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);

                for (int y = 0; y < bmp.Height; y++)
                {
                    int row = y * stride;
                    for (int x = 0; x < bmp.Width; x++)
                    {
                        int p = row + x * 4;
                        byte b = bytes[p + 0];
                        byte g = bytes[p + 1];
                        byte r = bytes[p + 2];
                        byte a = bytes[p + 3];
                        if (a == 0) continue;

                        int max = Math.Max(r, Math.Max(g, b));
                        int min = Math.Min(r, Math.Min(g, b));
                        int chroma = max - min;
                        if (max < 45 || chroma < 30) continue;
                        if (max > 245 && chroma < 20) continue;

                        double hue = Hue(r, g, b, max, min);
                        Color target;
                        if (hue < 24.0 || hue >= 340.0)
                        {
                            target = Voltage;
                        }
                        else if (hue >= 24.0 && hue < 70.0)
                        {
                            target = Temperature;
                        }
                        else if (hue >= 70.0 && hue < 245.0)
                        {
                            double dSoh = ColorDistance(r, g, b, SourceSoh);
                            double dCurrent = Math.Min(
                                ColorDistance(r, g, b, SourceCurrentA),
                                ColorDistance(r, g, b, SourceCurrentB)
                            );
                            target = dSoh < dCurrent ? Soh : Current;
                        }
                        else
                        {
                            continue;
                        }

                        double amount = Math.Max(0.18, Math.Min(1.0, chroma / 185.0));
                        Color bg = max > 235 ? Color.White : EstimateNeutralBackground(r, g, b);
                        Color converted = Mix(bg, target, amount);
                        bytes[p + 0] = converted.B;
                        bytes[p + 1] = converted.G;
                        bytes[p + 2] = converted.R;
                    }
                }

                Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
            if (File.Exists(targetPath))
            {
                File.Delete(targetPath);
            }
            bmp.Save(targetPath, ImageFormat.Png);
        }
    }

    static Color EstimateNeutralBackground(byte r, byte g, byte b)
    {
        int max = Math.Max(r, Math.Max(g, b));
        if (max > 225) return Color.White;
        return Color.FromArgb(0, 0, 0);
    }

    static double ColorDistance(byte r, byte g, byte b, Color target)
    {
        double dr = r - target.R;
        double dg = g - target.G;
        double db = b - target.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }

    static Color Mix(Color bg, Color fg, double amount)
    {
        amount = Math.Max(0.0, Math.Min(1.0, amount));
        return Color.FromArgb(
            Clamp(bg.R * (1.0 - amount) + fg.R * amount),
            Clamp(bg.G * (1.0 - amount) + fg.G * amount),
            Clamp(bg.B * (1.0 - amount) + fg.B * amount)
        );
    }

    static int Clamp(double value)
    {
        if (value < 0.0) return 0;
        if (value > 255.0) return 255;
        return (int)Math.Round(value);
    }

    static double Hue(byte r, byte g, byte b, int max, int min)
    {
        double chroma = max - min;
        if (chroma <= 0.0) return 0.0;
        double h;
        if (max == r)
        {
            h = ((g - b) / chroma) % 6.0;
        }
        else if (max == g)
        {
            h = ((b - r) / chroma) + 2.0;
        }
        else
        {
            h = ((r - g) / chroma) + 4.0;
        }
        h *= 60.0;
        if (h < 0.0) h += 360.0;
        return h;
    }
}
'@

Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$items = @(
    @{ Source = 'paper1_architecture.png'; Target = 'paper1_architecture.png' },
    @{ Source = 'paper1_lag_sequence.png'; Target = 'paper1_lag_sequence.png' }
)

foreach ($item in $items) {
    $source = Join-Path $sourceDir $item.Source
    $target = Join-Path $outDir $item.Target
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Missing Paper 1 source figure: $($item.Source)"
        continue
    }
    Write-Host "Rendering $($item.Target) with Paper 1 feature colours"
    [PaperOneEaaiFeatureRecolor]::ConvertImage($source, $target)
}

$duplicate = Join-Path $outDir 'paper1_mlp_architecture.png'
if (Test-Path -LiteralPath $duplicate) {
    Remove-Item -LiteralPath $duplicate -Force
    Write-Host "Removed duplicate paper1_mlp_architecture.png from eaai_palette."
}
