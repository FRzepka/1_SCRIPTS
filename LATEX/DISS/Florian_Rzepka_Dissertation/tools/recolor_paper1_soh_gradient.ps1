param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$sourceDir = Join-Path $DissertationRoot 'pictures'
$targetDir = Join-Path $sourceDir 'eaai_palette'

$converterCode = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class PaperOneSohGradientConverter
{
    static readonly Color[] SourceColors = new Color[] {
        Color.FromArgb(0x49, 0xCB, 0x40),
        Color.FromArgb(0x31, 0xAA, 0x8F),
        Color.FromArgb(0x2F, 0x7E, 0xD3),
        Color.FromArgb(0x6F, 0x37, 0xEF),
        Color.FromArgb(0xB0, 0x2D, 0xB4),
        Color.FromArgb(0xEF, 0x5F, 0x25),
        Color.FromArgb(0xE5, 0x43, 0x0D)
    };

    // Seven equally spaced samples of the light-palette OKLAB path
    // red -> purple -> blue -> green.
    static readonly Color[] TargetColors = new Color[] {
        Color.FromArgb(0xE2, 0x68, 0x68),
        Color.FromArgb(0xCD, 0x82, 0x9E),
        Color.FromArgb(0xB4, 0x95, 0xD1),
        Color.FromArgb(0x90, 0x9B, 0xCD),
        Color.FromArgb(0x62, 0xA0, 0xCA),
        Color.FromArgb(0x65, 0xAF, 0x9F),
        Color.FromArgb(0x6B, 0xBC, 0x6B)
    };

    public static void Convert(string sourcePath, string targetPath)
    {
        using (var output = new Bitmap(sourcePath))
        {
            var rect = new Rectangle(0, 0, output.Width, output.Height);
            var data = output.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                int stride = Math.Abs(data.Stride);
                byte[] pixels = new byte[stride * output.Height];
                Marshal.Copy(data.Scan0, pixels, 0, pixels.Length);

                for (int y = 0; y < output.Height; y++)
                {
                    int row = y * stride;
                    for (int x = 0; x < output.Width; x++)
                    {
                        int offset = row + x * 4;
                        double b = pixels[offset + 0];
                        double g = pixels[offset + 1];
                        double r = pixels[offset + 2];

                        double max = Math.Max(r, Math.Max(g, b));
                        double min = Math.Min(r, Math.Min(g, b));
                        if (max - min < 12.0)
                        {
                            continue;
                        }

                        int best = -1;
                        double bestError = Double.MaxValue;
                        double bestAlpha = 0.0;
                        double bestGray = 255.0;

                        for (int k = 0; k < SourceColors.Length; k++)
                        {
                            double sr = SourceColors[k].R;
                            double sg = SourceColors[k].G;
                            double sb = SourceColors[k].B;
                            double sm = (sr + sg + sb) / 3.0;
                            double pm = (r + g + b) / 3.0;
                            double denominator =
                                (sr - sm) * (sr - sm) +
                                (sg - sm) * (sg - sm) +
                                (sb - sm) * (sb - sm);
                            if (denominator < 1e-9)
                            {
                                continue;
                            }

                            double alpha =
                                ((r - pm) * (sr - sm) +
                                 (g - pm) * (sg - sm) +
                                 (b - pm) * (sb - sm)) / denominator;
                            alpha = Math.Max(0.0, Math.Min(1.0, alpha));
                            if (alpha < 0.08)
                            {
                                continue;
                            }

                            double gray = alpha < 0.999
                                ? (pm - alpha * sm) / (1.0 - alpha)
                                : 255.0;
                            gray = Math.Max(0.0, Math.Min(255.0, gray));

                            double rr = alpha * sr + (1.0 - alpha) * gray;
                            double gg = alpha * sg + (1.0 - alpha) * gray;
                            double bb = alpha * sb + (1.0 - alpha) * gray;
                            double error =
                                (r - rr) * (r - rr) +
                                (g - gg) * (g - gg) +
                                (b - bb) * (b - bb);

                            if (error < bestError)
                            {
                                best = k;
                                bestError = error;
                                bestAlpha = alpha;
                                bestGray = gray;
                            }
                        }

                        if (best < 0 || bestError > 140.0)
                        {
                            continue;
                        }

                        Color target = TargetColors[best];
                        pixels[offset + 0] = Clamp(bestAlpha * target.B + (1.0 - bestAlpha) * bestGray);
                        pixels[offset + 1] = Clamp(bestAlpha * target.G + (1.0 - bestAlpha) * bestGray);
                        pixels[offset + 2] = Clamp(bestAlpha * target.R + (1.0 - bestAlpha) * bestGray);
                    }
                }

                Marshal.Copy(pixels, 0, data.Scan0, pixels.Length);
            }
            finally
            {
                output.UnlockBits(data);
            }

            Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
            if (File.Exists(targetPath))
            {
                File.Delete(targetPath);
            }
            output.Save(targetPath, ImageFormat.Png);
        }
    }

    static byte Clamp(double value)
    {
        return (byte)Math.Round(Math.Max(0.0, Math.Min(255.0, value)));
    }
}
'@

if (-not ('PaperOneSohGradientConverter' -as [type])) {
    Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing
}

foreach ($name in @('paper1_soh_cycles.png', 'paper1_soh_time.png')) {
    $source = Join-Path $sourceDir $name
    $target = Join-Path $targetDir $name
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Source figure not found: $source"
    }
    [PaperOneSohGradientConverter]::Convert($source, $target)
    Write-Host "Updated SOH gradient figure: $target"
}

