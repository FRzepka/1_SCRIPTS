param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$source = Join-Path $DissertationRoot '..\..\EAAI\elsarticle\elsarticle\figures\Combined_Results\Figure_4_SOH_All_Days.png'
$target = Join-Path $DissertationRoot 'pictures\eaai_palette\embedded_soh_all_days.png'

$source = [System.IO.Path]::GetFullPath($source)
$target = [System.IO.Path]::GetFullPath($target)

if (-not (Test-Path -LiteralPath $source)) {
    throw "Source figure not found: $source"
}

$converterCode = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class SohAllDaysFourColorConverter
{
    // Original top-to-bottom colors from the 15 legend entries.
    static readonly Color[] SourceColors = new Color[] {
        Color.FromArgb(128, 219, 121),
        Color.FromArgb(123, 213, 135),
        Color.FromArgb(119, 207, 149),
        Color.FromArgb(115, 201, 163),
        Color.FromArgb(111, 195, 177),
        Color.FromArgb(107, 189, 191),
        Color.FromArgb(102, 183, 205),
        Color.FromArgb(98, 177, 219),
        Color.FromArgb(110, 165, 224),
        Color.FromArgb(121, 152, 229),
        Color.FromArgb(132, 140, 234),
        Color.FromArgb(143, 127, 239),
        Color.FromArgb(155, 115, 244),
        Color.FromArgb(166, 102, 249),
        Color.FromArgb(177, 90, 254)
    };

    // Fifteen equally spaced samples of the light-palette OKLAB path
    // red -> purple -> blue -> green.
    static readonly Color[] TargetColors = new Color[] {
        Color.FromArgb(0xE2, 0x68, 0x68),
        Color.FromArgb(0xD9, 0x74, 0x80),
        Color.FromArgb(0xD0, 0x7E, 0x97),
        Color.FromArgb(0xC6, 0x88, 0xAD),
        Color.FromArgb(0xBC, 0x90, 0xC3),
        Color.FromArgb(0xAF, 0x96, 0xD0),
        Color.FromArgb(0xA0, 0x99, 0xCF),
        Color.FromArgb(0x90, 0x9B, 0xCD),
        Color.FromArgb(0x7E, 0x9E, 0xCC),
        Color.FromArgb(0x6A, 0x9F, 0xCA),
        Color.FromArgb(0x62, 0xA5, 0xBE),
        Color.FromArgb(0x63, 0xAB, 0xAC),
        Color.FromArgb(0x65, 0xB1, 0x98),
        Color.FromArgb(0x68, 0xB7, 0x83),
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

if (-not ('SohAllDaysFourColorConverter' -as [type])) {
    Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing
}

[SohAllDaysFourColorConverter]::Convert($source, $target)
Write-Host "Updated four-color SOH figure: $target"
