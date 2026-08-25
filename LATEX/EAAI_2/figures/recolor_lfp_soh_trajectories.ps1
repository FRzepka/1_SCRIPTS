param(
    [string]$FigureRoot = $PSScriptRoot
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$source = Join-Path $FigureRoot 'source\lfp_soh_trajectories_dissertation_palette.png'
$target = Join-Path $FigureRoot 'lfp_soh_trajectories.png'

$converterCode = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class Eaai2SohPaletteConverter
{
    static readonly Color[] SourceColors = new Color[] {
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

    // Perceptually interpolated path through the four EAAI_2 colors:
    // pink, dark blue, turquoise, and green.
    static readonly Color[] TargetColors = new Color[] {
        Color.FromArgb(0xE7, 0x6B, 0x91),
        Color.FromArgb(0xBF, 0x66, 0x87),
        Color.FromArgb(0x98, 0x60, 0x7D),
        Color.FromArgb(0x70, 0x58, 0x73),
        Color.FromArgb(0x48, 0x4F, 0x69),
        Color.FromArgb(0x2C, 0x50, 0x69),
        Color.FromArgb(0x36, 0x6A, 0x7D),
        Color.FromArgb(0x41, 0x85, 0x91),
        Color.FromArgb(0x4B, 0xA1, 0xA6),
        Color.FromArgb(0x55, 0xBD, 0xBB),
        Color.FromArgb(0x58, 0xCC, 0xB4),
        Color.FromArgb(0x57, 0xD4, 0x9E),
        Color.FromArgb(0x57, 0xDB, 0x85),
        Color.FromArgb(0x57, 0xE2, 0x67),
        Color.FromArgb(0x59, 0xE8, 0x3A)
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
                        double b = pixels[offset];
                        double g = pixels[offset + 1];
                        double r = pixels[offset + 2];
                        double max = Math.Max(r, Math.Max(g, b));
                        double min = Math.Min(r, Math.Min(g, b));
                        if (max - min < 12.0) continue;

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
                            if (denominator < 1e-9) continue;

                            double alpha =
                                ((r - pm) * (sr - sm) +
                                 (g - pm) * (sg - sm) +
                                 (b - pm) * (sb - sm)) / denominator;
                            alpha = Math.Max(0.0, Math.Min(1.0, alpha));
                            if (alpha < 0.08) continue;

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

                        if (best < 0 || bestError > 140.0) continue;
                        Color target = TargetColors[best];
                        pixels[offset] = Clamp(bestAlpha * target.B + (1.0 - bestAlpha) * bestGray);
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

            output.Save(targetPath, ImageFormat.Png);
        }
    }

    static byte Clamp(double value)
    {
        return (byte)Math.Round(Math.Max(0.0, Math.Min(255.0, value)));
    }
}
'@

if (-not ('Eaai2SohPaletteConverter' -as [type])) {
    Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing
}

[Eaai2SohPaletteConverter]::Convert($source, $target)
Write-Host "Updated EAAI_2 SOH trajectory palette: $target"
