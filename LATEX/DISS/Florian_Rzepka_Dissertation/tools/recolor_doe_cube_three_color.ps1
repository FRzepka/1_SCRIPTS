param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$source = Join-Path $DissertationRoot '..\..\EAAI\elsarticle\elsarticle\figures\Schematics\Figure_2_DoE_Cube.png'
$target = Join-Path $DissertationRoot 'pictures\eaai_palette\embedded_doe_cube.png'

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

public static class DoeCubeThreeColorConverter
{
    static readonly Color[] SourceColors = new Color[] {
        Color.FromArgb(128, 0, 128),
        Color.FromArgb(0, 178, 0),
        Color.FromArgb(255, 128, 0)
    };

    // Center red, cube vertices green, and axial outer points blue.
    static readonly Color[] TargetColors = new Color[] {
        Color.FromArgb(226, 104, 104),
        Color.FromArgb(107, 188, 107),
        Color.FromArgb(98, 160, 202)
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

                        int best = -1;
                        double bestError = Double.MaxValue;
                        double bestAlpha = 0.0;

                        for (int k = 0; k < SourceColors.Length; k++)
                        {
                            double dr = SourceColors[k].R - 255.0;
                            double dg = SourceColors[k].G - 255.0;
                            double db = SourceColors[k].B - 255.0;
                            double denominator = dr * dr + dg * dg + db * db;
                            double alpha =
                                ((r - 255.0) * dr +
                                 (g - 255.0) * dg +
                                 (b - 255.0) * db) / denominator;
                            alpha = Math.Max(0.0, Math.Min(1.0, alpha));
                            if (alpha < 0.05)
                            {
                                continue;
                            }

                            double rr = 255.0 + alpha * dr;
                            double gg = 255.0 + alpha * dg;
                            double bb = 255.0 + alpha * db;
                            double error =
                                (r - rr) * (r - rr) +
                                (g - gg) * (g - gg) +
                                (b - bb) * (b - bb);

                            if (error < bestError)
                            {
                                best = k;
                                bestError = error;
                                bestAlpha = alpha;
                            }
                        }

                        if (best < 0 || bestError > 90.0)
                        {
                            continue;
                        }

                        Color target = TargetColors[best];
                        pixels[offset + 0] = Clamp(255.0 + bestAlpha * (target.B - 255.0));
                        pixels[offset + 1] = Clamp(255.0 + bestAlpha * (target.G - 255.0));
                        pixels[offset + 2] = Clamp(255.0 + bestAlpha * (target.R - 255.0));
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

if (-not ('DoeCubeThreeColorConverter' -as [type])) {
    Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing
}

[DoeCubeThreeColorConverter]::Convert($source, $target)
Write-Host "Updated three-color DoE cube: $target"

