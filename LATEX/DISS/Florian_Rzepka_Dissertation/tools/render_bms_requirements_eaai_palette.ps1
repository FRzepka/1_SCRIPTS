param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$sourceDir = Join-Path $DissertationRoot 'pictures'
$outDir = Join-Path $sourceDir 'eaai_palette'
$source = Join-Path $sourceDir 'bms_requirements.png'
$target = Join-Path $outDir 'bms_requirements.png'

$converterCode = @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class BmsRequirementsEaaiRecolor
{
    static readonly Color TileFill = Color.FromArgb(0xE8, 0xEE, 0xF2);
    static readonly Color TileEdge = Color.FromArgb(0x60, 0x7F, 0x94);
    static readonly Color CircleFill = Color.FromArgb(0xF2, 0xD0, 0xCC);
    static readonly Color CircleEdge = Color.FromArgb(0xC9, 0x2A, 0x2A);
    static readonly Color WireBlue = Color.FromArgb(0x4F, 0x72, 0x87);
    static readonly Color BmsCircleFill = Color.FromArgb(0xD9, 0xE8, 0xF1);
    static readonly Color BmsCircleEdge = Color.FromArgb(0x1F, 0x77, 0xB4);

    public static void Render(string sourcePath, string targetPath)
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
                        int i = row + x * 4;
                        byte b = bytes[i + 0];
                        byte g = bytes[i + 1];
                        byte r = bytes[i + 2];
                        byte a = bytes[i + 3];
                        if (a == 0) continue;

                        int max = Math.Max(r, Math.Max(g, b));
                        int min = Math.Min(r, Math.Min(g, b));
                        int chroma = max - min;
                        double lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                        if (max < 55 || (min < 35 && chroma < 45))
                        {
                            continue;
                        }

                        Color target;
                        double hue = chroma > 0 ? Hue(r, g, b, max, min) : 0.0;
                        double sat = max == 0 ? 0.0 : chroma / (double)max;

                        if (hue >= 150.0 && hue <= 195.0 && chroma > 35 && lum > 110.0)
                        {
                            target = CircleFill;
                        }
                        else if (hue >= 245.0 && hue <= 292.0 && (chroma > 24 || (b > r + 10 && r > g + 4)))
                        {
                            target = lum > 170.0 ? CircleFill : WireBlue;
                        }
                        else if (hue >= 208.0 && hue <= 230.0 && chroma > 55)
                        {
                            target = TileEdge;
                        }
                        else if (lum > 120.0 && (sat < 0.20 || b > r + 7))
                        {
                            target = Shade(TileFill, lum - 232.0);
                        }
                        else
                        {
                            continue;
                        }

                        bytes[i + 0] = target.B;
                        bytes[i + 1] = target.G;
                        bytes[i + 2] = target.R;
                    }
                }

                Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            DrawCircleOutlines(bmp);
            RecolorCentralBmsCircle(bmp);

            Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
            if (File.Exists(targetPath))
            {
                File.Delete(targetPath);
            }
            bmp.Save(targetPath, ImageFormat.Png);
        }
    }

    static void DrawCircleOutlines(Bitmap bmp)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        bool[] mask = new bool[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Color p = bmp.GetPixel(x, y);
                if (p.A > 0 && ColorDistance(p, CircleFill) < 34.0)
                {
                    mask[y * w + x] = true;
                }
            }
        }

        bool[] visited = new bool[w * h];
        var stack = new Stack<int>(4096);
        using (var g = Graphics.FromImage(bmp))
        using (var pen = new Pen(CircleEdge, 4.0f))
        {
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            for (int i = 0; i < mask.Length; i++)
            {
                if (!mask[i] || visited[i]) continue;
                int minX = w;
                int maxX = 0;
                int minY = h;
                int maxY = 0;
                int count = 0;
                visited[i] = true;
                stack.Push(i);

                while (stack.Count > 0)
                {
                    int q = stack.Pop();
                    count++;
                    int x = q % w;
                    int y = q / w;
                    if (x < minX) minX = x;
                    if (x > maxX) maxX = x;
                    if (y < minY) minY = y;
                    if (y > maxY) maxY = y;

                    Push(q - 1, x > 0, mask, visited, stack);
                    Push(q + 1, x < w - 1, mask, visited, stack);
                    Push(q - w, y > 0, mask, visited, stack);
                    Push(q + w, y < h - 1, mask, visited, stack);
                }

                int bw = maxX - minX + 1;
                int bh = maxY - minY + 1;
                double aspect = bw / Math.Max(1.0, (double)bh);
                if (count > 900 && bw >= 45 && bh >= 45 && bw <= 170 && bh <= 170 && aspect > 0.72 && aspect < 1.28)
                {
                    g.DrawEllipse(pen, minX + 1, minY + 1, bw - 3, bh - 3);
                }
            }
        }
    }

    static void RecolorCentralBmsCircle(Bitmap bmp)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        bool[] mask = new bool[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Color p = bmp.GetPixel(x, y);
                if (p.A > 0 && ColorDistance(p, CircleFill) < 34.0)
                {
                    mask[y * w + x] = true;
                }
            }
        }

        bool[] visited = new bool[w * h];
        var stack = new Stack<int>(4096);
        int bestMinX = 0;
        int bestMaxX = 0;
        int bestMinY = 0;
        int bestMaxY = 0;
        int bestCount = 0;

        for (int i = 0; i < mask.Length; i++)
        {
            if (!mask[i] || visited[i]) continue;
            int minX = w;
            int maxX = 0;
            int minY = h;
            int maxY = 0;
            int count = 0;
            visited[i] = true;
            stack.Push(i);

            while (stack.Count > 0)
            {
                int q = stack.Pop();
                count++;
                int x = q % w;
                int y = q / w;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;

                Push(q - 1, x > 0, mask, visited, stack);
                Push(q + 1, x < w - 1, mask, visited, stack);
                Push(q - w, y > 0, mask, visited, stack);
                Push(q + w, y < h - 1, mask, visited, stack);
            }

            double cx = (minX + maxX) / 2.0;
            double cy = (minY + maxY) / 2.0;
            int bw = maxX - minX + 1;
            int bh = maxY - minY + 1;
            double aspect = bw / Math.Max(1.0, (double)bh);
            bool centralBms = count > 900
                && bw >= 45 && bh >= 45 && bw <= 170 && bh <= 170
                && aspect > 0.72 && aspect < 1.28
                && Math.Abs(cx - w * 0.5) < w * 0.08
                && cy > h * 0.68;

            if (centralBms && count > bestCount)
            {
                bestMinX = minX;
                bestMaxX = maxX;
                bestMinY = minY;
                bestMaxY = maxY;
                bestCount = count;
            }
        }

        if (bestCount == 0) return;

        int x0 = Math.Max(0, bestMinX - 6);
        int x1 = Math.Min(w - 1, bestMaxX + 6);
        int y0 = Math.Max(0, bestMinY - 6);
        int y1 = Math.Min(h - 1, bestMaxY + 6);
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                Color p = bmp.GetPixel(x, y);
                if (p.A == 0) continue;
                if (ColorDistance(p, CircleFill) < 38.0)
                {
                    bmp.SetPixel(x, y, Color.FromArgb(p.A, BmsCircleFill.R, BmsCircleFill.G, BmsCircleFill.B));
                }
                else if (ColorDistance(p, CircleEdge) < 42.0)
                {
                    bmp.SetPixel(x, y, Color.FromArgb(p.A, BmsCircleEdge.R, BmsCircleEdge.G, BmsCircleEdge.B));
                }
            }
        }

        using (var g = Graphics.FromImage(bmp))
        using (var pen = new Pen(BmsCircleEdge, 4.5f))
        {
            g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
            g.DrawEllipse(pen, bestMinX + 1, bestMinY + 1, bestMaxX - bestMinX - 1, bestMaxY - bestMinY - 1);
        }
    }

    static void Push(int idx, bool valid, bool[] mask, bool[] visited, Stack<int> stack)
    {
        if (!valid || visited[idx] || !mask[idx]) return;
        visited[idx] = true;
        stack.Push(idx);
    }

    static Color Shade(Color baseColor, double delta)
    {
        delta = Math.Max(-58.0, Math.Min(24.0, delta));
        return Color.FromArgb(
            ClampToByte(baseColor.R + delta),
            ClampToByte(baseColor.G + delta),
            ClampToByte(baseColor.B + delta)
        );
    }

    static double ColorDistance(Color a, Color b)
    {
        double dr = a.R - b.R;
        double dg = a.G - b.G;
        double db = a.B - b.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
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

    static byte ClampToByte(double value)
    {
        if (value < 0.0) return 0;
        if (value > 255.0) return 255;
        return (byte)Math.Round(value);
    }
}
'@

if (-not (Test-Path -LiteralPath $source)) {
    throw "Missing source image: $source"
}

Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing
New-Item -ItemType Directory -Path $outDir -Force | Out-Null
Write-Host "Rendering bms_requirements.png with focused red / muted blue-gray palette"
[BmsRequirementsEaaiRecolor]::Render($source, $target)
