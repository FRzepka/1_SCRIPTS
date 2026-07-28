param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$sourceDir = Join-Path $DissertationRoot 'pictures'
$outDir = Join-Path $sourceDir 'eaai_palette'
$scriptsRoot = [System.IO.Path]::GetFullPath((Join-Path $DissertationRoot '..\..\..'))
$jesResultsDir = Join-Path $scriptsRoot 'LATEX\JES\paper_robustness_benchmark\figures\Results'

$converterCode = @'
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class JesEaaiPaletteRecolor
{
    struct MapColor
    {
        public Color Source;
        public Color Main;
        public Color Fill;

        public MapColor(Color source, Color main, Color fill)
        {
            Source = source;
            Main = main;
            Fill = fill;
        }
    }

    static readonly MapColor[] Maps = new MapColor[] {
        // Original JES result-figure colours: DM, HDM, HECM, DD.
        new MapColor(Color.FromArgb(0x6e, 0x2f, 0xc4), Color.FromArgb(0x2c, 0xa0, 0x2c), Color.FromArgb(0xa6, 0xd7, 0xa6)),
        new MapColor(Color.FromArgb(0x08, 0xbd, 0xba), Color.FromArgb(0x94, 0x67, 0xbd), Color.FromArgb(0xd2, 0xbf, 0xe3)),
        new MapColor(Color.FromArgb(0xd4, 0xbb, 0xff), Color.FromArgb(0x1f, 0x77, 0xb4), Color.FromArgb(0xa1, 0xc6, 0xe0)),
        new MapColor(Color.FromArgb(0x45, 0x89, 0xff), Color.FromArgb(0xd6, 0x27, 0x28), Color.FromArgb(0xee, 0xa4, 0xa5)),
    };

    public static void ConvertImage(string sourcePath, string targetPath, string mode)
    {
        using (var src = new Bitmap(sourcePath))
        using (var bmp = new Bitmap(src.Width, src.Height, PixelFormat.Format32bppArgb))
        {
            using (var g = Graphics.FromImage(bmp))
            {
                g.DrawImage(src, 0, 0, src.Width, src.Height);
            }

            bool useLightFills = String.Equals(mode, "bar", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed_rightbars", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed_no_outline", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed_signal", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed_spike", StringComparison.OrdinalIgnoreCase);
            bool outlineAllBars = String.Equals(mode, "bar", StringComparison.OrdinalIgnoreCase)
                || String.Equals(mode, "mixed", StringComparison.OrdinalIgnoreCase);
            bool outlineRightPanelBars = String.Equals(mode, "mixed_rightbars", StringComparison.OrdinalIgnoreCase);
            bool outlineSignalBars = String.Equals(mode, "mixed_signal", StringComparison.OrdinalIgnoreCase);
            bool outlineSpikeBars = String.Equals(mode, "mixed_spike", StringComparison.OrdinalIgnoreCase);

            var rect = new Rectangle(0, 0, bmp.Width, bmp.Height);
            var data = bmp.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                int stride = Math.Abs(data.Stride);
                byte[] bytes = new byte[stride * bmp.Height];
                Marshal.Copy(data.Scan0, bytes, 0, bytes.Length);

                int n = bmp.Width * bmp.Height;
                int[] cat = new int[n];
                byte[] strength = new byte[n];
                for (int i = 0; i < n; i++)
                {
                    cat[i] = -1;
                }

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
                        int idx = y * bmp.Width + x;
                        double k;
                        int c = Classify(r, g, b, out k);
                        if (c >= 0)
                        {
                            cat[idx] = c;
                            strength[idx] = (byte)Math.Max(25, Math.Min(255, Math.Round(k * 255.0)));
                        }
                    }
                }

                bool[] visited = new bool[n];
                var stack = new Stack<int>(4096);
                var pixels = new List<int>(4096);

                for (int i = 0; i < n; i++)
                {
                    if (visited[i] || cat[i] < 0) continue;

                    int c = cat[i];
                    pixels.Clear();
                    int minX = bmp.Width;
                    int maxX = 0;
                    int minY = bmp.Height;
                    int maxY = 0;

                    visited[i] = true;
                    stack.Push(i);
                    while (stack.Count > 0)
                    {
                        int q = stack.Pop();
                        pixels.Add(q);
                        int x = q % bmp.Width;
                        int y = q / bmp.Width;
                        if (x < minX) minX = x;
                        if (x > maxX) maxX = x;
                        if (y < minY) minY = y;
                        if (y > maxY) maxY = y;

                        TryPush(q - 1, x > 0, c, cat, visited, stack);
                        TryPush(q + 1, x < bmp.Width - 1, c, cat, visited, stack);
                        TryPush(q - bmp.Width, y > 0, c, cat, visited, stack);
                        TryPush(q + bmp.Width, y < bmp.Height - 1, c, cat, visited, stack);
                    }

                    int w = maxX - minX + 1;
                    int h = maxY - minY + 1;
                    bool area = useLightFills && pixels.Count > 180 && w > 10 && h > 10;
                    if (area && outlineSpikeBars)
                    {
                        double cx = (minX + maxX) / 2.0;
                        double cy = (minY + maxY) / 2.0;
                        area = cx < bmp.Width * 0.50 && cy > bmp.Height * 0.43;
                    }
                    Color target = area ? Maps[c].Fill : Maps[c].Main;

                    foreach (int q in pixels)
                    {
                        int x = q % bmp.Width;
                        int y = q / bmp.Width;
                        int p = y * stride + x * 4;
                        byte k = strength[q];
                        Color bg = EstimateBackground(bytes[p + 2], bytes[p + 1], bytes[p + 0]);
                        Color outColor = Mix(bg, target, k / 255.0);
                        bytes[p + 0] = outColor.B;
                        bytes[p + 1] = outColor.G;
                        bytes[p + 2] = outColor.R;
                    }
                }

                if (useLightFills && !outlineSpikeBars)
                {
                    ConvertWhiteAnnotations(bytes, stride, bmp.Width, bmp.Height, cat);
                }

                Marshal.Copy(bytes, 0, data.Scan0, bytes.Length);
            }
            finally
            {
                bmp.UnlockBits(data);
            }

            if (outlineAllBars || outlineRightPanelBars || outlineSignalBars || outlineSpikeBars)
            {
                using (var g = Graphics.FromImage(bmp))
                {
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                    if (outlineAllBars)
                    {
                        DrawModelOutlines(bmp, g);
                    }
                    else
                    {
                        if (outlineRightPanelBars)
                        {
                            DrawRightPanelBarOutlines(bmp, g);
                        }
                        else if (outlineSignalBars)
                        {
                            DrawProjectedBarOutlines(
                                bmp,
                                g,
                                (int)(bmp.Width * 0.52),
                                bmp.Width - 1,
                                (int)(bmp.Height * 0.06),
                                (int)(bmp.Height * 0.93),
                                55,
                                (int)(bmp.Width * 0.16),
                                90,
                                0.14
                            );
                        }
                        else if (outlineSpikeBars)
                        {
                            DrawProjectedBarOutlines(
                                bmp,
                                g,
                                (int)(bmp.Width * 0.03),
                                (int)(bmp.Width * 0.50),
                                (int)(bmp.Height * 0.43),
                                (int)(bmp.Height * 0.97),
                                18,
                                (int)(bmp.Width * 0.13),
                                22,
                                0.10
                            );
                        }
                    }
                }
            }

            Directory.CreateDirectory(Path.GetDirectoryName(targetPath));
            bmp.Save(targetPath, ImageFormat.Png);
        }
    }

    static void TryPush(int idx, bool valid, int c, int[] cat, bool[] visited, Stack<int> stack)
    {
        if (!valid || visited[idx] || cat[idx] != c) return;
        visited[idx] = true;
        stack.Push(idx);
    }

    static int Classify(byte r, byte g, byte b, out double strength)
    {
        strength = 0.0;
        int max = Math.Max(r, Math.Max(g, b));
        int min = Math.Min(r, Math.Min(g, b));
        int chroma = max - min;
        if (max < 35 || chroma < 24) return -1;
        if (max > 242 && chroma < 18) return -1;

        double bestScore = 1e9;
        int best = -1;
        double bestK = 0.0;
        for (int i = 0; i < Maps.Length; i++)
        {
            double kWhite;
            double scoreWhite = DistanceToBlend(r, g, b, Maps[i].Source, Color.White, out kWhite);
            double kPanel;
            double scorePanel = DistanceToBlend(r, g, b, Maps[i].Source, Color.FromArgb(247, 247, 247), out kPanel);
            double score = scoreWhite;
            double k = kWhite;
            if (scorePanel < score)
            {
                score = scorePanel;
                k = kPanel;
            }
            if (score < bestScore)
            {
                bestScore = score;
                best = i;
                bestK = k;
            }
        }

        if (best >= 0 && bestScore <= 42.0 && bestK >= 0.08)
        {
            strength = Math.Max(0.12, Math.Min(1.0, bestK));
            return best;
        }
        return -1;
    }

    static double DistanceToBlend(byte r, byte g, byte b, Color source, Color bg, out double k)
    {
        double pr = r - bg.R;
        double pg = g - bg.G;
        double pb = b - bg.B;
        double sr = source.R - bg.R;
        double sg = source.G - bg.G;
        double sb = source.B - bg.B;
        double denom = sr * sr + sg * sg + sb * sb;
        if (denom < 1.0)
        {
            k = 0.0;
            return 1e9;
        }
        k = (pr * sr + pg * sg + pb * sb) / denom;
        if (k < 0.0) k = 0.0;
        if (k > 1.0) k = 1.0;
        double rr = bg.R + k * sr;
        double gg = bg.G + k * sg;
        double bb = bg.B + k * sb;
        double dr = r - rr;
        double dg = g - gg;
        double db = b - bb;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }

    static Color EstimateBackground(byte r, byte g, byte b)
    {
        int max = Math.Max(r, Math.Max(g, b));
        if (max < 248) return Color.FromArgb(247, 247, 247);
        return Color.White;
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

    static int Clamp(double v)
    {
        if (v < 0.0) return 0;
        if (v > 255.0) return 255;
        return (int)Math.Round(v);
    }

    static void ConvertWhiteAnnotations(byte[] bytes, int stride, int width, int height, int[] cat)
    {
        int n = width * height;
        bool[] mask = new bool[n];
        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
            {
                int p = row + x * 4;
                byte b = bytes[p + 0];
                byte g = bytes[p + 1];
                byte r = bytes[p + 2];
                int max = Math.Max(r, Math.Max(g, b));
                int min = Math.Min(r, Math.Min(g, b));
                if (max < 210 || max - min > 35) continue;
                if (NearModelColor(cat, width, height, x, y, 3))
                {
                    mask[y * width + x] = true;
                }
            }
        }

        bool[] visited = new bool[n];
        var stack = new Stack<int>(256);
        var pixels = new List<int>(256);
        for (int i = 0; i < n; i++)
        {
            if (!mask[i] || visited[i]) continue;
            pixels.Clear();
            int minX = width;
            int maxX = 0;
            int minY = height;
            int maxY = 0;
            visited[i] = true;
            stack.Push(i);
            while (stack.Count > 0)
            {
                int q = stack.Pop();
                pixels.Add(q);
                int x = q % width;
                int y = q / width;
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                PushMask(q - 1, x > 0, mask, visited, stack);
                PushMask(q + 1, x < width - 1, mask, visited, stack);
                PushMask(q - width, y > 0, mask, visited, stack);
                PushMask(q + width, y < height - 1, mask, visited, stack);
            }

            int w = maxX - minX + 1;
            int h = maxY - minY + 1;
            if (pixels.Count >= 3 && pixels.Count <= 900 && w <= 130 && h <= 28)
            {
                foreach (int q in pixels)
                {
                    int x = q % width;
                    int y = q / width;
                    int p = y * stride + x * 4;
                    bytes[p + 0] = 20;
                    bytes[p + 1] = 20;
                    bytes[p + 2] = 20;
                }
            }
        }
    }

    static bool NearModelColor(int[] cat, int width, int height, int x, int y, int radius)
    {
        int x0 = Math.Max(0, x - radius);
        int x1 = Math.Min(width - 1, x + radius);
        int y0 = Math.Max(0, y - radius);
        int y1 = Math.Min(height - 1, y + radius);
        for (int yy = y0; yy <= y1; yy++)
        {
            int row = yy * width;
            for (int xx = x0; xx <= x1; xx++)
            {
                if (cat[row + xx] >= 0) return true;
            }
        }
        return false;
    }

    static void DrawModelOutlines(Bitmap bmp, Graphics g)
    {
        for (int c = 0; c < Maps.Length; c++)
        {
            using (var pen = new Pen(Maps[c].Main, 3.0f))
            {
                DrawLargeRegionOutlines(bmp, g, Maps[c].Fill, pen);
            }
        }
    }

    static void DrawRightPanelBarOutlines(Bitmap bmp, Graphics g)
    {
        int minX = bmp.Width / 2;
        int maxX = bmp.Width - 1;
        int minY = 0;
        int maxY = bmp.Height - 1;
        for (int c = 0; c < Maps.Length; c++)
        {
            bool[] mask = BuildColorMask(bmp, Maps[c].Fill, 42.0, minX, maxX, minY, maxY);
            using (var pen = new Pen(Maps[c].Main, 3.0f))
            {
                DrawVerticalBarRuns(g, pen, mask, bmp.Width, bmp.Height, minX, maxX, minY, maxY);
            }
        }
    }

    static void DrawProjectedBarOutlines(
        Bitmap bmp,
        Graphics g,
        int minX,
        int maxX,
        int minY,
        int maxY,
        int minColumnPixels,
        int maxBarWidth,
        int minBarHeight,
        double minDensity
    )
    {
        minX = Math.Max(0, minX);
        maxX = Math.Min(bmp.Width - 1, maxX);
        minY = Math.Max(0, minY);
        maxY = Math.Min(bmp.Height - 1, maxY);
        if (minX >= maxX || minY >= maxY) return;

        for (int c = 0; c < Maps.Length; c++)
        {
            bool[] mask = BuildColorMask(bmp, Maps[c].Fill, 42.0, minX, maxX, minY, maxY);
            using (var pen = new Pen(Maps[c].Main, 3.0f))
            {
                DrawProjectedBarRuns(
                    g,
                    pen,
                    mask,
                    bmp.Width,
                    minX,
                    maxX,
                    minY,
                    maxY,
                    minColumnPixels,
                    maxBarWidth,
                    minBarHeight,
                    minDensity
                );
            }
        }
    }

    static void DrawProjectedBarRuns(
        Graphics g,
        Pen pen,
        bool[] mask,
        int width,
        int minX,
        int maxX,
        int minY,
        int maxY,
        int minColumnPixels,
        int maxBarWidth,
        int minBarHeight,
        double minDensity
    )
    {
        int startX = -1;
        int lastActiveX = -1;
        for (int x = minX; x <= maxX; x++)
        {
            int count = 0;
            for (int y = minY; y <= maxY; y++)
            {
                if (mask[y * width + x]) count++;
            }

            if (count >= minColumnPixels)
            {
                if (startX < 0) startX = x;
                lastActiveX = x;
            }
            else if (startX >= 0 && x - lastActiveX > 8)
            {
                DrawProjectedBarRowRuns(g, pen, mask, width, startX, lastActiveX, minY, maxY, maxBarWidth, minBarHeight, minDensity);
                startX = -1;
                lastActiveX = -1;
            }
        }
        if (startX >= 0)
        {
            DrawProjectedBarRowRuns(g, pen, mask, width, startX, lastActiveX, minY, maxY, maxBarWidth, minBarHeight, minDensity);
        }
    }

    static void DrawProjectedBarRowRuns(
        Graphics g,
        Pen pen,
        bool[] mask,
        int width,
        int x0,
        int x1,
        int minY,
        int maxY,
        int maxBarWidth,
        int minBarHeight,
        double minDensity
    )
    {
        int bw = x1 - x0 + 1;
        if (bw < 18 || bw > maxBarWidth) return;

        int rowThreshold = Math.Max(3, Math.Min(bw / 4, bw - 2));
        int startY = -1;
        int lastActiveY = -1;
        for (int y = minY; y <= maxY; y++)
        {
            int count = 0;
            for (int x = x0; x <= x1; x++)
            {
                if (mask[y * width + x]) count++;
            }

            if (count >= rowThreshold)
            {
                if (startY < 0) startY = y;
                lastActiveY = y;
            }
            else if (startY >= 0 && y - lastActiveY > 8)
            {
                DrawProjectedBarIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY, minBarHeight, minDensity);
                startY = -1;
                lastActiveY = -1;
            }
        }
        if (startY >= 0)
        {
            DrawProjectedBarIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY, minBarHeight, minDensity);
        }
    }

    static void DrawProjectedBarIfDense(
        Graphics g,
        Pen pen,
        bool[] mask,
        int width,
        int x0,
        int x1,
        int y0,
        int y1,
        int minBarHeight,
        double minDensity
    )
    {
        int bw = x1 - x0 + 1;
        int bh = y1 - y0 + 1;
        if (bw < 18 || bh < minBarHeight) return;

        int count = 0;
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                if (mask[y * width + x]) count++;
            }
        }

        double density = count / Math.Max(1.0, (double)(bw * bh));
        if (density >= minDensity)
        {
            g.DrawRectangle(pen, x0, y0, bw - 1, bh - 1);
        }
    }

    static bool[] BuildColorMask(Bitmap bmp, Color fill, double threshold, int minX, int maxX, int minY, int maxY)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        bool[] mask = new bool[w * h];
        for (int y = Math.Max(0, minY); y <= Math.Min(h - 1, maxY); y++)
        {
            for (int x = Math.Max(0, minX); x <= Math.Min(w - 1, maxX); x++)
            {
                Color p = bmp.GetPixel(x, y);
                if (p.A > 0 && ColorDistance(p, fill) < threshold)
                {
                    mask[y * w + x] = true;
                }
            }
        }
        return mask;
    }

    static void DrawVerticalBarRuns(Graphics g, Pen pen, bool[] mask, int width, int height, int minX, int maxX, int minY, int maxY)
    {
        int panelHeight = maxY - minY + 1;
        int columnThreshold = Math.Max(8, panelHeight / 35);
        int startX = -1;
        int lastActiveX = -1;
        for (int x = minX; x <= maxX; x++)
        {
            int count = 0;
            for (int y = minY; y <= maxY; y++)
            {
                if (mask[y * width + x]) count++;
            }

            if (count >= columnThreshold)
            {
                if (startX < 0) startX = x;
                lastActiveX = x;
            }
            else if (startX >= 0 && x - lastActiveX > 8)
            {
                DrawVerticalBarRowRuns(g, pen, mask, width, height, startX, lastActiveX, minY, maxY);
                startX = -1;
                lastActiveX = -1;
            }
        }
        if (startX >= 0)
        {
            DrawVerticalBarRowRuns(g, pen, mask, width, height, startX, lastActiveX, minY, maxY);
        }
    }

    static void DrawVerticalBarRowRuns(Graphics g, Pen pen, bool[] mask, int width, int height, int x0, int x1, int minY, int maxY)
    {
        int bw = x1 - x0 + 1;
        if (bw < 24 || bw > width * 0.24) return;

        int rowThreshold = Math.Max(4, bw / 18);
        int startY = -1;
        int lastActiveY = -1;
        for (int y = minY; y <= maxY; y++)
        {
            int count = 0;
            for (int x = x0; x <= x1; x++)
            {
                if (mask[y * width + x]) count++;
            }

            if (count >= rowThreshold)
            {
                if (startY < 0) startY = y;
                lastActiveY = y;
            }
            else if (startY >= 0 && y - lastActiveY > 8)
            {
                DrawVerticalBarSubrectangleIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY);
                startY = -1;
                lastActiveY = -1;
            }
        }
        if (startY >= 0)
        {
            DrawVerticalBarSubrectangleIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY);
        }
    }

    static void DrawVerticalBarSubrectangleIfDense(Graphics g, Pen pen, bool[] mask, int width, int x0, int x1, int y0, int y1)
    {
        int bw = x1 - x0 + 1;
        int bh = y1 - y0 + 1;
        if (bw < 24 || bh < 20) return;

        int count = 0;
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                if (mask[y * width + x]) count++;
            }
        }

        double density = count / Math.Max(1.0, (double)(bw * bh));
        if (count > 500 && density > 0.16)
        {
            g.DrawRectangle(pen, x0, y0, bw - 1, bh - 1);
        }
    }

    static void DrawLargeRegionOutlines(Bitmap bmp, Graphics g, Color fill, Pen pen)
    {
        int w = bmp.Width;
        int h = bmp.Height;
        bool[] mask = new bool[w * h];
        for (int y = 0; y < h; y++)
        {
            for (int x = 0; x < w; x++)
            {
                Color p = bmp.GetPixel(x, y);
                if (p.A > 0 && ColorDistance(p, fill) < 42.0)
                {
                    mask[y * w + x] = true;
                }
            }
        }

        bool[] grown = DilateMask(mask, w, h, 3);
        bool[] visited = new bool[w * h];
        var stack = new Stack<int>(4096);
        for (int i = 0; i < grown.Length; i++)
        {
            if (!grown[i] || visited[i]) continue;
            int minX = w;
            int maxX = 0;
            int minY = h;
            int maxY = 0;
            int originalMinX = w;
            int originalMaxX = 0;
            int originalMinY = h;
            int originalMaxY = 0;
            int grownCount = 0;
            int originalCount = 0;
            visited[i] = true;
            stack.Push(i);
            while (stack.Count > 0)
            {
                int q = stack.Pop();
                grownCount++;
                int x = q % w;
                int y = q / w;
                if (mask[q])
                {
                    originalCount++;
                    if (x < originalMinX) originalMinX = x;
                    if (x > originalMaxX) originalMaxX = x;
                    if (y < originalMinY) originalMinY = y;
                    if (y > originalMaxY) originalMaxY = y;
                }
                if (x < minX) minX = x;
                if (x > maxX) maxX = x;
                if (y < minY) minY = y;
                if (y > maxY) maxY = y;
                PushMask(q - 1, x > 0, grown, visited, stack);
                PushMask(q + 1, x < w - 1, grown, visited, stack);
                PushMask(q - w, y > 0, grown, visited, stack);
                PushMask(q + w, y < h - 1, grown, visited, stack);
            }
            int bw = originalMaxX - originalMinX + 1;
            int bh = originalMaxY - originalMinY + 1;
            double density = originalCount / Math.Max(1.0, (double)(bw * bh));
            bool tooLarge = bw > w * 0.92 && bh > h * 0.92;
            bool rectangleLike = originalCount > 650 && bw > 22 && bh > 22 && density > 0.18 && !tooLarge;
            if (rectangleLike)
            {
                g.DrawRectangle(pen, originalMinX, originalMinY, bw - 1, bh - 1);
            }
            else if (originalCount > 650 && bw > 22 && bh > 22 && !tooLarge)
            {
                DrawDenseSubrectangles(g, pen, mask, w, originalMinX, originalMaxX, originalMinY, originalMaxY);
            }
        }
    }

    static void DrawDenseSubrectangles(Graphics g, Pen pen, bool[] mask, int width, int minX, int maxX, int minY, int maxY)
    {
        int boxWidth = maxX - minX + 1;
        int boxHeight = maxY - minY + 1;
        if (boxWidth <= 0 || boxHeight <= 0) return;

        int columnThreshold = Math.Max(3, boxHeight / 20);
        var xRanges = new List<int[]>();
        int startX = -1;
        int lastActiveX = -1;
        for (int x = minX; x <= maxX; x++)
        {
            int count = 0;
            for (int y = minY; y <= maxY; y++)
            {
                if (mask[y * width + x]) count++;
            }

            if (count >= columnThreshold)
            {
                if (startX < 0) startX = x;
                lastActiveX = x;
            }
            else if (startX >= 0 && x - lastActiveX > 5)
            {
                xRanges.Add(new int[] { startX, lastActiveX });
                startX = -1;
                lastActiveX = -1;
            }
        }
        if (startX >= 0) xRanges.Add(new int[] { startX, lastActiveX });

        foreach (var xRange in xRanges)
        {
            int x0 = xRange[0];
            int x1 = xRange[1];
            int localWidth = x1 - x0 + 1;
            if (localWidth < 22) continue;

            int rowThreshold = Math.Max(3, localWidth / 20);
            int startY = -1;
            int lastActiveY = -1;
            for (int y = minY; y <= maxY; y++)
            {
                int count = 0;
                for (int x = x0; x <= x1; x++)
                {
                    if (mask[y * width + x]) count++;
                }

                if (count >= rowThreshold)
                {
                    if (startY < 0) startY = y;
                    lastActiveY = y;
                }
                else if (startY >= 0 && y - lastActiveY > 5)
                {
                    DrawSubrectangleIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY);
                    startY = -1;
                    lastActiveY = -1;
                }
            }
            if (startY >= 0)
            {
                DrawSubrectangleIfDense(g, pen, mask, width, x0, x1, startY, lastActiveY);
            }
        }
    }

    static void DrawSubrectangleIfDense(Graphics g, Pen pen, bool[] mask, int width, int x0, int x1, int y0, int y1)
    {
        int bw = x1 - x0 + 1;
        int bh = y1 - y0 + 1;
        if (bw < 22 || bh < 22) return;

        int count = 0;
        for (int y = y0; y <= y1; y++)
        {
            for (int x = x0; x <= x1; x++)
            {
                if (mask[y * width + x]) count++;
            }
        }

        double density = count / Math.Max(1.0, (double)(bw * bh));
        if (count > 650 && density > 0.18)
        {
            g.DrawRectangle(pen, x0, y0, bw - 1, bh - 1);
        }
    }

    static bool[] DilateMask(bool[] mask, int width, int height, int radius)
    {
        bool[] grown = new bool[mask.Length];
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                if (!mask[y * width + x]) continue;
                int x0 = Math.Max(0, x - radius);
                int x1 = Math.Min(width - 1, x + radius);
                int y0 = Math.Max(0, y - radius);
                int y1 = Math.Min(height - 1, y + radius);
                for (int yy = y0; yy <= y1; yy++)
                {
                    int row = yy * width;
                    for (int xx = x0; xx <= x1; xx++)
                    {
                        grown[row + xx] = true;
                    }
                }
            }
        }
        return grown;
    }

    static void PushMask(int idx, bool valid, bool[] mask, bool[] visited, Stack<int> stack)
    {
        if (!valid || visited[idx] || !mask[idx]) return;
        visited[idx] = true;
        stack.Push(idx);
    }

    static double ColorDistance(Color a, Color b)
    {
        double dr = a.R - b.R;
        double dg = a.G - b.G;
        double db = a.B - b.B;
        return Math.Sqrt(dr * dr + dg * dg + db * db);
    }
}
'@

Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing

$figureMap = @(
    @{ Source = 'Figure_04_Baseline_Performance.png'; Target = 'robustness_baseline.png'; Mode = 'bar' },
    @{ Source = 'Figure_05_Current_Bias.png'; Target = 'robustness_current_bias.png'; Mode = 'line' },
    @{ Source = 'Figure_06_Noise_Robustness.png'; Target = 'robustness_noise.png'; Mode = 'line' },
    @{ Source = 'Figure_07_Initial_State_Recovery.png'; Target = 'robustness_init_recovery.png'; Mode = 'line' },
    @{ Source = 'Figure_08_Signal_Integrity.png'; Target = 'robustness_signal_integrity.png'; Mode = 'mixed_signal' },
    @{ Source = 'Figure_09_Burst_Dropout_Transition.png'; Target = 'robustness_dropout_transition.png'; Mode = 'line' },
    @{ Source = 'Figure_10_Burst_Dropout_Recovery.png'; Target = 'robustness_dropout_recovery.png'; Mode = 'line' },
    @{ Source = 'Figure_11_Voltage_Spike_Response.png'; Target = 'robustness_spike_response.png'; Mode = 'mixed_spike' },
    @{ Source = 'Figure_14_ADC_Quantization.png'; Target = 'robustness_adc_quantization.png'; Mode = 'mixed_rightbars' }
)

function Wait-ImageFile {
    param([Parameter(Mandatory)][string]$Path)
    $deadline = (Get-Date).AddMinutes(4)
    while ($true) {
        try {
            $stream = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
            $stream.Dispose()
            return
        }
        catch {
            if ((Get-Date) -gt $deadline) {
                throw "Timed out waiting for cloud file: $Path"
            }
            Start-Sleep -Seconds 5
        }
    }
}

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

foreach ($item in $figureMap) {
    $source = Join-Path $jesResultsDir $item.Source
    if (-not (Test-Path -LiteralPath $source)) {
        Write-Warning "Missing JES source figure: $($item.Source)"
        continue
    }
    $target = Join-Path $outDir $item.Target
    Wait-ImageFile -Path $source
    Write-Host "Recoloring $($item.Source) -> $($item.Target)"
    [JesEaaiPaletteRecolor]::ConvertImage($source, $target, $item.Mode)
}

Write-Host "Recolored JES result figures into $outDir"

$decisionRenderer = Join-Path $PSScriptRoot 'render_decision_eaai_palette.ps1'
if (Test-Path -LiteralPath $decisionRenderer) {
    Write-Host "Rendering decision synthesis from source tables."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $decisionRenderer -DissertationRoot $DissertationRoot
}
