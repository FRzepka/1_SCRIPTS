param(
    [string]$DissertationRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
)

$ErrorActionPreference = 'Stop'

Add-Type -AssemblyName System.Drawing

$sourceDir = Join-Path $DissertationRoot 'pictures'
$outDir = Join-Path $sourceDir 'eaai_palette'
$scriptsRoot = [System.IO.Path]::GetFullPath((Join-Path $DissertationRoot '..\..\..'))
$eaaiFigureDir = Join-Path $scriptsRoot 'LATEX\EAAI\elsarticle\elsarticle\figures'
$resolvedSource = [System.IO.Path]::GetFullPath($sourceDir)
$resolvedOut = [System.IO.Path]::GetFullPath($outDir)
if (-not $resolvedOut.StartsWith($resolvedSource, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing to write outside pictures directory: $resolvedOut"
}

$converterCode = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Runtime.InteropServices;

public static class EaaiPaletteImageConverter
{
    static readonly Color Green = Color.FromArgb(44, 160, 44);
    static readonly Color Red = Color.FromArgb(214, 39, 40);
    static readonly Color Blue = Color.FromArgb(31, 119, 180);
    static readonly Color Purple = Color.FromArgb(148, 103, 189);
    static readonly Color[] Categorical = new Color[] { Green, Red, Blue, Purple };
    static readonly Color[] Heatmap = new Color[] { Green, Red, Blue };
    static readonly double[] PaletteHues = new double[] { 120.0, 0.0, 206.6, 271.1 };

    public static void ConvertImage(string sourcePath, string targetPath, string mode)
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

                if (String.Equals(mode, "heatmap", StringComparison.OrdinalIgnoreCase))
                {
                    ConvertHeatmap(bytes, stride, bmp.Width, bmp.Height);
                }
                else if (String.Equals(mode, "generic", StringComparison.OrdinalIgnoreCase))
                {
                    ConvertCategoricalSimple(bytes, stride, bmp.Width, bmp.Height, false);
                }
                else if (String.Equals(mode, "line", StringComparison.OrdinalIgnoreCase))
                {
                    ConvertCategoricalSimple(bytes, stride, bmp.Width, bmp.Height, true);
                }
                else
                {
                    bool robustness = String.Equals(mode, "robustness", StringComparison.OrdinalIgnoreCase)
                        || String.Equals(mode, "decision", StringComparison.OrdinalIgnoreCase);
                    ConvertCategorical(bytes, stride, bmp.Width, bmp.Height, robustness);
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

    static void ConvertCategoricalSimple(byte[] bytes, int stride, int width, int height, bool robustness)
    {
        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
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
                double satForGray = max == 0 ? 0.0 : chroma / (double)max;

                bool white = max > 245 && chroma < 12;
                bool black = max < 55 || (min < 35 && chroma < 45);
                bool gray = (chroma < 24 || (chroma < 50 && satForGray < 0.16 && lum > 135.0)) && !white && !black;
                if (white || black) continue;

                if (gray)
                {
                    byte gv = ClampToByte(lum);
                    bytes[i + 0] = gv;
                    bytes[i + 1] = gv;
                    bytes[i + 2] = gv;
                    continue;
                }

                double hue = Hue(r, g, b, max, min);
                int cat = robustness ? RobustnessTargetIndex(hue, max, chroma) : NearestPaletteIndex(hue);
                Color converted = Mix(Categorical[cat], Color.White, robustness ? 0.24 : 0.30);
                bytes[i + 0] = converted.B;
                bytes[i + 1] = converted.G;
                bytes[i + 2] = converted.R;
            }
        }
    }

    static void ConvertCategorical(byte[] bytes, int stride, int width, int height, bool robustness)
    {
        int[] category = new int[width * height];
        double[] luminance = new double[width * height];
        for (int i = 0; i < category.Length; i++)
        {
            category[i] = -1;
        }

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
            {
                int i = row + x * 4;
                int idx = y * width + x;
                byte b = bytes[i + 0];
                byte g = bytes[i + 1];
                byte r = bytes[i + 2];
                byte a = bytes[i + 3];
                if (a == 0) continue;

                int max = Math.Max(r, Math.Max(g, b));
                int min = Math.Min(r, Math.Min(g, b));
                int chroma = max - min;
                double lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;

                bool white = max > 245 && chroma < 12;
                bool black = max < 55 || (min < 35 && chroma < 45);
                double satForGray = max == 0 ? 0.0 : chroma / (double)max;
                bool gray = (chroma < 24 || (chroma < 50 && satForGray < 0.16 && lum > 135.0)) && !white && !black;
                if (white || black) continue;

                if (gray)
                {
                    byte gv = ClampToByte(lum);
                    bytes[i + 0] = gv;
                    bytes[i + 1] = gv;
                    bytes[i + 2] = gv;
                    continue;
                }

                double hue = Hue(r, g, b, max, min);
                category[idx] = robustness ? RobustnessTargetIndex(hue, max, chroma) : NearestPaletteIndex(hue);
                luminance[idx] = lum;
            }
        }

        int[] smoothedCategory = (int[])category.Clone();
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                int cat = category[idx];

                int left = NearestCategory(category, width, height, x, y, -1, 0, 18);
                int right = NearestCategory(category, width, height, x, y, 1, 0, 18);
                int up = NearestCategory(category, width, height, x, y, 0, -1, 18);
                int down = NearestCategory(category, width, height, x, y, 0, 1, 18);

                int replacement = -1;
                if (left >= 0 && left == right)
                {
                    replacement = left;
                }
                else if (up >= 0 && up == down)
                {
                    replacement = up;
                }

                if (replacement >= 0 && replacement != cat)
                {
                    smoothedCategory[idx] = replacement;
                }
            }
        }
        category = smoothedCategory;

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
            {
                int idx = y * width + x;
                int cat = category[idx];
                if (cat < 0) continue;

                bool boundary = false;
                int sameCount = 0;
                for (int yy = Math.Max(0, y - 1); yy <= Math.Min(height - 1, y + 1); yy++)
                {
                    for (int xx = Math.Max(0, x - 1); xx <= Math.Min(width - 1, x + 1); xx++)
                    {
                        if (xx == x && yy == y) continue;
                        if (category[yy * width + xx] == cat) sameCount++;
                    }
                }
                if (x == 0 || y == 0 || x == width - 1 || y == height - 1)
                {
                    boundary = true;
                }
                else if (
                    IsBoundaryNeighbor(category, width, height, x, y, -1, 0, cat) ||
                    IsBoundaryNeighbor(category, width, height, x, y, 1, 0, cat) ||
                    IsBoundaryNeighbor(category, width, height, x, y, 0, -1, cat) ||
                    IsBoundaryNeighbor(category, width, height, x, y, 0, 1, cat))
                {
                    boundary = true;
                }
                else if (sameCount <= 2)
                {
                    boundary = true;
                }
                else if (IsExteriorNear(category, width, height, x, y, cat, 4))
                {
                    boundary = true;
                }

                if (boundary)
                {
                    int leftBridge = NearestCategory(category, width, height, x, y, -1, 0, 30);
                    int rightBridge = NearestCategory(category, width, height, x, y, 1, 0, 30);
                    int upBridge = NearestCategory(category, width, height, x, y, 0, -1, 30);
                    int downBridge = NearestCategory(category, width, height, x, y, 0, 1, 30);
                    bool nearVerticalExterior = HasExteriorToward(category, width, height, x, y, 0, -1, cat, 4)
                        || HasExteriorToward(category, width, height, x, y, 0, 1, cat, 4);
                    bool nearHorizontalExterior = HasExteriorToward(category, width, height, x, y, -1, 0, cat, 4)
                        || HasExteriorToward(category, width, height, x, y, 1, 0, cat, 4);

                    if (leftBridge >= 0 && leftBridge == rightBridge && !nearVerticalExterior)
                    {
                        cat = leftBridge;
                        boundary = false;
                    }
                    else if (upBridge >= 0 && upBridge == downBridge && !nearHorizontalExterior)
                    {
                        cat = upBridge;
                        boundary = false;
                    }
                }

                Color target = Categorical[cat];
                Color converted;
                if (boundary)
                {
                    converted = target;
                }
                else
                {
                    double whiteMix = 0.58;
                    if (luminance[idx] > 215.0) whiteMix = 0.68;
                    if (luminance[idx] < 95.0) whiteMix = 0.48;
                    converted = Mix(target, Color.White, whiteMix);
                }

                int i = row + x * 4;
                bytes[i + 0] = converted.B;
                bytes[i + 1] = converted.G;
                bytes[i + 2] = converted.R;
            }
        }
    }

    static bool IsBoundaryNeighbor(int[] category, int width, int height, int x, int y, int dx, int dy, int cat)
    {
        int nx = x + dx;
        int ny = y + dy;
        if (nx < 0 || ny < 0 || nx >= width || ny >= height) return true;
        int ncat = category[ny * width + nx];
        if (ncat == cat) return false;
        if (ncat >= 0)
        {
            for (int skip = 2; skip <= 5; skip++)
            {
                int sx = x + dx * skip;
                int sy = y + dy * skip;
                if (sx < 0 || sy < 0 || sx >= width || sy >= height) break;
                int scat = category[sy * width + sx];
                if (scat == cat) return false;
            }
            return true;
        }

        // Grid lines are often drawn on top of bars as narrow neutral gaps. If
        // the same category continues behind such a gap, the adjacent coloured
        // pixels are still interior fill, not an outline.
        for (int skip = 2; skip <= 18; skip++)
        {
            int sx = x + dx * skip;
            int sy = y + dy * skip;
            if (sx < 0 || sy < 0 || sx >= width || sy >= height) break;
            int scat = category[sy * width + sx];
            if (scat == cat) return false;
            if (scat >= 0 && scat != cat) return true;
        }
        return true;
    }

    static bool IsExteriorNear(int[] category, int width, int height, int x, int y, int cat, int radius)
    {
        for (int distance = 1; distance <= radius; distance++)
        {
            if (IsExteriorAtOffset(category, width, height, x, y, -1, 0, cat, distance)) return true;
            if (IsExteriorAtOffset(category, width, height, x, y, 1, 0, cat, distance)) return true;
            if (IsExteriorAtOffset(category, width, height, x, y, 0, -1, cat, distance)) return true;
            if (IsExteriorAtOffset(category, width, height, x, y, 0, 1, cat, distance)) return true;
        }
        return false;
    }

    static bool HasExteriorToward(int[] category, int width, int height, int x, int y, int dx, int dy, int cat, int radius)
    {
        for (int distance = 1; distance <= radius; distance++)
        {
            if (IsExteriorAtOffset(category, width, height, x, y, dx, dy, cat, distance)) return true;
        }
        return false;
    }

    static bool IsExteriorAtOffset(int[] category, int width, int height, int x, int y, int dx, int dy, int cat, int distance)
    {
        int sx = x + dx * distance;
        int sy = y + dy * distance;
        if (sx < 0 || sy < 0 || sx >= width || sy >= height) return true;
        int scat = category[sy * width + sx];
        if (scat == cat) return false;

        for (int skip = distance + 1; skip <= distance + 30; skip++)
        {
            int bx = x + dx * skip;
            int by = y + dy * skip;
            if (bx < 0 || by < 0 || bx >= width || by >= height) break;
            int bcat = category[by * width + bx];
            if (bcat == cat) return false;
        }
        return true;
    }

    static int NearestCategory(int[] category, int width, int height, int x, int y, int dx, int dy, int maxSkip)
    {
        for (int skip = 1; skip <= maxSkip; skip++)
        {
            int sx = x + dx * skip;
            int sy = y + dy * skip;
            if (sx < 0 || sy < 0 || sx >= width || sy >= height) return -1;
            int scat = category[sy * width + sx];
            if (scat >= 0) return scat;
        }
        return -1;
    }

    static void ConvertHeatmap(byte[] bytes, int stride, int width, int height)
    {
        double[] lumValues = new double[width * height];
        int count = 0;

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
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
                bool white = max > 252 && chroma < 6;
                bool blackText = max < 55 || (min < 35 && chroma < 45);
                bool darkAntiAlias = chroma < 15 && lum < 175;
                if (!(white || blackText || darkAntiAlias))
                {
                    lumValues[count++] = lum;
                }
            }
        }

        if (count == 0) return;

        Array.Sort(lumValues, 0, count);
        double low = lumValues[Math.Max(0, (int)Math.Floor(count * 0.02))];
        double high = lumValues[Math.Min(count - 1, (int)Math.Floor(count * 0.98))];
        if (high <= low) high = low + 1.0;

        for (int y = 0; y < height; y++)
        {
            int row = y * stride;
            for (int x = 0; x < width; x++)
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
                bool white = max > 252 && chroma < 6;
                bool blackText = max < 55 || (min < 35 && chroma < 45);
                bool darkAntiAlias = chroma < 15 && lum < 175;
                if (white || blackText || darkAntiAlias) continue;

                double t = (lum - low) / (high - low);
                t = Clamp(0.05 + 0.85 * t, 0.0, 0.90);
                Color c = Mix(Interpolate(Heatmap, t), Color.White, 0.38);
                bytes[i + 0] = c.B;
                bytes[i + 1] = c.G;
                bytes[i + 2] = c.R;
            }
        }
    }

    static int RobustnessTargetIndex(double hue, int max, int chroma)
    {
        bool lightPurple = hue >= 245.0 && hue < 285.0 && (max > 210 || chroma < 110);
        bool darkPurple = hue >= 245.0 && hue < 285.0 && !lightPurple;
        bool cyan = hue >= 165.0 && hue < 195.0;
        bool blue = hue >= 195.0 && hue < 245.0;
        bool redOrange = hue < 80.0 || hue >= 330.0;
        bool greenYellow = hue >= 80.0 && hue < 165.0;
        bool otherPurple = hue >= 285.0 && hue < 330.0;

        if (darkPurple || redOrange || otherPurple) return 0;
        if (cyan || greenYellow) return 1;
        if (lightPurple) return 2;
        if (blue) return 3;
        return 1;
    }

    static int NearestPaletteIndex(double hue)
    {
        double best = Double.MaxValue;
        int index = 0;
        for (int i = 0; i < PaletteHues.Length; i++)
        {
            double d = HueDistance(hue, PaletteHues[i]);
            if (d < best)
            {
                best = d;
                index = i;
            }
        }
        return index;
    }

    static Color Interpolate(Color[] colors, double t)
    {
        if (colors.Length == 1) return colors[0];
        double x = Clamp(t, 0.0, 1.0) * (colors.Length - 1);
        int i0 = (int)Math.Floor(x);
        int i1 = Math.Min(i0 + 1, colors.Length - 1);
        double f = x - i0;
        byte r = ClampToByte(colors[i0].R * (1.0 - f) + colors[i1].R * f);
        byte g = ClampToByte(colors[i0].G * (1.0 - f) + colors[i1].G * f);
        byte b = ClampToByte(colors[i0].B * (1.0 - f) + colors[i1].B * f);
        return Color.FromArgb(r, g, b);
    }

    static Color Mix(Color a, Color b, double bAmount)
    {
        bAmount = Clamp(bAmount, 0.0, 1.0);
        double aAmount = 1.0 - bAmount;
        return Color.FromArgb(
            ClampToByte(a.R * aAmount + b.R * bAmount),
            ClampToByte(a.G * aAmount + b.G * bAmount),
            ClampToByte(a.B * aAmount + b.B * bAmount)
        );
    }

    static double HueDistance(double a, double b)
    {
        double d = Math.Abs(a - b);
        return Math.Min(d, 360.0 - d);
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

    static Color FromHsv(double h, double s, double v)
    {
        h = ((h % 360.0) + 360.0) % 360.0;
        s = Clamp(s, 0.0, 1.0);
        v = Clamp(v, 0.0, 1.0);
        double c = v * s;
        double x = c * (1.0 - Math.Abs((h / 60.0) % 2.0 - 1.0));
        double m = v - c;
        double rp = 0.0, gp = 0.0, bp = 0.0;

        if (h < 60.0) { rp = c; gp = x; }
        else if (h < 120.0) { rp = x; gp = c; }
        else if (h < 180.0) { gp = c; bp = x; }
        else if (h < 240.0) { gp = x; bp = c; }
        else if (h < 300.0) { rp = x; bp = c; }
        else { rp = c; bp = x; }

        return Color.FromArgb(
            ClampToByte((rp + m) * 255.0),
            ClampToByte((gp + m) * 255.0),
            ClampToByte((bp + m) * 255.0)
        );
    }

    static double Clamp(double value, double min, double max)
    {
        if (value < min) return min;
        if (value > max) return max;
        return value;
    }

    static byte ClampToByte(double value)
    {
        if (value < 0.0) return 0;
        if (value > 255.0) return 255;
        return (byte)Math.Round(value);
    }
}
'@

Add-Type -TypeDefinition $converterCode -ReferencedAssemblies System.Drawing

$heatmapFiles = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
@(
    'paper1_correlation.png',
    'paper1_mae_matrix.png',
    'robustness_cross_scenario.png'
) | ForEach-Object { [void]$heatmapFiles.Add($_) }

$barStyleFiles = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
@(
    'robustness_adc_quantization.png',
    'robustness_baseline.png',
    'robustness_signal_integrity.png'
) | ForEach-Object { [void]$barStyleFiles.Add($_) }

$lineStyleFiles = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
@(
    'robustness_current_bias.png',
    'robustness_dropout_recovery.png',
    'robustness_dropout_transition.png',
    'robustness_init_recovery.png',
    'robustness_noise.png',
    'robustness_spike_response.png'
) | ForEach-Object { [void]$lineStyleFiles.Add($_) }

$skipFiles = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
@(
    'bms_board_render.png',
    'paper1_mlp_architecture.png'
) | ForEach-Object { [void]$skipFiles.Add($_) }

$aliases = @(
    @{ Source = 'embedded_architecture_rb.png'; Target = 'embedded_architecture.png'; Mode = 'generic' },
    @{ Source = 'robustness_dd_architecture_rb.png'; Target = 'robustness_dd_architecture.png'; Mode = 'generic' },
    @{ Source = 'robustness_decision_synthesis.png'; Target = 'robustness_decision.png'; Mode = 'decision' }
)

$eaaiSources = @{
    'embedded_soc_test_trajectory.png' = 'Combined_Results\Figure_3_SOC_Test_Trajectory.png'
    'embedded_soh_all_days.png' = 'Combined_Results\Figure_4_SOH_All_Days.png'
    'embedded_soc_dashboard.png' = 'Combined_Results\Figure_8_SOC_Streaming_Dashboard.png'
    'embedded_soc_error.png' = 'Combined_Results\Figure_9_SOC_MAE_Hist.png'
    'embedded_soh_error.png' = 'Combined_Results\Figure_10_SOH_MAE_Hist.png'
    'embedded_soh_dashboard.png' = 'Combined_Results\Figure_11_SOH_Streaming_Dashboard.png'
    'embedded_soc_zoom_pulse.png' = 'Combined_Results\Figure_12_SOC_Zoom_Pulse.png'
    'embedded_soc_zoom_checkup.png' = 'Combined_Results\Figure_13_SOC_Zoom_Checkup.png'
    'embedded_model_sizes.png' = 'Combined_Results\Figure_14_Model_Sizes.png'
    'embedded_latency_hist.png' = 'Combined_Results\Figure_15_Latency_Hist.png'
    'embedded_lstm_schematic.png' = 'Schematics\Figure_1_LSTM_MLP_Schematic.png'
    'embedded_doe_cube.png' = 'Schematics\Figure_2_DoE_Cube.png'
    'embedded_pipeline.png' = 'Schematics\Figure_5_Pipeline.png'
    'embedded_pruning_schematic.png' = 'Schematics\Figure_6_Pruning_Schematic.png'
    'embedded_quantization.png' = 'Schematics\Figure_7_Quantization_Schematic.png'
    'embedded_quantization_schematic.png' = 'Schematics\Figure_7_Quantization_Schematic.png'
}

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

function Convert-One {
    param(
        [Parameter(Mandatory)][string]$Source,
        [Parameter(Mandatory)][string]$Target,
        [Parameter(Mandatory)][string]$Mode
    )
    Wait-ImageFile -Path $Source
    [EaaiPaletteImageConverter]::ConvertImage($Source, $Target, $Mode)
}

function Copy-EaaiSourceIfAvailable {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Target
    )
    if (-not $eaaiSources.ContainsKey($Name)) {
        return $false
    }
    $src = Join-Path $eaaiFigureDir $eaaiSources[$Name]
    if (-not (Test-Path -LiteralPath $src)) {
        return $false
    }
    Wait-ImageFile -Path $src
    Copy-Item -LiteralPath $src -Destination $Target -Force
    return $true
}

New-Item -ItemType Directory -Path $outDir -Force | Out-Null

$count = 0
Get-ChildItem -LiteralPath $sourceDir -File -Filter '*.png' | Sort-Object Name | ForEach-Object {
    $name = $_.Name
    if ($skipFiles.Contains($name)) {
        $existing = Join-Path $outDir $name
        if (Test-Path -LiteralPath $existing) {
            Remove-Item -LiteralPath $existing -Force
        }
        return
    }
    if ($_.BaseName -match '_rb$|_original$|_synthesis$') {
        return
    }

    $mode = 'generic'
    if ($heatmapFiles.Contains($name)) {
        $mode = 'heatmap'
    }
    elseif ($barStyleFiles.Contains($name)) {
        $mode = 'robustness'
    }
    elseif ($lineStyleFiles.Contains($name)) {
        $mode = 'line'
    }

    $target = Join-Path $outDir $name
    Write-Host "Processing $name"
    if (-not (Copy-EaaiSourceIfAvailable -Name $name -Target $target)) {
        Convert-One -Source $_.FullName -Target $target -Mode $mode
    }
    $count++
}

foreach ($alias in $aliases) {
    $src = Join-Path $sourceDir $alias.Source
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Warning "Alias source missing: $($alias.Source)"
        continue
    }
    $target = Join-Path $outDir $alias.Target
    Write-Host "Processing $($alias.Target)"
    Convert-One -Source $src -Target $target -Mode $alias.Mode
    $count++
}

Write-Host "Converted $count PNG files into $outDir"
Write-Host "Palette: Base/main green #2CA02C, red #D62728, blue #1F77B4; fourth color purple #9467BD."

$paperOneRenderer = Join-Path $PSScriptRoot 'render_paper1_feature_figures_eaai_palette.ps1'
if (Test-Path -LiteralPath $paperOneRenderer) {
    Write-Host "Refreshing Paper 1 feature figures from original plots."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $paperOneRenderer -DissertationRoot $DissertationRoot
}

$paperOneSohGradientRenderer = Join-Path $PSScriptRoot 'recolor_paper1_soh_gradient.ps1'
if (Test-Path -LiteralPath $paperOneSohGradientRenderer) {
    Write-Host "Refreshing the Paper 1 SOH figures with the four-anchor gradient."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $paperOneSohGradientRenderer -DissertationRoot $DissertationRoot
}

$sohAllDaysRenderer = Join-Path $PSScriptRoot 'recolor_soh_all_days_four_color.ps1'
if (Test-Path -LiteralPath $sohAllDaysRenderer) {
    Write-Host "Refreshing the four-color SOH campaign figure."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $sohAllDaysRenderer -DissertationRoot $DissertationRoot
}

$doeCubeRenderer = Join-Path $PSScriptRoot 'recolor_doe_cube_three_color.ps1'
if (Test-Path -LiteralPath $doeCubeRenderer) {
    Write-Host "Refreshing the three-color DoE cube."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $doeCubeRenderer -DissertationRoot $DissertationRoot
}

$jesRecolorScript = Join-Path $PSScriptRoot 'recolor_jes_results_to_eaai_palette.ps1'
if (Test-Path -LiteralPath $jesRecolorScript) {
    Write-Host "Refreshing JES result figures from original paper plots."
    & $jesRecolorScript -DissertationRoot $DissertationRoot
}

$bmsRequirementsRenderer = Join-Path $PSScriptRoot 'render_bms_requirements_eaai_palette.ps1'
if (Test-Path -LiteralPath $bmsRequirementsRenderer) {
    Write-Host "Refreshing BMS requirements schematic with focused red palette."
    & powershell -NoProfile -ExecutionPolicy Bypass -File $bmsRequirementsRenderer -DissertationRoot $DissertationRoot
}
