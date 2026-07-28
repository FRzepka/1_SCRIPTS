using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;

namespace ReviewOne
{
    public sealed class ErrorStats
    {
        public long Count;
        public double Mae;
        public double Rmse;
        public double Bias;
        public double P95;
        public double Max;
    }

    [StructLayout(LayoutKind.Explicit)]
    internal struct FloatBits
    {
        [FieldOffset(0)] public float Float;
        [FieldOffset(0)] public int Bits;
    }

    public static class AnalysisCore
    {
        private static readonly CultureInfo Inv = CultureInfo.InvariantCulture;

        public static double[] LoadNpzDouble(string archivePath, string entryName)
        {
            using (var archive = ZipFile.OpenRead(archivePath))
            {
                var entry = archive.GetEntry(entryName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase)
                    ? entryName
                    : entryName + ".npy");
                if (entry == null)
                    throw new InvalidDataException("NPZ entry not found: " + entryName);

                using (var stream = entry.Open())
                using (var reader = new BinaryReader(stream, Encoding.ASCII, false))
                {
                    var magic = reader.ReadBytes(6);
                    if (magic.Length != 6 || magic[0] != 0x93 || Encoding.ASCII.GetString(magic, 1, 5) != "NUMPY")
                        throw new InvalidDataException("Invalid NPY header in " + entry.FullName);

                    byte major = reader.ReadByte();
                    reader.ReadByte();
                    int headerLength = major <= 1 ? reader.ReadUInt16() : checked((int)reader.ReadUInt32());
                    string header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));

                    var descrMatch = Regex.Match(header, "['\"]descr['\"]\\s*:\\s*['\"]([^'\"]+)['\"]");
                    var shapeMatch = Regex.Match(header, "['\"]shape['\"]\\s*:\\s*\\(([^\\)]*)\\)");
                    if (!descrMatch.Success || !shapeMatch.Success)
                        throw new InvalidDataException("Unsupported NPY header: " + header);

                    string descr = descrMatch.Groups[1].Value.Trim();
                    long count = 1;
                    foreach (Match m in Regex.Matches(shapeMatch.Groups[1].Value, "\\d+"))
                        count *= long.Parse(m.Value, Inv);
                    if (count > int.MaxValue)
                        throw new InvalidDataException("Array too large for local analysis: " + count);

                    int n = (int)count;
                    if (descr.EndsWith("f8", StringComparison.Ordinal))
                    {
                        byte[] bytes = ReadExact(reader, checked(n * 8));
                        var values = new double[n];
                        Buffer.BlockCopy(bytes, 0, values, 0, bytes.Length);
                        return values;
                    }
                    if (descr.EndsWith("f4", StringComparison.Ordinal))
                    {
                        byte[] bytes = ReadExact(reader, checked(n * 4));
                        var source = new float[n];
                        Buffer.BlockCopy(bytes, 0, source, 0, bytes.Length);
                        var values = new double[n];
                        for (int i = 0; i < n; i++) values[i] = source[i];
                        return values;
                    }
                    throw new InvalidDataException("Only float32/float64 NPY arrays are supported, got " + descr);
                }
            }
        }

        private static byte[] ReadExact(BinaryReader reader, int count)
        {
            byte[] data = new byte[count];
            int offset = 0;
            while (offset < count)
            {
                int read = reader.Read(data, offset, count - offset);
                if (read <= 0) throw new EndOfStreamException();
                offset += read;
            }
            return data;
        }

        public static ErrorStats CalculateStats(double[] reference, double[] prediction)
        {
            int n = Math.Min(reference.Length, prediction.Length);
            return CalculateStats(reference, prediction, 0, n, true);
        }

        private static ErrorStats CalculateStats(double[] reference, double[] prediction, int start, int end, bool percentile)
        {
            int n = Math.Max(0, end - start);
            var result = new ErrorStats { Count = n };
            if (n == 0) return result;

            double sumAbs = 0.0, sumSq = 0.0, sum = 0.0, max = 0.0;
            double[] absErrors = percentile ? new double[n] : null;
            for (int i = start, j = 0; i < end; i++, j++)
            {
                double e = prediction[i] - reference[i];
                double a = Math.Abs(e);
                sumAbs += a;
                sumSq += e * e;
                sum += e;
                if (a > max) max = a;
                if (absErrors != null) absErrors[j] = a;
            }
            result.Mae = sumAbs / n;
            result.Rmse = Math.Sqrt(sumSq / n);
            result.Bias = sum / n;
            result.Max = max;
            result.P95 = absErrors == null ? double.NaN : SelectQuantile(absErrors, 0.95);
            return result;
        }

        private static double SelectQuantile(double[] values, double q)
        {
            if (values.Length == 0) return double.NaN;
            int k = (int)Math.Floor(q * (values.Length - 1));
            int left = 0, right = values.Length - 1;
            while (left < right)
            {
                double pivot = values[(left + right) / 2];
                int i = left, j = right;
                while (i <= j)
                {
                    while (values[i] < pivot) i++;
                    while (values[j] > pivot) j--;
                    if (i <= j)
                    {
                        double tmp = values[i]; values[i] = values[j]; values[j] = tmp;
                        i++; j--;
                    }
                }
                if (k <= j) right = j;
                else if (k >= i) left = i;
                else break;
            }
            return values[k];
        }

        public static void WriteDownsampledSohFilteredTrajectory(string npzPath, string outputPath, int pointCount)
        {
            double[] reference = LoadNpzDouble(npzPath, "y_gt");
            int n = reference.Length;
            int points = Math.Max(2, Math.Min(pointCount, n));
            int[] indices = new int[points];
            for (int i = 0; i < points; i++)
                indices[i] = (int)Math.Round(i * (n - 1.0) / (points - 1.0));

            double[] referenceSample = SampleEntry(reference, indices);
            reference = null;
            double[] baseSample = LoadAndSampleEntry(npzPath, "C_Base", indices, n);
            double[] prunedSample = LoadAndSampleEntry(npzPath, "C_Pruned", indices, n);
            double[] quantizedSample = LoadAndSampleEntry(npzPath, "C_Quant", indices, n);

            var csv = new StringBuilder();
            csv.AppendLine("ProgressPercent,Reference,BaseFiltered,PrunedFiltered,QuantizedFiltered");
            for (int i = 0; i < points; i++)
            {
                csv.Append((100.0 * indices[i] / (n - 1.0)).ToString("G17", Inv)).Append(',')
                    .Append(referenceSample[i].ToString("G17", Inv)).Append(',')
                    .Append(baseSample[i].ToString("G17", Inv)).Append(',')
                    .Append(prunedSample[i].ToString("G17", Inv)).Append(',')
                    .Append(quantizedSample[i].ToString("G17", Inv)).AppendLine();
            }
            Directory.CreateDirectory(Path.GetDirectoryName(outputPath));
            File.WriteAllText(outputPath, csv.ToString(), new UTF8Encoding(false));
        }

        public static void RunLocalWindowsFilterAnalysis(string basePrunedNpz, string quantizedNpz, string outputRoot)
        {
            string outDir = Path.Combine(outputRoot, "filter");
            Directory.CreateDirectory(outDir);
            double[] reference = LoadNpzDouble(basePrunedNpz, "y_true");
            double[] quantizedReference = LoadNpzDouble(quantizedNpz, "y_true");
            int n = Math.Min(reference.Length, quantizedReference.Length);
            ErrorStats targetConsistency = CalculateStats(reference, quantizedReference, 0, n, true);
            using (var consistency = NewWriter(Path.Combine(outDir, "soh_filter_local_target_consistency.csv")))
            {
                consistency.WriteLine("Count,MAE,MaxAbs");
                consistency.WriteLine(string.Join(",", n.ToString(Inv), F(targetConsistency.Mae), F(targetConsistency.Max)));
            }
            quantizedReference = null;
            GC.Collect();

            int samplePoints = 2500;
            double[] referenceSmall = Downsample(reference, samplePoints);
            var small = new Dictionary<string, double[]>();
            var modelSources = new[]
            {
                Tuple.Create("Base", basePrunedNpz, "y_base"),
                Tuple.Create("Pruned", basePrunedNpz, "y_pruned"),
                Tuple.Create("Quantized", quantizedNpz, "y_quant")
            };

            using (var metrics = NewWriter(Path.Combine(outDir, "soh_filter_compression_local_windows.csv")))
            {
                metrics.WriteLine("Model,Filter,MAE_pct,RMSE_pct,Bias_pp,P95_pct,Max_pct,LimiterActivationFraction,MeanAbsPostprocessChange_pp,P95PostprocessChange_pp");
                foreach (var source in modelSources)
                {
                    double[] raw = LoadNpzDouble(source.Item2, source.Item3);
                    int count = Math.Min(reference.Length, raw.Length);
                    double factor = raw[0] == 0.0 ? 1.0 : reference[0] / raw[0];
                    for (int i = 0; i < count; i++) raw[i] *= factor;

                    double benchmarkActivation, manuscriptActivation, sequentialActivation;
                    double[] benchmark = ApplyBenchmarkFilterWithDiagnostics(raw, 1e-4, 1e-5, 0.02, out benchmarkActivation);
                    double[] manuscript = ApplyManuscriptFilterWithDiagnostics(raw, 1e-6, 2e-8, out manuscriptActivation);
                    double[] sequential = ApplyManuscriptFilterWithDiagnostics(benchmark, 1e-6, 2e-8, out sequentialActivation);
                    WriteLocalFilterMetric(metrics, source.Item1, "Raw_first-point-scaled", reference, raw, raw, 0.0);
                    WriteLocalFilterMetric(metrics, source.Item1, "BenchmarkCode_alpha0.02_symmetric-cap", reference, raw, benchmark, benchmarkActivation);
                    WriteLocalFilterMetric(metrics, source.Item1, "ManuscriptText_alpha1e-6_downward-cap", reference, raw, manuscript, manuscriptActivation);
                    WriteLocalFilterMetric(metrics, source.Item1, "FinalSequential_alpha0.02_then_alpha1e-6", reference, raw, sequential, sequentialActivation);

                    small[source.Item1 + "Raw"] = Downsample(raw, samplePoints);
                    small[source.Item1 + "Benchmark"] = Downsample(benchmark, samplePoints);
                    small[source.Item1 + "Manuscript"] = Downsample(manuscript, samplePoints);
                    small[source.Item1 + "Sequential"] = Downsample(sequential, samplePoints);
                    raw = null; benchmark = null; manuscript = null; sequential = null; GC.Collect();
                }
            }

            using (var trajectory = NewWriter(Path.Combine(outDir, "soh_filter_compression_local_trajectory.csv")))
            {
                trajectory.WriteLine("ProgressPercent,Reference,BaseRaw,BaseBenchmark,BaseManuscript,BaseSequential,PrunedRaw,PrunedBenchmark,PrunedManuscript,PrunedSequential,QuantizedRaw,QuantizedBenchmark,QuantizedManuscript,QuantizedSequential");
                for (int i = 0; i < samplePoints; i++)
                {
                    trajectory.WriteLine(string.Join(",", F(100.0*i/(samplePoints-1)), F(referenceSmall[i]),
                        F(small["BaseRaw"][i]), F(small["BaseBenchmark"][i]), F(small["BaseManuscript"][i]), F(small["BaseSequential"][i]),
                        F(small["PrunedRaw"][i]), F(small["PrunedBenchmark"][i]), F(small["PrunedManuscript"][i]), F(small["PrunedSequential"][i]),
                        F(small["QuantizedRaw"][i]), F(small["QuantizedBenchmark"][i]), F(small["QuantizedManuscript"][i]), F(small["QuantizedSequential"][i])));
                }
            }
        }

        private static void WriteLocalFilterMetric(StreamWriter writer, string model, string filter,
            double[] reference, double[] raw, double[] processed, double activationFraction)
        {
            ErrorStats targetStats = CalculateStats(reference, processed);
            ErrorStats processChange = CalculateStats(raw, processed);
            writer.WriteLine(string.Join(",", model, filter, F(100*targetStats.Mae), F(100*targetStats.Rmse),
                F(100*targetStats.Bias), F(100*targetStats.P95), F(100*targetStats.Max), F(activationFraction),
                F(100*processChange.Mae), F(100*processChange.P95)));
        }

        private static double[] LoadAndSampleEntry(string npzPath, string entry, int[] indices, int expectedLength)
        {
            double[] values = LoadNpzDouble(npzPath, entry);
            if (values.Length < expectedLength)
                throw new InvalidDataException(entry + " is shorter than y_gt.");
            return SampleEntry(values, indices);
        }

        private static double[] SampleEntry(double[] values, int[] indices)
        {
            var sample = new double[indices.Length];
            for (int i = 0; i < indices.Length; i++) sample[i] = values[indices[i]];
            return sample;
        }

        public static void RunSequenceAnalyses(string socNpz, string sohNpz, string rawSohNpz, string outputRoot)
        {
            Directory.CreateDirectory(Path.Combine(outputRoot, "long_horizon"));
            Directory.CreateDirectory(Path.Combine(outputRoot, "filter"));
            Directory.CreateDirectory(Path.Combine(outputRoot, "faults"));

            RunTaskSequence(
                "SOC", socNpz, "y",
                new Dictionary<string, string> { { "Base", "base" }, { "Pruned", "pruned" }, { "Quantized", "quant" } },
                outputRoot);

            RunTaskSequence(
                "SOH", sohNpz, "y_gt",
                new Dictionary<string, string> { { "Base", "C_Base" }, { "Pruned", "C_Pruned" }, { "Quantized", "C_Quant" } },
                outputRoot);

            RunFilterAnalysis(rawSohNpz, outputRoot);
            WriteFilterTheory(Path.Combine(outputRoot, "filter"));
        }

        private static void RunTaskSequence(string task, string npz, string referenceEntry,
            Dictionary<string, string> modelEntries, string outputRoot)
        {
            double[] reference = LoadNpzDouble(npz, referenceEntry);
            string longDir = Path.Combine(outputRoot, "long_horizon");
            string faultDir = Path.Combine(outputRoot, "faults");
            string taskLower = task.ToLowerInvariant();

            string windowCsv = Path.Combine(longDir, taskLower + "_windowed_stability.csv");
            string cumulativeCsv = Path.Combine(longDir, taskLower + "_cumulative_mae.csv");
            string deviationCsv = Path.Combine(longDir, taskLower + "_compression_deviation.csv");
            string missingCsv = Path.Combine(faultDir, taskLower + "_missing_update_robustness.csv");
            string bitflipCsv = Path.Combine(faultDir, taskLower + "_output_bitflip_robustness.csv");

            using (var windows = NewWriter(windowCsv))
            using (var cumulative = NewWriter(cumulativeCsv))
            using (var deviations = NewWriter(deviationCsv))
            using (var missing = NewWriter(missingCsv))
            using (var bitflips = NewWriter(bitflipCsv))
            {
                windows.WriteLine("Task,Model,Window,StartFraction,EndFraction,Count,MAE_pct,RMSE_pct,Bias_pp,P95_pct,Max_pct");
                cumulative.WriteLine("Task,Model,ProgressPercent,CumulativeMAE_pct");
                deviations.WriteLine("Task,Variant,Window,StartFraction,EndFraction,MeanAbsDeviation_pp,RMSDeviation_pp,BiasDeviation_pp,P95Deviation_pp,MaxDeviation_pp");
                missing.WriteLine("Task,Model,Scenario,DroppedFraction,MAE_pct,DeltaMAE_pp,RMSE_pct,MaxError_pct");
                bitflips.WriteLine("Task,Model,BitClass,Trials,NonFiniteRate,OutOfRangeRate,CatastrophicRate_gt10pp,P95CorruptedError_pct,P95MitigatedError_pct,MedianErrorIncrease_pp");

            if (!modelEntries.ContainsKey("Base"))
                throw new InvalidOperationException("A Base entry is required for sequence comparisons.");

            var orderedModels = new List<KeyValuePair<string, string>>
            {
                new KeyValuePair<string, string>("Base", modelEntries["Base"])
            };
            foreach (var pair in modelEntries)
            {
                if (!string.Equals(pair.Key, "Base", StringComparison.OrdinalIgnoreCase))
                    orderedModels.Add(pair);
            }

            double[] basePrediction = null;
            foreach (var pair in orderedModels)
                {
                    double[] prediction = LoadNpzDouble(npz, pair.Value);
                    WriteWindows(windows, task, pair.Key, reference, prediction, 10);
                    WriteCumulative(cumulative, task, pair.Key, reference, prediction, 100);
                    WriteMissingUpdateScenarios(missing, task, pair.Key, reference, prediction);
                    WriteBitflipScenarios(bitflips, task, pair.Key, reference, prediction, 20000, 1701 + pair.Key.Length + task.Length);

                    if (pair.Key == "Base")
                    {
                        basePrediction = prediction;
                    }
                    else
                    {
                        WriteDeviationWindows(deviations, task, pair.Key, basePrediction, prediction, 10);
                        prediction = null;
                    }
                    GC.Collect();
                }
                basePrediction = null;
            }
            reference = null;
            GC.Collect();
        }

        private static void WriteWindows(StreamWriter writer, string task, string model,
            double[] reference, double[] prediction, int windows)
        {
            int n = Math.Min(reference.Length, prediction.Length);
            for (int w = 0; w < windows; w++)
            {
                int start = (int)((long)n * w / windows);
                int end = (int)((long)n * (w + 1) / windows);
                var s = CalculateStats(reference, prediction, start, end, true);
                writer.WriteLine(string.Join(",", new[] {
                    task, model, (w + 1).ToString(Inv), F((double)w / windows), F((double)(w + 1) / windows),
                    s.Count.ToString(Inv), F(100*s.Mae), F(100*s.Rmse), F(100*s.Bias), F(100*s.P95), F(100*s.Max)
                }));
            }
        }

        private static void WriteCumulative(StreamWriter writer, string task, string model,
            double[] reference, double[] prediction, int points)
        {
            int n = Math.Min(reference.Length, prediction.Length);
            double running = 0.0;
            int nextPoint = 1;
            for (int i = 0; i < n; i++)
            {
                running += Math.Abs(prediction[i] - reference[i]);
                while (nextPoint <= points && i + 1 >= (long)n * nextPoint / points)
                {
                    writer.WriteLine(string.Join(",", task, model, F(100.0 * nextPoint / points), F(100.0 * running / (i + 1))));
                    nextPoint++;
                }
            }
        }

        private static void WriteDeviationWindows(StreamWriter writer, string task, string variant,
            double[] baseline, double[] prediction, int windows)
        {
            int n = Math.Min(baseline.Length, prediction.Length);
            for (int w = 0; w < windows; w++)
            {
                int start = (int)((long)n * w / windows);
                int end = (int)((long)n * (w + 1) / windows);
                var s = CalculateStats(baseline, prediction, start, end, true);
                writer.WriteLine(string.Join(",", new[] {
                    task, variant, (w + 1).ToString(Inv), F((double)w / windows), F((double)(w + 1) / windows),
                    F(100*s.Mae), F(100*s.Rmse), F(100*s.Bias), F(100*s.P95), F(100*s.Max)
                }));
            }
        }

        private static void WriteMissingUpdateScenarios(StreamWriter writer, string task, string model,
            double[] reference, double[] prediction)
        {
            var baseline = CalculateStats(reference, prediction, 0, Math.Min(reference.Length, prediction.Length), false);
            WriteMissingRow(writer, task, model, "Original", 0.0, baseline, baseline.Mae);

            foreach (int percent in new[] { 1, 5, 10 })
            {
                var stats = CalculateRandomDropStats(reference, prediction, percent / 100.0, 9100 + percent + model.Length);
                WriteMissingRow(writer, task, model, "RandomDrop" + percent + "pct_HoldLast", percent / 100.0, stats, baseline.Mae);
            }
            foreach (int gap in new[] { 60, 600, 3600 })
            {
                double dropped;
                var stats = CalculateBurstDropStats(reference, prediction, gap, 10, out dropped);
                WriteMissingRow(writer, task, model, "TenGaps" + gap + "Samples_HoldLast", dropped, stats, baseline.Mae);
            }
        }

        private static void WriteMissingRow(StreamWriter writer, string task, string model, string scenario,
            double droppedFraction, ErrorStats stats, double baselineMae)
        {
            writer.WriteLine(string.Join(",", task, model, scenario, F(droppedFraction), F(100*stats.Mae),
                F(100*(stats.Mae-baselineMae)), F(100*stats.Rmse), F(100*stats.Max)));
        }

        private static ErrorStats CalculateRandomDropStats(double[] reference, double[] prediction, double dropRate, int seed)
        {
            int n = Math.Min(reference.Length, prediction.Length);
            var rng = new Random(seed);
            double held = prediction[0], sumAbs = 0, sumSq = 0, max = 0;
            for (int i = 0; i < n; i++)
            {
                if (i == 0 || rng.NextDouble() >= dropRate) held = prediction[i];
                double e = held - reference[i], a = Math.Abs(e);
                sumAbs += a; sumSq += e*e; if (a > max) max = a;
            }
            return new ErrorStats { Count=n, Mae=sumAbs/n, Rmse=Math.Sqrt(sumSq/n), Max=max, P95=double.NaN };
        }

        private static ErrorStats CalculateBurstDropStats(double[] reference, double[] prediction, int gapLength,
            int gaps, out double droppedFraction)
        {
            int n = Math.Min(reference.Length, prediction.Length);
            var starts = new int[gaps];
            for (int g = 0; g < gaps; g++)
                starts[g] = Math.Max(1, (int)((long)n * (g + 1) / (gaps + 1)) - gapLength / 2);

            int currentGap = 0, dropped = 0;
            double held = prediction[0], sumAbs = 0, sumSq = 0, max = 0;
            for (int i = 0; i < n; i++)
            {
                while (currentGap < gaps && i >= starts[currentGap] + gapLength) currentGap++;
                bool drop = currentGap < gaps && i >= starts[currentGap] && i < starts[currentGap] + gapLength;
                if (drop) dropped++; else held = prediction[i];
                double e = held - reference[i], a = Math.Abs(e);
                sumAbs += a; sumSq += e*e; if (a > max) max = a;
            }
            droppedFraction = (double)dropped / n;
            return new ErrorStats { Count=n, Mae=sumAbs/n, Rmse=Math.Sqrt(sumSq/n), Max=max, P95=double.NaN };
        }

        private static void WriteBitflipScenarios(StreamWriter writer, string task, string model,
            double[] reference, double[] prediction, int trials, int seed)
        {
            foreach (string bitClass in new[] { "Mantissa", "Exponent", "Sign", "AnyBit" })
            {
                var rng = new Random(seed + bitClass.Length * 31);
                var corruptedErrors = new double[trials];
                var mitigatedErrors = new double[trials];
                var increases = new double[trials];
                int nonFinite = 0, outOfRange = 0, catastrophic = 0;

                for (int t = 0; t < trials; t++)
                {
                    int i = 1 + rng.Next(Math.Max(1, Math.Min(reference.Length, prediction.Length) - 1));
                    int bit;
                    if (bitClass == "Mantissa") bit = rng.Next(0, 23);
                    else if (bitClass == "Exponent") bit = rng.Next(23, 31);
                    else if (bitClass == "Sign") bit = 31;
                    else bit = rng.Next(0, 32);

                    var fb = new FloatBits { Float = (float)prediction[i] };
                    fb.Bits ^= (1 << bit);
                    double corrupted = fb.Float;
                    double originalError = Math.Abs(prediction[i] - reference[i]);
                    bool finite = !(double.IsNaN(corrupted) || double.IsInfinity(corrupted));
                    bool range = finite && corrupted >= 0.0 && corrupted <= 1.0;
                    if (!finite) nonFinite++;
                    if (!range) outOfRange++;

                    double corruptedError = finite ? Math.Abs(corrupted - reference[i]) : 10.0;
                    double mitigated = range ? corrupted : prediction[i - 1];
                    double mitigatedError = Math.Abs(mitigated - reference[i]);
                    if (corruptedError > 0.10) catastrophic++;
                    corruptedErrors[t] = corruptedError;
                    mitigatedErrors[t] = mitigatedError;
                    increases[t] = corruptedError - originalError;
                }

                Array.Sort(corruptedErrors);
                Array.Sort(mitigatedErrors);
                Array.Sort(increases);
                writer.WriteLine(string.Join(",", task, model, bitClass, trials.ToString(Inv),
                    F((double)nonFinite/trials), F((double)outOfRange/trials), F((double)catastrophic/trials),
                    F(100*AtQuantile(corruptedErrors,0.95)), F(100*AtQuantile(mitigatedErrors,0.95)),
                    F(100*AtQuantile(increases,0.50))));
            }
        }

        private static double AtQuantile(double[] sorted, double q)
        {
            if (sorted.Length == 0) return double.NaN;
            return sorted[(int)Math.Floor(q * (sorted.Length - 1))];
        }

        private static void RunFilterAnalysis(string rawSohNpz, string outputRoot)
        {
            string outDir = Path.Combine(outputRoot, "filter");
            double[] reference = LoadNpzDouble(rawSohNpz, "y_true");
            int samplePoints = 2500;
            var progress = new double[samplePoints];
            var referenceSmall = Downsample(reference, samplePoints);
            for (int i=0; i<samplePoints; i++) progress[i] = 100.0*i/(samplePoints-1);

            var small = new Dictionary<string,double[]>();
            using (var metrics = NewWriter(Path.Combine(outDir, "soh_filter_comparison.csv")))
            {
                metrics.WriteLine("Model,Filter,MAE_pct,RMSE_pct,Bias_pp,P95_pct,Max_pct");
                foreach (var pair in new[] { Tuple.Create("Base","y_base"), Tuple.Create("Pruned","y_pruned") })
                {
                    double[] raw = LoadNpzDouble(rawSohNpz, pair.Item2);
                    int n = Math.Min(reference.Length, raw.Length);
                    double factor = raw[0] == 0.0 ? 1.0 : reference[0] / raw[0];
                    for (int i=0; i<n; i++) raw[i] *= factor;

                    double[] benchmark = ApplyBenchmarkFilter(raw, 1e-4, 1e-5, 0.02);
                    double[] manuscript = ApplyManuscriptFilter(raw, 1e-6, 2e-8);
                    WriteFilterMetric(metrics, pair.Item1, "Raw_first-point-scaled", CalculateStats(reference,raw));
                    WriteFilterMetric(metrics, pair.Item1, "BenchmarkCode_alpha0.02_symmetric-cap", CalculateStats(reference,benchmark));
                    WriteFilterMetric(metrics, pair.Item1, "ManuscriptText_alpha1e-6_downward-cap", CalculateStats(reference,manuscript));

                    small[pair.Item1+"Raw"] = Downsample(raw,samplePoints);
                    small[pair.Item1+"Benchmark"] = Downsample(benchmark,samplePoints);
                    small[pair.Item1+"Manuscript"] = Downsample(manuscript,samplePoints);
                    raw=null; benchmark=null; manuscript=null; GC.Collect();
                }
            }

            using (var trajectory = NewWriter(Path.Combine(outDir, "soh_filter_trajectory_downsampled.csv")))
            {
                trajectory.WriteLine("ProgressPercent,Reference,BaseRaw,BaseBenchmark,BaseManuscript,PrunedRaw,PrunedBenchmark,PrunedManuscript");
                for (int i=0;i<samplePoints;i++)
                {
                    trajectory.WriteLine(string.Join(",", F(progress[i]), F(referenceSmall[i]),
                        F(small["BaseRaw"][i]), F(small["BaseBenchmark"][i]), F(small["BaseManuscript"][i]),
                        F(small["PrunedRaw"][i]), F(small["PrunedBenchmark"][i]), F(small["PrunedManuscript"][i])));
                }
            }
            reference=null; GC.Collect();
        }

        private static void WriteFilterMetric(StreamWriter writer, string model, string filter, ErrorStats s)
        {
            writer.WriteLine(string.Join(",", model, filter, F(100*s.Mae), F(100*s.Rmse), F(100*s.Bias), F(100*s.P95), F(100*s.Max)));
        }

        private static double[] ApplyBenchmarkFilter(double[] input, double relCap, double absCap, double alpha)
        {
            var output = new double[input.Length];
            double last=input[0], ema=input[0]; output[0]=last;
            for(int i=1;i<input.Length;i++)
            {
                double v=input[i], cap=Math.Min(Math.Abs(last)*relCap,absCap), delta=v-last;
                if(Math.Abs(delta)>cap) v=last+(delta>0?cap:-cap);
                ema=alpha*v+(1-alpha)*ema; v=ema; output[i]=v; last=v;
            }
            return output;
        }

        private static double[] ApplyBenchmarkFilterWithDiagnostics(double[] input, double relCap, double absCap,
            double alpha, out double activationFraction)
        {
            var output = new double[input.Length];
            double last=input[0], ema=input[0]; output[0]=last;
            long activated=0;
            for(int i=1;i<input.Length;i++)
            {
                double v=input[i], cap=Math.Min(Math.Abs(last)*relCap,absCap), delta=v-last;
                if(Math.Abs(delta)>cap) { v=last+(delta>0?cap:-cap); activated++; }
                ema=alpha*v+(1-alpha)*ema; v=ema; output[i]=v; last=v;
            }
            activationFraction=input.Length <= 1 ? 0.0 : (double)activated/(input.Length-1);
            return output;
        }

        private static double[] ApplyManuscriptFilter(double[] input, double alpha, double maxDrop)
        {
            var output=new double[input.Length];
            double ema=1.0,last=1.0;
            for(int i=0;i<input.Length;i++)
            {
                ema=alpha*input[i]+(1-alpha)*ema;
                double v=ema;
                if(v<last-maxDrop) v=last-maxDrop;
                output[i]=v; last=v;
            }
            return output;
        }

        private static double[] ApplyManuscriptFilterWithDiagnostics(double[] input, double alpha, double maxDrop,
            out double activationFraction)
        {
            var output=new double[input.Length];
            double ema=1.0,last=1.0;
            long activated=0;
            for(int i=0;i<input.Length;i++)
            {
                ema=alpha*input[i]+(1-alpha)*ema;
                double v=ema;
                if(v<last-maxDrop) { v=last-maxDrop; activated++; }
                output[i]=v; last=v;
            }
            activationFraction=input.Length == 0 ? 0.0 : (double)activated/input.Length;
            return output;
        }

        private static double[] Downsample(double[] values, int count)
        {
            var result=new double[count];
            for(int i=0;i<count;i++)
            {
                int idx=(int)((long)(values.Length-1)*i/(count-1));
                result[i]=values[idx];
            }
            return result;
        }

        private static void WriteFilterTheory(string outDir)
        {
            using(var writer=NewWriter(Path.Combine(outDir,"soh_filter_characteristics.csv")))
            {
                writer.WriteLine("Definition,Alpha,SamplingHz,TauSamples,TauDays,T50Days,T90Days,CutoffHz");
                WriteFilterCharacteristic(writer,"BenchmarkCode",0.02,1.0);
                WriteFilterCharacteristic(writer,"ManuscriptText",1e-6,1.0);
            }
            using(var writer=NewWriter(Path.Combine(outDir,"soh_filter_step_response.csv")))
            {
                writer.WriteLine("Samples,TimeSeconds,TimeDays,Benchmark_alpha0.02,Manuscript_alpha1e-6");
                writer.WriteLine("0,0,0,0,0");
                var samples=new SortedSet<long>();
                for(int i=0;i<=400;i++) samples.Add((long)Math.Round(Math.Pow(10.0,7.0*i/400.0)));
                foreach(long n in samples)
                {
                    double a=1.0-Math.Pow(1.0-0.02,n);
                    double b=1.0-Math.Pow(1.0-1e-6,n);
                    writer.WriteLine(string.Join(",",n.ToString(Inv),F(n),F(n/86400.0),F(a),F(b)));
                }
            }
        }

        private static void WriteFilterCharacteristic(StreamWriter writer,string name,double alpha,double fs)
        {
            double tau=-1.0/Math.Log(1.0-alpha);
            double t50=Math.Log(0.5)/Math.Log(1.0-alpha);
            double t90=Math.Log(0.1)/Math.Log(1.0-alpha);
            double oneMinus=1.0-alpha;
            double cosOmega=(1.0+oneMinus*oneMinus-2.0*alpha*alpha)/(2.0*oneMinus);
            cosOmega=Math.Max(-1.0,Math.Min(1.0,cosOmega));
            double cutoff=Math.Acos(cosOmega)/(2.0*Math.PI)*fs;
            writer.WriteLine(string.Join(",",name,F(alpha),F(fs),F(tau),F(tau/fs/86400.0),F(t50/fs/86400.0),F(t90/fs/86400.0),F(cutoff)));
        }

        private static StreamWriter NewWriter(string path)
        {
            Directory.CreateDirectory(Path.GetDirectoryName(path));
            return new StreamWriter(path,false,new UTF8Encoding(false));
        }

        private static string F(double value)
        {
            return value.ToString("G17",Inv);
        }
    }
}
