Benchmark-Übersicht (SOC / SOH)
================================

Ziel
----
- Vergleich der verschiedenen Modellvarianten (PC / C / Quant / später Pruned) **aus Ingenieur-Sicht**:
  - Genauigkeit: Wie gut trifft das Modell die Groundtruth?
  - Echtzeitfähigkeit: Latenz pro Schritt, Throughput (Samples/s).
  - Ressourcen: Flash-/RAM-Bedarf.
  - Energie: Energie pro Inferenz auf STM32.
  - Nutzen der Optimierungsschritte: Was bringt Quantisierung / Pruning in %?
- Alle Benchmarks so aufbauen, dass **Plotdaten als Dateien gespeichert** werden (NPZ/JSON/CSV), damit Paper-Plots ohne erneutes Durchlaufen reproduziert werden können.

Modell-Varianten
----------------
- PC (Python, FP32):
  - Referenz: seq2many (SOH/SOC), inkl. Online-Filter + PIN-Calib.
- C-Sim (PC, FP32-Gleichung wie STM32):
  - Schrittweise C-Referenz (ohne Embedded-Overhead).
- Quantisiert (PC, INT8/INT16-Gewichte + FP32 Aktivierungen):
  - Neues SOH_Quantized-Paket (`quant_state_soh_int16hh_p99_5.npz`).
- STM32:
  - Base-SOC/-SOH-Modell (FP32 / Mixed).
  - Quantized-SOC/-SOH-Modell (INT8/INT16).
  - Später: Pruned-Varianten (gleiche Metriken).

1. Genauigkeit (PC-Teil)
------------------------
Ziel: Zeigen, dass alle Varianten (Python, C-Sim, Quant) **inhaltsgleich** oder zumindest “engineering-good enough” sind.

Plots
- Serien-Overlay (global):
  - `overlay_full.png`: Groundtruth vs PC-Referenz vs C-Sim vs Quant (volle Länge).
  - Zweck: Visuales Vertrauen in die Kurvenform.
- Zoom (Anfang):
  - `overlay_firstN.png` (z. B. erste 2000 Punkte).
  - Fokus: Start bei 1, Verhalten von PIN/Filter, Overshoot.
- Zoom (kritische Bereiche / Tail):
  - `overlay_tailN.png` (z. B. letzte 2000–5000 Punkte).
  - Fokus: Drift am Ende, Stabilität bei niedrigem SOH/SOC.
- Fehler-Zeitverlauf:
  - `error_plot.png`: Fehler (z. B. y_pred − y_true oder |Δ|) über der Zeit.
  - Gut geeignet als Unterplot unter einem Overlay (oben Verlauf, unten Fehler).
- Fehler-Histogramm:
  - `diff_hist.png`: |Δ| zwischen Referenz und Variante.
  - Je nach Vergleich:
    - C vs Python
    - Quant vs C
    - später: Pruned vs C/Quant

Metrik-Tabellen (aus JSON)
- Pro Vergleich eine JSON-Datei, z. B.:
  - `streaming_benchmark.json`, `streaming_benchmark_no_pruned.json`, `*_metrics.json`.
- Inhalt je Modell:
  - `MAE`, `RMSE`, `MaxAbsError` (gegen Groundtruth).
  - `MAE_C_vs_QUANT`, `RMSE_C_vs_QUANT` (nur interner Abgleich).
  - `N` (Anzahl verglichener Schritte).
- Plot / Tabelle:
  - Balkenplot oder Tabelle: Modell vs. MAE/RMSE.
  - Fürs Paper: kompakte Tabelle (z. B. 3–5 Zeilen, 2–3 Spalten).

Wichtig
- Alle PC-Skripte, die Overlays erzeugen, sollen die verwendeten Befehle (z. B. `RUN_CMD.md`) und Metriken (`metrics.json`) **neben die Bilder schreiben**.
- Streckdaten (Predictions) bei Bedarf als `*.npz` mit Arrays
  - `y_true`, `y_py`, `y_c`, `y_quant`, …
  - Beispiel: `streaming_benchmark_preds.npz`.

2. Ressourcen / Größe (HPC / Build-Teil)
----------------------------------------
Ziel: Speicherbedarf und Modellkomplexität klar machen.

Metriken
- Modelldetails (aus Training / Checkpoint):
  - Parameteranzahl (LSTM/MLP, gesamt).
  - Hidden-Size, Repräsentation (FP32, INT8/INT16).
- Binärgrößen:
  - Größe des C-Weight-Headers / Library (Bytes, kB).
  - STM32-Firmware-Größe (Flash usage) aus Build-Output.
- RAM:
  - Laufzeit-RAM-Bedarf (Stack/Heap) – idealerweise aus Linker-Map oder einfacher RAM-Benchmark auf HPC (`ram_benchmark.py`) als Referenz.

Plots/Tabelle
- Balkenplot `hidden_size.png`:
  - x-Achse: Modellvarianten (z. B. FP32, INT8, Pruned).
  - y-Achse: Hidden-Size bzw. Parameterzahl.
- Modellgrößen-Plot `model_sizes.png` (aus `out/compare_fp32_fp16_int8_pruned_new/`):
  - Kombinierter Blick auf Param-Anzahl, Header-Größe und evtl. ONNX/PT-Größen pro Modell.
- Tabelle „Modellgrößen“ (fürs Paper):
  - Spalten: Modell, Param (#), Flash (kB), Weight-Header (kB), RAM (kB).
  - Alle Zahlen aus Skripten generiert und in JSON/CSV abgelegt.
  - Für das Paper kann FP16 entfallen; Fokus auf FP32 (Baseline), INT8/INT16 (Quant) und Pruned.

3. Laufzeit / Durchsatz (PC & STM32)
------------------------------------
Ziel: Zeigen, wie schnell die Modelle als **Streaming-Pipeline** laufen – einmal auf dem PC (repräsentativ), einmal auf der MCU (realistisch).

PC-Benchmark
- Skripte: z. B. `benchmark_onnx_streaming*.py` / `compare_all_models.py` / `plot_streaming_benchmark.py`.
- Messen:
  - `latency_ms_per_step` (Mittelwert, evtl. Verteilung).
  - `throughput_samples_per_s` (inverse Latenz).
- Ausgabedateien:
  - `streaming_benchmark.json` / `streaming_benchmark_no_pruned.json`.
  - `streaming_benchmark_preds.npz` für Plotdaten.

STM32-Benchmark
- Vorgehen:
  - Messung der Zyklen um **nur die Inferenz** (z. B. via DWT Cycle Counter).
  - Optional: Energiemessung über externes Messgerät (Shunt + Logger).
- Ausgabekanal:
  - MCU schreibt Zeilen wie:
    - `METRICS: model=BASE cycles=... us=... E_uJ=...`
    - `METRICS: model=QUANT cycles=... us=... E_uJ=...`
  - PC-Parser sammelt diese Zeilen in `.json`/`.csv`.

Plots
- `latency_bar_inline.png`:
  - kompakter Balkenvergleich der Latenz (PC-Streaming) für wenige Varianten (z. B. Baseline vs Quant).
- `latency_bar.png`:
  - ausführlicherer Balkenplot inkl. Pruned-Varianten; kann später auf FP32/INT8/Pruned reduziert werden, FP16 ist nicht zwingend nötig.
- `throughput_bar_inline.png`:
  - Samples/s pro Modell (Durchsatz); eignet sich gut als Gegenstück zu `latency_bar_inline.png`.
- `latency_hist.png`:
  - Verteilung der Schritt-Latenzen (Histogramm), zeigt Jitter / Worst-Case-Verhalten.
- Optional: Energie pro Sample (Barplot oder zusätzlich zur Latenz in einer kombinierte Figur).

4. Energieverbrauch (STM32)
---------------------------
Ziel: Für Ingenieure besonders interessant – Energie pro Vorhersage.

Messkonzept
- Hardware: Shunt-Widerstand oder Strommessgerät in der VDD-Leitung.
- Vorgehen:
  - MCU läuft in definiertem Modus (nur Modell, feste Frequenz).
  - Messfenster mit N Inferenzschritten, Energie ΔE messen.
  - Energie pro Schritt: `E_step = ΔE / N`.
- Einbindung in Benchmark-Output:
  - `energy_uJ_per_step` im JSON für jedes Modell.

Darstellung
- Tabelle im Paper:
  - Modell, Latenz (µs), Energie (µJ), Speedup (%), Energieeinsparung (%).
- Optionaler Plot:
  - Kombinierter Balkenplot mit zwei Skalen (Latenz vs Energie) oder zwei separate Plots.

5. SOC/SOH-Kurven – Vergleich der Implementierungen
---------------------------------------------------
Ziel: Für den Leser sichtbar machen, dass die integrierten C/Quant-Implementierungen im Zeitverlauf **gleich** zu Python arbeiten.

Plots (PC, gespeist von gespeicherten Arrays)
- Für jedes relevante Szenario:
  - `overlay_full.png` (z. B. im Ordner `5_benchmark/model_comparison/output`):
    - Kurven: Groundtruth, Python, C-Sim, Quant.
  - `overlay_firstN.png`:
    - Zoom in den kritischen Startbereich (PIN-Kalibrierung, Filter).
  - `overlay_tailN.png`:
    - Zoom in den Bereich kurz vor Ende (Drift/Offset).
  - Varianten wie `overlay_first5000.png` (SOC-Fokus) können zusätzlich genutzt werden, um SOC-Läufe getrennt von SOH darzustellen.

Datenablage
- Die Skripte, die diese Plots erzeugen, sollen vorher ein `.npz` schreiben, z. B.:
  - `streaming_benchmark_preds.npz` mit Schlüsseln:
    - `t` (Index), `y_true`, `y_py`, `y_c`, `y_quant`, ...
  - Optional: `mask_valid` oder Flags für Bereiche ohne Groundtruth.
- Damit kann man später problemlos neue Plots für Paper / Präsentationen erzeugen (ggf. in einem separaten Notebook).

6. Ordner- / Dateistruktur
--------------------------
- `5_benchmark/PC/`:
  - Skripte für reine PC-Referenz- und Vergleichsläufe (Py vs C vs Quant).
  - Alle Skripte sollen die Ergebnisse nach `5_benchmark/model_comparison/output/` bzw. `5_benchmark/out/` schreiben.
- `5_benchmark/model_comparison/output/`:
  - `overlay_full.png`, `overlay_firstN.png`, `overlay_tailN.png`.
  - `streaming_benchmark*.json`, `*preds.npz`.
  - `PLOTS_STREAMING_BENCHMARK.md` (Beschreibung der aktuell erzeugten Bilder).
- `5_benchmark/out/`:
  - Aggregierte Benchmarks über alle Modelle (Tabellen/JSON für Latenz, Durchsatz, Energie).
- `5_benchmark/HPC/`:
  - RAM/CPU-Benchmarks auf Großrechner (z. B. 32 GB vs 16 GB Runs), falls für das Paper relevant.

7. Was später ins Paper kommt
-----------------------------
Vorschlag für Paper-Figuren
- Figur A: SOC/SOH-Kurve mit Fehlerdarstellung (2-spaltig passend):
  - Oben: `overlay_firstN.png` oder `overlay_first5000.png` (Groundtruth vs PC vs C vs Quant).
  - Unten: `error_plot.png` (Fehlerverlauf derselben Modelle).
  - So sieht man Verlauf und Fehler direkt untereinander.
- Figur B: Ressourcen- und Modellgrößenübersicht:
  - Modell, Param (#), Flash (kB), RAM (kB), Latenz (µs), Energie (µJ).
- Figur C: Laufzeit und Throughput:
  - Linke Seite: `latency_bar_inline.png` (oder `latency_bar.png` mit FP32/INT8/Pruned).
  - Rechte Seite: `throughput_bar_inline.png`.
- Figur D (optional): Latenzverteilung und Modellgrößen:
  - Oben: `latency_hist.png` (zeigt Jitter / Worst-Case).
  - Unten: `model_sizes.png` oder `hidden_size.png` (Komplexitätsvergleich).
- Optional: Energieplot:
  - Energie pro Vorhersage für Base vs Quant vs Pruned (als zusätzliche Balkengrafik oder gemeinsam mit Latenz).

Hinweis für Implementierung
---------------------------
- Für jedes neue Skript in `5_benchmark`:
  - Immer:
    - Rohdaten (Predictions / Metriken) als `*.npz` / `*.json` speichern.
    - zusätzlich `RUN_CMD.md` mit dem genauen Aufruf ablegen.
  - Plots nur als „Ansicht“ verstehen – die eigentliche Wahrheit sind die gespeicherten Arrays/Metriken.
