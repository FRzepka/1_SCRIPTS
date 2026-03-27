PC / STM32 Benchmarks
=====================

Dieses Verzeichnis enthaelt die Skripte zur Simulation der Modelle auf dem PC
und zur Auswertung der STM32-Benchmarks (Latenz, Energie, Speicherbedarf).

Struktur
--------

### 1. SOC Benchmarks (`SOC/`)

Hier liegen die Skripte und Ergebnisse fuer die State-of-Charge (SOC) Modelle.

**Hauptskript (PC-Simulation):**

- `SOC/run_soc_base_quant_pruned_from_pt.py`
  - Laedt PyTorch-Checkpoints (`.pt`) und fuehrt eine Streaming-Inferenz
    fuer Base (FP32), Quantized (INT8-Simulation) und Pruned (FP32) durch.
  - Output: Plots und NPZ-Daten in `SOC/bench_v_soc_full/`.

**Beispielaufruf:**

```bash
python DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC/run_soc_base_quant_pruned_from_pt.py ^
    --out-dir DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC/bench_v_soc_full ^
    --max-steps -1
```

### 2. Archiv (`archive/`)

Sammelbecken fuer aeltere Testskripte (`test_int8_soc.py`,
`analyze_int8_quantization.py` etc.), die nicht mehr aktiv genutzt werden,
aber als Referenz aufgehoben wurden.

### 3. SOH Benchmarks (`SOH/`)

Hier liegen die Skripte und Ergebnisse fuer die State-of-Health (SOH) Modelle.

**Hauptskript (PC-Simulation):**

- `SOH/run_soh_benchmark.py` (je nach Version leicht anders benannt)
  - Fuehrt die Inferenz fuer Base, Pruned und Quantized Modelle durch.
  - Speichert Ergebnisse in `SOH/BENCH_SOH_FULL_FINAL_20251124/`.

**Post-Processing & STM32-Filter (wichtig):**

Da die Rohdaten der SOH-Modelle (insbesondere quantisiert) verrauscht sind,
wird ein STM32-Filter simuliert:

- Skript: `SOH/BENCH_SOH_FULL_FINAL_20251124/simulate_stm32_filter.py`
- Methode: Kombination aus
  - Exponential Moving Average (EMA) und
  - Slew-Rate-Limiter (Drop-Limiter).
- Typische Parameter:
  - `alpha = 1.0e-6` (entspricht ca. N = 2 000 000 Samples Glattung)
  - `max_drop = 2e-8` (maximaler Abfall pro Sample)
- Der Filter erzeugt eine sehr glatte, monotone Abnahme des SOH, passend zum
  physikalischen Alterungsprozess.

Zusammenhang zum STM32 Benchmark
--------------------------------

Der PC-Benchmark prueft hauptsaechlich:

- mathematische Korrektheit,
- Genauigkeitsverlust durch Quantisierung/Pruning
- Verhalten des STM32-Filters an echten Zeitreihen.

Die Skripte im Ordner `../STM32` messen dagegen die tatsaechliche
**Performance auf der Hardware**:

- Latenz pro Inferenz (`time_us`),
- Energie (`E_uJ`),
- RAM-Bedarf (Static + Stack).

STM32 Benchmark-Workflow (End-to-End)
-------------------------------------

Der komplette Hardware-Benchmark (alle 6 Modelle) laeuft mit einem einzigen
Kommando:

```bash
python DL_Models/LFP_LSTM_MLP/5_benchmark/run_full_suite.py
```

**Ablauf der Suite:**

Fuer jedes Modell in der Reihe

> `SOC Base`, `SOC Pruned`, `SOC Quantized`,
> `SOH Base`, `SOH Pruned`, `SOH Quantized`

wird nacheinander ausgefuehrt:

1. Build & Flash (STM32)
   - Skript: `DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/automate_stm32.py`
   - Aufruf: `python automate_stm32.py <Typ>_<Model> all`
     - Beispiel: `SOC_Base`, `SOH_Quantized`
   - Intern: `make clean`, `make -j16 all`, Flash mit
     `STM32_Programmer_CLI.exe`.

2. Benchmark auf dem Board
   - SOC: `STM32/SOC/run_single_benchmark.py`
   - SOH: `STM32/SOH/run_single_benchmark.py`
   - Auto-Reset des Boards per DTR-Pin (kein manueller Reset noetig).
   - Standard: `n_samples = 1000` (kann ueber zweiten Parameter geaendert
     werden).
   - Ergebnisse:
     - `DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOC/result_<Model>.json`
     - `DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOH/result_<Model>.json`

3. Report & kombinierte Plots
   - Skript:
     `DL_Models/LFP_LSTM_MLP/5_benchmark/PC/SOC_SOH_Combined_Results/generate_combined_report.py`
   - Wichtige Outputs:
     - `combined_model_sizes.png` (Parameter, Flash, RAM)
     - `combined_latency_hist.png` (Latenz-Verteilungen SOC vs. SOH)
     - diverse MAE- und Dashboard-Plots fuer SOC und SOH.

Voraussetzungen
---------------

- STM32CubeIDE 1.17.0 unter `C:\ST\STM32CubeIDE_1.17.0\...`
  (Pfad in `STM32/automate_stm32.py` hinterlegt).
- NUCLEO-H753ZI mit ST-LINK angeschlossen.
  - Die Benchmarks suchen den COM-Port automatisch (`find_com_port()`),
    Fallback sind COM7/COM8.
- Python-Umgebung mit:
  - `numpy`, `pandas`, `matplotlib`
  - `pyserial`

Beispiele fuer Einzel-Benchmarks
--------------------------------

Nur SOH Quantized mit 1000 Samples:

```bash
python DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOH/run_single_benchmark.py Quantized 1000
```

Nur SOC Base mit 10 000 Samples:

```bash
python DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOC/run_single_benchmark.py Base 10000
```

RAM-Messung SOC vs. SOH (combined_model_sizes.png)
--------------------------------------------------

Die Balken im Plot `combined_model_sizes.png` basieren auf zwei klar
dokumentierten Quellen.

### SOC (State of Charge)

- Die SOC-RAM-Werte kommen aus `SOC_SIZES` am Anfang von
  `PC/SOC_SOH_Combined_Results/generate_combined_report.py`.
- Diese Werte wurden einmalig aus den Linker-Map-Dateien der Projekte
  `AI_Project_LSTM_SOC_base`, `AI_Project_LSTM_SOC_pruned`,
  `AI_Project_LSTM_SOC_quantized` extrahiert
  (Data + BSS + reservierter Stack) und dort eingetragen.
- Vorteil: stabil, reproduzierbar, unabhaengig vom Benchmark-Run.

Aktuell verwendete SOC-RAM-Werte (statisch im Skript hinterlegt):

- Base: `5036 B`  (`≈ 4.9 KB`)
- Pruned: `4112 B`  (`≈ 4.0 KB`)
- Quantized: `3576 B`  (`≈ 3.5 KB`)

Die STM32-Benchmarks fuer SOC messen zusaetzlich den Stackverbrauch zur
Laufzeit, diese Messung wird aber **nicht** fuer den Plot genutzt, weil die
Werte zwischen den Varianten sehr aehnlich sind und das Bild eher
verfaelschen wuerden. Fuer die Darstellung der Speicherkosten werden daher
die Map-basierten, konservativen Werte verwendet.

### SOH (State of Health)

- Die SOH-RAM-Werte stammen direkt aus den STM32-Benchmarks im Ordner
  `DL_Models/LFP_LSTM_MLP/5_benchmark/STM32/SOH`.
- Jedes SOH-Firmwareprojekt
  (`AI_Project_LSTM_SOH_base`, `..._SOH_pruned`, `..._SOH_quantized`) gibt
  beim Start eine Zeile
  `RAM_MEASURE: Static=<...> (Data=..., BSS=...), ...`
  aus und fuegt in jeder `METRICS:`-Zeile sowohl
  `Stack=<...>` als auch `Static=<...>` an.
- Das Skript `STM32/SOH/run_single_benchmark.py` parst daraus:
  - `static_ram_bytes`  (Data + BSS)
  - `max_stack_bytes`   (maximal beobachteter Stackverbrauch)
  - `total_ram_bytes = static_ram_bytes + max_stack_bytes`
- In `generate_combined_report.py` aktualisiert
  `plot_model_sizes_combined()` die Eintraege in `SOH_SIZES[...]` genau mit
  diesen gemessenen `total_ram_bytes`.

Damit ergeben sich im Plot (Stand der letzten Messung):

- SOH Base: `8896 B`  (`≈ 8.7 KB`)
- SOH Pruned: `7128 B`  (`≈ 7.0 KB`)
- SOH Quantized: `6864 B`  (`≈ 6.7 KB`)

Kurzfassung
-----------

- **SOC-RAM**: aus Linker-Map (statisch, konservativ).  
- **SOH-RAM**: aus STM32-Laufzeitmessung (`Static + max Stack`).

Mit diesen Informationen laesst sich der komplette Zustand der Benchmarks
(PC + STM32) jederzeit reproduzieren und die Herkunft der im Report
gezeigten RAM-Werte ist nachvollziehbar dokumentiert.

