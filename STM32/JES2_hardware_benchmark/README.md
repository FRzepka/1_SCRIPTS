# JES2 STM32 SOC Hardware Benchmark

Dieses Paket bereitet den isolierten Hardwarevergleich von DM, HDM, HECM und DD
auf einem STM32H753ZI vor. Es verwendet nur nominale, vorab festgelegte
Testvektoren. Die Messfehlerkampagne wird nicht auf Hardware wiederholt.

## Verzeichnis

- `HARDWARE_TEST_HANDOFF.md`: verbindlicher Status und PC-Uebergabe.
- `SERIAL_PROTOCOL.md`: UART-Vertrag zwischen Host und Firmware.
- `config/models.json`: versionierte Modell- und Eingabevertraege.
- `scripts/collect_serial_benchmark.py`: sequenzieller UART-Benchmark.
- `scripts/extract_memory_report.py`: Flash/RAM-Bericht aus ELF-Dateien.
- `scripts/summarize_results.py`: gemeinsame Paper-Tabelle aus Einzelmessungen.
- `test_vectors/`: nominale Eingaben und Software-Referenzausgaben.

## Lokale Installation

```bash
python -m venv .venv
.venv/Scripts/activate
pip install -r STM32/JES2_hardware_benchmark/requirements.txt
```

Unter Linux lautet die Aktivierung `.venv/bin/activate`.

## Serieller Benchmark

```bash
python STM32/JES2_hardware_benchmark/scripts/collect_serial_benchmark.py \
  --port COM7 \
  --model DD \
  --vectors STM32/JES2_hardware_benchmark/test_vectors/jes2_nominal_vectors.csv \
  --out-dir STM32/JES2_hardware_benchmark/results/DD
```

Das DD-Image muss die ersten 2023 Samples als Fenster-Warm-up kennzeichnen. Bei
DM, HDM und HECM kann jede Zeile nach dem letzten Reset ausgewertet werden.

## DD-Export auf dem HPC

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  STM32/JES2_hardware_benchmark/scripts/export_dd_onnx.py
```

Primaer ist der Fixed-Window-Export mit 2024 Samples. Der Stateful-Export ist nur
eine zusaetzliche Deployment-Variante und nicht automatisch numerisch identisch
zum Rolling-Window-Verfahren der Softwarekampagne.

## Speicherbericht

```bash
python STM32/JES2_hardware_benchmark/scripts/extract_memory_report.py \
  --image DD=path/to/DD.elf \
  --image HECM=path/to/HECM.elf \
  --out STM32/JES2_hardware_benchmark/results/memory.json
```

Das Skript verwendet standardmaessig `arm-none-eabi-size -A`. Die resultierenden
statischen Werte ersetzen keine Peak-Stack-Messung auf dem Board.

## Zusammenfassung

```bash
python STM32/JES2_hardware_benchmark/scripts/summarize_results.py \
  --results-root STM32/JES2_hardware_benchmark/results \
  --out STM32/JES2_hardware_benchmark/results/hardware_summary.csv
```
