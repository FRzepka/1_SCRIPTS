# Testvektoren

Die Datei `jes2_nominal_vectors.csv` wurde auf dem HPC aus dem eingefrorenen
frischen C27-Baselinefenster erzeugt. Sie enthaelt 4096 ausschliesslich nominale
Messwerte, die gemeinsame kausale SOH-Spur und die Software-Referenzausgaben.

Pflichtspalten:

```text
sample_id,segment_id,reset,voltage_v,current_a,temperature_c,soh,q_c_ah,dv_dt_v_s,di_dt_a_s,dt_s,expected_dm,expected_hdm,expected_hecm,expected_dd
```

`reset=1` markiert den ersten Sample eines unabhaengigen Segments. Die Reihenfolge
darf auf dem PC nicht sortiert oder veraendert werden. Fuer DD sind Ausgaben vor
dem gefuellten 2024-Sample-Fenster leer.

## Sechs-Zellen-Vektoren

`multicell/` enthaelt dieselbe nominale 4096-Sample-Auswertung fuer alle sechs
Holdout-Zellen. Die feste Lastklasseneinteilung steht in
`multicell/jes2_multicell_manifest.json`:

- Low: C25, C27
- Medium: C09, C13, C15
- High: C29

Jede CSV enthaelt `soc_dataset` sowie deterministische Software-Sollwerte fuer
DM, HDM, HECM und DD. Die DD-Sollwerte werden mit dem validierten Fixed-Window-
ONNX-Modell auf CPU erzeugt, um CUDA-Laufstreuung aus dem Hardwarevergleich
auszuschliessen. C27 reproduziert den zuvor auf dem Board validierten DD-
Sollverlauf bis auf maximal `1.37e-7` SOC.

Reproduzierbarer HPC-Export:

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  STM32/JES2_hardware_benchmark/scripts/build_multicell_hardware_references.py
/home/florianr/anaconda3/envs/ml1/bin/python \
  STM32/JES2_hardware_benchmark/scripts/export_multicell_hardware_vectors.py
```
