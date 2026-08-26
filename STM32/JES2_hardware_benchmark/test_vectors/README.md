# Testvektoren

Die finale Datei `jes2_nominal_vectors.csv` wird auf dem HPC erzeugt und klein
genug fuer Git gehalten. Sie enthaelt ausschliesslich nominale Messwerte und die
zugehoerigen float32-Softwareausgaben.

Pflichtspalten:

```text
sample_id,segment_id,reset,voltage_v,current_a,temperature_c,soh,q_c_ah,dv_dt_v_s,di_dt_a_s,dt_s,expected_dm,expected_hdm,expected_hecm,expected_dd
```

`reset=1` markiert den ersten Sample eines unabhaengigen Segments. Die Reihenfolge
darf auf dem PC nicht sortiert oder veraendert werden. Fuer DD sind Ausgaben vor
dem gefuellten 2024-Sample-Fenster leer.
