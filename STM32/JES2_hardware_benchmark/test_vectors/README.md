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
