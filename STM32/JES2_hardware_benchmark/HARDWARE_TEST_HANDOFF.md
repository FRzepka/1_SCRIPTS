# JES2 STM32 Hardware-Test: Uebergabeprotokoll

Stand: 2026-08-26

## Zweck

Dieses Verzeichnis ist die verbindliche Uebergabe vom HPC auf den lokalen PC mit
angeschlossenem STM32H753ZI. Ein neuer Codex-Chat soll zuerst diese Datei und
anschliessend `README.md` sowie `SERIAL_PROTOCOL.md` lesen.

Starttext fuer den neuen Chat:

```text
Lies STM32/JES2_hardware_benchmark/HARDWARE_TEST_HANDOFF.md vollstaendig und
fahre mit dem JES2-STM32-Hardwarebenchmark fort. Das STM32H753ZI ist an diesem
PC angeschlossen. Pruefe zuerst Git-Stand, COM-Port und STM32CubeIDE/X-CUBE-AI.
```

## Festgelegter Umfang

Der Hardwaretest ist ein isolierter SOC-Ausfuehrungsbenchmark. Er wiederholt
keine Rausch-, Bias-, Quantisierungs-, Missing-Sample-, Jitter-, Dropout- oder
Spike-Tests. Diese Robustheitstests sind bereits Bestandteil der vollstaendigen
Softwarekampagne.

Auf dem STM32 werden gemessen:

1. reine Inferenzzeit ueber den DWT-Zykluszaehler,
2. Verteilung von Median, P95 und Maximum ueber mehrere Sequenzdurchlaeufe,
3. Flash-Belegung des finalen Images,
4. statisches RAM sowie Peak-Stack und Modell-/Aktivierungspuffer,
5. numerische Abweichung zur float32-Software-Referenz,
6. optional reale Energie pro Inferenz mit externem Messgeraet und Trigger-Pin.

Host-UART-Latenz wird nur als Diagnose gespeichert und darf nicht als reine
Inferenzzeit berichtet werden.

## Zu vergleichende SOC-Estimatoren

- `DM`: Coulomb Counting mit fester Nennkapazitaet.
- `HDM`: Coulomb Counting mit vorgegebener kausaler SOH-Spur.
- `HECM`: Zwei-RC-EKF mit vorgegebener kausaler SOH-Spur.
- `DD`: JES2-GRU-MLP aus `SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct`.

Das SOH-LSTM wird nicht auf dem Board ausgefuehrt. HDM, HECM und DD erhalten in
jedem Testschritt exakt dieselbe, auf dem HPC vorberechnete SOH-Spur. Damit misst
der Hardwaretest nur den estimator-spezifischen SOC-Aufwand. Im Paper muss dies
als isolierter SOC-Hardwarebenchmark bezeichnet werden. Der Softwarebenchmark
bleibt die End-to-End-Auswertung mit gemeinsamem kausalem SOH-LSTM.

## Kritischer Versionshinweis

Die vorhandenen Projekte unter `STM32/workspace_1.17.0/AI_Project_LSTM_SOC_*`
verwenden ein aelteres LSTM mit sechs Eingangsmerkmalen. Sie sind nicht direkt
das JES2-DD-Modell und duerfen nicht unveraendert als JES2-Hardwaremessung
verwendet werden.

Der JES2-DD-Vertrag lautet:

- Modell: GRU-MLP, hidden size 67, MLP hidden size 96, eine GRU-Schicht.
- Checkpoint: `DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/checkpoints/best_model_finetuned.pt`.
- Skalierer: `DL_Models/LFP_SOC_SOH_Model/2_models/SOC_1.7.0.0/PrunedFT_1.7.0.0_s30_struct/scaler_robust.joblib`.
- Sequenzlaenge: 2024 Samples.
- Eingabe: Spannung, Strom, Temperatur, SOH, Q_c, dU/dt, dI/dt, dt.
- Ausgabe: SOC im Bereich 0 bis 1.

Die JES2-Softwarekampagne nutzt den Rolling-Window-Pfad. Fuer eine direkte
Vergleichbarkeit muss die primaere STM32-Messung ebenfalls ein 2024-Sample-
Fenster auswerten. Eine stateful One-Step-Variante darf zusaetzlich gemessen
werden, ist aber als separate Deployment-Optimierung zu kennzeichnen und zuerst
gegen die Rolling-Window-Referenz zu quantifizieren.

## Bereits erledigt

- SOC- und SOH-Train/Validation-Splits wurden abgeglichen.
- Ein neues gemeinsames SOH-Modell fuer JES2 wurde trainiert.
- 19 reproduzierbare Softwaretestszenarien wurden implementiert.
- Die Testzellen C09, C13, C15, C25, C27 und C29 wurden nach SOH-Zustand und
  Lastklasse in feste 24-h-Fenster aufgeteilt.
- DD-Pilot: 304 von 304 Laeufen erfolgreich.
- Vier-Modell-Smoke-Test: erfolgreich.
- Vollkampagne: 6720 von 6720 Laeufen erfolgreich; keine fehlgeschlagenen Runs.
- Modelle: DM 1200 Runs, HDM 1840, HECM 1840, DD 1840.
- Finale Manifeste wurden zu
  `campaigns/jes2_full_holdout_merged_20260825.json` zusammengefuehrt.
- Die Skripte in diesem Verzeichnis definieren Host-Protokoll, Speicheranalyse
  und Ergebniszusammenfassung fuer den Hardwaretest.
- Der tatsaechlich verwendete pruned/fine-tuned JES2-DD-Checkpoint wurde als
  Fixed-Window- und Stateful-ONNX exportiert. Fixed-Window ONNX und PyTorch
  stimmen bis auf eine maximale Abweichung von 5.96e-08 ueberein.
- Eine nominale 4096-Sample-Sequenz aus dem eingefrorenen C27-Fresh-Fenster mit
  gemeinsamer SOH-Spur und Referenzausgaben aller vier Estimatoren liegt unter
  `test_vectors/jes2_nominal_vectors.csv`.

## Finale Softwareauswertung

Die finale Softwareauswertung der 6720 Runs mit 10000 hierarchischen Bootstrap-
Wiederholungen pro Metrikgruppe wurde am 2026-08-26 erfolgreich abgeschlossen.
Der Abschluss ist durch `FULL_POSTPROCESS_EXIT_STATUS=0` in
`DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_postprocess_20260825.log`
belegt. Die Kampagnen-Rohdaten sind absichtlich per `.gitignore` vom Git-
Upload ausgeschlossen; Paper-Tabellen und Diagramme werden versioniert.

## Noch auf dem HPC zu erledigen

- Finale Tabellen/Diagramme auf Plausibilitaet pruefen.
- Reviewer-To-do-Dateien auf den finalen Kampagnenstatus aktualisieren.

## Noch am lokalen STM32-PC zu erledigen

1. `git pull` ausfuehren und diese Datei lesen.
2. Boardbezeichnung, MCU, Takt, CubeIDE- und X-CUBE-AI-Version dokumentieren.
3. Fuer DM, HDM, HECM und DD reproduzierbare Release-Firmware bauen.
4. Das in `SERIAL_PROTOCOL.md` definierte Protokoll implementieren.
5. Jedes Image flashen und `scripts/collect_serial_benchmark.py` ausfuehren.
6. ELF/Map-Dateien mit `scripts/extract_memory_report.py` auswerten.
7. Peak-Stack per Stack-Painting/Linker-Unterstuetzung messen; statisches RAM
   allein reicht fuer die Reviewer-Antwort nicht.
8. Optional Trigger-Pin und externe Leistungsmessung ausfuehren.
9. Alle Resultate mit `scripts/summarize_results.py` zusammenfassen.
10. Rohresultate, Build-Metadaten und Firmware-Commit nach Git pushen.

## Akzeptanzkriterien

- Identische geordnete Testvektoren fuer alle vier Estimatoren.
- Mindestens drei vollstaendige Messrunden je Firmware nach einem Warm-up.
- Keine verlorenen oder vertauschten Sample-IDs.
- DWT-Zyklen fuer jede gueltige Inferenz vorhanden.
- Numerische Abweichung gegen die Software-Referenz berichtet, nicht nur gegen
  den Dataset-SOC.
- Compilerflags, MCU-Takt, Firmware-Hash und Toolversionen im Ergebnis enthalten.
- Flash, statisches RAM, Peak-Stack und Aktivierungspuffer getrennt ausgewiesen.
- Keine Robustheitsstoerungen im Hardwaredatensatz.

## Git-Regel

Kleine CSV/JSON-Resultate, Berichte und Build-Metadaten werden committed. Grosse
Rohlogs, Debug-Verzeichnisse und temporaere CubeIDE-Artefakte bleiben lokal. Vor
jedem Push `git status` und Dateigroessen kontrollieren.
