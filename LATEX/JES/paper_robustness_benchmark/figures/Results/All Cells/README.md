# All Cells – kuratierte Abbildungsübersicht

Die 25 ausgewählten PNG-Dateien sind lückenlos als `Figure_01` bis `Figure_25`
nummeriert. Figures 01–20 gehören in den Hauptteil; Figures 21–25 sind zusätzlich
mit `APPENDIX` gekennzeichnet und gehören in den Anhang. Quelldateien außerhalb
dieses Ordners bleiben unverändert. Das Farbschema ist durchgängig DM = grün,
HDM = lila, HECM = blau und DD = rot.

## Umgesetzte Auswahl

| Abbildung | Umsetzung | Prüfstatus |
|---|---|---|
| 01–03 | Dissertation-Fassungen aus der EAAI-Palette übernommen | Anforderungs-, Methoden- und Taxonomieübersichten in den finalen Dissertation-Farben. |
| 04 | Aktuelle Six-Cell-Baseline übernommen | Gewünschter Ersatz; zeigt Zellen als Punkte und Cell-Macro-Mittel mit 95%-KI. |
| 05–06 | Current-gain sensitivity und Lifecycle-Reset | Gepaarte Vorzeichen-Sensitivität des finalen Benchmark-Builds sowie C29-Diagnose zu Akkumulation und Reset. |
| 07–08 | Noise robustness | Dissertation-Detail und Six-Cell-Übersicht. |
| 09 | Initial-state recovery | Korrigierte dedizierte C29-Paarkurve sowie Six-Cell-Vergleich von erstem 300-s-Eintritt und persistenter Recovery. Endpunkte am gemeinsamen Start sind linkszensiert. |
| 10 | Signal integrity | Aktuelle Six-Cell-Übersicht; die redundante historische Detailabbildung wurde entfernt. |
| 11 | Burst dropout | Korrigierter Übergang mit eingefrorenen Online-Zuständen während des Gaps. |
| 12–13 | Voltage spikes | Six-Cell-Zusammenfassung und JES2-Verlaufsdarstellung. |
| 14 | Cross-scenario heatmap | Aktuelle Six-Cell-Fassung mit der starken Dissertation-Palette. |
| 15 | Decision synthesis | Illustrative Six-Cell-Scores einschließlich Burst-Dropout, gepaartem Gain-Sweep und beobachteter persistenter Recovery. Relapse bleibt ein separates Diagnostikum. |
| 16 | ADC quantization | Strom- und Spannungsdetail kombiniert mit Six-Cell-Mittel, hierarchischem 95%-KI sowie ΔMAE- und MAE-Werten. |
| 17–20 | STM32-Hardwarebenchmark, Hauptteil | Hardware-/Software-Äquivalenz, On-Device-Latenzen, Speicherbedarf und DD-Inferenzmodi. Figure 20 benötigt noch den unten dokumentierten Style-Rebuild. |
| 21–24 | Anhang (`APPENDIX`) | Holdout-Abdeckung, JES2-Testmatrix, Evaluationsfenster und detaillierte DD-Latenzverteilungen. |
| 25 | HECM Lookup-Sensitivität (`APPENDIX`) | 240 HECM-Läufe über 16 Fenster zeigen Baseline-Accuracy, adverse Current-Gain-Penalty und den Lookup-×-Gain-Interaktionseffekt für Widerstand ±10 % und OCV ±10 mV. |

## Wichtiger Rechenstatus

Die finalen Six-Cell-Abbildungen verwenden die korrigierte Online-`Q_c`-
Rekonstruktion und die neu berechnete Dauerreferenz für Burst-Dropout. Figure 09
liest ausschließlich die dedizierte korrigierte Initialisierungs-Paarkampagne.
Figure 15 verwendet persistente Recovery, Excess-Error-AUC und persistente
Zensierung. Ein Relapse nach erstem Eintritt wird separat berichtet und nicht in
den Recovery-Score eingerechnet, weil er für nie recoverte Läufe undefiniert ist.

Der Plotter für Figure 20 wurde auf den transparenten Balkenstil ohne Punkt- und
Schraffurmuster umgestellt. Das aktuelle PNG enthält noch den vorherigen Stil,
weil die dafür benötigten per-cell STM32-`summary.json`-Dateien nur auf dem
Hardware-PC liegen. Dort muss
`python STM32/JES2_hardware_benchmark/scripts/build_four_hardware_benchmark_figures.py`
ausgeführt und `figure_04_dd_inference_modes.png` anschließend als Figure 20
übernommen werden. Die Messwerte selbst ändern sich dadurch nicht.

Erzeugung: `python figures/build_all_cells_collection.py`

Nur Figure 14 neu erzeugen:
`python figures/build_revised_all_cells_figures.py --figure-12-only --figure-12-output "figures/Results/All Cells/Figure_14_Cross_Scenario_Heatmap_Six_Cell.png"`

Nur Figure 15 neu erzeugen:
`python figures/build_revised_all_cells_figures.py --figure-13-only --figure-13-output "figures/Results/All Cells/Figure_15_Decision_Synthesis_Six_Cell.png"`

Nur Figure 16 neu erzeugen:
`python figures/build_figure_16_adc_quantization.py`
