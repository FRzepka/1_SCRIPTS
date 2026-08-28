# All Cells – kuratierte Abbildungsübersicht

Die 24 ausgewählten PNG-Dateien sind lückenlos als `Figure_01` bis `Figure_24`
nummeriert. Quelldateien außerhalb dieses Ordners bleiben unverändert. Das
Farbschema ist durchgängig DM = grün, HDM = lila, HECM = blau und DD = rot.

## Umgesetzte Auswahl

| Abbildung | Umsetzung | Prüfstatus |
|---|---|---|
| 01–03 | Originale unverändert übernommen | Anforderungs-, Methoden- und Taxonomieübersichten. |
| 04 | Aktuelle Six-Cell-Baseline übernommen | Gewünschter Ersatz; zeigt Zellen als Punkte und Cell-Macro-Mittel mit 95%-KI. |
| 05–06 | Current-bias sensitivity und Lifecycle-Reset | Gepaarte Gain-Fehler sowie zeitliche Akkumulation und Recovery. |
| 07–08 | Noise robustness | Dissertation-Detail und Six-Cell-Übersicht. |
| 09 | Initial-state recovery | Gepaarte Six-Cell-Recovery mit stabilem Recovery-Kriterium. |
| 10–11 | Signal integrity | Detaildarstellung und Six-Cell-Übersicht. |
| 12 | Burst dropout | Korrigierter Übergang mit eingefrorenen Online-Zuständen während des Gaps. |
| 13–14 | Voltage spikes | Six-Cell-Zusammenfassung und JES2-Verlaufsdarstellung. |
| 15–16 | Cross-scenario heatmap | Aktuelle starke Palette sowie erhaltene gedämpfte Vergleichsversion. |
| 17 | Decision synthesis | Aktuelle Six-Cell-Scores einschließlich Burst-Dropout und Recovery. |
| 18–19 | ADC quantization | Detaildarstellung und Six-Cell-Fassung. |
| 20–24 | Methodische Ergänzungen | Zellabdeckung, Statistik, Testmatrix, Workflow und Evaluationsfenster. |

## Wichtiger Rechenstatus

Die vorhandenen Six-Cell-Abbildungen wurden vor der jüngsten Korrektur der
Online-`Q_c`-Rekonstruktion und des Burst-Dropout-Fensters erzeugt. Für reine
Sichtung und Layoutauswahl sind sie geeignet. Numerische Dropout-Aussagen sowie
davon abhängige Cross-Scenario- und Decision-Abbildungen gelten erst nach einem
vollständigen korrigierten Six-Cell-Rerun als final.

Erzeugung: `python figures/build_all_cells_collection.py`

Nur Figure 15 neu erzeugen:
`python figures/build_revised_all_cells_figures.py --figure-12-only --figure-12-output "figures/Results/All Cells/Figure_15_Cross_Scenario_Heatmap_Six_Cell.png"`

Nur Figure 17 neu erzeugen:
`python figures/build_revised_all_cells_figures.py --figure-13-only --figure-13-output "figures/Results/All Cells/Figure_17_Decision_Synthesis_Six_Cell.png"`
