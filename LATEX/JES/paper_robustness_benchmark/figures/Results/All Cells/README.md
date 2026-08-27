# All Cells – kuratierte Abbildungsübersicht

Diese Sammlung ist **nicht destruktiv**: Die Quelldateien in `JES_Upload/Figures`,
den Dissertation-Pictures und `figures/Results` bleiben unverändert. Das
Farbschema ist durchgängig DM = grün, HDM = lila, HECM = blau und DD = rot.

## Umgesetzte Auswahl

| Abbildung | Umsetzung | Prüfstatus |
|---|---|---|
| 01–03 | Originale unverändert übernommen | Inhaltlich weiterhin gültige Anforderungs-, Methoden- und Taxonomieübersichten; Fig. 16, 27–29 ergänzen die Six-Cell-Methodik. |
| 04 | Aktuelle Six-Cell-Baseline übernommen | Gewünschter Ersatz; zeigt Zellen als Punkte und Cell-Macro-Mittel mit 95%-KI. |
| 05 | Originale Panels (a) und (c); nur Panel (b) durch die aktuelle Six-Cell-Sensitivität ersetzt | Testidee kompatibel. Die aktuelle Auswertung verwendet gepaarte positive/negative Gain-Fehler und den ungünstigeren Effekt je Zelle. |
| 06 | Dissertation-Detailabbildung erhalten; aktuelle Six-Cell-Übersicht separat als 06b ergänzt | Kompatibel: Detail-Sweep und Multi-Channel/Six-Cell-Übersicht beantworten unterschiedliche Fragen. |
| 07 | Dissertation-Abbildung erhalten; aktuelle gepaarte Six-Cell-Recovery separat als 07b | Gemeinsamer physikalischer Voll-Ladeanker bei SOC = 1,0; DD erscheint erst nach seinem Sequenz-Warm-up. Die Zusammenfassung verwendet stabile Recovery ohne späteren Rückfall. |
| 08 | Dissertation-Abbildung erhalten; aktuelle Six-Cell-Übersicht separat als 08b | Missing-sample- und Timing-Teile gültig; historische Burst-Dropout-Anteile nicht für neue Schlussfolgerungen verwenden. |
| 09 | Durch `Figure_09_Burst_Dropout_Transition_CORR` ersetzt | Korrigierter C29-Übergang mit eingefrorenen Online-Zuständen während des Gaps. Das Six-Cell-Recovery-Panel stammt noch aus den bisherigen Vollkampagnen und muss nach einem vollständigen korrigierten Six-Cell-Rerun aktualisiert werden. |
| 10 | Original erhalten, aber als `LEGACY_REVIEW` markiert | Verwendet das frühere Dropout-Protokoll; nicht als finales Ergebnis zitieren. Die gültige Übergangsdarstellung ist Fig. 09. |
| 11 | Dissertation-Abbildung erhalten; 11b zeigt dieselbe >5%-Transientenmetrik als Six-Cell-Makro, Zellpunkte und HECM/DD-Aufschlüsselung nach SOH-Zustand | Der frühere DD-selektierte C29-Ausschnitt wurde entfernt, weil er die globale HECM-Anfälligkeit nicht erklären konnte. |
| 12 | Dissertation-Abbildung erhalten; 12b verwendet die gepaarte ±Gain-Fehler-Auswertung | Burst dropout ist als `rerun pending` ausgegraut und wird nicht mit veralteten Zahlen dargestellt. |
| 13 | Dissertation-Abbildung erhalten; Radar und Prioritätsprofile in 13b neu berechnet | Nutzt aktuelle Six-Cell-Baseline, gepaarte ±Gain-Fehler, korrigierte Initial-State-Kampagne; Dropout bis zum Rerun aus dem Score ausgeschlossen. |
| 14 | Dissertation-Abbildung erhalten; aktuelle Six-Cell-Fassung als 14b ergänzt | Six-Cell-Zahlen sind die bevorzugte quantitative Ebene. |
| 16, 17, 27–29 | Als besonders hilfreiche Ergänzungen aufgenommen | Dokumentieren Zellabdeckung, statistische Unsicherheit, Testmatrix, Auswertungslogik und gefrorene Evaluationsfenster. |

## Wichtiger Rechenstatus

Die vorhandenen Six-Cell-Abbildungen wurden vor der jüngsten Korrektur der
Online-`Q_c`-Rekonstruktion und des Burst-Dropout-Fensters erzeugt. Für reine
Sichtung und Layoutauswahl sind sie geeignet. Numerische Dropout-Aussagen sowie
davon abhängige Cross-Scenario- und Decision-Abbildungen gelten erst nach einem
vollständigen korrigierten Six-Cell-Rerun als final.

Erzeugung: `python figures/build_all_cells_collection.py`
