# Verteidigungsvorbereitung – aktualisierter LaTeX-Build

Stand: 23.09.2026. Die vorhandenen Markdown-Quellen wurden weitergeführt. Die Dissertation selbst wird nicht verändert.

## Inhalt und Änderungen

- Englische Fachbegriffe, weiterhin deutsche Erklärungen.
- F075: verständliche Erklärung von Channel-Auswahl, mitentfernten Verbindungen und Grenzen des Gewichtsscores.
- Alle 114 Fachfragen, 12 Rechenübungen, 67 Abbildungen und 28 Tabellenhinweise erhalten.
- Modellübersicht und englischer Begriffsschlüssel ergänzt.
- Einheitlicher Fließtext (Arial, 11 pt); Tabellen mit einheitlich 9,5 pt. Abbildungen bleiben Originalseiten.

## Bearbeiten und bauen

Die drei Markdown-Dateien sind die gepflegten Quellen für den Hauptteil. Die Modellübersicht und das Glossar stehen in `Modelluebersicht_und_Begriffe.tex`; das Layout in `preamble.tex`.

`python build_latex.py` erzeugt `Fragenkatalog.tex` daraus. Diese vollständige LaTeX-Datei kann auch direkt bearbeitet werden; ein erneuter Generatorlauf überschreibt solche direkten Änderungen.

Mit `python build_latex.py --compile` wird zusätzlich dreimal XeLaTeX ausgeführt, um Inhaltsverzeichnis und Seitenzahlen aufzulösen. Voraussetzungen: Python 3, XeLaTeX, Arial und die in der Präambel genannten üblichen LaTeX-Pakete. Der Generator braucht keine zusätzlichen Python-Pakete.

Alternativ die vorhandene `Fragenkatalog.tex` mit XeLaTeX und Jobname `Dissertation_Fragenkatalog_und_Lernskript` kompilieren (für aktualisierte Verweise mindestens zweimal).

Die Ausgabe heißt weiterhin `Dissertation_Fragenkatalog_und_Lernskript.pdf`.

`figures/thesis-page-*.pdf` enthält unveränderte einzelne Vektorseiten der Dissertation. Ihre Quelle ist durch `figure_inventory.json` dokumentiert. Diese Assets gehören zum LaTeX-Projekt.

Der bisherige `build_guide.py` bleibt als älterer ReportLab-Build erhalten; für das neue Layout bitte `build_latex.py` verwenden. Die bisherigen Dateien werden vor dem Ersetzen im Unterordner `backup_vor_latex_20260923` gesichert.

Die Überarbeitung ist eine sprachliche und typografische Pflege, keine erneute Validierung sämtlicher Studienergebnisse. Bereits gekennzeichnete offene Punkte bleiben offen. Es wurde nichts committed oder nach GitHub gepusht.
