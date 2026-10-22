# Verteidigungsvorbereitung

Separates Lernmaterial zur Dissertation. Originaltext, Originalbilder und `../main.pdf`
werden nicht geaendert.

## Hauptdatei

`Dissertation_Fragenkatalog_und_Lernskript.pdf`

Enthaelt 114 Fachfragen mit Antwortvorschlaegen, 12 geloeste Rechenuebungen,
10 kritische Pruefpunkte, einen Lern-/Rechercheplan, einen Atlas aller 67
nummerierten Abbildungen und Hinweise zu allen 28 nummerierten Tabellen.
Das Inhaltsverzeichnis und PDF-Lesezeichen erleichtern die Navigation.

## Bearbeitbare Quellen

- `Fragen_und_Antworten.md`: Fragen F001-F100, offene Punkte, Uebungen und Lernplan.
- `Abbildungsatlas.md`: Einzelinterpretationen mit Originalseiten aus der PDF.
- `Tabellen_und_Vertiefung.md`: Fragen F101-F114 und Tabellenbegleitung.
- `figure_inventory.json`: Abbildungsnummern, PDF-Seiten und SHA-256 der Original-PDF.
- `build_guide.py`: reproduzierbarer PDF-Build.

Die Interpretation basiert auf dem gelesenen Manuskript und seiner Quellenstruktur.
Es wurden keine Simulationen, Modelltrainings oder Hardwaremessungen wiederholt.
Nicht nachgewiesene Ursachen sind als Hypothesen oder offene Fragen gekennzeichnet.
Das Dokument ist keine Vorhersage persoenlicher Fragen der Pruefer und kein
vollstaendiges Audit aller experimentellen Rohdaten.

## Build auf diesem HPC

Python-Abhaengigkeiten: `reportlab`, `pymupdf`, `Pillow`.
DejaVu-Schriften werden aus `/usr/share/fonts/truetype/dejavu` verwendet.
Im vorhandenen Conda-Setup benoetigt Pillow die zugehoerige C++-Bibliothek:

```bash
env LD_LIBRARY_PATH=/home/florianr/anaconda3/lib python build_guide.py
```

Der Build erzeugt unter `assets/` ausschliesslich abgeleitete Seitenkopien und
Kontaktboegen. Sie koennen aus der unveraenderten Dissertation neu erzeugt werden.
Nach Aenderung der Dissertation diese Assets neu generieren, nicht alte Seitenkopien
mit einer neuen PDF mischen. Der Generator erkennt eine geaenderte Quell-PDF.

Stand: 2026-09-16, nach Fast-forward von GitHub auf `2d8fdc9`.
Neue Lernunterlagen wurden nicht automatisch committed oder gepusht.
