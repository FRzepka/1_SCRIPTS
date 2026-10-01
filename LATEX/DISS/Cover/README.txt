DISSCOVER - FLORIAN RZEPKA

Cover.tex              Editierbare LaTeX-Datei (pdfLaTeX).
Cover.pdf              Eine A4-Seite, schwarze Schrift auf weissem Grund.
TU_BERLIN_Logo_Lang_RGB_SR_rot.svg  Originaldatei vom TU-Server, unveraendert.
TU_BERLIN_Logo_Lang_schwarz.svg     Schwarze Version mit identischen Pfaden.
TU_BERLIN_Logo_Lang_schwarz.pdf     Vektor-PDF fuer das LaTeX-Cover.

Kompilieren im Cover-Ordner:
pdflatex -interaction=nonstopmode -halt-on-error Cover.tex

Schriftgroessen, Texte, Zeilenumbrueche und Datum stehen in Cover.tex.
Logo: 96 mm breit. Dissertation: 56 pt (etwa 96 mm Schriftzugbreite).
Haupttitel: 20 pt. Untertitel: 15 pt. Name: 20 pt.
Submitted by und Datum: 14 pt. Trennlinie: 40 mm.
Serifenschrift: TeX Gyre Termes (Times). Datum: 29. September 2026.
Die CoverAt-Angaben positionieren die Elemente in mm ab Blattoberkante.
Nach Aenderungen zweimal kompilieren (TikZ-Seitenpositionierung).
Fuer den Druck in Originalgroesse (100 Prozent) ausgeben.

Logoquelle (abgerufen am 29. September 2026):
https://svn.vsp.tu-berlin.de/repos/public-svn/ueber_uns/logo/TU_BERLIN_Logo_Lang_RGB_SR_rot.svg

Neue Langversion passend zum TU-Styleguide (02/2023), Seiten 7, 9 und 10.
Zeichen, Buchstabenformen und Abstaende wurden aus der Original-SVG
uebernommen. Fuer den Schwarz-Weiss-Druck ist ausschliesslich die Farbe
auf Schwarz gesetzt. Die PDF beschneidet nur den aeusseren Leerraum;
das Cover stellt den Schutzabstand um das Logo bereit.

Die alten Dateien TU_Logo_schwarz.pdf und TU_Zeichen_schwarz.pdf werden
vom Cover nicht mehr verwendet.

BEVORZUGTE GESTALTUNG - 29. SEPTEMBER 2026
Der Nutzer findet die Fassung mit dem originalen TU-Logo bereits sehr gut.
Diese Basis ist als Cover_TU_Originallogo.tex und Cover_TU_Originallogo.pdf
gesichert. Originales Logo, Schriftart und Anordnung sind die bevorzugte
Ausgangsbasis fuer weitere Anpassungen.

Aktueller Wunsch: Nur "Dissertation" vergroessern, bis der Schriftzug
ungefaehr so breit ist wie das 96 mm breite Logo. Dafuer wurde die
Ueberschrift von 44 auf 56 pt gesetzt. Alle anderen Elemente behalten
ihre Groessen und Positionen.
