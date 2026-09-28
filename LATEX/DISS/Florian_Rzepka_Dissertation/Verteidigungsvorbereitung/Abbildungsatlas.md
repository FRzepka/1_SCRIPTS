# 13. Abbildungsatlas: Lesen, erklären, hinterfragen

Dieser Atlas folgt den 67 tatsächlich nummerierten Abbildungen der angegebenen PDF. Originalseiten werden ausschließlich als separate Kopien in das Lernskript eingebettet. Die Abbildungen der Dissertation werden nicht verändert. Enthält eine Seite zwei Bilder, ist jeweils die bezeichnete Nummer gemeint. Die Seitenausschnitte zeigen bewusst auch die Caption, damit Unterschiede zwischen Grafik und Text erkennbar bleiben.

**Antwortschema für jede Abbildung:** Was ist auf den Achsen? Was sind Beobachtungseinheit und Zeitraum? Was wurde verändert und was konstant gehalten? Was zeigt das Ergebnis? Welche Erklärung ist belegt und welche nur plausibel? Was kann diese Grafik nicht beantworten? Bei einer schematischen Abbildung entfällt die empirische Interpretation, ihre Bausteine und Annahmen müssen stattdessen erklärt werden.

**Nummerierungsfalle:** Beispielsweise ist Figure_09 im Ergebnisordner hier Abbildung 6.9. Die ADC-Grafik heißt im Quellenordner Figure_16, aber im Dissertation-PDF Abbildung 6.14. Im Gespräch immer die Nummer der tatsächlich gezeigten Folie beziehungsweise PDF verwenden.

# Abbildung 2.1: Anforderungsrahmen
@figure 2.1

**Frage:** Warum steht das BMS im Zentrum und weshalb sind die drei Bereiche gleichberechtigt?

**Antwort:** Der Rahmen verknüpft Schätzqualität, Hardwaregrenzen und die Umsetzungs-/Wartungskette. Die Darstellung ist eine konzeptionelle Ordnung, keine gemessene Gleichgewichtung. Ein fehlender Sicherheits- oder Zeitnachweis lässt sich nicht durch bessere mittlere Accuracy ausgleichen. **Vertiefung:** Zeige für jeden Bereich eine konkrete Ergebnisabbildung der Dissertation. **Grenze:** Das Schaubild weist noch keine Vollständigkeit aller Anforderungen eines Serien-BMS nach. Siehe F005 und F006.

# Abbildung 3.1: Schätzerfamilien
@figure 3.1

**Frage:** Warum sind hybride Modelle keine völlig disjunkte vierte Physik?

**Antwort:** Hybrid bezeichnet Kombinationen von Integrations-, Modell- und Lernkomponenten. HDM verändert den Kapazitätspfad des Coulomb Counting, HECM verbindet einen Observer mit geschätztem SOH, DD verwendet ebenfalls physikalisch motivierte Features. Die Familien sind konzeptionelle Orientierung. **Grenze:** Aus den Kästen darf man nicht ableiten, dass alle Verfahren innerhalb einer Familie denselben Informationsgehalt oder dieselbe Güte hätten. Siehe F004 und F037.

# Abbildung 3.2: MLP
@figure 3.2

**Frage:** Was repräsentiert eine Verbindung und wie berechnet sich die nächste Schicht?

**Antwort:** Jede Verbindung ist ein Gewicht. Ein Neuron summiert gewichtete Eingänge plus Bias und wendet eine Activation Function an. Die ganze Schicht berechnet a = f(Wx+b). Training verändert W und b, nicht die aktuelle Eingangsmessung. **Vertiefung:** Für n Eingänge und m Ausgänge gibt es nm Gewichte plus m Biasparameter. **Grenze:** Die gezeichnete kleine Knotenzahl ist schematisch und keine Größenangabe des finalen SOH-Modells. Siehe F030.

# Abbildung 3.3: LSTM-Zelle
@figure 3.3

**Frage:** Warum gibt es zwei horizontale Zustandswege?

**Antwort:** c speichert den Cell State, h die nach außen gegebene rekurrente Repräsentation. Forget-Gate und Input-Gate bestimmen alte und neue Information im Cell State, Output-Gate steuert die sichtbare Ausgabe. Die obere additive Verbindung erleichtert lange Abhängigkeiten. **Grenze:** Ein offenes Gate garantiert weder physikalische Interpretierbarkeit noch beliebig langes korrektes Gedächtnis. Auf Nachfrage die Dimensionen sämtlicher Gates angeben: jeweils H Komponenten. Siehe F071 und F072.

# Abbildung 3.4: GRU-Zelle
@figure 3.4

**Frage:** Welcher Zweig wird hier mit z gewichtet und ist das überall gleich?

**Antwort:** Die Dissertation verwendet die Kandidatengewicht-Konvention. Andere Implementierungen gewichten mit z den bisherigen Zustand. Das ist nur bei konsistenter Umdefinition äquivalent. Reset-before und reset-after beeinflussen ebenfalls die konkrete Rechnung. **Grenze:** Nicht allein anhand des Diagramms einen Framework-Export bestätigen. Das endgültige C-Modell muss gegen exakt dessen Gateordnung, Biasbehandlung und Activations geprüft werden. Siehe F071.

# Abbildung 3.5: Pruning-Granularität
@figure 3.5

**Frage:** Warum sparen die roten Nullen im mittleren Bild nicht zwingend Laufzeit?

**Antwort:** Unstrukturierte Nullen behalten die ursprünglichen Matrixdimensionen. Eine dichte Schleife multipliziert sie weiterhin, sofern kein spezieller Sparse-Pfad verwendet wird. Rechts entfernt Structured Pruning ganze Zeilen und passende Spalten, sodass wirklich kleinere dichte Tensoren entstehen. **Grenze:** Die Zeichnung illustriert Speicherrepräsentation, keine gemessenen Zeitverhältnisse. Sparsity-Anteil und effektiver Speed-up sind getrennte Größen. Siehe F074.

# Abbildung 3.6: Abhängige Strukturen
@figure 3.6

**Frage:** Warum ist unabhängig beschnittenes Pruning vor einer Addition ungültig?

**Antwort:** Addierte Zweige müssen dieselben Dimensionen und die beabsichtigte semantische Zuordnung besitzen. Werden verschiedene Kanäle entfernt, können Dimensionen oder Bedeutungen nicht mehr zusammenpassen. Gemeinsame Auswahlregeln erhalten die Struktur. **Übertragung:** Im LSTM betrifft derselbe Hidden Channel mehrere Gates, recurrent columns, Zustände und MLP-Eingänge. Alle müssen zusammen angepasst werden. Siehe F075.

# Abbildung 3.7: Quantization Points und Gitter
@figure 3.7

@latex \clearpage
**Frage:** Was zeigt Abbildung 3.7, und welche Teile haben wir tatsächlich quantisiert?

**Kurzantwort:** Die Abbildung zeigt allgemeine Möglichkeiten der Quantization. Unsere Umsetzung verwendet Weight-only Quantization: Die beiden LSTM-Gewichtsmatrizen Input-to-Hidden und Hidden-to-Hidden werden als INT8 gespeichert. Beim Rechnen werden die Gewichtswerte in FP32 umgewandelt und ihre Scales berücksichtigt. Die MLP-Gewichte, Biases, States und Activations bleiben FP32. Es wird also nicht das gesamte Netz auf INT8 umgestellt.

**Welche Gewichte genau?** Sowohl W_ih (Input-to-Hidden) als auch W_hh (Hidden-to-Hidden), jeweils für alle vier LSTM-Transformationen. Der frühere Ausdruck „recurrent weights“ war hier missverständlich: Gemeint sind die LSTM-Gewichtsmatrizen insgesamt, nicht nur W_hh. „Zeilenweise symmetrisch“ bedeutet: Jede Matrixzeile erhält einen eigenen Scale; verwendet werden die Integer-Codes von -127 bis +127 mit Zero Point 0.

**Warum nicht alle Gewichte?** Die untersuchte Umsetzung konzentriert sich auf den großen LSTM-Gewichtsblock. Der MLP-Head bleibt FP32. Auch seine Gewichte zu quantisieren wäre ein zusätzlicher Optimierungsschritt, dessen Genauigkeit und Laufzeit separat geprüft werden müssten. Siehe F079 bis F082.

**Was unterscheidet Weights, Activations und Akkumulatoren?** Weights sind gespeicherte Modellparameter. Activations sind Zwischenwerte, die beim Verarbeiten der Eingaben entstehen. Ein Akkumulator sammelt die Produkte einer gewichteten Summe. Diese Summe kann einen größeren Wertebereich benötigen: Schon 100 mal 100 ergibt 10 000 und passt nicht in INT8. Integer Kernels verwenden deshalb häufig INT32 für die Summen. In unserer Umsetzung erfolgt die Rechnung weiterhin in FP32. INT8-Gewichte bedeuten also nicht automatisch INT8-Zwischenwerte oder INT8-Summen.

**Was ist der Zero Point?** Er ist die gespeicherte Ganzzahl z, die den realen Wert null repräsentiert. Im oberen, symmetrischen Beispiel gilt z = 0: Integer-Code 0 bedeutet realer Wert 0. Bei einer asymmetrischen Abbildung könnte beispielsweise Code 100 den realen Wert 0 darstellen. Es geht nicht bloß um positive und negative Werte; diese gibt es auch bei symmetrischer Quantization.

**Wie liest man die drei Gitter rechts?** Oben stehen die Integer-Codes q, darunter die zugehörigen realen Werte x. Das obere Gitter ist symmetrisch um null. Das mittlere nutzt UINT8 von 0 bis 255 für nichtnegative Werte, ebenfalls mit Zero Point 0. Das untere nutzt einen verschobenen Zero Point für einen asymmetrischen Wertebereich. „Gitter“ meint die diskreten darstellbaren Zahlenstufen; Werte dazwischen werden gerundet.

**Was bedeutet Full Integer?** Die Netzoperationen werden mit geeigneten Integer- beziehungsweise Fixed-Point-Verfahren ausgeführt, statt die Gewichte zum Rechnen wieder nach FP32 zu konvertieren. Dabei müssen nicht alle Zwischenwerte INT8 sein: INT32-Summen oder breitere States sind möglich. Dafür braucht man passende Kernels und Scales auch für die Zwischenwerte. Das ist eine andere Umsetzung als unsere Weight-only Quantization. Abbildung 3.7 beschreibt die Grundlagen und belegt keine Full-Integer-Ausführung unserer Firmware.

# Abbildung 4.1: NMC-Alterung über Zyklen und Zeit
@figure 4.1

**Frage:** Warum verändern sich scheinbare Abstände zwischen den oberen und unteren Kurven?

**Antwort:** Die obere Achse normiert auf Durchsatzzyklen, die untere auf Kalenderzeit. Zellen mit anderen Raten und DOD erreichen vergleichbaren Durchsatz in anderer Zeit. Daher erscheinen Alterungsunterschiede je nach Koordinate anders. Zwei Zellen pro Szenario zeigen zudem Replikatstreuung. **Grenze:** Aus den Kurven allein keine separate Aktivierungsenergie oder eindeutige Dominanz eines Alterungsmechanismus schätzen. Die Temperaturwirkung ist sichtbar, andere Faktoren wirken gekoppelt. Siehe F009 und F010.

# Abbildung 4.2: LFP-DoE-Würfel
@figure 4.2

@latex \clearpage
@latex {\large\bfseries\color{accent}Abbildung 4.2 -- DoE kurz erklärt}\par
**Frage:** Warum verwendet man ein Design of Experiments (DoE), und was zeigt der Würfel?

**Zweck:** DoE ist statistische Versuchsplanung. Lade-C-Rate, Entlade-C-Rate und DoD werden gezielt kombiniert, um ihre Einflüsse auf die Alterung mit begrenzter Versuchskapazität zu untersuchen. Ein Designpunkt ist eine Kombination dieser drei Einstellungen.

**Full Factorial Design:** Bei drei Faktoren mit je zwei Stufen (niedrig/hoch) werden alle 2 x 2 x 2 = 8 Kombinationen getestet. Sie bilden die acht grünen Würfelecken. Man kann damit Haupteffekte untersuchen, etwa den durchschnittlichen Einfluss einer höheren Lade-C-Rate, und Wechselwirkungen: Wirkt schnelles Laden bei hoher DoD anders als bei niedriger DoD?

**Central Composite Design (CCD):** Die gezeichnete Anordnung ergänzt die acht Eckpunkte um einen roten Mittelpunkt und sechs blaue Axialpunkte. An jedem Axialpunkt wird ein Faktor weiter nach oben oder unten variiert, während die anderen auf mittlerer Einstellung bleiben. Diese zusätzlichen Punkte ermöglichen eine quadratische Modellierung und damit die Untersuchung von Krümmung, etwa eines überproportional steigenden Kapazitätsverlusts bei hoher Belastung.

**Nutzen für unsere Modelle:** Die Kampagne liefert Messdaten aus systematisch unterschiedlichen Belastungs- und Alterungsverläufen für Training und Bewertung der SOC-/SOH-Schätzer. Wiederholungen derselben Einstellung würden zusätzlich die Zellstreuung erfassen. Ob Einflüsse tatsächlich nachgewiesen sind, ergibt erst die statistische Auswertung, nicht der Würfel allein.

**Einordnung der Unterlagen:** Die Grafik zeigt 15 unterschiedliche Positionen; der Begleittext nennt dagegen acht Betriebspunkte mit Wiederholungen. Die konkrete Zellzuordnung muss am Versuchsplan abgeglichen werden. Die frühere pauschale Erklärung, die 15 Zellen seien lediglich Wiederholungen von acht Punkten, ist damit nicht belegt. Siehe F025.

# Abbildung 4.3: Struktur eines LFP-Verlaufs
@figure 4.3

**Frage:** Warum wechseln regelmäßige Zyklen, Diagnoseblöcke und dynamische Abschnitte?

**Antwort:** Die Kampagne kombiniert beschleunigte zyklische Alterung mit Check-ups und anwendungsbezogenen Lastprofilen. Check-ups liefern vergleichbare Referenzinformation, dynamische Abschnitte beanspruchen die Schätzer anders als konstante Ströme. **Grenze:** Ein dargestelltes PV-Heimspeicherprofil repräsentiert nicht jede Microgrid-Nutzung. Außerdem muss erläutert werden, welche Diagnosephasen im jeweiligen Modellinput enthalten sind und welche für Labels verwendet werden. Siehe F020.

# Abbildung 4.4: LFP-SOH über die Kampagne
@figure 4.4

**Frage:** Warum haben einige Kurven starke Einbrüche und lokale Erholung, und welche ist C29?

**Antwort:** Die Kurven verbinden diskrete Kapazitätsanker über die Versuchszeit. Unterschiedliche Lastbedingungen und Eigenerwärmung gehen mit unterschiedlichen Alterungsverläufen einher. Lokale Anstiege sind gemessene Kapazitätsvariation, nicht automatisch irreversible Regeneration. **Konkreter offener Punkt:** Im vorhandenen PDF nennt die Legende Betriebsbedingungen, aber keine Zellnamen oder eindeutig ausgewiesenen Splits. C29 darf deshalb nicht allein aus einer vermuteten Farbe identifiziert werden. Für diese Zuordnung Tabelle 4.3 benutzen. Siehe P01.

# Abbildung 4.5: Holdout-Abdeckung
@figure 4.5

**Frage:** Weshalb bleibt eine Zelle ausschließlich fresh, und kann High eine Statistikgruppe sein?

**Antwort:** C27 deckt im vorhandenen Zeitbereich nur SOH 0,930 bis 1 ab. Es gibt keine beobachteten mid-life-/aged-Fenster dieser Zelle. High besteht allein aus C29 und ist deskriptiv. **Vertiefung:** Temperaturbereich, P95-C-Rate und SOH sind verschiedene Coverage-Größen. Ein breiter Balken bedeutet nicht viele unabhängige Wiederholungen. **Grenze:** Keine belastbare zwischenzellige Varianz für High und keine rein kausale Lastklassenwirkung. Siehe F022 und F023.

# Abbildung 4.6: BMS-Platine
@figure 4.6

**Frage:** Welche sichtbaren Hardwarebereiche sind für Ihre Beiträge entscheidend?

**Antwort:** Der Text ordnet Sensing/Balancing, Strommess- und Schutzpfad sowie Controller/Kommunikation als Funktionsdomänen ein. Die physische Trennung von Leistungs- und Signaldomäne unterstützt Messqualität und EMV. **Grenze:** Bauteilwerte, Kalibriergenauigkeit, reale Leiterplattenbestückung und Teststatus nicht aus dem Rendering erraten. Schaltplan, Stückliste und Messprotokoll bereithalten. Eine Platinenabbildung ist kein Funktions- oder Sicherheitsnachweis. Siehe F099 und P09.

# Abbildung 4.7: BMS-Systemblock
@figure 4.7

**Frage:** Wo verlaufen Messung, Schätzung und Schutz, und was passiert bei Ausfall der KI?

**Antwort:** Messfront-end und Stromsensor liefern Daten an die Controllerdomäne. Die konzeptionelle M4/M7-Trennung hält konventionelle BMS-Aufgaben von austauschbaren Schätzern getrennt. Eine reale Ausfallantwort benötigt aber Watchdog, sichere Grenzprüfung, Plausibilität und definierte Fallbacks. **Grenze:** Das Blockbild allein belegt nicht, dass alle diese Maßnahmen fertig implementiert und getestet sind. Die isolierten H753-Benchmarks nicht als vollständigen H755-Systemtest ausgeben. Siehe F099.

# Abbildung 5.1: Korrelationsmatrix
@figure 5.1

**Frage:** Warum sind negative und positive Zusammenhänge beide rot?

**Antwort:** Farbe stellt hier den Betrag dar. Die gedruckten Zahlen enthalten Vorzeichen und exakten Koeffizienten. Es ist eine symmetrische Matrix, daher genügt eine Dreieckshälfte. **Vertiefung:** Durchsatz und SOH haben gemeinsame zeitliche Entwicklung. Eine große Pearson-Korrelation beweist weder eine Ursache noch robuste Generalisierung auf andere Lastprofile. **Prüfübung:** Eine konkrete Zelle der Matrix mit Vorzeichen, Größenordnung und möglicher Confounding-Variable erklären. Siehe F028.

# Abbildung 5.2: Lag-Konstruktion
@figure 5.2

@latex \clearpage
@latex {\large\bfseries\color{accent}Abbildung 5.2 -- Resampling und Aggregation}\par
**Frage:** Wie werden die Messdaten für das MLP zeitlich aufbereitet, und welche Information bleibt erhalten?

**Die beiden Zeitraster auseinanderhalten:** Laut Dissertation werden die zunächst je Prozessschritt unterschiedlich abgetasteten NMC-Messungen auf ein Sekundenraster gebracht. Für die MLP-Historie folgt eine gröbere Zeitauflösung. Der Startpunkt der Untersuchung ist 30 Minuten mit 12 Lags; als beste untersuchte Kombination nennt die Arbeit 10 Minuten mit 16 Lags, also 160 Minuten Historie. Das sind zwei verschiedene Verarbeitungsschritte, keine widersprüchlichen Abtastraten.

**Resampling: Welches Zeitraster?** Es legt die neuen Zeitintervalle fest. Im gefundenen NMC-Vorverarbeitungsskript wird ein Sekundenraster erzeugt und interpoliert. Resampling erzeugt dabei keine neuen unabhängigen Messungen. Im späteren Lag-Code wird auf die übergebene gröbere Intervalllänge resampled; für die in der Arbeit genannte 10-Minuten-Konfiguration sind das 600 Sekunden pro Intervall.

**Aggregation: Wie wird ein Intervall zusammengefasst?** Der gefundene Lag-Code verwendet den arithmetischen Mittelwert pro Zeitintervall und Messspalte, anschließend lineare Interpolation fehlender Werte. Auf einem lückenlosen Sekundenraster enthält ein 10-Minuten-Intervall beispielsweise 600 Werte, aus denen ein Mittelwert wird. Es wird also nicht lediglich jeder 600. Messwert ausgewählt. Beispiel: 300 Sekunden mit 3,6 V und 300 Sekunden mit 3,8 V ergeben 3,7 V als Intervallmittel. Derselbe Mittelwert kann aus unterschiedlichen zeitlichen Verläufen entstehen.

**Was bleibt erhalten, was geht verloren?** Erhalten bleiben mittlere Signalniveaus, die Reihenfolge der Zeitintervalle, ihr fester Abstand und Veränderungen zwischen ihnen. Die genaue Reihenfolge innerhalb eines Intervalls, kurze Spitzen und schnelle Wechsel sind aus dem Mittelwert nicht rekonstruierbar. Interpolation füllt Lücken durch angenommene Zwischenwerte; sie stellt den unbekannten Verlauf nicht wieder her. Das MLP erhält damit eine gröbere zeitliche Darstellung, nicht den ursprünglichen Sekundenverlauf.

**Besonderheit der kumulierten Ströme:** Im gefundenen Lag-Code wird zuerst der Strom pro Intervall gemittelt. Danach werden positive und negative Intervallmittel getrennt aufsummiert und mit der Intervalllänge in Ah umgerechnet. Diese Summen werden anschließend ebenfalls verzögert als Features abgelegt. Sie tragen Nutzungsinformation von vor dem Lag-Fenster mit. Lade- und Entladeanteile innerhalb desselben Intervalls können sich allerdings schon beim Mitteln aufheben; ihre getrennte Integration vor dem Mitteln wäre nicht dieselbe Rechnung.

**Warum diese Aufbereitung?** Statt Tausenden Sekundenwerten bekommt das MLP eine kompakte Historie. Das reduziert Eingangsgröße und glättet schnelle Schwankungen. Zu grobe Intervalle verlieren relevante Dynamik; zu feine Intervalle decken bei gleicher Lag-Zahl nur wenig Zeit ab. Deshalb wurden in der Arbeit mehrere Zeitauflösungen und Lag-Längen verglichen.

@latex \clearpage
@latex {\large\bfseries\color{accent}Abbildung 5.2 -- Lag-Fenster und Überlappung}\par
**Frage:** Wie erhält ein MLP ohne Recurrent State eine Historie, wie überlappen die Fenster, und warum?

**Konstruktion im gefundenen Code:** Für jede Featuregröße werden der aktuelle Intervallwert und verschobene Kopien angelegt: t, t-1, t-2 bis t-L. Die Schleife läuft einschließlich L; L = 16 bedeutet hier daher 16 vergangene Werte plus den aktuellen, insgesamt 17 Werte je Feature. Die Kapazität des aktuellen Intervalls dient als Zielwert. Im gezeigten Trainingsaufruf werden Kapazitätsspalten aus den Inputs entfernt. Die Arbeit beschreibt das Ziel als aktuellen SOH, nicht als Zukunftsprognose.

**Konkrete Rechnung mit der in der Arbeit genannten Einstellung:** Bei 10 Minuten Abstand und L = 16 liegen zwischen ältestem und aktuellem Zeitindex 160 Minuten. Für Spannung, Temperatur sowie positive und negative kumulierte Ladung wären es nach dieser Code-Konvention 4 x 17 = 68 Eingabewerte. Das ist eine Ableitung aus der gefundenen Lag-Klasse, keine Bestätigung der Eingangsgröße des final trainierten Modells. 16 Werte einschließlich des aktuellen hätten dagegen nur 15 Abstände, also 150 Minuten zwischen den Zeitindizes. Bei Intervallmitteln muss zusätzlich zwischen Zeitindex-Abstand und der gesamten Breite der enthaltenen Intervalle unterschieden werden.

**Überlappung bei einem Schritt Vorschub:** Jede nächste Zeile wird ein neues Beispiel; ein zusätzlicher größerer Stride ist in dieser Lag-Klasse nicht eingebaut. Bei 10-Minuten-Auflösung würde das nächste Fenster daher zehn Minuten später beginnen und enden:

@latex \[\begin{aligned}A &: [0,10,20,\ldots,150,160]\ \text{min},\\B &: [10,20,30,\ldots,160,170]\ \text{min}.\end{aligned}\]

**Was wird wiederverwendet?** Beide Fenster enthalten die 16 Zeitindizes von Minute 10 bis Minute 160. Ein alter Eintrag fällt heraus, ein neuer kommt hinzu. Nach dieser Code-Konvention überlappen somit 16 von 17 Einträgen, etwa 94 Prozent. Die Zeitangaben sind schematische Rasterindizes; aus der linken Beschriftung eines gemittelten Intervalls darf nicht abgeleitet werden, dass sein Mittelwert schon zu Intervallbeginn verfügbar war.

**Warum überlappen lassen?** Nach jedem neuen Intervall kann eine Schätzung mit aktualisierter Historie entstehen, ohne auf einen komplett neuen, nicht überlappenden Datenblock zu warten. Im Training werden unterschiedliche Endzeitpunkte und Ausschnitte eines Verlaufs nutzbar. Die zeitliche Reihenfolge steckt in der festen Reihenfolge der Input-Spalten. Das MLP selbst behält zwischen zwei Aufrufen keinen Hidden State.

**Grenze bei Training und Validation:** Überlappende Fenster sind stark abhängig und keine neuen unabhängigen Versuche. Im gefundenen Projekt werden die Lag-Zeilen vor dem Training gemischt; der gezeigte Aufruf reserviert anschließend 20 Prozent für Validation. Dadurch können stark ähnliche Fenster auf beiden Seiten liegen. Ein separater Test auf vollständig zurückgehaltenen Zellen ist davon zu unterscheiden. Eine rein zeitliche Trennung müsste zusätzlich die Fensterüberlappung an der Grenze berücksichtigen.

**Quellenstatus:** Dissertation, Kapitel 4 (Sekundenraster) und 5 (Zeitauflösung/Lags); NMC-Projekt mg_farm_main_project, Klassen LagDataset und LagDataCollection sowie main.py; Vorverarbeitung resample_BX_SOH_Combined_to_second_v2.py. Die gefundene main.py nutzt einen älteren Aufruf mit 30 Minuten und unterschiedlichen Lag-Längen. Der finale 10-Minuten-/16-Lag-Trainingslauf ist damit nicht vollständig reproduziert. Die Code-Details sind deshalb ausdrücklich als Befund dieses Projektstands gekennzeichnet. Siehe F026 und F029.

# Abbildung 5.3: Trainings- und Evaluationsworkflow
@figure 5.3

**Frage:** An welcher Stelle darf das Testset Einfluss nehmen?

**Antwort:** Auf die finale Auswertung, nicht auf Scalerfitting, Hyperparameterwahl oder frühes Stoppen. Datenaufbereitung mit gelernten Parametern muss innerhalb der Entwicklungsgrenze bleiben. **Kritische Nachfrage:** Ist das im gezeichneten Ablauf oder in der Implementierung garantiert? Das Schema beschreibt Absicht, die tatsächliche Trennung muss der Code belegen. **Offen:** Zufällige überlappende Validation-Sequenzen und historischer Gebrauch der Lag-MAE-Matrix prüfen. Siehe F031 und F032.

# Abbildung 5.4: SOH-MLP
@figure 5.4

**Frage:** Warum sieht das Modell klein aus, während die Tabelle viele Schichten zeigt?

**Antwort:** Das Bild erklärt Feature Groups und Vollvernetzung schematisch. Die tatsächlich gewählten Breiten stehen in Tabelle 5.1. Die Darstellung ist kein maßstabsgetreues Netz mit der richtigen Neuronenzahl. **Grenze:** Einfach meint feedforward und nachvollziehbare Featurebildung, nicht zwingend wenige Parameter. Den Unterschied zur Eingangssuche mit acht Neuronen erklären. Die bedingte Parameterrechnung in F030 bereithalten.

# Abbildung 5.5: Historien-MAE-Matrix
@figure 5.5

@latex \clearpage
@latex {\Large\bfseries\color{blue} Abbildung 5.5 -- Einordnung und Prüfungsfragen}\par\medskip

**Was zeigt die Abbildung?** Drei MAE-Karten für Zelle 1, 7 und 9. Jedes Feld steht für eine Kombination aus Lag-Einstellung (8, 12, 16 oder 20) und Zeitauflösung (1, 10, 30 oder 60 Minuten). Blau steht auf der jeweiligen Skala für kleinere, Rot für größere Fehler. Die Kombination aus 16 Lag-Schritten und zehn Minuten liegt bei allen drei Zellen in einem günstigen Bereich; 60 Minuten sind häufig ungünstiger.

**Warum ist eine feinere Zeitauflösung nicht automatisch besser?** Bei gleicher Lag-Anzahl reicht die Historie weniger weit zurück. Nach der Definition der Arbeit entsprechen 16 Schritte mit einer Minute Abstand 16 Minuten Historie, mit zehn Minuten Abstand dagegen 160 Minuten. Feinere Auflösung bewahrt mehr zeitliche Details, kann aber zu wenig vom Lade-/Entladeverlauf abdecken. Gröbere Mittelung deckt mehr Zeit ab, kann jedoch kurze Änderungen verwischen. Es besteht ein Kompromiss zwischen Detailtiefe und zeitlicher Reichweite.

**Warum helfen mehr Lags nicht immer?** Bei gleicher Zeitauflösung verlängern sie die Historie und vergrößern den Eingabevektor. Ältere Werte können zusätzliche Information liefern, aber auch weitgehend redundant sein. Mehr Eingaben garantieren daher keinen kleineren Vorhersagefehler. Redundanz ist eine mögliche Erklärung, kein durch die Fehlerkarte allein nachgewiesener Mechanismus.

**Kann man die Farben zwischen den Zellen direkt vergleichen?** Nein: Jede Teilgrafik hat eine eigene Farbskala. Ein blaues Feld bei Zelle 9 kann einen deutlich höheren MAE darstellen als ein blaues Feld bei Zelle 1. Für den Vergleich sind die Zahlen der jeweiligen Farbleiste maßgeblich. Der MAE wird hier in SOH-Prozentpunkten angegeben; Zelle 9 weist insgesamt deutlich höhere Fehler auf.

**Sind zehn Minuten und 16 Lags allgemein optimal?** Nein. Sie sind eine günstige Kombination innerhalb des untersuchten diskreten Rasters und der dargestellten Zellen. Daraus folgt weder ein universelles Optimum noch, dass benachbarte, nicht getestete Einstellungen schlechter wären. Für andere Betriebsprofile oder Datensätze müsste die Wahl erneut bewertet werden. Die Grafik allein zeigt außerdem keine Streuung über wiederholte Trainingsläufe.

**Welche Rolle dürfen die Testzellen bei dieser Wahl spielen?** Hyperparameter sollten anhand der Validation gewählt und danach auf zurückgehaltenen Testzellen bewertet werden. Zeigt die Grafik nur eine nachträgliche Analyse eines bereits festgelegten Modells, ist das von einer Auswahl anhand der Testfehler zu unterscheiden. Werden die Testzellen zur Auswahl der Lag-Einstellung benutzt, sind sie keine von dieser Auswahl unabhängige abschließende Bewertung mehr. Welcher Ablauf für diese Abbildung tatsächlich verwendet wurde, ist anhand der Grafik allein nicht geklärt. Siehe F029 und F031.

**Hinweis zur X-Achse:** „Number of sequences“ ist missverständlich, weil hier die Historienlänge und nicht die Anzahl verschiedener Trainingsfenster variiert wird. „Number of lag steps“ wäre eindeutiger. Ob die dargestellte Einstellung den aktuellen Wert mitzählt, ist für den finalen Lauf noch offen; die inklusive Zählweise des gefundenen älteren Codes darf nicht ungeprüft übertragen werden. Siehe die Erläuterung zu Abbildung 5.2.

# Abbildung 5.6: NMC-SOH-Trajektorien
@figure 5.6

**Frage:** Warum weicht Zelle 9 stärker ab, obwohl die Form teilweise getroffen wird?

**Antwort:** Formtreue allein reicht nicht. Ein systematisches vertikales Verschieben erzeugt trotz ähnlichem Trend erhebliche Residuen. Die U-förmige Referenz ist atypisch und schlechter in Absolutwerten reproduziert. **Grenze:** Der Plot enthält nachträgliche Rolling-Glättung. Die sichtbare Ruhe ist deshalb nicht automatisch rohe Netzqualität. Fensterbreite und kausale beziehungsweise zentrierte Filterdefinition vor der Prüfung nachschlagen. Siehe F033 und F034.

# Abbildung 5.7: NMC-Scatter
@figure 5.7

**Frage:** Was bedeutet Abstand von der Diagonalen und warum geht Zeitinformation verloren?

**Antwort:** Jeder Punkt vergleicht Vorhersage und Referenz. Die Identitätslinie bedeutet perfekte Übereinstimmung. Systematische Abstände zeigen Bias oder Kalibrierfehler. Viele Punkte auf engem Bereich können durch lange Aufenthaltsdauer dort entstehen. **Grenze:** Scatter zeigt nicht, ob Fehler zusammenhängend über Stunden auftreten oder rasch wechseln. Deshalb zusammen mit der Trajektorie lesen. MSE-Einheiten sind Quadrat-Prozentpunkte, MAE-Einheiten Prozentpunkte. Siehe F034.

# Abbildung 6.1: Gemeinsame Benchmarkkette
@figure 6.1

**Frage:** Wo muss eine Störung eingespeist werden, damit der Vergleich fair bleibt?

**Antwort:** Sie muss die tatsächlichen Rohkanäle und alle daraus kausal gebildeten Features beeinflussen. Ein gestörter Strom bei ungestörtem Qc würde DD einen künstlich sauberen Nebenkanal lassen. Die Kette soll diese Inkonsistenz verhindern. **Vertiefung:** Labels bleiben unverändert, da die Zielbatterie nicht durch einen Sensorfehler physikalisch anders betrieben wird. **Grenze:** Ein realer Regelkreis, der auf falsche Schätzungen reagiert, wäre eine andere Versuchsart. Siehe F039 und F040.

# Abbildung 6.2: Störungstaxonomie
@figure 6.2

**Frage:** Warum gehören Gain, Offset, Rauschen, Verfügbarkeit und Initialzustand nicht in dieselbe Fehlerbeschreibung?

**Antwort:** Sie ändern unterschiedliche Mechanismen: Skalierung, Nullpunkt, zufällige Fluktuation, Informationsverlust oder internen Startzustand. Dadurch können sich gleiche nominale Modelle unter den Tests anders ordnen. **Grenze:** Die Taxonomie ist eine relevante Auswahl, kein vollständiger kartesischer Produktraum sämtlicher Sensorfehler. Zusammengesetzte Fehler und zeitabhängige Drift sind nicht umfassend abgedeckt. Die konkrete Stärke steht in der Szenariotabelle. Siehe F050 und F056.

# Abbildung 6.3: Fensterprotokoll
@figure 6.3

**Frage:** Warum ein sauberes Vorfenster, gemeinsamer Scorebeginn und verlängertes Dropoutfenster?

**Antwort:** Das SOH-Netz benötigt kausalen Kontext, die GRU ein gültiges Eingabefenster. Erst danach wird gleichzeitige Leistung verglichen. Dropout braucht ausreichend Nachbeobachtung und eine gleich lange Baseline. **Grenze:** Das ist keine Kaltstartvalidierung ohne Vorgeschichte. Der typische Fensterselektor erfasst nicht jeden seltenen ungünstigen Betriebspunkt. Auf Nachfrage die sechs Selection Features und die Behandlung fehlender Zell-SOH-Kombinationen nennen. Siehe F023, F024 und F042.

# Abbildung 6.4: Nominale Accuracy
@figure 6.4

**Frage:** Ist DD hier wirklich am besten und was bedeuten die Punkte?

**Antwort:** DD hat das kleinste zellgewichtete MAE und RMSE im dargestellten Protokoll. Die Punkte zeigen Zell-/Fensterwerte, die Balken den Makrowert und die Fehlerstriche ein hierarchisches 95-Prozent-Intervall. DD-MAE 0,0258 bedeutet 2,58 Prozentpunkte. **Grenze:** Nicht jede Zelle oder jeder Zeitpunkt muss dieselbe Rangfolge haben. Die korrigierten exakten Tests sind wegen kleiner Zellzahl nicht signifikant. Siehe F047, F063 und F065.

# Abbildung 6.5: Current Gain
@figure 6.5

**Frage:** Warum steigt das obere Diagramm monoton, während eine Einzelkurve gelegentlich näher an der Referenz liegen kann?

**Antwort:** Oben wird je Betrag und Zelle die ungünstigere von zwei Gainrichtungen zusammengefasst. Unten steht eine konkrete Realisierung und ein Ausschnitt einer Zelle. Eine lokale Kompensation ist damit vereinbar. **Grenze:** Die obere Kurve ist keine Messung einer zufälligen positiven Drift und kein allgemeiner Monotoniebeweis. Die +3-Prozent-Eingangskurve illustriert eine Richtung, nicht jede adverse Auswahl. Siehe F048 und F049.

# Abbildung 6.6: Gain über Lebenszeit und Ladeanker
@figure 6.6

@latex \clearpage
@latex {\Large\bfseries Abbildung 6.6 -- Fehleraufbau, Ladeanker und Alterung}\par\medskip

**Was zeigt die Abbildung?** Eine einzelne Hochlast-Testtrajektorie über den gesamten untersuchten Lebenszeitverlauf. (a) zeigt den zeitlichen Zusatzfehler durch den Strom-Gain, über 24 Stunden geglättet. (b) fasst 44 Abschnitte zwischen Vollladungen zusammen; die X-Achse ist der relative Fortschritt zwischen den Ereignissen, nicht der SOC. Die Linien zeigen Mittelwerte, die Balken den Min-Max-Bereich. (c) vergleicht den Zusatzfehler vor und nach Vollladungen. Erst (d) zeigt den absoluten Gesamt-MAE von Baseline und gestörter Berechnung.

**Bedeutet ein höherer Gesamt-MAE, dass sich der Fehler aufkumuliert?** Nein. Auch eine dauerhaft gleich große zusätzliche Abweichung erhöht den MAE. Die Differenz MAE mit Gain minus MAE ohne Gain ist keine fortlaufende Summe aller bisherigen Fehler. Akkumulieren kann die SOC-Abweichung durch die Integration eines fehlerhaften Stromsignals, nicht der Strom selbst.

**Was darf man aus Bild b über Akkumulation und Reset schließen?** Bei einigen Modellen steigt der zusätzliche Fehler zwischen Vollladungen an und nimmt im weiteren Verlauf wieder ab. Das ist mit zeitweisem Fehleraufbau und teilweiser Korrektur vereinbar. Die Abnahme schließt einen vorherigen Aufbau nicht aus. Weder ein vollständiger Reset noch ein unbegrenzt wachsender Fehler oder eine generell vernachlässigbare Akkumulation ist damit nachgewiesen. Auch ein Ladeanker beseitigt den Sensorfehler nicht; dieser wirkt anschließend weiter. Bei HECM ist der mittlere Beitrag nach der Vollladung in (c) sogar größer.

**Beweist der spätere Anstieg, dass ältere Zellen generell schwerer vorherzusagen sind?** Nein. Bild a zeigt den Zusatzfehler gegenüber der jeweiligen ungestörten Berechnung, nicht deren gesamten MAE. Beispiel in SOC-Prozentpunkten: Anfangs Baseline 2 und gestört 3 ergibt Zusatzfehler 1; später Baseline weiterhin 2 und gestört 4 ergibt Zusatzfehler 2. Die Differenz wächst, obwohl die Baseline unverändert bleibt. Alterung kann die Empfindlichkeit mitverändern, aber auch Betriebsbedingungen und verbleibende Fehlerakkumulation kommen infrage. Die Darstellung trennt diese Ursachen nicht und belegt daher keine Hauptursache.

**Kurze Formulierung für die Verteidigung:** Wir beobachten bei einigen Modellen einen zeitweisen Aufbau des zusätzlichen Fehlers und eine anschließende Abnahme. Das ist mit teilweise begrenzter Akkumulation vereinbar, aber kein Nachweis eines vollständigen Resets. Eine Mitwirkung der Alterung am späteren Anstieg bleibt eine Hypothese.

**Einordnung:** HECM hat in diesem einzelnen Hochlast-Lebenszeitlauf den kleinsten zusätzlichen MAE durch den Gain-Fehler. Das widerspricht nicht dem Sechs-Zellen-Fenstervergleich zugunsten DD: Datengrundlage und Gewichtung unterscheiden sich. Siehe F052.

# Abbildung 6.7: Lokales Stromrauschen
@figure 6.7

**Frage:** Warum schwankt DD lokal sichtbar, während die globale Fehleränderung klein bleibt?

**Antwort:** Rohstrom und Ableitung wirken direkt auf DD. Die integrierenden Verfahren glätten wechselnde Fehler stärker im sichtbaren SOC, obwohl jeder verrauschte Sample eingeht. Nullgemittelte lokale Änderungen können im Tagesmittel wenig ausmachen. **Grenze:** Ein Knick ist nicht automatisch ein Fehler im Modell. Resetlogs, Lastwechsel und SOH-Stundenupdates müssen unterschieden werden. Der gezeigte Ausschnitt wird als reset-frei beschrieben. Siehe F054 und F055.

# Abbildung 6.8: Rauschen über sechs Zellen
@figure 6.8

**Frage:** Darf man bei einem höheren Mittelbalken sicher größere Empfindlichkeit behaupten?

**Antwort:** Ein höherer Punktwert beschreibt die beobachtete Richtung, ein breites Intervall aber erhebliche Unsicherheit. Insbesondere hoher Stromnoise zeigt HECM mit größerer mittlerer Penalty, deren Intervall null einschließt. **Grenze:** Nicht aus dem Balken allein eine bewiesene Populationseigenschaft machen. Die Seeds messen Störungsvariation, die Zellen bleiben die unabhängigen Einheiten. Siehe F061 bis F066.

# Abbildung 6.9: Initialisierungs-Recovery
@figure 6.9

**Frage:** Warum bleibt DM nahe zehn Prozentpunkten, während DD unter die Schwelle fällt und wieder steigt?

**Antwort:** DM/HDM behalten die falsche Ladungsbilanz bis zu einer wirksamen Verankerung. HECM korrigiert über Spannung. DD kann die Qc-Abhängigkeit zeitweise kompensieren, aber bei anderem Kontext wieder stärker auf das gestörte Feature reagieren. Panel b zeigt gepaarte Ausgangsdifferenz, nicht Dataset-MAE. Panel c aggregiert mehrere Fenster und Zellen, nicht nur die Beispielkurve. **Grenze:** Erste Rückkehr ist nicht persistent. Siehe F041 bis F046.

# Abbildung 6.10: Signalverlust und Jitter
@figure 6.10

**Frage:** Weshalb ist 2-Prozent-Sample Loss gravierender als großer Zeitjitter?

**Antwort:** Gefrorene Werte verlieren reale Signaländerung, während der Jittertest erhaltene Werte zeitbewusst verarbeitet. Die integrativen Modelle verlieren insbesondere Ladungsinformation. **Grenze:** Das hängt von der exakten Interventionssemantik ab. Asynchrone Spannung/Strom-Paare oder falsche Zeitstempel könnten viel kritischer sein und sind nicht automatisch durch diese Balken abgedeckt. Fehlerstriche sind Konfidenzintervalle, keine Worst-Case-Grenzen. Siehe F056.

# Abbildung 6.11: Burst Dropout
@figure 6.11

**Frage:** Warum ist DD nach dem Ausfall relativ unauffällig, HECM aber länger verschoben?

**Antwort:** Im Rolling-DD verlassen eingefrorene Rohwerte später das Fenster. Neue Multikanalinformation kann den Einfluss des falschen Qc reduzieren. HECM muss eine verschobene Integrations-/Polarisationshistorie mit begrenzter LFP-Spannungsinformation korrigieren. Das sind plausible Mechanismen. **Grenze:** Der negative globale DD-ΔMAE mit nullüberlappendem Intervall ist kein Nachweis einer Verbesserung. Panel b misst Abweichung vom sauberen Modell, nicht absoluten Referenzfehler. Siehe F057.

# Abbildung 6.12: Spike-Penalty nach Zelle und SOH
@figure 6.12

**Frage:** Warum reagiert DD hier stärker als in der globalen Heatmap?

**Antwort:** Diese Grafik konzentriert sich auf Spike-Samples und löst nach Zelle und SOH auf. Die globale Heatmap mittelt über lange Fenster, in denen fast alle Samples ungestört sind. Beide Aussagen können gleichzeitig richtig sein. **Grenze:** Farbintensität nicht ohne jeweilige Skala vergleichen. Eine stärker gefärbte Zelle beweist keinen kausalen Alterungseinfluss, da Last und SOH gekoppelt sein können. Siehe F053 und F069.

# Abbildung 6.13: Ereignisausgerichteter Spikeverlauf
@figure 6.13

**Frage:** Was ist Excess Absolute Error und kann er negativ sein?

**Antwort:** Er beschreibt zusätzlichen absoluten Zielfehler gegenüber dem gepaarten sauberen Lauf. Er kann negativ sein, wenn die Störung eine bestehende Abweichung kompensiert. Davon verschieden ist der immer nichtnegative Betrag der Differenz zwischen gestörter und sauberer Ausgabe. **Prüfpunkt:** Ein kausaler Spike darf keine vorherige Wirkung verursachen. Bei vorgezogenen Unterschieden Zeitmarken, Rolling-Alignment, Vorereignisse und Mittelung prüfen. **Grenze:** Der plausible dU/dt-Pfad braucht Ablation zur isolierten Attribution. Siehe F053 und F054.

# Abbildung 6.14: ADC-Quantization
@figure 6.14

**Frage:** Warum sind Stufen im Sensorsignal sichtbar, aber nicht gleichermaßen im SOC?

**Antwort:** Integration, Observer und zeitliche Lernabbildung übertragen Eingangsrundung unterschiedlich. Schritte von 0,01 A, 5 mV und 0,5 °C werden gemeinsam im gewählten Szenario geprüft. Alle Makrointervalle überlappen hier null. **Grenze:** Das ist Sensor Quantization, nicht Weight Quantization. Es bewertet weder beliebige ADC-Bitzahlen noch Front-end-Kalibrierung vollständig. Full-scale-Bereich und effektive Auflösung wären nötig, um Bits aus Schrittweiten abzuleiten. Siehe F079 für den anderen Quantization-Begriff.

# Abbildung 6.15: Cross-Scenario-Heatmap
@figure 6.15

**Frage:** Was bedeuten positive, negative und sehr kleine Werte?

**Antwort:** Positiv bedeutet höheren MAE relativ zur jeweiligen Modellbaseline, negativ geringeren. Die Heatmap zeigt Zusatzfehler, keine absoluten nominalen Fehler. Current Offset dominiert mehrere Modelle. **Grenze:** Kleine globale Spikewerte verbergen lokale Peaks. Gain/Offset verwenden adverse Richtungen, andere Zeilen andere definierte Interventionen. Die Heatmap ist daher ein kompakter Index mit methodisch unterschiedlichen Störungsformen, kein universeller Sicherheitsmaßstab. Siehe F048 und F067.

# Abbildung 6.16: Entscheidungssynthese
@figure 6.16

**Frage:** Warum gewinnt nicht überall derselbe Vertreter und was bedeutet große Dreiecksfläche?

**Antwort:** Die Achsen fassen verschiedene relative Kriterien zusammen. DD ist bei nominaler Accuracy und mehreren Robustheitsdefinitionen stark, HECM bei Recovery. Die rechte Seite ändert Prioritätsgewichte. **Grenze:** Flächen sind nicht linear in jedem Einzelwert und die Achsenskalierung ist relativ. Ein Score nahe eins ist keine Zuverlässigkeitswahrscheinlichkeit. Für eine reale Entscheidung muss die konkrete Störung und ein absolutes Anforderungslimit betrachtet werden. Siehe F068.

# Abbildung 6.17: Hardware-Software-Äquivalenz
@figure 6.17

**Frage:** Ist der sehr kleine Fehler hier die Batterie-SOC-Genauigkeit?

**Antwort:** Nein. Verglichen werden zwei Implementierungen desselben Schätzers. Kleine MCU-zu-Software-Abweichung zeigt die numerische Konsistenz des Exports. Beide können gegenüber Dataset-SOC einen deutlich größeren Fehler besitzen. **Vertiefung:** Logarithmische Achse und Prozentpunkteinheit nennen. **Grenze:** Gleiche Ausgaben beweisen nicht die Richtigkeit der gemeinsamen Referenz oder des Modells. Siehe F084 und F089.

@latex \clearpage

## Warum rechnen DM, HDM und DD auf dem Mikrocontroller mit 32 Bit, HECM intern aber mit 64 Bit?

**Antwort:** Der STM32H753 besitzt trotz seines 32-Bit-Prozessorkerns eine Gleitkommaeinheit für 32- und 64-Bit-Rechnungen. Im geprüften Firmware-Code verwenden DM und HDM eine 32-Bit-Ladungssumme; ihre Python-Referenz integriert intern mit 64 Bit. DD verwendet in der Software-Referenz und im C-Modell 32-Bit-Gleitkommazahlen. HECM nutzt intern 64 Bit für Filterzustände, Kovarianz und Parameterinterpolation; Ein- und Ausgabe der Rechenfunktion sind 32 Bit.

**Einordnung:** Bei DD passt die 32-Bit-Implementierung zur vorhandenen Modellpräzision und benötigt weniger Speicher als eine 64-Bit-Darstellung. Bei DM und HDM können unterschiedliche Rundungen der fortlaufenden Ladungssumme zur Hardware-Software-Abweichung beitragen. Beim HECM kann höhere Genauigkeit für wiederholte Filter- und Matrixoperationen sinnvoll sein. Die gemessene Übereinstimmung ist für alle vier Implementierungen sehr gut. **Grenze:** Eine ausdrückliche historische Begründung dieser Datentypwahl und ein Nachweis ihrer Optimalität sind nicht dokumentiert. Weder die Notwendigkeit von 64 Bit beim HECM noch die vollständige Ursache der unterschiedlichen Balkenhöhen ist durch diesen Vergleich isoliert belegt.

# Abbildung 6.18: Inferenzlatenzen
@figure 6.18

**Frage:** Warum sind DM/HDM nahezu deterministisch und DD breiter verteilt?

**Antwort:** DM/HDM führen sehr wenige Operationen aus. DD berechnet zahlreiche Matrix- und Activation Operations, HECM Lookup/Observer-Schritte. Wiederholte Zellreplays zeigen beobachtete Laufzeitstreuung. **Grenze:** Die hier gezeigte DD-Ausführung ist continuous, nicht die teure Rolling-Variante des primären Robustheitsbenchmarks. Der Maximalwert ist ein beobachteter Extremwert, kein formaler WCET-Beweis. Siehe F088 bis F090.

@latex \clearpage

# Abbildung 6.18: Erklärungen der Laufzeitverteilungen

## Warum zeigen DM und HDM zwei deutliche Laufzeitgruppen?

**Antwort:** Die kleinere Spitze umfasst viele Messungen und ist kein einzelner Ausreißer. Im Code wird normalerweise die Ladungssumme durch Stromintegration fortgeschrieben. Nach mindestens 300 Sekunden oberhalb der Spannungsschwelle von 3,63 V wird sie stattdessen auf null gesetzt. Dann entfällt die Integrationsrechnung. Auch die SOC-Begrenzung erzeugt Verzweigungen. Diese unterschiedlichen Rechenwege sind eine plausible Erklärung. Die Zuordnung einer bestimmten Spitze zum Vollladezweig ist ohne Abgleich der einzelnen Laufzeiten mit den Betriebszuständen nicht bewiesen.

## Warum ist die Laufzeitverteilung des HECM breiter?

**Antwort:** Die Parametersuche durchläuft die SOC- und SOH-Stützstellen vom Tabellenanfang bis zum passenden Intervall. Je nach Zustand sind unterschiedlich viele Suchschritte erforderlich. An oder außerhalb der Tabellengrenzen liefert die Funktion das Randintervall unmittelbar. Das könnte schnelle Gruppen und breiter verteilte längere Laufzeiten erklären. Wie häufig ein Zeitbereich vorkommt, hängt auch von den Betriebszuständen der Testsequenzen ab. Der konkrete Beitrag dieser Suche zur beobachteten Verteilung ist nicht isoliert nachgewiesen.

## Warum schwankt die DD-Laufzeit trotz gleicher Matrizen und Schleifenlängen?

**Antwort:** Gewichtsmatrizen, Dimensionen und Schleifenlängen bleiben bei der kontinuierlichen GRU-Ausführung gleich. Eingaben und Zwischenwerte ändern sich jedoch. Die Sigmoid-Funktion verwendet je nach Vorzeichen unterschiedliche Zweige; außerdem werden Exponentialfunktion, Tangens hyperbolicus und ReLU berechnet. Wertabhängige Rechenwege sowie mögliche Cache- und Interrupt-Effekte können kleine Laufzeitunterschiede verursachen. Die mittleren 90 Prozent liegen bei etwa 423,8 bis 426,1 Mikrosekunden, der Median bei 424,9 Mikrosekunden. Die Streuung ist damit relativ klein. Die glockenartige Form beweist weder eine Normalverteilung noch eine einzelne Ursache. Welche Effekte dominieren, wurde nicht separat gemessen.

# Abbildung 6.19: Flash und RAM
@figure 6.19

**Frage:** Warum braucht HECM viel Flash, aber wenig RAM?

**Antwort:** Seine voridentifizierten Parameterflächen liegen als persistente Tabellen im Flash. Die aktuellen Observerzustände bleiben klein. DD speichert Gewichte im Flash und Zustände/Arbeitsdaten im RAM. Das Ergebnis betrifft die konkrete Tabellendichte und Firmware, nicht jede ECM-Implementierung. **Grenze:** Das gemeinsame SOH-LSTM ist nicht Teil dieser MCU-Messung. Ein vollständiger SOC/SOH-Systembedarf wäre größer und müsste separat gemessen werden. Siehe F087.

@latex \clearpage

## Warum braucht HECM trotz großer Kennfelder kaum mehr RAM als DM und HDM?

**Antwort:** Die Kennfelder sind im geprüften C-Code als konstante Daten angelegt und durch die Linker-Konfiguration dem Flash zugeordnet. Der Prozessor liest die benötigten Einträge direkt aus diesem Speicher. Eine vollständige Kopie der Tabellen im RAM ist nicht erforderlich. Bei der Interpolation werden nur benachbarte Tabellenwerte verwendet; Parameter und Zwischenwerte liegen vorübergehend in Registern oder im RAM.

**Warum liegen die RAM-Balken nahe beieinander?** HECM speichert zusätzlich einen kleinen Zustandsvektor und eine kleine Kovarianzmatrix, aber keine großen veränderlichen Tabellen. Alle drei Implementierungen enthalten zudem gemeinsamen Speicherbedarf der Firmware und Laufzeitumgebung. Die dokumentierte Messung ergibt 2432 Byte für DM und HDM sowie 2528 Byte für HECM, also 96 Byte Unterschied. Das entspricht gerundet 2,4 beziehungsweise 2,5 KiB.

**Einordnung:** HECM nutzt ebenfalls Strom zur SOC-Fortschreibung, berechnet aber zusätzlich RC-Zustände und eine Spannungskorrektur. Mehr Rechenarbeit bedeutet nicht automatisch viel mehr RAM. Die Spitzenwerte gelten für die vermessene Firmware und die geprüften Abläufe; sie sind kein universeller Worst-Case-Nachweis.

## Warum zeigt Abbildung 6.19 den Speicher in Kibibyte statt in Byte oder Megabyte?

**Antwort:** Speicher wird binär adressiert: Zehn Adressbits unterscheiden beispielsweise 1024 Adressen. Deshalb sind Speicherkapazitäten häufig in Zweierpotenzen organisiert. Dazu passt die binäre Einheit Kibibyte: 1 KiB entspricht zwei hoch zehn, also exakt 1024 Byte. KiB bezeichnet diese Umrechnung eindeutig und vermeidet die Verwechslung mit 1000 Byte pro Kilobyte. Zusätzlich bleiben die Zahlen gut lesbar: 2432 Byte sind 2,375 KiB, gerundet 2,4 KiB.

**Begriffe unterscheiden:** 1 KiB = 1024 Byte, 1 MiB = 1 048 576 Byte. Dagegen bedeuten die dezimalen Einheiten 1 kB = 1000 Byte und 1 MB = 1 000 000 Byte. KiB macht die verwendete Basis eindeutig. Byte ist die Grundeinheit und selbst weder ein dezimaler noch ein binärer Vorsatz. Ein Byte besteht aus acht Bit. Für diese Abbildung sind KiB eine Darstellungsentscheidung, keine technische Pflicht.

# Abbildung 6.20: DD-Ausführungsmodi
@figure 6.20

**Frage:** Wie können gleiche Gewichte gleichzeitig sehr unterschiedliche Fehler und Laufzeiten erzeugen?

**Antwort:** Rolling baut aus dem letzten vollständigen Fenster neu auf. Continuous trägt den Zustand fort. Periodic Reset löscht Kontext, ohne das ganze historische Fenster neu einzuspielen. Unterschiedliche zeitliche Information verändert die Funktion trotz gleicher Gewichte. **Beleg:** Rolling etwa 724 ms und 67,3 KiB, Continuous etwa 0,425 ms und 4,1 KiB. **Grenze:** Ähnliche nominale Accuracy ist keine vollständige Störungsäquivalenz. Siehe F085 und F086.

# Abbildung 7.1: Stateful LSTM plus Kopf
@figure 7.1

**Frage:** Was läuft einmal pro Sample und was bleibt über Aufrufe erhalten?

**Antwort:** Das LSTM verarbeitet den aktuellen Featurevektor. h und c bleiben erhalten, der MLP-Kopf bildet h auf SOC oder SOH ab. Für SOC ist H = 64, für SOH H = 128. **Grenze:** Das Diagramm ist nicht das stündliche SOH-Netz mit 20 aggregierten Eingängen aus Kapitel 6. Hier werden beide Aufgaben auf dem 1-Hz-Stream behandelt. Siehe F073 und F084.

# Abbildung 7.2: Aufgabenabhängige Architekturen
@figure 7.2

**Frage:** Warum unterschiedliche Features und verschiedene Ausgangsaktivierungen?

**Antwort:** SOC erhält unter anderem Qc und Ableitungen für kurzfristige Dynamik. SOH erhält Zeit und Durchsatz für langfristige Alterungsinformation. SOC wird durch Sigmoid auf 0 bis 1 begrenzt, SOH bleibt vor Nachverarbeitung linear. **Grenze:** Zeit kann ein starker Datensatzproxy sein. Eine Modellleistung mit Zeitfeature beweist keine robuste Extrapolation auf andere Alterungsraten. Ein SOH-Wert über eins kann bei nominaler Referenz physikalisch möglich sein, muss aber geprüft werden. Siehe F019.

# Abbildung 7.3: Deploymentpipeline
@figure 7.3

**Frage:** Wie kontrollieren Sie, dass Kompressionsvarianten fair verglichen werden?

**Antwort:** Gleicher Base-Ausgangspunkt, identischer Featurestream, dokumentierte Gewichte/Scaler und dieselbe Zustandsführung. Danach getrennte Kompression, C-Export, numerischer Vergleich und Hardwaremessung. **Grenze:** Unterschiedliches Fine-Tuning ist selbst eine Intervention und braucht für kausale Attribution eine passende Kontrolle. Das Schema allein ersetzt keine Versions- und Datenprovenienz. Siehe F078 und Quellenauftrag Q5.

# Abbildung 7.4: Numerisches Pruning-Beispiel
@figure 7.4

**Frage:** Warum verschwindet h2 nicht nur in einer Gatezeile?

**Antwort:** Alle vier Gates, der recurrent input, Bias, Zustände und MLP-Eingänge beziehen sich auf denselben Kanal. Entfernen muss diese Abhängigkeiten gemeinsam berücksichtigen. Im Beispiel hat h2 den kleinsten aggregierten L2-Score. **Grenze:** Die kleinen Matrixzahlen sind illustrativ, keine echten trainierten Gewichte. Der Score misst Gewichtsmagnitude, nicht direkt Sensitivität des SOH/SOC-Ausgangs. Siehe F075.

# Abbildung 7.5: Quantization und Export
@figure 7.5

**Frage:** Was bleibt grün/rot beziehungsweise als Ganzzahl oder Float gespeichert?

**Antwort:** Die Recurrent-Matrizen werden zeilenweise in Codes und Skalen aufgeteilt. Die effektiven Werte entstehen im Rechenpfad wieder in Float. Die Netzstruktur bleibt erhalten. **Grenze:** Ein gespeichertes INT8-Muster sagt nichts darüber, ob Multiplikation und Akkumulation als Integer laufen. Die Ein- und Ausgabe und recurrent states bleiben hier FP32. Die Detailgrenze steht zusätzlich in A.11. Siehe F079 bis F082.

# Abbildung 7.6: SOC über die gesamte Trajektorie
@figure 7.6

**Frage:** Warum sieht man oben fast nur eine dichte Fläche und unten trotzdem systematische Fehler?

**Antwort:** Viele Ladezyklen werden über lange Zeit zusammengedrängt. Kleine Unterschiede überlagern sich visuell. Das Fehlerpanel macht sie sichtbar. Base/Pruned/Quantized haben global MAE 2,68/2,34/2,79 Prozentpunkte. **Grenze:** Visuelle Überdeckung ist kein numerischer Gleichheitsnachweis. Es ist die repräsentative Einzelzelltrajektorie der Embedded-Studie, nicht automatisch das sechszellige Robustheitsmittel. Siehe F084 und F095.

# Abbildung 7.7: Gefilterter SOH
@figure 7.7

**Frage:** Wie viel der ruhigen Schätzung stammt vom Netz und wie viel vom Filter?

**Antwort:** Die gezeigte Ausgabe durchläuft Anfangsausrichtung, zwei EMAs und Rate-Limiter. Deshalb ist die Linienruhe eine Eigenschaft der gesamten Pipeline. Das primäre Ergebnis bevorzugt Base gegenüber den komprimierten Varianten. **Grenze:** Die zweite EMA hat eine sehr lange Antwortzeit. Die Diagrammwerte dürfen nicht als ungefilterter SOH-MAE zitiert werden. Andere Werte in A.6 benötigen klare Sequenzprovenienz. Siehe F092 bis F094.

# Abbildung 7.8: SOC-Fehlerverteilung
@figure 7.8

**Frage:** Warum sind Boxplot und Histogramm beide nötig?

**Antwort:** Die Boxplots zeigen Verteilung absoluter Fehler, Histogramme zusätzlich deren Vorzeichenstruktur. Pruned hat den kleinsten Median, aber seine Tails und lokalen Abschnitte müssen separat betrachtet werden. **Grenze:** Überlappende Histogramme können zeitlich völlig verschiedene Fehler enthalten. Die Beobachtungen sind korrelierte Samples, keine Millionen unabhängigen Versuche. Boxplot-Whisker nicht automatisch als Min-Max interpretieren. Siehe F065 und F078.

# Abbildung 7.9: SOH-Fehlerverteilung
@figure 7.9

**Frage:** Warum ist die Form unregelmäßiger als beim SOC?

**Antwort:** SOH verändert sich langsam, wird aus spärlichen Kapazitätsankern interpoliert und stark gefiltert. Bestimmte Alterungsabschnitte können lange einen ähnlichen Bias tragen und Häufungen erzeugen. **Grenze:** Verteilungsform identifiziert nicht eindeutig einen elektrochemischen Alterungsmechanismus. Ein größeres Absolutfehlerband beim komprimierten Modell beweist keine rekurrente Instabilität. Die Zeitlokalisation liefert A.5. Siehe F094 und F095.

# Abbildung 7.10: Dynamischer SOC-Ausschnitt
@figure 7.10

**Frage:** Warum ist Pruned trotz besserem Gesamt-MAE lokal teilweise weiter von Base entfernt?

**Antwort:** Pruning verändert die Funktion. Global günstigeres Verhalten muss nicht in jedem dynamischen Abschnitt günstiger sein. Außerdem ist die Base-Ausgabe keine Ground Truth. Der untere Residuenplot ist für die tatsächliche Genauigkeit wichtiger als allein die Überdeckung der Modellkurven. **Grenze:** Der gewählte 30k-bis-40k-Sekunden-Ausschnitt ist eine Illustration, kein repräsentatives Mittel aller Transienten. Siehe F078 und F095.

# Abbildung 7.11: Check-up-Ausschnitt
@figure 7.11

**Frage:** Was zeigt der längere Abschnitt, das im kurzen Pulsfenster fehlt?

**Antwort:** Längere Ladungsintegration, Betriebsphasenwechsel und Diagnoseintervalle zeigen andere Dynamik. Die Varianten bleiben im gezeigten Abschnitt grundsätzlich konsistent, mit sichtbaren lokalen Fehlerunterschieden. **Grenze:** Ein stabiler Ausschnitt beweist keine allgemeine Langzeitstabilität. Die Datenauswahl der Diagnosephasen und die Maskierung von Kapazitätstests müssen nachvollziehbar sein. Siehe F020 und F095.

# Abbildung 7.12: Ressourcenlandschaft
@figure 7.12

**Frage:** Warum stimmen theoretischer Gewichtsspeicher und gelinkter Flash nicht überein?

**Antwort:** Die Theorie zählt Parameterrepräsentationen. Die Firmware enthält zusätzlich Code, Skalierung, konstante Tabellen und Laufzeitunterstützung. RAM ist wiederum ein anderer Speicherbereich mit Zuständen, Puffern und Stack. **Grenze:** Weight-only-Quantization lässt viele FP32-Daten unverändert. Die genauen RAM-Unterschiede nicht allein mit einem Viertel pro Gewicht erklären. Für vollständige Attribution Linker-Sektionen und Stackmessung zeigen. Siehe F081 und F087.

# Abbildung 7.13: Hostlatenzverteilungen
@figure 7.13

**Frage:** Warum steht Quantized weiter rechts und weshalb sind SOH-Abstände größer?

**Antwort:** Der Mixed-Precision-Kernel erhöht hier den Aufwand, und SOH besitzt den größeren recurrent core. Host Latency umfasst aber auch UART und Rücktransport. **Grenze:** Ein fester Kommunikationsanteil kann Kernelunterschiede relativ verkleinern. Es sind keine unabhängigen Energieverteilungen. Bei einem Vergleich mit Kapitel 6 zuerst Messgrenzen und Modellsemantik klären. Siehe F082, F083 und F089.

# Abbildung A.1: Vollständige Testmatrix
@figure A.1

**Frage:** Können Sie jede Zeile einem Eingriff und einer Wiederholungszahl zuordnen?

**Antwort:** Die Matrix trennt manipulierte Kanäle/Zustände von wiederholten Seeds. Baseline und Initialisierung erklären zusammen mit 18 Störungszeilen die 20 Definitionen. Die beiden Vorzeichen von drei Gainstufen und einem Offset erweitern die Messstörungsfälle auf 22 Subfälle. **Grenze:** Nicht aus der Zahl der farbigen Punkte die unabhängige Stichprobengröße ableiten. Die unabhängigen Einheiten sind weiterhin sechs Zellen. Siehe F066 und F067.

# Abbildung A.2: DD-Latenzmodi
@figure A.2

**Frage:** Warum liegen kontinuierlich und periodisch reset nahezu zusammen, obwohl ihre Genauigkeit stark verschieden ist?

**Antwort:** Beide führen pro Update nur einen rekurrenten Schritt aus. Der gelegentliche Zustandsreset spart keine relevanten Matrixoperationen und kostet wenig. Er verändert aber die verfügbare Historie. Rolling wiederholt dagegen das gesamte Fenster. **Grenze:** Laufzeitähnlichkeit bedeutet keine funktionale Gleichwertigkeit. Dieses Anhangsbild ergänzt die Fehler- und Speicherpanels in Abbildung 6.20. Siehe F085 und F086.

# Abbildung A.3: Lookup-Sensitivität
@figure A.3

**Frage:** Kann man daraus sagen, die HECM-Tabelle sei unwichtig?

**Antwort:** Nein. Links sind lokale Änderungen der Störungspenalty relativ zur jeweiligen sauberen Variante, rechts Recovery unter derselben Lookupvariation. Viele Interaktionen sind klein gegenüber dem großen Offseteffekt, aber Widerstand -10 Prozent zeigt einen zensierten Zellfall und ein deutlich höheres Zeitmittel. **Grenze:** OAT-Variationen prüfen keine kombinierten Unsicherheiten oder falschen Kennlinienformen. Tabelle A.1 zusätzlich nennen, nicht nur farbige Felder. Siehe F059 und F060.

# Abbildung A.4: Pruning-Diagnostik
@figure A.4

**Frage:** Warum schneiden Sie links die kleinsten 19 Scores ab und was beweist die Gewichtsverteilung rechts?

**Antwort:** Das entspricht der implementierten deterministischen Saliency-Regel für 64 auf 45 SOC-Kanäle. Rechts wird eine zentrale Gewichtungsverteilung vor und nach Auswahl verglichen. **Grenze:** Eine ähnliche Verteilungsform bedeutet weder ähnliche Funktion noch bewiesene Regularisierung. Rekurrente Struktur und Vorzeichen-/Kanalzusammenhänge gehen im Histogramm verloren. Für Kausalität braucht es zusätzliche Kontrollen. Siehe F075 und F078.

# Abbildung A.5: Zehn Zeitsegmente
@figure A.5

**Frage:** Ist der steigende Fehler rechts im Leben ein Quantization-Drift-Problem?

**Antwort:** Nicht zwingend. Sowohl Base als auch Quantized werden gegenüber dem Ziel schlechter, während ihre gegenseitige Abweichung nicht entsprechend wächst. Das spricht in diesem Replay gegen akkumulierte Quantization-Abweichung als Hauptursache. **Grenze:** Die Segmente sind gleich viele Samples, nicht unabhängige Wiederholungen. Veränderungen von SOH, Last und Labelqualität können gemeinsam wirken. Siehe F095.

# Abbildung A.6: SOH-Filterstufen
@figure A.6

**Frage:** Warum wechselt das Modellranking nach dem Filter?

**Antwort:** Filterung wirkt auf Richtung, Frequenz und zeitliche Struktur der Residuen, nicht nur auf deren Amplitude. Ein Modell mit mehr hochfrequentem Anteil kann stärker profitieren als eines mit langsamem Bias. Die lange EMA verändert den Zeitbezug zusätzlich. **Grenze:** Die hier genannten MAE-Werte weichen von der Haupt-SOH-Auswertung ab. Die präzise Daten-/Zustandsprovenienz vor einer eindeutigen Erklärung klären. Siehe F092 bis F094 und P06.

# Abbildung A.7: Utility-Gewichtungen
@figure A.7

**Frage:** Bedeutet viel Rot, dass Pruning immer gewinnt?

**Antwort:** Es gewinnt für einen großen Teil der untersuchten diskreten Prioritätskombinationen, insbesondere beim SOC. Beim SOH können Accuracy-Priorität Base und Flash-Priorität Quantized attraktiv machen. **Grenze:** Die Häufigkeit über ein künstliches Gitter ist keine Nutzer- oder Einsatzwahrscheinlichkeit. Harte Grenzen gehören vor den gewichteten Score. **Rechenfrage:** 1771 Kombinationen über Stars-and-bars erklären. Siehe F098 und R11.

# Abbildung A.8: Eingangsbuffer-Fehler
@figure A.8

**Frage:** Warum können Peaks groß sein, während ΔMAE61 klein bleibt?

**Antwort:** Der Peak betrifft einen kurzen Moment, während der MAE-Kontrast 61 Samples mittelt. Ein wieder abklingender Zustandseffekt kann lokal relevant sein und global wenig beitragen. Die Fault-Clean-Differenz und die Änderung des Referenzfehlers sind verschiedene Metriken. **Grenze:** Nur ein bestimmtes Mantissenbit und begrenzte Events wurden getestet. SOH-Recoveryprozente beziehen sich auf 60 Sekunden, nicht auf den 24-h-Test in Kapitel 6. Siehe F097.

# Abbildung A.9: Komplexitätsskalierung
@figure A.9

**Frage:** Warum steigen die MAC-Kurven quadratisch und die Reduktionskurve nähert sich einer Grenze?

**Antwort:** Recurrent-Matrizen enthalten H mal H-Verbindungen pro Gate. Bei festem Input und Kopf dominiert dieser Term für große H. Entfernt man einen Anteil p, beträgt die asymptotische Einsparung 2p-p². Die endlichen tatsächlich gewählten Architekturen liegen darunter. **Grenze:** Analytische MACs sind keine direkte Hardwarezeit und enthalten Activation Functions nicht. Siehe F077 und R07.

# Abbildung A.10: Reichweite des L2-Kriteriums
@figure A.10

**Frage:** Warum nicht Gradient oder Aktivierung statt Gewichtsnorm?

**Antwort:** Der gewählte Score ist deterministisch aus den trainierten Gewichten berechenbar und passt zur Kanalstruktur. Gradientenkriterien benötigen Daten und Backward Pass, Activation-Kriterien einen repräsentativen Kalibrierstream und eine zeitliche Aggregation. **Grenze:** Die Tabelle vergleicht Informationsbedarf und Methodenscope, nicht experimentell gemessene Überlegenheit. Keine Aussage L2 ist am besten ohne Vergleichsexperiment. Siehe F075 und F076.

# Abbildung A.11: Mixed-Precision-Grenze
@figure A.11

**Frage:** Weshalb verbleibt ein erheblicher FP32-Anteil nach Quantization?

**Antwort:** Nur Wih und Whh werden als INT8 gespeichert. Kopf, Bias, Skalen und sämtliche dynamischen Zustände bleiben Float. Der gestapelte Speichervergleich macht diese Grenze sichtbar. **Grenze:** Die Grafik betrifft Modellkonstanten, nicht zwangsläufig denselben Gesamtumfang wie Firmware-Flash in Abbildung 7.12. Beide Größen nicht gegeneinander als Widerspruch lesen. Siehe F080 und F081.

# Abbildung A.12: Statische Operationen und gemessene Zeit
@figure A.12

**Frage:** Warum ist der statische Quantized/Base-Faktor etwa 1,81, aber der Zeitfaktor nicht?

**Antwort:** Das vereinfachte Zählen zusätzlicher Skalierungs-Multiplikationen bildet nicht alle Instruktionen, Konversionen, Speicherzugriffe, Schleifen und Activation Functions ab. Eine MAC und eine zusätzliche Multiplikation sind auch nicht zwingend zeitlich identisch teuer. **Grenze:** Ohne Profiling kann die genaue Diskrepanz nicht einer einzelnen Ursache zugeordnet werden. Das Bild belegt gerade die Grenze einer naiven FLOP-zu-Latenz-Umrechnung. Siehe F082 und F083.

# 14. Abschließende Selbstprüfung

Du bist nicht dann vorbereitet, wenn du alle Zahlen auswendig kannst, sondern wenn du jede wichtige Zahl korrekt einordnest. Für die Prüfung sollten mindestens die nominalen SOC-Rangfolgen, die abweichende Recovery-Rangfolge, der Offsetmechanismus, die lange SOH-Filterantwort, die Inferenzmodi und die Mixed-Precision-Grenze ohne Hilfsmittel erklärbar sein.

**Abschlussaufgabe:** Wähle zufällig eine Abbildung aus Kapitel 5, eine aus Kapitel 6 und eine aus Kapitel 7. Erkläre bei jeder Input, Output, Referenz, Beobachtungseinheit, Intervention, Kennzahl und Grenze. Verbinde sie anschließend mit einer der drei Forschungsfragen. Wenn du dabei unterschiedliche Modelle oder Fehlerdefinitionen vermischst, wiederhole die betreffende Querverbindung statt nur den Plot auswendig zu lernen.

**Wichtigste sachliche Schlussantwort:** Die Dissertation liefert keine universelle Garantie für neuronale Batteriezustandsschätzung. Sie liefert eine konkrete, messbare Vorgehensweise, um Repräsentation, nominalen Fehler, Störungsmechanismen und Embedded-Ausführung gemeinsam zu beurteilen. Genau diese Verbindung ist verteidigbar, solange die Grenzen der einzelnen Studien transparent bleiben.
