# Verteidigung der Dissertation

Fragenkatalog, begründete Antwortvorschläge, Rechenübungen und Abbildungsatlas für Florian Rzepka. Arbeitsstand 16. September 2026, nach Aktualisierung auf Git-Commit 2d8fdc9.

**Gegenstand:** Deployment-Oriented Neural Network State-Estimation Methods for Battery Management Systems: Ageing Estimation, Robustness Benchmarking, Embedded Optimization, and Hardware Integration.

**Quellenbasis:** Die angegebene main.pdf mit 230 PDF-Seiten, main.tex, das eingebundene Robustheitskapitel, dessen Anhang und die eingebundenen Ergebnistabellen. Das PDF trägt das Erstellungsdatum 3. September 2026. PDF-Seiten und gedruckte Seiten sind nicht identisch. Im Haupttext entspricht beispielsweise PDF-Seite 134 der gedruckten Seite 104. Der Atlas verwendet ausdrücklich PDF-Seiten und die tatsächlich gedruckten Abbildungsnummern, nicht historische Dateinamen wie Figure_09.

**Wozu dieses Dokument dient:** Antworten üben, Rechenwege verstehen, Grenzen offen benennen und passende Belege finden. Es sagt nicht vorher, was bestimmte Mitglieder der Kommission persönlich fragen werden. Die Fragen sind fachliche Simulationen aus Batteriephysik, Statistik, maschinellem Lernen und eingebetteten Systemen.

**Evidenzkennzeichnung:** BELEGT bezeichnet Angaben der Dissertation, nicht eine hier erneut durchgeführte Messung. HERLEITUNG bezeichnet eine nachvollziehbare Rechnung oder fachliche Einordnung. HYPOTHESE bezeichnet einen möglichen Mechanismus ohne isolierenden Nachweis. OFFEN bezeichnet Angaben, die vor der Verteidigung anhand von Skripten, Messprotokollen oder Originalarbeiten geklärt werden sollten. Diese Vorbereitung wiederholt keine Simulation und zertifiziert weder Firmware noch Versuchsdaten.

**Aufbau:** Teil 1 behandelt roten Faden und Grundbegriffe. Teil 2 behandelt Daten und Referenzen. Teil 3 behandelt den SOH-MLP. Teil 4 behandelt Robustheit und Statistik. Teil 5 behandelt Mikrocontroller, Pruning und Quantisierung. Teil 6 enthält kritische Rückfragen, Rechenübungen und einen Lernplan. Der anschließende Atlas behandelt alle 67 nummerierten Abbildungen einschließlich Anhang und zeigt die Originalseiten.

**Lernmethode:** Erst die Frage ohne Antwort lesen. Dann eine Antwort von 30 bis 60 Sekunden geben. Anschließend die ausführliche Begründung und die genannte Grenze erklären. Zahlen immer mit Modell, Datensatz, Einheit und Auswertungsfenster nennen. Ein plausibler Mechanismus ist noch kein experimentell identifizierter Mechanismus.

# 1. Roter Faden und wissenschaftlicher Beitrag

## F001. Was ist Ihre Dissertation in drei Sätzen?

**Antwort:** Ich untersuche Batteriezustandsschätzung als durchgängige Aufgabe von der zeitlichen Eingangsrepräsentation bis zur Ausführung im BMS. Zuerst zeige ich, wie ein feedforward SOH-Modell durch Historienmerkmale zeitliche Information nutzen kann, dann vergleiche ich vier SOC-Repräsentanten unter nominalen und gestörten Messungen, anschließend untersuche ich die Kompression und Mikrocontroller-Ausführung rekurrenter SOC- und SOH-Modelle. Die gemeinsame Aussage ist, dass geringer Testfehler allein nicht genügt, weil Störungsverhalten, Recovery, Zustandshandhabung, Speicher und Laufzeit eigenständige Anforderungen sind.

**Vertiefung:** Die Beiträge sind keine lückenlose Versuchsreihe mit einem einzigen Modell. Kapitel 5 verwendet NMC und ein MLP, Kapitel 6 LFP und unter anderem eine GRU, Kapitel 7 LFP und zwei LSTM-Modelle. Verbunden werden sie durch die Evaluationslogik. **Beleg:** Kapitel 1 und 8. **Nicht behaupten:** Alle Schlussfolgerungen seien am selben Modell oder in einem vollständig integrierten Feldsystem gemessen worden.

## F002. Was ist tatsächlich neu, wenn LSTM, EKF und Pruning bereits bekannt sind?

**Antwort:** Der Beitrag liegt in der anwendungsspezifischen Repräsentation, der kontrollierten Vergleichsmethodik und der nachvollziehbaren Implementierungs- und Messkette. Ich beanspruche weder die Erfindung einer neuen rekurrenten Zelle noch eines allgemein neuen Pruningverfahrens. Der Mehrwert entsteht beispielsweise durch die Trennung von Accuracy, Robustness und Recovery, die zellweise gepaarte Auswertung und den Nachweis, dass identische GRU-Gewichte bei unterschiedlicher Zustandshandhabung sehr unterschiedliche Kosten und Transienten erzeugen.

**Nachfrage:** Ist das nur Engineering? Eine sinnvolle Antwort benennt prüfbare Erkenntnisse: Wann hilft explizite Historie, welche Fehlermechanismen verändern einen nominalen Vergleich und welche Optimierung adressiert tatsächlich den Hardwareengpass? **Grenze:** Die Breite des Rahmens ersetzt keine isolierende Ablation jedes Mechanismus. Die eigene Rolle bei Datenaufnahme, Hardware, Software, Analyse und Manuskripten sollte anhand persönlicher Arbeitsnachweise erläutert werden, nicht aus diesem Text erfunden werden.

## F003. Welche drei Forschungsfragen beantworten welche Ergebnisse?

**Antwort:** Forschungsfrage 1 betrifft die zeitliche Repräsentation mit begrenzter architektonischer Komplexität. Kapitel 5 untersucht kumulierten Strom und Lag-Sequenzen. Forschungsfrage 2 betrifft praktische Robustheit. Kapitel 6 vergleicht nominale Genauigkeit, Messstörungen und Initialisierungs-Recovery. Forschungsfrage 3 betrifft Embedded-Umsetzung. Kapitel 7 vergleicht Base, Pruned und Quantized hinsichtlich Schätzfehler, Speicher und Laufzeit.

**Prüfpunkt:** Die Energieantwort ist eingeschränkt. Kapitel 7 verwendet eine aus Laufzeit und angenommenen 0,5 W berechnete Energiegröße, keine unabhängige Energiemessung. Die Forschungsfrage wird damit bezüglich Laufzeit und Speicher direkt, bezüglich Energie nur unter einer konstanten Leistungsannahme beantwortet. **Beleg:** Kapitel 1.1, 7.6 und 8.1.

## F004. Warum ist die Arbeit keine Suche nach dem universell besten Modell?

**Antwort:** Ein Ranking hängt von Referenzdefinition, Eingangsmerkmalen, Trainingsdaten, Störungen, Auswertungsfenstern und Hardware ab. DD hat in Kapitel 6 den kleinsten nominalen MAE und führt mehrere Robustheitszusammenfassungen an. HECM hat die bessere beobachtete Recovery. DM und HDM sind wesentlich billiger in der Ausführung. Schon diese Ergebnisse verhindern eine eindimensionale allgemeine Siegerbehauptung.

**Vertiefung:** Ein vollständiger Familienvergleich müsste mehrere Vertreter, Architekturen, Parametrierungen und Trainingsinitialisierungen einschließen. Hier werden vier festgelegte Repräsentanten verglichen. Die übertragbaren Aussagen betreffen ihre Mechanismen, zum Beispiel Integration ohne kontinuierliche Spannungskorrektur, nicht jede mögliche Implementierung einer Klasse.

## F005. Was bedeutet deployment-oriented konkret?

**Antwort:** Eingänge müssen zur Laufzeit verfügbar und kausal berechenbar sein. Skalierung, Zustände und Gewichtsexport müssen zwischen Training und Firmware konsistent sein. Die Implementierung muss in den Speicher passen und rechtzeitig fertig werden. Außerdem müssen Störungen und Wiederanlauf betrachtet werden. Deployment ist deshalb mehr als das Umwandeln einer Modelldatei in ein C-Array.

**Beispiel:** Ein Modell mit 2024 Samples Historie kann nominell kompakt sein und trotzdem bei jedem Ausgabeschritt 2024 rekurrente Schritte wiederholen. Ein dauerhaft fortgeschriebener Zustand vermeidet diese Wiederholung, verändert aber die zeitliche Semantik. Genau diese gekoppelte Entscheidung untersucht Kapitel 6. **Grenze:** Ausführbarkeit auf einem Entwicklungsboard ist noch keine funktionale Sicherheit oder Feldqualifikation.

## F006. Warum drei Anforderungsdimensionen statt eines einzigen Scores?

**Antwort:** Performance beschreibt die Brauchbarkeit der Schätzung, Hardware die Ressourcen und Deployment die Umsetzbarkeit und Wartbarkeit. Ein kleiner Fehler kann fehlende Echtzeitfähigkeit nicht kompensieren. Ein kleiner Speicher kann gefährliche Transienten nicht kompensieren. Manche Bedingungen sind harte Grenzen und gehören vor eine gewichtete Nutzwertoptimierung.

**Vertiefung:** Erst Anforderungen prüfen, etwa maximal zulässige Latenz und sichere Fehlerbehandlung. Danach innerhalb zulässiger Kandidaten Trade-offs vergleichen. Ein gewichteter Score ist eine Entscheidungshilfe, keine Naturkonstante und keine Sicherheitsfreigabe. **Beleg:** Abbildung 2.1, Kapitel 6.5 und 7.11.

# 2. Batteriephysik und Zustandsschätzung

## F007. Was unterscheidet SOC, SOH und verfügbare Energie?

**Antwort:** SOC ist ein relativer Ladungszustand gegenüber einer festgelegten nutzbaren Kapazität. Kapazitäts-SOH setzt aktuell gemessene Kapazität ins Verhältnis zu einer nominalen oder anfänglichen Referenz. Energie hängt zusätzlich vom Spannungsverlauf und den Betriebsgrenzen ab. Gleicher SOC bedeutet deshalb nicht automatisch gleiche verfügbare Energie bei unterschiedlichen Zellen oder Alterungszuständen.

**Herleitung:** Ladung ist das Integral des Stroms, elektrische Energie das Integral von Spannung mal Strom. Kapazitätsverlust und Widerstandsanstieg wirken unterschiedlich auf Energienutzung und Leistungsfähigkeit. Die Dissertation untersucht primär SOC und kapazitätsbezogenen SOH, nicht eine vollständige Leistungs- oder Sicherheitszustandsschätzung. **Beleg:** Kapitel 3.1 bis 3.2 und 4.1.3.

## F008. Warum ist LFP für spannungsbasierte SOC-Schätzung schwierig?

**Antwort:** Auf dem breiten OCV-Plateau ändert sich die Gleichgewichtsspannung nur wenig mit SOC. Damit erzeugt ein SOC-Unterschied nur ein kleines Spannungssignal, das von Sensorfehlern, Polarisation und Hysterese überlagert werden kann. Eine lokale Näherung lautet: SOC-Unsicherheit ist ungefähr Spannungsunsicherheit geteilt durch die OCV-SOC-Steigung.

**Grenze:** Eine kleine Steigung bedeutet eingeschränkte lokale Spannungsinformation, nicht dass LFP grundsätzlich unschätzbar wäre. Zeitliche Dynamik, Stromintegration, informative Randbereiche und zusätzliche Zustandsinformation können helfen. Der Datensatz belegt keine eindeutige Zerlegung aller dieser Informationsquellen. **Beleg:** Kapitel 3.1.1 und 3.3. **Übung:** Erkläre, warum eine große Spannungsänderung unter Last nicht automatisch eine große SOC-Änderung ist.

## F009. Warum kann gemessener SOH zwischendurch steigen?

**Antwort:** Das Diagramm zeigt verfügbare Kapazität aus diskreten Check-ups, nicht direkt die Menge irreversibel verlorenen aktiven Materials. Konditionierung, Relaxation, Temperatur, Abschaltkriterien und Messunsicherheit können einen späteren Kapazitätstest höher ausfallen lassen. Die lineare Interpolation übernimmt diese Schwankungen zwischen den Ankern.

**Wichtige Grenze:** Aus einer ansteigenden Linie lässt sich weder eine Selbstheilung noch genau eine Ursache ableiten. Die Arbeit nennt mehrere plausible Beiträge, isoliert sie aber nicht. Eine saubere Prüfung würde Check-up-Bedingungen, Rohstromintegration und Temperatur vergleichen und Wiederholungsmessungen auswerten. **Beleg:** Abbildungen 4.1 und 4.4. Dies ist besonders für die U-förmige NMC-Zelle 9 wichtig.

## F010. Wie unterscheiden Sie Kalenderalterung und zyklische Alterung?

**Antwort:** Kalenderalterung hängt unter anderem von verstrichener Zeit, Temperatur und Lager-SOC ab. Zyklische Alterung hängt zusätzlich von Durchsatz, Stromraten, Entladetiefe und Betriebsgrenzen ab. In einem laufenden Zyklierexperiment treten beide Beiträge gleichzeitig auf.

**Vertiefung:** Eine Darstellung über äquivalente Vollzyklen entfernt den Zeiteinfluss nicht. Schnell zyklierte Zellen erreichen denselben Durchsatz in anderer Zeit als langsam zyklierte. Um getrennte kausale Beiträge zu bestimmen, braucht man geeignete Kontrollbedingungen, etwa Kalenderlagerung und ein identifizierbares Versuchsdesign. Die vorliegenden Kurven erlauben zunächst einen Vergleich unter den gewählten Gesamtbedingungen.

## F011. Wie funktioniert Coulomb Counting und woher kommt Drift?

**Antwort:** Das Verfahren integriert den gemessenen Strom und normiert die Ladungsänderung durch die angenommene Kapazität. Bei positiver Laderichtung steigt SOC, bei positiver Entladerichtung fällt er. Eine Gleichung ist nur zusammen mit dieser Vorzeichenkonvention sinnvoll. Fehler entstehen aus Anfangszustand, Stromoffset, Gain, Zeitbasis, Integrationsnäherung und Kapazitätsannahme.

**Herleitung:** Ein konstanter Stromoffset b erzeugt ohne Korrektur einen SOC-Fehler mit Betrag b mal Zeit in Stunden geteilt durch Kapazität in Ah. Ein Gainfehler wirkt dagegen auf das vorzeichenbehaftete Stromintegral. Daher müssen beide Fehlerarten getestet werden. Die Dissertation verwendet in Kapitel 6 einen ladungspositiven Qc-Zustand und SOC = 1 + Qc/C. **Lernquelle:** Movassagh et al., Coulomb-Counting-Fehleranalyse, Quellenabschnitt.

## F012. Warum begrenzt ein Full-Charge-Reset die Drift nicht vollständig?

**Antwort:** Ein Reset setzt einen Zustand an einem erkannten Ladeanker zurück. Danach beginnt die Integration gestörter Ströme erneut. Zudem kann die Erkennung selbst durch Spannung, Timing und Messfehler beeinflusst werden. Die Schätzung kann nach einem Anker korrekt sein und später wieder abweichen.

**Vertiefung:** Clipping auf 0 bis 1 ist kein physikalisch nachgewiesener Reset. Clipping kann Fehler im Ausgang unsichtbar machen, obwohl intern noch ein Ladungsfehler existiert. In einem rekurrenten Modell kann außerdem eine weitere fehlerbehaftete Historie bestehen. **Beleg:** Kapitel 6.4.2 und Abbildung 6.6. Deshalb nicht sagen, ein einzelner Volladepunkt lösche jeden über die Lebenszeit entstandenen Fehler.

## F013. Was bedeuten die beiden RC-Zweige im HECM?

**Antwort:** Sie approximieren dynamische Spannungsanteile mit unterschiedlichen Zeitkonstanten. Ein ohmscher Widerstand beschreibt den unmittelbaren Strom-Spannungs-Anteil, die RC-Zweige verzögerte Polarisation. Im diskreten Modell steht exp(-Δt/τ) für das Abklingen eines vorherigen RC-Zustands.

**Grenze:** Zwei RC-Zweige sind eine reduzierte terminale Beschreibung. Sie identifizieren nicht eindeutig jeweils einen einzelnen elektrochemischen Prozess. Ob ein oder drei Zweige besser wären, müsste unter gleichen Daten- und Ressourcenbedingungen untersucht werden. **Übung:** Für τ = 10 s und Δt = 1 s bleiben nach einem Schritt rund 90,5 Prozent eines ungetriebenen Zustands übrig. Nach 10 s sind es rund 36,8 Prozent.

## F014. Erklären Sie den EKF ohne Formelsammlung.

**Antwort:** Zuerst sagt das Modell aus dem bisherigen Zustand und dem Strom den nächsten Zustand und die zu erwartende Spannung voraus. Dann wird die vorhergesagte mit der gemessenen Spannung verglichen. Der Kalman-Gain verteilt dieses Residuum auf die Zustandskorrektur, gewichtet mit geschätzter Modell- und Messunsicherheit.

**Vertiefung:** P beschreibt Zustandsschätzunsicherheit, Q Prozessunsicherheit und R Messunsicherheit. Die Jacobimatrizen linearisieren die nichtlinearen Zusammenhänge lokal. Ein größeres R verringert bei sonst gleichen Bedingungen den Einfluss der Messung. Das ist keine Garantie für einen besseren Filter. Falsch spezifizierte Unsicherheiten oder ein falsches OCV-Modell können systematische Fehler erzeugen. **Beleg:** Kapitel 3.3, Gleichungen 3.3 bis 3.4 im PDF prüfen.

## F015. Warum nicht UKF, elektrochemisches Modell oder neuronaler Observer?

**Antwort:** Der 2RC-EKF bietet einen verbreiteten, überschaubaren Repräsentanten mit expliziter Spannungskorrektur. Ein UKF vermeidet explizite Jacobimatrizen, benötigt aber mehrere Modellfortschreibungen. Ein elektrochemisches Modell kann reichere Zustände darstellen, verlangt mehr Parametrierung und meist mehr Rechenaufwand. Ein neuronaler Observer wäre ein weiterer Hybrid mit eigenem Training.

**Grenze:** Die Wahl ist eine Begrenzung des Vergleichsraums. Aus den Ergebnissen lässt sich keine Überlegenheit gegenüber nicht untersuchten Beobachtern ableiten. Eine gute Anschlussstudie vergleicht alternative Observer mit demselben Referenzziel, denselben Zellgrenzen und derselben Hardwaremessung.

# 3. Daten, Labels und experimentelle Aussagekraft

## F016. Welche Datensätze verwenden Sie genau?

**Antwort:** Kapitel 5 verwendet 14 NMC-Zellen mit 2,6 Ah nominaler Kapazität, sieben Alterungsszenarien und je zwei Zellen. Variiert werden mittlerer SOC, DOD, Entladerate und 35 beziehungsweise 45 °C. Kapitel 6 und 7 verwenden 15 LFP-Zellen mit 1,8 Ah aus der MG-FARM-DoE-Kampagne, mit nominal 25 °C Umgebung sowie variierter Lade- und Entladerate und DOD.

**Vertiefung:** Die gemessene Zelltemperatur ist nicht auf 25 °C beschränkt. Beispielsweise dokumentiert die Split-Tabelle für C29 bis 49,4 °C. Umgebungssollwert und Sensortemperatur dürfen nicht verwechselt werden. Die unterschiedlichen Datensätze erlauben keine direkte Rangfolge ihrer Modell-MAE. **Beleg:** Tabelle 4.1, Tabelle 4.3 und Kapitel 4.1.

## F017. Wie lauten die Zellgrenzen des Robustheitsbenchmarks?

**Antwort:** Training: C01, C03, C05, C11, C17, C23. Validation: C07, C19, C21. Test: C09, C13, C15, C25, C27, C29. Die Entwicklungsgrenze gilt für neuronale Modelle, Scaler, Pruning und Lookup-Identifikation.

**Nachfrage:** Sind das automatisch auch die Splits jeder anderen Studie? Nein. Die Dissertation dokumentiert den sechszelligen Split ausdrücklich für Kapitel 6. Kapitel 5 hat ein anderes NMC-Schema. Für Kapitel 7 muss die genaue Zuordnung aus der zugehörigen Trainingskonfiguration belegt werden, bevor die Robustheitssplits darauf übertragen werden. **OFFEN:** Eine Backup-Folie sollte für jede Studie separat Training, Validation, Test und gezeigte Beispielzelle enthalten.

## F018. Ist der Referenz-SOC wirklich Ground Truth?

**Antwort:** Er ist die operational definierte Dataset-Referenz, die aus Stromintegration, Kapazitätsinformationen und bestätigten Ladeankern aufgebaut wurde. Er ist keine direkte, fehlerfreie Messung des inneren SOC. Der Begriff Ground Truth kann als Bezeichnung des Auswertungsziels dienen, wenn seine Konstruktion und Unsicherheit ausdrücklich erklärt werden.

**Vertiefung:** Modell und Ziel können gemeinsame Messfehler oder Integrationsannahmen teilen. Ein niedriger MAE belegt Übereinstimmung mit dieser Referenz, nicht automatisch höchste physikalische Wahrheit. Unabhängige Referenzvalidierung könnte hochgenaue Strommessung, Unsicherheitsrechnung und kontrollierte Endpunkte umfassen. **Beleg:** Kapitel 4.1.3. **Nicht sagen:** Weil nichts Besseres verfügbar ist, sei die Referenz unabhängig oder fehlerfrei.

## F019. Warum zwei SOH-Normierungen und welche Falle entsteht dadurch?

**Antwort:** NMC und die Embedded-Studie beschreiben Kapazität relativ zur Nennkapazität. Der Robustheitsbenchmark normiert auf die erste gemessene Referenzkapazität der Trajektorie. Beide Größen sind kapazitätsbezogen, aber numerisch nicht automatisch identisch.

**Herleitung:** Bei Cnom = 1,8 Ah, Cref,0 = 1,9 Ah und C = 1,52 Ah ist nominaler SOH 0,8444 und relativer SOH 0,8. Für die Rekonstruktion der Kapazität muss der passende Nenner wieder verwendet werden. Cnom mal relativer SOH wäre hier 1,44 Ah statt 1,52 Ah. **OFFEN:** Kapitel 6 schreibt für HDM Ceff = Cnom mal SOH. Vor der Verteidigung klären, auf welcher Basis das tatsächlich eingespeiste LSTM-SOH normiert ist. Unterschiedliche Konventionen können korrekt sein, müssen aber konsistent ineinander überführt werden.

## F020. Ist eine offline interpolierte SOH-Referenz ein Kausalitätsverstoß?

**Antwort:** Nicht automatisch. Ein Label darf retrospektiv aus später verfügbaren Referenzmessungen erstellt werden. Eine Online-Schätzung darf diese späteren Informationen hingegen nicht als Eingang erhalten. Entscheidend ist die Trennung zwischen Zielkonstruktion und Featurepipeline.

**Vertiefung:** Interpolation glättet den Zielverlauf und begrenzt, welche Dynamik überhaupt bewertet wird. Eine Schätzung, die zwischen seltenen Check-ups glatt verläuft, ist damit noch nicht als Echtzeitsensor für schnelle Kapazitätsänderungen validiert. **OFFEN:** Für jedes Modell dokumentieren, welche Kapazitätstests maskiert wurden, welche Diagnosephasen als Eingänge verbleiben und wie initialer SOH verfügbar gemacht wird. Die Initialausrichtung von Kapitel 7 verwendet y0 und ist eine zusätzliche Annahme.

## F021. Warum zufällige Zeitfenster nicht als unabhängige Zellen zählen?

**Antwort:** Benachbarte Zeitfenster teilen Zellchemie, Fertigungseigenschaften, Historie und oft große Teile derselben Signale. Sie sind korreliert. Millionen Samples oder viele Seeds ersetzen deshalb keine zusätzlichen unabhängigen Zellen.

**Vertiefung:** Zell-disjunkte Testdaten prüfen Transfer auf neue Zellen innerhalb der abgedeckten Domäne. Ein zusätzlicher betriebsbedingungs-disjunkter Split würde stärker prüfen, ob neue Last- oder Temperaturregime generalisiert werden. Auch verschiedene Zellen aus derselben Charge sind nicht automatisch repräsentativ für die gesamte Produktpopulation. **Beleg:** Kapitel 6.2.6 und Grenzen in Kapitel 8.

## F022. Sind Low, Middle und High physikalische Alterungsklassen?

**Antwort:** Nein. Es sind deskriptive Lastgruppen anhand des gemessenen P95 des absoluten C-Rates über die ganze Trajektorie. Low umfasst C25 und C27, Middle C09, C13 und C15, High nur C29. Das beschreibt Belastungsabdeckung und ist keine isolierte Kausalanalyse der Alterung.

**Vertiefung:** P95 ist weniger empfindlich gegenüber einzelnen Maximalwerten als der absolute Peak, hängt aber von der Mischung aus Zyklierung und Diagnosephasen ab. C25 und C27 unterscheiden sich auch innerhalb Low deutlich. Für High gibt es keine zwischenzellige Streuung, weil n = 1. SOH, Last, Temperatur und Betriebsdauer können miteinander gekoppelt sein. **Beleg:** Tabelle 4.3 und Abbildung 4.5.

## F023. Warum gibt es 16 und nicht 18 Zell-SOH-Fenster?

**Antwort:** Sechs Zellen mal drei SOH-Bereiche ergäben 18 Kombinationen. C27 bleibt im betrachteten Datensatz vollständig im frischen Bereich, sodass zwei Kombinationen fehlen. Es werden nur beobachtete Zustände ausgewertet und keine künstlichen aged-Fenster ergänzt.

**Vertiefung:** Dadurch sind die Altersgruppen nicht vollständig balanciert. Ein Unterschied zwischen fresh und aged kann zum Teil daran liegen, dass unterschiedliche Zellen beitragen. Eine alterskausale Aussage verlangt innerhalb derselben Zellen gepaarte Vergleiche und die Kontrolle anderer Betriebsänderungen. **Beleg:** Kapitel 4.1.2 und 6.3.

## F024. Wie wurde verhindert, dass besonders günstige Testfenster ausgewählt wurden?

**Antwort:** Innerhalb jeder vorhandenen Zell-SOH-Kombination wird ein gemessenes Fenster anhand von SOH, Temperatur, P95-C-Rate, Durchsatz und Anteil niedrigen SOCs gewählt. Der Abstand zur robust skalierten typischen Merkmalslage entscheidet, nicht der Modellfehler. Damit ist die Auswahl nicht direkt auf einen Estimator optimiert.

**Kritische Präzisierung:** Die dargestellte Formel wählt den Punkt mit geringstem Abstand zum Koordinatenmedian, nicht zwingend den klassischen Medoid mit minimaler Summe aller paarweisen Abstände. Das sollte terminologisch erklärt werden. **OFFEN:** Wie behandelt das Skript eine MAD von null? Welche SOH-Grenzen gelten genau? Warum repräsentieren typische Fenster auch seltene gefährliche Zustände? Letzteres tun sie nicht vollständig. Daher ergänzen Störungsversuche und Ereignisanalysen die Auswahl.

## F025. Was bringt das DoE und was beweist es nicht?

**Antwort:** Das DoE verteilt endliche Versuchskapazität strukturiert über Lade-C-Rate, Entlade-C-Rate und Entladetiefe. Dadurch entstehen systematisch verschiedene Betriebsbedingungen statt zufälliger Einzelprofile. Die Arbeit nutzt diese Diversität für Zustandsschätzung und Robustheitsbewertung.

**Grenze:** Die Darstellung eines DoE-Würfels ist noch keine statistische Identifikation aller Haupteffekte und Interaktionen. Dafür braucht man den genauen Designplan, Wiederholungen, Auswertungsmodell und gegebenenfalls eine Alias-Struktur bei fraktionellen Designs. Die drei Faktoren sind nicht die einzigen physikalischen Einflussgrößen. Alterung, Eigenerwärmung und Fertigungsstreuung bleiben relevant.

# 4. SOH-MLP und zeitliche Merkmale

## F026. Wie erhält ein speicherloses MLP zeitliche Information?

**Antwort:** Die Historie wird vor dem Netz in einen festen Featurevektor geschrieben. Lag-Sequenzen enthalten vergangene Spannungs-, Temperatur- und kumulierte Stromwerte. Das Netz verarbeitet diese gemeinsam, ohne einen internen rekurrenten Zustand zwischen zwei Aufrufen zu benötigen.

**Vertiefung:** Speicherlos ist nur das Netz als Funktion. Das Gesamtsystem benötigt den Historienpuffer, Resampling und kumulative Zustände. Der Vergleich lautet daher nicht mit oder ohne Gedächtnis, sondern explizit repräsentierte versus intern gelernte Historie. **Beleg:** Abbildungen 5.2 und 5.4. **Nachfrage:** Warum nicht aktueller Strom allein? Durchsatzmerkmale enthalten längerfristige Betriebsinformation, können aber auch als Zeit- oder Altersproxy fungieren.

## F027. Warum getrennte kumulierte Lade- und Entladeströme?

**Antwort:** Ein vorzeichenbehaftetes Gesamtintegral kann nach einem vollständigen Zyklus wieder nahe null liegen, obwohl die Zelle erheblichen Durchsatz erfahren hat. Getrennte oder betragsbezogene Integrale erhalten diese Beanspruchungsinformation. Die physikalische Relevanz liegt in der Vorgeschichte, nicht in einer direkten Gleichsetzung von Ah und Kapazitätsverlust.

**Grenze:** Korrelation mit SOH kann durch gemeinsame Zeitabhängigkeit entstehen. Für einen Nachweis des zusätzlichen Merkmalsnutzens wären Ablationen gegen Zeit, Zykluszahl, kumulierten Durchsatz und einfache Regressionsbaselines hilfreich. **Beleg:** Kapitel 5.2.1 bis 5.2.2. Die Vorzeichen und Resetregeln dieser NMC-Merkmale nicht mit Qc des SOC-Benchmarks vermischen.

## F028. Bedeutet die rote Korrelationsfarbe positive Korrelation?

**Antwort:** In Abbildung 5.1 ausdrücklich nicht. Die Farbstärke kodiert den Betrag des Pearson-Koeffizienten. Sowohl -1 als auch +1 liegen am roten Ende, null am blauen. Das Vorzeichen muss aus der gedruckten Zahl gelesen werden.

**Vertiefung:** Pearson misst linearen Zusammenhang und ist empfindlich gegenüber Trend, Ausreißern und Mischungen verschiedener Zellen. Hohe Korrelation beweist weder Ursache noch Nichtredundanz. Ein Netz kann aus zwei stark korrelierten Merkmalen dennoch unterschiedliche Informationen gewinnen, oder nur dieselbe Zeitentwicklung doppelt sehen. **Übung:** Erkläre an einer negativen Durchsatz-SOH-Korrelation, warum ein positives Durchsatzsignal einen sinkenden Zielwert begleiten kann.

## F029. Warum gerade 16 Lag-Schritte und 10 Minuten?

**Antwort:** Das ist das im untersuchten Raster günstige Setting. Es verbindet eine relativ breite Historie mit einer zeitlichen Aggregation, die noch Informationen über typische Lade- und Entladephasen bewahrt. Sehr grobe Aggregation kann Dynamik verwischen, sehr kurze Intervalle können stark redundante Eingänge liefern.

**Präzisierung:** Die Arbeit nennt 160 Minuten. Bei 16 punktförmigen Stützstellen inklusive aktuellem Zeitpunkt beträgt der Abstand vom ältesten bis zum neuesten Punkt dagegen 15 mal 10 = 150 Minuten. 160 Minuten kann korrekt die Abdeckung von 16 aufeinanderfolgenden 10-Minuten-Bins bezeichnen. **OFFEN:** Die genaue Bin- und Indexkonvention am Skript prüfen. Das ist eine Definitionsfrage, keine Widerlegung des Befunds. **Beleg:** Abbildung 5.5 und Kapitel 5.5.

## F030. Ist Ihr MLP tatsächlich klein?

**Antwort:** Es ist strukturell einfach, weil es ein Feedforward-Netz ohne rekurrente Zustandsverwaltung ist. Die finale Tabelle listet aber neun versteckte Schichten mit 128, 256, 256, 256, 128, 512, 128, 256 und 256 Neuronen. Deshalb sollte einfach nicht mit nachgewiesen geringer Parameterzahl verwechselt werden.

**Herleitung:** Unter der Annahme von 48 Eingängen, also drei Features mal 16 Lag-Werte, ergeben die dichten Schichten einschließlich Bias 434561 Parameter. Bei FP32 sind das rund 1,74 MB reine Parameterdaten. Der Wert ist eine Architektur-Rechnung, kein verifizierter Modellexport. **OFFEN:** Tatsächliche Eingangsdimension und gespeichertes Modell prüfen. Der anfängliche Suchkandidat mit acht Neuronen beschreibt nicht das finale Modell. **Beleg:** Tabelle 5.1.

## F031. Was macht Hyperband und warum wurde es verwendet?

**Antwort:** Hyperband verteilt ein begrenztes Trainingsbudget auf viele Konfigurationen und verwirft schwache Kandidaten früh. Es kombiniert verschiedene Startbudgets und aufeinanderfolgende Ausleseschritte. Gesucht wurden unter anderem Historienlänge, Auflösung, Tiefe, Breite und Trainingsparameter.

**Grenze:** Frühe Trainingsleistung ist nicht immer ein zuverlässiger Prädiktor späterer Leistung. Ein langsamer konvergierender guter Kandidat kann zu früh ausscheiden. Die Auswahl muss an Validation erfolgen und das finale Testset unberührt lassen. **OFFEN:** Wenn Abbildung 5.5 Testzellen zeigt, darf deren Optimum nicht als ausschließlich validation-basierte Auswahl dargestellt werden, bevor die historische Auswertung geklärt ist. **Lernquelle:** Hyperband-Originalarbeit im Quellenabschnitt.

## F032. Wie zuverlässig ist die Validation mit 20 Prozent der Trainingssequenzen?

**Antwort:** Die vier vollständig zurückgehaltenen NMC-Testzellen schaffen eine wichtige unabhängige Testgrenze. Innerhalb des Trainingspools können zufällig getrennte, überlappende Sequenzen jedoch eine optimistische Validation erzeugen. Das hängt von der tatsächlichen Aufteilung und Fensterüberlappung ab.

**OFFEN:** Die Dissertation nennt keine vollständige genaue Sequenz-Partition in diesem Absatz. Zu prüfen sind zellweise oder zeitblockweise Trennung, Überschneidungen und zeitlicher Abstand zwischen Fenstern. Für eine strengere Wiederholung wären gruppierte Splits oder zeitliche Sperrzonen sinnvoll. Nicht behaupten, dass eine potenzielle Validierungsleckage tatsächlich vorlag, ohne die Skripte zu prüfen.

## F033. Warum ist die U-förmige Zelle 9 schlecht geschätzt?

**Antwort:** Das Modell folgt nur Teilen der Referenzentwicklung und hat größere absolute und systematische Abweichungen. Die U-Form allein erzeugt keinen Fehler. Würde sie exakt reproduziert, wären MAE und MSE klein. Der Befund spricht für eine Grenze der Generalisierung unter dieser ungewöhnlichen Trajektorie.

**Beleg:** Kapitel 5.5 nennt MAE 2,528 Prozentpunkte und MSE 8,327 Quadrat-Prozentpunkte für Zelle 9. Mögliche Ursachen sind unzureichende Trainingsabdeckung, Label-/Check-up-Besonderheiten und eine unpassende Merkmalsrepräsentation. Welche davon dominiert, ist nicht isoliert belegt. **Nicht sagen:** Das Modell versagt, weil SOH niemals steigen dürfe.

## F034. Was zeigen die guten NMC-Fehlerwerte wirklich?

**Antwort:** Für günstige Einzelzellen nennt die Arbeit MAE 0,487 und 0,253 Prozentpunkte, im Mittel 1,10 Prozentpunkte. Das ist eine Genauigkeitsaussage gegenüber der kapazitätsbasierten Referenz unter den untersuchten Bedingungen. 1,10 Prozentpunkte entsprechen 0,0110 auf der normierten Skala.

**Vertiefung:** MSE skaliert quadratisch. 2,95 Quadrat-Prozentpunkte entsprechen 0,000295 in normierten Einheiten, nicht 0,0295. Die dargestellte Glättung darf nicht mit ungefilterter Netzleistung gleichgesetzt werden. **OFFEN:** Vortragen können, auf welchen Zellen und welchen Roh-/Filterausgaben die angegebenen Mittel beruhen. Drei gezeigte Beispielzellen sind nicht automatisch das vollständige Testset aus vier Zellen.

## F035. Warum verbessert Entfernen problematischer Trainingszellen nicht alles?

**Antwort:** Das Entfernen von C3/C4 im NMC-Versuch reduziert mögliche unregelmäßige Daten, entfernt aber auch Variabilität. Der mittlere MAE sinkt laut Arbeit nur von 1,10 auf 1,07 Prozentpunkte, während die atypische Zelle 9 instabiler geschätzt wird. Datenbereinigung und Repräsentativität können gegensätzliche Effekte haben.

**Grenze:** Ohne Wiederholungen über Trainingsseeds ist diese kleine Differenz nicht automatisch statistisch belastbar. Eine klare Antwort lautet, dass die Untersuchung keinen einfachen Zusammenhang weniger problematische Daten gleich bessere Generalisierung zeigt. Dafür bräuchte man gepaarte Wiederholung, dokumentierte Auswahlregeln und Unsicherheiten.

## F036. Ist dies Alterungsprognose oder Zustandsschätzung?

**Antwort:** Das Netz schätzt den aktuellen SOH aus bis dahin vorliegenden Betriebsmerkmalen. Eine über die Zeit gezeichnete Folge solcher Schätzungen ist noch keine Vorhersage zukünftiger Restlebensdauer. Eine Prognose braucht einen Zukunftshorizont und Annahmen über die zukünftige Nutzung.

**Nachfrage:** Kann es für eine neue Chemie benutzt werden? Die Verfahrensidee ist übertragbar, die gelernten Gewichte und optimalen Zeitskalen nicht ohne Validierung. NMC- und LFP-Spannungsverläufe und Alterungsmechanismen unterscheiden sich. Neu trainieren oder anpassen und zell-disjunkt testen wäre der saubere Weg.

# 5. Robustheitsbenchmark und Recovery

## F037. Erklären Sie die vier Repräsentanten in einer Minute.

**Antwort:** DM integriert Strom mit fester Kapazität und einem Volladeanker. HDM benutzt denselben Kern, skaliert die Kapazität aber mit geschätztem SOH. HECM ergänzt eine 2RC-Zustandsbeschreibung, SOH-abhängige Tabellen und EKF-Spannungskorrektur. DD ist eine geprunte GRU, die Messkanäle und abgeleitete Merkmale gemeinsam aus einer kausalen Sequenz auf SOC abbildet.

**Präzisierung:** HDM, HECM und DD erhalten denselben kausalen SOH-Verlauf. Das kontrolliert einen gemeinsamen Eingang, macht die Modelle aber nicht identisch empfindlich gegen SOH-Fehler. DD ist kein raw-signal-only-Modell, weil Qc, Ableitungen, Zeitabstand und SOH bereits Vorwissen enthalten. **Beleg:** Kapitel 6.2.1 bis 6.2.3.

## F038. Wie sieht die DD-GRU exakt aus?

**Antwort:** Der Ausgangspunkt hat 96 rekurrente Einheiten. Strukturelles Pruning entfernt 29 vollständige Einheiten und zugehörige Verbindungen, sodass 67 bleiben. Danach folgen ein MLP mit 96 versteckten Einheiten und ein Sigmoid-Ausgang. Die Eingänge sind U, I, T, SOH, Qc, dU/dt, dI/dt und Δt. Im primären Benchmark verwendet jede Ausgabe die letzten 2024 Samples und beginnt die Fensterrechnung mit zurückgesetztem rekurrentem Zustand.

**Grenze:** Das ist nicht das 64er SOC-LSTM aus Kapitel 7. Die 30,2 Prozent Breitenreduktion entspricht auch nicht exakt einer 30,2-prozentigen Laufzeit- oder Parameterreduktion. Quadratische rekurrente Matrizen verändern sich stärker. **Beleg:** Kapitel 6.2.2.

## F039. Warum bekommt DD Qc, obwohl daraus direkt SOC berechnet werden kann?

**Antwort:** Qc ist ein kausal integriertes Ladungsmerkmal und damit ein starker, physikalisch sinnvoller SOC-Prädiktor. DD kann diese Information mit Spannungs-, Temperatur- und Zeitkontext verbinden. Es ist legitim, dieses Merkmal zu nutzen, solange seine Konstruktion online verfügbar ist und dieselben Störungen erfährt wie die Rohmessungen.

**Kritische Grenze:** Wenn das eingegebene Qc bereits offline driftkorrigierte Zielinformation enthielte, wäre der Vergleich problematisch. Deshalb trennt die Arbeit Online-Qc von der Dataset-Referenz. Ohne Feature-Ablation ist nicht gemessen, welcher Anteil des Erfolgs allein auf Qc oder auf die GRU zurückgeht. Eine sinnvolle Vergleichsbaseline wäre ein einfacher Regressor mit denselben Features.

## F040. Was bedeutet gemeinsame kausale SOH-Spur?

**Antwort:** Aus fünf Basissignalen werden stündlich Mittelwert, Standardabweichung, Minimum und Maximum gebildet. Diese 20 Merkmale gehen in ein gemeinsames SOH-Netz mit Feature-Embedding, zweilagigem LSTM und MLP-Blöcken. Seine Zustände laufen kausal weiter, und seine Ausgabe wird zwischen Stundenupdates gehalten. Alle SOH-abhängigen SOC-Zweige erhalten diese identische Information.

**Vertiefung:** Vor dem Test gibt es 192 Stunden ungestörten Kontext. Nach Störungsbeginn werden betroffene abgeleitete Merkmale aus dem gestörten Stream rekonstruiert. Das ist wichtig, weil andernfalls künstlich perfekte Nebenkanäle verbleiben könnten. **Grenze:** Gleiche SOH-Spur kontrolliert den Eingang, beseitigt aber keine SOH-Unsicherheit. Im Hardwareteil von Kapitel 6 wird sie extern geliefert, nicht auf dem Controller berechnet.

## F041. Wie unterscheiden sich Accuracy, Robustness und Recovery?

**Antwort:** Accuracy misst die Abweichung vom Referenz-SOC unter nominalen Messungen. Robustness beschreibt die zusätzliche Verschlechterung unter korrumpierten oder fehlenden Eingängen. Recovery beschreibt, wie eine initial gestörte Schätzung wieder an die entsprechend korrekt initialisierte Schätzung herankommt.

**Wichtig:** Recovery in dieser Arbeit ist eine gepaarte Modell-zu-Modell-Abweichung, keine Garantie kleiner Abweichung vom Dataset-SOC. Zwei gleich falsche Trajektorien können vollständig zueinander recovern. Deshalb muss die Accuracy separat gezeigt werden. **Beleg:** Kapitel 6.2.6 und Abbildung 6.9.

## F042. Wieso haben alle Modelle denselben Auswertungsbeginn?

**Antwort:** Die Rolling-GRU liefert erst nach ausreichender Historie eine gültige Ausgabe. Damit frühe schwierige oder leichte Samples nicht nur bei manchen Modellen in den Fehler eingehen, beginnt die gemeinsame Auswertung erst mit der ersten gültigen DD-Ausgabe. Alle Modelle werden auf demselben Zeitintervall verglichen.

**Grenze:** Das blendet einen Teil des realen Kaltstartproblems aus. Recovery-Zeit ist weiterhin auf die Intervention bezogen, während der gemeinsame beobachtete Bereich später beginnt. Die 2024 Samples sind bei 1 Hz ungefähr 33,7 Minuten. Sehr frühe Unterschiede und Recovery müssen deshalb hinsichtlich Beobachtungsgrenze vorsichtig interpretiert werden.

## F043. Was genau wird bei Initialisierungsfehlern verändert?

**Antwort:** Es wird ein SOC-äquivalenter Fehler von -0,10 eingebracht. Bei DM/HDM betrifft dies den zugänglichen Integrationszustand, bei HECM den expliziten SOC-Zustand, bei DD einen Qc-Offset von -0,10 mal nominaler Kapazität. DD besitzt keinen separaten SOC-Speicher, der direkt um zehn Prozentpunkte verschoben werden könnte.

**Grenze:** Die Interventionen sind funktional vergleichbar, aber nicht identisch in ihren latenten Zuständen oder in ihrer unmittelbaren Ausgangswirkung. Bei HDM hängt die SOC-Wirkung eines Ladungsoffsets zusätzlich vom Kapazitätsnenner ab. Deshalb ist es ein operationeller Re-Anker-Test und kein universeller Konvergenzvergleich identischer Anfangsfehler. **Beleg:** Kapitel 6.3 und 6.4.5.

## F044. Was bedeuten 0,02, 300 Sekunden und persistent?

**Antwort:** Zunächst muss der Betrag der Differenz zwischen gestörter und sauber initialisierter Ausgabe höchstens 0,02 SOC, also zwei Prozentpunkte, für mindestens 300 Sekunden betragen. Das ist ein erster qualifizierter Bandeintritt. Persistent ist die Rückkehr erst, wenn die Differenz danach bis zum Ende des 24-Stunden-Horizonts innerhalb des Bandes bleibt.

**Beispiel:** DD tritt nach einer Stunde für zehn Minuten ein, verlässt das Band wieder und bleibt erst nach drei Stunden dauerhaft darin. First entry ist eine Stunde, persistent recovery drei Stunden. Läuft der Versuch ohne dauerhafte Rückkehr aus, ist er rechtszensiert. **Grenze:** Das nachträgliche persistent-Kriterium ist eine Auswertedefinition. Online kann man noch nicht wissen, ob später ein Rückfall kommt.

## F045. Warum kann DD kurz recovern und danach wieder schlechter werden?

**Antwort:** DD kombiniert Qc mit wechselnden Messverläufen innerhalb eines rollenden Fensters. Die Vorhersage kann zeitweise wenig und später wieder stärker vom fehlerhaften Merkmal abhängen. Außerdem ändern sich SOC-Region und Dynamik. Der erste Bandeintritt beweist keine dauerhafte Entfernung aller Ursachen.

**Beleg:** DD hat eine Rückfallfraktion von 0,78 gegenüber 0,17 bei HECM. Die beobachteten persistenten Recovery-oder-Zensur-Mittel sind 2,60 h für DD und 1,20 h für HECM. **Grenze:** Die Feature-Erklärung ist plausibel, aber keine gemessene Attribution ohne Ablation. DM/HDM haben jeweils eine hohe persistente Zensurfraktion von 0,89.

## F046. Warum darf man zensierte 24 Stunden nicht als echte Recovery interpretieren?

**Antwort:** Bei einem zensierten Lauf ist nur bekannt, dass Recovery bis zum Ende nicht beobachtet wurde. Der tatsächliche Zeitpunkt könnte später liegen oder nie eintreten. Der als Recovery-or-censor zusammengefasste Wert ist daher eine begrenzte Beobachtungskennzahl, nicht die mittlere physikalische Konvergenzzeit.

**Vertiefung:** Eine Survival-Auswertung könnte Zeit-bis-Ereignis mit Zensierung expliziter behandeln. Mit sechs Zellen bliebe ihre Unsicherheit groß. Für die Verteidigung die Zensurquote immer zusammen mit der Zeit nennen. Gerade ein Balken bei rund 24 h bedeutet häufig keine erfolgreiche Recovery nach 24 h.

## F047. Welche nominalen Zahlen müssen Sie sicher kennen?

**Antwort:** Im Robustheitsbenchmark beträgt der zellweise gleichgewichtete MAE DM 0,0690, HDM 0,0412, HECM 0,0337 und DD 0,0258. Die RMSE-Werte sind 0,0740, 0,0442, 0,0388 und 0,0300. Das entspricht MAE von 6,90, 4,12, 3,37 und 2,58 Prozentpunkten.

**Grenze:** Es sind sechs Zellen und 16 standardisierte Fenster, nicht alle Samples aller Lebenszeiten gleich gewichtet. Die Hierarchie ist für diese Vertreter nominal klar, aber die Holm-korrigierten Signifikanztests liegen über 0,05. Deshalb praktische Effektgröße und Unsicherheit nennen, nicht eine pauschal statistisch gesicherte universelle Überlegenheit. **Beleg:** Abbildung 6.4.

## F048. Warum kann ein Bias den MAE verbessern?

**Antwort:** MAE vergleicht mit einer Referenz, und die nominale Schätzung kann bereits einen gerichteten Fehler haben. Eine zusätzliche Störung kann diesen Fehler teilweise kompensieren. Das ist keine Aussage, dass ungenauere Sensoren grundsätzlich besser wären.

**Herleitung:** Hat die nominale Schätzung Fehler +0,04 und verschiebt die Störung sie um -0,02, sinkt der absolute Fehler von 0,04 auf 0,02. Bei +0,02 zusätzlicher Verschiebung steigt er auf 0,06. Deshalb werden Gain und Stromoffset mit beiden Vorzeichen getestet. **Nicht tun:** Ergebnisse oder Vorzeichen nach gewünschter Schlussfolgerung auswählen, ohne die adverse Richtungsdefinition offenzulegen.

## F049. Was bedeutet adverse-direction und ist das ein Worst-Case-Beweis?

**Antwort:** Für jede Zelle und Betragsstufe wird die größere MAE-Änderung der beiden getesteten Vorzeichen verwendet. Anschließend werden Zellen gleich gewichtet. Das ist die ungünstigere der zwei untersuchten Richtungen, nicht der mathematische Worst Case aller möglichen Fehlerverläufe.

**Vertiefung:** Diese nichtlineare Auswahl unterscheidet sich vom Mittel beider Vorzeichen. Sie kann auch kleine positive Effekte betonen und muss als Auswahlregel vor der Aggregation bekannt sein. Ein monotoner Verlauf im betrachteten Raster ist ein Befund, kein allgemeiner Satz. Bei Clipping, Resets und nichtlinearen Modellen muss Monotonie nicht immer gelten. **Beleg:** Abbildung 6.5.

## F050. Warum ist additiver Stromoffset viel kritischer als Gain?

**Antwort:** Offset wird auch bei nominal null Strom integriert. Gain ist proportional zum Strom und kann sich zwischen Laden und Entladen teilweise kompensieren. Bei 50 mA entstehen in 24 Stunden 1,2 Ah falsche Ladung, was bei 1,8 Ah etwa 66,7 Prozentpunkten ungebremster SOC-Verschiebung entspräche.

**Grenze:** Das ist keine Vorhersage des gemessenen MAE, weil Anker, Clipping, Lastwechsel und Korrektur wirken. Tatsächlich sind die adversen MAE-Zunahmen DD 0,0401, HECM 0,1763, DM 0,2467 und HDM 0,2727. Die Größenordnung erklärt, warum Nullpunktkalibrierung wichtig ist. **Beleg:** Kapitel 6.4.3.

## F051. Warum ist HDM beim Offset schlechter als DM, obwohl nominal besser?

**Antwort:** HDM reduziert nominal Kapazitätsmismatch, besitzt aber weiterhin den Integrationskern. Bei gealterter Zelle kann der kleinere effektive Kapazitätsnenner denselben falschen Ladungsbetrag in eine größere SOC-Abweichung umsetzen. Damit kann bessere nominale Kapazitätsanpassung zugleich die Offsetempfindlichkeit erhöhen.

**Grenze:** Dieses Argument erklärt eine strukturelle Möglichkeit. Der exakte Zahlenunterschied enthält zusätzlich SOH-Verlauf, Anker, Clipping und Zellprofil. Die Arbeit isoliert nicht jeden Anteil. Die richtige Schlussfolgerung ist, dass nominal bessere Adaptation nicht automatisch höhere Robustheit gegen einen anderen Fehlermechanismus liefert.

## F052. Akkumuliert der Gainfehler über die gesamte Lebenszeit?

**Antwort:** Nicht zwangsläufig monoton. Ohne Korrektur folgt der Integrationsfehler dem vorzeichenbehafteten Ladungsdurchsatz. Volladeanker verändern den Zustand, Lade- und Entladephasen ändern das Vorzeichen des Integrals, und Modellkorrekturen wirken zusätzlich. Die Lebenszeitanalyse zeigt eher wiederkehrenden Aufbau und teilweise Re-Ankerung als unbegrenzte Drift.

**Beleg:** Im kontinuierlichen High-Load-Replay steigen MAE um 0,0102 für DM, 0,0072 für HDM, 0,0037 für HECM und 0,0048 für DD. Hier hat HECM die kleinste Zusatzlast, obwohl DD im sechszelligen Fenstervergleich günstiger ist. **Grenze:** Eine Zelle über die volle Lebenszeit und mehrere standardisierte Testfenster beantworten verschiedene Fragen. Dieser Unterschied ist kein logischer Widerspruch.

## F053. Warum verursachen hohe lokale Ausschläge manchmal kaum globalen MAE?

**Antwort:** Ein Mittel über viele Stunden verdünnt kurze Ereignisse. Ein Modell kann einen einzelnen gefährlichen Peak haben und trotzdem fast unveränderten mittleren Fehler. Deshalb verwendet die Arbeit ereignisausgerichtete Maxima, P95-Peaks und Excess Error zusätzlich zu globalen Kennzahlen.

**Herleitung:** Ein Fehler von 0,2 für einen einzigen Sample unter 86400 Samples trägt isoliert nur etwa 0,00000231 zum Tages-MAE bei. Ob der restliche Verlauf zusätzlich gestört wird, ist separat zu prüfen. **Beleg:** Abbildungen 6.12 und 6.13. DD zeigt eine lokale Spike-Empfindlichkeit, die in der globalen Heatmap fast unsichtbar ist.

## F054. Warum verstärken Ableitungsfeatures Rauschen und Spikes?

**Antwort:** Eine Rückwärtsdifferenz zieht zwei Messungen voneinander ab und teilt durch den Zeitabstand. Unabhängiges Messrauschen addiert sich in der Varianz. Bei gleichem Δt hat die Differenz zweier unabhängiger Fehler mit Varianz σ² die Varianz 2σ². Kleine Zeitabstände vergrößern zusätzlich die Ableitungsamplitude.

**Vertiefung:** Ein einzelner Spannungsspike erzeugt typischerweise eine große positive und danach negative Ableitungsstörung. DD sieht damit sowohl absolute Spannung als auch dU/dt. Eine Filterung könnte helfen, verändert aber Verzögerung und Trainingsverteilung. **HYPOTHESE:** Dieser Pfad ist ein plausibler Grund für DD-Transienten, nicht durch kanalweise Ablation bewiesen.

## F055. Warum kann DM trotz fehlender Spannungsregelung auf Spannungsrauschen reagieren?

**Antwort:** Spannung beeinflusst die Volladeerkennung. Wenn Rauschen den Zeitpunkt des bestätigten Ankers verändert, ändert sich der Qc-Reset und damit der spätere SOC-Verlauf. Spannung wird also nicht kontinuierlich zur Korrektur benutzt, ist aber nicht wirkungslos.

**Beleg:** Kapitel 6.4.4 berichtet kleine MAE-Zunahmen bei Spannungsrauschen auch für DM und HDM. **Grenze:** Ohne Ereignislog der Resetentscheidungen ist die Ankererklärung plausibel, aber nicht für jeden konkreten Knick direkt bewiesen. Eine sichere Diagrammantwort nennt den beobachteten Knick und den zu prüfenden Resetzeitpunkt, statt jede Bewegung nachträglich zu erklären.

## F056. Warum ist Jitter hier weniger kritisch als Sampleverlust?

**Antwort:** Der Jittertest erhält Messinformation und rekonstruiert zeitabhängige Größen mit den gestörten Zeitabständen. Beim Sampleverlust wird Information durch gehaltene Werte ersetzt. Insbesondere fehlende Ladung kann Integration nicht nachträglich rekonstruieren.

**Grenze:** Das ist keine Entwarnung für beliebige asynchrone Sensoren. Kanalversatz, falsche Timestamps, Frequenzdrift oder aliasierte hochfrequente Signale sind andere Fehler. Die konkrete Jitterimplementierung muss zu der behaupteten realen Störung passen. **Beleg:** Abbildung 6.10, drei Stufen ±0,1, ±0,5 und ±0,9 s.

## F057. Was wird beim Burst Dropout eingefroren und warum 48 Stunden?

**Antwort:** Der Versuch simuliert einen einstündig eingefrorenen Messstream. Der Start liegt im zulässigen Abschnitt mit größtem Betrag der integrierten Nettoladung. Es müssen mindestens zwölf Stunden davor und 24 Stunden danach verfügbar sein. Ein 48-Stunden-Baseline-Lauf ist nötig, um die Störung nicht gegen einen unterschiedlich langen nominalen Versuch zu vergleichen.

**Vertiefung:** Der letzte gültige Wert kann während des Ausfalls physikalisch falsch werden. Es geht nicht bloß um fehlende Bildpunkte. Abgeleitete Features müssen konsistent aus den eingefrorenen Messungen entstehen. **Beleg:** Abbildung 6.11. DD hat eine negative mittlere globale Änderung, deren Intervall null einschließt. Das belegt keine heilsame Wirkung von Ausfällen.

## F058. Warum verbessert perfekterer SOH nicht automatisch DD?

**Antwort:** HDM benutzt SOH in einem expliziten Kapazitätsnenner, HECM zusätzlich zur Tabellenwahl. DD benutzt SOH innerhalb einer gelernten mehrdimensionalen Abbildung. Ein Austausch eines Features kann dessen Zusammenhang mit anderen Features oder mit dem Training verändern.

**Beleg:** Der Referenz-SOH-Austausch senkt den nominalen MAE für HDM um 0,0370 und für HECM um 0,0166. DD verschlechtert sich im Mittel um 0,0049. **HYPOTHESE:** Ko-Adaption und Featureverteilungsänderung können das erklären. Für einen Nachweis müsste man Trainingseingänge, Kalibrierung und Ablationen untersuchen. Es folgt nicht, dass bessere SOH-Messungen prinzipiell unerwünscht wären.

## F059. Was beweist die HECM-Sensitivitätsanalyse?

**Antwort:** Sie prüft lokale Veränderungen der Widerstands- und Zeitkonstantentabellen um ±10 Prozent und OCV-Offsets um ±10 mV, jeweils einzeln und über alle betrachteten Störungssubfälle. Bewertet wird, wie sich die Störungspenalty relativ zur zugehörigen nominalen Variante verändert. Die größte absolute Interaktion beträgt 0,006116 MAE.

**Grenze:** Das ist keine Tabellenunabhängigkeit. Baseline-MAE und Recovery ändern sich. Bei Widerstand -10 Prozent steigt der mittlere Recovery-oder-Zensur-Wert von 1,20 auf 5,10 Stunden, weil eine Zelle nicht recovern kann. Es wurden keine gemeinsamen Parameterverteilungen, keine beliebigen Kennlinienformänderungen und keine globale Unsicherheitsanalyse untersucht. **Beleg:** Abbildung A.3 und Tabelle A.1.

## F060. Warum betrachtet man ΔΔMAE statt nur absoluten MAE?

**Antwort:** Eine Lookup-Änderung kann bereits nominal einen anderen Fehler erzeugen. Die Störungswechselwirkung soll davon getrennt werden. Zuerst wird für dieselbe Lookup der gestörte minus nominale MAE gebildet. Davon wird die entsprechende Differenz unter der ursprünglichen Lookup abgezogen.

**Herleitung:** ΔΔ = (MAEgestört,perturbiert - MAEsauber,perturbiert) - (MAEgestört,nominal - MAEsauber,nominal). Kleine ΔΔ bedeutet ähnliche zusätzliche Störungskosten, nicht gleichen absoluten Fehler. **Kritischer Punkt:** In mehreren Subfällen schließen punktweise Intervalle null aus. Daher ist die Aussage gar keine Auswirkung sachlich zu stark, auch wenn der größte Offseteffekt überwiegend durch den Sensorfehler erklärt wird.

# 6. Statistik und wissenschaftliche Fairness

## F061. Warum Cell-Macro statt alle Samples zusammenwerfen?

**Antwort:** Zuerst werden Fenster und Wiederholungen innerhalb einer Zelle zusammengefasst. Danach bekommt jede der sechs Zellen dasselbe Gewicht. Andernfalls könnten lange Trajektorien, viele verfügbare Fenster oder viele Seeds die Gesamtaussage dominieren.

**Vertiefung:** Macro beantwortet die Frage nach dem durchschnittlichen Zellverhalten in diesem Holdout-Set. Ein samplegewichteter Wert würde eher einen zufällig gezogenen Zeitpunkt unter der vorhandenen Datendauer repräsentieren. Beide sind definierbar, aber nicht austauschbar. Die Auswahl der Gewichtung muss vor dem Ergebnis begründet werden. **Beleg:** Kapitel 6.2.6.

## F062. Was machen 10000 Bootstrap-Wiederholungen?

**Antwort:** Aus den vorhandenen unabhängigen Einheiten werden wiederholt Stichproben mit Zurücklegen gezogen und die gewünschte Kennzahl neu berechnet. Hier werden Zellen und darin geschachtelte stochastische Wiederholungen berücksichtigt. Aus der resultierenden Verteilung wird ein Unsicherheitsintervall geschätzt.

**Grenze:** Zehntausend Wiederholungen stabilisieren die numerische Bootstrap-Schätzung, schaffen aber keine neuen Zellen und beheben keine fehlende Repräsentativität. Mit sechs Zellen bleiben extreme oder bislang unbesetzte Betriebsbedingungen schlecht erfasst. Bei gepaarten Modellen muss die Paarung im Resampling erhalten bleiben. **OFFEN:** Die genaue Behandlung mehrerer deterministischer SOH-Fenster sollte am Auswertungscode gezeigt werden können.

## F063. Warum ist der Signifikanztest bei sechs Zellen so grob?

**Antwort:** Ein exakter Sign-Flip-Test mit sechs unabhängigen gepaarten Differenzen hat nur 2 hoch 6, also 64 Vorzeichenkombinationen. Für einen üblichen zweiseitigen extremen Test ist die kleinste erreichbare Wahrscheinlichkeit 2/64 = 0,03125, sofern keine degenerierten Bindungen vorliegen.

**Herleitung:** Bei sechs paarweisen Modellvergleichen beginnt Holm mit einem Vergleich gegen 0,05/6 = 0,00833. Schon der kleinste mögliche unadjustierte Wert ist größer. Unter dieser Konstellation kann deshalb kein Test die erste Holm-Stufe bestehen. Das erklärt fehlende korrigierte Signifikanz ohne den praktischen Effekt zu widerlegen. **Grenze:** Das Ergebnis hängt von der tatsächlichen Testfamilie und Testdefinition ab. **Beleg:** Kapitel 6.2.6 und 6.4.1.

## F064. Warum kann ein Konfidenzintervall null ausschließen, während p nicht signifikant ist?

**Antwort:** Bootstrap-Intervalle, diskrete Sign-Flip-Tests und Holm-Korrektur benutzen unterschiedliche Verfahren und gegebenenfalls unterschiedliche Fehlerkontrolle. Ein punktweises unadjustiertes Intervall ist nicht dasselbe wie eine simultane Aussage über viele Tests. Daher ist die Konstellation nicht automatisch widersprüchlich.

**Vertiefung:** Aussagen sauber trennen: Effektgröße, punktweises Intervall, Testverfahren, Zahl der Vergleiche und korrigierter p-Wert. Nicht ein Verfahren auswählen, nur weil es die gewünschte Aussage liefert. Die Dissertation sollte durchgängig sagen, dass die kleine Zahl unabhängiger Zellen die inferenzielle Stärke begrenzt.

## F065. Was ist der Unterschied zwischen Fehlerbalken, Quartilen und Min-Max?

**Antwort:** Quartile beschreiben die Verteilung beobachteter Werte, Min-Max deren Extremwerte und ein Konfidenzintervall die Unsicherheit einer geschätzten Kennzahl. Die Fehlerbalken im Robustheitskapitel sind überwiegend hierarchische 95-Prozent-Konfidenzintervalle, nicht automatisch Min-Max oder Interquartilsabstände.

**Vertiefung:** Boxplots im Embedded-Kapitel beschreiben dagegen zeitliche Fehlerverteilungen. Ein schmales Intervall um einen Mittelwert kann mit einer breiten Verteilung einzelner Fehler koexistieren. **Übung:** Vor jedem Diagramm zuerst sagen, was Punkt, Balken, Box und Strich jeweils darstellen. Die genaue Whisker-Regel eines Boxplots muss aus Skript oder Legende kommen, nicht aus Gewohnheit angenommen werden.

## F066. Sind gleiche Seeds für alle Modelle unfair?

**Antwort:** Nein, gleiche Störungsrealisierungen sind für gepaarte Vergleiche sinnvoll. Alle Modelle sehen dann dieselben zufälligen Messfehler und dieselben Ausfallereignisse. Mehrere Seeds prüfen die Abhängigkeit vom konkreten Zufallsverlauf.

**Grenze:** Störungsseeds sind keine Trainingsseeds. Sie quantifizieren nicht die Variabilität neu trainierter Netze. Ebenso messen wiederholte MCU-Replays primär Ausführungsstreuung und nicht neue Zellgeneralisation. **Beleg:** Kapitel 6.3 nennt zehn Seeds für Gaußrauschen und fünf für zufällige Verluste, Jitter und Spikes, deterministische Fälle einmal.

## F067. Wie kann man 20 Szenarien, 22 Subfälle und 18 Heatmap-Zeilen erklären?

**Antwort:** Es sind unterschiedliche Zählebenen. Das Protokoll enthält 20 Definitionen einschließlich Baseline und Initialisierung. Entfernt man diese beiden, bleiben 18 Messstörungs- und Integritätszeilen. Drei Gainstufen und eine Offsetstufe werden jeweils mit beiden Vorzeichen ausgeführt, wodurch vier zusätzliche Subfälle und damit 22 entstehen.

**Vertiefung:** Seeds, Zellfenster, Modelle und Diagnosezweige multiplizieren die tatsächliche Zahl der Auswertungen weiter. Die Arbeit nennt 7040 Modell-Fenster-Auswertungen für die Hauptkampagne. Ein prüfbarer Run-Manifest ist belastbarer als aus dem Gedächtnis eine vereinfachte Multiplikation zu konstruieren. **Beleg:** Abbildung A.1 und Kapitel 6.3.

## F068. Wie interpretieren Sie Radarplot und Robustheitsscore?

**Antwort:** Die Skalen sind relative Zusammenfassungen innerhalb der betrachteten Kandidaten. Familienbalancierte Robustheit verhindert, dass eine Fehlerfamilie allein wegen vieler Severity-Stufen mehr Gewicht erhält. Andere Gewichtungen führen zu anderen Abständen und teilweise zu anderen Reihenfolgen.

**Grenze:** 0,967 ist keine Zuverlässigkeit von 96,7 Prozent und keine Fehlerfreiheitswahrscheinlichkeit. Min-Max-Normierung kann sich ändern, sobald ein weiterer Kandidat aufgenommen wird. Im getesteten Satz führt DD mehrere Aggregationen an, aber einzelne lokale Schwächen bleiben relevant. **Beleg:** Abbildung 6.16. Für Sicherheitsentscheidungen zuerst absolute Anforderungen und Szenariowerte prüfen.

## F069. Was wäre eine saubere Studie zur Altersabhängigkeit?

**Antwort:** Für jede Zelle die gepaarte Störungspenalty in vergleichbaren SOH-Phasen betrachten, dazu Last, Temperatur und Resetabstände dokumentieren. Der Zielkontrast lautet Änderung der Penalty mit SOH innerhalb der Zelle, nicht bloß schlechterer MAE in späteren Daten.

**Grenze:** Alter und Betriebsbedingungen sind nicht unabhängig randomisiert. Fehlende aged-Zellen und veränderte Lastprofile können Scheineffekte erzeugen. Eine Regression mit Zellstruktur oder ein Mixed-Effects-Modell kann helfen, aber bei sechs Zellen und wenigen Fenstern wäre ein komplexes Modell instabil. Erst deskriptiv transparent berichten, dann gezielte neue Versuche planen.

## F070. Wie vermeiden Sie nachträgliches Schönreden eines Diagramms?

**Antwort:** Vorzeichen, Fenster, Seeds, Aggregation, Einheiten und Metrik müssen nachvollziehbar sein. Ein unerwartet negativer ΔMAE wird nicht entfernt, sondern als mögliche Kompensation analysiert. Ein schönes Einzelzellbild wird nicht als Beleg für alle Zellen verkauft.

**Praktischer Ablauf:** Erst Roh- und Baselinekurven paaren, dann Ereigniszeitpunkt und Kausalität prüfen, dann lokale und globale Metriken unterscheiden, zuletzt interpretieren. Wenn sich die Story dadurch ändert, ändert man die Aussage und nicht die Daten. Explorative Analysen dürfen Teil einer kohärenten Arbeit sein, müssen aber nicht fälschlich als vorab registrierte Bestätigung bezeichnet werden.

# 7. Neuronale Grundlagen, Pruning und Quantisierung

## F071. Wie erklären Sie LSTM und GRU an der Tafel?

**Antwort:** Ein LSTM besitzt einen Zellzustand c und einen sichtbaren Zustand h. Forget-, Input- und Output-Gates steuern Behalten, Schreiben und Ausgeben. Die Kandidateninformation wird typischerweise mit tanh berechnet. Eine GRU hat nur einen rekurrenten Zustand und verwendet Reset- und Update-Gate, um alte und neue Information zu mischen.

**Prüfpunkt:** Update-Gate-Konventionen unterscheiden sich. Im Grundlagenbild der Dissertation gewichtet z den Kandidaten, während beispielsweise PyTorch in seiner Dokumentation z den alten Zustand gewichten lässt. Das kann durch z gegen 1-z mathematisch äquivalent sein, wenn Parameter und Gleichungen konsistent sind. Auch reset-before und reset-after unterscheiden sich. Beim C-Export zählt die konkret trainierte Framework-Semantik, nicht nur der Name GRU. **Beleg:** Abbildungen 3.3 und 3.4, PyTorch-Quelle im Quellenabschnitt.

## F072. Warum helfen Gates gegen verschwindende Gradienten, ohne Stabilität zu garantieren?

**Antwort:** Der additive Zellzustandspfad im LSTM erleichtert den Informationstransport über viele Schritte gegenüber einer ausschließlichen wiederholten nichtlinearen Transformation. Gates können steuern, wie viel Information erhalten bleibt. Dennoch können Aktivierungen sättigen, Gradienten verschwinden oder wachsen und lange Extrapolationen problematisch sein.

**Vertiefung:** Gradient Clipping beschränkt große Gradienten beim Training, beweist aber keine BIBO-Stabilität des trainierten Modells. Endlich lange fehlerfreie Replays liefern empirische Evidenz, keinen allgemeinen mathematischen Stabilitätsbeweis. Ein Sigmoid-Ausgang begrenzt SOC, nicht zwingend alle internen Zustände.

## F073. Was unterscheidet Trainingschunk, Rolling Window und dauerhaften Zustand?

**Antwort:** Ein Trainingschunk begrenzt den Backpropagation-Horizont. Ein Rolling Window berechnet jede Ausgabe aus einem begrenzten Fenster mit neu initialisiertem Zustand. Dauerhaftes Streaming führt h und gegebenenfalls c von Sample zu Sample weiter. Diese Begriffe beschreiben unterschiedliche Ebenen und dürfen nicht gleichgesetzt werden.

**Beleg:** Kapitel 6 primär Rolling-GRU mit 2024 Samples, Hardware zusätzlich continuous und periodic reset. Kapitel 7 stateful LSTM-Ausführung mit Training über begrenzte Chunks. **OFFEN:** Ob Trainingszustände zwischen Chunks übernommen und nur vom Gradienten getrennt wurden oder tatsächlich zurückgesetzt wurden, muss die jeweilige Trainingsschleife belegen. Das beeinflusst die Interpretation des Trainings-Deployment-Mismatches.

## F074. Warum ist strukturiertes Pruning für den Mikrocontroller attraktiv?

**Antwort:** Ganze versteckte Kanäle und abhängige Matrixdimensionen werden entfernt. Es entstehen kleinere dichte Arrays, die bestehende dichte C-Schleifen unmittelbar verarbeiten können. Gewichte, recurrent state und ein Teil des Rechenaufwands schrumpfen gemeinsam.

**Grenze:** Unstrukturiertes Nullsetzen verkleinert dichte Arrays zunächst nicht und kann ohne Sparse-Kernel kaum Laufzeit sparen. Sparse-Indexierung verursacht eigenen Speicher- und Steuerungsaufwand. Strukturiertes Pruning ist hier eine hardwaregerechte Wahl, aber nicht allgemein genauer als unstrukturiertes. **Beleg:** Abbildungen 3.5 und 3.6 sowie 7.4.

## F075. Was genau wird beim LSTM-Pruning entfernt?

**Antwort:** Für einen versteckten Kanal werden die zugehörigen Zeilen sämtlicher vier Gate-Matrizen, Bias-Einträge, die entsprechenden rekurrenten Spalten, Zustandskomponenten und MLP-Eingangsspalten entfernt. Nur so bleiben alle Abhängigkeiten und Dimensionen kompatibel.

**Vertiefung:** Der verwendete L2-Score summiert Normen der Input- und Recurrent-Gatezeilen. Die aus Konsistenzgründen mit entfernten Spalten gehen nicht als eigener Term in genau diesen Score ein. Aus kleiner Gewichtsnorm folgt nicht automatisch geringe dynamische Bedeutung. **Beleg:** Kapitel 7.4, Abbildung 7.4 und A.10. Nach dem einmaligen Schnitt folgt kurzes Fine-Tuning.

## F076. Warum 30 Prozent und warum nicht viel mehr?

**Antwort:** Die Studie verwendet einen moderaten Betriebspunkt, SOC 64 auf 45 und SOH 128 auf 90 versteckte Einheiten. Das ist kein experimentell bewiesenes globales Optimum. Ein breiter Sweep müsste für jede Rate Auswahl, Fine-Tuning und unabhängige Bewertung wiederholen.

**Grenze:** Bei stärkerer Reduktion kann die zeitliche Repräsentation oder die lokale Dynamik leiden. Die ideale Rate hängt von Redundanz, Datensatz, Verlustfunktion und Hardwareengpass ab. Ein einzelner günstiger Betriebspunkt reicht zur Demonstration einer Möglichkeit, nicht zur Behauptung, 30 Prozent seien universell optimal.

## F077. Warum spart 30 Prozent Breite etwa 45 Prozent MACs?

**Antwort:** Der rekurrente Anteil enthält Matrizen, deren Größe quadratisch mit der versteckten Breite wächst. Wird H auf 0,7H reduziert, bleibt asymptotisch 0,49 des quadratischen Anteils. Das entspricht 51 Prozent Reduktion für diesen Grenzfall. Lineare Eingangs- und Kopfanteile verringern den tatsächlichen Gesamteffekt.

**Beleg:** Die Arbeit zählt für SOC 22080 auf 12124 MACs und für SOH 85120 auf 46208. Das sind ungefähr 45,1 und 45,7 Prozent. Bias, Aktivierungen, Speicherbewegungen und Schleifen sind darin nicht enthalten. Deshalb folgt aus der MAC-Reduktion nicht exakt derselbe Laufzeitfaktor. **Übung:** Die Rechnung am Ende selbst ausführen.

## F078. Warum kann ein gepruntes Modell genauer werden?

**Antwort:** Pruning und Fine-Tuning verändern die Funktion. Eine solche Änderung kann redundante oder ungünstige Abhängigkeiten reduzieren und auf einem bestimmten Testverlauf näher an das Ziel führen. In Kapitel 7 ist der SOC-MAE für Pruned niedriger als für Base.

**Grenze:** Eine Regularisierungswirkung ist eine mögliche Erklärung, keine isoliert bewiesene Ursache. Unterschiedliches Fine-Tuning-Budget kann ebenfalls wirken. Zur Attribution braucht man eine weitertrainierte ungeprunte Kontrollgruppe, mehrere Initialisierungen und mehrere Zellen. SOH verschlechtert sich bei Kompression im primären Vergleich, was eine allgemeine Verbesserung widerlegt.

## F079. Wie funktioniert die verwendete INT8-Quantisierung?

**Antwort:** Für jede Matrixzeile wird ein Maßstab aus ihrem maximalen Absolutgewicht geteilt durch 127 berechnet. Die Gewichte werden durch diesen Maßstab geteilt, auf ganze Zahlen gerundet und im symmetrischen Bereich -127 bis 127 gespeichert. Im Kernel wird der effektive Gewichtswert mit dem zeilenspezifischen FP32-Maßstab rekonstruiert.

**Herleitung:** Bei endlichen Werten und der beschriebenen Max-Skalierung beträgt der Rundungsfehler eines Gewichts höchstens eine halbe Schrittweite. Das ist keine Schranke für den finalen SOC-Fehler, da viele Operationen und rekurrente Rückkopplungen folgen. **Beleg:** Kapitel 7.5. Ein all-zero-row-Schutz verhindert einen Maßstab von null.

## F080. Warum werden -128 und volle Integer-Arithmetik nicht genutzt?

**Antwort:** Der genau symmetrische Bereich -127 bis 127 besitzt gleich große positive und negative Endpunkte und stellt null exakt dar. Der zusätzliche negative INT8-Code bleibt ungenutzt. Vollständige Integer-Arithmetik wäre eine andere Implementierung mit quantisierten Aktivierungen, Akkumulatoren, Requantisierung und geeigneten Kernen.

**Beleg:** In dieser Arbeit sind nur die recurrent matrices Wih und Whh INT8 gespeichert. Bias, Zeilenskalen, Zustände, Aktivierungen, MLP und Rechenpfad bleiben FP32. Das Modell ist daher weight-only mixed precision, nicht vollständig INT8. Diese Präzisionsgrenze muss bei jedem Speicher- und Laufzeitargument genannt werden.

## F081. Warum spart Quantisierung nicht 75 Prozent des gesamten Flash?

**Antwort:** Nur ein Teil der Daten wird von vier auf ein Byte reduziert. Code, Bibliotheken, MLP-Gewichte, Bias und Skalen bleiben vorhanden. Die Zeilenskalen erzeugen zusätzlichen Speicher. Deshalb ist die 4:1-Ersparnis eine idealisierte Aussage für die quantisierten Matrixelemente, nicht für die komplette Firmware.

**Beleg:** SOC-Flash sinkt von 105,32 auf 52,48 KB, SOH von 335 auf 138 KB. Das ist deutlich, aber nicht ein Viertel der Gesamtfirmware. Für RAM gilt die Gewichtsspeicherargumentation erst recht nicht, weil Zustände und Arbeitsdaten weiterhin FP32 sind. **OFFEN:** KB/KiB-Konvention des ursprünglichen Messskripts für genaue Bytevergleiche dokumentieren.

## F082. Warum ist Ihr quantisiertes Netz langsamer?

**Antwort:** Der untersuchte Kernel konvertiert INT8-Gewichte während der Akkumulation nach FP32 und wendet die Skalen an. Es wird kein optimierter vollständig ganzzahliger Dot-Product-Pfad genutzt. Die Topologie und die Zahl der Modell-MACs bleiben unverändert, zusätzliche Operationen kommen hinzu.

**Beleg:** SOC-Kernelzeit steigt von 1,40 auf 6,99 ms, SOH von 22,73 auf 29,21 ms. Die Projekte sind laut Text ohne Compileroptimierung gebaut. **Grenze:** Das beweist nicht, dass Quantisierung grundsätzlich langsamer ist. Auch die genaue Verteilung auf Konversion, Cache, Schleifen und Speicher wurde nicht profiliert. CMSIS-NN zeigt, dass optimierte Integer-Kernel einen anderen Implementierungsraum darstellen.

## F083. Warum unterscheiden sich SOC- und SOH-Laufzeitfaktoren so stark?

**Antwort:** MAC-Zahl ist nur ein statischer Teil der Kosten. Aktivierungen, Indexierung, Schleifen, Speicherzugriffe und Funktionsgrenzen haben andere Skalierungen. Außerdem sind die Zeitmessgrenzen nicht vollständig gleich, etwa bezüglich SOH-Skalierung im Quantized-Pfad.

**Grenze:** Es gibt in der Arbeit keine operation-level-Zerlegung, mit der der Faktor 4,99 bei SOC gegenüber 1,29 bei SOH eindeutig erklärt werden könnte. Die ehrliche Antwort benennt gemessenen Unterschied und plausible Beiträge, ohne einen speziellen Cacheeffekt als bewiesene Ursache auszugeben. Ein nächster Versuch würde identische Messgrenzen, Optimierungsflags und Kernelprofiling verwenden.

# 8. Embedded-Messung, Filterung und Systemgrenzen

## F084. Warum darf man die GRU- und LSTM-Hardwarezahlen nicht direkt vergleichen?

**Antwort:** Es sind unterschiedliche Modelle, Eingangsvektoren, Messgrenzen und Firmwarepfade. Kapitel 6 misst isolierte SOC-Kerne mit extern bereitgestelltem kausalem SOH. Kapitel 7 misst eigenständige stateful SOC- und SOH-LSTM-Modelle mit vorbereiteten Featurevektoren.

**Beleg:** Continuous-DD in Kapitel 6 benötigt etwa 0,425 ms und 4,1 KiB Peak-RAM. Das SOC-Base-LSTM in Kapitel 7 benötigt 1,40 ms Kernelzeit und 4,93 KB RAM. Daraus lässt sich kein sauberer Architektursieger GRU gegen LSTM ableiten. Dafür müssten Features, Genauigkeitsziel, Toolchain und Rechenmodus kontrolliert werden.

## F085. Was zeigen 724 ms gegenüber 0,425 ms?

**Antwort:** Beim Rolling-Verfahren werden für jede Ausgabe 2024 Schritte mit neuem Zustand berechnet. Continuous verarbeitet nur den neuen Schritt. Der Quotient liegt bei rund 1704, nicht exakt 2024, weil unterschiedliche feste Kosten und Ausführungsdetails beitragen.

**Grenze:** Es ist kein zusätzlicher Pruninggewinn und keine algebraisch identische Beschleunigung desselben zeitlichen Algorithmus. Das Gedächtnis der kontinuierlichen Ausführung kann länger zurückreichen. Die ähnlichen nominalen Fehler der sechs Replays validieren den untersuchten Betrieb, aber nicht automatisch die identische Robustheit unter allen Störungen. **Beleg:** Abbildung 6.20.

## F086. Warum sind periodische Resets trotz ähnlicher Laufzeit schlecht?

**Antwort:** Ein Reset verwirft die akkumulierte versteckte Repräsentation. Das Netz muss Kontext neu aufbauen und kann währenddessen große Fehler erzeugen. Das ist nicht dasselbe wie das rollende Neuberechnen eines vollständigen historischen Fensters.

**Beleg:** Der Mittelwert der zellweisen maximalen Dataset-SOC-Fehler liegt bei periodischem Reset um 29 Prozentpunkte, einzelne Zellen erreichen etwa 39. Continuous und Rolling haben hier viel günstigere Verläufe. **Nachfrage:** Was bei Stromausfall tun? Zustandscheckpointing, definierte Kaltstartprozedur, verifizierte Anker und ein Fallback sind mögliche Erweiterungen, aber ihre sichere Funktion ist nicht durch diese Studie bereits nachgewiesen.

## F087. Wie wurden Flash und RAM bestimmt?

**Antwort:** Flash stammt aus den belegten Bereichen des gelinkten Programms. Das ist eine präzise Build-Eigenschaft, keine Schätzung aus der Parameterzahl. Runtime-RAM kombiniert statische Daten mit tatsächlich beobachtetem dynamischem Bedarf und Stack-Hochwassermarke entsprechend dem jeweiligen Kapitel.

**Grenze:** Ein beobachteter Peak ist nicht automatisch eine formal bewiesene Obergrenze für jeden Interrupt-, Fehler- und Schedulingpfad. Die Rolling-DD-Messung muss nach gefülltem Fenster eine gültige Inferenz umfassen. Unbenutzte Reserven dürfen nicht als tatsächlich benötigter Speicher missverstanden werden. **Beleg:** Kapitel 6.2.4, 7.6.2, Abbildungen 6.19 und 7.12.

## F088. Warum ändern Zellen den Flash kaum, aber die Laufzeit möglicherweise schon?

**Antwort:** Bei identischer Firmware bleiben Code, Gewichte und statische Puffer unabhängig von den Eingangssequenzen gleich. Laufzeit kann dagegen durch datenabhängige Zweige, Aktivierungsfunktionen, Cachezustand oder Interrupts variieren. Wiederholungen prüfen diese Variation.

**Grenze:** Laufzeitverteilungen sind keine verschiedenen Modellgrößen. Umgekehrt kann eine andere Eingangsfolge einen höheren Stack- oder Heap-Peak auslösen, wenn datenabhängige Pfade existieren. Deswegen bleibt der Messumfang wichtig. In deterministischen dichten Kernen ist geringe Streuung durchaus plausibel und kein Beweis für fehlende Zellvielfalt.

## F089. Was misst der DWT-Cycle-Counter, was die Hostlatenz?

**Antwort:** Der Cycle-Counter zählt CPU-Zyklen zwischen instrumentierten Grenzen auf dem Mikrocontroller. Teilt man durch den tatsächlichen Prozessortakt, erhält man die verstrichene Zeit dieses Bereichs. Hostlatenz enthält zusätzlich UART-Transfer, Protokoll, Scheduling und Rücktransport.

**Vertiefung:** Interrupts innerhalb der Messgrenzen können mitzählen. Messoverhead, Taktkonfiguration, Cache/Warm-up und Zählerüberlauf müssen dokumentiert werden. Bei 480 MHz entsprechen 204000 Zyklen 0,425 ms. **Beleg:** Kapitel 6.2.4 und 7.6.2. Eine hohe Hostlatenz widerlegt nicht automatisch einen schnellen Kern.

## F090. Haben Sie Echtzeitfähigkeit bewiesen?

**Antwort:** Die untersuchten isolierten Ausführungen bleiben innerhalb der 1-s-Abtastperiode. Das zeigt Machbarkeit unter dem Messaufbau. Ein formaler Worst-Case-Execution-Time-Nachweis oder die volle Terminierbarkeit aller parallelen BMS-Aufgaben ist damit nicht erbracht.

**Vertiefung:** Für ein Gesamtsystem müssen Sensorerfassung, Featurebildung, Kommunikation, SOH-Service, Schutzaufgaben, Interrupts und Fehlerbehandlung hinzugerechnet werden. Besonders Rolling-DD belegt mit 724 ms bereits etwa 72,4 Prozent der Periode. Continuous-DD hat mehr Reserve. **Nicht sagen:** Ein schneller Mittelwert allein garantiere jede Deadline im Feld.

## F091. Warum keine unabhängige Energieaussage?

**Antwort:** Eest wird aus angenommener konstanter Boardleistung 0,5 W mal gemessener Kernelzeit berechnet. Der Wert besitzt damit exakt dieselbe relative Rangfolge wie die Zeit. Es wurde für die Varianten keine unabhängige Leistungsaufnahme gemessen.

**Herleitung:** 0,5 W mal 0,8 ms sind 0,4 mJ. Bei 6,99 ms sind es 3,495 mJ. **Grenze:** Wirkliche Energie hängt von Aktivität, Speicher, Peripherie, Spannungsreglern und Schlafphasen ab. Für eine reale Messung wären synchronisierte Strom-/Spannungserfassung, definierte Systemgrenzen und Wiederholungen erforderlich. Der Proxy ist nützlich, darf aber nicht als zusätzlicher unabhängiger Erfolg verkauft werden.

## F092. Welche SOH-Filterung wird verwendet und warum ist sie kritisch?

**Antwort:** Zuerst wird die Rohvorhersage auf den anfänglichen SOH ausgerichtet. Eine erste Stufe begrenzt Änderungen und mittelt mit α = 0,02. Die zweite Stufe verwendet α = 0,000001 und begrenzt zusätzlich Abwärtsänderungen. Bei 1 Hz hat der lineare EMA-Anteil dieser zweiten Stufe eine Zeitkonstante von etwa 11,57 Tagen und eine 90-Prozent-Antwortzeit von 26,65 Tagen.

**Grenze:** Eine ruhige Kurve kann stark verzögert sein. Kleine Fehler gegenüber einer langsam interpolierten Referenz beweisen nicht, dass das Rohnetz den SOH schnell erkennt. Rate-Limiter sind nichtlinear und besitzen nicht einfach dieselbe Grenzfrequenz wie ein EMA. **Beleg:** Kapitel 7.2, Abbildung A.6.

## F093. Ist die initiale SOH-Ausrichtung kausal?

**Antwort:** Die Formel verwendet den initialen Referenzwert y0. Das ist zeitlich kausal, wenn dieser Startwert vor Einsatz aus einem tatsächlichen Kapazitätstest oder verifizierten Inbetriebnahmewert bekannt ist. Es ist aber eine zusätzliche Informationsannahme, kein kostenlos verfügbarer Online-Messkanal.

**Grenze:** Bei unbekanntem gebrauchten Akku oder Neustart ohne Historie wäre y0 nicht selbstverständlich bekannt. Die Sensitivität gegenüber falschem Start-SOH müsste separat untersucht werden. Eine genaue Antwort sagt deshalb initial kalibriert statt vollständig blind geschätzt. Außerdem sollte der Nenner der Ausrichtung auf problematisch kleine Anfangsausgaben geprüft werden.

## F094. Warum unterscheiden sich SOH-Zahlen zwischen Dashboard und Filteranalyse?

**Antwort:** Die Hauptauswertung nennt final 0,85, 1,46 und 1,41 Prozentpunkte MAE für Base, Pruned und Quantized. Die kontrollierte Filteranalyse nennt final 1,476, 1,695 und 1,028. Der Text beschreibt die zweite Auswertung als kontrollierte Sequenz, spezifiziert im unmittelbaren Absatz aber nicht vollständig die Unterschiede der Datenbasis.

**OFFEN:** Vor einer sicheren Erklärung müssen Zelle, Zeitraum, Maskierung, Vorinitialisierung, Filterzustand und Versionsstand beider Exporte gegenübergestellt werden. Unterschiedliche Sequenzen können andere Rangfolgen erklären, sind hier aber nicht allein aus der Zahl bewiesen. Eine belastbare Antwort lautet, dass die Ergebnisse unterschiedliche Auswertungspfade betreffen und ihre genaue Provenienz vor der Verteidigung tabellarisch nachgewiesen werden muss. Nicht behaupten, beide seien dieselbe vollständige Trajektorie.

## F095. Was beweist die Langhorizontanalyse zur Quantisierung?

**Antwort:** Die Trajektorie wird in zehn aufeinanderfolgende Teile geteilt, ohne recurrent state zurückzusetzen. Beim SOC steigen sowohl Base- als auch Quantized-Zielfehler. Gleichzeitig sinkt ihre mittlere gegenseitige Abweichung von 0,332 auf 0,271 Prozentpunkte. Das spricht gegen wachsende quantisierungsspezifische Abweichung in diesem Replay.

**Grenze:** Das beweist keine generelle rekurrente Stabilität über beliebige Laufzeiten. Gemeinsame Zielabweichung kann durch schwierigere späte Betriebsbedingungen entstehen. Die Pruned-zu-Base-Differenz wächst, obwohl Pruned insgesamt kleineres MAE hat. Nähe zum Base ist daher nicht identisch mit Genauigkeit. **Beleg:** Abbildung A.5.

## F096. Warum ist Report-Loss nicht dasselbe wie Input-Dropout?

**Antwort:** Im Embedded-Report-Loss-Test läuft die Inferenz im Hintergrund weiter. Nur der ausgegebene Bericht fehlt, und die letzte gültige Ausgabe wird gehalten. Im Messstream-Dropout aus Kapitel 6 fehlen dem Estimator dagegen neue Eingangsinformationen. Die internen Zustände können sich dadurch falsch entwickeln.

**Beleg:** Kapitel 7.6.1 und 7.10. Bei langsamem SOH verändert gehaltene Ausgabe wenig, bei dynamischem SOC kann ein langer Berichtsausfall relevante Abweichungen erzeugen. **Nicht sagen:** Kaum Fehler bei 10 Prozent Reportverlust belege Robustheit des rekurrenten Modells gegen 10 Prozent fehlende Sensorwerte.

## F097. Was zeigt der Bitflip-Versuch und was nicht?

**Antwort:** In einer einzelnen FP32-Eingangszahl wird das höchstwertige Fraction-Bit 22 verändert. Vorzeichen und Exponent bleiben gleich. Die gestörte und saubere Modellkopie starten aus demselben Zustand und werden anschließend mit denselben 60 sauberen Samples weitergeführt. Damit wird die Nachwirkung eines begrenzten Eingangsbufferfehlers geprüft.

**Grenze:** Es werden keine Gewichts-, Zustands-, Exponenten- oder Hardware-Strahlungsfehler untersucht. Die Recovery-Schwelle ist ereignisrelativ, maximal aus zehn Prozent des Peaks und einem sehr kleinen Mindestwert gebildet. Daher ist sie nicht identisch mit der Zwei-Prozentpunkte-Recovery aus Kapitel 6. **Beleg:** Abbildung A.8 und Tabelle A.7.

## F098. Was kann der Utility-Score und was bedeuten 1771 Gewichtungen?

**Antwort:** U ist eine gewichtete Summe von Verhältnissen zum Base für MAE, Flash, RAM und Energieproxy. Niedriger ist günstiger, Base ist eins. Die 1771 Kombinationen entstehen aus vier nichtnegativen Gewichten in Schritten von 0,05 mit Summe eins.

**Herleitung:** 20 Gittereinheiten werden auf vier Gewichte verteilt. Die Anzahl ist der Binomialkoeffizient 23 über 3 = 1771. **Grenze:** 94,64 Prozent gewonnene Gewichtungen sind keine Erfolgswahrscheinlichkeit in realen Anwendungen. Das Ergebnis hängt vom Gitter, den Kriterien und der Base-Normierung ab. Der Energieproxy bringt nur Timinginformation ein. **Beleg:** Abbildung A.7.

## F099. Warum dual-core und warum ist das noch keine vollständige Isolation?

**Antwort:** Das Plattformkonzept legt klassische Überwachung und Schutzaufgaben auf den Cortex-M4 und die schwereren Schätzer auf den Cortex-M7. Dadurch lassen sich Algorithmen austauschen, ohne jede BMS-Funktion neu zu strukturieren. Gemeinsame Busse, Speicher und Kommunikation können trotzdem Interferenzen erzeugen.

**Grenze:** Eine Architekturzeichnung beweist weder Freedom from Interference noch eine geprüfte Safety-Partition. Benötigt werden definierte Schnittstellen, Zeitbudgets, Watchdogs, Datenkonsistenz, Fehlerpfade und Systemtests. **Beleg:** Kapitel 4.3.3. Die Custom-Plattform verwendet STM32H755, die isolierten Benchmarks STM32H753. Diese Hardwareebenen getrennt benennen.

## F100. Kann das System direkt ein Feld-BMS ersetzen?

**Antwort:** Die Arbeit zeigt eine Laborplattform und reproduzierbare Embedded-Schätzung. Sie ersetzt keine Produktqualifikation. Feldbetrieb bringt weitere Zellstreuung, Temperaturgradienten, EMV, Sensoralterung, Versorgungsausfälle und Packkopplung mit sich.

**Vertiefung:** Zell-SOC ist nicht automatisch Pack-SOC. In einem Serienpack teilen Zellen den Strom, besitzen aber unterschiedliche Kapazität und Grenzen. Die schwächste Zelle kann nutzbare Energie begrenzen. Ein BMS braucht unabhängige Schutzfunktionen auch dann, wenn die ML-Schätzung plausibel wirkt. **Beleg:** Kapitel 4.3 und 8.5. Konkrete Normkonformität wird hier weder behauptet noch geprüft.

# 9. Kritische Punkte vor der Verteidigung

Diese Liste enthält keine automatischen Änderungen an der Dissertation. Sie unterscheidet beobachtete Darstellungsprobleme von offenen Implementierungsfragen. Ziel ist eine ehrliche Antwort, nicht das Einüben einer unbelegten Rechtfertigung.

## P01. Abbildung 4.4: Split-Zuordnung ist im vorhandenen PDF nicht erkennbar

**Beobachtung:** Die Legende auf PDF-Seite 85 enthält Lade-C-Rate, Entlade-C-Rate und DOD. Zellnamen und explizite Training-/Validation-/Testzuordnung fehlen in der sichtbaren Grafik. Die Bildunterschrift behauptet dagegen eine Farbunterscheidung dieser drei Splits.

**Antwort auf eine Rückfrage:** Die verlässliche Zuordnung steht in Tabelle 4.3. Aus dieser Grafik allein kann ich den Split nicht eindeutig zeigen. **To-do:** Vor endgültiger Verteidigungsversion Bild und Caption abgleichen. Eine Backup-Folie mit Zellen und Splits erstellen. Hier wurde das Original nicht verändert.

## P02. Kapazitätsnenner zwischen Label und HDM

**Beobachtung:** Relatives SOH wird auf Cref,0 normiert, HDM verwendet in der gedruckten Gleichung Cnom mal SOH. **To-do:** Den tatsächlich verwendeten SOH-Scaler, die Trainingstargets und die Kapazitätskonstante im Runner gemeinsam zeigen. Falls es eine bewusste unterschiedliche Normierung gibt, Umrechnung dokumentieren. Nicht ohne Implementierungsprüfung einen Fehler behaupten, aber auch nicht Konsistenz voraussetzen.

## P03. EFC-Konvention und Faktor zwei

**Beobachtung:** Kapitel 6 schreibt EFC = gesamte absolute Ladung geteilt durch Cnom. Bei vollständig mitgezähltem Laden und Entladen ergibt ein kompletter Zyklus damit zwei. Häufig wird ein vollständiger Lade-Entlade-Durchlauf als ein EFC über Division durch 2C definiert.

**Antwort:** Die verwendete Durchsatzkoordinate kann als Feature konsistent sein, benötigt aber eine explizite Konvention. **To-do:** Prüfen, ob nur eine Richtung oder tatsächlich beide Richtungen gezählt werden. Datensatz-EFC, Feature-EFC und Achsen der NMC-Grafiken nicht ungeprüft gleichsetzen.

## P04. Dimensionen und Vorzeichen der Integrationsgleichungen

**Beobachtung:** Wenn Zeit in Sekunden und Kapazität in Ah angegeben werden, ist ein Faktor 3600 erforderlich. In der HECM-Übergangsgleichung steht ηΔt/Cb ohne expliziten Umrechnungsfaktor. Die allgemeine SOC-Integralgleichung hat eine andere Vorzeichenkonvention als das ladungspositive Qc.

**Antwort:** Beides kann korrekt sein, wenn Cb in Coulomb beziehungsweise die Zeit in Stunden und die Stromkonvention passend definiert sind. **To-do:** Einheiten und Codekoordinaten offenlegen. Die Vorbereitung liefert keine Bestätigung der Firmwareeinheiten.

## P05. MLP-Komplexität und Hyperparameterwahl

**Beobachtung:** Strukturell einfach ist nicht gleich klein. Das finale MLP ist breit und tief. Die Lag-Matrix zeigt exemplarische Testzellen, weshalb die Trennung von Auswahl und Demonstration nachvollziehbar bleiben muss. **To-do:** model.summary beziehungsweise Parameterzahl, Suchprotokoll und unberührte Testentscheidung bereithalten. Den vierten NMC-Testfall ausdrücklich nennen können.

## P06. Filterprovenienz und Anfangskalibrierung

**Beobachtung:** Haupt-SOH-Auswertung und Filteranalyse haben verschiedene Werte und Rangfolgen. Initialalignment verwendet den Anfangsreferenzwert. **To-do:** Für beide Exporte Zell-ID, Samplebereich, Filterparameter, initiale Zustände, Masken und MAE-Berechnung nebeneinander dokumentieren. Nicht aus dem glatten Bild auf ungefilterte Onlinegenauigkeit schließen.

## P07. Robustheit einer Semantik ist nicht automatisch Robustheit der anderen

**Beobachtung:** Der große Robustheitsvergleich verwendet Rolling-GRU. Die schnelle Hardwareempfehlung nutzt Continuous-State. Ähnliche nominale Ergebnisse über sechs Zellen schließen Unterschiede unter Störungen nicht aus. **To-do:** Als Grenze nennen. Eine vollständige Übertragung würde die relevanten Störungs- und Recoveryfälle im Continuous-Modus wiederholen.

## P08. Hardwaremessgrenzen und Compilerflags

**Beobachtung:** Kapitel 7 misst handgeschriebene, nicht optimiert kompilierte Referenzkerne. Bei SOH liegt die Skalierung nicht für alle Varianten innerhalb derselben Zeitgrenze. **To-do:** Compiler, Flags, Takt, Warm-up, Interruptbedingungen, gemessene Funktion und Hostprotokoll auf einer Backup-Folie festhalten. Die Ergebnisse nicht als maximale Leistungsfähigkeit des STM32 oder von INT8 ausgeben.

## P09. Tatsächliche Systemintegration versus Konzept

**Beobachtung:** Die Plattformbeschreibung umfasst Schutz, Sensing und Dual-Core-Trennung. Die berichteten Kernbenchmarks schließen zahlreiche Systemfunktionen aus. **To-do:** Genau trennen, was auf der eigenen Platine tatsächlich getestet wurde, was auf dem Nucleo gemessen wurde und was eine Architekturabsicht ist. Dafür Messprotokolle statt nur Blockdiagramme bereithalten.

## P10. Keine Tabellenunabhängigkeit, keine Sicherheit aus einem Score

**Beobachtung:** Die HECM-Analyse zeigt lokale Robustheitsstabilität in vielen Fällen, aber eine relevante Recovery-Änderung. Radar- und Utility-Scores hängen von Normierung und Gewichtung ab. **To-do:** Die Formulierungen unabhängig, sicherstes Modell und universell robust vermeiden, wenn sie nicht durch ein definiertes absolutes Sicherheitskriterium gedeckt sind.

# 10. Rechenübungen mit Lösungen

## R01. Stromoffset und Zeit

**Aufgabe:** Eine 1,8-Ah-Zelle wird mit einem konstanten Messoffset von +50 mA integriert. Wie groß ist der rein rechnerische SOC-Fehler nach einer, sechs und 24 Stunden ohne Anker oder Clipping?

**Lösung:** 0,05 A mal t / 1,8 Ah ergibt 0,02778, 0,16667 und 0,66667. Das sind 2,78, 16,67 und 66,67 Prozentpunkte. Das Vorzeichen hängt von der Stromkonvention ab. Diese Werte sind Endabweichungen, nicht automatisch zeitliche MAE. Bei linear von null wachsendem Fehler wäre der mittlere absolute Fehler über den Zeitraum halb so groß.

## R02. Gainfehler bei Lade-Entlade-Wechsel

**Aufgabe:** Eine Zelle erfährt zuerst +1 Ah und anschließend -1 Ah Nettoladung. Der Strom hat +3 Prozent Gainfehler. Was passiert mit dem idealisierten Integrationsfehler?

**Lösung:** Nach der ersten Phase sind +0,03 Ah Zusatzfehler vorhanden. Nach beiden Phasen ist die Summe null. Zwischenzeitlich bestand trotzdem ein Fehler. Bei 1,8 Ah entspricht der Peak 1,67 Prozentpunkten. Reale Asymmetrien, Resetbedingungen, Effizienz und Kapazitätsänderungen verhindern eine automatische vollständige Kompensation.

## R03. MAE und RMSE

**Aufgabe:** Normierte Fehler sind 0,01, -0,01 und 0,10. Berechne MAE, RMSE und Bias.

**Lösung:** MAE = 0,12/3 = 0,04. MSE = 0,0102/3 = 0,0034. RMSE = 0,05831. Bias = 0,10/3 = 0,03333. In Prozentpunkten sind dies 4,00, 5,83 und 3,33. RMSE betont den großen Ausreißer stärker. Der positive Bias zeigt die gerichtete Abweichung, obwohl ein negativer Einzelfehler vorkommt.

## R04. Makro- und Mikrogewichtung

**Aufgabe:** Zelle A hat MAE 0,02 über 100 Samples, Zelle B MAE 0,08 über 900 Samples. Was ergibt sich?

**Lösung:** Cell-Macro = (0,02 + 0,08)/2 = 0,05. Samplegewichtetes Mittel = (100 mal 0,02 + 900 mal 0,08)/1000 = 0,074. Keines ist allein durch Mathematik richtiger. Die Forschungsfrage bestimmt, ob jede Zelle oder jeder Zeitpunkt gleich gewichtet werden soll.

## R05. EMA-Zeitkonstante

**Aufgabe:** Warum dauert α = 10^-6 bei 1 Hz so lange?

**Lösung:** Nach n Schritten ist der verbleibende Anteil eines Sprungs (1-α)^n. Die Zeitkonstante ist -1/ln(1-α), näherungsweise eine Million Sekunden beziehungsweise 11,57 Tage. Für 90 Prozent Antwort setzt man den Rest auf 0,1. Daraus folgen ln(0,1)/ln(1-α), etwa 2,303 Millionen Sekunden oder 26,65 Tage. Nichtlineare Rate-Limiter können das reale Verhalten zusätzlich verändern.

## R06. Quantisierung einer Zeile

**Aufgabe:** Die Gewichtzeile lautet -0,8, 0,1, 0,4. Bestimme Skala, Codes und rekonstruierten mittleren Wert.

**Lösung:** s = 0,8/127 = 0,0062992. Gerundet ergeben sich -127, 16 und 64, wobei exakte Halbwerte von der verwendeten Rundungsregel abhängen können. Für 0,1 erhält man 16s = 0,100787. Der Fehler 0,000787 liegt unter s/2 = 0,003150. Die Rekonstruktion von 0,4 kann an der Halbwertgrenze je nach Regel geringfügig anders ausfallen. Framework- und C-Rundung müssen zusammenpassen.

## R07. SOC-MACs vor und nach Pruning

**Aufgabe:** Nutze D = 6, M = 64 und N = 4H(D+H) + HM + M für H = 64 und H = 45.

**Lösung:** Base: 4 mal 64 mal 70 + 64 mal 64 + 64 = 22080. Pruned: 4 mal 45 mal 51 + 45 mal 64 + 64 = 12124. Die Reduktion ist 9956/22080 = 45,09 Prozent. Es werden 19 von 64 recurrent channels entfernt, also 29,69 Prozent. Der MLP-Kopf behält M = 64.

## R08. Speicher des rollenden Eingabefensters

**Aufgabe:** Wie viel Speicher benötigen 2024 Samples mit acht FP32-Features allein als Rohpuffer?

**Lösung:** 2024 mal 8 mal 4 = 64768 Byte = 63,25 KiB. Das erklärt einen großen Teil des berichteten Rolling-Peaks von 67,3 KiB. Restlicher Speicher entfällt auf Zustände, temporäre Daten und Stack. Continuous braucht diesen vollständigen Rohfensterpuffer nicht für dieselbe rekurrente Schrittausführung.

## R09. Zensur verändert den Mittelwert

**Aufgabe:** Fünf Zellen recovern je nach 1,2 h. Eine bleibt bis 24 h außerhalb des Bandes. Wie sieht ein vereinfachtes Recovery-or-censor-Mittel aus?

**Lösung:** (5 mal 1,2 + 24)/6 = 5 h. Ein einzelner zensierter Fall kann damit einen ungefähr vierfachen Mittelwert erzeugen, obwohl die anderen fünf fast unverändert sind. Das erklärt qualitativ die Empfindlichkeit des HECM-Resistance-Falls. Die Rechnung ist ein Lehrbeispiel, nicht die genaue Rekonstruktion der Fensteraggregation des Experiments.

## R10. Energieproxy und Auslastung

**Aufgabe:** Rechne 0,5 W bei 1,40 ms und 0,80 ms. Wie viel Prozent einer 1-s-Periode benötigt 724 ms?

**Lösung:** 0,70 mJ und 0,40 mJ. Der relative Proxy-Rückgang beträgt 42,86 Prozent und ist identisch zum Zeitrückgang. 724 ms entsprechen 72,4 Prozent der Periode. Bei zusätzlicher BMS-Last kann die verbleibende Reserve knapp werden, während ein Median keine Worst-Case-Garantie bietet.

## R11. Gewichtsraum des Utility-Scores

**Aufgabe:** Warum sind es 1771 Gewichtskombinationen und nicht 21 hoch 4?

**Lösung:** Jedes Gewicht kann zwar 21 Werte von 0 bis 1 annehmen, aber die vier Gewichte müssen sich zu eins addieren. Gesucht sind nichtnegative ganzzahlige Lösungen von a+b+c+d = 20. Stars-and-bars liefert (23 mal 22 mal 21)/(3 mal 2 mal 1) = 1771. Gleiches Rastergewicht ist keine empirische Häufigkeit realer Nutzerpräferenzen.

## R12. Was muss man aus 0,006116 gegenüber 0,1763 schließen?

**Aufgabe:** Setze maximale Lookup-Störungsinteraktion und nominale HECM-Offsetpenalty ins Verhältnis.

**Lösung:** Der Quotient beträgt ungefähr 0,0347, also 3,47 Prozent. Das ordnet die lokale Interaktion gegenüber dem großen Offseteffekt ein. Es ist keine globale Sensitivitätskonstante. Die Maxima und signed Subfälle müssen in ihrer Definition beachtet werden, und Recovery kann trotzdem stark reagieren. Ein kleines Verhältnis rechtfertigt deshalb nicht die Aussage, die Lookup habe keine Auswirkung.

# 11. Lern- und Rechercheplan

## Priorität A: Ohne Nachschlagen erklären können

- Die drei Forschungsfragen, je ein Ergebnis und je eine Grenze. Dabei MLP/NMC, GRU/LFP und LSTM/LFP sauber trennen.
- SOC/SOH-Definitionen, Vorzeichen, Ah-zu-Coulomb-Umrechnung, Offset- und Gainfehler einschließlich Rechenbeispiel.
- Unterschied von nominalem MAE, gepaarter Störungspenalty und gepaarter Recovery. Zwei Prozentpunkte sind nicht zwei Prozent relativer Fehler.
- Cell-Macro, sechs unabhängige Zellen, Bootstrap, Zensierung und warum 10000 Wiederholungen keine 10000 unabhängigen Experimente sind.
- Weight-only INT8, nicht Full-Integer. Gemessene Laufzeit, gelinkter Flash, beobachteter RAM-Peak und Energieproxy klar auseinanderhalten.
- Die vollständige SOH-Filterkette mit initialer Information und 26,65 Tagen EMA-Antwortzeit.

## Priorität B: Mit einer Skizze herleiten können

- 2RC-Modell, EKF-Prädiktion, Spannungsresiduum und lokale Beobachtbarkeit auf dem LFP-Plateau.
- LSTM- und GRU-Gates einschließlich Framework-Konventionen. Zeige, wo der Zustand gespeichert wird.
- Gate-konsistentes Pruning, Matrixdimensionen und quadratische MAC-Skalierung.
- Zeilenweise Quantisierung, Halbschritt-Fehlergrenze und warum sie keine Ausgangsfehlerschranke ist.
- Gepaarten Kontrast, Lookup-Interaktion und unterschiedliche Unsicherheitsdarstellungen.

## Priorität C: Vor der Verteidigung anhand von Unterlagen klären

- Exakte NMC-Testzellen, Split- und Skalierungslogik, Historienbins und Rolle der MAE-Matrix bei der Auswahl.
- Kapitel-7-Testzelle, genauer Samplebereich und vollständiger Trainingssplit. Nicht automatisch die Kapitel-6-Zellen einsetzen.
- Unterschiedliche SOH-Filterauswertungen, Anfangskalibrierung und verwendete Referenzdenominator.
- EFC-Konvention, HECM-Einheiten, Full-Charge-Schwellen und Reset-Logs.
- Hardwaremessgrenzen, Compilerflags, Takt, Speicherdateien und tatsächlicher Integrationsstand der Custom-Platine.

## Vorschlag für zehn Lerntage

**Tag 1:** F001 bis F015. Eine dreiminütige Gesamterklärung aufnehmen. SOC-Offset, LFP-Plateau und EKF ohne Unterlagen zeichnen.

**Tag 2:** F016 bis F025. Die Zell-/Split-Tabelle auswendig erklären, Labels von Onlinefeatures trennen. P01 bis P04 mit Originalunterlagen abgleichen.

**Tag 3:** F026 bis F036. Kapitel 5 anhand aller sieben Abbildungen erklären. Parameterzahl und Lag-Zeiten nachrechnen. Hyperparameterauswahl rekonstruieren.

**Tag 4:** F037 bis F046. Vier Modelle, Eingänge und Zustandseingriffe erklären. Ein künstliches Recovery-Beispiel mit Rückfall zeichnen.

**Tag 5:** F047 bis F060. Für jede Störungsfamilie Eingriff, sichtbaren Befund, plausiblen Mechanismus und Grenze in vier Sätzen notieren.

**Tag 6:** F061 bis F070. Statistik an den Rechenübungen trainieren. Alle Fehlerbalken im Robustheitskapitel korrekt benennen.

**Tag 7:** F071 bis F083. LSTM/GRU zeichnen, Pruning-Matrizen ausschneiden, eine Quantisierungszeile von Hand berechnen.

**Tag 8:** F084 bis F100. Hardwaremesskette und Filterung erklären. Hauptzahlen auf eine einzige Backup-Seite bringen.

**Tag 9:** Den vollständigen Abbildungsatlas durchgehen. Pro Abbildung 30 Sekunden Aussage, 30 Sekunden Methodik und 30 Sekunden Einschränkung. Die kritischen Punkte mit offenen Nachweisen nicht überspringen.

**Tag 10:** Zwei Probeprüfungen zu je 30 bis 45 Minuten. Eine konzentriert sich auf Batterien und Referenzen, die andere auf Statistik und Embedded-Umsetzung. Ungeklärte Antworten als offene Punkte notieren statt improvisieren.

## Simulation einer kritischen Prüfung

**Prüfpfad A:** Sie nennen niedrigen SOH-MAE. Wie wurde SOH gemessen? Warum steigt er? Nutzt die Interpolation Zukunftsinformation? Welche Informationen bekommt das Modell tatsächlich? Warum ist die Filterantwort länger als viele betriebliche Ereignisse? Welche Aussage bleibt dann vom Netz selbst?

**Antwortkern:** Kapazität als operative Referenz erklären, offline Label von Onlineinput trennen, Rohnetz und Filter getrennt bewerten, initiale Information offenlegen und keine schnelle Kapazitätserkennung aus glatten Kurven ableiten.

**Prüfpfad B:** DD gewinnt. Ist der Vergleich fair? Qc enthält doch schon SOC. Wurden alle Features gestört? Warum andere Initialzustände? Warum kein signifikanter Test? Warum nehmen Sie anschließend eine andere Inferenzsemantik auf der Hardware?

**Antwortkern:** Gleiche kausale Informationsbasis und rekonstruierte Features, aber unterschiedliche Hypothesenräume erläutern. Repräsentantenvergleich statt Universalaussage. Paarung und kleine Zellzahl erklären. Continuous als separat nominal validierten Deployment-Modus behandeln, dessen volle Störungsübertragung nicht bereits bewiesen ist.

**Prüfpfad C:** Quantisierung soll effizient sein. Warum ist sie langsamer? Was ist überhaupt quantisiert? Wo sind Energie und RAM gemessen? Wie viel schneller würde optimierter Code sein?

**Antwortkern:** Weight-only-Speicherung von Recurrent-Matrizen, FP32-Rekonstruktion, unveränderte Topologie und nicht optimierte Referenzkerne erklären. Messgrenzen und Proxy nennen. Keine nicht gemessene Beschleunigung durch einen optimierten Kernel versprechen.

## Wie mit einer unbekannten Frage umgehen?

**Antwortmuster:** Zuerst den gesicherten Teil nennen. Dann präzise abgrenzen, welcher zusätzliche Nachweis fehlt. Abschließend einen Versuch oder eine Analyse vorschlagen, die die Frage beantworten würde. Beispiel: Die Tabelle zeigt Recovery-Sensitivität bei -10 Prozent Widerstand. Welcher RC-Zweig den Effekt dominiert, habe ich nicht isoliert. Dazu würde ich Ri, R1 und R2 einzeln perturbieren und Innovation, Kalman-Gain und SOC-Korrektur ereignisweise auswerten.

Nicht verwenden: Das muss am Rauschen liegen. Der Plot sieht gut aus. Mehr Daten machen es automatisch besser. Das Framework wird schon stimmen. Gerade transparente Grenzen zeigen wissenschaftliches Verständnis.

# 12. Quellen und konkrete Rechercheaufträge

**Primärquelle der Ergebnisse:** main.pdf und die zugehörigen TeX-Dateien im Dissertation-Verzeichnis, Stand nach Git-Aktualisierung auf 2d8fdc9. Der Abbildungsatlas enthält eine direkt aus dem PDF erzeugte Seitenzuordnung. figure_inventory.json speichert zusätzlich den SHA-256-Hash der gelesenen PDF. Die folgenden externen Quellen unterstützen Hintergrundlernen, nicht die erneute Verifikation der eigenen Messdaten.

## Q1. Rekurrente Gleichungen und Export

[PyTorch GRU-Dokumentation](https://docs.pytorch.org/docs/main/generated/torch.nn.GRU.html). Nachschlagen: Update-Gate-Konvention und abweichende Position des Reset-Gates gegenüber ursprünglichen Formulierungen. Lernziel: Eine Framework-GRU nicht nur anhand einer allgemeinen Blockskizze nachimplementieren. Diese offizielle Dokumentation wurde für die Vorbereitung geprüft.

Die LSTM-Grundlagen und Framework-Referenzen stehen außerdem in der Dissertation, insbesondere in Kapitel 3.4 und der Bibliografie unter den LSTM-/PyTorch-Referenzen. Lernauftrag: Gate-Reihenfolge und getrennte Biasvektoren im tatsächlich exportierten Modell identifizieren. Bei konkreter Softwareversion deren Dokumentation verwenden, nicht unbesehen eine aktuelle API.

## Q2. Stromintegration und Unsicherheit

[Movassagh et al., A Critical Look at Coulomb Counting](https://arxiv.org/abs/2101.05435). Die Originalarbeit untersucht unter anderem Strommessung, Integrationsnäherung, Kapazitätsunsicherheit und Zeitbasis. Lernauftrag: Eigene Definition von Offset, Gain, Clock-Fehler und Kapazitätsfehler danebenstellen. Die konkrete Fehlerformel dieses Lernskripts selbst herleiten und ihre Voraussetzungen nennen.

## Q3. Optimierte Embedded-Kernel

[Arm CMSIS-NN, offizielles Repository](https://github.com/ARM-software/CMSIS-NN) und [offizielle Fully-Connected-API](https://arm-software.github.io/CMSIS-NN/latest/group__FC.html). Lernauftrag: Gewichts-, Aktivierungs-, Bias- und Akkumulator-Datentypen vergleichen. Herausarbeiten, weshalb ein INT8-Gewichtsarray in einem FP32-C-Kernel noch keine solche Integer-Ausführung ist. Die Quellen wurden zur Einordnung geprüft. Keine daraus abgeleitete Beschleunigungszahl für die eigene Firmware behaupten.

## Q4. Literatur bereits in der Dissertation

**Plett, Battery Management Systems:** Zustandsdefinition, Coulomb Counting, OCV, Observer und Parametrierung wiederholen. **Hyperband, Li et al.:** Budgetzuteilung und Early-Stopping-Verfahren lernen. **Structured Sparsity, Wen et al., sowie recurrent pruning nach Narang/Lobacheva:** Strukturauswahl von Deploymentrepräsentation unterscheiden. **Quantisierung nach Jacob et al. und Krishnamoorthi:** Post-Training-Verfahren, Quantization-Aware Training und Requantisierung auseinanderhalten. Die genauen bibliografischen Angaben stehen in bib/Dissertation.bib. Diese Werke sind gezielte Leseaufträge, keine Behauptung, sämtliche Volltexte seien für diese Vorbereitung erneut geprüft worden.

## Q5. Eigene Belegmappe

Für jede Ergebnisfamilie einen Belegordner anlegen: Konfiguration, Zellliste, Modellhash, Scaler, Skriptversion, Laufmanifest, Ergebnistabelle und Abbildung. Für Hardware zusätzlich Toolchain, Flags, Linker-Map, Firmwarehash und Zeitmessgrenzen. Für die beiden SOH-Auswertungen außerdem Filterzustand und Labeldefinition. Diese Mappe beantwortet die Rückfrage Woher genau stammt diese Zahl wesentlich besser als weitere schöne Diagramme.
