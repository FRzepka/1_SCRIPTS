# 15. Weitere Grundlagenfragen und Tabellenprüfung

Dieser Teil ergänzt die Fragen F001 bis F100 um häufige Nachfragen zur Methodenwahl und um eine vollständige Orientierung durch die 28 nummerierten Tabellen. Seiten beziehen sich wieder auf die PDF, nicht auf die gedruckte Seitenzählung.

## F101. Was unterscheidet ReLU, Sigmoid, tanh und Softmax?

**Antwort:** ReLU gibt bei positiven Eingängen den Wert weiter und bei negativen null. Sigmoid bildet auf 0 bis 1 ab und wird für Gates oder begrenzte Ausgaben verwendet. tanh bildet auf -1 bis 1 ab und kann positive und negative Kandidatenzustände darstellen. Softmax normalisiert mehrere Ausgänge auf eine Summe von eins und passt zu konkurrierenden Klassenwahrscheinlichkeiten.

**Kritische Nachfrage:** Warum wurde Softmax für SOH erwähnt? Bei nur einem Softmax-Ausgang ist die Ausgabe immer eins. Eine sinnvolle Konfiguration müsste sich auf versteckte Schichten oder mehrere Ausgänge beziehen. Der Text nennt eine Aktivierungsvergleichsstudie, aber die konkrete Platzierung muss aus dem ursprünglichen Tuning hervorgehen. Ohne diesen Nachweis nicht behaupten, die Arbeit belege allgemeine Untauglichkeit von Softmax.

## F102. Warum überhaupt skalieren und ist RobustScaler eine Robustheitsgarantie?

**Antwort:** Skalierung verhindert, dass numerisch große Merkmale wie kumulierte Zeit das Training allein durch ihre Größenordnung dominieren. Min-Max verwendet Bereichsgrenzen, RobustScaler typischerweise Median und Interquartilsabstand. Beide müssen mit ausschließlich Entwicklungsdaten bestimmt und unverändert auf Test und MCU angewendet werden.

**Grenze:** RobustScaler ist robust gegenüber manchen extremen Trainingswerten bei der Lage-/Skalenbestimmung. Er macht das neuronale Modell nicht automatisch robust gegen Sensorfehler oder neue Betriebsbereiche. Ein außerhalb des Trainingsbereichs liegendes Merkmal bleibt ein Extrapolationsproblem. **Beleg:** Kapitel 5.2.5 und 7.2.

## F103. Warum MSE als Loss, wenn Sie später MAE berichten?

**Antwort:** MSE ist glatt und gewichtet große Residuen stärker, was für Training praktisch sein kann. MAE bleibt in derselben Einheit wie die Zielgröße und ist gut interpretierbar. Ein MSE-minimierender Kandidat muss nicht der MAE-Minimierer sein.

**Vertiefung:** Die Modellwahl sollte das spätere Anwendungsziel spiegeln. Will man insbesondere Maximalfehler, Recovery oder Störungsresistenz verbessern, reicht nominale MSE möglicherweise nicht. Eine geänderte Loss wäre eine neue Trainingsentscheidung und müsste validation-basiert erfolgen. RMSE enthält dieselbe Ordnung wie MSE auf identischen Samples, aber nicht notwendigerweise nach unterschiedlichen Aggregationen über Zellen.

## F104. Was unterscheidet AdamW von Adam, und was macht Gradient Clipping?

**Antwort:** Adam passt Aktualisierungsschritte anhand von Momentenschätzungen der Gradienten an. AdamW trennt Gewichtszerfall konzeptionell von der adaptiven Gradientenaktualisierung. Gradient Clipping begrenzt große Gradienten, etwa über ihre Norm, und kann instabile Trainingsschritte verhindern.

**Grenze:** Diese Verfahren garantieren weder Generalisierung noch stabile Onlinezustände. Die Dissertation setzt sie als Trainingswerkzeuge ein, nicht als eigenständige wissenschaftliche Neuerung. Warm Restarts ändern die Lernratenplanung, nicht den Controllerzustand. **Recherche:** Die in Kapitel 7 zitierten Originalarbeiten von Loshchilov/Hutter und Pascanu sowie die tatsächlich gespeicherten Trainingskonfigurationen lesen.

## F105. Was unterscheidet globales und lokales Pruning?

**Antwort:** Lokales Pruning legt Auswahl oder Rate schichtweise fest. Globales Pruning vergleicht Strukturen über mehrere Schichten und kann Kapazität ungleich verteilen. Scores verschiedener Schichten müssen dafür sinnvoll vergleichbar sein.

**Grenze:** Die in Kapitel 7 untersuchten Modelle besitzen einen einzelnen recurrent layer und einen spezifischen Gate-Gruppenscore. Daraus folgt keine Aussage über global optimales Pruning eines tiefen Netzes. In den Grundlagen aufgeführte Strategien sind ein Überblick, keine sämtlich durchgeführten Experimente. **Beleg:** Tabellen 3.3 und 3.4.

## F106. Was unterscheidet iterative Auswahl, Pruning-aware Training und Sparse Training?

**Antwort:** One-shot entfernt Strukturen einmal, häufig mit anschließendem Fine-Tuning. Iteratives Pruning entfernt mehrfach kleinere Anteile. Pruning-aware Training berücksichtigt die gewünschte Struktur bereits während der Optimierung. Sparse Training kann von Beginn an mit beschränkter Konnektivität arbeiten und Strukturen dynamisch ändern.

**Grenze:** Diese Ansätze haben verschiedene Trainingskosten und führen nicht automatisch zu gleicher Deploymentrepräsentation. Die Dissertation nutzt in Kapitel 7 einen One-shot-Schnitt mit kurzer Nachanpassung. Nicht versehentlich die Grundlagenübersicht als eigenen Methodenvergleich darstellen.

## F107. Was ist der Unterschied zwischen PTQ und QAT?

**Antwort:** Post-training quantization transformiert ein bereits trainiertes Modell, gegebenenfalls mithilfe von Kalibrierdaten. Quantization-aware training simuliert relevante Quantisierungseffekte während der Gewichtsanpassung. QAT kann das Netz auf die späteren Rundungs- und Clippingeffekte vorbereiten, verursacht aber zusätzlichen Trainingsaufwand.

**Grenze:** Die hier untersuchte zeilenweise Max-Skalierung der Gewichte benötigt kein repräsentatives Aktivierungskalibrierset wie eine vollständige Aktivierungsquantisierung. Sie ist trotzdem nicht allgemein optimal. Die Speicher- und Laufzeitgrenze hängt vom Exportkernel ab, nicht allein von PTQ versus QAT. **Beleg:** Tabellen 3.6 und 3.7 sowie Kapitel 7.5.

## F108. Warum Per-Row statt Per-Tensor oder Per-Group?

**Antwort:** Ein gemeinsamer Skalenfaktor für die gesamte Matrix kann von einer großen Ausreißerzeile bestimmt werden und kleine Zeilen grob auflösen. Zeilenweise Skalen passen sich jedem Ausgangskanal an. Kleinere Gruppen erlauben noch feinere Anpassung, benötigen aber mehr Skalen und komplexeren Zugriff.

**Grenze:** Der hier gewählte Modus ist ein nachvollziehbarer Kompromiss, kein experimenteller Sieg über jede andere Granularität. Für eine Hardwareentscheidung zählen außerdem Skalenanwendung, Vektorisierung und Datentransfer. **Beleg:** Tabelle 3.6 und Abbildung 7.5.

## F109. Warum kann eine einzelne große Gewichtszahl Quantisierung verschlechtern?

**Antwort:** Bei Max-Abs-Skalierung bestimmt das größte Gewicht die Schrittweite der ganzen Zeile. Viele kleine Gewichte können dadurch auf denselben Code oder null runden. Ein größerer darstellbarer Bereich bedeutet bei festem Bitbudget gröbere Auflösung.

**Vertiefung:** Clipping könnte die Mehrheit genauer repräsentieren, verzerrt aber große Gewichte. Ob das günstiger ist, muss mit Daten und finaler Task-Metrik geprüft werden. Die Dissertation nutzt einen einfach reproduzierbaren Max-Abs-Pfad. Die Halbschrittgrenze gilt für genau diesen Pfad ohne zusätzliches Clipping, nicht für jede Quantisierungsstrategie.

## F110. Kann man Pruning und Quantisierung kombinieren?

**Antwort:** Prinzipiell ja, weil sie unterschiedliche Stellgrößen verändern: Topologie und numerische Repräsentation. Man könnte zuerst Kanäle auswählen und fine-tunen und anschließend das kleinere Netz quantisieren. Reihenfolge und gemeinsame Nachanpassung können die Güte beeinflussen.

**Grenze:** Die berichteten drei Hauptvarianten sind Base, Pruned und Quantized. Ohne eine vierte kombinierte Evaluation darf kein additiver oder multiplikativer Gewinn versprochen werden. Kompressionseffekte können interagieren. Eine neue Variante braucht denselben vollständigen Accuracy-, Transienten-, Speicher- und Laufzeittest.

## F111. Warum STM32 statt GPU, FPGA oder ASIC?

**Antwort:** Der Zielanwendungsfall benötigt lokale, autonome BMS-Ausführung mit begrenztem Speicher, Kosten und Energie. Ein Mikrocontroller passt zu Messung, Schutz und Kommunikation. GPU, FPGA und ASIC bieten andere Parallelitäts-, Flexibilitäts- und Entwicklungsprofile.

**Grenze:** Der Vergleich in Tabelle 3.9 ist funktional, keine eigene Messkampagne über diese Plattformen. Eine GPU kann Training beschleunigen, ohne für das spätere Modul geeignet zu sein. Ein FPGA kann deterministische Datenpfade liefern, aber seine Entwicklung und Verifikation sind andere Aufgaben. Keine pauschale Energiehierarchie ohne Arbeitslast und Messung behaupten.

## F112. Warum manuelle C-Kerne statt automatischem Konverter?

**Antwort:** Sie machen recurrent states, Gateordnung, Skalierung und Quantisierungsgrenzen explizit und erlauben eine nachvollziehbare Referenzimplementierung. Die Arbeit kann so Operationsfolge und Speicher besser kontrollieren.

**Grenze:** Manuelle Umsetzung ist fehleranfällig und nicht automatisch schneller. Numerische Äquivalenztests sind daher notwendig. Automatische Toolchains können hochoptimierte Operatoren bereitstellen, deren Zustandssemantik und Modellabdeckung aber geprüft werden müssen. Tabellen 3.5 und 3.8 sind Softwareübersichten, keine Garantie der Unterstützung genau des eigenen Modellgraphen in jeder Version.

## F113. Was ist ein sinnvoller Fallback für ein neuronales BMS-Modell?

**Antwort:** Ein unabhängig überprüfter einfacher Schätzer, Plausibilitätsgrenzen und konservative Betriebsgrenzen könnten bei unplausiblen Eingängen oder Ausgaben aktiviert werden. Der Wechsel müsste Zustandskonsistenz und Übergangssprünge berücksichtigen.

**Grenze:** Das ist ein Entwicklungsvorschlag und kein bereits validiertes Ergebnis der Dissertation. Ein Fallback-Coulomb-Counter kann denselben Stromsensorfehler teilen. Diversität der Modelle allein garantiert daher keine unabhängige Fehlerabsicherung. Schutzfunktionen müssen unabhängig von einer optimistischen SOC-Prognose wirken.

## F114. Welche zusätzliche Untersuchung hätte den größten Erkenntnisgewinn?

**Antwort:** Je nach Ziel: Für Klassenübertragbarkeit mehr unabhängige Zellen und Betriebsdomänen. Für schnelle DD-Deploymentaussagen die wichtigsten Störungstests im Continuous-Modus. Für die SOH-Netzleistung eine eindeutig dokumentierte Raw-/Filter-Ablation mit bekannten Anfangsbedingungen. Für Energie echte synchronisierte Leistungsmessung. Für INT8 optimierte, identisch instrumentierte Kernel.

**Priorisierung:** Nicht alles gleichzeitig versprechen. Eine gute Prüfungsantwort nennt den konkreten offenen Schluss und den kleinsten Versuch, der ihn beantwortet. Für den behaupteten Übergang vom robusten Softwaremodell zum schnellen Controller ist die Prüfung der Inferenzsemantik besonders unmittelbar.

## Tabellen 1.1 und 8.1: Beitrag und Synthese

**PDF-Seiten 33 und 185.** Tabelle 1.1 ordnet Studien, Beiträge und Publikationsbasis zu. Tabelle 8.1 verknüpft die Beiträge mit Anforderungen. **Prüfungsfrage:** Welche Zelle der Synthesetabelle ist direkt gemessen und welche indirekt relevant? **Antwort:** MLP-Repräsentation ist primär eine Schätzstudie, kein dortiger vollständiger MCU-Test. Kapitel 6 und 7 enthalten direkte Kernmessungen, Energie in Kapitel 7 aber nur Proxy. **Offen:** Publikationsstatus ist zeitabhängig. Für die tatsächliche Verteidigung den dann gültigen Status prüfen, nicht aus der September-PDF dauerhaft übernehmen.

## Tabellen 3.1 und 3.2: Chemie und Schätzerprinzipien

**PDF-Seiten 40 und 52.** Die Tabellen bündeln typische Eigenschaften von NMC/LFP sowie Kalman-/Neuronalschätzung. **Prüfungsfrage:** Sind das harte Ausschlusskriterien? **Antwort:** Nein, es sind typische Stärken und Grenzen unter den jeweils notwendigen Voraussetzungen. Ein neuronales Netz kann Physikfeatures nutzen, ein Observer adaptive oder gelernte Komponenten. Beobachtbarkeit und Trainingsabdeckung sind unterschiedliche Arten begrenzter Information. **Lernauftrag:** Zu jeder behaupteten Stärke ein Gegenbeispiel oder eine Bedingung nennen.

## Tabellen 3.3 bis 3.5: Pruningmethoden und Werkzeuge

**PDF-Seiten 56, 61 und 63.** Granularität, Optimierungsverfahren und Softwareunterstützung sind drei unterschiedliche Achsen. **Prüfungsfrage:** Erzeugt eine hohe Sparse-Rate schon eine schnelle Firmware? **Antwort:** Nein, Format, Operator und Kernel müssen die Sparsity ausnutzen. **Lernauftrag:** One-shot/local/gate-group/structured/dense-export als genaue Beschreibung der eigenen Kapitel-7-Variante zusammensetzen. Frameworkfähigkeiten sind versionsabhängig und dürfen nicht ungeprüft als aktuelle Vollabdeckung ausgegeben werden. Siehe F105, F106 und F112.

## Tabellen 3.6 bis 3.8: Quantisierungsmethoden und Werkzeuge

**PDF-Seiten 68, 70 und 72.** Die Tabellen behandeln Granularität, Strategie und Software. **Prüfungsfrage:** An welcher Stelle liegen Kalibrierungsdaten, Rundung, Integerarithmetik und Deploymentoperator? **Antwort:** Das sind getrennte Designentscheidungen. Der eigene Versuch ist zeilenweise symmetrische PTQ nur für Recurrent-Gewichte mit FP32-Zuständen. **Grenze:** Nicht alle aufgeführten Modi wurden selbst gemessen. Siehe F107 bis F109.

## Tabelle 3.9: Rechenplattformen

**PDF-Seite 75.** **Prüfungsfrage:** Welche Plattform würden Sie bei härteren Latenzgrenzen wählen? **Antwort:** Erst Rechenlast, Parallelität, Energie, Stückzahl, Entwicklungsaufwand und Sicherheitsanforderungen festlegen. Die Dissertation misst nur die gewählten STM32-Pfade. Ein Tabellenvergleich der Prinzipien ersetzt keine Portierungsstudie. Siehe F111.

## Tabellen 4.1 bis 4.3: Daten, NMC-Szenarien und Zellgrenzen

**PDF-Seiten 80, 81 und 86.** **Prüfungsfrage:** Sind Zellen, Szenarien und Samples dasselbe n? **Antwort:** Nein. 14 NMC-Zellen stehen sieben Szenarien gegenüber. Beim LFP-Test sind sechs Zellen die primären unabhängigen Einheiten, viele Samples beschreiben deren Verlauf. Die vollständige Split-Tabelle ist verbindlicher als eine unklare Farblegende. **Lernauftrag:** Für C09/C13/C15 ähnliche mittlere Lastgruppe, für C29 explorative High-Gruppe und für C27 fehlende Alterungsfenster erklären. Siehe F016 bis F025.

## Tabelle 5.1: Finale MLP-Größe

**PDF-Seite 105.** **Prüfungsfrage:** Wie viele trainierbare Parameter hat diese Tabelle? **Antwort:** Erst tatsächliche Eingangsgröße klären, dann pro Schicht Eingänge mal Ausgänge plus Bias summieren. Bei 48 Inputs ergibt die gedruckte Breitenfolge 434561 Parameter. **Grenze:** Das ist eine bedingte Rechnung, keine Zählung aus einem geladenen Checkpoint. Die schematische Grafik und der anfängliche Acht-Neuronen-Kandidat sind kein Ersatz für die Tabelle. Siehe F030.

## Tabellen 6.1 und 6.2: Repräsentanten und Interventionen

**PDF-Seiten 115 und 121 bis 123.** **Prüfungsfrage:** Welche Teile der Klasse wurden nicht untersucht? **Antwort:** Zum Beispiel alternative ECM-Ordnungen, Hysteresezustände, adaptive Kovarianzen, andere neuronale Architekturen und kombinierte Sensorfehler. **Lernauftrag:** Jede Interventionsstärke mit Einheit nennen und physikalischen Gain von Offset unterscheiden. Die Parameter sind kontrollierte Stressdefinitionen, keine geschätzte Wahrscheinlichkeit realer Sensorfehler. Siehe F037 bis F060.

## Tabelle 6.3: Empfehlung nach Priorität

**PDF-Seite 150.** **Prüfungsfrage:** Welchen Vertreter würden Sie tatsächlich einsetzen? **Antwort:** Bei primärer nominaler Genauigkeit und geringer systematischer Stromfehlersensitivität spricht der Versuch für DD. Bei Recovery HECM, bei minimalem Rechenbudget DM/HDM. Die schnelle DD-Empfehlung verwendet Continuous und verlangt zusätzliche Störungsvalidierung dieser Semantik. **Grenze:** Eine Prioritätsempfehlung ist keine pauschale Systemfreigabe.

## Tabellen 7.1 und 7.2: Hardware-KPIs

**PDF-Seite 167.** **Prüfungsfrage:** Sind Latency und Inference doppelte Messungen? **Antwort:** Nein, Hostlatenz und Kernzeit haben unterschiedliche Grenzen. Flash ist gelinkter Speicher, RAM enthält statische Daten und beobachteten Stack, Eest ist aus Inference berechnet. **Lernauftrag:** SOC Base/Pruned/Quantized mit 1,40/0,80/6,99 ms und SOH mit 22,73/12,72/29,21 ms erklären. Keine unabhängige Energiebestätigung aus der letzten Spalte ableiten. Siehe F087 bis F091.

## Tabelle 7.3: Kompressionstrade-off

**PDF-Seite 178.** **Prüfungsfrage:** Warum ist negativer ΔMAE gut, negativer ΔFlash ebenfalls gut und U kleiner eins günstig? **Antwort:** Es sind Änderungen gegenüber Base beziehungsweise normierte Kosten. Die Vorzeichen verschiedener Tabellen dürfen nicht mit den höher-ist-besser-Robustheitsscores aus Kapitel 6 verwechselt werden. **Grenze:** Gerundete Tabellenwerte können kleine Unterschiede zur Rechnung aus ungerundeten Daten erzeugen. Mehrere Prozentänderungen sind nicht einfach zu addieren.

## Tabelle A.1: Lookupvariation

**PDF-Seite 219.** **Prüfungsfrage:** Was bedeuten eckige Angaben wie 6/22? **Antwort:** Die Caption definiert Anzahlen der Subfälle mit null-ausschließendem Interaktionsintervall, nicht Grenzen eines einzigen Intervalls. Der Spaltenkopf mit CI kann deshalb missverständlich wirken. **Wichtig:** Die Tabelle enthält tatsächlich nicht nur Widerstand und OCV, sondern auch Zeitkonstanten. -10 Prozent Widerstand erzeugt 5,10 h Recovery-or-censor und 17 Prozent zensierte Zellen. Siehe F059 und F060.

## Tabelle A.2: Nominale Unsicherheiten

**PDF-Seite 219.** **Prüfungsfrage:** Was ist P95 und warum nicht derselbe Wert wie RMSE? **Antwort:** P95 ist ein Quantil der absoluten Fehler, RMSE ein quadratisches Mittel aller Fehler. Beide beschreiben verschiedene Aspekte. DD-P95 beträgt hier 0,0508, während MAE 0,0258 ist. **Grenze:** Ein quantilbezogener Makrowert ist nicht automatisch das Quantil aller gepoolten Samples. Aggregationsreihenfolge beachten.

## Tabelle A.3: Gepaarte Tests

**PDF-Seite 220.** **Prüfungsfrage:** Warum durchweg pHolm = 0,1875? **Antwort:** Diskrete exakte Tests mit sechs Zellen und Korrektur über sechs Vergleiche liefern eine grobe Ergebnisauflösung. Die Tabelle enthält etwa HECM gegen DD mit Mittelwertdifferenz -0,0079 und einem punktweisen Intervall unter null, aber exaktem p = 0,0938. **Grenze:** Standardisiertes dz kann bei kleiner Stichprobe instabil sein. Rohe Differenzen und Einheiten ebenfalls nennen. Siehe F063 und F064.

## Tabelle A.4: Ausgewählte Störungspenalties

**PDF-Seite 220.** **Prüfungsfrage:** Ist -0,0000 eine reale Verbesserung? **Antwort:** Es ist ein gerundeter sehr kleiner negativer Wert. Für Interpretation ungerundete Daten und Unsicherheit prüfen. Die Tabelle fasst globale MAE-Kontraste zusammen und ersetzt keine lokale Spikeanalyse. Die Offsetwerte sind zusätzliche MAE, nicht gesamte MAE. Siehe F048 und F053.

## Tabelle A.5: Score-Sensitivität

**PDF-Seite 220.** **Prüfungsfrage:** Warum ändern sich Abstände stark mit Gewichtung? **Antwort:** Familien, Severity-Level und Einzelszenarien erhalten unterschiedliche relative Bedeutung. DD bleibt im dargestellten Kandidatensatz führend, was begrenzte Rankingstabilität zeigt. **Grenze:** Kein frei gewähltes Gewichtungssystem ist eine gemessene Häufigkeit von Feldfehlern. Siehe F068.

## Tabelle A.6: Recovery und Hardware zusammen

**PDF-Seite 221.** **Prüfungsfrage:** Warum ist DM-Relapse null, obwohl DM nicht recovern kann? **Antwort:** Ohne qualifizierten Bandeintritt gibt es auch keinen anschließenden Rückfall. Niedriger Relapse allein ist deshalb nicht automatisch gut. DM/HDM haben hohe Zensur. **Zweiter Prüfpunkt:** Frühe bereits erfüllte Endpunkte sind konservative obere Zeitgrenzen, weil der gemeinsame Scorebeginn später liegt. DD-Hardwarewerte betreffen Continuous. Siehe F042 bis F046 und F085.

## Tabelle A.7: Fehleränderung nach Bitflip

**PDF-Seite 226.** **Prüfungsfrage:** Warum ist der SOH-Median negativ und trotzdem kein Robustheitsgewinn? **Antwort:** Ein einzelner Fault kann eine bestehende Zielabweichung zufällig kompensieren. Die gepaarte Outputabweichung, Peak und Recovery bleiben zusätzliche Kriterien. P95 der ereignisweisen MAE-Änderung ist nicht P95 aller instantaneous errors. **Grenze:** Ein bestimmtes Fraction-Bit und wenige Ereignisse erfassen keine vollständige Fehlertoleranz. Siehe F097.
