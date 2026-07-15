# Response to Reviewer #2

Manuscript: *Embedded Artificial Intelligence in Battery Management Systems: Pruning and Quantization for Efficient State-of-Charge and State-of-Health Estimation*

Revision working copy: `review_1/Embedded_Ai_Manuscript.tex`

Compiled revision: `review_1/review_build/Embedded_Ai_Manuscript.pdf`

Line-numbered anonymous revision: `review_1/review_build/Embedded_Ai_Manuscript_Anonymized.pdf`

The page and line references below refer to the final 57-page line-numbered anonymous revision compiled on 15 July 2026. No new model training, hardware measurement, calculation, or figure was required specifically to address these comments. The revised literature positioning for Reviewer 2 used verified references already contained in the manuscript bibliography; two additional Zotero-verified references were subsequently added for Reviewer 4. All page and line references in the responses to Reviewers 1, 3, and 4 were refreshed after the final integration.

## General response

We thank the reviewer for the constructive comments concerning the positioning and presentation of the work. We substantially rewrote the Introduction and Related Work sections, formulated one concise objective followed by three explicit contributions, and separated the demonstrated novelty from claims that are outside the scope of the study. We also rewrote the Conclusion and Outlook to quantify the principal results, state their practical meaning and limitations, and identify concrete future work. Finally, the manuscript received a language and logic pass, with particular attention to the Abstract, Introduction, Related Work, Results, Discussion, and Conclusion.

## Comment 1

> The novelty of the work must be clearly addressed and discussed, compare your research with existing research findings and highlight novelty, (compare your work with existing research findings and highlight novelty).

**Response:** We agree. The revision now distinguishes the contribution from both battery-estimation studies and generic model-compression studies. The Related Work section compares the present benchmark with prior embedded SOC and SOH implementations, including studies reporting microcontroller timing and memory and recent work on compact or quantized estimators. It also contrasts the present battery-specific, continuous-state benchmark with a generic pruning-and-quantization benchmark. This comparison identifies the remaining gap: existing work generally reports either estimation accuracy, one embedded implementation, or one compression method, but does not provide the same paired, auditable comparison of structured pruning and recurrent-weight quantization for both SOC and SOH under one STM32 measurement protocol. We now state explicitly that the novelty lies in this end-to-end comparative deployment benchmark and its transparent evidence boundaries, not in proposing a new recurrent cell or a new compression algorithm.

**Changes in the manuscript:** motivation, research gap, objective, and contribution statement (pages 4--5, lines 79--127); expanded comparison with prior embedded and compression studies (pages 5--8, lines 128--217, especially pages 7--8, lines 169--217); bounded contribution statement in the Conclusion and Outlook (pages 39--40, lines 826--864).

## Comment 2

> The main objective of the work must be written on the more clear and more concise way at the end of introduction section.

**Response:** We agree. The end of the Introduction now contains one direct objective sentence: the study quantifies how structured pruning and recurrent-weight quantization change estimation error, memory footprint, inference time, and a timing-derived energy proxy for continuously executed SOC and SOH estimators on the same STM32 platform. This sentence is followed by three compact contributions covering the common benchmark, the two transparent compression paths, and the joint accuracy/resource analysis.

**Changes in the manuscript:** concise objective and three-item contribution statement at the end of the Introduction (page 5, lines 112--127).

## Comment 3

> Introduction section must be written on more quality way, i.e. more up-to-date references addressed. Research gap should be delivered on more clear way with directed necessity for the conducted research work.

**Response:** We agree. The Introduction was restructured to move from the current SOC/SOH estimation context to embedded constraints and then to the specific need for a common compression benchmark. The literature discussion now includes recent 2024 and 2025 review and embedded-deployment studies already present and verified in the project bibliography, alongside the relevant foundational work. We also explain why cross-study accuracy values cannot be compared without considering chemistry, data split, target construction, and error definitions. The resulting research gap motivates the study directly: deployment decisions require accuracy, Flash, RAM, timing, and implementation precision to be reported under one controlled protocol. The existing verified bibliography was sufficient for this purpose, so no new or unverified source was added.

**Changes in the manuscript:** rewritten Introduction with current context and embedded motivation (pages 4--5, lines 79--127); updated and more critical literature synthesis and explicit research gap (pages 5--8, lines 128--217).

## Comment 4

> Conclusion section is missing some perspective related to the future research work, quantify main research findings.

**Response:** We agree. The Conclusion and Outlook was rewritten. It now reports the principal changes in Flash, RAM, inference time, and MAE separately for SOC and SOH and for both compression variants. It distinguishes measured inference time from the constant-power energy proxy and explains the practical operating points: structured pruning is the more balanced option for the evaluated firmware, whereas recurrent-weight quantization is mainly attractive when Flash is the dominant constraint and its mixed-precision runtime overhead is acceptable. The conclusion also states the limits of transfer to other chemistries, controllers, initialisations, and fault classes. Future work is prioritised around kernel optimisation and direct power profiling, broader validation across seeds, temperatures, chemistries, and controller families, training-aware compression, and internal embedded fault injection.

**Changes in the manuscript:** revised Conclusion and Outlook (pages 39--40, lines 826--864); quantified findings (page 39, lines 834--844); engineering interpretation and scope (page 39, lines 845--854); prioritised future work (pages 39--40, lines 855--864).

## Comment 5

> English language should be carefully checked and carefully check paper for language typos.

**Response:** We agree. We performed a language and logic pass throughout the manuscript. This included correcting grammar and terminology in the Abstract, rewriting long and repetitive passages in the Introduction and Related Work, clarifying the independence of the SOC and SOH estimators in the Methods, removing formulaic and unsupported wording from the Results, and rewriting the Discussion and Conclusion for more direct technical interpretation. We also standardised terms such as `mean absolute error`, `inference time`, `estimated energy`, `FP32`, and `INT8` according to their defined meanings.

**Representative changes in the manuscript:** revised Abstract (page 1, lines 6--24); rewritten Introduction and Related Work (pages 4--8, lines 79--217); clarified model-output wording (page 12, lines 262--270); revised Results opening and interpretation (pages 26--34, lines 567--711); revised Conclusion and Outlook (pages 39--40, lines 826--864).
