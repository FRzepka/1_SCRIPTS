# Response to Reviewer #4

Manuscript: *Embedded Artificial Intelligence in Battery Management Systems: Pruning and Quantization for Efficient State-of-Charge and State-of-Health Estimation*

Revision working copy: `review_1/Embedded_Ai_Manuscript.tex`

Compiled revision: `review_1/review_build/Embedded_Ai_Manuscript.pdf`

Line-numbered anonymous revision: `review_1/review_build/Embedded_Ai_Manuscript_Anonymized.pdf`

The page and line references below refer to the final 57-page line-numbered anonymous revision compiled on 15 July 2026. Figure captions and other float contents are identified by figure number and page because the `lineno` package does not assign reliable line numbers inside floats. No new battery experiment, model-training campaign, or STM32 measurement was performed for these comments. The two papers explicitly suggested by the reviewer were imported from the author's Zotero bibliography and are cited as Refs. [16] and [17]. Figures 2, 6, and 7 are revised explanatory schematics based on the documented dataset design and verified implementation; they do not introduce new experimental results.

## General response

We thank the reviewer for the detailed comments on presentation, literature context, and mathematical clarity. We shortened and refocused the Abstract, added and analysed the two suggested references, reorganised the method description around the actual Base, Pruned, and Quantized transformations, and added implementation-level equations for pruning, quantization, mixed-precision inference, and evaluation metrics. We also divided long paragraphs, performed a manuscript-wide language and logic pass, and regenerated three method figures at higher resolution with larger text and code-consistent content.

## Comment 1

> The expression of the abstract should be improved. The length should be suitable for the abstract, not too long.

**Response:** We agree. The Abstract was rewritten as a compact 169-word summary. It now states the engineering application, identifies the implemented artificial intelligence, describes the common STM32 benchmark, quantifies the main SOC/SOH accuracy and resource results, distinguishes the timing-derived energy proxy from a power measurement, and closes with the scope limitation. All acronyms are defined at first use. The title contains no acronym, and the keyword list has been reduced to six written-out terms.

**Changes in the manuscript:** revised Abstract and keywords (page 1, lines 6--27; Abstract lines 6--24 and keywords lines 25--27).

## Comment 2

> Please conduct more reference analysis for the research topic, such as An improved parameter identification and radial basis correction-differential support vector machine strategies for state-of-charge estimation of urban-transportation-electric-vehicle lithium-ion batteries, Improved singular filtering-Gaussian process regression-long short-term memory model for whole-life-cycle remaining capacity estimation of lithium-ion batteries adaptive to fast aging and multi-current variations, and so on.

**Response:** We added both suggested articles from the author's Zotero export and analysed their relationship to the present work. The revised Related Work explains that the first combines online parameter identification, adaptive radial-basis correction, and a differential support-vector machine for SOC estimation under dynamic vehicle tests, while the second combines singular filtering, Gaussian-process regression, and an LSTM for whole-life capacity estimation under fast ageing and varying currents. We then state the relevant distinction: these studies broaden estimation methods and operating conditions, whereas neither reports the paired post-training pruning/quantization comparison with linked-firmware memory and recurrent microcontroller timing investigated here. They are therefore treated as complementary evidence, not as direct benchmark substitutes.

**Changes in the manuscript:** new comparative literature analysis and citations [16,17] (page 6, lines 144--155); full bibliography entries with DOI metadata (page 53, lines 969--980).

## Comment 3

> The expression logic can be improved for your proposed method so that the innovation can be clearly understood. More mathematical analysis should be conducted to clarify the methods.

**Response:** We reorganised the methods to distinguish three operations explicitly: the common FP32 Base equations, structured pruning that changes the hidden dimension and dense parameter submatrices, and quantization that changes the stored representation of selected recurrent weights without changing topology. The revised text now gives the analytical MAC expression, the retained-index submatrix construction in Eq. (17), the exact symmetric per-row quantizer in Eqs. (18)--(20), the reconstruction-error bound in Eq. (21), and the mixed-precision STM32 gate equation in Eq. (22). It also states which quantities remain FP32 and why storage reduction does not imply fully integer execution. These additions clarify that the contribution is an auditable paired deployment benchmark rather than a new recurrent cell or compression algorithm.

**Changes in the manuscript:** common architecture and analytical MAC scaling (pages 15--16, lines 328--353); structured pruning definition and dense submatrix construction (pages 17--19, lines 384--425; Fig. 6 on page 19); exact quantization and mixed-precision equations (pages 20--21, lines 432--462; Fig. 7 on page 22); explicit benchmark objective and contributions (pages 4--5, lines 79--127).

## Comment 4

> Grammar should be checked and improved for the entire content. Please try to make every sentence correct and easy to understand.

**Response:** We performed a manuscript-wide language pass and simplified ambiguous or overly compressed sentences. Particular attention was given to the Abstract, Introduction, Related Work, preprocessing definitions, pruning and quantization methods, Results interpretation, Discussion, Limitations, Conclusion, and declarations. We also corrected an author-name encoding error and removed duplicated acknowledgement text.

**Representative changes in the manuscript:** Abstract (page 1, lines 6--24); Introduction and Related Work (pages 4--8, lines 79--217); Methods (pages 9--25, lines 221--566); Results and Discussion (pages 26--38, lines 567--825); Conclusion and declarations (pages 39--51, lines 826--909).

## Comment 5

> Please pay more attention to the expression logic and ensure the logic is correct for every sentence.

**Response:** We agree and revised the argument flow from motivation to evidence boundary. The Introduction now closes with one objective and three contributions; Related Work ends with the precise research gap; the methods separate data, targets, common architecture, compression transformations, and measurement definitions; the Results distinguish observations from causal claims; and the Discussion separates measured findings, implementation-based interpretations, and unevaluated transferability. Unsupported statements, including a causal regularisation explanation for the lower Pruned SOC MAE and independent energy-measurement wording, were removed or bounded.

**Changes in the manuscript:** objective and contribution sequence (page 5, lines 112--127); research gap and implementation boundary (pages 7--8, lines 169--217); Base/Pruned/Quantized method boundary (pages 15--21, lines 328--462); non-causal interpretation and evidence limitations (pages 35--38, lines 726--825); bounded Conclusion and Outlook (pages 39--40, lines 826--864).

## Comment 6

> Some paragraphs are too long. Please optimize your expression.

**Response:** We split and tightened long paragraphs throughout the manuscript. The revised structure separates dataset design from check-up procedures, training settings from deployment state handling, benchmark replay from firmware measurement, static operation counts from timing interpretation, and demonstrated limitations from future work. This reduces sentence load while preserving the quantitative details required for reproducibility.

**Representative changes in the manuscript:** dataset and preprocessing (pages 9--14, lines 221--327); training and deployment descriptions (pages 16--17, lines 354--383); benchmark methodology (pages 21--26, lines 471--566); Discussion and limitations (pages 35--38, lines 726--825); Conclusion and Outlook (pages 39--40, lines 826--864).

## Comment 7

> Some figures are unclear. Please try to optimize them.

**Response:** We regenerated three explanatory figures at high resolution with larger labels and a consistent colour system. Figure 2 now shows the actual low/high values for all three design-of-experiments factors and clarifies the eight operating-point corners. Figure 6 now depicts the dense hidden-channel pruning path and the implemented SOC and SOH dimension changes. Figure 7 now shows the verified mixed-precision storage and STM32 computation path; it also corrects the old schematic's inconsistent quantization range from `-128...127` to the implemented symmetric set `-127...127`, with `-128` unused. The generator is retained in the review folder for reproducibility.

**Changes in the manuscript:** revised Fig. 2 and caption (page 10); revised Fig. 6 and caption (page 19); revised Fig. 7 and caption (page 22); reproducible source in `review_analysis/tools/generate_reviewer4_figures.ps1`.

## Comment 8

> More expression about the mathematical analysis should be conducted to show your idea clearly. Please try to highlight your proposed method and focus on it.

**Response:** In addition to the method equations described in our response to Comment 3, we formalised the evaluation criteria so the relation between predictions and reported percentages is unambiguous. The revised manuscript defines signed percentage-point error, MAE, RMSE, and the empirical P95 threshold in Eq. (23) and the following expression. It also connects analytical MAC reductions with the implemented hidden-size changes and explicitly distinguishes these architecture-level counts from measured flash, latency, and the timing-derived energy proxy. This mathematical treatment focuses the contribution on a transparent, reproducible comparison of two deployment transformations.

**Changes in the manuscript:** analytical MAC count and asymptotic reduction (pages 15--16, lines 336--348); pruning equations (page 18, lines 391--407); quantization and reconstruction equations (pages 20--21, lines 439--462); formal error metrics (page 24, lines 536--541); measured/static interpretation (pages 36--37, lines 743--775); analytical scaling in Appendix A.6 and Fig. A.21 (page 46).

## Verification

- The author manuscript compiles to 36 pages.
- The line-numbered anonymous manuscript compiles to 57 pages.
- The final logs contain no undefined citations, undefined references, or LaTeX errors.
- The only remaining BibTeX warnings concern eight pre-existing entries with missing year or journal metadata; neither newly added Wang entry produces a warning.
