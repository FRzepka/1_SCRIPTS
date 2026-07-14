# Response to Reviewer #1

Manuscript: *Embedded Artificial Intelligence in Battery Management Systems: Pruning and Quantization for Efficient State-of-Charge and State-of-Health Estimation*

Revision working copy: `review_1/Embedded_Ai_Manuscript.tex`

Compiled revision: `review_1/review_build/Embedded_Ai_Manuscript.pdf`

Line-numbered anonymous revision: `review_1/Embedded_Ai_Manuscript_Anonymized.pdf`

The page and line references below refer to the 49-page line-numbered anonymous revision compiled on 14 July 2026. Figure captions and other float contents are identified by figure number and page because the `lineno` package does not assign reliable line numbers inside floats. No new STM32 measurements, battery experiments, HPC training runs, random-seed study, or validation on another chemistry were performed. Added analyses use archived model artefacts and saved output trajectories, and the revised manuscript states this boundary explicitly.

## General response

We thank the reviewer for the detailed comments. We revised the method descriptions, removed unsupported causal claims, documented the exact scope of the energy estimate and mixed-precision quantization, and added five supplementary analyses. These analyses cover pruning diagnostics, temporal stability, SOH filtering, utility-weight sensitivity, and limited software-level output faults. We have carefully distinguished observations from the evaluated checkpoints and replay from conclusions that would require new chemistry, training, power-measurement, or internal fault-injection campaigns.

## Comment 1: Cell-chemistry generalisation

> This study only evaluated the estimator on the LFP DoE cycle aging dataset containing a single battery chemical system. The authors need to discuss whether the observed trade-off between accuracy and compression can be generalized to other chemical systems with different degradation characteristics, such as NMC or LCO.

**Response:** We agree that the original manuscript did not delimit chemistry transferability clearly enough. The revised Discussion now separates architecture-level storage effects from model accuracy. For a fixed architecture, removing the same number of hidden channels or storing the same recurrent matrices as INT8 produces comparable parameter-count changes independent of chemistry. However, the accuracy-compression trade-off depends on chemistry, operating range, degradation trajectory, and training data. Our numerical MAE and preferred compression level are therefore demonstrated only for the evaluated JGC LFP cells at 25 degrees Celsius and cannot be transferred to NMC or LCO without retraining and validation. We added relevant NMC SOC/SOH literature to show feasibility of compact data-driven models while explicitly stating that those studies do not validate our numerical trade-off.

**Changes in the manuscript:** scope statement in the Abstract (page 1, lines 24--27); chemistry-transferability discussion (pages 34--35, lines 715--725); expanded limitations (page 35, lines 734--740); Conclusion and Outlook (page 36, lines 758--768).

## Comment 2: Quantization-aware and pruning-aware training

> The pruning and quantization experiments were conducted on the fully-precision floating-point trained model. The authors need to investigate whether quantization-aware training or pruning-aware training, compared to post-compression training, can further improve the trade-off between accuracy and efficiency.

**Response:** We clarified that the present study evaluates two deployable transformations of trained FP32 checkpoints: one-shot structured pruning followed by brief post-pruning fine-tuning, and post-training recurrent-weight quantization. Quantization-aware training, pruning-aware training, and an iterative pruning schedule were not performed because they require a new training campaign. The revised paper no longer suggests that the tested variants represent the best attainable jointly trained compression result. It explains why training-aware compression may recover accuracy and identifies this comparison as future work.

**Changes in the manuscript:** structured magnitude pruning and the explicit post-pruning fine-tuning boundary (page 16, lines 348--360); post-training mixed-precision quantization details (page 17, lines 367--379); Discussion of the compression boundary and missing training-aware comparison (page 35, lines 726--740); future work (page 36, lines 762--769).

## Comment 3: Evidence for the claimed pruning regularisation effect

> The SOC estimator showed a slight improvement in MAE after pruning, which is attributed to the regularization effect. The authors need to provide evidence of weight distribution or unit activation statistics to support this regularization explanation.

**Response:** We agree that the original causal interpretation was too strong. We removed the claim that pruning acted as a regulariser. The revised text reports the lower SOC MAE as an observation for one checkpoint and test split. We added a post-hoc L2 unit-saliency ranking and recurrent-weight distribution comparison. The 19 removed SOC units occupy the lowest saliency ranks by construction, which verifies implementation of the stated criterion. The figure is explicitly described as diagnostic evidence, not proof of improved generalisation. We also state that repeated seeds, cross-validation, and a matched fine-tuning ablation were not available.

**Changes in the manuscript:** pruning criterion and diagnostic scope (page 16, lines 337--360); Error Distribution Analysis and removal of the causal regularisation claim (page 27, lines 565--574); Discussion (page 33, lines 671--679); Appendix A.1 and Fig. A.16 (page 37).

## Comment 4: Scope of the inference-energy values

> The inference energy consumption is based on hardware measurement reports. The authors need to clarify whether this includes the energy consumed by memory access or only the core computation. In microcontroller deployments, flash access may dominate energy consumption.

**Response:** We corrected the terminology throughout. On-device kernel time was measured with the STM32 DWT cycle counter. Energy was not measured independently for each model. The reported value is now defined as the proxy `E_est = 0.5 W * measured inference time`, where 0.5 W is an assumed representative average power for the complete development board. The timed kernel interval includes the instructions and memory accesses executed by the LSTM and MLP, but the constant-power calculation cannot separate CPU computation, flash access, SRAM access, UART, regulator losses, or other board consumers. It therefore cannot establish whether flash access dominates actual consumption. Tables, abstract, utility score, Discussion, and Conclusion now consistently call this value an estimate or timing-derived proxy.

**Changes in the manuscript:** Abstract (page 1, lines 18--22); measured kernel interval (pages 21--22, lines 470--476); energy-proxy definition and limitations (page 22, lines 477--494); utility definition (pages 31--32, lines 638--645); Discussion (page 34, lines 698--706); Conclusion (pages 35--36, lines 746--755). The revised KPI table captions and headings appear on pages 21--22.

## Comment 5: Temporal dependence and very long sequences

> The streaming replay test used the complete test set and did not consider the temporal dependence of predictions. The authors need to evaluate whether the compressed model maintains stability on very long sequences, as quantization errors may accumulate.

**Response:** We added a temporal analysis of the saved full-stream predictions. The naturally ordered replay is divided into ten consecutive equal-count segments without resetting recurrent states. We report target MAE and P95 error in each segment and the mean absolute deviation of each compressed output from Base. Quantized-to-Base SOC deviation changes from 0.332 percentage points in the first tenth to 0.271 percentage points in the final tenth, so the evaluated output does not exhibit quantization-specific monotonic drift. SOH deviations fluctuate with ageing phase and likewise show no monotonic quantization accumulation. We limit this conclusion to the recorded replay and do not claim stability for arbitrary sequence lengths.

**Changes in the manuscript:** host-side temporal-analysis method (page 20, lines 431--442); Temporal Stability and Filter Interaction results (pages 25--26, lines 536--549); bounded conclusion (page 36, lines 755--757); Appendix A.2 and Fig. A.17 (page 38).

## Comment 6: SOH filter design, cutoff frequency, delay, and compression

> The SOH estimator operates on the filtered signal. The authors need to record the filter design and cutoff frequency and analyze how compression interacts with filtering to affect the overall estimation delay and accuracy.

**Response:** We reconstructed and documented the complete two-stage causal pipeline. Every raw model trajectory is first aligned to the initial SOH label. Stage 1 applies a symmetric step limiter and an EMA with `alpha_1 = 0.02`. Stage 2 consumes the stage-1 output, applies an EMA with `alpha_2 = 1e-6`, and limits downward changes to `2e-8` per sample. At 1 Hz, the EMA-only equivalent values are: stage 1 time constant 49.5 s, cutoff 3.22 mHz, and 90% response time 114 s; stage 2 time constant 11.57 days, cutoff `1.59e-7 Hz`, and 90% response time 26.65 days. The nonlinear limiters have no single cutoff frequency.

We also added a controlled local C-model re-execution for Base, Pruned, and Quantized, showing raw, stage-1, and final sequential outputs and MAE. The results demonstrate that filtering changes both absolute error and model ranking, while the strong final stage introduces substantial delay. Because the local Windows trajectory is not numerically identical to the archived Linux output over the complete recurrent sequence, it is used only for the filter-interaction analysis and does not replace the main benchmark values.

**Changes in the manuscript:** new SOH prediction post-processing subsection (pages 12--13, lines 269--288); revised SOH Results text (page 25, lines 527--535); compression/filter comparison (pages 26--27, lines 550--560); Discussion (page 33, lines 680--686); bounded conclusion (page 36, lines 755--757); Appendix A.3 and Fig. A.18 (page 39). The revised SOH dashboard caption appears on page 26.

## Comment 7: Utility weights and ranking sensitivity

> The comparison between pruning and quantization variants used the normalized utility score U. The authors need to clarify the weight factors used in this score and conduct a sensitivity analysis to show how the ranking changes with the variation of application priorities.

**Response:** We generalised the utility equation to four explicit non-negative weights for accuracy, flash, RAM, and estimated energy, constrained to sum to one. The originally reported score uses equal weights of 0.25. We evaluated all 1,771 combinations on a 0.05 grid and added focused sweeps in which one objective receives 25--85% weight and the remainder is divided equally. Pruned is top-ranked for 94.64% of SOC combinations and 53.36% of SOH combinations; Base wins 0% and 28.85%, and Quantized wins 5.36% and 17.79%, respectively. Thus, SOC Pruned is robust across most tested priorities, while the SOH ranking changes when accuracy or flash dominates. We also clarify that the energy objective is a timing-derived proxy and is not statistically independent of inference time under the fixed-power assumption.

**Changes in the manuscript:** revised utility equation, explicit equal weights, and sensitivity results (pages 31--32, lines 632--655); interpretation in the Discussion (page 34, lines 707--714); Appendix A.4 and Fig. A.19 (page 40).

## Comment 8: Robustness to embedded-system faults

> The current limitations mentioned the lack of fault injection research. The authors need to assess the robustness of the compressed model to common embedded system faults, which may affect the correctness of inference.

**Response:** We added a deliberately limited software-level sensitivity analysis using saved outputs. Random missing reported outputs are handled by hold-last. In a second experiment, 20,000 trials per task and model flip one random bit in an FP32 estimator output. A finite physical-range check followed by hold-last rejects invalid outputs. Arbitrary-bit corruption produces errors above 10 percentage points in approximately 33--34% of trials. The guard lowers P95 error but leaves approximately 72--73 percentage points for SOC and 94 percentage points for SOH because plausible in-range corruption remains undetected. This shows that a range check is useful but insufficient.

The revised manuscript explicitly states that this is not internal embedded fault injection. It does not corrupt weights, activations, recurrent states, inputs, communication packets, or MCU memory and does not support a hardware-safety claim. Those tests remain future work.

**Changes in the manuscript:** Limited Software-Level Fault Sensitivity method (page 21, lines 450--463); Limited Output-Fault Sensitivity results (pages 32--33, lines 656--669); expanded limitations (page 35, lines 734--740); future-work boundary (page 36, lines 765--768); Appendix A.5 and Fig. A.20 (page 41).

## Files added or updated

- Updated manuscript: `review_1/Embedded_Ai_Manuscript.tex`
- Compiled manuscript: `review_1/review_build/Embedded_Ai_Manuscript.pdf`
- Updated line-numbered anonymous manuscript: `review_1/Embedded_Ai_Manuscript_Anonymized.tex`
- Compiled line-numbered anonymous manuscript: `review_1/Embedded_Ai_Manuscript_Anonymized.pdf`
- Figure generator: `review_1/review_analysis/tools/generate_reviewer1_figures.ps1`
- Added appendix figures: `review_1/figures/Review_1_Additional/rev_1_*.png` through `rev_5_*.png`
- Analysis data and provenance: `review_1/review_analysis/results/` and `review_1/review_analysis/results/analysis_provenance.json`
