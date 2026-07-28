# Response to Reviewer #1

Manuscript: *Embedded Artificial Intelligence in Battery Management Systems: Pruning and Quantization for Efficient State-of-Charge and State-of-Health Estimation*

The page and line references below refer to the final 62-page line-numbered anonymous revision compiled on 22 July 2026. Figure captions and other float contents are identified by figure number and page because the `lineno` package does not assign reliable line numbers inside floats. No new STM32 measurements, battery experiments, HPC training runs, random-seed study, or validation on another chemistry were performed. Added analyses use the existing model artefacts, exported C implementations, test data, and recorded benchmark results. The revised manuscript states these boundaries explicitly. All references below were refreshed after integration of the Reviewer 2, Reviewer 3, and Reviewer 4 revisions.

## General response

We thank the reviewer for the detailed comments. We revised the method descriptions, removed unsupported causal claims, documented the exact scope of the energy estimate and mixed-precision quantization, and added five supplementary analyses. These analyses cover pruning diagnostics, temporal stability, SOH filtering, utility-weight sensitivity, and transient software input-buffer faults. We have carefully distinguished observations from the evaluated models and replay from conclusions that would require new chemistry, training, power-measurement, or broad hardware fault-injection campaigns.

## Comment 1: Cell-chemistry generalisation

> This study only evaluated the estimator on the LFP DoE cycle aging dataset containing a single battery chemical system. The authors need to discuss whether the observed trade-off between accuracy and compression can be generalized to other chemical systems with different degradation characteristics, such as NMC or LCO.

**Response:** We agree that the original manuscript did not delimit chemistry transferability clearly enough. The revised Discussion now separates architecture-level storage effects from model accuracy. For a fixed architecture, removing the same number of hidden channels or storing the same recurrent matrices as INT8 produces comparable parameter-count changes independent of chemistry. However, the accuracy-compression trade-off depends on chemistry, operating range, degradation trajectory, and training data. Our numerical MAE and preferred compression level are therefore demonstrated only for the evaluated JGC LFP cells at 25 degrees Celsius and cannot be transferred to NMC or LCO without retraining and validation. We added relevant NMC SOC/SOH literature to show feasibility of compact data-driven models while explicitly stating that those studies do not validate our numerical trade-off.

**Changes in the manuscript:** general discussion of transfer across cell chemistries (pages 39--40, lines 838--848); platform and scope discussion (page 40, lines 849--876); Conclusion and Outlook (pages 41--42, lines 877--917).

## Comment 2: Quantization-aware and pruning-aware training

> The pruning and quantization experiments were conducted on the fully-precision floating-point trained model. The authors need to investigate whether quantization-aware training or pruning-aware training, compared to post-compression training, can further improve the trade-off between accuracy and efficiency.

**Response:** We clarified that the present study evaluates two deployable transformations of trained FP32 models: one-shot structured pruning followed by brief post-pruning fine-tuning, and post-training recurrent-weight quantization. Quantization-aware training, pruning-aware training, and an iterative pruning schedule were not performed because they require a new training campaign. The revised paper no longer suggests that the tested variants represent the best attainable jointly trained compression result. It explains why training-aware compression may recover accuracy and identifies this comparison as future work.

**Changes in the manuscript:** structured magnitude pruning and the explicit post-pruning fine-tuning boundary (pages 17--20, lines 387--440); post-training mixed-precision quantization details (pages 20--22, lines 441--480); Discussion of the compression boundary and training-aware comparison (page 40, lines 849--856); future work (page 42, lines 908--917).

## Comment 3: Evidence for the claimed pruning regularisation effect

> The SOC estimator showed a slight improvement in MAE after pruning, which is attributed to the regularization effect. The authors need to provide evidence of weight distribution or unit activation statistics to support this regularization explanation.

**Response:** We agree that the original causal interpretation was too strong. We removed the claim that pruning acted as a regulariser. The revised text reports the lower SOC MAE as an observation for one trained model and test split. We added an L2 unit-saliency ranking and recurrent-weight distribution comparison. The 19 removed SOC units occupy the lowest saliency ranks by construction, which verifies implementation of the stated criterion. The figure is explicitly described as diagnostic evidence, not proof of improved generalisation. We also state that repeated seeds, cross-validation, and a matched fine-tuning ablation were not available.

**Changes in the manuscript:** pruning criterion and diagnostic scope (pages 18--19, lines 394--434); Error Distribution Analysis and removal of the causal regularisation claim (page 31, lines 643--660); Discussion (page 37, lines 761--779); Appendix A.1 (page 43, line 922) and Fig. A.16 (page 44).

## Comment 4: Scope of the inference-energy values

> The inference energy consumption is based on hardware measurement reports. The authors need to clarify whether this includes the energy consumed by memory access or only the core computation. In microcontroller deployments, flash access may dominate energy consumption.

**Response:** We corrected the terminology throughout. On-device kernel time was measured with the STM32 DWT cycle counter. Energy was not measured independently for each model. The reported value is now defined as the proxy `E_est = 0.5 W * measured inference time`, where 0.5 W is an assumed representative average power for the complete development board. The timed kernel interval includes the instructions and memory accesses executed by the LSTM and MLP, but the constant-power calculation cannot separate CPU computation, flash access, SRAM access, UART, regulator losses, or other board consumers. It therefore cannot establish whether flash access dominates actual consumption. The Methods, KPI tables, utility score, Discussion, and Conclusion consistently call this value an estimate or timing-derived proxy. The revised Abstract focuses on measured flash and inference-time outcomes rather than presenting the proxy as independent energy evidence.

**Changes in the manuscript:** measured resource results in the Abstract (page 1, lines 17--23); measured kernel interval and build settings (page 25, lines 538--558); energy-proxy definition and interpretation (pages 25--27, lines 559--589); utility definition and sensitivity results (pages 34--35, lines 711--732); Discussion (pages 38--39, lines 814--837); Conclusion (page 41, lines 877--894). The revised KPI table captions and headings appear in Tables 1 and 2 on pages 26--27.

## Comment 5: Temporal dependence and very long sequences

> The streaming replay test used the complete test set and did not consider the temporal dependence of predictions. The authors need to evaluate whether the compressed model maintains stability on very long sequences, as quantization errors may accumulate.

**Response:** We added a temporal analysis of the full-stream output trajectories. The naturally ordered replay is divided into ten consecutive equal-count segments without resetting recurrent states. We report target MAE and P95 error in each segment and the mean absolute deviation of each compressed output from Base. Quantized-to-Base SOC deviation changes from 0.332 percentage points in the first tenth to 0.271 percentage points in the final tenth, so the evaluated output does not exhibit quantization-specific monotonic drift. SOH deviations fluctuate with ageing phase and likewise show no monotonic quantization accumulation. We limit this conclusion to the recorded replay and do not claim stability for arbitrary sequence lengths.

**Changes in the manuscript:** host-side temporal-analysis method (page 23, lines 492--503); Temporal Stability and Filter Interaction results (pages 29--30, lines 618--642); conclusion (pages 41--42, lines 895--907); Appendix A.2 and Fig. A.17 (page 45).

## Comment 6: SOH filter design, cutoff frequency, delay, and compression

> The SOH estimator operates on the filtered signal. The authors need to record the filter design and cutoff frequency and analyze how compression interacts with filtering to affect the overall estimation delay and accuracy.

**Response:** We documented the complete two-stage causal pipeline. Every raw model trajectory is first aligned to the initial SOH label. Stage 1 applies a symmetric step limiter and an EMA with `alpha_1 = 0.02`. Stage 2 consumes the stage-1 output, applies an EMA with `alpha_2 = 1e-6`, and limits downward changes to `2e-8` per sample. At 1 Hz, the EMA-only equivalent values are: stage 1 time constant 49.5 s, cutoff 3.22 mHz, and 90% response time 114 s; stage 2 time constant 11.57 days, cutoff `1.59e-7 Hz`, and 90% response time 26.65 days. The nonlinear limiters have no single cutoff frequency.

We also evaluated the exported Base, Pruned, and Quantized C models on a common ordered input sequence and report the raw, stage-1, and final sequential outputs and MAE. The results demonstrate that filtering changes both absolute error and model ranking, while the strong final stage introduces substantial delay. This controlled comparison is used specifically to examine the interaction between compression and filtering and does not replace the primary benchmark values.

**Changes in the manuscript:** SOH prediction post-processing subsection (pages 14--15, lines 307--326); SOH Results text (pages 27--28, lines 608--617); compression/filter comparison (pages 29--30, lines 618--642); Discussion (page 37, lines 773--779); conclusion (pages 41--42, lines 895--907); Appendix A.3 and Fig. A.18 (page 46). The SOH dashboard caption appears in Fig. 11 on page 30.

## Comment 7: Utility weights and ranking sensitivity

> The comparison between pruning and quantization variants used the normalized utility score U. The authors need to clarify the weight factors used in this score and conduct a sensitivity analysis to show how the ranking changes with the variation of application priorities.

**Response:** We generalised the utility equation to four explicit non-negative weights for accuracy, flash, RAM, and estimated energy, constrained to sum to one. The originally reported score uses equal weights of 0.25. We evaluated all 1,771 combinations on a 0.05 grid and added focused sweeps in which one objective receives 25--85% weight and the remainder is divided equally. Pruned is top-ranked for 94.64% of SOC combinations and 53.36% of SOH combinations; Base wins 0% and 28.85%, and Quantized wins 5.36% and 17.79%, respectively. Thus, SOC Pruned is robust across most tested priorities, while the SOH ranking changes when accuracy or flash dominates. We also clarify that the energy objective is a timing-derived proxy and is not statistically independent of inference time under the fixed-power assumption.

**Changes in the manuscript:** revised utility equation, explicit equal weights, and sensitivity results (pages 34--35, lines 711--732); interpretation of the purpose and decision value of the score in the Discussion (page 39, lines 823--837); Appendix A.4 and Fig. A.19 (page 47).

## Comment 8: Robustness to embedded-system faults

> The current limitations mentioned the lack of fault injection research. The authors need to assess the robustness of the compressed model to common embedded system faults, which may affect the correctness of inference.

**Response:** We added two deliberately limited software-level fault analyses. Randomly unavailable output reports are handled with hold-last while the underlying inference trajectory continues without interruption. More importantly, a stateful input-buffer experiment now evaluates whether a transient fault propagates through the recurrent estimator. The exported C implementations of Base, Pruned, and Quantized process a continuous 8000-sample segment from the evaluation trajectory of one representative LFP cell at 1 Hz. For each task and model, 30 independent events are distributed over the sequence after at least 2048 clean samples. Ten events affect voltage, ten current, and ten temperature. FP32 bits are indexed from zero at the least significant bit. Bit 22 is therefore the highest-order bit of the 23-bit fraction field. Flipping this bit produces a pronounced but finite input change without changing the sign or exponent. At each event, it is flipped in one input value for exactly one sample. The clean and disturbed branches start from the same recurrent state, after which both receive the same 60 unmodified inputs without resetting the LSTM state.

We report three complementary properties. The disturbed-clean peak quantifies the immediate fault effect, a sustained threshold determines recovery time, and the change in target MAE quantifies the resulting accuracy loss. The evaluation window contains the corrupted sample and 60 subsequent clean samples. It therefore contains 61 samples while covering a 60-s recovery horizon at 1 Hz. This separation is necessary because a transient can accidentally move an imperfect prediction closer to the target and because a window MAE alone can hide one short peak. All SOC variants recover in all 30 events within 60 s. Their median peak deviations are 3.29--3.82 percentage points, P95 peak deviations are 16.18--23.81 percentage points, and P95 increases in window MAE are 1.42--1.59 percentage points. For SOH, median peaks are 0.83--1.59 percentage points and P95 peaks are 3.37--5.85 percentage points. The shares recovered within 60 s are 93.3% for Base, 90.0% for Pruned, and 86.7% for Quantized. SOH P95 increases in window MAE remain 0.053--0.122 percentage points. No non-finite output occurs in the evaluated events.

This experiment directly tests short-term recurrent propagation after a transient software input-buffer upset, but it remains a bounded sensitivity analysis. It does not inject faults into physical sensors, communication links, weights, activations, recurrent-state memory, or MCU memory and does not support a hardware-safety claim. These broader fault classes remain future work.

**Changes in the manuscript:** Limited Software-Level Fault Sensitivity method and MAE definition (page 24, lines 511--537); Limited Software Fault Sensitivity results (pages 36--37, lines 735--760); interpretation in the Discussion (page 40, lines 865--876) and Conclusion (pages 41--42, lines 895--907); Appendix A.5 and Fig. A.20 (page 48), with the model-specific target-MAE values in Table A.4 (page 49).
