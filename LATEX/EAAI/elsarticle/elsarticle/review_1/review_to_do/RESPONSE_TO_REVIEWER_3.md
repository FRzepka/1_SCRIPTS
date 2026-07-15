# Response to Reviewer #3

Manuscript: *Embedded Artificial Intelligence in Battery Management Systems: Pruning and Quantization for Efficient State-of-Charge and State-of-Health Estimation*

Revision working copy: `review_1/Embedded_Ai_Manuscript.tex`

Compiled revision: `review_1/review_build/Embedded_Ai_Manuscript.pdf`

Line-numbered anonymous revision: `review_1/review_build/Embedded_Ai_Manuscript_Anonymized.pdf`

The page and line references below refer to the final 57-page line-numbered anonymous revision compiled on 15 July 2026. Figure captions and other float contents are identified by figure number and page because the `lineno` package does not assign reliable line numbers inside floats. No new STM32 measurements, model-training runs, random-seed study, cross-validation, or benchmark on another controller family were performed. The additional quantitative analyses use the evaluated architectures, inspected C firmware, archived model artefacts, and previously recorded DWT kernel times. The manuscript now states these evidence boundaries explicitly. All references below were refreshed after integration of the Reviewer 2 and Reviewer 4 revisions.

## General response

We thank the reviewer for identifying several places where the implementation and the scope of the conclusions required clearer documentation. We revised the abstract, feature-engineering description, pruning and quantization methods, STM32 benchmark protocol, Discussion, limitations, and Outlook. Four supplementary figures were added for analytical model-complexity scaling, the scope of the L2 pruning criterion, the verified mixed-precision boundary, and static quantized-runtime accounting. Where the requested experiment was not available, we removed unsupported causal or transferability claims and state precisely what can and cannot be concluded from the evaluated checkpoints and firmware.

## Comment 1: Model complexity and hardware transferability

> The abstract reports that pruning reduces flash footprint and energy per inference by approximately 40%, yet it does not clarify whether these savings remain consistent across different model complexities (e.g., varying hidden dimensions) or across other STM32 families (e.g., F4 series). Given the diversity of embedded AI deployments, it would strengthen the paper to briefly discuss the model sensitivity and hardware transferability of these energy savings, or at least delineate the scope within which the current conclusions hold, thereby improving the generalizability of the findings for practitioners working on different BMS platforms.

**Response:** We agree and have separated architecture-level scaling from hardware-specific measurements. For input dimension `D`, hidden size `H`, and MLP width `M`, the revised manuscript gives the analytical count

`N_MAC(H) = 4H(D + H) + HM + M`.

This gives 22,080 to 12,124 MACs for the implemented SOC change from 64 to 45 hidden units and 85,120 to 46,208 MACs for the SOH change from 128 to 90 hidden units. For a nominal 30% hidden-size reduction, the quadratic recurrent term approaches a 51% MAC reduction as `H` grows, while the linear input, MLP, and output terms keep finite models below that limit. Figure A.21 visualises this analytical dependence.

The approximately 40% savings in the Abstract are not presented as universal hardware factors. They apply to the evaluated checkpoints, model dimensions, STM32H753ZI firmware, GCC `-O0` build, and constant-power timing proxy. Clock rate, FPU/DSP support, compiler optimisation, cache, and memory organisation can change the measured flash and timing ratios. We therefore make no numerical transfer claim for STM32F4 or another controller family without a new hardware benchmark.

**Changes in the manuscript:** bounded Abstract statement (page 1, lines 17--24); analytical MAC model and implemented hidden-size reductions (pages 15--16, lines 328--348); inspected firmware and build settings (pages 23--24, lines 515--535); measured/static comparison and implementation limits (pages 36--37, lines 743--775); explicit hardware-transfer limitation (page 38, lines 812--818); bounded Conclusion and Outlook (pages 39--40, lines 834--864); Appendix A.6 and Fig. A.21 (page 46).

## Comment 2: Derivative implementation and non-ideal sampling

> The paper employs centered finite differences for voltage and current derivatives but does not address efficient implementation on embedded platforms, nor does it discuss numerical stability under variable sampling rates or data loss. In real-world BMS applications, sensor noise and communication delays are inevitable. It would be valuable to elaborate on the robustness strategies employed when computing derivatives on the STM32, or to suggest alternative filtering approaches that could maintain numerical stability under non-ideal conditions, thereby ensuring reliable real-time operation.

**Response:** An audit of the preprocessing code showed that the original term "centered finite differences" was incorrect. The evaluated features use timestamp-aware backward differences,

`Delta t_k = max(t_k - t_(k-1), epsilon_t)`,

`dU/dt = (U_k - U_(k-1)) / Delta t_k`, and analogously for current.

This formulation is causal, stores only the previous accepted sample and timestamp, and has constant work per update. However, the derivative and cumulative channels in the reported STM32 benchmark were prepared on the host and transmitted as part of the complete six-feature UART vector. The benchmark deliberately isolates the LSTM/MLP inference kernel; its DWT times do not include acquisition or feature construction. We now document this boundary instead of implying that centered differences were calculated on the microcontroller.

Because variable sampling, missing measurements, derivative noise, and causal derivative filtering or limiting were not evaluated as embedded modules in this compression study, we do not claim measured robustness for those conditions. The limitations now state that a production feature front end would require a separate validation of timestamp checks, missing-sample handling, and suitable causal filtering or limiting.

**Changes in the manuscript:** corrected timestamp-aware backward-difference equations, causal implementation cost, and host/STM32 boundary (pages 10--11, lines 246--261); DWT measurement interval and scaler placement (pages 23--24, lines 515--535); explicit non-ideal-sampling limitation (page 38, lines 819--825). No additional figure was included because the implementation boundary is expressed more precisely by equations and text than by a process diagram.

## Comment 3: Rationale for the L2 pruning score

> The pruning score is defined using L2 norms, but the authors do not justify why this criterion was chosen over alternatives such as gradient-based sensitivity or activation distribution, nor do they provide experimental comparisons. Given that pruning strategies directly impact the accuracy-efficiency trade-off, it would enhance the paper to include comparative experiments or at least provide a theoretical rationale for the chosen method, thereby helping readers assess its suitability for LSTM-based battery estimators and facilitating informed design decisions in similar embedded applications.

**Response:** We expanded the rationale for the gate-group L2 saliency score. It combines the four gate rows belonging to one hidden unit, is data independent once the checkpoint has been saved, and maps directly to removal of complete dense rows, recurrent columns, and the associated MLP input. It therefore yields a smaller dense LSTM kernel rather than an irregular sparse mask. Gradient-based sensitivity would require training data and a backward pass, while activation-based ranking would require a representative calibration stream and an aggregation rule across time.

We did not run an experimental comparison of pruning criteria. The revised manuscript states this explicitly and does not claim universal superiority of L2 ranking. Figure A.22 provides a method-property rationale and the observed cross-task L2 rankings, while the existing diagnostics in Fig. A.16 show that the intended low-saliency channels were removed from the evaluated checkpoint.

**Changes in the manuscript:** expanded criterion definition, implementation rationale, comparison boundary, and removal of any universal-superiority implication (pages 17--19, lines 384--425); checkpoint-specific diagnostic interpretation (pages 29--30, lines 620--637); bounded Discussion (pages 35--36, lines 727--735); Appendix A.7 and Fig. A.22 (page 47). The supporting checkpoint diagnostics remain in Appendix A.1 (page 40, line 869) and Fig. A.16 (page 41).

## Comment 4: Weight-only mixed-precision quantization

> The paper applies per-row symmetric quantization exclusively to weights, while biases and activations remain in floating-point precision. Although this simplifies implementation, it may forgo additional opportunities for reducing memory and computational overhead. It would be insightful to discuss whether activation quantization was evaluated, or to explain the specific impact of retaining floating-point activations on both accuracy and hardware efficiency. Such clarification would help readers understand the design trade-offs and assess whether alternative quantization schemes could yield better resource-accuracy balances for BMS deployments.

**Response:** We verified the deployed SOC and SOH source paths against the exported headers and now describe the implementation consistently as weight-only mixed precision. Only the recurrent `W_ih` and `W_hh` matrices are stored as INT8 with FP32 row scales. Biases, hidden and cell states, gate activations, the MLP weights and computation, and the output remain FP32. The recurrent matrices constitute approximately 80% of the raw Base model constants, which motivated targeting them first.

Retaining FP32 activations and states avoids introducing another quantization source into the recurrent update, but it preserves FP32 runtime-state storage and arithmetic. The evaluated implementation therefore cannot provide the RAM and latency benefits of a full-integer path. Activation quantization was not evaluated, and the accuracy contribution of retaining FP32 activations was not isolated experimentally. We identify activation/state quantization and an optimised integer kernel as future work rather than implying that the current implementation exhausts the available efficiency gains.

**Changes in the manuscript:** exact precision boundary and its consequences (pages 20--21, lines 432--462); resource interpretation (pages 32--33, lines 675--687); quantized-kernel interpretation and possible optimisations (pages 36--37, lines 752--775); future-work scope (pages 39--40, lines 855--860); Appendix A.8 and Fig. A.23 (page 48).

## Comment 5: Interpretation of the lower pruned SOC MAE

> The SOC error distribution reveals that the pruned model achieves slightly better MAE than the baseline, which the authors attribute to a regularization effect. However, this observation may be sensitive to training data splits or model initialization. It would be prudent to investigate whether this advantage persists across different random seeds or cross-validation settings, or to provide a theoretical explanation for how pruning could enhance generalization. Such analysis would prevent overinterpretation of what might be a dataset-specific phenomenon and strengthen the claims about pruning's benefits.

**Response:** We agree that the original causal interpretation was not supported by the available evidence. The manuscript no longer attributes the lower SOC Pruned MAE to a regularisation effect. It now reports a 0.35-percentage-point improvement only as an observation for the evaluated checkpoint and test split. The L2 saliency and recurrent-weight diagnostics verify which channels were selected, but they cannot demonstrate that pruning caused improved generalisation.

No repeated-seed, cross-validation, or matched unpruned fine-tuning experiment was available, so we do not claim persistence across model initialisations or data partitions. This limitation is stated in the Results, Discussion, limitations, Conclusion, and future-work paragraph.

**Changes in the manuscript:** non-causal Results interpretation (pages 29--30, lines 620--637); checkpoint- and split-specific Discussion (pages 35--36, lines 727--735); repeated-seed limitation and bounded conclusion (pages 38--39, lines 812--854); future random-seed validation (pages 39--40, lines 855--860); supporting diagnostic Fig. A.16 (page 41) and criterion-scope Fig. A.22 (page 47).

## Comment 6: Root causes of quantized inference time

> The increased inference time of quantized models is noted but not deeply analyzed regarding its root causes, such as dequantization overhead, changes in memory access patterns, or cycle consumption from floating-point conversions. Providing performance profiling data-including cycle counts for key operations and memory bandwidth utilization-would help identify specific bottlenecks. Additionally, discussing potential optimizations like operator fusion or hardware acceleration would offer practical guidance for future work aiming to improve quantized LSTM deployment on resource-constrained microcontrollers.

**Response:** We expanded the implementation-level analysis while keeping static accounting separate from measured profiling. Base and Quantized have the same topology and therefore the same model MAC counts: 22,080 for SOC and 85,120 for SOH. In the inspected quantized C expressions, each recurrent INT8 weight is converted to FP32 and multiplied by its row scale inside the innermost accumulation. This adds 17,920 source-level FP32 scale multiplications for SOC and 68,608 for SOH. Counting one model MAC and one additional scale multiplication as one operation gives a static Quantized/Base ratio of about 1.81 for both tasks.

The measured DWT kernel-time ratios are 4.99 for SOC and 1.29 for SOH. Their disagreement with the identical static ratio shows why a simple operation count cannot explain cross-kernel timing. The count omits cast cost, loop and index overhead, loads/stores, nonlinear functions, cache/flash behaviour, and compiler effects. The evaluated firmware was built at `-O0` and does not use an optimised integer dot-product path. Reduced recurrent-weight traffic may offset more arithmetic overhead in the larger SOH model, but memory bandwidth was not measured, so this remains an implementation-based interpretation rather than a causal attribution.

We did not have operation-level DWT traces or memory-bandwidth measurements and therefore do not present the new analysis as cycle-level hardware profiling. The revised Discussion instead identifies concrete future optimisations: move shared row scales outside the innermost accumulation where mathematically valid, fuse scale and bias operations, use DSP-optimised or integer dot-product kernels, and quantize activations and recurrent states for a complete integer path. Figure A.24 places the static source-level counts beside the previously measured total kernel times and labels the evidence types explicitly.

**Changes in the manuscript:** measurement interval, scaler placement, compiler and FPU settings (pages 23--24, lines 515--535); mixed-precision KPI interpretation (pages 24--26, lines 536--566); latency interpretation (page 33, lines 681--687); static counts, measured ratios, omitted costs, and optimisation options (pages 36--37, lines 752--775); future-work paragraph (pages 39--40, lines 855--860); Appendix A.9 and Fig. A.24 (page 49).

## Files added or updated

- Updated manuscript: `review_1/Embedded_Ai_Manuscript.tex`
- Compiled manuscript: `review_1/review_build/Embedded_Ai_Manuscript.pdf`
- Updated line-numbered anonymous manuscript: `review_1/Embedded_Ai_Manuscript_Anonymized.tex`
- Compiled line-numbered anonymous manuscript: `review_1/review_build/Embedded_Ai_Manuscript_Anonymized.pdf`
- Added appendix figures: `review_1/figures/Review_1_Additional/rev_6_model_complexity_scaling.png`, `rev_8_pruning_criterion_scope.png`, `rev_9_mixed_precision_quantization.png`, and `rev_10_quantized_runtime_accounting.png`
- Figure generator and analysis provenance: `review_1/review_analysis/tools/` and `review_1/review_analysis/results/`
