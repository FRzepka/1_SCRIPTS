# Review 1 work list

Decision letter deadline stated in the email: **2 August 2026**. Confirm the date
shown in Editorial Manager because the same email also contains a generic 45-day
statement.

## Highest priority before editing claims

1. **Correctly document the verified two-stage SOH-filter pipeline.** The saved
   `C_Base`, `C_Pruned`, and `C_Quant` trajectories were first calibrated to the
   initial label and filtered with `alpha=0.02` plus a symmetric limiter using
   `min(1e-4 * |previous|, 1e-5)`. The final manuscript figure then applies a second
   EMA with `alpha=1e-6` and a downward-only limit of `2e-8` per sample. Both stages
   were therefore used sequentially. The manuscript currently describes only the
   second stage and must be corrected without changing the reported main results.
2. **Correct the energy terminology and units.** The available values equal measured
   inference time multiplied by an assumed constant power of 0.5 W. They do not
   separate core computation, flash access, SRAM access, or UART power. Call them
   estimates, remove the duplicate contradictory paragraph, and verify uJ versus mJ
   in text, tables, captions, abstract, and conclusion.
3. **Remove unsupported causal claims.** The lower pruned SOC MAE is an observation
   from one trained model/split, not proof of regularization. Use the new weight
   analysis as descriptive evidence only and explicitly state that seed robustness
   was not tested.
4. **State the exact quantization scope.** Only recurrent weights are INT8. Biases,
   activations, recurrent states, and the MLP remain FP32. Quantized weights are
   multiplied by FP32 scales inside the current kernel, which explains added work.
5. **Fix derivative language.** The manuscript says `centred finite differences`,
   but the audited feature scripts use timestamp-aware backward differences:
   `np.diff(signal) / max(np.diff(time), 1e-6)`. These derivative features were
   prepared before UART replay and were not computed or stress-tested on the MCU.
   Correct the method description and do not claim an embedded robustness test that
   was not run.

## Analyses completed locally

- Full-sequence stability in ten consecutive windows for SOC and SOH, including
  MAE, P95, bias, cumulative MAE, and compressed-to-Base deviation.
- SOH-filter accuracy, theoretical time constants, and raw-to-filter trajectory
  comparisons for locally re-executed Base, Pruned, and Quantized C models.
- Utility sensitivity for all four-objective 5% weight combinations.
- Recurrent-weight distributions and L2 unit-saliency plots.
- INT8 reconstruction error and explicit FP32/INT8 storage accounting.
- Static kernel operation counts related to the saved inference times.
- Limited missing-output analysis and a stateful transient input-buffer bit-flip
  test using the six exported C models.

The existing filter comparison under `review_analysis/results/filter` treats
`alpha=0.02` and `alpha=1e-6` primarily as alternatives applied to locally generated
raw outputs. It does not yet reproduce the exact archived two-stage processing chain
used for the final manuscript figure. Do not present it as the final filter evidence
until the sequential analysis described below has been generated.

The 39 candidate figures are in `figures/Review_1_Additional`. Raw values are under
`review_analysis/results`, and the response status for every comment is in
`REVIEWER_RESPONSE_MATRIX.csv`.

## Manuscript work completed in the review copy

- [x] Shorten and restructure the abstract; define every acronym on first use.
- [x] Remove acronyms from title and keywords. Ensure Editorial Manager entries match.
- [x] State one concise objective and explicit contributions at the end of Introduction.
- [x] Strengthen novelty against current embedded SOC/SOH, pruning, and quantization
  literature. Verify every suggested reference before adding it.
- [x] Add a chemistry-scope paragraph: results are demonstrated only for LFP and cannot
  be numerically generalized to NMC or LCO without validation.
- [x] Add the L2-pruning rationale and distinguish one-shot structured pruning plus
  fine-tuning from pruning-aware or iterative training.
- [x] Replace the incorrect statement about successive pruning rounds. The implemented
  method performs one 30% structured pruning operation followed by short fine-tuning.
- [x] Add utility-weight definitions and the new sensitivity result.
- [x] Add a precise FP32 activation/state and INT8 recurrent-weight implementation note.
- [x] Add limitations for one chemistry, one split/seed, one STM32 family, estimated
  energy, no QAT, no broad hardware fault-injection campaign, and no operation-level
  profiling.
- [x] Quantify the main results in the conclusion and add concrete future work.
- [x] Break up long paragraphs, proofread the whole manuscript, and improve unclear
  figures. Submit the revision in single-column format with all source files.

## Work deliberately not performed

- Quantization-aware training or new pruning-aware training.
- New HPC training, random-seed study, cross-validation, or architecture sweep.
- New STM32/F4 runs, power-rail measurements, cycle-level profiling, or memory-bus
  instrumentation.
- Validation on NMC/LCO cells.
- Broad hardware fault injection into model-state, activation, weight memory, physical
  sensors, communication links, or MCU memory. The completed input-buffer test covers
  only a transient software upset before inference.

These omissions are possible to disclose and discuss, but the strongest reviewer
requests may not be fully satisfied by discussion alone. In particular, QAT, seed
robustness, energy-component profiling, and broad hardware fault injection remain
acceptance risks and should be answered transparently rather than implied as done.

## Binding decisions from author discussion (14 July 2026)

These notes are the agreed basis for the manuscript revision and reviewer response.
Do not silently replace them with stronger claims unless new evidence is found.

### A. SOH filtering and Reviewer 1.6

**Verified provenance**

- `run_benchmark_soh.py` first calibrates every model output to the first SOH label.
- It then applies an EMA with `alpha=0.02` and a symmetric step limiter with
  `rel_cap=1e-4` and `abs_cap=1e-5`; the effective limit is the smaller of the two.
- The resulting saved arrays `C_Base`, `C_Pruned`, and `C_Quant` are therefore
  already filtered.
- `simulate_stm32_filter.py` loads these arrays and applies a second online filter:
  EMA `alpha=1e-6` followed by a downward-only rate limit of `2e-8` per sample.
- `figures/Combined_Results/Figure_11_SOH_Streaming_Dashboard.png` is the output of
  this second script. The reported final SOH figure therefore uses both stages in
  sequence.

**Interpretation agreed with the author**

- The first, weaker stage follows the original model-prediction dynamics more
  closely.
- The very strong final smoothing yields a smaller global MAE against the slowly
  changing SOH labels, which motivated its use in the final result.
- A smaller MAE does not by itself establish a better online dynamic response. The
  strong filter can suppress short-term variation while introducing substantial
  response delay.
- Use precise terms: distinguish the unfiltered model prediction, the SOH target or
  reference label, the stage-1 output, and the final two-stage output.

**Required appendix addition; no HPC or STM32 rerun needed**

- Compare unfiltered output, stage 1 (`alpha=0.02` plus symmetric limiter), and the
  final sequential stage 1 + stage 2 (`alpha=1e-6` plus downward limiter).
- Show Base, Pruned, and Quantized variants under identical processing.
- Report MAE as well as a response-delay or settling-time measure so that smoothing
  quality is not judged from MAE alone.
- State the sampling interval used for converting EMA alpha to an equivalent time
  constant or cutoff frequency. A cutoff frequency is meaningful only together with
  the sampling interval.
- The nonlinear rate limiters have no single cutoff frequency. Report the equivalent
  EMA cutoff/time constant separately and describe the limiter in the time domain.

**Minimal main-text changes**

- Keep the established main result figure and numerical results.
- Correct the Methods description so that both sequential stages are disclosed.
- Add a short Results/Discussion sentence about the accuracy-delay trade-off and
  refer to the appendix comparison.
- Do not describe the final curve as the result of only `alpha=1e-6` applied directly
  to the raw network output.

### B. Energy estimate and Reviewer 1.4 / Reviewer 3.6

**What was actually done**

- On-device inference time was measured with the STM32 DWT cycle counter.
- Energy was not measured separately for each model with a power analyser.
- A representative average Nucleo development-board power of `0.5 W` was assumed.
- The reported value is `E_est = 0.5 W * measured inference time`.
- Example: `1.40 ms * 0.5 W = 0.70 mJ = 700 uJ`.

**Required terminology and scope**

- Call the values an **estimated energy-per-inference proxy under a constant 0.5 W
  board-power assumption**, not direct measured inference energy.
- Under this constant-power assumption, percentage energy changes are mathematically
  identical to percentage inference-time changes. Energy is not an independent
  hardware measurement.
- The estimate cannot separate CPU computation, flash access, SRAM access, UART,
  regulator losses, or other development-board consumers and cannot establish
  whether flash access dominates.
- The paper's intended scope is compression, deployment feasibility, and potential
  efficiency benefits. Detailed board-level energy profiling belongs to the newer,
  more detailed paper and was not performed for this revision.
- Tone down `measured energy` claims in the abstract, Methods, utility definition,
  Results, Discussion, and Conclusion. Use `estimated` consistently.
- Remove the duplicated KPI paragraph. Use one unit consistently; the existing table
  values are correctly expressed in mJ.

**Reviewer-response position**

- Explain the DWT timing measurement and constant-power calculation transparently.
- Explicitly acknowledge that memory-access energy was not isolated.
- Do not claim a new power-rail experiment or component-level profiling.

### C. Centered finite differences and Reviewer 3.2

- The manuscript wording `centered finite differences` is inconsistent with the
  audited feature code and should be corrected rather than defended.
- The relevant scripts calculate `dt[k] = max(t[k]-t[k-1], 1e-6)` and then use
  `(x[k]-x[k-1])/dt[k]` for voltage and current. This is a timestamp-aware backward
  difference, not a centered difference.
- In the STM32 benchmark, the resulting prepared feature vectors were transmitted
  via UART; derivative computation itself was not evaluated as an on-device module.
- Clarify this boundary in Methods so the reviewer does not infer that derivative
  handling was tested on the MCU under jitter or packet loss.
- Do not add another robustness experiment to this paper. Variable sampling,
  missing data, sensor corruption, and causal derivative handling are a separate
  research topic covered by the author's robustness work.
- Add a concise limitation and, only if bibliographically available and genuinely
  applicable, cite the separate robustness paper. Do not claim results from an
  unpublished or unavailable source.
- A possible deployment note may mention timestamp-aware backward differences and
  causal filtering as future implementation options, but these were not validated
  here.

### Reviewer 3 figure candidates generated on 14 July 2026

- `Figure_21_Model_Complexity_Scaling.png` addresses Reviewer 3.1 with analytical MAC
  scaling over hidden size. The highlighted points identify the implemented
  64-to-45 SOC and 128-to-90 SOH architecture changes and report only their
  analytical MAC reductions. Measured flash, runtime, and energy-proxy values remain
  separate and the figure must not be presented as an STM32F4 or cross-hardware
  benchmark.
- `rev_7_derivative_deployment_boundary.png` was generated as an internal reasoning
  aid for Reviewer 3.2, but the final decision is not to include it in the manuscript
  or appendix. The point is clearer as a corrected equation plus bounded prose.
- `Figure_22_Pruning_Criterion_Scope.png` addresses Reviewer 3.3 and supports Reviewer
  3.5. It combines cross-task L2 rankings with a method-property rationale. It is
  not an experimental comparison against gradient- or activation-based pruning.
- `Figure_23_Mixed_Precision_Quantization.png` addresses Reviewer 3.4 by showing the
  exact INT8/FP32 boundary and persistent storage composition. Transient activation
  buffers remain FP32 but were not separately profiled.
- `Figure_24_Quantized_Runtime_Accounting.png` addresses Reviewer 3.6 by placing static
  MAC and FP32 scale-multiplication counts beside measured kernel times. It supports
  a code-based explanation but is not cycle-level or memory-bandwidth profiling.
- No new figure was generated for a random-seed or cross-validation study in
  Reviewer 3.5 because those experiments were not performed. The response must
  remove the causal regularisation claim and state the single-checkpoint boundary.

### Reviewer 3 discussion decisions recorded on 14 July 2026

**Reviewer 3.1 and final Figure A.21 interpretation**

- The final `Figure_21_Model_Complexity_Scaling.png` layout uses three wide panels stacked
  vertically. Panels (a) and (b) retain the accepted model-size and evaluated 30%
  operating-point views. Panel (c) adds the general pruning-fraction limit.
- Panel (a) reports architecture-level MAC counts only. For input size `D=6`, hidden
  size `H`, and MLP width `M`, the count is
  `N(H) = 4H(D+H) + HM + M`. The plotted SOC and SOH differences arise from their
  MLP widths (`M=64` and `M=128`), not from measured hardware behaviour.
- For a general pruning fraction `p`, the retained hidden size is `H_p = (1-p)H` and
  the analytical MAC reduction is
  `[4(2p-p^2)H^2 + p(4D+M)H] / [4H^2 + (4D+M)H + M]`. Its large-`H` limit is
  `2p-p^2` because the quadratic recurrent term dominates.
- Panel (b) applies the evaluated `p=0.30` operating point. Its retained quadratic
  share approaches `(1-p)^2 = 0.49`, so the MAC reduction approaches 51%.
- Panel (c) shows the general asymptotic curve and highlights `p=0.30`. The reference
  values are 19%, 36%, 51%, 64%, and 75% maximum analytical reduction for pruning
  fractions of 10%, 20%, 30%, 40%, and 50%, respectively.
- The implemented architecture points are shown separately: SOC `64 -> 45` gives
  an analytical 45.1% MAC reduction and SOH `128 -> 90` gives 45.7%.
- The 51% dashed line is an asymptotic architecture-level limit for this calculation,
  not a measured flash, latency, or energy reduction and not a guaranteed saving on
  every MCU. The earlier mixed flash/runtime/energy points and 40% reference line
  were removed because they combined quantities with different meanings.
- The manuscript explains the linear and fixed terms that keep finite models below
  the limit for the selected `p`. It also states that `p=0.30` is a pragmatic moderate
  operating point rather than the optimum of an exhaustive pruning-rate sweep.
- A rate sweep would require a separately pruned and fine-tuned model at every
  candidate fraction. This remains a pruning-focused follow-up and must not be implied
  by the analytical curve, which predicts architecture-level MAC scaling but not MAE.
- The manuscript must also state that transfer to another STM32
  family was not measured; only the operation-count trend is architecture based.

**Reviewer 3.2 and `rev_7` scope**

- Final decision: retain `rev_7` only as an internal work product. The reviewer did
  not request a figure, and the diagram gives disproportionate visual weight to two
  derivative features while its untested deployment branch could be mistaken for an
  implemented method.
- Answer Reviewer 3.2 in text: correct `centred` to timestamp-aware backward
  differences, show the equation, state that features were prepared host-side, and
  explain that the benchmark deliberately isolates LSTM/MLP inference. Reported
  kernel timing therefore excludes sensor acquisition and feature extraction.
- Add a short implementation note that a causal version stores only the previous
  accepted voltage/current samples and timestamp and has constant work per sample.
  Add a limitation that jitter, missing samples, derivative noise, filtering, and
  derivative limiting were not evaluated in this compression benchmark.
- The internal process diagram was initially generated because the reviewer asks
  both how derivatives were formed in the evaluated benchmark and how non-ideal
  sampling could be handled in an embedded implementation.
- Panel (a) is the audited path: timestamp-aware backward differences and robust
  scaling were calculated before the prepared six-feature vectors were replayed by
  UART. Derivative computation itself was neither timed nor stress-tested on the MCU.
- Panel (b) is only a proposed causal deployment path: validate timestamps/samples,
  skip and flag invalid derivatives, resynchronise state, and optionally apply causal
  filtering or derivative limiting. None of these safeguards was evaluated in the
  current compression benchmark.
- Current colour intent is visual rather than quantitative: blue marks the main data
  path and data/state objects, green marks validation, preprocessing, or mitigation
  operations, and red highlights the derivative issue, invalid-sample branch, and
  scope warnings. The colours do not encode measured quality or implementation
  status. No colour key is needed while the figure remains internal; add one or
  simplify the palette only if the figure is reused elsewhere.
- The phrase `resynchronise state` in the invalid-sample branch is ambiguous. It is
  intended to mean resynchronising the accepted-sample/timestamp buffer used for the
  derivative, not resetting or correcting the LSTM hidden or cell state. Rename this
  box before any future reuse of the diagram.
- Do not add a new derivative-robustness experiment to this compression paper. The
  author's separate robustness work covers that broader topic; the response should
  correct the factual method description and define the scope honestly.

**Remaining Reviewer 3 boundaries**

- Figure A.22 may justify why gate-group L2 ranking is simple, structured, and directly
  compatible with dense dimension reduction, but it must not be described as an
  experimental comparison with gradient- or activation-based criteria.
- Figure A.23 must call the implementation weight-only mixed precision: recurrent weights
  are INT8 with FP32 scales, while biases, states, activations, and MLP computation
  remain FP32. Activation quantisation was not evaluated.
- Figure A.24 may connect the quantised runtime increase to explicit per-weight FP32
  scale multiplications and dequantisation work. It cannot provide cycle attribution,
  memory-bandwidth use, or operator-level hardware profiling that was not measured.
- For Reviewer 3.5, report the lower SOC Pruned MAE only as a checkpoint- and split-
  specific observation. Do not retain the unsupported causal `regularisation effect`
  explanation and do not imply repeated-seed or cross-validation evidence.

**Reviewer 3.4 and Figure A.23 verification**

- The evaluated quantized deployment is now verified from the final model packages,
  STM32 source selected by `main.c`, and the linked map sections for both tasks.
- SOC calls `lstm_model_predict_int8()` from `lstm_model_int8.c`. That source uses
  `model_weights_lstm_int8_manual.h` and `mlp_weights_fp32.h`.
- SOH calls `lstm_model_soh_int8_forward()` from `lstm_model_soh_int8.c`. That source
  uses `model_weights_lstm_int8_manual_soh.h` and `mlp_weights_fp32_soh.h`.
- In both final deployments only `W_ih` and `W_hh` are stored as INT8. The per-row
  scales, fused LSTM bias, input and recurrent states, gate activations, MLP weights
  and activations, and estimator output remain FP32.
- The MLP declarations are `static const float` in both exported headers. The linker
  maps independently confirm FP32 sizes for `MLP0_WEIGHT` and `MLP1_WEIGHT`.
- The quantization scripts explicitly implement recurrent-weight-only quantization
  and export the MLP unchanged. They do not document a separate empirical comparison
  that proves why MLP quantization was rejected. The defensible rationale is scope:
  the experiment targets the recurrent matrices, which account for about 80% of the
  raw Base model constants, while retaining a simple, directly validated FP32 MLP
  path. Do not present this rationale as a measured superiority of FP32 MLP weights.
- Activation quantization and a fully integer kernel were not evaluated. State this
  directly. Retaining FP32 activations avoids introducing another quantization source,
  but its accuracy benefit was not isolated experimentally.
- The STM32 quantized-project README files are stale and describe an older opposite
  hybrid variant (FP32 LSTM and INT8 MLP). Several unused alternative headers also
  remain in the project folders. Do not infer the evaluated method from those files;
  use the called kernel and linked map sections.
- The analytical stored-model-constant totals are 87.5/37.0 KiB for SOC Base/Quantized
  and 335.0/138.0 KiB for SOH Base/Quantized. These values are computed from recurrent
  weights, LSTM bias/scales, and MLP parameters only.
- Hidden/cell states and transient activations are runtime RAM and must not be stacked
  into those model-constant bars. This was corrected in Figure A.23.
- The analytical totals are not the complete linked-firmware Flash values. The measured
  map-derived firmware values remain 105.32/52.48 KiB for SOC and 335.00/138.00 KiB
  for SOH. The matching SOH pairs are coincidental at the displayed precision.
- Figure A.23 now uses `model constants` rather than `persistent storage`. Panel (a)
  contains only the verified precision path; the complete explanatory block below
  that path was removed at the author's request. Limitations and interpretation
  belong in the future caption or manuscript text, not inside the graphic.

**Reviewer 3.6 and Figure A.24 assessment**

- Reviewer 3.6 asks why the quantized implementation is slower despite its smaller
  weight storage. In particular, the reviewer suggests operation-level cycle counts,
  memory-bandwidth analysis, and a discussion of implementation optimizations.
- Figure A.24 combines two different evidence types that must remain explicitly
  distinguished: static operation counts derived from the architecture and C source,
  and already available total STM32 kernel-time measurements.
- The Base and Quantized models have the same topology and therefore the same model
  MAC counts: 22,080 for SOC and 85,120 for SOH. Pruning reduces these counts to
  12,124 for SOC and 46,208 for SOH by reducing the hidden dimension.
- The current quantized C kernels execute the per-row FP32 scale multiplication inside
  the inner recurrent-weight loops. The static accounting therefore contains one
  additional FP32 scale multiplication for each recurrent weight term: 17,920 for
  SOC and 68,608 for SOH. These are source-level counts, not measured instruction or
  processor-cycle counts.
- The measured total kernel times are 1.40/0.80/6.99 ms for SOC and
  22.73/12.72/29.21 ms for SOH (Base/Pruned/Quantized). Relative to Base, the
  Quantized kernel is approximately 5.0 times as slow for SOC and 1.29 times as slow
  for SOH. The different penalties already show that static multiplication counts
  alone do not establish a complete causal explanation.
- The timing values are averages over 10,000 streaming inferences on the STM32H753ZI;
  the on-device inference interval was measured with the DWT cycle counter. They are
  not host-side UART latency values.
- All inspected Base and Quantized STM32 projects were compiled as Debug builds with
  `-O0`, hard-float ABI, and the Cortex-M7 FPv5 FPU. Consequently, the written
  per-weight cast and scale expression remains a particularly expensive code path,
  while no optimized integer dot-product kernel is used.
- For pruning, static and measured ratios agree closely because the arithmetic type
  and kernel structure remain FP32: SOC retains 54.9% of the counted MACs and 57.2%
  of the measured time; SOH retains 54.3% and 56.0%, respectively. This supports the
  MAC-based size analysis within the same implementation family.
- The same proportionality must not be expected for Quantized because it changes the
  operation types and memory footprint. Counting a MAC and a separate FP32 multiply
  as one operation each omits the INT8-to-FP32 conversion, loop/index overhead,
  loads, stores, nonlinearities, and cache/Flash behaviour. Static operation ratios
  predict about 1.81 times Base for both tasks, whereas the measured ratios are about
  4.99 for SOC and 1.29 for SOH.
- A plausible explanation for the smaller relative SOH penalty is that the much
  larger SOH model is more strongly affected by FP32 weight traffic; the smaller INT8
  recurrent matrices can offset part of the conversion and scaling cost. For the
  smaller SOC model, the additional cast/scale and unoptimized inner-loop overhead
  dominate more visibly. The Quantized variants also use much larger transient stack
  arrays than Base. These are code- and footprint-based hypotheses, not measured
  cycle attribution or memory-bandwidth evidence.
- The figure can support the limited conclusion that the weight-only mixed-precision
  kernel saves Flash but does not provide a native integer compute path: INT8 weights
  are converted into FP32 arithmetic and repeatedly combined with FP32 scales. This
  code structure is consistent with additional runtime overhead.
- Figure A.24 does not measure cycles for individual operations, cast costs, Flash/cache
  traffic, memory bandwidth, or compiler effects. Do not describe it as hardware
  profiling and do not attribute a measured percentage of the delay to any one cause.
- Potential optimizations belong in the discussion: move each row scale outside the
  innermost accumulation where mathematically valid, use optimized dot-product or
  integer kernels, fuse scale/bias work, and evaluate a complete activation-quantized
  path. These were not evaluated and must be presented as future work.
- Current recommendation: Figure A.24 is optional rather than essential. The measured
  times already appear in the main results, while the new contribution is the static
  source-level accounting. A compact appendix table plus precise manuscript text may
  answer the point more clearly than a four-panel figure. Keep the figure only if its
  caption states the static/measured distinction and the absence of cycle-level and
  memory-bandwidth profiling.

### Reviewer 2 completion status (15 July 2026)

- [x] No new calculation, model run, hardware measurement, or figure is required
  for the five Reviewer 2 comments; they concern positioning, objective, literature
  synthesis, conclusion, and language.
- [x] No new Zotero entry is required. The revised positioning uses the already
  verified recent references in the bibliography, including 2024 and 2025 reviews,
  embedded SOC/SOH studies, and a current pruning/quantization benchmark.
- [x] The Introduction was rewritten from the current SOC/SOH context through the
  embedded resource problem to one explicit research gap.
- [x] One concise objective and three contributions now close the Introduction.
- [x] Related Work now compares the study directly with prior embedded SOC/SOH and
  compression studies and states that the novelty is the paired, auditable STM32
  benchmark, not a new recurrent cell or compression algorithm.
- [x] The Conclusion and Outlook now quantifies Flash, RAM, inference-time, and MAE
  changes for SOC and SOH, distinguishes estimated energy from direct power
  measurement, interprets the operating points, and prioritises future work.
- [x] A manuscript-wide language and logic pass corrected the Abstract,
  Introduction, Related Work, Methods wording, Results interpretation, Discussion,
  and Conclusion.
- [x] `RESPONSE_TO_REVIEWER_2.md` documents every response with references to the
  final 61-page line-numbered anonymous revision.
- [x] The page and line references in `RESPONSE_TO_REVIEWER_1.md` and
  `RESPONSE_TO_REVIEWER_3.md` were refreshed after the Reviewer 2 integration.

### Reviewer 3 status after the completed technical discussion (15 July 2026)

- [x] All six Reviewer 3 comments have now been discussed and checked against the
  available model code, STM32 projects, benchmark results, and generated analyses.
- [x] No additional experiment requiring the HPC or a new STM32 run is currently
  planned for Reviewer 3.
- [x] Manuscript integration, appendix placement, and the point-by-point response
  are complete. `RESPONSE_TO_REVIEWER_3.md` refers to the 61-page line-numbered
  anonymous revision compiled on 20 July 2026.

**3.1 Model complexity and hardware transferability**

- [x] `Figure_21_Model_Complexity_Scaling.png` now includes the general pruning-fraction
  limit `2p-p^2`, example limits from 10% to 50%, and the highlighted evaluated
  `p=0.30` point.
- [x] Add text deriving the general analytical hidden-size scaling, the implemented
  `64 -> 45` and `128 -> 90` points, and the 51% asymptotic MAC-reduction limit for
  the evaluated 30% fraction.
- [x] State that 30% is a single moderate operating point and is not claimed to be
  the optimum of an exhaustive pruning-rate sweep.
- [x] State that the approximately 40% measured Flash/time/energy-proxy findings are
  specific to the tested checkpoints, STM32H753ZI, firmware, and build. Do not claim
  transfer to STM32F4 or another controller family; only the architecture-level MAC
  trend transfers without a new hardware benchmark.

**3.2 Derivative implementation and non-ideal sampling**

- [x] Final decision is text only; `rev_7` remains internal and is not inserted.
- [x] Correct the original statement `centred finite differences`. The audited
  implementation uses timestamp-aware backward differences.
- [x] Add the causal equation, state that the six prepared features were computed
  host-side and replayed over UART, and clarify that reported kernel timing isolates
  LSTM/MLP inference and excludes acquisition and feature extraction.
- [x] Add a bounded limitation: variable sampling, missing samples, derivative noise,
  and causal filtering/limiting were not evaluated in this compression benchmark.
  Refer to the separate robustness work only if it is citable at submission time.

**3.3 L2 pruning criterion**

- [x] `Figure_22_Pruning_Criterion_Scope.png` and the verified saliency definition are
  accepted.
- [x] The current method already explains the structured L2 criterion, but add an
  explicit sentence contrasting it with gradient- and activation-based sensitivity:
  those alternatives require backward-pass or representative-activation statistics,
  whereas the selected gate-group L2 score is data independent and directly maps to
  removal of complete dense hidden channels.
- [x] State explicitly that no experimental comparison of pruning criteria was run;
  the rationale is methodological and implementation based, not evidence that L2 is
  universally superior.

**3.4 Weight-only mixed-precision quantization**

- [x] `Figure_23_Mixed_Precision_Quantization.png` was corrected against the called SOC
  and SOH kernels, exported headers, and linker maps. Its panel (a) contains only the
  precision path.
- [x] Keep the verified scope in the method: only recurrent `W_ih` and `W_hh` are
  INT8; row scales, bias, states, activations, MLP, and output remain FP32.
- [x] Add the missing explicit consequence: retaining FP32 activations/states avoids
  introducing another quantization source but preserves FP32 RAM and arithmetic cost,
  so the implementation cannot obtain the latency and RAM benefits of a full-integer
  path. The accuracy benefit of this choice was not isolated experimentally.
- [x] State directly that activation quantization and fully integer deployment were
  not evaluated. Explain recurrent-weight-only scope via the approximately 80% share
  of recurrent matrices in the Base model constants, without claiming that retaining
  the FP32 MLP was experimentally superior.

**3.5 Lower SOC Pruned MAE**

- [x] This point is already handled correctly in the current manuscript: the causal
  regularisation claim has been removed, the 0.35-percentage-point difference is
  checkpoint- and split-specific, and the absence of repeated seeds, cross-validation,
  and matched unpruned fine-tuning is stated.
- [x] No new diagram or experiment is required. Keep the existing saliency/weight
  diagnostics descriptive and do not use them as proof of improved generalisation.

**3.6 Quantized runtime increase**

- [x] `Figure_24_Quantized_Runtime_Accounting.png`, its MAC formula, additional scale
  counts, measured DWT times, and the static-versus-measured interpretation have been
  discussed and accepted as useful evidence.
- [x] Expand the current discussion beyond `consistent with the observed timing
  penalty`: report that static ratios are about 1.81 for both tasks, whereas measured
  Quantized/Base ratios are 4.99 for SOC and 1.29 for SOH. Explain why static counts
  are predictive within the FP32 pruning family but not across a changed kernel.
- [x] Record that the inspected firmware used `-O0`, hard-float FP32, per-weight
  INT8-to-FP32 casts, and per-weight row scaling without an optimized integer
  dot-product path. Present reduced SOH weight traffic as a plausible offsetting
  factor, not as measured memory-bandwidth attribution.
- [x] Add future optimization options requested by the reviewer: hoist row scales
  outside inner accumulations where valid, fuse scale/bias work, use optimized DSP or
  integer dot-product kernels, and evaluate activation quantization/full-integer
  execution. These are proposals, not evaluated results.
- [x] State explicitly that no operation-level cycle breakdown or memory-bandwidth
  profile was available. The existing DWT measurements are total kernel times averaged
  over 10,000 inferences.

### Reviewer 4 completion status (15 July 2026)

- [x] The Abstract was reduced to approximately 169 words and now directly identifies
  the engineering application of artificial intelligence, the implemented artificial
  intelligence, the paired compression benchmark, the principal numerical results,
  and the evidence boundary. All abstract acronyms are defined on first use.
- [x] The keyword list was reduced to six written-out terms. The title and keywords
  contain no unexplained acronym.
- [x] The two reviewer-suggested papers were imported from the author's Zotero export
  `bib/paper2_socsoh_ref_fr.bib` into the isolated review bibliography as
  `wang_improved_2024` (DOI `10.1016/j.est.2023.110222`) and
  `wang_improved_2023` (DOI `10.1016/j.energy.2023.128677`). No bibliographic data
  were invented or inferred.
- [x] Related Work now explains what the two hybrid estimators contribute and why
  their accuracy evidence complements, but does not replace, the paired embedded
  pruning/quantization benchmark.
- [x] The method logic now explicitly separates the common Base equations, structured
  dimension reduction, and representation-only quantization.
- [x] Mathematical detail was added for dense pruning submatrix construction,
  code-exact symmetric per-row quantization over `-127...127`, the reconstruction
  error bound, the mixed-precision gate calculation, and MAE/RMSE/P95 definitions.
- [x] The quantization description was checked against the SOC and SOH export code.
  The previous schematic/caption range `-128...127` was corrected to the implemented
  symmetric set `-127...127` with `-128` deliberately unused.
- [x] Long Dataset, Training, Benchmark, Results, Discussion, and Limitations passages
  were split or tightened, and a manuscript-wide grammar and logic pass was performed.
- [x] Figures 2, 6, and 7 were regenerated at high resolution with larger text and
  code-consistent content. Figure 2 shows the real DoE factor levels, Figure 6 shows
  dense hidden-channel removal and the `64 -> 45` / `128 -> 90` dimensions, and
  Figure 7 shows the verified mixed-precision path.
- [x] The reproducible generator is
  `review_analysis/tools/generate_reviewer4_figures.ps1`.
- [x] Author and anonymous review manuscripts compile without undefined citations,
  undefined references, or LaTeX errors. The final PDFs contain 37 and 61 pages,
  respectively.
- [x] `RESPONSE_TO_REVIEWER_4.md` records the point-by-point response using the final
  61-page line-numbered anonymous PDF. Responses 1--3 were refreshed after this
  integration.

### D. Verified pruning implementation

**Method**

- The final models use one-shot, magnitude-based structured pruning of complete LSTM
  hidden units, not iterative pruning and not pruning-aware training.
- Unit saliency is the sum of L2 norms of input and recurrent weight rows over all
  four LSTM gates.
- The lowest-saliency units are removed jointly from all gate matrices. Associated
  recurrent columns and the input columns of the downstream MLP are sliced to form
  a physically smaller dense network.
- SOC changes from 64 to 45 hidden units; SOH changes from 128 to 90 hidden units,
  corresponding to approximately 30% removal.
- The reduced model is then briefly fine-tuned with AdamW and MSE loss.

**Parameters that are actually evidenced**

- SOH is fully documented by `archive/run_pruning.sh` and the valid final manifest:
  30% pruning, 3 fine-tuning epochs, learning rate `1e-6`, training batch size 512,
  and saved validation metrics after fine-tuning.
- The final SOC folder and 64-to-45 checkpoint match the simple one-shot pruning
  script and the project README states that the existing model received short
  fine-tuning.
- The final SOC `manifest.json` is truncated to 108 bytes. The exact SOC epoch count
  and learning rate are therefore not recoverable from that file. Script defaults
  are 5 epochs and `5e-5`, while another archived usage example shows 3 epochs.
- Until an original SOC command or log is found, do not state an exact SOC epoch
  count or learning rate as fact. Write `brief post-pruning fine-tuning` and disclose
  the reproducibility limitation if necessary.

**Required corrections and claims**

- Replace the current sentence `After each pruning step ... before the next pruning
  round` because there was one pruning step and no next round.
- Weight-distribution and L2-saliency diagrams provide descriptive support for the
  pruning criterion, but they do not prove that pruning caused regularization.
- Describe the lower SOC MAE as an observed result for the evaluated checkpoint and
  split. No new random-seed, cross-validation, pruning-aware, or QAT experiments are
  planned.

### E. Revision boundary

- No new HPC training or architecture sweep.
- No new STM32 or STM32F4 deployment.
- No new power measurement or operation-level hardware profiling.
- No quantization-aware or pruning-aware training.
- No new derivative-robustness campaign in this optimisation paper.
- No new NMC/LCO measurements.
- No broad hardware fault-injection campaign. The software input-buffer test below is
  the only newly evaluated fault class.
- The revision will rely on verified artefacts, additional offline analyses, clearer
  mathematical descriptions, appendices, carefully bounded claims, and explicit
  limitations.

### F. Reviewer 1.8 transient input-buffer fault test (20 July 2026)

**Final decision**

- [x] The earlier post-inference output-register bit-flip analysis was rejected because
  independent corruption of reported scalar outputs does not exercise LSTM state or
  reveal recovery behaviour. It must not appear in the manuscript or reviewer response.
- [x] Replace it with the minimal stateful test in
  `DL_Models/LFP_LSTM_MLP/5_benchmark/PC/input_bitflip_recovery/`.
- [x] Keep the separate missing-output and hold-last results in concise prose because
  they cover a second common failure mode without another figure.

**Verified protocol**

- The actual exported SOC and SOH Base, Pruned, and Quantized C implementations process
  8000 finite C07 samples in their natural 1-Hz order.
- Thirty independent events per task and model are distributed after a minimum of 2048
  preceding clean samples. Ten events affect voltage, ten current, and ten temperature.
- Every event starts from the corresponding clean recurrent state. Bit 22, the most
  significant FP32 mantissa bit, is flipped in one input-buffer value before inference
  for exactly one sample. The following 60 inputs are clean and the LSTM state is not
  reset.
- The events are independent of one another so that overlapping faults do not obscure
  the response to one transient upset. Within each event, propagation through the
  recurrent state is retained for the full 60-s recovery horizon.
- Recovery is the first time after the peak at which disturbed-clean absolute deviation
  remains below `max(10% of peak, 0.0001 pp)` for five consecutive samples.

**Metrics and interpretation**

- `d(t) = 100 * |fault output - clean output|` measures only the effect of the injected
  fault. Peak, P95 peak, residual, and recovery time characterise transient magnitude
  and persistence.
- Target accuracy is reported with a 61-s window MAE for the clean and disturbed paths.
  `Delta MAE_61 = MAE_fault - MAE_clean` isolates the change caused by the fault.
  Normalised errors are multiplied by 100 and reported in percentage points.
- MAE does not replace disturbed-clean deviation. A fault can accidentally move an
  imperfect prediction closer to the target, and averaging over 61 samples can hide one
  short peak. The paper therefore reports both target-MAE change and transient response.
- Negative `Delta MAE_61` values mean accidental movement toward the target and must not
  be interpreted as improved fault robustness.

**Results used in the manuscript**

- SOC median peak deviations are 3.29--3.82 pp and P95 peaks are 16.18--23.81 pp. Every
  SOC event recovers within 60 s, with median recovery times of 10.0--12.5 s.
- SOC median `Delta MAE_61` is 0.046--0.079 pp and P95 is 1.42--1.59 pp. Quantized is
  close to Base, while Pruned has the largest upper-tail peak and P95 MAE increase.
- SOH median peak deviations are 0.83--1.59 pp and P95 peaks are 3.37--5.85 pp. Recovery
  within 60 s is 93.3% for Base, 90.0% for Pruned, and 86.7% for Quantized.
- SOH median `Delta MAE_61` is approximately zero and P95 is 0.053--0.122 pp. Quantized
  has the largest P95 peak and the largest share without confirmed 60-s recovery.
- No non-finite output occurs in the 180 evaluated task/model events. The result is not
  a hardware-safety claim and does not cover physical sensors, communication, weight or
  recurrent-state memory, activations, or MCU memory.

**Files and manuscript integration**

- [x] Reproducible runner, wrapper, README, and figure generator are stored in the new
  benchmark folder. Progress, elapsed time, and ETA are printed during execution.
- [x] Final numerical run:
  `results/BITFLIP_INPUT_20260720_124813/`.
- [x] Figure A.20 is stored as `Figure_20_Limited_Fault_Sensitivity.png`. Panels (a) and
  (b) show representative SOC and SOH propagation, panel (c) shows median and P95 peak,
  and panel (d) shows recovery within 60 s.
- [x] The manuscript now defines the stateful protocol, disturbed-clean response,
  `Delta MAE_61`, and recovery criterion. Model-specific MAE changes are placed in an
  Appendix A.5 table.
- [x] Reviewer 1 response 8 is rewritten around the input-buffer experiment and the
  remaining hardware fault-injection boundary.
