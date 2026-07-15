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
- Limited missing-output and output-register bit-flip simulations.

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
  energy, no QAT, no internal fault injection, and no operation-level profiling.
- [x] Quantify the main results in the conclusion and add concrete future work.
- [x] Break up long paragraphs, proofread the whole manuscript, and improve unclear
  figures. Submit the revision in single-column format with all source files.

## Work deliberately not performed

- Quantization-aware training or new pruning-aware training.
- New HPC training, random-seed study, cross-validation, or architecture sweep.
- New STM32/F4 runs, power-rail measurements, cycle-level profiling, or memory-bus
  instrumentation.
- Validation on NMC/LCO cells.
- Internal model-state, activation, weight-memory, sensor, or MCU fault injection.

These omissions are possible to disclose and discuss, but the strongest reviewer
requests may not be fully satisfied by discussion alone. In particular, QAT, seed
robustness, energy-component profiling, and broad internal fault injection remain
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

- `rev_6_model_complexity_scaling.png` addresses Reviewer 3.1 with analytical MAC
  scaling over hidden size. The highlighted points identify the implemented
  64-to-45 SOC and 128-to-90 SOH architecture changes and report only their
  analytical MAC reductions. Measured flash, runtime, and energy-proxy values remain
  separate and the figure must not be presented as an STM32F4 or cross-hardware
  benchmark.
- `rev_7_derivative_deployment_boundary.png` was generated as an internal reasoning
  aid for Reviewer 3.2, but the final decision is not to include it in the manuscript
  or appendix. The point is clearer as a corrected equation plus bounded prose.
- `rev_8_pruning_criterion_scope.png` addresses Reviewer 3.3 and supports Reviewer
  3.5. It combines cross-task L2 rankings with a method-property rationale. It is
  not an experimental comparison against gradient- or activation-based pruning.
- `rev_9_mixed_precision_quantization.png` addresses Reviewer 3.4 by showing the
  exact INT8/FP32 boundary and persistent storage composition. Transient activation
  buffers remain FP32 but were not separately profiled.
- `rev_10_quantized_runtime_accounting.png` addresses Reviewer 3.6 by placing static
  MAC and FP32 scale-multiplication counts beside measured kernel times. It supports
  a code-based explanation but is not cycle-level or memory-bandwidth profiling.
- No new figure was generated for a random-seed or cross-validation study in
  Reviewer 3.5 because those experiments were not performed. The response must
  remove the causal regularisation claim and state the single-checkpoint boundary.

### Reviewer 3 discussion decisions recorded on 14 July 2026

**Reviewer 3.1 and final `rev_6` interpretation**

- The final `rev_6_model_complexity_scaling.png` layout uses two wide panels stacked
  vertically. This version has been reviewed and accepted for later manuscript use.
- Panel (a) reports architecture-level MAC counts only. For input size `D=6`, hidden
  size `H`, and MLP width `M`, the count is
  `N(H) = 4H(D+H) + HM + M`. The plotted SOC and SOH differences arise from their
  MLP widths (`M=64` and `M=128`), not from measured hardware behaviour.
- Panel (b) applies an analytical 30% hidden-size reduction (`H' = 0.70H`). As the
  quadratic recurrent term dominates for large `H`, its retained share approaches
  `0.70^2 = 0.49`; the corresponding MAC reduction therefore approaches 51%.
- The implemented architecture points are shown separately: SOC `64 -> 45` gives
  an analytical 45.1% MAC reduction and SOH `128 -> 90` gives 45.7%.
- The 51% dashed line is an asymptotic architecture-level limit for this calculation,
  not a measured flash, latency, or energy reduction and not a guaranteed saving on
  every MCU. The earlier mixed flash/runtime/energy points and 40% reference line
  were removed because they combined quantities with different meanings.
- Later manuscript text must explain the linear and fixed terms that keep smaller
  models below the 51% limit. It must also state that transfer to another STM32
  family was not measured; only the operation-count trend is architecture based.

**Reviewer 3.2 and `rev_7` scope**

- Final decision: retain `rev_7` only as an internal work product. The reviewer did
  not request a figure, and the diagram gives disproportionate visual weight to two
  derivative features while its untested deployment branch could be mistaken for an
  implemented method.
- Answer Reviewer 3.2 in tÛ¿4¶‰žËkºwµçA­•É¹•°ÍÑÉÕÑÕÉ”É•µ…¥¸@ÌÈèM=É•Ñ…¥¹Ì€ÔÐ¸ä”½˜Ñ¡”½Õ¹Ñ•5Ì…¹€ÔÜ¸È”(€½˜Ñ¡”µ•…ÍÕÉ•Ñ¥µ”ìM= É•Ñ…¥¹Ì€ÔÐ¸Ì”…¹€ÔØ¸À”°É•ÍÁ•Ñ¥Ù•±ä¸Q¡¥ÌÍÕÁÁ½ÉÑÌÑ¡”(€5µ‰…Í•Í¥é”…¹…±åÍ¥ÌÝ¥Ñ¡¥¸Ñ¡”Í…µ”¥µÁ±•µ•¹Ñ…Ñ¥½¸™…µ¥±ä¸(´Q¡”Í…µ”ÁÉ½Á½ÉÑ¥½¹…±¥ÑäµÕÍÐ¹½Ð‰”•áÁ•Ñ•™½ÈEÕ…¹Ñ¥é•‰•…ÕÍ”¥Ð¡…¹•ÌÑ¡”(€½Á•É…Ñ¥½¸ÑåÁ•Ì…¹µ•µ½Éä™½½ÑÁÉ¥¹Ð¸½Õ¹Ñ¥¹œ„5…¹„Í•Á…É…Ñ”@ÌÈµÕ±Ñ¥Á±ä(€…Ì½¹”½Á•É…Ñ¥½¸•… ½µ¥ÑÌÑ¡”%9PàµÑ¼µ@ÌÈ½¹Ù•ÉÍ¥½¸°±½½À½¥¹‘•à½Ù•É¡•…°(€±½…‘Ì°ÍÑ½É•Ì°¹½¹±¥¹•…É¥Ñ¥•Ì°…¹…¡”½±…Í ‰•¡…Ù¥½ÕÈ¸MÑ…Ñ¥Œ½Á•É…Ñ¥½¸É…Ñ¥½Ì(€ÁÉ•‘¥Ð…‰½ÕÐ€Ä¸àÄÑ¥µ•Ì	…Í”™½È‰½Ñ Ñ…Í­Ì°Ý¡•É•…ÌÑ¡”µ•…ÍÕÉ•É…Ñ¥½Ì…É”…‰½ÕÐ(€€Ð¸ää™½ÈM=…¹€Ä¸Èä™½ÈM= ¸(´Á±…ÕÍ¥‰±”•áÁ±…¹…Ñ¥½¸™½ÈÑ¡”Íµ…±±•ÈÉ•±…Ñ¥Ù”M= Á•¹…±Ñä¥ÌÑ¡…ÐÑ¡”µÕ (€±…É•ÈM= µ½‘•°¥Ìµ½É”ÍÑÉ½¹±ä…™™•Ñ•‰ä@ÌÈÝ•¥¡ÐÑÉ…™™¥ŒìÑ¡”Íµ…±±•È%9Pà(€É•ÕÉÉ•¹Ðµ…ÑÉ¥•Ì…¸½™™Í•ÐÁ…ÉÐ½˜Ñ¡”½¹Ù•ÉÍ¥½¸…¹Í…±¥¹œ½ÍÐ¸½ÈÑ¡”(€Íµ…±±•ÈM=µ½‘•°°Ñ¡”…‘‘¥Ñ¥½¹…°…ÍÐ½Í…±”…¹Õ¹½ÁÑ¥µ¥é•¥¹¹•Èµ±½½À½Ù•É¡•…(€‘½µ¥¹…Ñ”µ½É”Ù¥Í¥‰±ä¸Q¡”EÕ…¹Ñ¥é•Ù…É¥…¹ÑÌ…±Í¼ÕÍ”µÕ ±…É•ÈÑÉ…¹Í¥•¹ÐÍÑ…¬(€…ÉÉ…åÌÑ¡…¸	…Í”¸Q¡•Í”…É”½‘”´…¹™½½ÑÁÉ¥¹Ðµ‰…Í•¡åÁ½Ñ¡•Í•Ì°¹½Ðµ•…ÍÕÉ•(€å±”…ÑÑÉ¥‰ÕÑ¥½¸½Èµ•µ½Éäµ‰…¹‘Ý¥‘Ñ •Ù¥‘•¹”¸(´Q¡”™¥ÕÉ”…¸ÍÕÁÁ½ÉÐÑ¡”±¥µ¥Ñ•½¹±ÕÍ¥½¸Ñ¡…ÐÑ¡”Ý•¥¡Ðµ½¹±äµ¥á•µÁÉ•¥Í¥½¸(€­•É¹•°Í…Ù•Ì±…Í ‰ÕÐ‘½•Ì¹½ÐÁÉ½Ù¥‘”„¹…Ñ¥Ù”¥¹Ñ••È½µÁÕÑ”Á…Ñ è%9PàÝ•¥¡ÑÌ(€…É”½¹Ù•ÉÑ•¥¹Ñ¼@ÌÈ…É¥Ñ¡µ•Ñ¥Œ…¹É•Á•…Ñ•‘±ä½µ‰¥¹•Ý¥Ñ @ÌÈÍ…±•Ì¸Q¡¥Ì(€½‘”ÍÑÉÕÑÕÉ”¥Ì½¹Í¥ÍÑ•¹ÐÝ¥Ñ …‘‘¥Ñ¥½¹…°ÉÕ¹Ñ¥µ”½Ù•É¡•…¸(´É•Ù|ÄÁ€‘½•Ì¹½Ðµ•…ÍÕÉ”å±•Ì™½È¥¹‘¥Ù¥‘Õ…°½Á•É…Ñ¥½¹Ì°…ÍÐ½ÍÑÌ°±…Í ½…¡”(€ÑÉ…™™¥Œ°µ•µ½Éä‰…¹‘Ý¥‘Ñ °½È½µÁ¥±•È•™™•ÑÌ¸¼¹½Ð‘•ÍÉ¥‰”¥Ð…Ì¡…É‘Ý…É”(€ÁÉ½™¥±¥¹œ…¹‘¼¹½Ð…ÑÑÉ¥‰ÕÑ”„µ•…ÍÕÉ•Á•É•¹Ñ…”½˜Ñ¡”‘•±…äÑ¼…¹ä½¹”…ÕÍ”¸(´A½Ñ•¹Ñ¥…°½ÁÑ¥µ¥é…Ñ¥½¹Ì‰•±½¹œ¥¸Ñ¡”‘¥ÍÕÍÍ¥½¸èµ½Ù”•… É½ÜÍ…±”½ÕÑÍ¥‘”Ñ¡”(€¥¹¹•Éµ½ÍÐ…ÕµÕ±…Ñ¥½¸Ý¡•É”µ…Ñ¡•µ…Ñ¥…±±äÙ…±¥°ÕÍ”½ÁÑ¥µ¥é•‘½ÐµÁÉ½‘ÕÐ½È(€¥¹Ñ••È­•É¹•±Ì°™ÕÍ”Í…±”½‰¥…ÌÝ½É¬°…¹•Ù…±Õ…Ñ”„½µÁ±•Ñ”…Ñ¥Ù…Ñ¥½¸µÅÕ…¹Ñ¥é•(€Á…Ñ ¸Q¡•Í”Ý•É”¹½Ð•Ù…±Õ…Ñ•…¹µÕÍÐ‰”ÁÉ•Í•¹Ñ•…Ì™ÕÑÕÉ”Ý½É¬¸(´ÕÉÉ•¹ÐÉ•½µµ•¹‘…Ñ¥½¸èÉ•Ù|ÄÁ€¥Ì½ÁÑ¥½¹…°É…Ñ¡•ÈÑ¡…¸•ÍÍ•¹Ñ¥…°¸Q¡”µ•…ÍÕÉ•(€Ñ¥µ•Ì…±É•…‘ä…ÁÁ•…È¥¸Ñ¡”µ…¥¸É•ÍÕ±ÑÌ°Ý¡¥±”Ñ¡”¹•Ü½¹ÑÉ¥‰ÕÑ¥½¸¥ÌÑ¡”ÍÑ…Ñ¥Œ(€Í½ÕÉ”µ±•Ù•°…½Õ¹Ñ¥¹œ¸½µÁ…Ð…ÁÁ•¹‘¥àÑ…‰±”Á±ÕÌÁÉ•¥Í”µ…¹ÕÍÉ¥ÁÐÑ•áÐµ…ä(€…¹ÍÝ•ÈÑ¡”Á½¥¹Ðµ½É”±•…É±äÑ¡…¸„™½ÕÈµÁ…¹•°™¥ÕÉ”¸-••ÀÑ¡”™¥ÕÉ”½¹±ä¥˜¥ÑÌ(€…ÁÑ¥½¸ÍÑ…Ñ•ÌÑ¡”ÍÑ…Ñ¥Œ½µ•…ÍÕÉ•‘¥ÍÑ¥¹Ñ¥½¸…¹Ñ¡”…‰Í•¹”½˜å±”µ±•Ù•°…¹(€µ•µ½Éäµ‰…¹‘Ý¥‘Ñ ÁÉ½™¥±¥¹œ¸((ŒŒŒI•Ù¥•Ý•È€È½µÁ±•Ñ¥½¸ÍÑ…ÑÕÌ€ ÄÔ)Õ±ä€ÈÀÈØ¤((´mát9¼¹•Ü…±Õ±…Ñ¥½¸°µ½‘•°ÉÕ¸°¡…É‘Ý…É”µ•…ÍÕÉ•µ•¹Ð°½È™¥ÕÉ”¥ÌÉ•ÅÕ¥É•(€™½ÈÑ¡”™¥Ù”I•Ù¥•Ý•È€È½µµ•¹ÑÌìÑ¡•ä½¹•É¸Á½Í¥Ñ¥½¹¥¹œ°½‰©•Ñ¥Ù”°±¥Ñ•É…ÑÕÉ”(€Íå¹Ñ¡•Í¥Ì°½¹±ÕÍ¥½¸°…¹±…¹Õ…”¸(´mát9¼¹•Üi½Ñ•É¼•¹ÑÉä¥ÌÉ•ÅÕ¥É•¸Q¡”É•Ù¥Í•Á½Í¥Ñ¥½¹¥¹œÕÍ•ÌÑ¡”…±É•…‘ä(€Ù•É¥™¥•É••¹ÐÉ•™•É•¹•Ì¥¸Ñ¡”‰¥‰±¥½É…Á¡ä°¥¹±Õ‘¥¹œ€ÈÀÈÐ…¹€ÈÀÈÔÉ•Ù¥•ÝÌ°(€•µ‰•‘‘•M=½M= ÍÑÕ‘¥•Ì°…¹„ÕÉÉ•¹ÐÁÉÕ¹¥¹œ½ÅÕ…¹Ñ¥é…Ñ¥½¸‰•¹¡µ…É¬¸(´mátQ¡”%¹ÑÉ½‘ÕÑ¥½¸Ý…ÌÉ•ÝÉ¥ÑÑ•¸™É½´Ñ¡”ÕÉÉ•¹ÐM=½M= ½¹Ñ•áÐÑ¡É½Õ Ñ¡”(€•µ‰•‘‘•É•Í½ÕÉ”ÁÉ½‰±•´Ñ¼½¹”•áÁ±¥¥ÐÉ•Í•…É …À¸(´mát=¹”½¹¥Í”½‰©•Ñ¥Ù”…¹Ñ¡É•”½¹ÑÉ¥‰ÕÑ¥½¹Ì¹½Ü±½Í”Ñ¡”%¹ÑÉ½‘ÕÑ¥½¸¸(´mátI•±…Ñ•]½É¬¹½Ü½µÁ…É•ÌÑ¡”ÍÑÕ‘ä‘¥É•Ñ±äÝ¥Ñ ÁÉ¥½È•µ‰•‘‘•M=½M= …¹(€½µÁÉ•ÍÍ¥½¸ÍÑÕ‘¥•Ì…¹ÍÑ…Ñ•ÌÑ¡…ÐÑ¡”¹½Ù•±Ñä¥ÌÑ¡”Á…¥É•°…Õ‘¥Ñ…‰±”MQ4ÌÈ(€‰•¹¡µ…É¬°¹½Ð„¹•ÜÉ•ÕÉÉ•¹Ð•±°½È½µÁÉ•ÍÍ¥½¸…±½É¥Ñ¡´¸(´mátQ¡”½¹±ÕÍ¥½¸…¹=ÕÑ±½½¬¹½ÜÅÕ…¹Ñ¥™¥•Ì±…Í °I4°¥¹™•É•¹”µÑ¥µ”°…¹5(€¡…¹•Ì™½ÈM=…¹M= °‘¥ÍÑ¥¹Õ¥Í¡•Ì•ÍÑ¥µ…Ñ••¹•Éä™É½´‘¥É•ÐÁ½Ý•È(€µ•…ÍÕÉ•µ•¹Ð°¥¹Ñ•ÉÁÉ•ÑÌÑ¡”½Á•É…Ñ¥¹œÁ½¥¹ÑÌ°…¹ÁÉ¥½É¥Ñ¥Í•Ì™ÕÑÕÉ”Ý½É¬¸(´mátµ…¹ÕÍÉ¥ÁÐµÝ¥‘”±…¹Õ…”…¹±½¥ŒÁ…ÍÌ½ÉÉ•Ñ•Ñ¡”‰ÍÑÉ…Ð°(€%¹ÑÉ½‘ÕÑ¥½¸°I•±…Ñ•]½É¬°5•Ñ¡½‘ÌÝ½É‘¥¹œ°I•ÍÕ±ÑÌ¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¸°¥ÍÕÍÍ¥½¸°(€…¹½¹±ÕÍ¥½¸¸(´mátIMA=9M}Q=}IY%]I|È¹µ‘€‘½Õµ•¹ÑÌ•Ù•ÉäÉ•ÍÁ½¹Í”Ý¥Ñ É•™•É•¹•ÌÑ¼Ñ¡”(€™¥¹…°€ÔÜµÁ…”±¥¹”µ¹Õµ‰•É•…¹½¹åµ½ÕÌÉ•Ù¥Í¥½¸¸(´mátQ¡”Á…”…¹±¥¹”É•™•É•¹•Ì¥¸IMA=9M}Q=}IY%]I|Ä¹µ‘€…¹(€IMA=9M}Q=}IY%]I|Ì¹µ‘€Ý•É”É•™É•Í¡•…™Ñ•ÈÑ¡”I•Ù¥•Ý•È€È¥¹Ñ•É…Ñ¥½¸¸((ŒŒŒI•Ù¥•Ý•È€ÌÍÑ…ÑÕÌ…™Ñ•ÈÑ¡”½µÁ±•Ñ•Ñ•¡¹¥…°‘¥ÍÕÍÍ¥½¸€ ÄÔ)Õ±ä€ÈÀÈØ¤((´mát±°Í¥àI•Ù¥•Ý•È€Ì½µµ•¹ÑÌ¡…Ù”¹½Ü‰••¸‘¥ÍÕÍÍ•…¹¡•­•……¥¹ÍÐÑ¡”(€…Ù…¥±…‰±”µ½‘•°½‘”°MQ4ÌÈÁÉ½©•ÑÌ°‰•¹¡µ…É¬É•ÍÕ±ÑÌ°…¹•¹•É…Ñ•…¹…±åÍ•Ì¸(´mát9¼…‘‘¥Ñ¥½¹…°•áÁ•É¥µ•¹ÐÉ•ÅÕ¥É¥¹œÑ¡”!A½È„¹•ÜMQ4ÌÈÉÕ¸¥ÌÕÉÉ•¹Ñ±ä(€Á±…¹¹•™½ÈI•Ù¥•Ý•È€Ì¸(´mát5…¹ÕÍÉ¥ÁÐ¥¹Ñ•É…Ñ¥½¸°…ÁÁ•¹‘¥àÁ±…•µ•¹Ð°…¹Ñ¡”Á½¥¹Ðµ‰äµÁ½¥¹ÐÉ•ÍÁ½¹Í”(€…É”½µÁ±•Ñ”¸IMA=9M}Q=}IY%]I|Ì¹µ‘€É•™•ÉÌÑ¼Ñ¡”€ÔÜµÁ…”±¥¹”µ¹Õµ‰•É•(€…¹½¹åµ½ÕÌÉ•Ù¥Í¥½¸½µÁ¥±•½¸€ÄÔ)Õ±ä€ÈÀÈØ¸((¨¨Ì¸Ä5½‘•°½µÁ±•á¥Ñä…¹¡…É‘Ý…É”ÑÉ…¹Í™•É…‰¥±¥Ñä¨¨((´mátÉ•Ù|Ù}µ½‘•±}½µÁ±•á¥Ñå}Í…±¥¹œ¹Á¹€…¹¥ÑÌ¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¸…É”…•ÁÑ•¸(´mát‘Ñ•áÐ•áÁ±…¥¹¥¹œÑ¡”…¹…±åÑ¥…°¡¥‘‘•¸µÍ¥é”Í…±¥¹œ°¥µÁ±•µ•¹Ñ•€ØÐ€´ø€ÐÕ€(€…¹€ÄÈà€´ø€äÁ€Á½¥¹ÑÌ°…¹Ñ¡”€ÔÄ”…ÍåµÁÑ½Ñ¥Œ5µÉ•‘ÕÑ¥½¸±¥µ¥Ð™½È„¹½µ¥¹…°(€€ÌÀ”¡¥‘‘•¸µÍ¥é”É•‘ÕÑ¥½¸¸(´mátMÑ…Ñ”Ñ¡…ÐÑ¡”…ÁÁÉ½á¥µ…Ñ•±ä€ÐÀ”µ•…ÍÕÉ•±…Í ½Ñ¥µ”½•¹•ÉäµÁÉ½áä™¥¹‘¥¹Ì…É”(€ÍÁ•¥™¥ŒÑ¼Ñ¡”Ñ•ÍÑ•¡•­Á½¥¹ÑÌ°MQ4ÌÉ ÜÔÍi$°™¥ÉµÝ…É”°…¹‰Õ¥±¸¼¹½Ð±…¥´(€ÑÉ…¹Í™•ÈÑ¼MQ4ÌÉÐ½È…¹½Ñ¡•È½¹ÑÉ½±±•È™…µ¥±äì½¹±äÑ¡”…É¡¥Ñ•ÑÕÉ”µ±•Ù•°5(€ÑÉ•¹ÑÉ…¹Í™•ÉÌÝ¥Ñ¡½ÕÐ„¹•Ü¡…É‘Ý…É”‰•¹¡µ…É¬¸((¨¨Ì¸È•É¥Ù…Ñ¥Ù”¥µÁ±•µ•¹Ñ…Ñ¥½¸…¹¹½¸µ¥‘•…°Í…µÁ±¥¹œ¨¨((´mát¥¹…°‘•¥Í¥½¸¥ÌÑ•áÐ½¹±äìÉ•Ù|Ý€É•µ…¥¹Ì¥¹Ñ•É¹…°…¹¥Ì¹½Ð¥¹Í•ÉÑ•¸(´mát½ÉÉ•ÐÑ¡”½É¥¥¹…°ÍÑ…Ñ•µ•¹Ð•¹ÑÉ•™¥¹¥Ñ”‘¥™™•É•¹•Í€¸Q¡”…Õ‘¥Ñ•(€¥µÁ±•µ•¹Ñ…Ñ¥½¸ÕÍ•ÌÑ¥µ•ÍÑ…µÀµ…Ý…É”‰…­Ý…É‘¥™™•É•¹•Ì¸(´mát‘Ñ¡”…ÕÍ…°•ÅÕ…Ñ¥½¸°ÍÑ…Ñ”Ñ¡…ÐÑ¡”Í¥àÁÉ•Á…É•™•…ÑÕÉ•ÌÝ•É”½µÁÕÑ•(€¡½ÍÐµÍ¥‘”…¹É•Á±…å•½Ù•ÈUIP°…¹±…É¥™äÑ¡…ÐÉ•Á½ÉÑ•­•É¹•°Ñ¥µ¥¹œ¥Í½±…Ñ•Ì(€1MQ4½51@¥¹™•É•¹”…¹•á±Õ‘•Ì…ÅÕ¥Í¥Ñ¥½¸…¹™•…ÑÕÉ”•áÑÉ…Ñ¥½¸¸(´mát‘„‰½Õ¹‘•±¥µ¥Ñ…Ñ¥½¸èÙ…É¥…‰±”Í…µÁ±¥¹œ°µ¥ÍÍ¥¹œÍ…µÁ±•Ì°‘•É¥Ù…Ñ¥Ù”¹½¥Í”°(€…¹…ÕÍ…°™¥±Ñ•É¥¹œ½±¥µ¥Ñ¥¹œÝ•É”¹½Ð•Ù…±Õ…Ñ•¥¸Ñ¡¥Ì½µÁÉ•ÍÍ¥½¸‰•¹¡µ…É¬¸(€I•™•ÈÑ¼Ñ¡”Í•Á…É…Ñ”É½‰ÕÍÑ¹•ÍÌÝ½É¬½¹±ä¥˜¥Ð¥Ì¥Ñ…‰±”…ÐÍÕ‰µ¥ÍÍ¥½¸Ñ¥µ”¸((¨¨Ì¸Ì0ÈÁÉÕ¹¥¹œÉ¥Ñ•É¥½¸¨¨((´mátÉ•Ù|á}ÁÉÕ¹¥¹}É¥Ñ•É¥½¹}Í½Á”¹Á¹€…¹Ñ¡”Ù•É¥™¥•Í…±¥•¹ä‘•™¥¹¥Ñ¥½¸…É”(€…•ÁÑ•¸(´mátQ¡”ÕÉÉ•¹Ðµ•Ñ¡½…±É•…‘ä•áÁ±…¥¹ÌÑ¡”ÍÑÉÕÑÕÉ•0ÈÉ¥Ñ•É¥½¸°‰ÕÐ…‘…¸(€•áÁ±¥¥ÐÍ•¹Ñ•¹”½¹ÑÉ…ÍÑ¥¹œ¥ÐÝ¥Ñ É…‘¥•¹Ð´…¹…Ñ¥Ù…Ñ¥½¸µ‰…Í•Í•¹Í¥Ñ¥Ù¥Ñäè(€Ñ¡½Í”…±Ñ•É¹…Ñ¥Ù•ÌÉ•ÅÕ¥É”‰…­Ý…ÉµÁ…ÍÌ½ÈÉ•ÁÉ•Í•¹Ñ…Ñ¥Ù”µ…Ñ¥Ù…Ñ¥½¸ÍÑ…Ñ¥ÍÑ¥Ì°(€Ý¡•É•…ÌÑ¡”Í•±•Ñ•…Ñ”µÉ½ÕÀ0ÈÍ½É”¥Ì‘…Ñ„¥¹‘•Á•¹‘•¹Ð…¹‘¥É•Ñ±äµ…ÁÌÑ¼(€É•µ½Ù…°½˜½µÁ±•Ñ”‘•¹Í”¡¥‘‘•¸¡…¹¹•±Ì¸(´mátMÑ…Ñ”•áÁ±¥¥Ñ±äÑ¡…Ð¹¼•áÁ•É¥µ•¹Ñ…°½µÁ…É¥Í½¸½˜ÁÉÕ¹¥¹œÉ¥Ñ•É¥„Ý…ÌÉÕ¸ì(€Ñ¡”É…Ñ¥½¹…±”¥Ìµ•Ñ¡½‘½±½¥…°…¹¥µÁ±•µ•¹Ñ…Ñ¥½¸‰…Í•°¹½Ð•Ù¥‘•¹”Ñ¡…Ð0È¥Ì(€Õ¹¥Ù•ÉÍ…±±äÍÕÁ•É¥½È¸((¨¨Ì¸Ð]•¥¡Ðµ½¹±äµ¥á•µÁÉ•¥Í¥½¸ÅÕ…¹Ñ¥é…Ñ¥½¸¨¨((´mátÉ•Ù|å}µ¥á•‘}ÁÉ•¥Í¥½¹}ÅÕ…¹Ñ¥é…Ñ¥½¸¹Á¹€Ý…Ì½ÉÉ•Ñ•……¥¹ÍÐÑ¡”…±±•M=(€…¹M= ­•É¹•±Ì°•áÁ½ÉÑ•¡•…‘•ÉÌ°…¹±¥¹­•Èµ…ÁÌ¸%ÑÌÁ…¹•°€¡„¤½¹Ñ…¥¹Ì½¹±äÑ¡”(€ÁÉ•¥Í¥½¸Á…Ñ ¸(´mát-••ÀÑ¡”Ù•É¥™¥•Í½Á”¥¸Ñ¡”µ•Ñ¡½è½¹±äÉ•ÕÉÉ•¹Ð]}¥¡€…¹]}¡¡€…É”(€%9PàìÉ½ÜÍ…±•Ì°‰¥…Ì°ÍÑ…Ñ•Ì°…Ñ¥Ù…Ñ¥½¹Ì°51@°…¹½ÕÑÁÕÐÉ•µ…¥¸@ÌÈ¸(´mát‘Ñ¡”µ¥ÍÍ¥¹œ•áÁ±¥¥Ð½¹Í•ÅÕ•¹”èÉ•Ñ…¥¹¥¹œ@ÌÈ…Ñ¥Ù…Ñ¥½¹Ì½ÍÑ…Ñ•Ì…Ù½¥‘Ì(€¥¹ÑÉ½‘Õ¥¹œ…¹½Ñ¡•ÈÅÕ…¹Ñ¥é…Ñ¥½¸Í½ÕÉ”‰ÕÐÁÉ•Í•ÉÙ•Ì@ÌÈI4…¹…É¥Ñ¡µ•Ñ¥Œ½ÍÐ°(€Í¼Ñ¡”¥µÁ±•µ•¹Ñ…Ñ¥½¸…¹¹½Ð½‰Ñ…¥¸Ñ¡”±…Ñ•¹ä…¹I4‰•¹•™¥ÑÌ½˜„™Õ±°µ¥¹Ñ••È(€Á…Ñ ¸Q¡”…ÕÉ…ä‰•¹•™¥Ð½˜Ñ¡¥Ì¡½¥”Ý…Ì¹½Ð¥Í½±…Ñ••áÁ•É¥µ•¹Ñ…±±ä¸(´mátMÑ…Ñ”‘¥É•Ñ±äÑ¡…Ð…Ñ¥Ù…Ñ¥½¸ÅÕ…¹Ñ¥é…Ñ¥½¸…¹™Õ±±ä¥¹Ñ••È‘•Á±½åµ•¹ÐÝ•É”(€¹½Ð•Ù…±Õ…Ñ•¸áÁ±…¥¸É•ÕÉÉ•¹ÐµÝ•¥¡Ðµ½¹±äÍ½Á”Ù¥„Ñ¡”…ÁÁÉ½á¥µ…Ñ•±ä€àÀ”Í¡…É”(€½˜É•ÕÉÉ•¹Ðµ…ÑÉ¥•Ì¥¸Ñ¡”	…Í”µ½‘•°½¹ÍÑ…¹ÑÌ°Ý¥Ñ¡½ÕÐ±…¥µ¥¹œÑ¡…ÐÉ•Ñ…¥¹¥¹œ(€Ñ¡”@ÌÈ51@Ý…Ì•áÁ•É¥µ•¹Ñ…±±äÍÕÁ•É¥½È¸((¨¨Ì¸Ô1½Ý•ÈM=AÉÕ¹•5¨¨((´mátQ¡¥ÌÁ½¥¹Ð¥Ì…±É•…‘ä¡…¹‘±•½ÉÉ•Ñ±ä¥¸Ñ¡”ÕÉÉ•¹Ðµ…¹ÕÍÉ¥ÁÐèÑ¡”…ÕÍ…°(€É•Õ±…É¥Í…Ñ¥½¸±…¥´¡…Ì‰••¸É•µ½Ù•°Ñ¡”€À¸ÌÔµÁ•É•¹Ñ…”µÁ½¥¹Ð‘¥™™•É•¹”¥Ì(€¡•­Á½¥¹Ð´…¹ÍÁ±¥ÐµÍÁ•¥™¥Œ°…¹Ñ¡”…‰Í•¹”½˜É•Á•…Ñ•Í••‘Ì°É½ÍÌµÙ…±¥‘…Ñ¥½¸°(€…¹µ…Ñ¡•Õ¹ÁÉÕ¹•™¥¹”µÑÕ¹¥¹œ¥ÌÍÑ…Ñ•¸(´mát9¼¹•Ü‘¥…É…´½È•áÁ•É¥µ•¹Ð¥ÌÉ•ÅÕ¥É•¸-••ÀÑ¡”•á¥ÍÑ¥¹œÍ…±¥•¹ä½Ý•¥¡Ð(€‘¥…¹½ÍÑ¥Ì‘•ÍÉ¥ÁÑ¥Ù”…¹‘¼¹½ÐÕÍ”Ñ¡•´…ÌÁÉ½½˜½˜¥µÁÉ½Ù••¹•É…±¥Í…Ñ¥½¸¸((¨¨Ì¸ØEÕ…¹Ñ¥é•ÉÕ¹Ñ¥µ”¥¹É•…Í”¨¨((´mátÉ•Ù|ÄÁ}ÅÕ…¹Ñ¥é•‘}ÉÕ¹Ñ¥µ•}…½Õ¹Ñ¥¹œ¹Á¹€°¥ÑÌ5™½ÉµÕ±„°…‘‘¥Ñ¥½¹…°Í…±”(€½Õ¹ÑÌ°µ•…ÍÕÉ•]PÑ¥µ•Ì°…¹Ñ¡”ÍÑ…Ñ¥ŒµÙ•ÉÍÕÌµµ•…ÍÕÉ•¥¹Ñ•ÉÁÉ•Ñ…Ñ¥½¸¡…Ù”‰••¸(€‘¥ÍÕÍÍ•…¹…•ÁÑ•…ÌÕÍ•™Õ°•Ù¥‘•¹”¸(´mátáÁ…¹Ñ¡”ÕÉÉ•¹Ð‘¥ÍÕÍÍ¥½¸‰•å½¹½¹Í¥ÍÑ•¹ÐÝ¥Ñ Ñ¡”½‰Í•ÉÙ•Ñ¥µ¥¹œ(€Á•¹…±Ñå€èÉ•Á½ÉÐÑ¡…ÐÍÑ…Ñ¥ŒÉ…Ñ¥½Ì…É”…‰½ÕÐ€Ä¸àÄ™½È‰½Ñ Ñ…Í­Ì°Ý¡•É•…Ìµ•…ÍÕÉ•(€EÕ…¹Ñ¥é•½	…Í”É…Ñ¥½Ì…É”€Ð¸ää™½ÈM=…¹€Ä¸Èä™½ÈM= ¸áÁ±…¥¸Ý¡äÍÑ…Ñ¥Œ½Õ¹ÑÌ(€…É”ÁÉ•‘¥Ñ¥Ù”Ý¥Ñ¡¥¸Ñ¡”@ÌÈÁÉÕ¹¥¹œ™…µ¥±ä‰ÕÐ¹½Ð…É½ÍÌ„¡…¹•­•É¹•°¸(´mátI•½ÉÑ¡…ÐÑ¡”¥¹ÍÁ•Ñ•™¥ÉµÝ…É”ÕÍ•€µ<Á€°¡…Éµ™±½…Ð@ÌÈ°Á•ÈµÝ•¥¡Ð(€%9PàµÑ¼µ@ÌÈ…ÍÑÌ°…¹Á•ÈµÝ•¥¡ÐÉ½ÜÍ…±¥¹œÝ¥Ñ¡½ÕÐ…¸½ÁÑ¥µ¥é•¥¹Ñ••È(€‘½ÐµÁÉ½‘ÕÐÁ…Ñ ¸AÉ•Í•¹ÐÉ•‘Õ•M= Ý•¥¡ÐÑÉ…™™¥Œ…Ì„Á±…ÕÍ¥‰±”½™™Í•ÑÑ¥¹œ(€™…Ñ½È°¹½Ð…Ìµ•…ÍÕÉ•µ•µ½Éäµ‰…¹‘Ý¥‘Ñ …ÑÑÉ¥‰ÕÑ¥½¸¸(´mát‘™ÕÑÕÉ”½ÁÑ¥µ¥é…Ñ¥½¸½ÁÑ¥½¹ÌÉ•ÅÕ•ÍÑ•‰äÑ¡”É•Ù¥•Ý•Èè¡½¥ÍÐÉ½ÜÍ…±•Ì(€½ÕÑÍ¥‘”¥¹¹•È…ÕµÕ±…Ñ¥½¹ÌÝ¡•É”Ù…±¥°™ÕÍ”Í…±”½‰¥…ÌÝ½É¬°ÕÍ”½ÁÑ¥µ¥é•M@½È(€¥¹Ñ••È‘½ÐµÁÉ½‘ÕÐ­•É¹•±Ì°…¹•Ù…±Õ…Ñ”…Ñ¥Ù…Ñ¥½¸ÅÕ…¹Ñ¥é…Ñ¥½¸½™Õ±°µ¥¹Ñ••È(€•á•ÕÑ¥½¸¸Q¡•Í”…É”ÁÉ½Á½Í…±Ì°¹½Ð•Ù…±Õ…Ñ•É•ÍÕ±ÑÌ¸(´mátMÑ…Ñ”•áÁ±¥¥Ñ±äÑ¡…Ð¹¼½Á•É…Ñ¥½¸µ±•Ù•°å±”‰É•…­‘½Ý¸½Èµ•µ½Éäµ‰…¹‘Ý¥‘Ñ (€ÁÉ½™¥±”Ý…Ì…Ù…¥±…‰±”¸Q¡”•á¥ÍÑ¥¹œ]Pµ•…ÍÕÉ•µ•¹ÑÌ…É”Ñ½Ñ…°­•É¹•°Ñ¥µ•Ì…Ù•É…•(€½Ù•È€ÄÀ°ÀÀÀ¥¹™•É•¹•Ì¸((ŒŒŒI•Ù¥•Ý•È€Ð½µÁ±•Ñ¥½¸ÍÑ…ÑÕÌ€ ÄÔ)Õ±ä€ÈÀÈØ¤((´mátQ¡”‰ÍÑÉ…ÐÝ…ÌÉ•‘Õ•Ñ¼…ÁÁÉ½á¥µ…Ñ•±ä€ÄØäÝ½É‘Ì…¹¹½Ü‘¥É•Ñ±ä¥‘•¹Ñ¥™¥•Ì(€Ñ¡”•¹¥¹••É¥¹œ…ÁÁ±¥…Ñ¥½¸½˜…ÉÑ¥™¥¥…°¥¹Ñ•±±¥•¹”°Ñ¡”¥µÁ±•µ•¹Ñ•…ÉÑ¥™¥¥…°(€¥¹Ñ•±±¥•¹”°Ñ¡”Á…¥É•½µÁÉ•ÍÍ¥½¸‰•¹¡µ…É¬°Ñ¡”ÁÉ¥¹¥Á…°¹Õµ•É¥…°É•ÍÕ±ÑÌ°(€…¹Ñ¡”•Ù¥‘•¹”‰½Õ¹‘…Éä¸±°…‰ÍÑÉ…Ð…É½¹åµÌ…É”‘•™¥¹•½¸™¥ÉÍÐÕÍ”¸(´mátQ¡”­•åÝ½É±¥ÍÐÝ…ÌÉ•‘Õ•Ñ¼Í¥àÝÉ¥ÑÑ•¸µ½ÕÐÑ•ÉµÌ¸Q¡”Ñ¥Ñ±”…¹­•åÝ½É‘Ì(€½¹Ñ…¥¸¹¼Õ¹•áÁ±…¥¹•…É½¹å´¸(´mátQ¡”ÑÝ¼É•Ù¥•Ý•ÈµÍÕ•ÍÑ•Á…Á•ÉÌÝ•É”¥µÁ½ÉÑ•™É½´Ñ¡”…ÕÑ¡½ÈÌi½Ñ•É¼•áÁ½ÉÐ(€‰¥ˆ½Á…Á•ÈÉ}Í½Í½¡}É•™}™È¹‰¥‰€¥¹Ñ¼Ñ¡”¥Í½±…Ñ•É•Ù¥•Ü‰¥‰±¥½É…Á¡ä…Ì(€Ý…¹}¥µÁÉ½Ù•‘|ÈÀÈÑ€€¡=$€ÄÀ¸ÄÀÄØ½¨¹•ÍÐ¸ÈÀÈÌ¸ÄÄÀÈÈÉ€¤…¹(€Ý…¹}¥µÁÉ½Ù•‘|ÈÀÈÍ€€¡=$€ÄÀ¸ÄÀÄØ½¨¹•¹•Éä¸ÈÀÈÌ¸ÄÈàØÜÝ€¤¸9¼‰¥‰±¥½É…Á¡¥Œ‘…Ñ„(€Ý•É”¥¹Ù•¹Ñ•½È¥¹™•ÉÉ•¸(´mátI•±…Ñ•]½É¬¹½Ü•áÁ±…¥¹ÌÝ¡…ÐÑ¡”ÑÝ¼¡å‰É¥•ÍÑ¥µ…Ñ½ÉÌ½¹ÑÉ¥‰ÕÑ”…¹Ý¡ä(€Ñ¡•¥È…ÕÉ…ä•Ù¥‘•¹”½µÁ±•µ•¹ÑÌ°‰ÕÐ‘½•Ì¹½ÐÉ•Á±…”°Ñ¡”Á…¥É••µ‰•‘‘•(€ÁÉÕ¹¥¹œ½ÅÕ…¹Ñ¥é…Ñ¥½¸‰•¹¡µ…É¬¸(´mátQ¡”µ•Ñ¡½±½¥Œ¹½Ü•áÁ±¥¥Ñ±äÍ•Á…É…Ñ•ÌÑ¡”½µµ½¸	…Í”•ÅÕ…Ñ¥½¹Ì°ÍÑÉÕÑÕÉ•(€‘¥µ•¹Í¥½¸É•‘ÕÑ¥½¸°…¹É•ÁÉ•Í•¹Ñ…Ñ¥½¸µ½¹±äÅÕ…¹Ñ¥é…Ñ¥½¸¸(´mát5…Ñ¡•µ…Ñ¥…°‘•Ñ…¥°Ý…Ì…‘‘•™½È‘•¹Í”ÁÉÕ¹¥¹œÍÕ‰µ…ÑÉ¥à½¹ÍÑÉÕÑ¥½¸°(€½‘”µ•á…ÐÍåµµ•ÑÉ¥ŒÁ•ÈµÉ½ÜÅÕ…¹Ñ¥é…Ñ¥½¸½Ù•È€´ÄÈÜ¸¸¸ÄÈÝ€°Ñ¡”É•½¹ÍÑÉÕÑ¥½¸(€•ÉÉ½È‰½Õ¹°Ñ¡”µ¥á•µÁÉ•¥Í¥½¸…Ñ”…±Õ±…Ñ¥½¸°…¹5½I5M½@äÔ‘•™¥¹¥Ñ¥½¹Ì¸(´mátQ¡”ÅÕ…¹Ñ¥é…Ñ¥½¸‘•ÍÉ¥ÁÑ¥½¸Ý…Ì¡•­•……¥¹ÍÐÑ¡”M=…¹M= •áÁ½ÉÐ½‘”¸(€Q¡”ÁÉ•Ù¥½ÕÌÍ¡•µ…Ñ¥Œ½…ÁÑ¥½¸É…¹”€´ÄÈà¸¸¸ÄÈÝ€Ý…Ì½ÉÉ•Ñ•Ñ¼Ñ¡”¥µÁ±•µ•¹Ñ•(€Íåµµ•ÑÉ¥ŒÍ•Ð€´ÄÈÜ¸¸¸ÄÈÝ€Ý¥Ñ €´ÄÈá€‘•±¥‰•É…Ñ•±äÕ¹ÕÍ•¸(´mát1½¹œ…Ñ…Í•Ð°QÉ…¥¹¥¹œ°	•¹¡µ…É¬°I•ÍÕ±ÑÌ°¥ÍÕÍÍ¥½¸°…¹1¥µ¥Ñ…Ñ¥½¹ÌÁ…ÍÍ…•Ì(€Ý•É”ÍÁ±¥Ð½ÈÑ¥¡Ñ•¹•°…¹„µ…¹ÕÍÉ¥ÁÐµÝ¥‘”É…µµ…È…¹±½¥ŒÁ…ÍÌÝ…ÌÁ•É™½Éµ•¸(´mát¥ÕÉ•Ì€È°€Ø°…¹€ÜÝ•É”É••¹•É…Ñ•…Ð¡¥ É•Í½±ÕÑ¥½¸Ý¥Ñ ±…É•ÈÑ•áÐ…¹(€½‘”µ½¹Í¥ÍÑ•¹Ð½¹Ñ•¹Ð¸¥ÕÉ”€ÈÍ¡½ÝÌÑ¡”É•…°½™…Ñ½È±•Ù•±Ì°¥ÕÉ”€ØÍ¡½ÝÌ(€‘•¹Í”¡¥‘‘•¸µ¡…¹¹•°É•µ½Ù…°…¹Ñ¡”€ØÐ€´ø€ÐÕ€€¼€ÄÈà€´ø€äÁ€‘¥µ•¹Í¥½¹Ì°…¹(€¥ÕÉ”€ÜÍ¡½ÝÌÑ¡”Ù•É¥™¥•µ¥á•µÁÉ•¥Í¥½¸Á…Ñ ¸(´mátQ¡”É•ÁÉ½‘Õ¥‰±”•¹•É…Ñ½È¥Ì(€É•Ù¥•Ý}…¹…±åÍ¥Ì½Ñ½½±Ì½•¹•É…Ñ•}É•Ù¥•Ý•ÈÑ}™¥ÕÉ•Ì¹ÁÌÅ€¸(´mátÕÑ¡½È…¹…¹½¹åµ½ÕÌÉ•Ù¥•Üµ…¹ÕÍÉ¥ÁÑÌ½µÁ¥±”Ý¥Ñ¡½ÕÐÕ¹‘•™¥¹•¥Ñ…Ñ¥½¹Ì°(€Õ¹‘•™¥¹•É•™•É•¹•Ì°½È1…Q•`•ÉÉ½ÉÌ¸Q¡”™¥¹…°AÌ½¹Ñ…¥¸€ÌØ…¹€ÔÜÁ…•Ì°(€É•ÍÁ•Ñ¥Ù•±ä¸(´mátIMA=9M}Q=}IY%]I|Ð¹µ‘€É•½É‘ÌÑ¡”Á½¥¹Ðµ‰äµÁ½¥¹ÐÉ•ÍÁ½¹Í”ÕÍ¥¹œÑ¡”™¥¹…°(€€ÔÜµÁ…”±¥¹”µ¹Õµ‰•É•…¹½¹åµ½ÕÌA¸I•ÍÁ½¹Í•Ì€Ä´´ÌÝ•É”É•™É•Í¡•…™Ñ•ÈÑ¡¥Ì(€¥¹Ñ•É…Ñ¥½¸¸((ŒŒŒ¸Y•É¥™¥•ÁÉÕ¹¥¹œ¥µÁ±•µ•¹Ñ…Ñ¥½¸((¨©5•Ñ¡½¨¨((´Q¡”™¥¹…°µ½‘•±ÌÕÍ”½¹”µÍ¡½Ð°µ…¹¥ÑÕ‘”µ‰…Í•ÍÑÉÕÑÕÉ•ÁÉÕ¹¥¹œ½˜½µÁ±•Ñ”1MQ4(€¡¥‘‘•¸Õ¹¥ÑÌ°¹½Ð¥Ñ•É…Ñ¥Ù”ÁÉÕ¹¥¹œ…¹¹½ÐÁÉÕ¹¥¹œµ…Ý…É”ÑÉ…¥¹¥¹œ¸(´U¹¥ÐÍ…±¥•¹ä¥ÌÑ¡”ÍÕ´½˜0È¹½ÉµÌ½˜¥¹ÁÕÐ…¹É•ÕÉÉ•¹ÐÝ•¥¡ÐÉ½ÝÌ½Ù•È…±°(€™½ÕÈ1MQ4…Ñ•Ì¸(´Q¡”±½Ý•ÍÐµÍ…±¥•¹äÕ¹¥ÑÌ…É”É•µ½Ù•©½¥¹Ñ±ä™É½´…±°…Ñ”µ…ÑÉ¥•Ì¸ÍÍ½¥…Ñ•(€É•ÕÉÉ•¹Ð½±Õµ¹Ì…¹Ñ¡”¥¹ÁÕÐ½±Õµ¹Ì½˜Ñ¡”‘½Ý¹ÍÑÉ•…´51@…É”Í±¥•Ñ¼™½É´(€„Á¡åÍ¥…±±äÍµ…±±•È‘•¹Í”¹•ÑÝ½É¬¸(´M=¡…¹•Ì™É½´€ØÐÑ¼€ÐÔ¡¥‘‘•¸Õ¹¥ÑÌìM= ¡…¹•Ì™É½´€ÄÈàÑ¼€äÀ¡¥‘‘•¸Õ¹¥ÑÌ°(€½ÉÉ•ÍÁ½¹‘¥¹œÑ¼…ÁÁÉ½á¥µ…Ñ•±ä€ÌÀ”É•µ½Ù…°¸(´Q¡”É•‘Õ•µ½‘•°¥ÌÑ¡•¸‰É¥•™±ä™¥¹”µÑÕ¹•Ý¥Ñ ‘…µ\…¹5M±½ÍÌ¸((¨©A…É…µ•Ñ•ÉÌÑ¡…Ð…É”…ÑÕ…±±ä•Ù¥‘•¹•¨¨((´M= ¥Ì™Õ±±ä‘½Õµ•¹Ñ•‰ä…É¡¥Ù”½ÉÕ¹}ÁÉÕ¹¥¹œ¹Í¡€…¹Ñ¡”Ù…±¥™¥¹…°µ…¹¥™•ÍÐè(€€ÌÀ”ÁÉÕ¹¥¹œ°€Ì™¥¹”µÑÕ¹¥¹œ•Á½¡Ì°±•…É¹¥¹œÉ…Ñ”€Å”´Ù€°ÑÉ…¥¹¥¹œ‰…Ñ Í¥é”€ÔÄÈ°(€…¹Í…Ù•Ù…±¥‘…Ñ¥½¸µ•ÑÉ¥Ì…™Ñ•È™¥¹”µÑÕ¹¥¹œ¸(´Q¡”™¥¹…°M=™½±‘•È…¹€ØÐµÑ¼´ÐÔ¡•­Á½¥¹Ðµ…Ñ Ñ¡”Í¥µÁ±”½¹”µÍ¡½ÐÁÉÕ¹¥¹œ(€ÍÉ¥ÁÐ…¹Ñ¡”ÁÉ½©•ÐI5ÍÑ…Ñ•ÌÑ¡…ÐÑ¡”•á¥ÍÑ¥¹œµ½‘•°É••¥Ù•Í¡½ÉÐ(€™¥¹”µÑÕ¹¥¹œ¸(´Q¡”™¥¹…°M=µ…¹¥™•ÍÐ¹©Í½¹€¥ÌÑÉÕ¹…Ñ•Ñ¼€ÄÀà‰åÑ•Ì¸Q¡”•á…ÐM=•Á½ ½Õ¹Ð(€…¹±•…É¹¥¹œÉ…Ñ”…É”Ñ¡•É•™½É”¹½ÐÉ•½Ù•É…‰±”™É½´Ñ¡…Ð™¥±”¸MÉ¥ÁÐ‘•™…Õ±ÑÌ(€…É”€Ô•Á½¡Ì…¹€Õ”´Õ€°Ý¡¥±”…¹½Ñ¡•È…É¡¥Ù•ÕÍ…”•á…µÁ±”Í¡½ÝÌ€Ì•Á½¡Ì¸(´U¹Ñ¥°…¸½É¥¥¹…°M=½µµ…¹½È±½œ¥Ì™½Õ¹°‘¼¹½ÐÍÑ…Ñ”…¸•á…ÐM=•Á½ (€½Õ¹Ð½È±•…É¹¥¹œÉ…Ñ”…Ì™…Ð¸]É¥Ñ”‰É¥•˜Á½ÍÐµÁÉÕ¹¥¹œ™¥¹”µÑÕ¹¥¹€…¹‘¥Í±½Í”(€Ñ¡”É•ÁÉ½‘Õ¥‰¥±¥Ñä±¥µ¥Ñ…Ñ¥½¸¥˜¹••ÍÍ…Éä¸((¨©I•ÅÕ¥É•½ÉÉ•Ñ¥½¹Ì…¹±…¥µÌ¨¨((´I•Á±…”Ñ¡”ÕÉÉ•¹ÐÍ•¹Ñ•¹”™Ñ•È•… ÁÉÕ¹¥¹œÍÑ•À€¸¸¸‰•™½É”Ñ¡”¹•áÐÁÉÕ¹¥¹œ(€É½Õ¹‘€‰•…ÕÍ”Ñ¡•É”Ý…Ì½¹”ÁÉÕ¹¥¹œÍÑ•À…¹¹¼¹•áÐÉ½Õ¹¸(´]•¥¡Ðµ‘¥ÍÑÉ¥‰ÕÑ¥½¸…¹0ÈµÍ…±¥•¹ä‘¥…É…µÌÁÉ½Ù¥‘”‘•ÍÉ¥ÁÑ¥Ù”ÍÕÁÁ½ÉÐ™½ÈÑ¡”(€ÁÉÕ¹¥¹œÉ¥Ñ•É¥½¸°‰ÕÐÑ¡•ä‘¼¹½ÐÁÉ½Ù”Ñ¡…ÐÁÉÕ¹¥¹œ…ÕÍ•É•Õ±…É¥é…Ñ¥½¸¸(´•ÍÉ¥‰”Ñ¡”±½Ý•ÈM=5…Ì…¸½‰Í•ÉÙ•É•ÍÕ±Ð™½ÈÑ¡”•Ù…±Õ…Ñ•¡•­Á½¥¹Ð…¹(€ÍÁ±¥Ð¸9¼¹•ÜÉ…¹‘½´µÍ••°É½ÍÌµÙ…±¥‘…Ñ¥½¸°ÁÉÕ¹¥¹œµ…Ý…É”°½ÈEP•áÁ•É¥µ•¹ÑÌ…É”(€Á±…¹¹•¸((ŒŒŒ¸I•Ù¥Í¥½¸‰½Õ¹‘…Éä((´9¼¹•Ü!AÑÉ…¥¹¥¹œ½È…É¡¥Ñ•ÑÕÉ”ÍÝ••À¸(´9¼¹•ÜMQ4ÌÈ½ÈMQ4ÌÉÐ‘•Á±½åµ•¹Ð¸(´9¼¹•ÜÁ½Ý•Èµ•…ÍÕÉ•µ•¹Ð½È½Á•É…Ñ¥½¸µ±•Ù•°¡…É‘Ý…É”ÁÉ½™¥±¥¹œ¸(´9¼ÅÕ…¹Ñ¥é…Ñ¥½¸µ…Ý…É”½ÈÁÉÕ¹¥¹œµ…Ý…É”ÑÉ…¥¹¥¹œ¸(´9¼¹•Ü‘•É¥Ù…Ñ¥Ù”µÉ½‰ÕÍÑ¹•ÍÌ…µÁ…¥¸¥¸Ñ¡¥Ì½ÁÑ¥µ¥Í…Ñ¥½¸Á…Á•È¸(´9¼¹•Ü95½1<µ•…ÍÕÉ•µ•¹ÑÌ¸(´Q¡”É•Ù¥Í¥½¸Ý¥±°É•±ä½¸Ù•É¥™¥•…ÉÑ•™…ÑÌ°…‘‘¥Ñ¥½¹…°½™™±¥¹”…¹…±åÍ•Ì°±•…É•È(€µ…Ñ¡•µ…Ñ¥…°‘•ÍÉ¥ÁÑ¥½¹Ì°…ÁÁ•¹‘¥•Ì°…É•™Õ±±ä‰½Õ¹‘•±…¥µÌ°…¹•áÁ±¥¥Ð(€±¥µ¥Ñ…Ñ¥½¹Ì¸(