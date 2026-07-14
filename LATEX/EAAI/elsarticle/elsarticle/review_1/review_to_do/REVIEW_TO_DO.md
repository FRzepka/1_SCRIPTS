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

## Manuscript work still required

- Shorten and restructure the abstract; define every acronym on first use.
- Remove acronyms from title and keywords. Ensure Editorial Manager entries match.
- State one concise objective and explicit contributions at the end of Introduction.
- Strengthen novelty against current embedded SOC/SOH, pruning, and quantization
  literature. Verify every suggested reference before adding it.
- Add a chemistry-scope paragraph: results are demonstrated only for LFP and cannot
  be numerically generalized to NMC or LCO without validation.
- Add the L2-pruning rationale and distinguish one-shot structured pruning plus
  fine-tuning from pruning-aware or iterative training.
- Replace the incorrect statement about successive pruning rounds. The implemented
  method performs one 30% structured pruning operation followed by short fine-tuning.
- Add utility-weight definitions and the new sensitivity result.
- Add a precise FP32 activation/state and INT8 recurrent-weight implementation note.
- Add limitations for one chemistry, one split/seed, one STM32 family, estimated
  energy, no QAT, no internal fault injection, and no operation-level profiling.
- Quantify the main results in the conclusion and add concrete future work.
- Break up long paragraphs, proofread the whole manuscript, and improve unclear
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
- `rev_7_derivative_deployment_boundary.png` addresses Reviewer 3.2. Panel (a)
  documents the audited offline feature path; panel (b) shows a possible causal
  handling strategy for non-ideal sampling and is explicitly marked as not tested.
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
- The revision will rely on verified artefacts, additional offline analyses, clearer
  mathematical descriptions, appendices, carefully bounded claims, and explicit
  limitations.
