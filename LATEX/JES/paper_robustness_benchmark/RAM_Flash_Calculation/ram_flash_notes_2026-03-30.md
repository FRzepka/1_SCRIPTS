# RAM and Flash Notes

These notes document the internal reasoning behind the flash and RAM values used for the embedded-feasibility discussion in the manuscript.

## Scope

- `Flash` refers to estimated persistent model/storage footprint.
- `RAM*` in the manuscript refers only to persistent numerical states that are intrinsic to the implemented architecture.
- The following are deliberately excluded from `RAM*`:
  - temporary buffers
  - framework overhead
  - stack usage
  - allocator behavior
  - driver/RTOS/runtime overhead
  - system-level runtime memory

## Securely Known Persistent State Contributions

### SOC-GRU core

- Compact deployment-prepared variant: hidden size `67`
- One GRU hidden state is stored persistently
- Values:
  - `1 x 67 = 67` float values
  - `67 x 4 B = 268 B` for float32

### SOH-LSTM core

- Compact deployment-prepared variant: `2` LSTM layers, hidden size `112`
- Persistent states:
  - hidden state `h`
  - cell state `c`
- Values:
  - hidden values: `2 x 112 = 224`
  - cell values: `2 x 112 = 224`
  - total: `448` float values
  - `448 x 4 B = 1792 B` for float32

### DM core

Persistent states directly visible in `CC_1.0.0/cc_model.py`:

- `q_m_new`
- `soc`
- `cv_time_s`
- `last_timestamp_s`
- `_capacity_override_init_done`

Interpretation used for the manuscript table:

- `4` persistent numerical states
- `1` logical state flag
- table value kept intentionally minimal: `16 B`

Note:
- the boolean flag is not expanded into a platform-specific byte count in the manuscript calculation
- the goal of the table is a strict lower-bound style state-memory indication

### HECM core

Persistent states directly visible in `ECM_0.0.3/fast_ecm.py`:

- state vector `x` with `3` values
- covariance matrix `P` with `3 x 3 = 9` values
- previous current `I_prev` with `1` value

Values:

- total: `13` float values
- `13 x 4 B = 52 B` for float32

## Estimator-Level RAM* Aggregation Used in the Manuscript

- `DM`: `16 B`
- `HDM`: `16 B + 1792 B = 1808 B`
- `HECM`: `52 B + 1792 B = 1844 B`
- `DD`: `268 B + 1792 B = 2060 B`

## Why This Was Kept Out of the Main Text

The manuscript keeps the table notation compact and readable. The full derivation is documented here instead of in the main methodology text to avoid overloading the paper with implementation-detail prose.
