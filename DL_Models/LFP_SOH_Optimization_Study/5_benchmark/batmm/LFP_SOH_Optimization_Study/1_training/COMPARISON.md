# SOH Model Comparison

## Version Overview

| Family | Meaning |
|--------|---------|
| **0.1** | LSTM-based SOH models |
| **0.1.1** | Per-second SOH prediction (raw sequence inputs) |
| **0.1.2** | Sampled/aggregated SOH prediction (e.g., hourly means) |
| **0.2** | TCN-based SOH models |
| **0.2.1** | Per-second SOH prediction (raw sequence inputs) |
| **0.2.2** | Sampled/aggregated SOH prediction (hourly stats) |
| **0.3** | GRU-based SOH models |
| **0.4** | CNN-based SOH models |

## Architecture Overview

⭐ = current best per family

| Version | Architecture | Stateful | Parameters | Hidden Size | Features | Seq Len | Stride | Batches | Speed | Notes |
|---------|-------------|----------|------------|-------------|----------|---------|--------|---------|-------|-------|
| **0.1.0.0** | LSTM Hybrid | ✅ | ~280K | 128 | 8 (with derivatives) | 2048 | 1 | ~860K | - | **CRASHED** - BatchNorm issue |
| **0.1.1.0** | LSTM + Attention | ✅ | 414K | 160 | 6 | 2048 | 1 | 862K | 7.4 it/s | **SLOW** - Too many samples |
| **0.1.1.1** | LSTM + Attention | ✅ | 414K | 160 | 6 | **512** | **8** | **40K** | ~37 it/s | **SLOW RESULTS** - R²=-0.30 @ epoch 12 ❌ |
| **0.1.1.2** | LSTM + Attention | ✅ | 414K | 160 | 7 (derivatives) | **512** | **8** | **~50K** | ~35 it/s | **NEW DATA SPLIT** - Better train/val coverage 🔄 |
| **0.1.1.3** | LSTM Seq2Seq (no attention) | ✅ | - | 160 | 7 (derivatives) | **512** | **8** | **~50K** | TBD | **STATEFUL-ALIGNED** - Stopped |
| **0.1.1.4** | LSTM Seq2Seq (long seq) | ✅ | - | 160 | 7 (derivatives) | **2048** | **500** | **~?** | TBD | **LONG SEQ** - Wide stride |
| **0.1.1.5** | LSTM Seq2Seq (smooth) | ✅ | - | 160 | 5 (no derivatives) | **1024** | **128** | **~?** | TBD | **SMOOTH LOSS** - Less noisy |
| **0.1.2.1** | LSTM Seq2Seq (hourly) | ✅ | - | 160 | 5 (no derivatives) | **168h** | **24h** | **~?** | TBD | **HOURLY MEAN** - MCU-friendly |
| **0.1.2.2** | LSTM Seq2Seq (hourly stats) | ✅ | ~394K | 160 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **HOURLY STATS** - mean/std/min/max |
| **0.1.2.3 ⭐** | LSTM Seq2Seq (larger, hourly stats) | ✅ | - | 192 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **FINAL LSTM** - best overall |
| **0.2.1.0** | TCN (Complex) | ❌ | 379K | 128 | 6 | 2048 | 1 | 647K | 1.0 it/s | **VERY SLOW** - Global pool (not stateful) |
| **0.2.1.1** | TCN (Complex) | ❌ | 379K | 128 | 6 | **512** | **8** | **32K** | <1 it/s | **STILL SLOW** - Complex arch ❌ |
| **0.2.1.3** | TCN (Ultra-Simple) | ❌ | 165,377 | 128 | 6 | **512** | **8** | **32K** | ~51 it/s | **FAST** - Global avg pool (not stateful) |
| **0.2.1.4** | TCN (Ultra-Simple) | ❌ | 165,377 | 128 | 6 | **512** | **8** | **32K** | ~? it/s | **DATA SPLIT FIX** - Same arch, better cells |
| **0.2.1.5** | Causal TCN Seq2Seq | ✅* | - | 128 | 6 | **512** | **8** | **32K** | TBD | **STATEFUL** - causal + seq2seq (slow) |
| **0.2.1.6** | Fast Causal TCN Seq2Seq | ✅* | - | 96 | 6 | **512** | **8** | **32K** | TBD | **FASTER** - lighter + TF32/cudnn |
| **0.2.1.7** | Fast Causal TCN Seq2Seq (smooth) | ✅* | - | 96 | 6 | **512** | **8** | **32K** | TBD | **SMOOTH LOSS** - superseded by 0.2.2.1 |
| **0.2.2.1 ⭐** | Fast Causal TCN Seq2Seq (hourly stats) | ✅* | - | 96 | 5 → 20 (hourly stats) | **168h** | **1h** | **~?** | TBD | **RESAMPLED** - larger RF + mean target |
| **0.3.1.0** | GRU Seq2Seq (hourly stats) | ✅ | - | 192 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **GRU BASELINE** |
| **0.3.1.1 ⭐** | GRU Seq2Seq (larger, hourly stats) | ✅ | - | 256 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **BEST GRU** - deeper head |
| **0.4.1.0** | CNN Seq2Seq (causal) | ✅* | - | 192 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **CNN BASELINE** |
| **0.4.1.1 ⭐** | CNN Seq2Seq (dilated causal) | ✅* | - | 224 | 5 → 20 (hourly stats) | **168h** | **24h** | **~?** | TBD | **BEST CNN** - dilations + smoother loss |

---

✅* = Stateful with causal buffer (no global pooling)

## Detailed Comparison

### 0.1.0.0 - LSTM Hybrid (Base)
- **Architecture**: Feature Projection → LSTM(2 layers) → MLP Head
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c, dU_dt, dI_dt
- **Embedding**: 64
- **Hidden Size**: 128
- **Layers**: 2 LSTM layers
- **Dropout**: 0.1
- **Stateful**: Yes (unidirectional)
- **Parameters**: ~280,000
- **Status**: ❌ Crashed during training (BatchNorm/derivatives issue)
- **Training Time**: N/A (crashed)

### 0.1.1.0 - LSTM Hybrid + Attention
- **Architecture**: Feature Projection (BN) → LSTM(2 layers) → Temporal Attention → MLP Head(3 layers)
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c (removed derivatives)
- **Embedding**: 96
- **Hidden Size**: 160
- **Layers**: 2 LSTM + 3 MLP
- **Dropout**: 0.15
- **Stateful**: Yes (unidirectional)
- **Parameters**: 414,210
- **Batch Size**: 96 (effective 192 with accum=2)
- **Learning Rate**: 0.00015 (warmup 5 epochs)
- **Epochs**: 200 (early stop 30)
- **Status**: 🟢 Running
- **Est. Training Time**: 4-5 hours

### 0.2.1.0 - TCN (Temporal Convolutional Network)
- **Architecture**: Feature Projection → TCN Blocks(4) → Global Pooling → MLP Head
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: 2048, **Stride**: 1
- **Embedding**: 96
- **TCN Channels**: [96, 128, 128, 128]
- **Kernel Size**: 3 (causal padding)
- **Dilation**: [1, 2, 4, 8] (exponential)
- **Dropout**: 0.15
- **Stateful**: ❌ No (global average pooling)
- **Parameters**: 379,169
- **Batch Size**: 128 (effective 256 with accum=2)
- **Learning Rate**: 0.0002
- **Epochs**: 200 (early stop 30)
- **Status**: 🔴 **TOO SLOW** (1.0 it/s, ~200h est.)
- **Issue**: Too many samples (647K batches)

### 0.1.1.1 - LSTM Hybrid + Attention (FAST)
- **Architecture**: Same as 0.1.1.0
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512** (was 2048), **Stride**: **8** (was 1)
- **Embedding**: 96
- **Hidden Size**: 160
- **Layers**: 2 LSTM + 3 MLP
- **Dropout**: 0.15
- **Stateful**: Yes (unidirectional)
- **Parameters**: 414,210
- **Batch Size**: **256** (was 96), **No accum**
- **Learning Rate**: 0.0002
- **Epochs**: 150 (early stop 25)
- **Batches**: **40,438** (was 862K) - **21x fewer!**
- **Status**: ❌ **POOR RESULTS** - R²=-0.30 @ epoch 12 (negative = bad)
- **Problem**: Validation cells (C07, C19, C21) only cover SOH 0.82-1.00, but training has 0.66-1.00
- **Result**: Data distribution mismatch → model confused

### 0.1.1.2 - LSTM Hybrid + Attention (IMPROVED DATA SPLIT) 🎯
- **Architecture**: Same as 0.1.1.1
- **Features**: Voltage, Current, Temp, EFC, Q_c, dU/dt, dI/dt (**7 features**)
- **Seq Length**: **512**, **Stride**: **8**
- **Embedding**: 96
- **Hidden Size**: 160
- **Layers**: 2 LSTM + 3 MLP
- **Dropout**: 0.15
- **Stateful**: Yes (unidirectional)
- **Parameters**: 414,210
- **Batch Size**: **256**, **No accum**
- **Learning Rate**: 0.0002
- **Epochs**: 150 (early stop 25)
- **Batches**: **~50,000** (more training data)
- **Status**: 🟡 Superseded (replaced by 0.1.1.3)
- **KEY CHANGES**:
  - ✅ **Train cells**: C01,C03,C05,**C09,C13**,C17,C23,**C27** (SOH 0.62-1.00 with more degraded cells!)
  - ✅ **Val cells**: C07,**C15**,C21 (SOH 0.63-1.00 - C15 is degraded!)
  - ✅ **Test cells**: C11,C19,C25 (SOH 0.65-1.00 - reserved for final test)
  - ✅ Reduced features (7 vs 16) - removed redundant/noisy features
  - ✅ Q_c kept (tracks cycle position), Q_m removed (98% correlated with Q_c)
  - ✅ C_Rate removed (100% redundant with Current)
  - ✅ SOC/Resistance_Approx removed (noisy/leaky)

### 0.1.1.3 - LSTM Seq2Seq (STATEFUL-ALIGNED) ✅
- **Architecture**: Feature Projection → LSTM(2 layers) → MLP Head (per timestep)
- **Features**: Voltage, Current, Temp, EFC, Q_c, dU/dt, dI/dt (7)
- **Seq Length**: **512**, **Stride**: **8**
- **Embedding**: 96
- **Hidden Size**: 160
- **Layers**: 2 LSTM + MLP
- **Dropout**: 0.15
- **Stateful**: Yes (unidirectional)
- **Loss**: **Seq2Seq** with warmup_steps (ignores first 32 steps)
- **Status**: 🔴 Stopped (val not improving)

### 0.1.1.4 - LSTM Seq2Seq (LONG SEQ, WIDE STRIDE)
- **Architecture**: Same as 0.1.1.3
- **Features**: Voltage, Current, Temp, EFC, Q_c, dU/dt, dI/dt (7)
- **Seq Length**: **2048**, **Stride**: **500**
- **Batch Size**: 128
- **Stateful**: Yes (unidirectional)
- **Status**: 🟡 Completed (validation still noisy)

### 0.1.1.5 - LSTM Seq2Seq (SMOOTH LOSS, NO DERIVS)
- **Architecture**: Same as 0.1.1.3
- **Features**: Voltage, Current, Temp, EFC, Q_c (5)
- **Seq Length**: **1024**, **Stride**: **128**
- **Batch Size**: 192
- **Loss**: Seq2Seq + **Smoothness penalty** on |y_t − y_{t-1}|
- **Stateful**: Yes (unidirectional)
- **Status**: 🔵 Planned / ready to run

### 0.1.2.1 - LSTM Seq2Seq (HOURLY SAMPLING)
- **Architecture**: Same as 0.1.1.5
- **Features**: Voltage, Current, Temp, EFC, Q_c (5)
- **Sampling**: **1h mean** per sample (Testtime[s] → hourly bins)
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Batch Size**: 256
- **Loss**: Seq2Seq + **Smoothness penalty**
- **MCU Fit**: One inference per hour, no per‑second buffer needed
- **Status**: 🟡 Superseded by 0.1.2.2

### 0.1.2.2 - LSTM Seq2Seq (HOURLY STATS)
- **Architecture**: Feature Projection → LSTM(2 layers) → MLP Head (per timestep)
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 160
- **Parameters**: ~394,145
- **Loss**: Seq2Seq + **Smoothness penalty**
- **MCU Fit**: One inference per hour, no per‑second buffer needed
- **Status**: 🟢 Completed (baseline for hourly LSTM)

### 0.1.2.3 - LSTM Seq2Seq (LARGER, HOURLY STATS) ✅ FINAL
- **Architecture**: Deeper feature projection → LSTM(3 layers) → Residual MLP blocks → Deeper MLP head
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 192
- **Loss**: Seq2Seq + **Smoothness penalty**
- **Status**: ✅ Final (best LSTM result)

### 0.3.1.0 - GRU Seq2Seq (HOURLY STATS)
- **Architecture**: Feature projection → GRU(3 layers) → Residual MLP blocks → Deeper MLP head
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 192
- **Status**: ✅ Completed (GRU baseline)

### 0.3.1.1 - GRU Seq2Seq (LARGER, HOURLY STATS) ⭐
- **Architecture**: Deeper GRU(4 layers) + larger embeddings → Residual MLP blocks → Deeper MLP head
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 256
- **Status**: ⭐ Best GRU so far

### 0.4.1.0 - CNN Seq2Seq (CAUSAL)
- **Architecture**: Causal CNN blocks → 1x1 Conv head (per timestep)
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 192
- **Status**: ✅ Completed (CNN baseline)

### 0.4.1.1 - CNN Seq2Seq (DILATED CAUSAL) ⭐
- **Architecture**: Dilated causal CNN blocks → 1x1 Conv head (per timestep)
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **last-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **24 hours**
- **Hidden Size**: 224
- **Status**: ⭐ Best CNN so far

### 0.2.1.3 - TCN (ULTRA-SIMPLE & FAST) 🎯
- **Architecture**: 1x1 Conv Embedding → Conv1d Stack → Global Pool → MLP Head
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512** (was 2048), **Stride**: **8** (was 1)
- **Hidden Size**: 128
- **TCN Layers**: 4 (dilations: 1, 2, 4, 8)
- **Kernel Size**: 3
- **Dropout**: 0.05
- **Stateful**: ❌ No (global pooling)
- **Parameters**: 165,377
- **Batch Size**: **256**, **No accum**
- **Learning Rate**: 0.0002
- **Epochs**: 150 (early stop 25)
- **Batches**: **~32,000** (was 647K) - **20x fewer!**
- **Status**: 🟡 Superseded (replaced by 0.2.1.6)
- **Key Changes**:
  - ✅ Removed complex feature projection with BatchNorm reshaping
  - ✅ Simple 1x1 conv for embedding (no B*T reshape bottleneck)
  - ✅ Removed double BatchNorm (was slowing down 0.2.1.1)
  - ✅ Simpler MLP head (2 layers like 2.1.0.0)
  - ✅ Based on working 2.1.0.0 architecture
-- **Speedup**: ~50x faster than 0.2.1.1!

### 0.2.1.4 - TCN (ULTRA-SIMPLE & FAST, NEW SPLIT)
- **Architecture**: Same as 0.2.1.3
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512**, **Stride**: **8**
- **Hidden Size**: 128
- **Stateful**: ❌ No (global average pooling)
- **Status**: 🔄 Replaced by 0.2.1.6 (kept for comparison)

### 0.2.1.5 - Causal TCN Seq2Seq (STATEFUL)
- **Architecture**: Causal dilated TCN (residual blocks) → 1x1 Conv head (per timestep)
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512**, **Stride**: **8**
- **Hidden Size**: 128
- **Stateful**: ✅* (streaming with buffer; no global pooling)
- **Loss**: **Seq2Seq** with warmup_steps = receptive_field - 1
- **Status**: 🟡 Slow in practice (replaced by 0.2.1.6)

### 0.2.1.6 - Fast Causal TCN Seq2Seq (STATEFUL)
- **Architecture**: Causal dilated TCN (lighter) → 1x1 Conv head (per timestep)
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512**, **Stride**: **8**
- **Hidden Size**: 96 (reduced)
- **Stateful**: ✅* (streaming with buffer)
- **Perf Flags**: TF32 + cudnn benchmark enabled
- **Status**: 🟡 Completed (val not improving)

### 0.2.1.7 - Fast Causal TCN Seq2Seq (SMOOTH LOSS)
- **Architecture**: Same as 0.2.1.6
- **Features**: Testtime, Voltage, Current, Temp, EFC, Q_c
- **Seq Length**: **512**, **Stride**: **8**
- **Hidden Size**: 96
- **Loss**: Seq2Seq + **Smoothness penalty** on |y_t − y_{t-1}|
- **Split**: Updated train/val/test (closer SOH distribution)
- **Status**: 🟡 Superseded by 0.2.2.1

### 0.2.2.1 - Fast Causal TCN Seq2Seq (HOURLY STATS)
- **Architecture**: Causal dilated TCN (residual blocks) → 1x1 Conv head (per timestep)
- **Features**: Voltage, Current, Temp, EFC, Q_c (5) → hourly stats (mean/std/min/max = 20)
- **Sampling**: **1h stats**, target = **mean-of-hour**
- **Seq Length**: **168 hours**, **Stride**: **1 hour**
- **Kernel Size / Dilations**: k=5, [1,2,4,8,16,32] (larger RF)
- **Hidden Size**: 96
- **Loss**: Seq2Seq + **Smoothness penalty**
- **Status**: 🟢 Running (resampled TCN approach)

---

## Feature Comparison

| Feature | 0.1.0.0 | 0.1.1.0 | 0.2.1.0 |
|---------|---------|---------|---------|
| Testtime[s] | ✅ | ✅ | ✅ |
| Voltage[V] | ✅ | ✅ | ✅ |
| Current[A] | ✅ | ✅ | ✅ |
| Temperature[°C] | ✅ | ✅ | ✅ |
| EFC | ✅ | ✅ | ✅ |
| Q_c | ✅ | ✅ | ✅ |
| dU_dt[V/s] | ✅ | ❌ | ❌ |
| dI_dt[A/s] | ✅ | ❌ | ❌ |

---

## Performance Targets

| Metric | Excellent | Good | Acceptable |
|--------|-----------|------|------------|
| MAE | < 0.015 | < 0.025 | < 0.040 |
| RMSE | < 0.020 | < 0.035 | < 0.050 |
| R² | > 0.98 | > 0.95 | > 0.90 |

---

## Training Configuration

| Config | 0.1.0.0 | 0.1.1.0 | 0.1.1.1 | 0.2.1.0 | 0.2.1.1 |
|--------|---------|---------|---------|---------|---------|
| Seq Length | 2048 | 2048 | **512** | 2048 | **512** |
| Window Stride | 1 | 1 | **8** | 1 | **8** |
| Batch Size | 128 | 96 | **256** | 128 | **256** |
| Accum Steps | 1 | 2 | 1 | 2 | 1 |
| Effective Batch | 128 | 192 | 256 | 256 | 256 |
| Base LR | 0.0002 | 0.00015 | 0.0002 | 0.0002 | 0.0003 |
| Weight Decay | 0.0001 | 0.00015 | 0.00015 | 0.0001 | 0.0001 |
| Warmup Epochs | 0 | 5 | 3 | 5 | 3 |
| Scheduler | CosineWarmRestart | CosineWarmRestart | CosineWarmRestart | CosineWarmRestart | CosineWarmRestart |
| Max Epochs | 150 | 200 | 150 | 200 | 150 |
| Early Stop | 25 | 30 | 25 | 30 | 25 |
| **Total Batches** | ~860K | 862K | **40K** | 647K | **32K** |
| **Training Speed** | - | 7.4 it/s | **~37 it/s** | 1.0 it/s | **~51 it/s** |
| **Est. Time/Epoch** | - | ~32h | **~18min** | ~180h | **~10min** |

---

## Training Configuration (Fast Runs)

| Config | 0.1.1.2 | 0.1.1.3 | 0.1.1.4 | 0.1.1.5 | 0.1.2.1 | 0.2.1.3 | 0.2.1.4 | 0.2.1.5 | 0.2.1.6 | 0.2.1.7 |
|--------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| Seq Length | 512 | 512 | 2048 | 1024 | **168h** | 512 | 512 | 512 | 512 | 512 |
| Window Stride | 8 | 8 | 500 | 128 | **24h** | 8 | 8 | 8 | 8 | 8 |
| Batch Size | 256 | 256 | 128 | 192 | 256 | 256 | 256 | 256 | 256 | 256 |
| Loss Mode | last | **seq2seq** | **seq2seq** | **seq2seq** | **seq2seq** | last | last | **seq2seq** | **seq2seq** | **seq2seq** |
| Warmup Steps | 0 | 32 | 32 | 32 | 8 | 0 | 0 | auto | auto | auto |
| Hidden Size | 160 | 160 | 160 | 160 | 160 | 128 | 128 | 128 | 96 | 96 |
| Stateful | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | ❌ | ✅* | ✅* | ✅* |

---

## STM32 Deployment Considerations

| Aspect | LSTM (0.1.x.x) | TCN (0.2.x.x) |
|--------|----------------|---------------|
| **Inference Speed** | Slower (sequential) | **Faster** (parallel) |
| **Memory (RAM)** | Higher (hidden states) | **Lower** (fixed buffers) |
| **Flash Size** | ~1.5MB | **~1.2MB** |
| **Latency** | ~10ms | **~3ms** |
| **Ease of Impl.** | Complex (LSTM ops) | **Simple** (1D conv) |
| **Stateful Mode** | Supported | **Supported** (causal buffer) |

**Winner for STM32: TCN (0.2.x.x)** ✅

---

## Version Naming Convention

**Format: `X.Y.Z.W`**

- **X (Major)**: Task type (0=SOH only, 1=ECM+SOH, 2=future)
- **Y (Architecture)**: Network type (1=LSTM, 2=TCN, 3=CNN, 4=Transformer, etc.)
- **Z (Variant)**: Architecture modifications
- **W (Hyperparams)**: Same arch, different hyperparameters

**Examples:**
- `0.1.0.0` = SOH, LSTM, base variant, default params
- `0.1.1.0` = SOH, LSTM, improved variant (attention), default params
- `0.1.1.1` = SOH, LSTM, improved variant, tuned params
- `0.2.1.0` = SOH, TCN, base variant, default params

---

## Results Summary (to be updated after training)

| Version | Best Epoch | Val RMSE | Val MAE | Val R² | Est. Time | Status |
|---------|------------|----------|---------|--------|-----------|--------|
| 0.1.0.0 | - | - | - | - | - | ❌ Crashed |
| 0.1.1.0 | TBD | TBD | TBD | TBD | ~100h | 🔴 Too slow |
| **0.1.1.1** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Superseded |
| **0.1.1.2** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Superseded |
| **0.1.1.3** | TBD | TBD | TBD | TBD | **~5h** | 🔴 Stopped |
| **0.1.1.4** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Completed |
| **0.1.1.5** | TBD | TBD | TBD | TBD | **~5h** | 🔵 Planned |
| **0.1.2.1** | TBD | TBD | TBD | TBD | **~5h** | 🔵 Prepared |
| 0.2.1.0 | TBD | TBD | TBD | TBD | ~300h | 🔴 Too slow |
| 0.2.1.1 | TBD | TBD | TBD | TBD | ~100h+ | 🔴 **Too slow** (still epoch 1) |
| **0.2.1.3** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Superseded |
| **0.2.1.4** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Superseded |
| **0.2.1.5** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Stopped |
| **0.2.1.6** | TBD | TBD | TBD | TBD | **~5h** | 🟡 Completed |
| **0.2.1.7** | TBD | TBD | TBD | TBD | **~5h** | 🟢 Running |

---

## Recommendations

1. **PRIMARY**: Use **0.1.1.3** (LSTM seq2seq) + **0.2.1.6** (fast causal TCN) for current runs.
2. **For Best Accuracy**: Start with 0.1.1.3 (stateful-aligned seq2seq).
3. **For STM32 Deployment**: Prefer 0.2.1.6 (causal TCN, smaller + faster).
4. **OLD versions**: ❌ Stop 0.2.1.1 (too slow, complex architecture).
5. **Planned**: 0.1.2.1 with **target downsampling** to reduce jitter (evaluate later).
6. **Prepared**: 0.1.2.1 hourly sampling (1h means) for MCU-friendly inference.

### Key Insights

**Problem with 0.2.1.1**:
- Complex feature projection with `reshape(B*T, F)` + BatchNorm1d = **huge bottleneck**
- Double BatchNorm (feature_proj + CausalConv1d) = **unnecessary overhead**
- Complex 3-layer MLP head = **slower**
- Result: <1 it/s, ~100+ hours training time ❌

**Speed Solution in 0.2.1.3/0.2.1.4**:
- Simple 1x1 conv for embedding (parallel, **no reshape**)
- Single BatchNorm removed (use ReLU only)
- 2-layer MLP head (like working 2.1.0.0)
- Result: ~51 it/s, ~5 hours training time ✅
- **Note**: 0.2.1.3/0.2.1.4 are **not stateful** (global pooling)

**Stateful Solution in 0.2.1.5/0.2.1.6**:
- Causal dilated TCN + seq2seq head (no global pooling)
- Streaming possible with buffer of receptive_field-1 samples
- 0.2.1.6 reduces hidden/layers + enables TF32 for speed

**Architecture Comparison**:
- 0.2.1.1: 379K params, complex, **<1 it/s**
- 0.2.1.3: 165,377 params, simple, **~51 it/s** (fast, not stateful)
- 0.2.1.6: lighter causal TCN (stateful, faster than 0.2.1.5)

---

**Last Updated**: 2026-01-08 12:35
**Problem with 0.1.1.0 / 0.2.1.0**: 
- `seq_chunk_size=2048` + `window_stride=1` = way too many overlapping samples
- Result: 800K+ batches → 100+ hours training time

**Solution in 0.1.1.2+ / 0.2.1.3+**:
- `seq_chunk_size=512` + `window_stride=8` = 40K batches
- Result: ~2 hours training time (50-100x faster!)
- **Quality**: Should be similar or better (less overfitting on similar samples)
