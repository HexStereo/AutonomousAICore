# Autonomous Optimization Controller & Closed-Loop Telemetry

## 1. Motivation
Deep neural network pretraining often suffers from catastrophic failure modes:
- Sudden loss spikes and loss divergence.
- Unstable gradient dynamics ($L_2$ explosion or vanishing).
- Numerical underflow/overflow yielding NaNs or Infs.
- Hardware Out-Of-Memory (OOM) due to sequence length variance or memory fragmentation.

In traditional frameworks, human intervention is required to stop, inspect, and restart training with lower learning rates or smaller batch sizes. The **Autonomous Optimization Controller** implements automated, closed-loop supervision, anomaly detection, and self-healing actions.

---

## 2. Telemetry Tracker Mathematics

The `TelemetryTracker` (`include/autonomous/telemetry.hpp`) maintains continuous, online statistical counters:

### A. Dual Exponential Moving Averages (EMA)
To distinguish between normal batch-level noise and true divergence, dual EMAs are tracked:
$$\text{EMA}_{\text{short}, t} = \alpha_{\text{short}} \cdot \text{EMA}_{\text{short}, t-1} + (1 - \alpha_{\text{short}}) \cdot \mathcal{L}_t \quad (\alpha_{\text{short}} = 0.8)$$
$$\text{EMA}_{\text{long}, t} = \alpha_{\text{long}} \cdot \text{EMA}_{\text{long}, t-1} + (1 - \alpha_{\text{long}}) \cdot \mathcal{L}_t \quad (\alpha_{\text{long}} = 0.98)$$

### B. Rolling Loss Variance
Across a sliding window of size $W = 50$:
$$\mu = \frac{1}{W} \sum_{i=1}^W \mathcal{L}_i, \quad \sigma^2 = \frac{1}{W} \sum_{i=1}^W \mathcal{L}_i^2 - \mu^2$$

### C. Computational Throughput (TFLOP/s)
Using the standard forward + backward Transformer FLOP formulation ($6P$ operations per token for a model with $P$ parameters):
$$\text{FLOPs} = 6.0 \times P \times \text{tokens}$$
$$\text{Throughput (TFLOP/s)} = \frac{\text{FLOPs}}{\Delta t_{\text{sec}} \times 10^{12}}$$

### D. Hardware Memory Utilization
- **Host RAM**: Queried via Win32 `GlobalMemoryStatusEx`.
- **GPU VRAM**: Live query via `cudaMemGetInfo(&free_bytes, &total_bytes)`.

---

## 3. Anomaly Taxonomy & Detection Rules

| Anomaly | Trigger Condition | Severity |
|---|---|---|
| `NanOrInfLoss` | `std::isnan(loss) || std::isinf(loss)` | **Critical** |
| `NanOrInfGradient` | `std::isnan(grad_norm) || std::isinf(grad_norm)` | **Critical** |
| `LossDivergence` | $\mathcal{L}_t > 1.4 \times \text{EMA}_{\text{long}}$ for $M \ge 3$ consecutive steps | **High** |
| `GradientExplosion` | $\|g\|_2 > 4.0 \times \text{EMA}(\|g\|_2)$ | **Medium** |
| `GradientVanishing` | $\|g\|_2 < 10^{-7}$ | **Medium** |
| `MemoryCritical` | Free GPU VRAM ratio $< 8\%$ | **High** |

---

## 4. Closed-Loop Self-Healing Policies

```
             +-----------------------+
             |   HealthState::Optimal|
             +-----------------------+
                         |
           [Anomaly Detected: Spike / Memory]
                         v
             +-----------------------+
             |   HealthState::Warning| ---> [Action: TightenClipping / ScaleLR / TuneGradAccum]
             +-----------------------+
                         |
           [Anomaly Detected: NaN / Divergence]
                         v
             +-----------------------+
             |  HealthState::Diverging|
             +-----------------------+
                         |
           [Rollback to Last Stable Checkpoint]
                         v
             +-----------------------+
             | HealthState::Recovering| ---> [Action: Rollback + LR Decay 0.5x + Skip Batch]
             +-----------------------+
                         |
           [If Retries > max_retries]
                         v
             +-----------------------+
             |   HealthState::Halted | ---> [Emergency Safety Halt]
             +-----------------------+
```

### Self-Healing Actions:
1. **RollbackCheckpoint**:
   - Model weights restored from last validated stable checkpoint (`.aic`).
   - Optimizer learning rate halved: $\eta \leftarrow \eta \times 0.5$.
   - Dataloader advances to skip the bad batch to prevent re-entering numerical instability.
   - Health state transitions to `Recovering`.
2. **TightenGradientClipping**:
   - Dynamically lowers clipping threshold: $\text{clip} \leftarrow \max(\text{clip}_{\min}, \text{clip} \times 0.7)$.
   - Prevents destabilizing weight updates from gradient spikes.
3. **ScaleLearningRate**:
   - Decays optimizer learning rate smoothly when high loss variance is detected.
4. **TuneGradientAccumulation**:
   - If memory pressure is detected ($< 8\%$ free), decreases `grad_accum_steps`.
   - If memory headroom is abundant ($> 50\%$ free) and throughput can be improved, increases `grad_accum_steps` to maximize Blackwell SM occupancy.
