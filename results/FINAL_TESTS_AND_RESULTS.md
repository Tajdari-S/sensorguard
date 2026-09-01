# SensorGuard: final tests and results

This is the single paper-facing inventory of completed SensorGuard and Rahman
experiments. Development results must not be described as sealed or independent
final tests.

## Detector and decision rule

The primary implementations are 400-tree random forests trained separately on
1 Hz NVML features, 10 kS/s GPU-current features, and their feature-level
concatenation. Every headline comparison uses causal 30-second windows, a
15-second stride, probability threshold 0.85, and the fixed 3-of-5 run rule.
A calibrated logistic late-fusion diagnostic combines frozen NVML and current
probabilities. CUSUM is implemented as a secondary causal diagnostic; its old
comparison is invalid because it used the mismatched model lineage and will be
rerun on corrected frozen probabilities before any paper claim is made.

## GPU-ordinal validity audit

The original synchronized-30 runner used CUDA ordinal 1 while its NVML logger
and mapped current probe monitored physical GPU1. Once failed physical GPU0
was omitted from CUDA enumeration, CUDA ordinal 1 resolved to physical GPU2.
The audit confirmed that the recorded NVML utilization was exactly 0% for
every original family, including ordinary training. Therefore all results
derived from that synchronized-30 frozen model are invalid: the synchronized
30-run metrics, sealed fused-update scores, CUSUM comparison, application
privacy score, simulated health score, hard-family predictions, broad negative
predictions, and minimum retained sensor decision. Their raw hard-family and
broad-negative traces remain usable and will be rescored with corrected
models. Replacement collection pins physical UUIDs and uses CUDA ordinal zero
inside that isolated device list.

## Completed evidence

| Evidence | Setup and protocol | Result | Paper interpretation |
| --- | --- | --- | --- |
| Rahman NVML corpus | 1,396 classifier-eligible runs; nine NVIDIA GPU models; grouped 30-second evaluation | 98.21% pooled accuracy and 98.20% macro-F1; committed unseen-family variants range from 11.1% to 100% detection | Prior-state-of-the-art motivation, not a SensorGuard result |
| Rahman fixed run rule | 1,302 runs; threshold 0.75 and 3-of-5 rule | 731/760 training alerts, 5/542 false-alert runs, 0.059 alerts/GPU-hour, median alert 60 s | Prior reference only; its threshold differs from SensorGuard's 0.85 threshold |
| Matched 36-run physical cohort (development) | RTX 3090; 26 training and 10 non-training runs; leave-one-family-out | NVML 18/26 TP; current 24/26 TP; feature fusion 22/26 TP; every method 0/10 FP | GPU current is useful on this matched cohort, but fusion is not automatically better |
| Sensor-subset ablation (development) | Eight NVML-anchored subsets on the same 36 runs | Only GPU current changes decisions: NVML+current detects 22/26 versus 18/26 for NVML. Motherboard current and UltraMic add zero detections | Retain NVML and one GPU-current channel; do not retain motherboard current or UltraMic |
| Extended 61-run adaptive stress cohort (development) | 51 training and 10 non-training runs | NVML 44/51 TP and 0/10 FP; current 42/51 TP and 2/10 FP | Current can be worse under adaptive development; do not use this cohort as a sealed claim |
| Original synchronized 30-run cohort | Five training and five control families, three repetitions each | **INVALIDATED:** workload ran on physical GPU2 while NVML/current monitored GPU1 | Replaced by UUID-isolated collection |
| Original sealed fused-update test | Three fused-update, three AdamW, and nine controls | **INVALIDATED:** same workload/sensor GPU mismatch and invalid frozen model | Recollect after corrected freeze |
| Fresh fused-recovery validation | Three new shape/seed variants | **INVALIDATED MODEL LINEAGE:** recovery model was derived from mismatched synchronized data | Do not use in paper |
| Rahman hard-family RTX replay | Seven attack families x three plus one inference control x three; 24 synchronized 300-second runs | Raw collection valid; **old predictions invalid** because the frozen models came from the mismatched corpus | Rescore raw traces with corrected models |
| Hard-family contract audit | Frozen hashes, features, and scores checked | Reproduction audit valid for the old artifacts, but those artifacts are now invalid | Repeat against corrected frozen hashes |
| Broad NVML negative exposure | 121 healthy node2 runs; 60.581 GPU-hours | Raw exposure valid; **99/121 old score is invalid** because the detector artifact came from mismatched data | Rescore with corrected NVML artifact |
| New synchronized physical negative exposure | GPU1 and mapped current channel; GEMM, FFT, memcpy, and idle; three 300-second repetitions each | **PENDING_PHYSICAL_NEGATIVE_RESULT** | Independent repeated controls from known negative families; not an external benign-family test |
| Simulated sensor-health tests | Probability-trace fault injection | **INVALIDATED MODEL LINEAGE** | Repeat on corrected probabilities; still label as simulation |
| Privacy diagnostic | Current features; leave-one-repetition-out | **INVALIDATED:** workload and current channel were not aligned | Repeat on corrected traces |
| CUSUM diagnostic | Calibration on original development repetitions | **INVALIDATED MODEL LINEAGE** | Repeat on corrected probabilities |
| Application roofline | Five RTX 3090 applications x three repetitions | Training and inference overlap in normalized arithmetic intensity and throughput | Motivation only; roofline is not a training detector |
| WAVE and base-logger overhead | Same six GPT-2, LLaMA, and Qwen configurations | WAVE mean runtime multiplier 28.497x; NVML+DCGM base logger 1.050x | SensorGuard's always-on telemetry is much less expensive than profiler-based verification |
| Full-monitor overhead | Five alternating paired repetitions; NVML, DCGM, and six PicoScopes stream concurrently | Mean useful-work penalty -0.181%, 95% CI [-0.548%, 0.186%]; all six scopes present and no overflow | No slowdown is resolved; do not describe the negative point estimate as a speedup |
| Minimum sensor specification | Original sampling/quantization replay | **INVALIDATED:** baseline decisions came from mismatched traces | Recompute on corrected synchronized traces |
| Hardware acquisition health | PicoScope 2204A serial 12789/2929 | Mean effective sampling 9,991.99 samples/s; 30/30 captures at 100 us interval, zero overflow, less than 1% clipping; native 8-bit device | Hardware capability is confirmed, but the retained decision point must be recomputed |

The PicoScope model and serial were read directly through the installed driver.
Pico Technology's official
[PicoScope 2000 Series data sheet](https://www.picotech.com/download/datasheets/picoscope-2000-series-data-sheet-en.pdf)
specifies 8-bit vertical resolution for the 2204A.

## Threat model and tested adversaries

### Adversary capability and trust boundary

The evaluated adversary is a GPU tenant who wants to perform real optimization
while being classified as inference or non-ML work. The tenant controls the
model and workload code, PyTorch and custom CUDA behavior, batch and sequence
size, precision, gradient accumulation, scheduling, duty cycle, workload
interleaving, and tenant-accessible clock or power requests. In the stronger
tests, the tenant also knows the feature vocabulary and an earlier detector,
may train a surrogate, and may sacrifice throughput for evasion. A positive
training label is accepted only when the workload records nonzero weight change
and meaningful loss reduction; merely running backward-shaped matrix
operations is not labelled training.

The tenant cannot modify the host driver or operating system, the verifier's
logger, the experiment schedule, the PicoScope trace, or the physical mapping
from the GPU rail to the scope channel. NVML is therefore *shaped, not forged*:
the tenant may arrange computation so that authentic 1 Hz counters resemble an
allowed workload, but may not replace those counters with invented values. GPU
current provides an independently collected, 10 kS/s view of the same device;
it is not assumed to reveal the semantics of a memory write.

The following table states what was actually executed rather than treating all
red-team ideas as equivalent evidence. A machine-readable version is in
[threat-model-test-matrix.csv](tables/threat-model-test-matrix.csv).

| Adversary/test | Knowledge and action | Executed workload and controls | Origin and status |
| --- | --- | --- | --- |
| Native timing and resource shaping | Chooses duty cycle, mixes training with inference, reduces the training microbatch, or migrates work between GPUs | Ordinary SGD, duty-shaped SGD, interleaved SGD/inference, memory-minimal SGD, and migration; matched GEMM, FFT, memcpy, inference, and idle controls; three repetitions per family | Designed and implemented for SensorGuard in [development_evasion_workload.py](../scripts/workloads/development_evasion_workload.py); corrected UUID-isolated collection is in progress |
| Held-out Rahman families | Uses previously unseen utilization, temporal, mimicry, composite, or dilution strategies; white-box variants know the earlier NVML detector | Low utilization B, temporal disruption D, 30% interleaving F, composite idle dilution, and white-box LoRA dilution N=5/10/20; white-box inference is the negative control; three 300-second repetitions per family | Downloaded unchanged workload logic from [Rahman's public GPU-monitoring repository](https://github.com/robirahman/GPU-monitoring) and scheduled by [build_hard_family_plan.py](../scripts/experiments/build_hard_family_plan.py). Script SHA-256 prefixes: `80a1d3ee`, `7bf1f560`, `5b9acda8`, `80f583f4`. Raw RTX 3090 traces are complete and await scoring by the corrected frozen models |
| Custom-kernel semantic evasion | Knows that aggregate activity is being measured and removes a separately visible optimizer phase | A custom autograd layer computes forward work and gradients and updates weights in-place; ordinary AdamW is a positive control, while one-forward, three-forward, and gradient-like dummy writes are matched negatives | SensorGuard-native adapter in [fused_update_workload.py](../scripts/workloads/fused_update_workload.py), inspired by Rahman's symmetric-GEMM/TinyLlama experiment. The corrected post-freeze run is a confirmation test, not an untouched attack family |
| Telemetry-channel faults | Corrupts an already produced detector stream | Drop, freeze, bias/scale, time shift, benign-control replay, and current-channel swap are applied to recorded probability traces | Designed for SensorGuard and explicitly reported as simulation; it is not a physical cable, logger, or attestation attack |

Rahman et al.'s TIGAR study introduced 20 families over five red-team rounds:
utilization manipulation; memory reduction and workload mimicry;
training-as-inference; composite and multi-GPU evasion; and white-box attacks.
Its evaluation trained on all other families and tested the held-out family,
which is the source of our unseen-family criterion. We do **not** claim to
re-run all 20 families with current sensing: the synchronized physical replay
uses the seven difficult families listed above because they include the
reported low-detection cases. Cite the TIGAR paper as `rahman2026hidden` and
the public code repository for the downloaded implementations.

### Out-of-scope adversaries

A privileged hardware operator who controls the driver, verifier, logger, or
physical sensors is outside the demonstrated threat model. The probability-
trace injections do not prove resistance to an operator who rewires a probe or
injects authenticated-looking samples. That setting requires a sealed shunt
and ADC, authenticated timestamps and transport, channel-identity attestation,
and tamper evidence; the current prototype does not implement those guarantees.
Thermal cameras, thermistors, a network tap, RF, and visible-light sensing are
also outside the headline classifier because no synchronized labelled test has
shown that they improve held-out-family decisions. Visible light is intended
for physical-integrity evidence, not workload classification.

### Paper-ready threat-model text

```latex
\paragraph{Threat model.}
We consider a tenant who performs real parameter updates while attempting to
appear as inference or non-ML work. The tenant controls workload and kernel
code, framework configuration, batch size, precision, gradient accumulation,
scheduling, duty cycle, workload interleaving, and tenant-accessible clock and
power requests. In our strongest tests, the tenant also knows the feature
vocabulary and an earlier detector and may trade throughput for evasion. The
tenant does not control the host driver, verifier, synchronized logger, or
physical probe mapping. Thus NVML may be \emph{shaped} by workload choice but
not forged, while the independently logged GPU-current trace cannot be
replaced by tenant software.

Our native development suite executes ordinary SGD, duty-shaped training,
training interleaved with inference, memory-minimal microbatch training, and
GPU migration, with matched GEMM, FFT, memcpy, inference, and idle controls.
Every positive workload records nonzero weight change and loss reduction. We
also replay seven difficult families from Rahman et al.~\cite{rahman2026hidden}
using their public implementations\footnote{\url{https://github.com/robirahman/GPU-monitoring}}:
low-utilization training, temporal
disruption, 30\% interleaving, composite dilution, and white-box LoRA dilution
at three strengths, together with a white-box inference control. These attacks
are downloaded from the authors' repository, not designed by us. Finally, we
design a custom fused-update attack that computes gradients and updates weights
in-place, with AdamW as a positive control and forward, FLOP-matched
three-forward, and gradient-like scratch writes as negatives.

Rahman et al. evaluate 20 families across utilization manipulation, memory
reduction and workload mimicry, training-as-inference, composite and multi-GPU
evasion, and white-box attacks. We do not claim a current-sensor replication of
all 20: our synchronized replay selects seven reported hard families and tests
them with a detector fitted and frozen only on separate development workloads;
the hard-family traces are not used for fitting, calibration, or threshold
selection, but the families were selected before this corrected analysis and
are therefore not described as an untouched sealed set.
Missing, frozen, biased, shifted, replayed, and swapped telemetry channels are
evaluated only by injecting faults into recorded probability traces; these are
software simulations, not physical tamper tests. A privileged operator who can
alter the driver, verifier, logger, or sensor wiring is out of scope and would
require authenticated telemetry, channel-identity attestation, and tamper-
evident sensing hardware.
```

## What the results support

1. In the matched 36-run development cohort, GPU current detects 24/26
   training runs versus 18/26 for NVML; both have 0/10 false-alert runs.
2. In the same cohort, GPU current is the only tested added physical modality
   that changes run decisions. Motherboard current and UltraMic add zero
   detections. NVML plus current detects 22/26, so fusion is not claimed to
   dominate current alone.
3. Across five paired repetitions, running NVML, DCGM, and six PicoScopes
   together has a -0.181% mean point estimate with 95% CI [-0.548%, 0.186%].
   This supports "no resolved slowdown," not a speedup claim.
4. The PicoScope 2204A channel acquires at approximately 10 kS/s without
   overflow in the audited captures and has native 8-bit resolution. The
   minimum decision-preserving sample rate remains pending corrected replay.
5. Worst-family gain, corrected hard-family detection, fused-update detection,
   false alerts/GPU-hour, time to alert, privacy leakage, simulated fault
   detection, and the minimum retained sample rate remain pending the
   UUID-isolated replacement analysis.

## What must not be claimed

- Do not claim zero false alerts in general. Zero is observed only in specified
  matched cohorts.
- Do not state a worst-family gain until the corrected hard-family scoring is
  complete.
- Do not claim fused-update detection until the corrected post-freeze
  confirmation is complete.
- Do not call the system privacy-preserving; corrected leakage evaluation is
  pending, and current traces are inherently a workload side channel.
- Do not call simulated trace faults physical tamper tests.
- Do not claim that motherboard current, ultrasound, thermal, network, RF, or
  visible-light sensing improves the detector.

## Abstract text to add

**PENDING_FINAL_ABSTRACT_TEXT**

## Introduction text to add

**PENDING_FINAL_INTRODUCTION_TEXT**

## Primary artifact locations

- [Matched 36 and sensor subsets](tables/)
- [Synchronized development and sealed fused update](current_paper_physical_20260831/)
- [Hard-family evaluation, late fusion, health, and privacy](final_paper_completion/hard_family_evaluation/)
- [Hard-family integrity audit](final_paper_completion/hard_family_audit/)
- [Broad frozen NVML negative exposure](final_paper_completion/negative_exposure/)
- [Full-monitor overhead](final_paper_completion/full_monitor_overhead/)
- [Hardware sensor specification](final_paper_completion/hardware_sensor_spec/)
- [Fused-update recovery](../next_paper/FUSED_UPDATE_RECOVERY.md)
- [Rahman/SensorGuard evidence inventory](tables/combined_evidence_inventory.csv)
- [Threat-model test matrix](tables/threat-model-test-matrix.csv)
