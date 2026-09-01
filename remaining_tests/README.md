# Current-paper tests and results

Updated 2026-09-01. This is the single short status note for the paper. It uses
all relevant committed evidence from `robirahman/GPU-monitoring` and
`Tajdari-S/sensorguard` without altering or duplicating Robi's data. The
machine-readable provenance and rerun decisions are in
[`../results/tables/combined_evidence_inventory.csv`](../results/tables/combined_evidence_inventory.csv).

## Detector being evaluated

The primary comparison is **NVML alone vs GPU current alone vs NVML + GPU
current**. All three must use the same causal features, run IDs, grouped or
held-out-family splits, 30-second windows, 15-second stride, fixed probability
threshold, and run rule. The current fixed operating point is probability
`0.85` with a 3-of-5 rule. CUSUM is reported as a secondary decision rule on
the same frozen probabilities; it is not allowed to use a different test set.

Thermal, acoustic, motherboard-current, network, RF, and visible-light inputs
are not part of the headline system unless a new synchronized test shows an
incremental gain on a family that NVML + GPU current misses. Visible light is
for physical-integrity/tamper monitoring, not workload classification.

## Completed results from Robi Rahman

These results were already present in Robi's repository or the NeurIPS
author-response material. We reuse the files without modifying them.

- **Large NVML corpus:** 1,404 runs were collected across nine NVIDIA GPU
  models and four architecture generations; 1,396 runs are classifier-eligible.
  A random forest reaches **98.21% accuracy and 98.20% macro-F1** on 30-second
  windows under run-grouped cross-validation. This is the broad prior NVML
  result, not a held-family SensorGuard result.
- **Prior fixed run rule:** on 760 training and 542 non-training runs, 30-second
  windows, 15-second stride, threshold 0.75, and the 3-of-5 rule alert on
  **731/760 training runs** and **5/542 non-training runs**. The reported rate
  is **0.059 false alerts/GPU-hour**; median time to alert is **60 seconds** and
  the maximum is **225 seconds**. Because our current threshold is 0.85, these
  values remain a separate prior reference.
- **Unseen NVML attack families:** exact committed window-detection rates for
  difficult held families include low-utilization **60.6%**, temporal
  disruption **67.9%**, 30% training/inference interleaving **64.3%**, one
  composite attack **50.0%**, and white-box LoRA variants **61.1%**, **66.7%**,
  and **11.1%**. These results explain why pooled 98.21% accuracy is not a
  robustness guarantee.
- **RTX 3090 physical-sensor pilot:** the completed channels are the GPU-current
  clamp, motherboard-current clamp, and Dodotronic UltraMic. Under five-fold
  `StratifiedGroupKFold` grouped by run ID, GPU current obtains **99.2%, 100%,
  and 100% window accuracy** at 15, 30, and 60 seconds. Motherboard current
  obtains **77.7%, 74.5%, and 66.4%**; UltraMic obtains **47.5%, 60.6%, and
  55.1%**. GPU plus motherboard current and the full electrical-plus-acoustic
  group give the same window accuracies as GPU current alone. This small,
  single-host, run-grouped pilot is proof of signal, not unseen-family evidence.
- **Physical red-team development:** gradient accumulation, randomized batch
  timing, and gap-filling ballast were caught without large throughput loss.
  Continuous co-filling remained caught at about **105x slowdown**; a combined
  attack barely evaded at **112x slowdown**; dominant ballast dilution evaded
  at about **50x slowdown**. After the detector was hardened using attack
  examples, fresh dilution and cuFFT-mimicry variants were caught. These are
  adaptive development rounds, not a sealed final attack test.
- **Hardware scope:** Robi's large software-telemetry corpus includes H200 and
  other GPU models, but the completed physical pilot is RTX 3090. We have no
  H200 GPU-current SensorGuard result and do not present one.

## Completed results from this SensorGuard work

- **Matched 36-run comparison:** the same 26 training and 10 non-training RTX
  3090 runs are scored with leave-one-workload-family-out evaluation, threshold
  0.85, and the 3-of-5 rule. NVML detects **18/26**, GPU current **24/26**, and
  NVML plus current **22/26**; all three produce **0/10 false-alert runs**. This
  is development evidence and shows that fusion is not automatically better.
- **All completed sensor subsets:** we tested the GPU-current clamp,
  motherboard-current clamp, and UltraMic individually and in physical-only
  groups, then evaluated all eight NVML-anchored subsets on the same 36 runs.
  NVML plus GPU current detects **22/26**, compared with **18/26** for NVML, a
  **15.4-point gain** with paired-bootstrap 95% interval **3.85--30.77 points**.
  NVML plus motherboard current, NVML plus UltraMic, and NVML plus both make
  exactly the same decisions as NVML. Every subset containing GPU current makes
  the same decisions as NVML plus GPU current. Therefore the minimum useful
  added sensor in this cohort is GPU current.
- **Extended 61-run adaptive stress test:** NVML detects **44/51** training runs
  with **0/10** false alerts, while current alone detects **42/51** with **2/10**
  false alerts. This prevents a claim that current always outperforms NVML.
- **Two-seen/one-unseen transfer:** models trained with timing shaping,
  interleaving, or both detect **0/3** disjoint-GPU memory-minimal training runs
  at threshold 0.85, despite window AUCs of **0.968, 0.992, and 0.986**. There
  are **0 alerts over 0.843 control GPU-hours**. High AUC therefore does not
  imply transfer at the fixed run-level operating point.
- **Independent synchronized 30-run corpus:** all 30 node/verifier pairs are
  valid on node1 RTX 3090 GPU1 with PicoScope `12789/2929` channel A at 10 kS/s.
  NVML detects **0/15** training runs with **2/15** false-alert runs; current and
  NVML-plus-current each detect **6/15** with **0/15** false alerts. Current
  detects duty shaping and GPU migration but misses ordinary, interleaved, and
  memory-minimal training. Overall gain is **40 percentage points**, but
  worst-family detection is 0% for every method, so worst-family gain is
  **0 percentage points**.
- **Sealed fused-update test:** three fused-update runs, three AdamW positive
  controls, and nine matched forward/inference/dummy-write controls all have
  valid synchronized traces. Fused update is detected **0/3** by NVML, current,
  and fusion. NVML detects AdamW **3/3**; current and fusion detect it **0/3**.
  Every method produces **0/9** matched-negative alerts. This is a completed
  negative result, not a missing experiment.
- **Released paper metrics:** across the 0.6625 matched-negative GPU-hours from
  the synchronized development and sealed tests, current and fusion produce
  **0.000 alerts/GPU-hour**, while NVML produces **3.019 alerts/GPU-hour**.
  Median alert time among the six current-detected development runs is
  **90 seconds**. Fused-update time to alert is not estimable because no method
  alerts.
- **Fused-recovery attempt:** a candidate fitted after opening fused examples
  detects **3/3** fresh fused variants but also alerts on **9/9** matched
  negatives, equal to **48 false alerts/GPU-hour**. The candidate is rejected;
  aggregate current cannot distinguish real weight updates from matched scratch
  writes.
- **CUSUM comparison:** on held development repetition 3, current CUSUM detects
  the same **2/5** positives as 3-of-5 and reduces median alert time from 90 to
  **52.5 seconds**, but adds one false alert. Fusion CUSUM also detects **2/5**,
  reduces median latency to **75 seconds**, and adds one false alert. Every CUSUM
  configuration detects **0/3** fused-update runs, so CUSUM is not selected.
- **Matched monitor overhead:** on the same six GPT-2, LLaMA, and Qwen
  configurations, WAVE averages **28.497x runtime**, while the SensorGuard
  NVML-plus-DCGM base logger averages **1.050x**. In a separate five-repetition
  test, streaming all six PicoScopes at 10 kS/s gives a mean useful-work penalty
  of **-0.045%** with 95% CI **-0.353% to 0.262%**, so no slowdown is resolved.
  Logger CPU use is 14.28% of one core-equivalent and raw output is 863.49 MB per
  monitored GPU-hour.
- **Sampling and quantization:** offline replay preserves all full-rate
  decisions at **10 kS/s and 8 bits**. At 1 kS/s, detection falls from 6/15 to
  **5/15**; at 100 Hz and 10 Hz it is **0/15**. All current configurations have
  0/15 false alerts. The 10 kS/s, 8-bit point still needs hardware-in-loop
  confirmation.
- **RTX 3090 application roofline:** five cases with three repetitions each
  cover ResNet-50 training/inference and GPT-2 training/prefill/decode. Training
  and inference overlap in arithmetic intensity and throughput, demonstrating
  that roofline is a workload-matching tool rather than a detector.
- **Negative-exposure accounting:** 14.454316 GPU-hours have valid manifests
  and checksums, including 11.45 prior hours. The final frozen detector has not
  yet been applied to all of this exposure, so this is completed exposure
  accounting but not a released final false-alert rate.
- **Optional-sensor readiness:** six PicoScopes and the verified GPU1 current
  map are available. No UltraMic, thermal/visible camera, wired thermistor,
  or RTL-SDR is currently exposed to our accounts; the old SAIGE endpoint
  returns HTTP 530. A normal Intel X710 interface exists, but no passive network
  tap is verified and packet capture lacks `CAP_NET_RAW`. Therefore no result is
  claimed for thermal, contact-temperature, network, RF, or visible-light data.

## Queued on 2026-09-01

- **Hard-family replay:** 24 synchronized cells are running: seven difficult
  Rahman attack variants with three repetitions each and three matched
  white-box inference controls. Node1 records 1 Hz NVML from RTX 3090 GPU1;
  the verifier records the mapped 10 kS/s GPU-current channel.
- **Calibrated late fusion:** after collection, node2 fits logistic late fusion
  with isotonic calibration using development-only out-of-family modality
  probabilities, then applies the frozen 0.85 threshold and 3-of-5 rule to the
  new attack families. No new-family label is used for fitting or calibration.
- **Simulated health, consistency, replay, and spoofing:** queued injections
  cover modality freeze, dropout, bias/scale, time shift, control replay, and
  channel swap. These are explicitly probability-trace injections, not claims
  about physically unplugged or rewired sensors.
- **Privacy:** queued leave-one-repetition-out classification measures how much
  application-family identity can be recovered from GPU-current features.
  Higher accuracy means greater privacy leakage.
- **Sensor roofline:** `results/tables/sensor-roofline.csv` records sensing
  bitrate/storage against detection, false alerts, decision retention, and the
  measured full-rate overhead. It identifies 10 kS/s at 8 bit as the minimum
  offline-retained point, while retaining the measured 0% worst-family result.

The feature exports run automatically after all 24 node and verifier cells
pass. Node2 waits for both exports, runs every offline analysis above, and
publishes compact CSV and Markdown results through the private tailnet transfer
path. Raw Robi data are neither modified nor transferred.

## Do not rerun

Do not repeat Rahman's large NVML corpus, the 36-run paired corpus, the 61-run
stress cohort, the 30 synchronized development runs, the sealed fused matrix,
the application roofline, or the existing sensor-subset ablation. The short
node2 NVML reruns started on 2026-09-01 were stopped when this duplication was
identified; their partial files were retained and Robi's data were untouched.

## Minimum remaining work

1. **Analysis only, no GPU recollection:** apply the final common feature
   contract, grouped splits, threshold 0.85, fixed 3-of-5 rule, and CUSUM to
   every compatible existing run. Keep the 0.75 Rahman result separate unless
   its stored probabilities are re-scored at 0.85. Score all existing negative
   exposure before collecting more hours.
2. **Freeze the paper endpoint:** the current sealed result is negative. The
   paper may report that limitation. A positive fused-update claim instead
   requires a new trusted semantic signal and then a newly sealed attack
   implementation; fused-update cannot be reused as an untouched test.
3. **Software tamper tests:** run replay/freeze, bias/scale, clipping, sample
   drop, timestamp shift, channel swap, and disconnect injections on existing
   traces. This needs no new GPU workload.
4. **Optional physical sensors only on misses:** after hardware is attached and
   mapped, collect synchronized thermal/contact-temperature, RF, or UltraMic
   data only for ordinary training, interleaving, memory-minimal training,
   fused update, and their matched inference/forward/dummy-write/FFT controls.
   A sensor enters the system only if it improves held-family decisions without
   increasing false alerts. Network capture is relevant mainly to distributed
   training; visible light remains tamper-only.
5. **Small confirmations:** confirm the 10 kS/s/8-bit point in hardware and
   measure overhead once with the final retained NVML + current pipeline.

## Fast two-node execution

- **Node1 + verifier:** synchronized physical acquisition only. Use validated
  RTX 3090 GPU1 and scope `12789/2929` channel A. GPU0 remains quarantined.
  Because acoustic and thermal channels are node-shared, run one target
  workload at a time unless per-GPU isolation is first demonstrated.
- **Node2:** analysis, full-corpus re-scoring, negative-exposure scoring, and
  software tamper injection. Do not collect duplicate NVML workload runs and
  do not use held-out GPU4 for development.
- Long jobs use named `screen` sessions. Existing Robi directories and runs are
  read-only; new artifacts go to a separate worktree or results directory.

Until optional sensors are physically attached, the fastest valid strategy is
parallel **analysis on node2** and **no duplicate collection on node1**. More
GPU runs alone cannot fix the fused-update failure because the missing
information is semantic, not statistical power.

## Optional-sensor readiness test (2026-09-01)

The non-invasive readiness probe is
`scripts/experiments/probe_optional_sensors.py`; the frozen miss-only protocol
is `configs/optional_sensor_miss_campaign.json`.  On the current machines it
finds six PicoScopes on the verifier, but no attached UltraMic, thermal/visible
camera, thermistor interface, or RTL-SDR.  The old SAIGE health endpoint returns
HTTP 530 and neither remote account has SAIGE credentials.  Node1 has an Intel
X710 10GBASE-T interface, but this is not proof of a passive tap and the account
lacks `CAP_NET_RAW`, so packet capture cannot start yet.

Once a modality becomes ready, rerun only frozen NVML+current false-negative
families (three repetitions) and their matched controls.  The first known block
is fused update versus forward-only, dummy-write, and inference.  Network is
tested separately with two-node DDP training versus distributed inference and
matched all-reduce.  Visible light is evaluated only for obstruction/disconnect
health checks.  A modality is retained only if it adds at least one held-family
detection without adding a false-alert run; otherwise it remains excluded from
the headline system.
