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

## Results already available

- **Large prior NVML corpus:** Rahman's committed classifier evaluates 1,396
  eligible runs across nine NVIDIA GPU models and reports 98.21% random-forest
  accuracy at 30 seconds. The prior paper covers 1,404 collected runs; the
  difference is collection count versus classifier-eligible count. Its fixed
  run-level rebuttal result used threshold 0.75, not the current 0.85 threshold,
  so it is prior context rather than a direct SensorGuard comparison.
- **Matched 36-run RTX 3090 development comparison:** 26 training and 10
  non-training runs. NVML detects 18/26, GPU current detects 24/26, and
  NVML + GPU current detects 22/26; every method has 0/10 false-alert runs.
  These are exactly matched runs. They are development evidence, not a sealed
  final-family test.
- **Sensor subsets on the same 36 runs:** GPU current is the only added sensor
  that changes decisions. Adding motherboard current or UltraMic to NVML, or
  adding either after GPU current, changes no run decision. The minimum useful
  development subset is therefore GPU current; fusion is not automatically
  better than current alone.
- **Extended 61-run adaptive stress test:** NVML detects 44/51 training runs
  with 0/10 false alerts; current alone detects 42/51 with 2/10 false alerts.
  This harder development result prevents a claim that current always wins.
- **New independent synchronized corpus:** 30 valid paired runs on node1 RTX
  3090 GPU1 with PicoScope `12789/2929` channel A at 10 kS/s. NVML detects
  0/15 training runs and gives 2/15 false-alert runs; current and fusion each
  detect 6/15 with 0/15 false alerts. Current catches duty shaping and GPU
  migration but misses ordinary training, interleaving, and memory-minimal
  training. Worst-family detection is 0% for every method, so worst-family gain
  is **0 percentage points**.
- **Sealed fused-update test:** three fused-update runs, three AdamW positive
  controls, and nine matched negatives all completed with valid synchronized
  traces. Fused update is detected 0/3 by NVML, current, and fusion. NVML detects
  AdamW 3/3; current and fusion detect it 0/3. Every method gives 0/9 matched
  false alerts. Fused update is therefore a measured failure, not a missing
  experiment.
- **Fused-recovery attempt:** a candidate trained after opening fused examples
  detected 3/3 new fused variants but also alerted on all 9/9 matched negatives,
  equal to 48 false alerts/GPU-hour. The candidate is rejected. The held-variant
  diagnostic also fails to provide a stable fused solution. Aggregate NVML and
  current do not reveal whether a write updates real model weights or scratch
  memory; solving this requires a trusted semantic observable, not more blind
  threshold tuning on the same runs.
- **CUSUM:** on held development repetition 3, CUSUM produces the same current
  and fusion detections as the fixed rule, alerts earlier on detected runs, but
  adds one false alert. It detects 0/3 fused-update runs. CUSUM is useful as a
  secondary latency comparison but does not solve the fused failure.
- **Overhead:** on six matched GPT-2, LLaMA, and Qwen configurations, WAVE
  averages 28.497x runtime and the SensorGuard NVML+DCGM logger averages 1.050x.
  The separate physical logger test measures -0.045% useful-work penalty, with
  a 95% confidence interval of -0.353% to 0.262%; no physical-monitor slowdown
  is resolved at this sample size.
- **Minimum current specification:** offline replay finds that 10 kS/s at 8 bit
  preserves all 30 full-rate decisions, while 1 kS/s loses one of six
  detections. This is an offline result; a small hardware-in-loop confirmation
  remains.
- **Application roofline:** five RTX 3090 application cases with three
  repetitions each show training/inference overlap in arithmetic intensity and
  throughput. This supports motivation only; roofline is not the detector.
- **Negative exposure:** 14.454316 committed, auditable GPU-hours are presently
  counted. The final frozen detector has not yet been applied to all exposure,
  so a final false-alerts/GPU-hour claim is not released.

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
