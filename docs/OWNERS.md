# Team ownership

Fill this table at kickoff. Each person owns one lane and reviews one other lane. Do not assign issues until GitHub usernames are verified.

| Lane | Primary owner | Reviewer | Responsibilities |
|---|---|---|---|
| Paper and integration | TBD | TBD | Scope, novelty, WAVE framing, writing, figures, HotCRP |
| Telemetry and physical sensors | TBD | TBD | Electrical, thermal, ultrasound, RF, and network loggers; calibration, synchronization, safety |
| Workloads and adversarial evaluation | TBD | TBD | Training/inference/non-ML corpus, custom kernels, evasion families, useful-work metrics |
| Analysis, statistics, reproducibility | TBD | TBD | Splits, feature pipeline, random-forest fusion, fixed 3-of-5 rule, bootstrap CIs, manifests, clean reruns |

## Ownership rules

- Every result has one producer and one independent reviewer.
- Only the paper/integration owner edits headline claims after Gate 2.
- Only the analysis owner can unseal the test split, with another collaborator present.
- Hardware safety and probe placement require approval from the sensor owner and a qualified lab supervisor.
