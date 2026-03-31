# AADMF — Evaluation Plan & Thesis Outline
## Agentic Adaptive Data Mining Framework

**Version:** 1.0 | **Date:** March 31, 2026
**Document:** 5 of 5 — Evaluation Plan + Thesis Structure

---

## Table of Contents

1. [Evaluation Framework Overview](#1-evaluation-framework-overview)
2. [Metrics — Complete Definitions](#2-metrics--complete-definitions)
3. [Experiment Configurations](#3-experiment-configurations)
4. [Ablation Study Design](#4-ablation-study-design)
5. [Results Reporting Template](#5-results-reporting-template)
6. [Thesis Chapter Structure](#6-thesis-chapter-structure)
7. [IEEE Paper Structure (arXiv Submission)](#7-ieee-paper-structure-arxiv-submission)
8. [Viva Preparation Guide](#8-viva-preparation-guide)

---

## 1. Evaluation Framework Overview

### 1.1 What You Are Evaluating

Your evaluation answers three research questions, each mapped to a patent claim:

| Research Question | Patent Claim | Primary Metric |
|---|---|---|
| RQ1: Does the scoring matrix select better algorithms than random/static selection under drift? | Claim 1 | quality_score improvement vs baseline |
| RQ2: Does the hypothesis template generate statistically valid hypotheses? | Claim 2 | Hypothesis Validity Rate (HVR) |
| RQ3: Does the hash-chained provenance detect 100% of tampers? | Claim 3 | Tamper detection rate |

### 1.2 Evaluation Philosophy

**Quantitative:** Compare AADMF vs 4 ablation variants across 125 experiment runs. Report mean ± std. Confirm significance with Wilcoxon test.

**Qualitative:** Manual inspection of 10–15 generated hypotheses — do they make domain sense? (Gas sensor domain: does correlation between sensor_0 and sensor_3 under ethanol exposure make physical sense?)

**Reproducibility:** All experiments run with `seed=42` baseline. Sensitivity analysis across seeds 42–46.

---

## 2. Metrics — Complete Definitions

### 2.1 Quality Score (Primary Mining Metric)

Measures mining output quality per batch per algorithm.

```
IsolationForest:  quality_score = 1.0 - anomaly_rate
DBSCAN:           quality_score = n_clusters / (n_clusters + 1)
StatisticalRules: quality_score = min(rules_found / 5.0, 1.0)

Aggregate (per run):
  mean_quality_score = mean(quality_score over all batches)
```

**Why this metric:** Each algorithm has a different natural output. Quality score normalises all to [0, 1] so Planner's accuracy_history is comparable across algorithm types.

**Limitation to acknowledge in thesis:** Quality score is a proxy — it does not directly measure classification accuracy. Acknowledge and discuss.

### 2.2 Drift Detection Latency

```
drift_detection_latency = first_batch_where_drift_detected - drift_injection_batch

Target: latency = 1 batch (detect in the batch immediately after injection)
```

**How to measure:**
1. Inject drift at batch `drift_after`
2. Record first batch where `drift_detected == True`
3. Latency = detected_batch - drift_after

**Expected result:** With tuned PH parameters, latency ≤ 2 batches for all drift levels.

### 2.3 Hypothesis Validity Rate (HVR)

```
HVR = n_valid_hypotheses / n_total_hypotheses

n_valid_hypotheses = count where chi-square p < 0.05 on BOTH real and synthetic
n_total_hypotheses = all hypotheses generated across all batches

Target: HVR > 0.70
```

**Per-confidence breakdown:**
```
HVR_HIGH   = count(confidence == "HIGH") / n_total
HVR_MEDIUM = count(confidence == "MEDIUM") / n_total
HVR_LOW    = count(confidence == "LOW") / n_total
```

### 2.4 Provenance Completeness

```
completeness = n_events_logged / n_events_expected

n_events_expected = 1 (SYSTEM_START)
                  + n_batches × 4 (DATA_INGESTED, DRIFT_CHECK, ALGO_SELECTED, MINING_RESULT)
                  + n_drift_batches × 1 (HYPOTHESES_GENERATED)
                  + n_hypotheses_validated × 1 (HYPOTHESIS_VALIDATED)

Target: completeness = 1.00 (100%)
```

### 2.5 Tamper Detection Rate

```
tamper_detection_rate = n_detected_tampers / n_injected_tampers

Test: inject 10 tampers at random event positions
      count how many are detected by verify_integrity()

Target: 100% (all 10 detected)
```

### 2.6 Runtime Overhead

```
overhead = (runtime_AADMF - runtime_baseline) / runtime_baseline × 100%

Target: overhead < 20%
```

Measure wall-clock time for identical dataset and batch count.

### 2.7 Algorithm Selection Accuracy (ASA)

```
ASA = n_correct_algorithm_selections / n_total_batches

"Correct" definition:
  - Under high drift (drift_score > 0.5): IsolationForest is correct
  - Under no drift (drift_score < 0.05): StatisticalRules is correct
  - Under moderate drift: either IF or DBSCAN is acceptable

Target: ASA > 0.80 (80%)
```

---

## 3. Experiment Configurations

### 3.1 Dataset Configurations

| Config ID | Dataset | Batches | Batch Size | Drift Level |
|---|---|---|---|---|
| E1 | Synthetic | 10 | 150 | No drift (drift_after=999) |
| E2 | Synthetic | 10 | 150 | Early drift (drift_after=2) |
| E3 | Synthetic | 10 | 150 | Mid drift (drift_after=4) |
| E4 | Synthetic | 10 | 150 | Late drift (drift_after=7) |
| E5 | Synthetic | 10 | 150 | Severe drift (drift_after=0) |
| E6 | UCI Gas Sensor | 10 | ~1400 | Real drift (use all batches) |

### 3.2 System Variants

| Variant ID | Scoring Matrix | Hypothesizer | Provenance | Description |
|---|---|---|---|---|
| V1 | ✅ | ✅ | ✅ | Full AADMF (your system) |
| V2 | ❌ (random) | ✅ | ✅ | Ablate scoring matrix |
| V3 | ✅ | ❌ | ✅ | Ablate hypothesizer |
| V4 | ✅ | ✅ | ❌ | Ablate provenance |
| V5 | ❌ | ❌ | ❌ | Static baseline (IsolationForest fixed) |

### 3.3 Seeds

Run each (Config × Variant) combination with seeds: **[42, 43, 44, 45, 46]**

**Total experiment runs:** 6 configs × 5 variants × 5 seeds = **150 runs**
**Estimated runtime:** ~4 minutes total (each run ≈ 1.5 seconds)

### 3.4 Experiment Runner Code

```python
# aadmf/evaluation/ablation.py

import pandas as pd
import time
import yaml
from itertools import product

VARIANTS = {
    "V1_full":           {"use_scoring_matrix": True,  "use_hypothesizer": True,  "use_provenance": True},
    "V2_no_matrix":      {"use_scoring_matrix": False, "use_hypothesizer": True,  "use_provenance": True},
    "V3_no_hypothesizer":{"use_scoring_matrix": True,  "use_hypothesizer": False, "use_provenance": True},
    "V4_no_provenance":  {"use_scoring_matrix": True,  "use_hypothesizer": True,  "use_provenance": False},
    "V5_static":         {"use_scoring_matrix": False, "use_hypothesizer": False, "use_provenance": False},
}

DRIFT_LEVELS = [999, 2, 4, 7, 0]   # drift_after values
SEEDS = [42, 43, 44, 45, 46]

def run_all_experiments(base_config: dict) -> pd.DataFrame:
    results = []
    total = len(VARIANTS) * len(DRIFT_LEVELS) * len(SEEDS)
    i = 0

    for variant_name, variant_flags in VARIANTS.items():
        for drift_after in DRIFT_LEVELS:
            for seed in SEEDS:
                i += 1
                print(f"[{i}/{total}] variant={variant_name} drift_after={drift_after} seed={seed}")

                config = {**base_config}
                config["streaming"]["drift_after"] = drift_after
                config["streaming"]["seed"] = seed
                config.update(variant_flags)

                t0 = time.time()
                result = run_single_experiment(config)
                runtime = time.time() - t0

                results.append({
                    "variant": variant_name,
                    "drift_after": drift_after,
                    "seed": seed,
                    "runtime_s": round(runtime, 3),
                    **result
                })

    return pd.DataFrame(results)
```

---

## 4. Ablation Study Design

### 4.1 Primary Comparison: Full AADMF vs Static Baseline

**Hypothesis (to prove):** Full AADMF achieves significantly higher mean quality_score than static baseline across all drift levels.

**Statistical test:** Wilcoxon signed-rank test (non-parametric; no normality assumption required)

```python
from scipy.stats import wilcoxon

v1 = df[df.variant == "V1_full"]["mean_quality_score"].values
v5 = df[df.variant == "V5_static"]["mean_quality_score"].values

stat, p = wilcoxon(v1, v5)
print(f"W={stat:.2f}, p={p:.4f}")
# Expect p < 0.05 → AADMF improvement is statistically significant
```

### 4.2 Component Contribution Analysis

For each ablated component, compute:

```
contribution_scoring_matrix  = mean(V1) - mean(V2)  / mean(V5)  [normalised gain]
contribution_hypothesizer    = qualitative (hypotheses generated vs none)
contribution_provenance      = tamper_detection_rate: V1=100%, V4=0%
```

### 4.3 Expected Results Table (Fill In After Running)

| Variant | Mean Quality Score | Std | Drift Detection Latency | HVR | Runtime (s) |
|---|---|---|---|---|---|
| V1 Full AADMF | [fill] | [fill] | [fill] | [fill] | [fill] |
| V2 No Scoring Matrix | [fill] | [fill] | [fill] | [fill] | [fill] |
| V3 No Hypothesizer | [fill] | [fill] | [fill] | — | [fill] |
| V4 No Provenance | [fill] | [fill] | [fill] | [fill] | [fill] |
| V5 Static Baseline | [fill] | [fill] | N/A | — | [fill] |

### 4.4 Drift Level Sensitivity (Chart 1 in Thesis)

Plot: mean_quality_score vs drift_level (drift_after value) for all 5 variants.

**Expected pattern:**
- V5 (static): degrades as drift increases (no adaptation)
- V1 (full): stays relatively stable (adapts via scoring matrix)
- V2 (no matrix): partial degradation (hypothesizer still helps)
- Gap between V1 and V5 widens at high drift levels → proves adaptation value

---

## 5. Results Reporting Template

Use this exact structure in your thesis Chapter 5:

### 5.1 Section: Experimental Setup

```
Hardware: [CPU, RAM, OS]
Python: 3.12.x
Dataset: UCI Gas Sensor Array Drift (Batches 1–10) + Synthetic
Seeds: 42, 43, 44, 45, 46 (5 runs per configuration)
Total experiments: 150
Key parameters: [list your final config.yaml values]
```

### 5.2 Section: RQ1 Results — Algorithm Selection

```
Table X: Mean quality_score ± std by variant and drift level

Figure X: Line chart — quality_score vs drift_after, all variants

Key finding: AADMF (V1) achieves [X]% higher mean quality_score than
static baseline (V5) across all drift levels (Wilcoxon p=[p value], W=[W value]).
The scoring matrix contributes [X]% improvement (V1 vs V2 comparison).
```

### 5.3 Section: RQ2 Results — Hypothesis Generation

```
Table X: Hypothesis statistics by variant and drift level

| Drift Level | n_generated | n_valid | HVR   | HVR_HIGH | HVR_MEDIUM |
|-------------|-------------|---------|-------|----------|------------|
| No drift    | [fill]      | [fill]  | [fill]| [fill]   | [fill]     |
| Mid drift   | [fill]      | [fill]  | [fill]| [fill]   | [fill]     |
| High drift  | [fill]      | [fill]  | [fill]| [fill]   | [fill]     |

Figure X: Bar chart — HVR by drift level

Key finding: The hypothesizer achieves HVR=[X]% at high drift levels
(drift_after=2), confirming that drift-triggered hypothesis generation
produces statistically significant patterns.

Example hypothesis (select your best one):
  "sensor_0 and sensor_3 show co-pattern (r=0.61, MI≈0.31) under drift
   [drift_score=0.42] → investigate combined feature for gas classification"
  Validation: chi-square p=0.0023 (HIGH confidence), robust to ±5% noise (p=0.011)
```

### 5.4 Section: RQ3 Results — Provenance

```
Table X: Provenance integrity test results

| Test | Result |
|------|--------|
| Completeness | 100% (all expected events logged) |
| Tamper detection rate | 100% (10/10 injected tampers detected) |
| Mean tamper detection position | exact event modified |
| Chain verification time | [X] ms for 42 events |

Figure X: Provenance graph screenshot (Neo4j browser or pyvis)

Key finding: The SHA-256 hash-chained provenance logger achieves 100%
tamper detection at exact event granularity, with negligible verification
overhead ([X] ms per 42 events).
```

### 5.5 Section: Runtime Overhead

```
Table X: Runtime comparison

| Component | Runtime (ms) | % of Total |
|-----------|-------------|------------|
| Streaming + Drift | [fill] | [fill] |
| Planner | [fill] | [fill] |
| Miner | [fill] | [fill] |
| Hypothesizer | [fill] | [fill] |
| Validator | [fill] | [fill] |
| Provenance | [fill] | [fill] |
| Total AADMF | [fill] | 100% |
| Static Baseline | [fill] | — |
| Overhead | [fill] | < 20% target |
```

---

## 6. Thesis Chapter Structure

### Chapter 1: Introduction (~2,500 words)

```
1.1 Motivation
    - Data streams are ubiquitous: IoT sensors, financial feeds, healthcare monitors
    - Three unsolved gaps: no adaptation, no hypothesis generation, no provenance
    - Quote: "The cost of silent model degradation under concept drift is [cite]"

1.2 Research Questions
    RQ1: Can an autonomous agent using a drift-weighted scoring matrix
         select better mining algorithms than static selection?
    RQ2: Can a statistical hypothesis template generate valid (p<0.05)
         patterns from streaming sensor data under drift?
    RQ3: Can SHA-256 hash-chaining provide tamper-proof provenance for
         agent decisions in a data mining pipeline?

1.3 Contributions
    - Novel scoring matrix with EMA accuracy feedback for algorithm selection
    - Original hypothesis generation template (correlation + MI + drift trigger)
    - Tamper-proof hash-chained provenance for agentic data mining
    - Open-source implementation with reproducible experiments

1.4 Thesis Organisation
    [One paragraph describing each chapter]
```

### Chapter 2: Literature Review (~4,500 words)

```
2.1 Data Stream Mining
    - Concept drift: types (sudden, gradual, incremental, recurring)
    - Cite: Gama et al. (2014), Bifet & Gavaldà (2007)
    - Gap: existing systems are static after training

2.2 Drift Detection Methods
    - DDM, EDDM, ADWIN, KSWIN
    - Cite: Gama et al. (2004), Bifet & Gavaldà (2007), Ross et al. (2012)
    - Your contribution: EWMA-enhanced PH for streaming sensors

2.3 Multi-Agent Systems for Data Mining
    - Agent-based ML: decentralised, parallel, adaptive
    - Cite: Cao (2010), Jennings et al. (1998)
    - Gap: no agent system combines drift adaptation + hypothesis generation

2.4 Automated Hypothesis Generation
    - Scientific discovery automation: Robot Scientist
    - Cite: King et al. (2009), Muggleton (2014)
    - Gap: domain-specific (biology); not general streaming data

2.5 Data Provenance
    - PROV-DM, W3C provenance model, lineage tracking
    - Cite: Moreau et al. (2013), Buneman et al. (2001)
    - Gap: no provenance system integrated with agentic mining decisions

2.6 Gap Analysis Table

| System/Paper | Drift Adaptive | Hypothesis Gen | Agent-Based | Provenance |
|---|---|---|---|---|
| [Paper 1] | ✅ | ❌ | ❌ | ❌ |
| [Paper 2] | ❌ | ✅ | ❌ | ❌ |
| [Paper 3] | ✅ | ❌ | ✅ | ❌ |
| AADMF (Ours) | ✅ | ✅ | ✅ | ✅ |
```

### Chapter 3: System Design (~3,000 words)

```
3.1 Design Goals and Principles
    - Agent isolation, layered complexity, reproducibility first

3.2 Architecture Overview
    [Include draw.io architecture diagram with all 7 layers]

3.3 Drift Detection Design
    [Page-Hinkley + EWMA enhancement — show the math]

3.4 Scoring Matrix Design (Claim 1)
    [Full mathematical derivation — show score formula, weights, EMA]

3.5 Hypothesis Template Design (Claim 2)
    [Formal definition of template — pseudocode from Algorithms doc Section 5.1]

3.6 Provenance Design (Claim 3)
    [Hash chain mechanism — show SHA-256 formula, verification algorithm]

3.7 LangGraph Integration Design
    [State graph diagram; agent nodes; conditional edges]
```

### Chapter 4: Implementation (~3,000 words)

```
4.1 Technology Stack Selection
    [Justified table from TDD Document Section 2]

4.2 Phase 1 Implementation (PoC)
    [Key code snippets in pseudocode]
    [PoC execution proof table]

4.3 Phase 2 Implementation (Full System)
    [LangGraph integration]
    [Neo4j schema]
    [Ollama integration]

4.4 Dashboard Implementation
    [Streamlit architecture]
    [Screenshot of live dashboard]

4.5 Implementation Challenges and Solutions
    [Honest discussion: what was hard; how you solved it]
```

### Chapter 5: Evaluation (~4,500 words)

```
5.1 Experimental Setup
    [Hardware, dataset, parameters — use template from Section 5.1 above]

5.2 RQ1: Algorithm Selection Results
    [Table + chart from Section 5.2 above]
    [Wilcoxon test result]

5.3 RQ2: Hypothesis Generation Results
    [Table + chart from Section 5.3 above]
    [Best hypothesis example with interpretation]

5.4 RQ3: Provenance Integrity Results
    [Table from Section 5.4 above]
    [Provenance graph screenshot]

5.5 Runtime Overhead Analysis
    [Table from Section 5.5 above]

5.6 Discussion
    - Why scoring matrix beats static: adapts to drift signal; learns from history
    - Why hypothesis template achieves >70% HVR: dual filter (r + MI) reduces noise
    - Why 100% tamper detection: cryptographic property of SHA-256 avalanche effect
    - Limitations: MI proxy vs real MI; single-node not distributed; synthetic dataset

5.7 Threats to Validity
    - Internal: same seed range; controlled dataset
    - External: results on UCI; may differ on other domains
    - Construct: quality_score is a proxy metric
```

### Chapter 6: Conclusion (~2,000 words)

```
6.1 Summary of Contributions
    - Answered RQ1: scoring matrix achieves [X]% improvement (statistically significant)
    - Answered RQ2: hypothesis template achieves HVR=[X]% at high drift
    - Answered RQ3: 100% tamper detection; negligible overhead

6.2 Future Work
    1. Replace MI proxy with real mutual information (Phase 2 enhancement)
    2. Deploy on edge device (Raspberry Pi) for IoT scenarios
    3. Extend to multimodal data (sensor + text + images)
    4. Add distributed provenance via blockchain backend
    5. Apply to healthcare monitoring (ECG drift detection)
    6. Explore reinforcement learning for scoring weight adaptation

6.3 Patent Status
    [Filing date; claim summary; university IPR cell reference]

6.4 Reproducibility Statement
    Code: github.com/[username]/aadmf
    Data: UCI Gas Sensor Array Drift (publicly available)
    Seed: 42 (deterministic results)
```

---

## 7. IEEE Paper Structure (arXiv Submission)

For your 6-page IEEE format paper derived from the thesis:

```
Title: "AADMF: Autonomous Multi-Agent Framework for Hypothesis-Driven
        Pattern Discovery in Streaming Data with Tamper-Proof Provenance"

Abstract (150 words):
  Problem → Gap → Proposed System → Key Results → Contribution

I. Introduction (0.5 pages)
  - Problem + 3 gaps → your system → contributions

II. Related Work (0.5 pages)
  - Condensed literature; gap analysis table

III. System Design (1.5 pages)
  - Architecture diagram
  - Scoring matrix formula
  - Hypothesis template pseudocode
  - Hash chain mechanism

IV. Experimental Setup (0.5 pages)
  - Dataset, variants, seeds, metrics

V. Results (2 pages)
  - Table: RQ1 results (quality_score by variant)
  - Table: RQ2 results (HVR by drift level)
  - Table: RQ3 results (provenance integrity)
  - 1 key chart: quality_score vs drift level

VI. Conclusion (0.5 pages)
  - Findings summary + future work

References (target: 15–20)
```

**Target conferences/journals:**
- IEEE ICDM Workshop on Mining Data Streams
- ACM SIGKDD (workshop track)
- arXiv cs.AI preprint (publish before conference submission)

---

## 8. Viva Preparation Guide

### 8.1 Questions You Will Definitely Get

**Q1: "Why did you choose Page-Hinkley over ADWIN?"**
> Answer: PH is implementable from scratch without external libraries, making it fully transparent and patentable as a custom implementation. ADWIN requires `scikit-multiflow` and is harder to tune. My EWMA enhancement to PH specifically addresses the weakness of standard PH (slow mean adaptation) with a simple, justified modification.

**Q2: "How do you prove your scoring matrix is better than random algorithm selection?"**
> Answer: Ablation variant V2 uses random algorithm selection. Across 150 experiment runs, V1 (scoring matrix) achieves [X]% higher mean quality_score than V2. The difference is statistically significant (Wilcoxon p=[value]). I also show the scoring matrix correctly escalates to IsolationForest within 1 batch of drift injection in [X]% of runs.

**Q3: "What is the difference between your hypothesis generation and just computing correlations?"**
> Answer: Three key differences. First, hypotheses are only triggered when drift_score > 0.1, meaning we generate hypotheses about patterns that are *emerging under change* — not static correlations. Second, we use dual filtering (both Pearson r AND MI proxy must exceed thresholds), reducing spurious hypotheses. Third, every hypothesis is double-validated on real AND synthetic data. This combination is the novelty.

**Q4: "Is SHA-256 hash chaining really novel? Blockchains already use this."**
> Answer: The hash chain mechanism itself is not novel. What is novel is applying it specifically to agentic data mining decision provenance — an area with no prior implementation in the patent literature. The novelty is the *application and integration*, not the cryptographic primitive. My patent claim is specifically about "tamper-proof provenance for agent-based data mining decisions" — the integrated system, not SHA-256 itself.

**Q5: "Why not use a deep learning model instead of IsolationForest?"**
> Answer: The novelty in this project is the *agent orchestration*, not the mining algorithm. Using simple, interpretable algorithms (IsolationForest, DBSCAN) makes the ablation study cleaner and the results more attributable to the agentic components. Deep learning would add GPU dependency, training overhead, and make it impossible to isolate the contribution of the scoring matrix vs the algorithm quality.

**Q6: "Can you demonstrate the tamper test live?"**
> Answer: Yes. [Have your Streamlit dashboard open with the Tamper Demo button ready. Click it during viva. Show chain breaks at exact event index.]

### 8.2 Live Demo Checklist (Prepare 2 Days Before Viva)

- [ ] `poc.py` runs cleanly from cold start: `python poc.py`
- [ ] Streamlit dashboard starts: `streamlit run aadmf/dashboard/app.py`
- [ ] Drift score chart visible and animating
- [ ] Hypothesis feed showing at least 2 valid hypotheses
- [ ] Tamper demo button works; shows break at correct event
- [ ] Neo4j browser open (localhost:7474) with provenance graph loaded
- [ ] GitHub repo open in browser — code visible
- [ ] Results notebook open with ablation charts

### 8.3 Key Numbers to Memorise

Before your viva, know these by heart:

```
PoC runtime:            1.42 seconds (10 batches, 150 rows each)
Provenance events:      42 per 10-batch run
Tamper detection:       100% (breaks at exact modified event)
Baseline improvement:   +20% quality_score (from PoC results)
HVR target:             > 70%
Wilcoxon p-value:       [fill after running full experiments]
Total experiments run:  150 (5 variants × 6 configs × 5 seeds)
Patent claims:          4 (1 independent method, 1 independent system,
                           1 independent method, 1 dependent integrated)
```

---

## Quick Reference — Document Index

| # | Document | Purpose | Build in |
|---|---|---|---|
| 1 | Technical Design (TDD) | Architecture, classes, data flow | Week 1 (reference) |
| 2 | Algorithms & Methodology | Full math for every algorithm | Week 1 (reference) |
| 3 | Sprint Plan | Week-by-week task breakdown | Use throughout |
| 4 | Codebase Skeleton | Python stubs ready to fill | Week 1–4 |
| 5 | Evaluation + Thesis | Metrics, results template, thesis outline | Week 13–24 |

**The order to read these documents:**
1. Sprint Plan (Document 3) — understand the journey
2. Technical Design (Document 1) — understand the system
3. Algorithms (Document 2) — understand the math
4. Codebase Skeleton (Document 4) — start coding
5. Evaluation + Thesis (Document 5) — when you reach Week 13

---

*AADMF — Evaluation Plan & Thesis Outline v1.0 | Document 5 of 5*
*Full Implementation Plan Complete — 5 Documents, 24 Weeks*