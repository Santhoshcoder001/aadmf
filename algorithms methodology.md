# AADMF — Algorithms & Methodology
## Agentic Adaptive Data Mining Framework

**Version:** 1.0 | **Date:** March 31, 2026
**Document:** 2 of 5 — Algorithms & Methodology

---

## Table of Contents

1. [Research Methodology](#1-research-methodology)
2. [Drift Detection — Page-Hinkley (Deep Dive)](#2-drift-detection--page-hinkley-deep-dive)
3. [Scoring Matrix — Mathematical Foundation](#3-scoring-matrix--mathematical-foundation)
4. [Mining Algorithms — Complete Specification](#4-mining-algorithms--complete-specification)
5. [Hypothesis Generation — Your Template (Full Math)](#5-hypothesis-generation--your-template-full-math)
6. [Validation — Statistical Tests Explained](#6-validation--statistical-tests-explained)
7. [Provenance — Hash Chain Cryptography](#7-provenance--hash-chain-cryptography)
8. [LLM Integration Methodology (Phase 2)](#8-llm-integration-methodology-phase-2)
9. [Baseline Comparison Methodology](#9-baseline-comparison-methodology)

---

## 1. Research Methodology

### 1.1 Overall Approach: Design Science Research (DSR)

Your project follows **Design Science Research** — the standard methodology for computer science engineering projects that produce artefacts (systems, algorithms, frameworks).

```
┌─────────────────────────────────────────────────────────────────┐
│                  DSR CYCLE FOR AADMF                            │
│                                                                 │
│  PROBLEM        DESIGN         BUILD          EVALUATE          │
│  AWARENESS  →   SOLUTION   →   ARTEFACT   →   ARTEFACT         │
│                                                                 │
│  Gap in         AADMF          poc.py +       Ablation          │
│  adaptive       architecture   full system    studies +         │
│  mining                                       metrics           │
│  + provenance                                                   │
└─────────────────────────────────────────────────────────────────┘
```

**In your thesis, map each chapter to a DSR phase:**
- Chapter 1 (Introduction) → Problem Awareness
- Chapter 2 (Literature Review) → Existing solutions & gaps
- Chapter 3 (Design) → Your solution design
- Chapter 4 (Implementation) → Artefact build
- Chapter 5 (Evaluation) → Metrics + ablation
- Chapter 6 (Conclusion) → Contribution statement + patent

### 1.2 Experimental Methodology

**Variables you control:**

| Variable Type | Variables |
|---|---|
| Independent | Drift level (0, 2, 4, 6, 8 injected batches), Algorithm variant |
| Dependent | quality_score, drift_detection_latency, hypothesis_validity_rate, runtime |
| Controlled | seed=42, batch_size=150, n_features=16, same dataset |

**Statistical rigour:**
- Run each configuration 5 times (different seeds: 42, 43, 44, 45, 46)
- Report mean ± std deviation
- Use Wilcoxon signed-rank test to confirm AADMF vs baseline is statistically significant (p < 0.05)

---

## 2. Drift Detection — Page-Hinkley (Deep Dive)

### 2.1 Problem Statement

Given a stream of batches `B_1, B_2, ..., B_n`, detect when the underlying distribution `P(X)` changes — specifically when the mean of a reference feature shifts significantly.

### 2.2 Standard Page-Hinkley Test

The standard PH test tracks cumulative deviations from the observed mean:

```
For each new observation x_t:

  x̄_t  = (1/t) × Σ(x_i)          [running sample mean]
  m_t   = x_t − x̄_t − δ           [deviation from mean minus tolerance δ]
  M_t   = M_{t-1} + m_t            [cumulative sum]
  M*_t  = min(M_0, M_1, ..., M_t)  [running minimum]
  PH_t  = M_t − M*_t               [Page-Hinkley statistic]

ALARM if PH_t > λ  (threshold λ)
```

### 2.3 YOUR Enhanced Version (Novelty Point)

**Problem with standard PH:** The running sample mean `x̄_t` is equally influenced by all past observations, making it slow to adapt to gradual drift — by the time it detects, you've already drifted significantly.

**Your enhancement:** Replace running mean with **exponentially weighted moving mean (EWMA)**:

```
x̄_t = α × x̄_{t-1} + (1−α) × x_t    [EWMA with forgetting factor α]
```

This means recent observations are weighted more heavily. As drift occurs, the EWMA adapts faster, making deviations from it smaller — but a sudden sharp shift still triggers the PH alarm.

**Your complete algorithm:**

```
INPUTS:
  x_t   = mean of primary sensor column in batch t
  α     = 0.9999 (forgetting factor — tunes sensitivity)
  δ     = 0.005  (minimum detectable change)
  λ     = 50.0   (alarm threshold)

INITIALISE:
  x̄ ← 0, M ← 0, M* ← 0

FOR each batch t:
  x̄  ← α × x̄ + (1−α) × x_t
  M  ← M + x_t − x̄ − δ
  M* ← min(M*, M)
  PH ← M − M*

  drift_score    ← min(PH / λ, 1.0)    [normalised 0–1]
  drift_detected ← PH > λ
```

**Why α = 0.9999?**
At α = 0.9999, the effective window length is `1/(1−α) = 10,000` observations. With batch_size=150, this means ~67 batches of memory. This makes the EWMA stable enough not to false-alarm but reactive enough to catch real drift.

**Tuning guide for your experiments:**

| α value | Effective window | Behaviour |
|---|---|---|
| 0.9 | 10 obs | Very reactive; high false alarm rate |
| 0.999 | 1,000 obs | Balanced for streaming |
| 0.9999 | 10,000 obs | Stable; catches sustained drift |
| 0.99999 | 100,000 obs | Very conservative; misses short drift |

**Document this enhancement in your thesis as Novelty Point 1.**

### 2.4 Drift Score Interpretation

| drift_score range | Meaning | Planner response |
|---|---|---|
| 0.00 – 0.05 | Stable data | Prefer StatisticalRules (low cost) |
| 0.05 – 0.20 | Early drift signal | Begin switching to IsolationForest |
| 0.20 – 0.60 | Moderate drift | IsolationForest dominant |
| 0.60 – 1.00 | Severe drift | IsolationForest + hypothesize aggressively |

---

## 3. Scoring Matrix — Mathematical Foundation

### 3.1 Formal Definition

Let `A = {a_1, a_2, ..., a_k}` be the set of available mining algorithms.
For each algorithm `a_i` and each batch `t`, the planner computes:

```
score(a_i, t) = w_d × drift_weight(a_i) × drift_score(t)
              + w_a × accuracy_history(a_i, t)
              + w_c × (1 − cost(a_i))

where: w_d + w_a + w_c = 1.0
       w_d = 0.4, w_a = 0.3, w_c = 0.3  (your default weights)

chosen_algorithm(t) = argmax_{a_i ∈ A} score(a_i, t)
```

### 3.2 Accuracy History — EMA Update Rule

After Miner returns `quality_score(t)` for the chosen algorithm:

```
accuracy_history(a, t+1) = α_ema × accuracy_history(a, t)
                          + (1 − α_ema) × quality_score(t)

where α_ema = 0.7 (configurable)
```

This ensures the planner learns from experience: algorithms that consistently produce high quality scores get a higher accuracy history, biasing future selections toward them.

### 3.3 Algorithm Registry Values (Justified)

| Algorithm | drift_weight | cost | Justification |
|---|---|---|---|
| IsolationForest | 0.9 | 0.3 | Explicitly designed for anomaly detection in shifted distributions; O(n log n) |
| DBSCAN | 0.6 | 0.4 | Density-based; moderately robust to shift; O(n²) worst case |
| StatisticalRules | 0.4 | 0.1 | Best on stable data; minimal compute; degrades with drift |
| Apriori (Phase 2) | 0.3 | 0.5 | Association rules meaningful only in stable regimes; expensive |

### 3.4 Weight Sensitivity Analysis (for Thesis)

Run your ablation study varying weights and report:

| Variant | w_d | w_a | w_c | Expected Effect |
|---|---|---|---|---|
| Drift-dominant | 0.7 | 0.2 | 0.1 | Aggressive drift response; may over-switch |
| Your default | 0.4 | 0.3 | 0.3 | Balanced (patent claim) |
| Cost-dominant | 0.1 | 0.2 | 0.7 | Always picks StatisticalRules; ignores drift |
| Accuracy-dominant | 0.2 | 0.7 | 0.1 | Exploits best-past-performer; slow to adapt |

---

## 4. Mining Algorithms — Complete Specification

### 4.1 IsolationForest

**Theory:** Anomaly isolation. Randomly selects a feature, then randomly selects a split value between feature min and max. Anomalies are isolated in fewer splits (shorter path lengths).

**Anomaly score:**
```
score(x) = 2^(−E[h(x)] / c(n))

where h(x)  = path length for point x
      c(n)  = average path length for dataset of size n
      E[h(x)] = expected path length over all trees

Points with score ≈ 1 are anomalies.
Points with score ≈ 0.5 are normal.
```

**Your adaptive contamination (novelty):**
```python
contamination = max(0.05, min(0.45, 0.10 + 0.20 * drift_score))
```

When `drift_score = 0`: contamination = 0.10 (standard)
When `drift_score = 1`: contamination = 0.30 (expect more anomalies under severe drift)

**Parameters:**
```python
IsolationForest(
    n_estimators=100,         # 100 trees (standard)
    contamination=adaptive,   # YOUR adaptive logic
    max_samples="auto",       # sqrt(n_samples)
    random_state=config.seed,
    n_jobs=-1                 # use all cores
)
```

**Quality score formula:**
```
quality_score = 1.0 − anomaly_rate
anomaly_rate  = |{x : predict(x) == −1}| / n
```

### 4.2 DBSCAN

**Theory:** Density-Based Spatial Clustering of Applications with Noise. A point is a core point if it has ≥ `min_samples` neighbours within distance `eps`. Clusters form from reachable core points. Noise points (label = -1) are isolated.

**DBSCAN is useful here because:**
- It does not assume spherical clusters (unlike K-Means)
- It identifies outliers naturally (noise = anomalies)
- No need to specify k in advance
- Under drift, cluster structure changes — detecting this is informative

**Your adaptive eps (novelty):**
```python
# Base eps from data scale; tighten under drift (drift creates outliers)
eps = base_eps * (1.0 − 0.3 * drift_score)
```

When drift_score = 0: eps = 1.5 (standard clustering)
When drift_score = 1: eps = 1.05 (tighter → more noise points detected)

**Parameters:**
```python
DBSCAN(
    eps=adaptive,             # YOUR adaptive logic
    min_samples=5,
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1
)
```

**Quality score formula:**
```
n_clusters   = |unique_labels| − (1 if −1 in labels else 0)
quality_score = n_clusters / (n_clusters + 1)
```

This saturates toward 1.0 as more meaningful clusters are found, and is 0 when only noise is returned.

### 4.3 Statistical Rules (Hand-Coded)

**Theory:** Discover linear association patterns between feature pairs. Binarise features at their column mean, then measure correlation.

**Your algorithm:**
```
FOR each pair (feature_i, feature_j) in first K features:
  b_i ← (feature_i > mean(feature_i)).astype(int)
  b_j ← (feature_j > mean(feature_j)).astype(int)
  r, p ← pearsonr(b_i, b_j)
  IF |r| > corr_threshold:
    rule: "HIGH feature_i → HIGH feature_j" (if r > 0)
          "HIGH feature_i → LOW feature_j"  (if r < 0)
```

**Quality score:**
```
quality_score = min(n_rules_found / 5.0, 1.0)
```

5 rules = full score; 0 rules = 0 score.

**Why this is fast:** Only O(K²) pairs checked; K ≤ 8 in default config → 28 pairs max.

### 4.4 Apriori (Phase 2 only)

**Theory:** Classic association rule mining. Finds frequent itemsets in a transactional dataset, then derives rules `A → B` with support and confidence measures.

**Adaptation for sensor data:**
- Discretise continuous sensor values into bins: LOW / MEDIUM / HIGH
- Treat each row as a "transaction" of sensor states
- Mine frequent itemsets across the transaction set

**Key metrics:**
```
support(A)    = |transactions containing A| / |all transactions|
confidence(A→B) = support(A ∪ B) / support(A)
lift(A→B)     = confidence(A→B) / support(B)

Rule is interesting if: support > min_support
                        confidence > min_confidence
                        lift > 1.0
```

**Library:** `mlxtend.frequent_patterns.apriori` + `association_rules`

**Parameters:**
```python
frequent_itemsets = apriori(df_binary, min_support=0.3, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1.2)
```

---

## 5. Hypothesis Generation — Your Template (Full Math)

### 5.1 Formal Template Definition

**This is your core original contribution. Document it verbatim in your patent application.**

Given:
- Feature set `F = {f_1, f_2, ..., f_16}` (sensor columns)
- Current batch window `W_t` of `n` observations
- Drift score `d_t ∈ [0, 1]`
- Correlation threshold `τ_r = 0.3`
- Mutual information threshold `τ_MI = 0.05`

**Hypothesis Generation Algorithm:**

```
FUNCTION GenerateHypotheses(W_t, d_t, drift_detected):

  IF d_t ≤ 0.1 AND NOT drift_detected:
    RETURN []   // no hypotheses in stable regime

  H ← []

  FOR i IN range(min(8, |F|)):
    FOR j IN range(i+1, min(8, |F|)):

      // Step 1: Pearson correlation
      r_{ij}, p_{ij} ← PearsonR(W_t[:, f_i], W_t[:, f_j])

      // Step 2: Mutual Information proxy
      MI_{ij} ← |r_{ij}| × 0.5

      // Step 3: Apply template filter
      IF |r_{ij}| > τ_r AND MI_{ij} > τ_MI:

        // Step 4: Construct hypothesis
        condition ← "under drift" if drift_detected else "in stable regime"
        statement ← TEMPLATE(f_i, f_j, r_{ij}, MI_{ij}, d_t, condition)

        H ← H ∪ {Hypothesis(id, f_i, f_j, r_{ij}, MI_{ij}, p_{ij}, statement)}

  // Step 5: Rank by |r|, return top 3
  RETURN TOP_K(H, k=3, key=|correlation|)
```

**Template string (Phase 1):**
```
"{f_i} and {f_j} show co-pattern (r={r:.2f}, MI≈{MI:.2f}) {condition}
[drift_score={d_t}] → investigate combined feature ({f_i}×{f_j})
for gas classification"
```

### 5.2 Why Pearson + MI Proxy?

**Pearson correlation alone** catches linear relationships but misses non-linear co-patterns.

**Mutual Information (MI)** catches any dependency (linear or not). True MI requires:
```
MI(X,Y) = Σ_x Σ_y P(x,y) × log(P(x,y) / (P(x)×P(y)))
```

In Phase 1, we approximate MI with `|r| × 0.5` — this is conservative and fast.

In Phase 2, replace with:
```python
from sklearn.metrics import mutual_info_score
MI = mutual_info_score(
    pd.cut(X[f_i], bins=5, labels=False),
    pd.cut(X[f_j], bins=5, labels=False)
)
```

**Combined filter rationale:** A feature pair must pass BOTH tests:
- High |r| → strong linear relationship
- High MI → relationship is not random

This dual filter reduces false hypotheses compared to using either test alone.

### 5.3 Hypothesis Quality Metric (for Thesis)

Define **hypothesis validity rate** over a full run:

```
HVR = |{h ∈ H : h.valid == True}| / |H|

where H = all hypotheses generated across all batches
      h.valid = chi-square test passed (p < 0.05) on both real + synthetic
```

Your target: HVR > 0.70 (70% of generated hypotheses are statistically valid).

### 5.4 Phase 2 Enhancement — Real Mutual Information

In Phase 2, the hypothesizer also ranks by **MI decay** — the change in MI between the current window and the previous window:

```
ΔMI_{ij}(t) = MI_{ij}(W_t) − MI_{ij}(W_{t-1})

If ΔMI_{ij}(t) > threshold: hypothesize about this pair even if |r| is moderate
```

This detects pairs where the relationship is *changing*, which is often more interesting than pairs with consistently high correlation.

---

## 6. Validation — Statistical Tests Explained

### 6.1 Chi-Square Test of Independence

**Purpose:** Test whether two binary variables are statistically independent.

**Setup:**
```
Given feature pair (f_i, f_j):
  a ← (W_t[f_i] > median(W_t[f_i])).astype(int)   // 1 if above median
  b ← (W_t[f_j] > median(W_t[f_j])).astype(int)

Contingency table:
          b=0      b=1
  a=0  [  n_00  |  n_01  ]
  a=1  [  n_10  |  n_11  ]
```

**Chi-square statistic:**
```
χ² = Σ_{i,j} (O_{ij} − E_{ij})² / E_{ij}

where O_{ij} = observed count in cell (i,j)
      E_{ij} = expected count = (row_sum × col_sum) / n
```

**p-value:** probability of observing χ² this large under the null hypothesis (independence). If p < 0.05, reject null → features are dependent → hypothesis supported.

**Degrees of freedom:** dof = (rows−1)×(cols−1) = 1 for 2×2 table.

**In Python:**
```python
from scipy.stats import chi2_contingency
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

### 6.2 Synthetic Augmentation (Your Novelty in Validation)

Standard validation uses only the current window. Your validator also checks on a noisy copy:

```
noise_factor_i ~ Uniform(1−ε, 1+ε)   for each observation i
                 where ε = 0.05

X_noisy[f_j, i] ← X[f_j, i] × noise_factor_i
```

If the chi-square test still passes on `X_noisy`, the hypothesis is **robust** — not just a statistical artefact of the specific sample.

**This robustness check is your patent novelty in the validation step.** Document it as: *"double-validation via synthetic perturbation augmentation."*

### 6.3 Confidence Levels

| Confidence | Condition | Interpretation |
|---|---|---|
| HIGH | p_real < 0.01 | Very strong evidence; suitable for reporting |
| MEDIUM | 0.01 ≤ p_real < 0.05 | Statistically valid; interpret with context |
| LOW | p_real ≥ 0.05 | Not statistically significant; generated but not valid |

### 6.4 Phase 2 — Spearman Rank Correlation (Additional Check)

For Phase 2, add Spearman as a non-parametric alternative:

```python
from scipy.stats import spearmanr
rho, p_spearman = spearmanr(X[f_i], X[f_j])
```

A hypothesis is `confidence=HIGH` in Phase 2 only if:
- Pearson p < 0.01 AND
- Spearman p < 0.01 AND
- Chi-square on synthetic p < 0.05

---

## 7. Provenance — Hash Chain Cryptography

### 7.1 SHA-256 Hash Function Properties

SHA-256 is a cryptographic hash function. For AADMF provenance:

| Property | Meaning for Provenance |
|---|---|
| Deterministic | Same input always gives same hash |
| Avalanche effect | Changing 1 bit in input changes ~50% of output bits |
| Pre-image resistance | Cannot reverse-engineer input from hash |
| Collision resistance | Cannot find two inputs with same hash |

### 7.2 Hash Chain Mechanism

```
Event 0: prev_hash = "GENESIS"
         payload   = JSON({seq:0, type:"SYSTEM_START", ts:..., details:..., prev_hash:"GENESIS"})
         hash_0    = SHA256(payload)[:16]

Event 1: prev_hash = hash_0
         payload   = JSON({seq:1, type:"DATA_INGESTED", ..., prev_hash:hash_0})
         hash_1    = SHA256(payload)[:16]

Event 2: prev_hash = hash_1
         payload   = JSON({seq:2, type:"DRIFT_CHECK", ..., prev_hash:hash_1})
         hash_2    = SHA256(payload)[:16]
...
```

**Tamper detection:**
If an attacker modifies `details` of Event 1:
```
Modified payload → SHA256 → different hash_1'
Event 2's prev_hash = hash_1 ≠ hash_1'
→ verify_integrity() detects mismatch at Event 2
```

### 7.3 Why Not Full Blockchain?

| Feature | Blockchain | AADMF Hash Chain |
|---|---|---|
| Tamper detection | ✅ | ✅ |
| Distributed consensus | ✅ | ❌ (not needed) |
| Cryptographic overhead | High | Low |
| Laptop runnable | ❌ Needs nodes | ✅ Pure Python |
| Patent novelty | Low (common) | High (novel in data mining) |

For a single-system audit trail, full blockchain is unnecessary overhead. Your SHA-256 chain provides the same tamper-resistance property at 1% of the cost. **Document this design decision explicitly in your thesis.**

### 7.4 Verification Algorithm Complexity

```
Time complexity:  O(n) where n = number of events
Space complexity: O(n) for storing the chain
Hash computation: O(1) per event (fixed payload size)
```

For 42 events (a full 10-batch run): verification completes in microseconds.

---

## 8. LLM Integration Methodology (Phase 2)

### 8.1 Role — Narrow and Controlled

The LLM is used ONLY for one task: converting a hypothesis dict into a richer natural language sentence.

```
INPUT (your algorithm output):
{
  "feature_a": "sensor_0",
  "feature_b": "sensor_3",
  "correlation": 0.42,
  "mutual_info_proxy": 0.21,
  "drift_detected": true
}

PROMPT TO LLM:
"You are a data science assistant. Given this hypothesis data,
write one clear sentence suitable for a data analyst:
{json_input}
Rules: keep it under 30 words; include the feature names and
correlation value; do not invent new statistics."

OUTPUT (LLM):
"Sensor 0 and sensor 3 show a significant positive co-pattern
(r=0.42) under data drift, suggesting a new combined feature
may improve gas classification."
```

### 8.2 Why Phi-3 Mini?

| Model | Size | RAM | Speed (laptop) | Quality |
|---|---|---|---|---|
| Phi-3 Mini (3.8B) | ~2.4GB | ~4GB | ~5 tok/s | Excellent for structured tasks |
| TinyLlama (1.1B) | ~0.7GB | ~2GB | ~15 tok/s | Good for simple phrasing |
| Llama 3.2 (3B) | ~2GB | ~4GB | ~6 tok/s | Comparable to Phi-3 |
| GPT-4o | 0 (API) | 0 | Fast | Best quality; costs money |

Phi-3 Mini runs locally via Ollama with no internet. It is specifically trained for instruction-following structured tasks — ideal for your narrow use case.

### 8.3 Ollama Setup (Month 3)

```bash
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Pull Phi-3 Mini
ollama pull phi3:mini

# Test
ollama run phi3:mini "Describe this in one sentence: sensor_0 r=0.42"
```

**Python integration:**
```python
import ollama

def phrase_hypothesis(hyp: dict) -> str:
    prompt = f"""
    You are a data science assistant. Write ONE sentence (max 30 words)
    describing this hypothesis for a data analyst.
    Include feature names and correlation value exactly.
    Data: {json.dumps(hyp)}
    """
    response = ollama.chat(
        model="phi3:mini",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.2}  # low temp = deterministic
    )
    return response["message"]["content"].strip()
```

### 8.4 Hallucination Guard

After LLM phrasing, validate that:
- Feature names `feature_a` and `feature_b` appear in the output
- Correlation value (rounded) appears in the output
- Output length < 50 words

If any check fails → fall back to template string.

```python
def safe_phrase(hyp: dict, llm_output: str) -> str:
    checks = [
        hyp["feature_a"] in llm_output,
        hyp["feature_b"] in llm_output,
        str(round(hyp["correlation"], 1)) in llm_output,
        len(llm_output.split()) < 50
    ]
    if all(checks):
        return llm_output
    return template_phrase(hyp)  # fallback
```

---

## 9. Baseline Comparison Methodology

### 9.1 Static Baseline Definition

The baseline is a **static scikit-learn pipeline** — what most students build:

```python
# baseline.py
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class StaticBaseline:
    """No agents. No drift detection. No hypothesis generation.
    Always uses IsolationForest with fixed contamination=0.1.
    No provenance logging.
    """
    def __init__(self):
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()

    def run_batch(self, X):
        Xs = self.scaler.fit_transform(X)
        preds = self.model.fit_predict(Xs)
        anomaly_rate = (preds == -1).mean()
        return {"quality_score": 1.0 - anomaly_rate}
```

### 9.2 Comparison Metrics

| Metric | AADMF | Baseline | Expected Delta |
|---|---|---|---|
| Avg quality_score (all batches) | Adaptive | Fixed ~0.60 | AADMF > +15% |
| Quality after drift injection | Recovers | Degrades | AADMF significantly better |
| Drift detection latency | Measured | N/A (no detector) | AADMF: 1 batch |
| Hypotheses generated | Yes | 0 | Qualitative advantage |
| Provenance completeness | 100% | 0% | Qualitative advantage |
| Runtime overhead | Higher | Lower | < 20% overhead target |

### 9.3 Ablation Study Design

Run 5 variants × 5 seeds × 5 drift levels = **125 experiment runs**. With ~1.5s per run, total runtime ≈ 3 minutes.

```python
# evaluation/ablation.py

VARIANTS = {
    "full":               {"scoring_matrix": True,  "hypothesizer": True,  "provenance": True},
    "no_scoring_matrix":  {"scoring_matrix": False, "hypothesizer": True,  "provenance": True},
    "no_hypothesizer":    {"scoring_matrix": True,  "hypothesizer": False, "provenance": True},
    "no_provenance":      {"scoring_matrix": True,  "hypothesizer": True,  "provenance": False},
    "static_baseline":    {"scoring_matrix": False, "hypothesizer": False, "provenance": False},
}

SEEDS = [42, 43, 44, 45, 46]
DRIFT_LEVELS = [0, 2, 4, 6, 8]  # drift_after values

results = []
for variant_name, variant_config in VARIANTS.items():
    for seed in SEEDS:
        for drift_level in DRIFT_LEVELS:
            result = run_experiment(variant_config, seed, drift_level)
            results.append({
                "variant": variant_name,
                "seed": seed,
                "drift_level": drift_level,
                **result
            })

df = pd.DataFrame(results)
df.to_csv("experiments/results/ablation_results.csv", index=False)
```

### 9.4 Statistical Significance Test

To prove AADMF outperforms baseline:

```python
from scipy.stats import wilcoxon

aadmf_scores   = df[df.variant == "full"]["quality_score"].values
baseline_scores = df[df.variant == "static_baseline"]["quality_score"].values

stat, p = wilcoxon(aadmf_scores, baseline_scores)
print(f"Wilcoxon p-value: {p:.4f}")
# If p < 0.05: AADMF improvement is statistically significant
```

Include this test in your thesis results section. It is required for IEEE conference submissions.

---

*AADMF — Algorithms & Methodology v1.0 | Document 2 of 5*