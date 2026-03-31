# AADMF — Week-by-Week Implementation Sprint Plan
## Agentic Adaptive Data Mining Framework

**Version:** 1.0 | **Date:** March 31, 2026
**Document:** 3 of 5 — Sprint Plan (24 Weeks)

---

## How to Use This Plan

- Each week has **Daily Tasks** broken into AM / PM halves
- Each week ends with a **Done Criteria** checklist — do not proceed to next week until all are ticked
- Time estimate assumes **15–20 hours/week** of focused work
- All code goes to GitHub at end of each week (commit + push)
- All design decisions go into your **Novelty Log** (a markdown file you keep alongside code)

---

## PHASE 1 — FOUNDATION & PoC (Weeks 1–4)

---

### Week 1 — Environment + Streaming Simulator + Basic Provenance

**Goal:** Development environment ready; can stream synthetic data; can log events with hash chaining.

#### Day 1 (Mon) — Environment Setup
- AM: Create GitHub repo `aadmf`; init Python venv; install: `pip install pandas numpy scikit-learn scipy streamlit plotly`
- PM: Create folder structure (see TDD Document 1); create `config.yaml`; write `README.md` skeleton

#### Day 2 (Tue) — StreamingSimulator
- AM: Write `aadmf/streaming/simulator.py` — `StreamingSimulator` class with `next_batch()` method
- PM: Add drift injection logic (sensor mean shift after `drift_after` batch); write unit test `tests/unit/test_simulator.py`

#### Day 3 (Wed) — ProvenanceLogger (Phase 1)
- AM: Write `aadmf/provenance/dict_chain.py` — `DictChainLogger` with `log()`, `verify_integrity()`, `export_json()`
- PM: Write unit test `tests/unit/test_provenance.py` — test chain intact; test tamper breaks at correct index

#### Day 4 (Thu) — SystemState + BaseAgent
- AM: Write `aadmf/core/state.py` — `SystemState` TypedDict + `BatchResult` dataclass
- PM: Write `aadmf/agents/base.py` — `BaseAgent` abstract class

#### Day 5 (Fri) — Integration + Novelty Log
- AM: Write a minimal `poc_v1.py` that: creates simulator → logs 3 batches → verifies provenance chain
- PM: Start `NOVELTY_LOG.md` — document your EMA enhancement to Page-Hinkley + your reasoning

**Week 1 Done Criteria:**
- [ ] GitHub repo live with correct folder structure
- [ ] `simulator.py` generates batches with and without drift; unit test passes
- [ ] `dict_chain.py` logs events; `verify_integrity()` returns True; tamper test breaks at correct event
- [ ] `NOVELTY_LOG.md` started with at least 2 documented design decisions
- [ ] All code committed and pushed

---

### Week 2 — Drift Detector + Planner Agent

**Goal:** Drift detected correctly; Planner selects algorithm based on scoring matrix.

#### Day 1 (Mon) — Page-Hinkley Drift Detector
- AM: Write `aadmf/drift/page_hinkley.py` — implement your EWMA-enhanced PH test
- PM: Write unit test: inject zero drift → score stays near 0; inject step shift → score rises

#### Day 2 (Tue) — Drift Detector Tuning
- AM: Plot drift_score over 10 synthetic batches with drift at batch 4 — verify visual rise
- PM: Experiment with α values (0.99, 0.999, 0.9999); document chosen value in Novelty Log

#### Day 3 (Wed) — PlannerAgent + Scoring Matrix
- AM: Write `aadmf/agents/planner.py` — `PlannerAgent` with scoring matrix + EMA accuracy update
- PM: Write unit test: drift_score=0 → StatisticalRules chosen; drift_score=0.8 → IsolationForest chosen

#### Day 4 (Thu) — Algorithm Registry
- AM: Add all 3 algorithms to registry with documented drift_weight and cost values
- PM: Test `update_accuracy()` — verify EMA converges correctly over 10 calls

#### Day 5 (Fri) — End-to-End Test Phase 1 Agents
- AM: Write integration test: 5 batches → drift at batch 3 → verify planner switches algorithm
- PM: Commit all; update Novelty Log with scoring matrix design decisions + weight justification

**Week 2 Done Criteria:**
- [ ] `page_hinkley.py` detects drift within 1 batch of injection in unit test
- [ ] `planner.py` selects IsolationForest when drift_score > 0.5
- [ ] Accuracy EMA updates correctly after each batch
- [ ] Integration test: algorithm switches from StatisticalRules to IsolationForest after drift injection
- [ ] All code committed

---

### Week 3 — Miner Agent + All 3 Mining Algorithms

**Goal:** All 3 mining algorithms working; quality_score returned; feeds back to Planner.

#### Day 1 (Mon) — BaseMiner + IsolationForest Wrapper
- AM: Write `aadmf/mining/base.py` — `BaseMiner` abstract class with `mine(X, drift_score) → dict`
- PM: Write `aadmf/mining/isolation_forest.py` — IsolationForest wrapper with adaptive contamination

#### Day 2 (Tue) — DBSCAN Wrapper
- AM: Write `aadmf/mining/dbscan.py` — DBSCAN wrapper with adaptive eps
- PM: Unit test: high drift → tighter eps → more noise points detected

#### Day 3 (Wed) — StatisticalRules Miner
- AM: Write `aadmf/mining/statistical_rules.py` — hand-coded Pearson correlation rules
- PM: Unit test: correlated synthetic features → at least 1 rule found

#### Day 4 (Thu) — MinerAgent Dispatcher
- AM: Write `aadmf/agents/miner.py` — `MinerAgent` that dispatches to correct mining wrapper
- PM: Test full dispatch: for each of 3 chosen algorithms, verify correct miner called + quality_score ∈ [0,1]

#### Day 5 (Fri) — Full Phase 1 Pipeline Test
- AM: Write `poc_v2.py`: stream → drift detect → planner → miner → provenance log per batch
- PM: Run 10 batches; print results table; commit

**Week 3 Done Criteria:**
- [ ] All 3 mining wrappers return `quality_score ∈ [0, 1]`
- [ ] `MinerAgent` dispatches correctly for all 3 algorithm choices
- [ ] Adaptive contamination / adaptive eps logic verified
- [ ] `poc_v2.py` runs 10 batches end-to-end without errors
- [ ] Provenance chain has correct number of events after full run

---

### Week 4 — Hypothesizer + Validator + Full PoC

**Goal:** Hypotheses generated and validated; complete PoC running; PoC report written.

#### Day 1 (Mon) — HypothesizerAgent
- AM: Write `aadmf/agents/hypothesizer.py` — full implementation of your correlation+MI template
- PM: Unit test: inject correlated features (sensor_0 = sensor_3 + noise) → at least 1 hypothesis generated

#### Day 2 (Tue) — HypothesizerAgent Tuning
- AM: Test trigger condition (drift_score > 0.1); verify 0 hypotheses in stable regime
- PM: Test max_per_batch=3 cap; verify ranking by |r|; document template logic in Novelty Log

#### Day 3 (Wed) — ValidatorAgent
- AM: Write `aadmf/agents/validator.py` — chi-square + synthetic augmentation validation
- PM: Unit test: known correlated features → valid=True; uncorrelated features → valid=False

#### Day 4 (Thu) — Full PoC Assembly
- AM: Write final `poc.py` — complete orchestrator loop connecting all 7 components
- PM: Run full 10-batch PoC; verify: 42 events logged; tamper test works; results table printed

#### Day 5 (Fri) — PoC Documentation
- AM: Write `experiments/notebooks/poc_results.ipynb` — charts of drift_score, algorithm selection, hypothesis count
- PM: Update `README.md` with quick start; record screen demo video for GitHub; commit everything

**Week 4 Done Criteria:**
- [ ] `poc.py` runs without errors from cold start: `python poc.py`
- [ ] Hypotheses generated when drift injected; 0 in stable regime
- [ ] At least 1 valid hypothesis (p < 0.05) over full run
- [ ] Provenance chain: `verify_integrity()=True`; tamper test breaks at correct event
- [ ] Screen demo video recorded and linked in README
- [ ] All code committed + tagged `v0.1-poc`

---

## PHASE 2 — REAL DATA + AGENTS (Weeks 5–10)

---

### Week 5 — UCI Gas Sensor Dataset Integration

**Goal:** Real dataset loading; streaming simulation on real data.

#### Day 1–2 — Dataset Download & Exploration
- Download UCI Gas Sensor Array Drift Dataset (10 batch files)
- Load in Jupyter notebook; check: shape, feature distributions, batch-by-batch drift patterns
- Visualise mean of sensor_0 across batches → should show gradual shift

#### Day 3 — UCI Loader
- Write `aadmf/streaming/uci_loader.py` — `UCIGasSensorLoader` class
- Loads batch CSVs; normalises features; returns same interface as `StreamingSimulator`

#### Day 4 — Cross-Validation with Synthetic
- Run `poc.py` with `UCIGasSensorLoader` instead of `StreamingSimulator`
- Compare drift_score curves: real vs synthetic — should show similar patterns

#### Day 5 — Tune PH Parameters on Real Data
- Experiment with δ and λ values on real dataset; find settings that detect drift between batch 3 and 5
- Document final values in `config.yaml` and Novelty Log

**Week 5 Done Criteria:**
- [ ] UCI dataset loaded and preprocessed
- [ ] `poc.py` runs with real data without errors
- [ ] Drift detected in real dataset between batches 3–6
- [ ] Tuned PH parameters documented

---

### Week 6 — Scoring Matrix v2 + Algorithm Tuning

**Goal:** Scoring matrix performs better on real data; all 3 miners tuned for real sensor data.

#### Day 1–2 — Scoring Weight Experiments
- Run 5 weight configurations on real data; log quality_score for each
- Select best configuration; update `config.yaml`; document in Novelty Log

#### Day 3–4 — Miner Tuning on Real Data
- Tune IsolationForest contamination range on real data
- Tune DBSCAN eps range; test adaptive eps formula
- Log results: which algorithm performs best at which drift level

#### Day 5 — Scoring Matrix v2
- If needed: add 4th algorithm (Apriori) to registry with tuned values
- Write `experiments/results/scoring_matrix_v2_results.csv`

**Week 6 Done Criteria:**
- [ ] Best scoring weights documented with experimental evidence
- [ ] All miners tuned for real UCI data
- [ ] Scoring matrix v2 implemented if changes made

---

### Week 7 — Hypothesizer v2 + Real MI

**Goal:** Replace MI proxy with real mutual information; test on real data.

#### Day 1–2 — Real Mutual Information
- Implement real MI using `sklearn.metrics.mutual_info_score` on discretised features
- Compare with proxy: are the same pairs identified?

#### Day 3 — MI Decay Hypothesis (Phase 2 Enhancement)
- Implement ΔMI tracking: `MI_history` dict keyed by feature pair
- Generate hypothesis if ΔMI > threshold even when |r| is moderate

#### Day 4–5 — Test on Real Data
- Run full pipeline on UCI data; count hypotheses per batch
- Validate all hypotheses; compute HVR; target > 70%

**Week 7 Done Criteria:**
- [ ] Real MI implemented and tested
- [ ] ΔMI tracking working
- [ ] HVR computed on real data; documented

---

### Week 8 — LangGraph Integration

**Goal:** Agents wrapped as LangGraph nodes; state graph working.

#### Day 1–2 — LangGraph Setup
- `pip install langgraph langchain`
- Study: LangGraph StateGraph, add_node, add_edge, add_conditional_edges
- Write minimal test: 2 nodes passing state → works

#### Day 3 — Wrap Agents as LangGraph Nodes
- Write `aadmf/orchestrator/langgraph_flow.py`
- Each agent's `run()` method becomes a LangGraph node
- Test: `SystemState` TypedDict passes through all nodes correctly

#### Day 4 — Conditional Edges
- Add conditional edge: mine → hypothesize (if drift > threshold) OR mine → log_batch
- Test: at drift=0 → hypothesizer skipped; at drift=0.5 → hypothesizer runs

#### Day 5 — Full LangGraph Run
- Replace manual orchestrator in `poc.py` with LangGraph flow
- Verify: same results as manual orchestrator; same provenance events

**Week 8 Done Criteria:**
- [ ] LangGraph StateGraph with all 5 agent nodes working
- [ ] Conditional drift-triggered edge working
- [ ] Results identical to Phase 1 manual orchestrator
- [ ] Code committed

---

### Week 9 — Neo4j Provenance Backend

**Goal:** Neo4j replacing dict-chain for provenance storage; graph queries working.

#### Day 1 — Neo4j Setup
- Install Neo4j Community Edition (free); start local instance on bolt://localhost:7687
- `pip install neo4j`
- Create database; set password; test connection from Python

#### Day 2–3 — Neo4j Logger
- Write `aadmf/provenance/neo4j_graph.py` — `Neo4jLogger` implementing same interface as `DictChainLogger`
- Map event types to Neo4j node labels; map event flow to relationships

#### Day 4 — Provenance Queries
- Write Cypher queries for: "show all decisions in batch 5", "show hypothesis lineage", "find all IsolationForest decisions"
- Expose as `ProvenanceLogger.query(cypher_string)` method

#### Day 5 — Switch and Verify
- Run full pipeline with Neo4j backend; verify same events logged
- View provenance graph in Neo4j Browser (visual graph)
- Screenshot for thesis

**Week 9 Done Criteria:**
- [ ] Neo4j running locally; Python driver connected
- [ ] All 7 event types logged as nodes with correct properties
- [ ] 3 Cypher queries working
- [ ] Provenance graph screenshot taken
- [ ] Hash-chain integrity maintained in Neo4j (prev_hash stored as property)

---

### Week 10 — Ollama + Phi-3 Mini LLM Integration

**Goal:** LLM phrasess hypotheses in natural language; hallucination guard working.

#### Day 1 — Ollama Setup
- Install Ollama; pull Phi-3 Mini: `ollama pull phi3:mini`
- Test: `ollama run phi3:mini "Write one sentence about sensor_0 correlation 0.42"`

#### Day 2–3 — LLM Hypothesizer
- Write `aadmf/llm/ollama_client.py` — wrapper around Ollama Python client
- Integrate into `HypothesizerAgent`: if `config.use_llm=True` → call LLM phrasing

#### Day 4 — Hallucination Guard
- Implement safety checks: feature names present, correlation value present, length < 50 words
- Test: deliberately give bad prompt → guard triggers → fallback to template

#### Day 5 — Evaluation
- Run 10 batches with LLM phrasing; assess quality manually
- Compare template vs LLM phrasing; document findings in Novelty Log

**Week 10 Done Criteria:**
- [ ] Phi-3 Mini running locally via Ollama
- [ ] LLM phrasing integrated with config toggle
- [ ] Hallucination guard passing all test cases
- [ ] Example LLM outputs documented

---

## PHASE 3 — DASHBOARD + EXPERIMENTS (Weeks 11–16)

---

### Week 11 — Streamlit Dashboard — Core

**Goal:** Working dashboard showing live drift + algorithm selection.

#### Day 1–2 — App Skeleton
- Write `aadmf/dashboard/app.py` — Streamlit app with sidebar config + main content area
- Sections: Live Stream Panel, Hypothesis Feed, Provenance Graph, Metrics Panel

#### Day 3 — Drift Chart
- Write `aadmf/dashboard/charts.py` — Plotly line chart of `drift_score` per batch
- Add red threshold line at 0.1; colour-code: green (stable) / red (drift)

#### Day 4 — Algorithm Selection Chart
- Add bar chart: algorithm selection count over all batches
- Add table: per-batch breakdown (batch, drift_score, algo_chosen, quality_score)

#### Day 5 — Run and Demo
- Run `streamlit run aadmf/dashboard/app.py`; test all charts update correctly
- Screenshot for thesis

**Week 11 Done Criteria:**
- [ ] Streamlit app runs without errors
- [ ] Drift score chart renders with correct values
- [ ] Algorithm selection bar chart renders
- [ ] Per-batch table renders

---

### Week 12 — Dashboard — Hypothesis Feed + Provenance Graph

**Goal:** Hypotheses visible in dashboard; provenance graph visualised.

#### Day 1–2 — Hypothesis Feed
- Add hypothesis panel: scrollable list; each entry shows feature pair, r value, confidence badge (colour-coded)
- Valid hypotheses: green badge; Invalid: red badge

#### Day 3–4 — Provenance Graph Visualisation
- `pip install pyvis networkx`
- Write `aadmf/dashboard/graph_viz.py` — converts provenance chain to NetworkX graph → pyvis HTML
- Embed pyvis HTML in Streamlit using `st.components.v1.html`

#### Day 5 — Tamper Demo Button
- Add "Inject Tamper" button: modifies one event; shows chain break at event index
- Add "Reset Chain" button: restores original chain

**Week 12 Done Criteria:**
- [ ] Hypothesis feed renders with colour-coded badges
- [ ] Provenance graph renders as interactive network in dashboard
- [ ] Tamper demo button works; shows break event index
- [ ] Screen recording of full dashboard made

---

### Week 13 — Ablation Study Runner

**Goal:** Automated ablation runner producing CSV results.

#### Day 1–2 — Ablation Framework
- Write `aadmf/evaluation/ablation.py` — `AblationRunner` class
- Implements 5 variants: full, no_scoring_matrix, no_hypothesizer, no_provenance, static_baseline

#### Day 3 — Metrics Module
- Write `aadmf/evaluation/metrics.py` — all metric calculations
- `compute_quality_score_mean`, `compute_hvr`, `compute_drift_latency`, `compute_runtime`

#### Day 4–5 — Run All Experiments
- Run 125 experiments (5 variants × 5 seeds × 5 drift levels)
- Save to `experiments/results/ablation_results.csv`
- Runtime: ~3–5 minutes total

**Week 13 Done Criteria:**
- [ ] `ablation.py` runner works for all 5 variants
- [ ] 125 experiments completed without errors
- [ ] `ablation_results.csv` saved with all columns
- [ ] Quick sanity check: full variant outperforms static baseline

---

### Week 14 — Results Analysis + Visualisation

**Goal:** Results analysed; charts ready for thesis.

#### Day 1–2 — Analysis Notebook
- Write `experiments/notebooks/ablation_analysis.ipynb`
- Load CSV; compute mean ± std for each variant × drift level
- Wilcoxon test: AADMF full vs static baseline

#### Day 3 — Charts for Thesis
- Chart 1: Quality score vs drift level (line chart; 5 variants)
- Chart 2: Hypothesis validity rate bar chart
- Chart 3: Algorithm selection distribution (stacked bar)
- Chart 4: Runtime overhead (AADMF vs baseline)

#### Day 4–5 — Refine and Document
- Ensure all charts are publication-quality (labels, legends, gridlines)
- Export as PNG + SVG for thesis
- Write `experiments/results/key_findings.md` — bullet summary of results

**Week 14 Done Criteria:**
- [ ] Wilcoxon test confirms AADMF vs baseline is significant (p < 0.05)
- [ ] 4 publication-quality charts exported
- [ ] Key findings documented in 1 page

---

### Week 15–16 — System Polish + Demo Preparation

**Goal:** Production-grade code; impressive demo ready.

#### Week 15
- Code review: add docstrings to all classes and methods
- Add `requirements.txt` (PoC) and `requirements_full.txt` (full system)
- Write `INSTALL.md` with step-by-step setup guide (5 minutes to first run)
- Add type hints to all agent method signatures
- Fix any remaining bugs found during demo prep

#### Week 16
- Record 5-minute screen demo video: start → streaming → drift detected → hypothesis generated → provenance graph
- Write GitHub README with: badges, architecture diagram (draw.io export), quick start, demo GIF
- Tag repo: `v1.0-full`
- Share repo link with supervisor for pre-thesis review

**Weeks 15–16 Done Criteria:**
- [ ] All methods have docstrings
- [ ] `INSTALL.md` tested on fresh machine (or fresh venv)
- [ ] Demo video recorded (5 min)
- [ ] GitHub README with architecture diagram and demo GIF
- [ ] Tagged `v1.0-full`

---

## PHASE 4 — THESIS + PATENT (Weeks 17–24)

---

### Week 17 — Thesis Chapter 1: Introduction + Chapter 2: Literature Review

**Chapter 1 (~2,000 words):**
- Problem statement (3 gaps in existing systems)
- Research questions (3 specific RQs mapped to your 3 patent claims)
- Contributions (bullet list: scoring matrix, hypothesis template, provenance chain)
- Thesis structure

**Chapter 2 (~4,000 words) — Literature Review:**
Survey at least 10 papers across these areas:

| Area | Key Papers to Review |
|---|---|
| Data stream mining | Gama et al. (2014) "Survey on concept drift adaptation" |
| Drift detection | Gama et al. (2004) "Learning with drift detection" (DDM) |
| Agentic AI | Park et al. (2023) "Generative agents" |
| Multi-agent data mining | Cao (2010) "Scalable ML with agents" |
| Data provenance | Moreau et al. (2013) "PROV-DM model" |
| Hypothesis generation | King et al. (2009) "Robot scientist" |

**Gap Analysis table:** For each existing system, show which of your 3 components it is missing.

---

### Week 18 — Thesis Chapter 3: System Design

**Chapter 3 (~3,000 words):**
- Architecture diagram (draw.io — Layer 1 through Layer 7)
- Agent design decisions and rationale
- Scoring matrix derivation (show the math)
- Hypothesis template formal definition
- Provenance hash chain mechanism
- Data flow diagram

Write from your TDD Document 1. All design decisions should reference Novelty Log entries.

---

### Week 19 — Thesis Chapter 4: Implementation

**Chapter 4 (~3,000 words):**
- Tech stack justification table (why each library was chosen)
- Phase 1 vs Phase 2 comparison
- Key code snippets (scoring matrix, hypothesis template, hash chain) — in pseudocode for thesis
- Challenges encountered and solutions
- PoC execution proof table (from SRS Appendix)

---

### Week 20 — Thesis Chapter 5: Evaluation

**Chapter 5 (~4,000 words):**
- Experimental setup (hardware, dataset, parameters)
- Ablation study results (all 4 charts)
- Statistical significance (Wilcoxon test result)
- Hypothesis validity rate analysis
- Provenance integrity results (tamper test)
- Runtime overhead analysis
- Discussion: why AADMF outperforms baseline (link to algorithm design)
- Limitations (honest assessment)

---

### Week 21 — Thesis Chapter 6 + Abstract + References

**Chapter 6 (~1,500 words):**
- Summary of contributions (3 claims)
- Answers to research questions
- Future work (6 concrete directions)
- Patent filing status

**Abstract** (~300 words): Write last; covers problem, method, results, contribution.

**References:** Use IEEE citation format. Target 30–40 references.

---

### Week 22 — Patent Draft

**Provisional Patent Application (Form 1 & Form 2 — India):**

**Form 2 — Complete Specification Draft:**

**Title:** "Method and System for Autonomous Hypothesis-Driven Pattern Discovery in Streaming Data with Tamper-Proof Provenance Using Multi-Agent Architecture"

**Abstract** (~150 words): Summarise the integrated system.

**Claims to draft (with supervisor + IPR cell):**

```
Claim 1 (Independent — Method):
A computer-implemented method for autonomous algorithm selection
in adaptive data mining comprising:
  (a) computing a drift score for a streaming data batch using an
      exponentially-weighted Page-Hinkley test;
  (b) computing a selection score for each candidate mining algorithm
      using a weighted scoring matrix comprising drift weight,
      accuracy history weight, and computational cost weight;
  (c) selecting the algorithm with highest selection score;
  (d) updating the accuracy history using exponential moving average
      of the returned quality score.

Claim 2 (Independent — System):
A system for hypothesis generation in streaming data comprising:
  (a) a hypothesizer agent that generates candidate hypotheses
      by identifying feature pairs satisfying both a Pearson
      correlation threshold and a mutual information threshold;
  (b) a validator agent that validates each hypothesis using
      chi-square independence test on both real and
      Gaussian-perturbed synthetic data.

Claim 3 (Independent — Method):
A computer-implemented method for tamper-proof provenance tracking
comprising:
  (a) creating a hash-chained event ledger where each event includes
      the SHA-256 hash of its predecessor;
  (b) verifying chain integrity by recomputing each hash and
      detecting any mismatch as evidence of tampering.

Claim 4 (Dependent on Claims 1–3):
The integrated framework of Claims 1–3 wherein all three methods
operate cooperatively on the same streaming data pipeline.
```

**Evidence to attach:** PoC execution output; Novelty Log excerpts; ablation study results.

---

### Week 23 — Thesis Revision + Supervisor Feedback

- Submit complete thesis draft to supervisor
- Address feedback; revise all chapters
- Final proofread (Grammarly + manual)
- Format check: page numbers, headings, figure captions, table numbers
- Ensure all figures are referenced in text

---

### Week 24 — Final Submission + Patent Filing + arXiv

#### Final Week Tasks:
- **Monday:** Submit thesis (hard copy + digital per university format)
- **Tuesday:** File Provisional Patent (Form 1 + Form 2) at university IPR cell
- **Wednesday:** Prepare arXiv submission (6-page IEEE format paper from thesis Chapter 3+5)
- **Thursday:** Submit arXiv preprint; make GitHub repo public
- **Friday:** Update LinkedIn/resume; send supervisor thank-you

---

## Progress Tracking Template

Copy this into your `PROGRESS.md` and update weekly:

```markdown
# AADMF Progress Tracker

## Phase 1 — Foundation & PoC
- [x] Week 1: Environment + Simulator + Provenance
- [ ] Week 2: Drift Detector + Planner Agent
- [ ] Week 3: Miner Agent + Mining Algorithms
- [ ] Week 4: Hypothesizer + Validator + Full PoC

## Phase 2 — Real Data + Agents
- [ ] Week 5: UCI Dataset Integration
- [ ] Week 6: Scoring Matrix v2
- [ ] Week 7: Hypothesizer v2 + Real MI
- [ ] Week 8: LangGraph Integration
- [ ] Week 9: Neo4j Provenance Backend
- [ ] Week 10: Ollama + Phi-3 LLM

## Phase 3 — Dashboard + Experiments
- [ ] Week 11: Streamlit Dashboard Core
- [ ] Week 12: Dashboard Hypothesis + Provenance Graph
- [ ] Week 13: Ablation Study Runner
- [ ] Week 14: Results Analysis
- [ ] Week 15–16: Polish + Demo

## Phase 4 — Thesis + Patent
- [ ] Week 17: Ch1 + Ch2
- [ ] Week 18: Ch3
- [ ] Week 19: Ch4
- [ ] Week 20: Ch5
- [ ] Week 21: Ch6 + Abstract + References
- [ ] Week 22: Patent Draft
- [ ] Week 23: Revision
- [ ] Week 24: Final Submission
```

---

*AADMF — Sprint Plan v1.0 | Document 3 of 5*