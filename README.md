# AADMF

Adaptive Agentic Drift Mining Framework (AADMF) is a multi-agent Python system for:

- Streaming-batch drift detection
- Adaptive mining algorithm selection
- Hypothesis generation and validation
- Tamper-evident provenance logging
- Optional LangGraph orchestration, Neo4j provenance storage, and Ollama-based phrasing

This README gives a complete execution procedure from environment setup to running core flows, tests, dashboard, and optional components.

---

## Full Project Execution (Step-by-Step)

Use this section as the run order for the project. Each step includes the purpose so you know why it is required.

### Step 1: Open the repository root

Move into the project folder before running anything else.

```powershell
cd C:\Users\santh\OneDrive\Desktop\aadmf\aadmf-main
```

Why this matters: all paths in the project, including `config.yaml`, `provenance.json`, and the dashboard, are resolved relative to the repository root.

### Step 2: Create and activate the virtual environment

Set up an isolated Python environment for this project.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Why this matters: the project depends on several third-party packages, and the virtual environment keeps them separate from your system Python.

### Step 3: Install dependencies

Install the libraries used by the pipeline, tests, dashboard, and optional integrations.

```powershell
pip install -r requirements.txt
```

Why this matters: without these packages, the orchestrator, mining algorithms, dashboard, or tests may fail to import.

### Step 4: Configure the dataset location

Update `config.yaml` so the loader can find your UCI batch files.

```yaml
uci_loader:
	data_dir: dataset1
	batch_numbers: null
	normalize: false
	use_ucimlrepo: true

uci_streaming:
	data_dir: dataset1
	batch_numbers: null
	normalize: false
	use_ucimlrepo: true

drift_detection:
	delta: 0.0
	threshold: 0.05
	alpha: 0.99

drift_detector:
	method: page_hinkley
	threshold: 0.05
	alpha: 0.99
	min_batch_size: 100
	use_relative_change: true

isolation_forest:
	base_contamination: 0.05
	adaptive_formula: fixed
	seed: 42

dbscan:
	base_eps: 2.0
	adaptive_formula: fixed
	min_samples: 5

statistical_rules:
	corr_threshold: 0.2
```

Why this matters: the pipeline needs a valid batch source. If the local files are missing, the fallback UCI download path can still be used when `use_ucimlrepo` is `true`.

### Step 5: Choose the orchestration mode

Select whether the project runs through the manual orchestrator or the LangGraph flow.

```yaml
execution:
	mode: full
	use_langgraph: true
```

Why this matters: `use_langgraph: true` exercises the graph-based execution path, which is the recommended full validation mode.

### Step 6: Run the main pipeline

Start the full end-to-end mining run.

```powershell
python poc.py
```

Why this matters: this generates the batch-by-batch results, updates `provenance.json`, and produces the data that the dashboard reads.

### Step 7: Run the loader smoke test

Check that the dataset can be loaded and processed for a short run.

```powershell
python test_uci_loader.py
```

Why this matters: this is the fastest way to confirm the data source and basic orchestration are working before spending time on the full pipeline.

### Step 8: Run the LangGraph parity test

Verify that the LangGraph path behaves as expected and matches the manual flow on key outputs.

```powershell
python test_langgraph.py
```

Why this matters: this confirms the graph-based orchestration is routing correctly and not changing the core batch results.

### Step 9: Run the unit tests

Execute the planner unit test suite.

```powershell
python -m pytest tests/unit/test_planner.py -q
```

Why this matters: the planner decides which mining algorithm is selected, so this test protects one of the main decision points in the system.

### Step 10: Launch the dashboard

Open the Streamlit UI that visualizes drift, algorithm selection, hypotheses, and provenance.

```powershell
streamlit run aadmf/dashboard/app.py
```

Why this matters: the dashboard is the easiest way to confirm that the pipeline produced usable provenance events and results.

### Step 11: Confirm the dashboard data sources

In the dashboard sidebar, make sure the file paths point to the project outputs.

- Provenance JSON path: `provenance.json`
- Config YAML path: `config.yaml`

Why this matters: if the dashboard points to the wrong files, it can appear empty even when the pipeline already produced data.

### Step 12: Verify the dashboard sections

Check that all five sections appear and contain data when the pipeline has run.

1. Live drift score chart.
2. Algorithm selection frequency chart.
3. Hypothesis feed.
4. Provenance graph.
5. Tamper demo status and button.

Why this matters: this is the visual proof that the run completed correctly and that the provenance chain is available for inspection.

### Step 13: Run the tamper demo

Use the dashboard button to simulate a provenance modification.

Why this matters: the tamper demo shows that the hash chain detects edits and that the audit trail is actually being enforced.

Expected behavior:

1. The chain starts as intact.
2. The tampered version reports a broken integrity check.

### Step 14: Optional Ollama hypothesis phrasing

Enable this only if you want generated hypothesis wording from Ollama.

```powershell
ollama pull phi3:mini
python test_llm_hypothesis.py
```

```yaml
hypothesizer:
	use_llm: true

llm:
	provider: ollama
	model: phi3:mini
	temperature: 0.2
```

Why this matters: this path is optional and only affects how hypotheses are phrased, not the mining or drift logic.

### Step 15: Optional Neo4j provenance backend

Use this only if you want provenance stored in Neo4j instead of the local hash chain.

```yaml
provenance:
	backend: neo4j
	export_path: provenance.json
	neo4j:
		uri: bolt://localhost:7687
		user: neo4j
		password: your_password_here
		enabled: true
```

```powershell
python poc.py
```

Why this matters: Neo4j is optional. If it is unavailable, the pipeline now prints a clearer install/start hint and falls back to the local dict-chain logger.

### Step 16: Optional week-6 tuning workflow

Run this when you want to reproduce the tuning and recommendation outputs.

```powershell
python tune_week6.py
```

Why this matters: this script generates the tuning CSVs and recommendation document used for analysis and reporting.

Expected outputs:

1. `experiments/results/scoring_matrix_tuning.csv`
2. `experiments/results/miner_tuning_results.csv`
3. `experiments/results/week6_final_recommendation.md`

### Step 17: Run everything in order

If you want a compact command sequence, run these from the repository root after installing dependencies.

```powershell
python poc.py
python test_uci_loader.py
python test_langgraph.py
python -m pytest tests/unit/test_planner.py -q
streamlit run aadmf/dashboard/app.py
```

Why this matters: this is the shortest practical sequence for validating the pipeline, tests, and dashboard together.

---

## Demonstration of Adaptive Behavior

Use this checklist when presenting the system behavior in a live demo.

### What to run

```powershell
python poc.py
streamlit run aadmf/dashboard/app.py
```

### What to verify in output

1. Drift score is non-zero on selected batches (spikes are expected in later UCI batches).
2. Planner does not stay fixed on one miner; it should switch between `StatisticalRules` and `IsolationForest` as drift changes.
3. Hypothesis feed includes batch context and validation state for each generated statement.
4. Provenance integrity check remains intact before tamper demo and fails after tampering.

### Current tuned settings applied

1. `IsolationForest`: `base_contamination=0.05`, `adaptive_formula=fixed`
2. `DBSCAN`: `base_eps=2.0`, `adaptive_formula=fixed`
3. `StatisticalRules`: `corr_threshold=0.2`

These values are derived from Week 6 experiment artifacts in `experiments/results/week6_final_recommendation.md` and `experiments/results/miner_tuning_results.csv`.

### Demonstration Results (Latest Successful Run)

Latest `python poc.py` run confirms adaptive behavior is active:

1. Drift spikes detected on batches **4, 5, 8, and 9** (`drift_detected=True`, `drift_score=1.000000`).
2. Planner switched algorithms by regime:
	- `StatisticalRules`: 6 batches (stable/low drift)
	- `IsolationForest`: 4 batches (high drift)
3. Average quality score: **0.9398**.
4. Hypotheses generated: **30**; valid hypotheses: **30**; HVR: **1.0000**.
5. Provenance integrity check remains intact (`intact=True`) before tamper demo.

This run demonstrates end-to-end adaptation: drift detection influences planner choice, miner behavior changes by batch, and validated hypotheses/provenance remain consistent.

---

## 1) Project Structure (Quick Orientation)

- `poc.py`: Main entry point for end-to-end execution
- `config.yaml`: Runtime configuration (streaming, drift, planner, LLM, provenance)
- `aadmf/`: Framework package (agents, orchestrators, mining, provenance, dashboard)
- `requirements.txt`: Pinned dependency set for setup and reproducibility
- `test_uci_loader.py`: UCI loader + orchestrator smoke test
- `test_langgraph.py`: Manual vs LangGraph parity and routing test
- `test_llm_hypothesis.py`: Template vs Ollama-phrased hypothesis comparison
- `tune_week6.py`: Week-6 tuning workflow and recommendation report generation
- `experiments/results/`: Generated CSV and markdown outputs

---

## 2) Prerequisites

### Required

1. Python 3.11+
2. `pip` (latest recommended)
3. Git

### Optional (only if you use these features)

1. Neo4j 5.x (for graph provenance backend)
2. Ollama (for LLM hypothesis phrasing)
3. scikit-learn extras already included through `requirements.txt`

---

## 3) Setup Environment (Windows PowerShell)

From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install core dependencies used by this codebase:

```powershell
pip install -r requirements.txt
```

Notes:

- `requirements.txt` is pinned so the dashboard, agents, and tests all use the same versions.
- `ollama`, `neo4j`, and `pyvis` are optional if you do not use those paths.

---

## 4) Dataset Configuration and Loading

AADMF supports two data paths:

1. Local UCI batch files (`batch1.dat ... batch10.dat`)
2. Automatic UCI fallback via `ucimlrepo` (dataset id `270`)

Current config behavior:

- `poc.py` defaults full mode to UCI dataset.
- `config.yaml` currently points `uci_loader.data_dir` to `data/raw`.

The current implementation also auto-resolves common local folders such as `dataset1` and `dataset2` before falling back to `ucimlrepo`.

### Recommended local setup (already common in this repo)

If your batches are under `dataset1/`, the following config is the recommended default:

```yaml
uci_loader:
	data_dir: dataset1
	batch_numbers: null
	normalize: false
	use_ucimlrepo: true

uci_streaming:
	data_dir: dataset1
	batch_numbers: null
	normalize: false
	use_ucimlrepo: true

drift_detector:
	method: page_hinkley
	threshold: 0.05
	alpha: 0.99
	min_batch_size: 100
	use_relative_change: true

isolation_forest:
	base_contamination: 0.05
	adaptive_formula: fixed
	seed: 42

dbscan:
	base_eps: 2.0
	adaptive_formula: fixed
	min_samples: 5

statistical_rules:
	corr_threshold: 0.2
```

If you keep `data/raw` and files are missing, loader attempts `ucimlrepo` fallback when `use_ucimlrepo: true`.

---

## 5) Main End-to-End Run

Run the full pipeline:

```powershell
python poc.py
```

What this does internally:

1. Loads `config.yaml`
2. Chooses execution mode (`full` defaults to UCI)
3. Builds streamer from config (`UCIGasSensorLoader` or synthetic simulator)
4. Runs orchestrator (manual wrapper; can route to LangGraph)
5. Prints per-batch results table and summary metrics
6. Exports provenance JSON (default: `provenance.json`)

Expected output highlights:

- Drift score and drift flags per batch
- Selected algorithm and quality score
- Hypothesis counts and validated hypothesis counts
- Provenance integrity summary

---

## 6) Control Execution Mode (Manual vs LangGraph)

You can explicitly control orchestration path via `config.yaml`:

```yaml
execution:
	mode: full            # full or debug
	use_langgraph: true   # true -> LangGraph flow, false -> manual loop
```

Behavior summary:

- If `use_langgraph` is absent, `mode: full` implies LangGraph usage.
- Manual orchestrator path remains available for compatibility and debugging.

---

## 7) Run Validation and Test Scripts

### A) UCI loader smoke + 3-batch orchestrator run

```powershell
python test_uci_loader.py
```

Use this first to confirm:

- batches are loading correctly
- basic drift evidence appears
- pipeline can process a short sequence

### B) LangGraph parity + conditional routing

```powershell
python test_langgraph.py
```

Confirms:

- LangGraph and manual outputs match on key metrics
- low drift branch skips hypothesis/validation
- high drift branch executes hypothesis/validation

### C) Unit tests (planner)

```powershell
python -m pytest tests/unit/test_planner.py -q
```

---

## 8) Run Dashboard (Streamlit)

Start UI:

```powershell
streamlit run aadmf/dashboard/app.py
```

Dashboard sections include:

1. Live drift score line chart
2. Algorithm selection frequency bar chart
3. Hypothesis feed with confidence badges
4. Provenance graph visualization
5. Tamper-demo interaction

The provenance graph now uses `st.iframe` to avoid Streamlit component deprecation warnings.

In sidebar, verify paths:

- Provenance JSON path: `provenance.json`
- Config YAML path: `config.yaml`

Tip: run `python poc.py` first so the dashboard has fresh provenance events to display.

---

## 9) Optional: Enable Ollama-Based Hypothesis Phrasing

### Step 1: Install and start Ollama service

Install Ollama from official installer, then run:

```powershell
ollama pull phi3:mini
```

### Step 2: Enable LLM in config

```yaml
hypothesizer:
	use_llm: true

llm:
	provider: "ollama"
	model: "phi3:mini"
	temperature: 0.2
```

### Step 3: Run comparison script

```powershell
python test_llm_hypothesis.py
```

This script compares template statements vs LLM-phrased statements and exercises fallback guard behavior.

---

## 10) Optional: Enable Neo4j Provenance Backend

### Step 1: Start Neo4j and create/update credentials

Ensure Neo4j is reachable at configured URI (default `bolt://localhost:7687`).

### Step 2: Configure provenance backend in `config.yaml`

```yaml
provenance:
	backend: "neo4j"
	export_path: provenance.json
	neo4j:
		uri: "bolt://localhost:7687"
		user: "neo4j"
		password: "your_password_here"
		enabled: true
```

### Step 3: Execute pipeline

```powershell
python poc.py
```

If Neo4j is unavailable, orchestrator code falls back to in-memory dict-chain logger and prints a warning.

The warning now includes a concise setup hint: install the `neo4j` Python package, start Neo4j Desktop or Docker, and verify the Bolt URI.

---

## 11) Run Week-6 Tuning Workflow

```powershell
python tune_week6.py
```

Generates:

- `experiments/results/scoring_matrix_tuning.csv`
- `experiments/results/miner_tuning_results.csv`
- `experiments/results/week6_final_recommendation.md`

Use this for data-backed parameter recommendations under real UCI drift.

---

## 12) Typical End-to-End Command Sequence

If you want a practical "do this in order" flow:

```powershell
# 1) Environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# 2) Run core pipeline
python poc.py

# 3) Validate loader + graph routing
python test_uci_loader.py
python test_langgraph.py

# 4) Unit tests
python -m pytest tests/unit/test_planner.py -q

# 5) Dashboard
streamlit run aadmf/dashboard/app.py
```

---

## 13) Troubleshooting

### Issue: `FileNotFoundError` for UCI batches

Cause:

- `uci_loader.data_dir` points to a folder without `batch*.dat`

Fix:

1. Point `data_dir` to `dataset1` (or actual location)
2. Keep `use_ucimlrepo: true` for fallback

### Issue: Neo4j connection warning

Cause:

- Neo4j service not running, wrong URI, or wrong credentials

Fix:

1. Start Neo4j
2. Verify URI/user/password in `config.yaml`
3. Or set `provenance.backend: "dict"` to bypass Neo4j

### Issue: Ollama phrasing not used

Cause:

- Ollama service/model unavailable or `use_llm: false`

Fix:

1. `ollama pull phi3:mini`
2. Set `hypothesizer.use_llm: true`
3. Ensure `pip install ollama` is present in environment

### Issue: Dashboard empty

Cause:

- No recent provenance events

Fix:

1. Run `python poc.py`
2. Confirm dashboard path points to `provenance.json`

---

## 14) Output Artifacts You Should Expect

After successful runs, common outputs include:

- `provenance.json` (event chain export)
- Console batch summary table from orchestrator
- CSV files in `experiments/results/` from tuning/evaluation scripts
- Optional HTML graph files from Neo4j visualization scripts
- `PROJECT_SUMMARY.md` for a one-page architecture overview

---

## 15) License / Notes

Project currently focuses on executable research pipeline behavior and experiment reproducibility. If you plan distribution, add explicit dependency lock files (`requirements.txt` / `pyproject.toml`) and environment-specific run profiles.