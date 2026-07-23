# Monopath DAGs: Structuring Patient Trajectories from Clinical Case Reports

Monopath DAGs is a modular pipeline that converts clinical case reports into
structured, **causal-model-ready** representations of a patient's trajectory as
Directed Acyclic Graphs (DAGs). Each graph captures temporally ordered clinical
states and the objective changes between them, supporting causal modeling,
semantic retrieval, and synthetic case generation.

## The graph model

Each DAG encodes one patient's course over time:

- **Nodes = patient states.** A snapshot of the patient at a point in the
  narrative — a human-readable `content` field plus structured `clinical_data`
  (medications, labs, vitals, imaging, procedures, diagnoses, …), grounded in
  UMLS concepts where possible.

- **Edges = objective value-changes.** Each transition records *what changed and
  by how much* — e.g. heart rate 70→110 bpm, a lab crossing a threshold, or a
  categorical change such as a medication started. Edges carry both the
  narrative and a structured `change` (variable, direction, from/to,
  delta, unit).

### Branches, inverses, and a collapsible structure

The trajectory has a main **spine** of persistent changes (disease progression,
durable state transitions). Layered on top are **branches** — *ephemeral*
excursions that later resolve:

> A branch **opens** when an objective value moves (e.g. heart rate rises) and
> **closes** when an **inverse** edge reverses it (heart rate is brought back
> down). The opening edge and its inverse closing edge form a matched pair.

Matched pairs behave like balanced brackets and can be **collapsed**: collapsing
a branch hides the transient excursion and leaves the net effect on the spine.
Branches **nest** (a perturbation occurring inside another unresolved one),
producing a stack whose depth is recorded on every edge. This gives the DAG an
explicit, interpretable separation between **branch** edges (reversible,
transient) and **non-branch** edges (persistent), which is exactly the signal a
downstream causal model uses to reason about interventions and their recovery.

Branch/non-branch classification and inverse matching are LLM-driven; the DAG
structure, stack nesting, and collapse operations are deterministic Python so
they remain reproducible and testable.

## What's in this repository

- A DSPy-driven pipeline for extracting DAGs from PubMed Central (PMC) HTML case reports
- Ontology-grounded node and objective value-change edge generation via LLMs
- Branch detection with inverse-edge matching and a collapsible, stacked structure
- A synthetic generation module for producing realistic case narratives
- Evaluation utilities for semantic fidelity and structural correctness
- The full dataset of Monopath DAGs, extracted metadata, and synthetic cases

---

## Where the data comes from

The inputs are **open-access clinical case reports from PubMed Central (PMC)** —
not a single journal. `src/data/web/scrape.py` queries PubMed through the NCBI
Entrez API (Biopython) for lung-cancer case reports:

- **Topic:** lung cancer / lung carcinoma / NSCLC / SCLC / mesothelioma / pulmonary
  carcinoma / bronchogenic carcinoma (Title/Abstract)
- **Filters:** Publication Type = *Case Reports*, *free full text*, published 2019→present

It downloads each article's full-text **HTML** and records metadata in
`lung_cancer_case_reports.csv` (`PMID, PMC_ID, Title, Journal, PublicationDate,
Authors, Abstract, MeSH_Terms, URL, Citation, HTML_FILE`). The working input set
ships in `webapp/static/pmc_htmls/*.html`; point `--input_dir` at that folder (or
your own directory of PMC HTMLs). Re-scraping needs `ncbi_api_key` and `email`
environment variables.

## How it works: data → DAG

Per HTML file, `run_pipeline()` (`src/agent/dspy/graph_generation.py`) runs:

1. **Preprocess** — `preprocess_pmc_article_text` (BeautifulSoup) strips abstract,
   references/bibliography, nav/script/style; keeps body paragraphs + tables.
2. **Paragraphs → patient timeline** — the `PatientTimeline` LLM signature distills
   a patient-only, chronologically ordered narrative (labs/imaging kept verbatim).
3. **Nodes (patient states)** — sentence-chunked generation with a running memory
   that merges overlapping states, then per-node `clinical_data` extraction
   (meds/labs/vitals/imaging/diagnoses, UMLS-grounded).
4. **Edges (objective value-changes)** — one edge per adjacent node pair, each with
   a `change` (variable, direction, from/to, delta, unit) plus narrative.
5. **Branches (collapsible)** — `match_branches` pairs each ephemeral excursion with
   its inverse closing edge (LLM decides "is B the inverse of A?"; a deterministic
   Python stack in `src/graph/branch.py` owns nesting/`branch_depth`).
6. **Serialize** — `{nodes, edges}` JSON per graph + a row in `graph_metadata.csv`.

```
PMC HTML ─▶ preprocess ─▶ patient timeline ─▶ nodes(+clinical_data)
        ─▶ edges(+change) ─▶ branch matching ─▶ graph_<n>.json
```

---

## Installation

### First-time setup (fresh clone)

A clone contains **code only** — the case-report dataset, the Python venv, the
model weights, and the Ollama container are all git-ignored and are **not** in
the repo. Bootstrap the environment in one step (run it inside a compute
allocation, not the login node):

```bash
salloc -p dev -c 4 --mem 8G -t 60      # or: sh_dev
bash scripts/setup.sh
```

`scripts/setup.sh` creates `.venv` + installs dependencies, builds the local GPU
Ollama container (`.ollama_dist/ollama.sif` via Apptainer), and writes a
`.config/.env` template. It does **not** fetch the dataset — supply that
yourself:

- **Data:** put PMC case-report HTMLs in `webapp/static/pmc_htmls/` — either
  re-scrape via `src/data/web/scrape.py` (needs `ncbi_api_key` + `email`), or
  copy an existing corpus into that folder.

Then run the pipeline (see [PIPELINE.md](PIPELINE.md) and the run commands below).

> **On Sherlock (Stanford HPC):** never `pip install` or run generation on the
> login node. Use `sh_dev`/`salloc`, keep outputs on `$SCRATCH`, and keep API
> keys and caches off `$HOME`.

### Manual steps (if not using setup.sh)

```bash
git clone <repo-url> <project_directory> && cd <project_directory>
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Configuration

Store API keys and model names in `.config/.env` (git-ignored).

```
DSPY_MODEL="gemini/gemini-2.0-flash"
GEMINI_APIKEY="your-gemini-api-key"
GPTKEY="your-openai-api-key"
```

`DSPY_MODEL` may point at a local Ollama model (e.g. `ollama/llama3.1`); the
pipeline routes to `http://localhost:11434` automatically when the name
contains `ollama`.

---

## Graph generation pipeline

Converts PMC HTML case reports into DAGs.

**Input** — place HTML files in `./pmc_htmls/`.

```bash
python main.py generate-graphs --input_dir ./pmc_htmls --output_dir ./webapp/static/graphs
```

**Output**
- Graph JSONs: `webapp/static/graphs/`
- Metadata: `webapp/static/graphs/graph_metadata.csv`

### Backends: Gemini (key) or Ollama (local, no key)

Selected via `DSPY_MODEL`:

- **Gemini** — `DSPY_MODEL=gemini/gemini-2.0-flash` + `GEMINI_APIKEY` in `.config/.env`.
- **Ollama** — any `DSPY_MODEL=ollama_chat/<model>`; the pipeline auto-targets
  `http://localhost:11434` and needs **no API key**, only a running Ollama server
  with the model pulled.

### Running on Sherlock with Ollama (no API key)

Never run this on the login node — use a GPU allocation and keep the venv + model
weights on `$SCRATCH`:

```bash
salloc -p gpu -G 1 -c 8 --mem 32G -t 02:00:00      # GPU compute shell
ml python/3.12.1                                    # system python is too old (2.7/3.6)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

ml devel ollama/0.30.5
export OLLAMA_MODELS=$SCRATCH/ollama_models
ollama serve &                                      # background server
ollama pull llama3.1:8b

export DSPY_MODEL=ollama_chat/llama3.1:8b
python main.py generate-graphs \
  --input_dir webapp/static/pmc_htmls --output_dir $SCRATCH/graphs
```

Each graph JSON has the form:

```json
{
  "nodes": [{ "id": "N1", "label": "Step 1", "customData": { "content": "...", "clinical_data": {} } }],
  "edges": [{ "from": "N1", "to": "N2", "data": {
    "edge_id": "A_to_B",
    "content": "...",
    "change": [{ "variable": "heart_rate", "direction": "increase", "from_value": 70, "to_value": 110, "delta": 40, "unit": "bpm" }],
    "branch_flag": true,
    "branch_role": "open",
    "branch_id": "b1",
    "branch_depth": 1
  }}]
}
```

---

## Synthetic case generation

Generates synthetic narratives from graph paths using LLMs.

**Prerequisite:** `webapp/static/graphs/graph_metadata.csv` exists.

```bash
python main.py generate-synthetic \
  --csv webapp/static/graphs/graph_metadata.csv \
  --output_dir synthetic_outputs \
  --model gemini/gemini-2.0-flash
```

**Output**
- Text outputs: `synthetic_outputs/*.txt`
- Metadata index: `synthetic_outputs/index.jsonl`

---

## Web server

Serve the interactive viewer (FastAPI + Uvicorn + vis-network):

```bash
python main.py run-server   # http://127.0.0.1:8000
```

---

## Project structure

```
MonopathDAGs/
├── .config/                     # API keys and model config (git-ignored)
├── main.py                      # CLI entry point
├── src/
│   ├── agent/dspy/              # DSPy pipeline (graph_generation.py, testing_more.py)
│   ├── data/                    # preprocessing, synthetic generation, stats
│   ├── graph/                   # DAG model: nodes, edges, branches, builder
│   └── benchmark/modules/       # fidelity / clustering / reconstruction eval
└── webapp/
    └── static/                  # input HTMLs, output graphs, synthetic outputs, viewer
```

See [CLAUDE.md](CLAUDE.md) for a developer-oriented map of the pipeline and the
graph model.

---

## Citation

```bibtex
@misc{zhou2025monopath,
  title  = {Monopath DAGs: Structuring Patient Trajectories from Clinical Case Reports},
  author = {Zhou, Anson and Fanous, Aaron and Bikia, Vasiliki and Xu, Sonnet and Agarwal, Ank A. and Fanous, Noah and Huang, Lyndon and Luu, Jonathan and Tolbert, Preston and Alsentzer, Emily and Daneshjou, Roxana},
  note   = {Manuscript under review},
  year   = {2025},
  url    = {https://github.com/DaneshjouLab/DynamicData}
}
```

---

## License

Licensed under the Apache License 2.0. See the `LICENSE` file for details.
