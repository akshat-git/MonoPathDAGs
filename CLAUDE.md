# CLAUDE.md

Guidance for Claude Code (and humans) working in this repository.

## What this project is

**Monopath DAGs** converts clinical case reports (PubMed Central HTML) into
structured **Directed Acyclic Graphs** of a patient's trajectory:

- **Nodes** = snapshots of the patient's clinical state (narrative `content` +
  structured `clinical_data`: meds, labs, vitals, imaging, diagnoses, …).
- **Edges** = transitions between adjacent states.

The graph is a *monopath* (a linear A→B→C→… spine). Extraction is driven by
[DSPy](https://github.com/stanfordnlp/dspy) signatures over an LLM
(Gemini by default, Ollama supported).

### Current direction (in progress)

We are reworking DAG creation to make the output **causal-model-ready**:

- Edges gain a structured **objective value-change** (`change`:
  variable, direction, from/to, delta, unit) alongside the narrative.
- **Branches** are *ephemeral excursions* — an opening edge (e.g. HR 70→110)
  matched to a downstream **inverse closing edge** (HR 110→75). Matched pairs
  nest via a stack (`branch_depth`) and form a **collapsible** bracket
  structure; collapsing a matched span isolates transient perturbations from
  persistent state change.
- Branch vs non-branch edges are explicitly encoded so a downstream causal
  model can consume the DAG. Branch matching is LLM-driven
  (`ClassifyBranchingEdges`); the DAG/collapse logic is pure Python.

When editing branch/edge logic, preserve this separation: **LLM decides
semantics (is B the inverse of A?); Python owns the DAG structure, stack
nesting, and collapse** so it stays deterministic and testable.

## Layout

```
main.py                              # CLI entry: generate-graphs | generate-synthetic | run-server
src/
  agent/dspy/graph_generation.py     # ★ core pipeline: nodes, edges, branch eval, JSON output
  agent/dspy/testing_more.py         # DSPy signatures: timeline, node build, ClassifyBranchingEdges
  data/data_processors/pdf_to_text.py
  data/data_sets/generate_synthetic_case.py
  graph/                             # DAG model layer (protocols + partial impls; being wired in)
    build_graph.py                   # GraphBuilder
    graph_protocol.py                # Graph / DAG protocols, GraphImplemented
    node/node_protocol.py            # NodeData, ImmutableNode/MutableNode, NodeFactory
    edge/edge_protocol.py            # EdgeData, DynamicDataEdge, OrderedEdge
  benchmark/modules/                 # fidelity / clustering / reconstruction evaluation
webapp/                              # FastAPI + vis-network viewer & human-eval tooling
```

## Data ingestion

Inputs are open-access **PubMed Central** case-report HTMLs (not one journal).
`src/data/web/scrape.py` queries PubMed via NCBI Entrez (Biopython) for lung-cancer
case reports (lung cancer / NSCLC / SCLC / mesothelioma …, Publication Type = Case
Reports, free full text, 2019+), downloads full-text HTML, and catalogs metadata in
`lung_cancer_case_reports.csv` (`PMID, PMC_ID, Title, Journal, …, HTML_FILE`). The
working set lives in `webapp/static/pmc_htmls/*.html`; that's what `--input_dir`
points at. Re-scraping needs env vars `ncbi_api_key` and `email`.

## Key pipeline flow (`src/agent/dspy/graph_generation.py`)

`generate_all_graphs` (configures the LM) → per HTML → `run_pipeline(html_path)`:

1. `preprocess_html_article` (BeautifulSoup: drop abstract/refs/nav, keep body+tables)
   → `generate_paragraphs` → `generate_patient_timeline` (LLM → patient-only timeline)
2. `run_node_generation` (sentence-chunked, memory-merged states)
   → `annotate_nodes_with_clinical_data` (UMLS-grounded meds/labs/vitals/…)
3. `run_edge_generation` (one edge per adjacent node pair, each with a structured
   `change`) → `match_branches` (LLM inverse-matching + Python stack nesting)
4. `process_and_save_graph` → `format_nodes`/`format_edges` → JSON + `graph_metadata.csv`

```
PMC HTML → preprocess → patient timeline → nodes(+clinical_data)
         → edges(+change) → branch matching → graph_<n>.json
```

Output JSON shape: `{"nodes": [{id, label, customData}], "edges": [{from, to, data}]}`.
Each edge's `data` now carries `change` plus `branch_flag`/`branch_role`/
`branch_id`/`branch_depth` from the collapsible branch structure.

### Known rough edges (fix as you touch them)
- `GraphImplemented.edges()` returns the dict, not `.values()`. (Phase 3)
- `DAG_Implemented` and `build_from_adjacency_matrix` are stubs. (Phase 3)
- `graph_generation.py` still has a bare top-level `from testing_more import ...`
  (~line 31) next to the correct package-relative import; prefer the latter.

Recently fixed (Phase 0/2): `evaluate_branching` (replaced by the stack-based
`match_branches` + `src/graph/branch.py`); the hard-coded docstrings-ini load
(now optional via the `PROMPT_INI` env var, else built-in prompts); the
`GEMINIAPIKEY` vs `GEMINI_APIKEY` mismatch; the import-time `teleprompter.compile`
(now lazy — see `get_optimized_branch_classifier`); and `main.py generate-graphs`
referencing an undefined `--prompt_ini`.

## Running

Config lives in `.config/.env` (`DSPY_MODEL`, `GEMINI_APIKEY`, `GPTKEY`). Never
commit it. Backend: Gemini needs `GEMINI_APIKEY`; an `ollama/*` `DSPY_MODEL` runs
locally with no key (needs an Ollama server at `localhost:11434`).

Requires **Python ≥3.9** (`list[dict]`, `from __future__ import annotations`).
On Sherlock the system `python`/`python3` are 2.7/3.6 — load a module instead
(e.g. `ml python/3.12.1`) inside an `sh_dev`/`salloc` session. The pure
`src/graph` logic + `tests/` need only stdlib; the DSPy pipeline needs the full
`requirements.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python main.py generate-graphs  --input_dir ./pmc_htmls --output_dir ./webapp/static/graphs
python main.py generate-synthetic --csv webapp/static/graphs/graph_metadata.csv \
       --output_dir synthetic_outputs --model gemini/gemini-2.0-flash
python main.py run-server        # FastAPI at http://127.0.0.1:8000
```

## Conventions
- LLM extraction is DSPy `Signature` + `Module`; docstrings on signatures are
  the prompts. Keep node/edge output **JSON-serializable** (list-of-dicts).
- Reference UMLS CUIs for clinical concepts where possible.
- Node ids are letters (`A`, `B`, …) internally; storage ids are `N1`, `N2`.
- New structural/graph logic goes in `src/graph/` with unit tests, not in the
  DSPy pipeline.

## Environment note (Sherlock HPC)
This repo lives on the Stanford Sherlock cluster. **Do not run graph generation,
`pip install`, or any heavy Python on the login node** — use `sh_dev`/`salloc`
or submit via Slurm, write outputs to `$SCRATCH`, and keep API keys and caches
(`HF_HOME`, `PIP_CACHE_DIR`) off `$HOME`.
