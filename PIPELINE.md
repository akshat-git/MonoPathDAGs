# Monopath DAGs — End-to-End Pipeline (files & formats)

Every stage, what runs it, what it reads, and what it writes. Backend is a local
GPU **Ollama** (Llama/Qwen) served in an Apptainer container, or a hosted model
(`gemini/*`) — selected by `DSPY_MODEL` / `MONO_MODEL`.

```
PMC HTML ─▶ preprocess ─▶ patient timeline ─▶ nodes(+clinical_data)
        ─▶ edges(+change) ─▶ branch matching ─▶ graph_<n>.json
        ─▶ (synthetic) ─▶ control + sample narratives (.txt + index.jsonl)
```

---

## Stage 0 — Acquire data (one-time)
- **Runs:** `src/data/web/scrape.py` (NCBI Entrez / Biopython)
- **Reads:** PubMed — lung-cancer case reports, free full text, 2019+ (env `ncbi_api_key`, `email`)
- **Writes:**
  - `webapp/static/pmc_htmls/*.html` — full-text PMC articles (the pipeline input)
  - `lung_cancer_case_reports.csv` — catalog: `PMID, PMC_ID, Title, Journal, PublicationDate, Authors, Abstract, MeSH_Terms, URL, Citation, HTML_FILE`
- **Note:** this corpus is **gitignored** (kept local), not committed.

## Stage 1 — Generate DAGs  (`main.py generate-graphs`)
Entry: `scripts/run_smoke.sbatch` (1 case) / `scripts/run_all.sbatch` (corpus) →
`generate_all_graphs()` → per HTML → `run_pipeline()` in `src/agent/dspy/graph_generation.py`.

| # | Step (function) | Reads | Writes / returns | LLM? |
|---|---|---|---|---|
| 1 | `preprocess_html_article` → `preprocess_pmc_article_text` (`testing_more.py`) | one `.html` | `raw_text: str` (body+tables, no abstract/refs) | no (BeautifulSoup) |
| 2 | `generate_paragraphs` → `split_into_sentences` | `raw_text` | `List[str]` paragraph chunks | no |
| 3 | `generate_patient_timeline` (`PatientTimeline`) | paragraphs | `case_report: str` (patient-only timeline) | **yes** |
| 4 | `run_node_generation` (`nodeConstruct` + memory merge + `ReorganizeNodes`) | `case_report` | `nodes_obj: List[dict]` `{node_id:"A", node_step_index, content}` | **yes** |
| 5 | `annotate_nodes_with_clinical_data` (`nodeClinicalDataExtract`) | each node's `content` | adds `node["clinical_data"]` `{medications, labs, vitals, imaging, diagnoses, …}` | **yes** |
| 6 | `run_edge_generation` (`edgeConstruct`) | `nodes_obj` | `edges_obj: List[dict]` (see edge shape below) | **yes** |
| 7 | `match_branches` (domain gate + `ClassifyBranchingEdges` + Python stack) | `edges_obj` | annotates `branch_flag / branch_role / branch_id / branch_depth` in place | **yes** (inverse only) |
| 8 | `process_and_save_graph` → `format_nodes`/`format_edges` + `BranchDAG` | nodes+edges | `graph_<n>.json` + row in `graph_metadata.csv` | no |

**Edge dict (from step 6, the objective value-change):**
```json
{"edge_id":"A_to_B","content":"…",
 "change":[{"variable":"…","variable_cui":null,
   "domain":"vital_sign|lab|symptom|functional_status|medication|procedure|diagnosis|imaging|genetic|other",
   "direction":"increase|decrease|new|resolved|unchanged",
   "from_value":…,"to_value":…,"delta":…,"unit":"…"}],
 "transition_event":{…}}
```

**Branch rule (step 7):** a branch may open **only** on a *reversible patient-state* domain
(`vital_sign|lab|symptom|functional_status`) that moved (increase/decrease); it closes only on the
**same variable, opposite direction** (LLM confirms). Treatment/diagnosis/genetic/imaging are
spine-only. Unmatched openers are demoted → a branch is always a completed inverse pair.
LLM decides semantics; the stack/nesting/collapse is deterministic Python (`src/graph/branch.py`).

**Graph JSON (step 8 output):**
```json
{
  "nodes":  [{"id":"N1","label":"Step 1","customData":{ /* node dict incl. content, clinical_data */ }}],
  "edges":  [{"from":"N1","to":"N2","data":{ /* edge dict incl. change + branch_* */ }}],
  "branches":     [{"branch_id":"b1","open_edge_id":"A_to_B","close_edge_id":"C_to_D","depth":0,"variable":"…","matched":true}],
  "branch_stack": [ /* nested containment tree of branches */ ]
}
```
- **Writes:** `logs/runs/<kind>_<jid>/output/graph_*.json` and `…/output/graph_metadata.csv`
  (`graph_id, json_path, source_file`).

## Stage 1b — DAG model / inspection (`src/graph/`, pure Python, no LLM)
- `BranchDAG.from_pipeline(nodes, edges)` → `collapse_all()` (persistent spine), `branch_stack()`,
  `to_causal_export()` (flat nodes + value-change edges + branch index + spine — the causal-model input).
- `scripts/summarize_graphs.py <output_dir>` → per-graph `nodes/edges/branches/matched/spine` table.

## Stage 2 — Synthetic narratives  (`main.py generate-synthetic`)
Entry: `scripts/run_synthetic.sbatch` (needs `GRAPH_CSV=<a graph_metadata.csv>`) →
`run_batch()` in `src/data/data_sets/generate_synthetic_case.py`. Per CSV row:

| # | Step | Reads | Writes | LLM? |
|---|---|---|---|---|
| 1 | `load_graph_from_file` + `lift_custom_data` | `graph_<n>.json` | networkx `DiGraph` (node `content`/`clinical_data` lifted to attrs) | no |
| 2 | **control** narrative — `prompts.ini [generate_control_nodes]` then `[path2text]` | `source_file` HTML | control `.txt` | **yes** |
| 3 | **sample** narrative — `LLMReconstructor.reconstruct` over the DAG path (`generate_paths` root→leaf) | the `DiGraph` | sample `.txt` | **yes** |
| 4 | `save_text_and_metadata` | text + meta | `<graph_id>__{control\|sample}__<uid>.txt` + append `index.jsonl` | no |

- **`index.jsonl` row:** `{graph_id, html_file, is_control, model, node_path_used, uid, text_file}`
- **Writes:** `logs/runs/synth_<jid>/synthetic/*.txt` + `index.jsonl`

---

## Config & infrastructure
- **`.config/.env`** — `DSPY_MODEL`, `GEMINI_APIKEY`, `GPTKEY` (git-ignored).
- **`.config/prompts.ini`** — `[generate_control_nodes]`, `[path2text]`, `[prompt_control]`.
- **Local model backend** — `.ollama_dist/ollama.sif` (Apptainer image, GPU via `--nv`); weights in
  `.ollama/` staged to `$L_SCRATCH`; select model with `MONO_MODEL` (e.g. `qwen2.5:14b`).
- **Scripts** — `scripts/{pipeline_common.sh, run_smoke.sbatch, run_all.sbatch, run_synthetic.sbatch, summarize_graphs.py}`.
- **Logs (git-ignored)** — `logs/slurm/` (scheduler), `logs/ollama/` (server + GPU check),
  `logs/runs/<kind>_<jid>/{input,output,synthetic}/`.

## One full local run (GPU, no API key)
```bash
sbatch --export=ALL,MONO_MODEL=qwen2.5:14b scripts/run_all.sbatch      # HTML -> graphs
# then, using the graph_metadata.csv it produced:
sbatch --export=ALL,MONO_MODEL=qwen2.5:14b,\
GRAPH_CSV=logs/runs/all_<jid>/output/graph_metadata.csv scripts/run_synthetic.sbatch
```
