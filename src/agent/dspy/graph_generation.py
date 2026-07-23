
from __future__ import annotations
import csv
import dspy
import json
import time
from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="urllib3")

from pprint import pprint
from tqdm import tqdm
from dspy import Example
from src.data.data_processors.pdf_to_text import extract_text_from_pdf
from src.agent.dspy.testing_more import (extract_paragraphs, recursively_decompose_to_atomic_sentences, ClassifyBranchingEdges)
from src.graph.branch import (
    match_branch_structure, BranchSpan, spans_from_edges,
    is_reversible_state_edge, is_direct_state_inverse,
)
from src.graph.branch_dag import BranchDAG

import pandas as pd
import configparser
import json
import requests
import os
import json
import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from src.agent.dspy.testing_more import (
    preprocess_pmc_article_text,
    split_into_sentences,
    PatientTimeline
)
# =====================================
# DOCSTRING CONTENT
# =====================================

docstring_dict = {
    "dag_primer":
        """
    You are an assistant that converts clinical case narratives into dynamic Directed Acyclic Graphs (DAGs).

    Each DAG consists of:
    - Nodes = snapshots of the patient's state.
    - Edges = transitions between those states.

    Terminology guidance:
    - Use UMLS-standard concepts when possible for consistency and interoperability.
    - If a concept isn't covered by UMLS, use clear, logical labeling.

    Text extraction guidance:
    - When looking at the case report input, ignore the references, background, conclusions etc. sections
    - We only want to extract on content relating to the specific patient discussed in the case report


    """,

    "node_instructions":
    """
    You are given a clinical case report. Your task is to extract a sequence of nodes
    representing the patient's evolving clinical STATE. A node is a snapshot of the patient
    AT A POINT IN TIME (the "state between transitions"); the change/cause that moves the
    patient from one node to the next lives on the EDGE, not the node.

    Guidelines:
    - Create one node per clinically meaningful state.
    - A node is the FULL patient state at that time, not a one-line note. Carry forward
      everything still true from node_memory (active diagnoses, ongoing medications,
      persistent findings) and add what is new — the node should read as a complete
      snapshot of the patient, not just the latest event.
    - Combine co-occurring labs/imaging into the same node.
    - Use separate nodes for clearly sequential or distinct events.
    - Do not return anything outside the list format. Should be in JSON compatible style.
    - Keep imaging content packaged in one node if no clear temporal change is indicated
    - Keep pathology / histology content packaged in one node if no clear temporal change is indicated
    - node_memory is a running memory that updates as we add new nodes; use it to preserve context, merge overlapping details, and avoid redundant or stale states
    - TIMESTAMP: whenever the report gives a date or relative time for this state
      (e.g. "November 2022", "on day 5", "two weeks later"), record it in `timestamp`.
      Prefer ISO8601 when a real date is given; otherwise keep the report's own phrasing.
    - STATE, NOT CHANGE: `content` describes what IS TRUE of the patient in this state —
      current findings and values (e.g. "hemoglobin 8.4 g/dL", "3.9 cm frontal lesion
      present"). Do NOT put transitions/deltas here ("rose from 70 to 110", "started drug X",
      "tumor grew") — those are CHANGES and belong on the EDGE, not the node. Use the
      report's facts and values; do not invent.

    Output format:
    Return a list of node dictionaries, in this order from top to bottom, each with:
    - node_id (In ascending alphabetical order, e.g., "A", "B", "C")
    - node_step_index (integer for order)
    - content (the patient's STATE at this time — current facts/values from the report; NOT changes/deltas, which go on edges)
    - timestamp (date/relative time from the report if stated; ISO8601 preferred, else the report's phrasing)
    - clinical_data (flat list of separate state facts; see the clinical_data instructions)

    Example:
    [
      {
        "node_id": "A",
        "node_step_index": 0,
        "content": "The patient presented with bilateral painless testicular masses.",
        "timestamp": "2022-11"
      }
    ]


    """,

    "edge_instructions":
        """
    Each edge is the DELTA between two adjacent patient-state nodes: it carries both the
    CAUSE of the transition (what happened — a treatment/intervention, a procedure, a
    spontaneous event) and the resulting CHANGE(s). Nodes are states; edges are what moved
    the patient between them. There should be one edge between every pair of adjacent nodes.
    The cause is captured in `transition_event` (trigger_type / trigger_entities); the
    change(s) — including a treatment being started — are captured in `change` (see below).

    Guidelines for edges:
    - Create edges only when there is a clear clinical progression or change between nodes.
    - Maintain narrative or logical order — edges should flow from earlier to later events.
    - Combine co-occurring findings into the same node, not across multiple edges.

    Output format:
    Return a list of edge dictionaries. Each edge dictionary MUST have EXACTLY these keys
    and NO others: edge_id, branch_flag, content, change, transition_event.

    CRITICAL: an edge is a TRANSITION, not a state. Do NOT put a `clinical_data` field on an
    edge, and do NOT list medications / diagnoses / labs / imaging as node-style records.
    Those belong to NODES. On an edge you describe only WHAT CHANGED, as `change`.

    Each edge, in order:
    - edge_id: Unique identifier (Use format "node_id"_to_"node_id", such that the first "node_id" is the upstream node and the second "node_id" is the downstream node bounding the edge)
    - branch_flag: Boolean if this starts a side branch, default = True
    - content: Exhuastive clinical content, include all relevant details for the given node
    - change: REQUIRED list describing the objective change(s) this edge encodes (see below).
      Never omit this and never replace it with clinical_data.

    Objective value-change (the core of every edge):
    Every edge encodes one or more *objective* changes in a named clinical variable — a
    measurable move (heart rate 70 -> 110), a lab crossing (Hb 12.1 -> 8.4), or a
    categorical change (medication started, diagnosis resolved). Capture these in a
    `change` list; one dict per variable that changed:

    change = [
        {
            "variable": "human-readable name of what changed (e.g., 'heart rate', 'hemoglobin', 'carboplatin')",
            "variable_cui": "UMLS CUI if known, else null",
            "domain": "OPTIONAL rough label for the kind of change (e.g. vital_sign, lab, symptom, medication, diagnosis, imaging) — NOT a fixed enum; leave blank if unsure. Final categorization is a later step.",
            "direction": "increase | decrease | new | resolved | unchanged",  # 'new' = newly started/appeared, 'resolved' = discontinued/cleared
            "from_value": "prior numeric or string value, or null if not stated / newly appearing",
            "to_value": "new numeric or string value, or null if resolved/discontinued",
            "delta": "signed numeric change if both values are numeric, else null",
            "unit": "unit string (e.g., 'bpm', 'g/dL', 'mg'), or null"
        }
    ]

    Guidance for change:
    - Prefer numeric from/to/delta when the narrative gives numbers; otherwise use
      descriptive strings and leave delta null.
    - A single edge may carry several change entries if multiple variables move together.
    - Do NOT categorize during extraction beyond the optional rough `domain` hint. Keep each
      change as a separate raw item; deciding "this is a treatment vs a lab vs a vital" is a
      LATER step, not something to force here.
    - `direction` should reflect the clinical/quantitative sense of the move so that an
      opposite move on the SAME variable (e.g., heart rate up, then heart rate back down) can
      later be recognized as an inverse that closes a transient branch. A treatment is not the
      inverse of a patient-state change.

    Optional structured field for edge-level transitions:
    transition_event = {
        "trigger_type": "procedure | lab_change | medication_change | symptom_onset | interpretation | spontaneous",
        "trigger_entities": ["UMLS_CUI_1", "UMLS_CUI_2"],  # e.g., C0025598 = Metformin, C0011581 = Chest Pain
        "change_type": "addition | discontinuation | escalation | deescalation | reinterpretation | resolution | progression | other",  # Nature of the change
        "target_domain": "medication | symptom | diagnosis | lab | imaging | procedure | functional_status | vital_sign",  # What category was affected
        "timestamp": "ISO 8601 datetime (e.g., "2025-03-01T10:00:00Z"), only include if explicitly given and can be converted to datetime"
    }

    """,

    "node_clinical_data_instructions":
        """
    A node is a snapshot of the patient's FULL clinical STATE at this point in time.
    Put the state in `clinical_data` as a FLAT LIST of SEPARATE facts — do NOT group or
    categorize them (no medications / labs / vitals buckets). Calling a fact a treatment,
    lab value, vital, diagnosis, etc. is a LATER step, not something to do here.

    clinical_data = {
        "facts": [
            {
                "item": "what it is, in the report's words (e.g. 'hemoglobin', 'frontal lesion', 'pembrolizumab', 'PD-L1 TPS')",
                "value": "value if any (numeric or string), else null",
                "unit": "unit if any, else null",
                "timestamp": "date/relative time if the report gives one, else null",
                "cui": "UMLS CUI if you happen to know it, else null"
            }
        ]
    }

    Rules:
    - Keep every fact separate; one entry per fact. Do NOT merge or categorize.
    - Capture the FULL current state: carry forward still-active facts from node_memory and
      add new ones. NEVER drop a fact just because it has no CUI.
    - STATE only — do NOT record changes/transitions/deltas here; those belong on edges.
    """,

    "branch_instructions":
        """
    Branches arise when physiologic changes or complications aren't part of the main pathway but impact patient states. Specifically, we are thinking of ephemeral changes.

    Mark side branches clearly:
    - Edge initiating a new branch: branch_flag = True
    """
}
# remove and place in an n

def load_docstrings_from_ini(file_path):
    """Override the built-in prompts from an .ini file (section [Docstrings])."""
    config = configparser.ConfigParser()
    config.read(file_path)

    docstring_dict.update({
        "dag_primer": config.get("Docstrings", "dag_primer"),
        "node_instructions": config.get("Docstrings", "node_instructions"),
        "edge_instructions": config.get("Docstrings", "edge_instructions"),
        "node_clinical_data_instructions": config.get("Docstrings", "node_clinical_data_instructions"),
        "branch_instructions": config.get("Docstrings", "branch_instructions")
    })


# Optionally override the built-in prompts from an .ini (via the PROMPT_INI env
# var, e.g. set by `main.py --prompt_ini`). If unset/missing/malformed we keep the
# built-in docstring_dict above rather than crashing at import.
_PROMPT_INI = os.environ.get("PROMPT_INI")
if _PROMPT_INI and os.path.exists(_PROMPT_INI):
    try:
        load_docstrings_from_ini(_PROMPT_INI)
        print(f"Loaded prompt overrides from {_PROMPT_INI}")
    except Exception as exc:
        print(f"Warning: could not load prompts from {_PROMPT_INI} ({exc}); using built-in prompts.")

# =====================================
# SELECTED LLM
# =====================================









# =====================================
# DSPY SIGNATURES
# =====================================

class nodeConstruct(dspy.Signature):
    text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_memory: list[dict] = dspy.InputField(optional=True, desc="List of nodes generated so far (memory). Use as context when generating new nodes so there is no duplication.") # Sliding window, grows as we make new nodes
    node_output = dspy.OutputField(type=list[dict], desc="A list of node dictionaries with node_id, node_step_index, content, optional timestamp and clinical_data")


class edgeConstruct(dspy.Signature):
    #text_input: str = dspy.InputField(desc="body of text extracted from a case report")
    node_input: list[dict] = dspy.InputField(desc="Ordered list of node dictionaries as generated by nodeConstruct")
    edge_output: list[dict] = dspy.OutputField(desc="A list of edge dictionaries, one per adjacent node pair. Each edge has EXACTLY: edge_id, branch_flag, content, change, transition_event. change is a REQUIRED non-empty list of objective changes {variable, variable_cui, domain, direction, from_value, to_value, delta, unit}. Do NOT put a clinical_data field or node-style medication/diagnosis/lab records on an edge — edges hold change only.")

class nodeClinicalDataExtract(dspy.Signature):
    # Optional: You can comment this out to rely only on atomic_sentences
    content: str = dspy.InputField(desc="Narrative clinical content from a node")
    atomic_sentences: list[str] = dspy.InputField(desc="List of atomic-level clinical statements derived from the node content")
    clinical_data: dict = dspy.OutputField(desc="Structured clinical data dictionary with fields like medications, labs, imaging, etc.")

# =====================================
# FORM AND APPLY DOCSTRINGS
# =====================================

nodeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['node_instructions']
edgeConstruct.__doc__ = docstring_dict["dag_primer"] + docstring_dict['edge_instructions']
nodeClinicalDataExtract.__doc__ = docstring_dict["dag_primer"] + docstring_dict["node_clinical_data_instructions"]

# =====================================
# MODULES
# =====================================


class NodeEdgeGenerate(dspy.Module):
    def __init__(self):
        super().__init__()
        self.node_module = dspy.Predict(nodeConstruct)
        # Attach few-shot demonstrations via `.demos` (the correct DSPy mechanism).
        # NOTE: do NOT pass `examples=` to dspy.Predict — in this DSPy version that
        # kwarg leaks the Example objects into the LM request payload, which litellm
        # then fails to JSON-serialize ("Object of type Example is not JSON serializable").
        self.edge_module = dspy.Predict(edgeConstruct)
        self.edge_module.demos = edge_fewshot_examples


    def generate_node(self, text_input, node_memory=None):
        # Call the LLM to get raw nodes
        result = self.node_module(
            text_input=text_input,
            node_memory=node_memory or []
        )
        nodes = result.get("node_output", [])

        # If the LLM returns a JSON string, parse it
        if isinstance(nodes, str):
            try:
                parsed = json.loads(nodes)
                if isinstance(parsed, list):
                    nodes = parsed
                else:
                    raise ValueError("Parsed node output is not a list.")
            except Exception as e:
                print(f"Warning: Node output not valid JSON. Error: {e}")
                nodes = []

        # If have prior memory, merge new with old
        if node_memory:
            nodes = merge_memory_nodes(node_memory, nodes)

        # Return the unified list under the same key
        return {"node_output": nodes}


    def generate_edge(self, node_input):
        # For edge generation, the node_input is the dict of nodes that are output from generate_node
        result = self.edge_module(node_input=node_input)
        edges = result.get("edge_output", [])

        # Handle stringified JSON if returned as text
        if isinstance(edges, str):
            try:
                parsed = json.loads(edges)
                if isinstance(parsed, list):
                    edges = parsed
                else:
                    raise ValueError("Parsed edge output is not a list.")
            except Exception as e:
                print(f"Warning: Edge output not valid JSON. Error: {e}")
                edges = []

        return {"edge_output": edges}



class ClinicalDataExtractor(dspy.Module):
    def __init__(self, max_retries: int = 2):
        super().__init__()
        self.extractor = dspy.Predict(nodeClinicalDataExtract)
        self.max_retries = max_retries

    def forward(self, content: str = "", atomic_sentences: list[str] = []):
        attempt = 0
        clinical_data = None

        while attempt <= self.max_retries:
            result = self.extractor(content=content, atomic_sentences=atomic_sentences)
            raw_output = result.get("clinical_data", {})

            print(f"\n=== Attempt {attempt + 1}: Raw clinical_data output ===")
            print(raw_output)

            # --- FIX: unwrap list if needed ---
            if isinstance(raw_output, list):
                # Take first dict if exists
                clinical_data = next((item for item in raw_output if isinstance(item, dict)), None)
            elif isinstance(raw_output, dict):
                clinical_data = raw_output
            else:
                clinical_data = None

            # If we got valid dict → DONE
            if isinstance(clinical_data, dict):
                return clinical_data

            print(f"⚠️ Attempt {attempt + 1}: clinical_data invalid → retrying...")
            attempt += 1

        # After retries failed → fallback
        print(f"❌ Failed to obtain valid clinical_data after {self.max_retries + 1} attempts. Returning empty dict.")
        return {}



class InverseEdgeMatcher(dspy.Module):
    """LLM wrapper around ClassifyBranchingEdges.

    Sole responsibility: decide the *semantics* — is edge B the inverse / reversal
    of edge A (e.g. heart rate 70->110 later undone by 110->75)? The DAG structure,
    stack nesting, and collapse logic live in pure Python (`match_branches`) so they
    stay deterministic and testable; this module is the only non-deterministic piece.
    """

    def __init__(self):
        super().__init__()
        self.program = dspy.Predict(ClassifyBranchingEdges)

    def forward(self, edge_a: Dict[str, Any], edge_b: Dict[str, Any]):
        return self.program(
            EdgeA=describe_edge_for_matching(edge_a),
            EdgeB=describe_edge_for_matching(edge_b),
        )




# ---------------------------------------
# FEW-SHOT EXAMPLES - EDGE CONSTRUCTION
# ---------------------------------------

# Structure not coming out right with edges so doing these to reinforce output

# NOTE: these few-shot examples are ANONYMIZED — they use generic category words
# ("vital sign", "therapy") and illustrative numbers, so they carry NO real disease,
# drug name, or UMLS CUI to bias the model. They ARE concrete/valid (not <placeholder>
# strings) so the model reliably learns the exact change schema and the
# open/inverse-close pattern. Examples 1+2 are an inverse pair on the SAME variable.

edge_example_1 = dspy.Example(
    node_input=[{"node_id": "A"}, {"node_id": "B"}],
    edge_output=[{
        "edge_id": "A_to_B",
        "branch_flag": True,
        "content": "A reversible patient-state measurement rose above its baseline.",
        "change": [{
            "variable": "vital sign",
            "variable_cui": None,
            "domain": "vital_sign",   # a reversible patient-state domain
            "direction": "increase",
            "from_value": 100,
            "to_value": 130,
            "delta": 30,
            "unit": "units"
        }],
        "transition_event": {
            "trigger_type": "spontaneous",
            "trigger_entities": [],
            "change_type": "escalation",
            "target_domain": "vital_sign"
        }
    }]
).with_inputs("node_input")

edge_example_2 = dspy.Example(
    node_input=[{"node_id": "B"}, {"node_id": "C"}],
    edge_output=[{
        "edge_id": "B_to_C",
        "branch_flag": True,
        "content": "The same measurement returned toward its baseline (inverse of the opener).",
        "change": [{
            "variable": "vital sign",   # identical variable to the opener
            "variable_cui": None,
            "domain": "vital_sign",
            "direction": "decrease",
            "from_value": 130,
            "to_value": 105,
            "delta": -25,
            "unit": "units"
        }],
        "transition_event": {
            "trigger_type": "interpretation",
            "trigger_entities": [],
            "change_type": "resolution",
            "target_domain": "vital_sign"
        }
    }]
).with_inputs("node_input")

edge_example_3 = dspy.Example(
    node_input=[{"node_id": "C"}, {"node_id": "D"}],
    edge_output=[{
        "edge_id": "C_to_D",
        "branch_flag": False,
        "content": "A persistent, non-reversible change (a therapy was started) — never a branch.",
        "change": [{
            "variable": "therapy",
            "variable_cui": None,
            "domain": "medication",   # persistent / interventional domain, not branch-eligible
            "direction": "new",
            "from_value": None,
            "to_value": "started",
            "delta": None,
            "unit": None
        }],
        "transition_event": {
            "trigger_type": "medication_change",
            "trigger_entities": [],
            "change_type": "addition",
            "target_domain": "medication"
        }
    }]
).with_inputs("node_input")

edge_fewshot_examples = [edge_example_1, edge_example_2, edge_example_3]



# =====================================
# EXTRACTION PROCESS AND ORG FUNCTIONS
# =====================================


def decompose_content_to_atomic_statements(content_block: str) -> list[str]:
    """
    Given a content string from a node or edge, decompose it into atomic clinical statements.

    Intended for use *after* nodes or edges are generated from the full case report.

    Parameters:
    - content_block (str): A single content string from a node or edge.

    Returns:
    - List of atomic clinical sentences (List[str])
    """
    atomic_statements = []

    for sentence in content_block.split(". "):
        if sentence.strip():
            decomposed = recursively_decompose_to_atomic_sentences(sentence.strip())
            atomic_statements.extend(decomposed)

    return atomic_statements




class ChunkedNodeGenerator(dspy.Signature):
    case_report: str = dspy.InputField(desc="Full clinical case report text")
    max_words_per_chunk: int = dspy.InputField(desc="Maximum number of words per chunk", default=250)
    max_chunks: int = dspy.InputField(desc="Maximum number of chunks to process (optional)", default=None)
    node_output: list[dict] = dspy.OutputField(desc="List of generated node dictionaries")


class ChunkingNodeModule(dspy.Module):
    def __init__(self, generator: NodeEdgeGenerate):
        super().__init__()
        self.generator = generator

    def forward(self, case_report: str, max_chunks: int = None) -> Dict[str, List[Dict]]:
        """
        Processes each sentence individually through the generator.

        Args:
            case_report (str): Input case report text.
            max_chunks (int, optional): Maximum number of sentences to process.

        Returns:
            Dict[str, List[Dict]]: Generated node outputs.
        """
        sentences = split_into_sentences(case_report,4)  # List[str]

        if max_chunks is not None:
            sentences = sentences[:max_chunks]

        memory_nodes: List[Dict] = []
        failed_chunks = 0
        start_time = time.time()

        for sentence in tqdm(sentences, desc="Generating nodes", unit="sentence"):
            try:
                text_input = sentence.strip()
                output = self.generator.generate_node(
                    text_input=text_input,
                    node_memory=memory_nodes
                )
                memory_nodes = output["node_output"]
            except Exception as exc:
                print(f"Warning: node generation failed on sentence. Error: {exc}")
                failed_chunks += 1

        elapsed_time = time.time() - start_time
        print(f"\nNode generation completed in {elapsed_time:.2f} seconds.")
        print(f"Failed sentences: {failed_chunks}/{len(sentences)}")

        return {"node_output": memory_nodes}


# Creates a dynamic memory to give nodes context as it is constructing

def merge_memory_nodes(prev_nodes: List[Dict], new_nodes: List[Dict]) -> List[Dict]:
    """
    Merge `new_nodes` into `prev_nodes` by:
      1. Appending truly new nodes.
      2. For matching node_ids, concatenating content and merging clinical_data.
      3. Re-assigning node_ids (“A”, “B”, …) and step_indices.
    """
    # shallow-copy previous
    merged = [n.copy() for n in prev_nodes]
    #lookup dictionary for merged. 
    lookup = {n["node_id"]: n for n in merged}
    # for cand in new nodes, 
    for cand in new_nodes:
        # candidate id
        cid = cand["node_id"]
        # if this in lookup
        if cid in lookup:
            # set base to the node
            base = lookup[cid]
            # merge content... if the content not in base you add it in. 
            if cand["content"] not in base["content"]:
                base["content"] += " " + cand["content"]
            # merge clinical_data->> close basically you take the cat and item
            for cat, items in cand.get("clinical_data", {}).items():
                bucket = base.setdefault("clinical_data", {}).setdefault(cat, [])
                for itm in items:
                    if itm not in bucket:
                        bucket.append(itm)
        else:
            merged.append(cand.copy())

    # re-index
    for idx, node in enumerate(merged):
        node["node_step_index"] = idx
        node["node_id"] = chr(ord("A") + idx)

    return merged


# Does final pass through completed sequence of nodes and helps to organize them
# problems with thios, we make heavy assumptions on the Node_ID being correct, which is close, 


class ReorganizeNodes(dspy.Signature):
    """
    Take a full list of extracted nodes, deduplicate or merge any
    overlapping/fragmented entries, then reindex them cleanly.
    """
    nodes_sequence: List[Dict] = dspy.InputField(
        desc="List of node dicts to be cleaned and merged"
    )
    node_output: List[Dict] = dspy.OutputField(
        desc="Reorganized list of nodes (deduped, merged, re‐indexed A, B, C…)"
    )

# lets try it this way
# 1: take in the text generated by the case, identify 



# =====================================
# RUN PIPELINE
# =====================================






def extract_case_report_text(pdf_path: str) -> str:
    """Extract text from a PDF case report."""
    return extract_text_from_pdf(pdf_path)


def preprocess_html_article(html_path: str) -> str:
    """Preprocess a PMC HTML article."""
    return preprocess_pmc_article_text(html_path)


def generate_paragraphs(raw_text: str, split_length: int = 10) -> List[str]:
    """Split preprocessed text into sentences/paragraphs."""
    return split_into_sentences(raw_text, split_length)


def generate_patient_timeline(paragraphs: List[str]) -> str:
    """Run DSPy patient timeline predictor across paragraphs."""
    predict_patient_timeline = dspy.Predict(PatientTimeline)
    prev_memory: List[str] = [""]

    for paragraph in paragraphs:
        output = predict_patient_timeline(
            paragraph=paragraph,
            previous_memory=prev_memory[-3:]
        )
        prev_memory.append(output.pt_timeline)
        print("-" * 30, output.pt_timeline)

    return prev_memory[-1]  # Final case report timeline


def run_node_generation(case_report: str) -> List[Dict[str, Any]]:
    """Run node generation pipeline."""
    generator = NodeEdgeGenerate()
    chunked_node_module = ChunkingNodeModule(generator)

    node_result = chunked_node_module(
        case_report=case_report,
        max_chunks=None
    )
    nodes_obj = node_result["node_output"]
    nodes_obj = merge_memory_nodes([], nodes_obj)

    nodes_obj = dspy.Predict(ReorganizeNodes)(nodes_sequence=nodes_obj).node_output
    return nodes_obj


def run_edge_generation(nodes_obj: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Run edge generation pipeline."""
    generator = NodeEdgeGenerate()
    edge_result = generator.generate_edge(node_input=nodes_obj)
    return edge_result["edge_output"]


def annotate_nodes_with_clinical_data(nodes_obj: List[Dict[str, Any]]) -> None:
    """Append clinical data extraction to each node."""
    clinical_data_extractor = ClinicalDataExtractor()
    for node in nodes_obj:
        content_str = node.get("content", "")
        clinical_data = clinical_data_extractor(
            content=content_str,
            atomic_sentences=None
        )
        node["clinical_data"] = clinical_data


def describe_edge_for_matching(edge: Dict[str, Any]) -> str:
    """Compact human-readable rendering of an edge for the inverse-matching LLM.

    Includes the narrative plus any objective value-changes so the classifier can
    judge whether one edge reverses another on the same variable.
    """
    parts = [edge.get("content", "") or ""]
    for vc in edge.get("change", []) or []:
        if not isinstance(vc, dict):
            continue
        parts.append(
            "[{var}: {frm} -> {to} ({dir_}{unit})]".format(
                var=vc.get("variable", "?"),
                frm=vc.get("from_value"),
                to=vc.get("to_value"),
                dir_=vc.get("direction", "?"),
                unit=(" " + vc["unit"]) if vc.get("unit") else "",
            )
        )
    return " ".join(p for p in parts if p).strip()


def is_inverse_edge(open_edge: Dict[str, Any], candidate: Dict[str, Any],
                    inverse_matcher: "InverseEdgeMatcher") -> bool:
    """Does `candidate` close the excursion opened by `open_edge`?

    Two gates, strictest first:
    1. Deterministic: the candidate must move the SAME reversible patient-state
       variable in the OPPOSITE direction (a treatment or an unrelated variable
       can never close a branch). Prevents over-eager / treatment-vs-state matches.
    2. LLM confirmation that it is a genuine physiologic reversal.
    """
    if not is_direct_state_inverse(open_edge, candidate):
        return False
    try:
        verdict = inverse_matcher(open_edge, candidate)
        return bool(getattr(verdict, "branch_reversible", False))
    except Exception as exc:  # keep the pipeline resilient to LLM/parse errors
        print(f"Warning: inverse-match failed "
              f"({open_edge.get('edge_id')} vs {candidate.get('edge_id')}): {exc}")
        return False


def is_branch_opening(edge: Dict[str, Any]) -> bool:
    """A branch opens only for a *reversible patient-state* change (vital / lab /
    symptom / functional status that measurably moved). Treatment, diagnosis,
    genetic and imaging changes are persistent spine edges and never open a
    branch — this is the deterministic guard against over-eager branching.
    """
    return is_reversible_state_edge(edge)


def match_branches(edges_obj: List[Dict[str, Any]],
                   inverse_matcher: "InverseEdgeMatcher" = None) -> List[Dict[str, Any]]:
    """Annotate edges with a collapsible, stack-nested branch structure.

    The LLM decides the *semantics* — does this edge open an ephemeral excursion?
    does it invert the currently-open one? — while the deterministic stack/nesting
    logic lives in :func:`src.graph.branch.match_branch_structure`. Here we simply
    inject the two LLM-backed predicates and adapt the result to a JSON-serializable
    list of span dicts (the collapsible index). Edges are annotated in place with
    branch_flag / branch_role / branch_id / branch_depth.
    """
    if not edges_obj:
        print("No edges available to evaluate.")
        return []

    inverse_matcher = inverse_matcher or InverseEdgeMatcher()

    def _is_inverse(open_edge: Dict[str, Any], candidate: Dict[str, Any]) -> bool:
        return is_inverse_edge(open_edge, candidate, inverse_matcher)

    def _is_opening(edge: Dict[str, Any]) -> bool:
        return is_branch_opening(edge)

    spans = match_branch_structure(edges_obj, is_opening=_is_opening, is_inverse=_is_inverse)

    matched = sum(1 for s in spans if s.matched)
    unresolved = [s.open_edge_id for s in spans if not s.matched]
    if unresolved:
        print(f"Unresolved (open, never inverted) branches: {unresolved}")
    print(f"Branch matching complete: {len(spans)} opened, {matched} matched/collapsible.")
    return [s.to_dict() for s in spans]


def print_nodes(nodes_obj: List[Dict[str, Any]]) -> None:
    """Pretty print all nodes."""
    print("=" * 60)
    print("Nodes (with clinical data):")
    print("=" * 60)

    for i, node in enumerate(nodes_obj):
        print(f"\n Node {i + 1}:")
        pprint(node)
        print("-" * 60)


def print_edges(edges_obj: List[Dict[str, Any]]) -> None:
    """Pretty print all edges."""
    print("=" * 60)
    print("Edges:")
    print("=" * 60)

    for i, edge in enumerate(edges_obj):
        print(f"\n Edge {i + 1}: edge_id = {edge.get('edge_id', 'N/A')}")
        pprint(edge)
        print("-" * 60)


def run_pipeline(html_path: str) -> tuple[list[dict], list[dict]]:
    """Main orchestrator function for running the entire pipeline.

    Returns:
        nodes_obj: List of node dictionaries
        edges_obj: List of edge dictionaries
    """
    global_start = time.time()

    # STEP 1: Extract and preprocess text
    
    raw_text = preprocess_html_article(html_path)
    paragraphs = generate_paragraphs(raw_text, split_length=10)

    # STEP 2: Patient timeline prediction
    case_report = generate_patient_timeline(paragraphs)

    print("\n🧹 Cleaned Case Report (filtered output):")
    print("=" * 60)
    print(case_report)
    print("=" * 60)

    # STEP 3: Node and edge generation
    nodes_obj = run_node_generation(case_report)
    annotate_nodes_with_clinical_data(nodes_obj)

    edges_obj = run_edge_generation(nodes_obj)
    # Annotate edges in place with the collapsible, stack-nested branch structure
    # (branch_flag / branch_role / branch_id / branch_depth). The returned span
    # index is serialized into the graph JSON in a later step.
    match_branches(edges_obj)

    # STEP 4: Print outputs
    print_nodes(nodes_obj)
    print_edges(edges_obj)

    global_end = time.time()
    print(f"\n=== Full DAG generation pipeline completed in {global_end - global_start:.2f} seconds ===")

    return nodes_obj, edges_obj



















def format_nodes(raw_nodes):
    """
    Format node list for storage, including custom data.
    """
    return [
        {
            "id": f"N{idx + 1}",
            "label": f"Step {idx + 1}",
            "customData": node
        }
        for idx, node in enumerate(raw_nodes)
    ]

def format_edges(raw_edges, node_lookup):
    """
    Format directed edge list for storage, including full edge data.

    Resilient to LLM output that doesn't follow the ``X_to_Y`` edge_id format or
    references unknown nodes: such edges are skipped with a warning rather than
    crashing the whole graph.
    """
    formatted = []
    for edge in raw_edges:
        edge_id = edge.get("edge_id", "") if isinstance(edge, dict) else ""
        if "_to_" not in edge_id:
            print(f"Warning: skipping edge with malformed edge_id: {edge_id!r}")
            continue
        source, target = edge_id.split("_to_", 1)
        if source not in node_lookup or target not in node_lookup:
            print(f"Warning: skipping edge {edge_id!r}; unknown source/target node.")
            continue
        formatted.append({
            "from": f"N{node_lookup[source] + 1}",
            "to": f"N{node_lookup[target] + 1}",
            "data": edge
        })
    return formatted

def save_json_file(data, output_path):
    """
    Save dictionary data as a JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)
    print(f"✅ Saved graph JSON: {output_path}")

def append_csv_row(csv_path, row, header):
    """
    Append a row to a CSV file, creating the file if it doesn't exist.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = csv_path.is_file()

    with csv_path.open('a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
    print(f"✅ Updated CSV: {csv_path}")

def process_and_save_graph(raw_nodes, raw_edges, graph_id, source_file, output_dir):
    """
    Process raw graph data, save JSON, and log metadata.
    

    """
    

    node_lookup = {
    node["node_id"]: idx
    for idx, node in enumerate(raw_nodes)
    if node and "node_id" in node
}
    valid_raw_nodes = [node for node in raw_nodes if node and "node_id" in node]
    formatted_nodes = format_nodes(valid_raw_nodes)
    formatted_edges = format_edges(raw_edges, node_lookup)

    # Collapsible branch index for downstream causal modeling — reconstructed
    # deterministically from the edge annotations (no LLM at save time).
    try:
        bdag = BranchDAG.from_pipeline(valid_raw_nodes, raw_edges)
        branches = [span.to_dict() for span in bdag.spans]
        branch_stack = bdag.branch_stack()
    except Exception as exc:
        print(f"Warning: could not build branch index for {graph_id}: {exc}")
        branches, branch_stack = [], []

    graph_filename = f"{graph_id}.json"
    graph_path = Path(output_dir) / graph_filename
    metadata_csv = Path(output_dir) / "graph_metadata.csv"

    save_json_file({
        "nodes": formatted_nodes,
        "edges": formatted_edges,
        "branches": branches,
        "branch_stack": branch_stack,
    }, graph_path)

    metadata_row = {
        "graph_id": graph_id,
        "json_path": str(graph_path),
        "source_file": source_file
    }
    append_csv_row(metadata_csv, metadata_row, header=["graph_id", "json_path", "source_file"])








def generate_all_graphs(input_folder: str, output_dir: str):

    DSPY_MODEL = os.environ.get("DSPY_MODEL", "gemini/gemini-2.0-flash")
    # Convention is GEMINI_APIKEY (see .config/.env); tolerate the legacy
    # GEMINIAPIKEY spelling too.
    gemini_api_key = os.environ.get("GEMINI_APIKEY") or os.environ.get("GEMINIAPIKEY")

    if "ollama" in DSPY_MODEL:
        lm = dspy.LM(DSPY_MODEL, api_base='http://localhost:11434', api_key='')
    elif not gemini_api_key:
        raise RuntimeError(
            "No Gemini API key found. Set GEMINI_APIKEY in .config/.env, or set "
            "DSPY_MODEL to an ollama/* model to run locally without a key."
        )
    else:
        lm = dspy.LM(DSPY_MODEL, api_key=gemini_api_key, temperature=0.21, cache=False)

    dspy.configure(lm=lm, adapter=dspy.ChatAdapter())

    os.makedirs(output_dir, exist_ok=True)
    graph_counter = 1

    for html_filename in os.listdir(input_folder):
        if not html_filename.lower().endswith(".html"):
            continue

        html_path = os.path.join(input_folder, html_filename)
        graph_id = f"graph_{graph_counter:03d}"
        graph_json_path = os.path.join(output_dir, f"{graph_id}.json")

        if os.path.exists(graph_json_path):
            print(f"✅ Skipping {html_filename} → {graph_id} already exists.")
            graph_counter += 1
            continue

        print(f"\n=== Processing {html_filename} as {graph_id} ===")

        # Isolate per-case failures so one bad report can't abort a corpus run.
        try:
            nodes_obj, edges_obj = run_pipeline(html_path)
            process_and_save_graph(
                raw_nodes=nodes_obj,
                raw_edges=edges_obj,
                graph_id=graph_id,
                source_file=html_path,
                output_dir=output_dir
            )
        except Exception as exc:
            import traceback
            print(f"❌ Failed on {html_filename} ({graph_id}): {exc}")
            traceback.print_exc()

        graph_counter += 1


if __name__ == "__main__":
    env_path = os.environ.get("config_path")
    load_dotenv(env_path)
    generate_all_graphs(
        input_folder="./pmc_htmls",
        output_dir="./webapp/static/graphs"
    )





