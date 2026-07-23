import json
import os
import numpy as np
import pandas as pd
from pathlib import Path

# Load filtered graph IDs from CSV
ids_df = pd.read_csv("final_dataset_ids.csv")
graph_ids = set(ids_df["graph_id"].astype(str).str.zfill(3))


# Define fields
summary_rows = []
detailed_rows = []
acyclic_false_ids = []

# Search for matching files
results_dir = Path(".")
json_files = list(results_dir.glob("results_graph_*.json"))

# Parse only selected files
for file_path in json_files:
    graph_id = file_path.stem.replace("results_graph_", "")
    if graph_id not in graph_ids:
        continue

    row = {"graph_id": graph_id}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # BERTScore
        if "bertscore" in data:
            row["bertscore_precision"] = data["bertscore"]["precision"]
            row["bertscore_recall"] = data["bertscore"]["recall"]
            row["bertscore_f1"] = data["bertscore"]["f1"]

        # BLEU and ROUGE
        row["bleu"] = data.get("bleu", np.nan)
        row["rouge1"] = data.get("rouge1", np.nan)
        row["rougeL"] = data.get("rougeL", np.nan)

        # Topology
        topo = data.get("topology", {})
        row["is_acyclic"] = topo.get("is_acyclic", False)
        row["timestamps_in_order"] = topo.get("timestamps_in_order", False)
        row["all_nodes_have_timestamps"] = topo.get("all_nodes_have_timestamps", False)
        row["weakly_connected_components"] = topo.get("weakly_connected_components", np.nan)
        row["node_count"] = topo.get("node_count", np.nan)
        row["edge_count"] = topo.get("edge_count", np.nan)
        row["avg_in_degree"] = topo.get("avg_in_degree", np.nan)
        row["density"] = topo.get("density", np.nan)

        if not row["is_acyclic"]:
            acyclic_false_ids.append(graph_id)

        detailed_rows.append(row)

    except Exception as e:
        print(f"⚠️ Skipping {file_path.name}: {e}")

# Convert to DataFrame
detailed_df = pd.DataFrame(detailed_rows)

# Compute summary stats
for column in detailed_df.columns:
    if column == "graph_id":
        continue
    if detailed_df[column].dtype == bool:
        percent_true = 100 * detailed_df[column].mean()
        summary_rows.append({
            "field": column,
            "type": "boolean",
            "percent_true": percent_true
        })
    elif pd.api.types.is_numeric_dtype(detailed_df[column]):
        summary_rows.append({
            "field": column,
            "type": "numeric",
            "mean": detailed_df[column].mean(),
            "std": detailed_df[column].std()
        })

summary_df = pd.DataFrame(summary_rows)

# Write to Excel with two sheets
output_path = "summary_stats_with_raw.xlsx"
with pd.ExcelWriter(output_path) as writer:
    summary_df.to_excel(writer, sheet_name="summary_stats", index=False)
    detailed_df.to_excel(writer, sheet_name="raw_per_graph_data", index=False)

print(f"✅ Excel saved to {output_path}")

# Save list of non-acyclic graph IDs
if acyclic_false_ids:
    print(f"\n⛔ Found {len(acyclic_false_ids)} non-acyclic graphs:")
    print(", ".join(acyclic_false_ids))
    with open("non_acyclic_graph_ids.txt", "w") as f:
        f.write("\n".join(acyclic_false_ids))
    print("📝 Saved: non_acyclic_graph_ids.txt")
else:
    print("\n✅ All selected graphs are acyclic.")
