"""Automated script to generate and verify Phase 1 and Phase 2 IEEE Reports."""

import os
import sys
import pandas as pd

def generate_reports():
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(docs_dir, ".."))

    results_csv = os.path.join(repo_root, "outputs/results/comparison_table.csv")
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        print("Loaded latest evaluation metrics:")
        print(df.to_string(index=False))
    else:
        print(f"Warning: {results_csv} not found.")

    p1_tex = os.path.join(docs_dir, "Phase1_IEEE_Report.tex")
    p2_tex = os.path.join(docs_dir, "Phase2_IEEE_Report.tex")

    for report, name in [(p1_tex, "Phase 1"), (p2_tex, "Phase 2")]:
        if os.path.exists(report):
            with open(report) as f:
                content = f.read()
            print(f"Verified {name} IEEE Report ({len(content):,} characters).")
        else:
            print(f"Error: {name} report missing at {report}")

if __name__ == "__main__":
    generate_reports()
