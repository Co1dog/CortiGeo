"""
This script applies Benjamini-Hochberg FDR correction to p-value columns in a
batch of CSV result files. It scans an input directory for CSV files, detects the
most likely p-value column using flexible column-name matching, computes
FDR-adjusted q-values while preserving invalid entries as NaN, replaces the
original p-value column with the adjusted values, and saves corrected CSV files
to the output directory.
"""

import os
import numpy as np
import pandas as pd


INPUT_DIR = r"correlation_supplement_results/voxel_wise_k100_results"
OUTPUT_DIR = "fdr_corrected_results_k100"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def bh_fdr_adjust(p_values):
    p_values = np.asarray(p_values, dtype=float)
    invalid_mask = ~np.isfinite(p_values)
    valid_p_values = p_values[~invalid_mask]

    if valid_p_values.size == 0:
        return np.full_like(p_values, np.nan, dtype=float)

    n_tests = valid_p_values.size
    order = np.argsort(valid_p_values)

    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, n_tests + 1, dtype=float)

    q_values = valid_p_values * n_tests / ranks
    q_values_sorted = q_values[order]
    q_values_monotone = np.minimum.accumulate(q_values_sorted[::-1])[::-1]

    adjusted = np.empty_like(q_values)
    adjusted[order] = q_values_monotone
    adjusted = np.clip(adjusted, 0.0, 1.0)

    output = np.full_like(p_values, np.nan, dtype=float)
    output[~invalid_mask] = adjusted

    return output


def normalize_column_name(name):
    return "".join(
        character
        for character in str(name).lower()
        if character.isalnum()
    )


def find_p_value_column(columns):
    candidates = [
        "p",
        "pval",
        "pvalue",
        "pvalueadj",
        "pvaluefdr",
        "padj",
        "q",
        "qval"
    ]

    normalized_map = {
        normalize_column_name(column): column
        for column in columns
    }

    for candidate in candidates:
        key = normalize_column_name(candidate)

        if key in normalized_map:
            return normalized_map[key]

    return None


csv_files = []

if os.path.isdir(INPUT_DIR):
    for file_name in os.listdir(INPUT_DIR):
        if file_name.lower().endswith(".csv"):
            csv_files.append(os.path.join(INPUT_DIR, file_name))
else:
    print(f"[Error] Directory does not exist: {INPUT_DIR}")
    raise SystemExit(1)

print(f"[Info] Found {len(csv_files)} CSV files")

for input_path in csv_files:
    file_name = os.path.basename(input_path)
    output_path = os.path.join(
        OUTPUT_DIR,
        file_name.replace(".csv", "_fdr.csv")
    )

    print(f"[Info] Processing file: {file_name}")

    df = pd.read_csv(input_path)
    p_value_column = find_p_value_column(df.columns)

    if p_value_column is None:
        print(f"[Error] No p-value column found in {file_name}. Skipping.")
        continue

    print(f"  - Using column: {p_value_column}")

    df[p_value_column] = bh_fdr_adjust(df[p_value_column].to_numpy())
    df.to_csv(output_path, index=False)

    print(f"  - Saved: {os.path.basename(output_path)}")

print(f"\n[OK] All files processed. Results saved to: {OUTPUT_DIR}")