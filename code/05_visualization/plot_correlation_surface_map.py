"""
This script visualizes region-wise correlation results on the cortical surface.
It loads a CSV file containing anatomical region names and correlation values,
standardizes column names, matches each region to the Harvard-Oxford cortical
atlas using numeric, exact-name, loose-name, or fuzzy matching, and writes the
correlation values back into atlas voxels. The resulting statistical volume is
resampled to the MNI152 template and displayed on the cortical surface using
lateral, medial, and dorsal views for both hemispheres.
"""

import re
import difflib
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
import matplotlib.pyplot as plt

from nilearn import datasets
from nilearn.image import resample_to_img
from nilearn.plotting import plot_img_on_surf


CSV_PATH = "correlations_wasserstein_pairs_wk2.csv"


def normalize_column_name(column_name):
    return re.sub(r"[\s\-]+", "", str(column_name).strip().lower())


def normalize_text(text):
    return re.sub(r"[^a-z]+", "", str(text).lower())


def load_table(csv_path):
    df = pd.read_csv(csv_path)
    df.rename(
        columns={column: normalize_column_name(column) for column in df.columns},
        inplace=True
    )

    region_candidates = [
        "region",
        "roi",
        "label",
        "name",
        "regionname",
        "roiname",
        "areaname"
    ]

    correlation_candidates = [
        "correlation",
        "corr",
        "r",
        "coef",
        "pearsonr",
        "spearmanr"
    ]

    def pick_column(columns, candidates):
        for candidate in candidates:
            if candidate in columns:
                return candidate
        return None

    region_column = pick_column(df.columns, region_candidates) or list(df.columns)[0]
    correlation_column = pick_column(df.columns, correlation_candidates)

    if correlation_column is None:
        numeric_columns = [
            column
            for column in df.columns
            if column != region_column and np.issubdtype(df[column].dtype, np.number)
        ]

        if not numeric_columns:
            for column in df.columns:
                if column == region_column:
                    continue

                converted_column = pd.to_numeric(df[column], errors="coerce")

                if converted_column.notna().any():
                    df[column] = converted_column
                    numeric_columns.append(column)

        correlation_column = numeric_columns[0] if numeric_columns else None

        if correlation_column is None:
            raise ValueError(
                "Unable to identify a correlation column in the CSV file."
            )

    result_df = df[[region_column, correlation_column]].copy()
    result_df.rename(
        columns={
            region_column: "region",
            correlation_column: "correlation"
        },
        inplace=True
    )

    def prettify_region_name(region_name):
        region_name = str(region_name)
        region_name = re.sub(r"^region_\d+_", "", region_name)
        region_name = re.sub(r"^region_", "", region_name)
        region_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", region_name)
        region_name = re.sub(r"([a-z])division", r"\1 division", region_name)
        return region_name.strip()

    result_df["pretty"] = result_df["region"].apply(prettify_region_name)
    result_df["correlation"] = pd.to_numeric(
        result_df["correlation"],
        errors="coerce"
    )

    return result_df


df = load_table(CSV_PATH)

regions_to_hide = set()
df.loc[df["region"].isin(regions_to_hide), "correlation"] = np.nan

atlas = datasets.fetch_atlas_harvard_oxford(
    "cort-maxprob-thr0-2mm",
    symmetric_split=True
)

atlas_img = atlas.maps if not isinstance(atlas.maps, str) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata().astype(int)

atlas_labels = [
    label.decode() if isinstance(label, bytes) else label
    for label in atlas.labels
]

label_lookup = {
    label: index
    for index, label in enumerate(atlas_labels)
}

stat_data = np.zeros_like(atlas_data, dtype="float32")

for _, row in df.iterrows():
    region = str(row["region"])
    value = row["correlation"]

    if pd.isna(value):
        continue

    match = re.match(r"(?:region_)?(\d+)", region)

    if match:
        atlas_index = int(match.group(1))

        if (atlas_data == atlas_index).any():
            stat_data[atlas_data == atlas_index] = float(value)
            continue

    candidate_name = row["pretty"]
    normalized_candidate = normalize_text(candidate_name)

    matched_labels = [
        label
        for label in atlas_labels
        if (
            normalized_candidate in normalize_text(label)
            or normalize_text(label) in normalized_candidate
        )
    ]

    if not matched_labels:
        best_label, best_score = max(
            (
                (
                    label,
                    difflib.SequenceMatcher(
                        None,
                        candidate_name.lower(),
                        label.lower()
                    ).ratio()
                )
                for label in atlas_labels
            ),
            key=lambda item: item[1]
        )

        if best_score >= 0.75:
            matched_labels = [best_label]

    for label in matched_labels:
        label_id = label_lookup[label]
        stat_data[atlas_data == label_id] = float(value)

stat_img = nib.Nifti1Image(
    stat_data,
    affine=atlas_img.affine,
    header=atlas_img.header
)

mni_template = datasets.load_mni152_template(resolution=2)

stat_img = resample_to_img(
    stat_img,
    target_img=mni_template,
    interpolation="nearest"
)

nonzero_values = stat_data[(stat_data != 0) & ~np.isnan(stat_data)]

if nonzero_values.size:
    vmin_raw, vmax_raw = np.percentile(nonzero_values, [2, 98])
    max_abs = max(abs(vmin_raw), abs(vmax_raw))
else:
    max_abs = 1.0

if max_abs == 0:
    max_abs = 1.0

vmin = -max_abs
vmax = max_abs

display = plot_img_on_surf(
    stat_map=stat_img,
    views=["lateral", "medial", "dorsal"],
    hemispheres=["left", "right"],
    inflate=False,
    bg_on_data=True,
    cmap="RdBu_r",
    threshold=None,
    vmin=vmin,
    vmax=vmax,
    colorbar=True
)

try:
    colorbar = display.colorbar
except AttributeError:
    colorbar = plt.gcf().axes[-1]

ticks = [-max_abs, 0, max_abs]
tick_labels = [f"{tick:.2f}" for tick in ticks]

if isinstance(colorbar, matplotlib.colorbar.Colorbar):
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels(tick_labels)
else:
    colorbar.set_xticks(ticks)
    colorbar.set_xticklabels(tick_labels)

plt.tight_layout()
plt.show()