"""
This script visualizes ROI-level correlation results on the cortical surface.
It loads a CSV file containing brain-region correlation values, standardizes
region names, excludes predefined unreliable regions, matches each region to
the Harvard-Oxford cortical atlas using loose and fuzzy name matching, maps the
correlation coefficients back into atlas voxels, resamples the statistical image
to the MNI152 template, and displays the result on lateral, medial, and dorsal
surface views for both hemispheres.
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


CSV_PATH = "correlations_dim.csv"

EXCLUDED_REGIONS = [
    "region_13_InferiorTemporalGyrusanteriordivision",
    "region_45_TemporalFusiformCortexanteriordivision",
    "region_27_OccipitalPole"
]


def prettify_region_name(region_name):
    region_name = str(region_name)
    region_name = re.sub(r"region_\d+_", "", region_name)
    region_name = re.sub(r"([a-z])([A-Z])", r"\1 \2", region_name)
    region_name = re.sub(r"([a-z])division", r"\1 division", region_name)
    return region_name.strip()


def normalize_text(text):
    return re.sub(r"[^a-z]+", "", str(text).lower())


def load_correlation_table(csv_path):
    df = pd.read_csv(csv_path)

    if "region" not in df.columns:
        df.rename(columns={df.columns[0]: "region"}, inplace=True)

    if "correlation" not in df.columns:
        numeric_columns = [
            column
            for column in df.columns
            if column != "region" and np.issubdtype(df[column].dtype, np.number)
        ]

        if not numeric_columns:
            raise ValueError("The CSV file must contain a correlation column.")

        df.rename(columns={numeric_columns[0]: "correlation"}, inplace=True)

    df["pretty"] = df["region"].apply(prettify_region_name)
    df["correlation"] = pd.to_numeric(df["correlation"], errors="coerce")

    df.loc[df["region"].isin(EXCLUDED_REGIONS), "correlation"] = np.nan

    return df


df = load_correlation_table(CSV_PATH)

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
    value = row["correlation"]

    if pd.isna(value):
        continue

    region = str(row["region"])
    numeric_match = re.match(r"(?:region_)?(\d+)", region)

    if numeric_match:
        atlas_index = int(numeric_match.group(1))

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