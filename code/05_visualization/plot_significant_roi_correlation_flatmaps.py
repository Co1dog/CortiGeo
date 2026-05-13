"""
This script generates significance-filtered Pycortex flatmaps from ROI-level
correlation CSV files. It robustly detects region, correlation, and p-value
columns, maps each ROI to the Harvard-Oxford cortical atlas, keeps only regions
with p-values below the significance threshold, resamples the resulting
statistical volume into Pycortex subject space, fills small missing areas using
local nearest-neighbor interpolation, and saves high-resolution cortical
flatmaps for each input CSV file.
"""

import os
import re
import tempfile
from typing import Optional

import cortex
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nilearn import datasets, image
from scipy.ndimage import generic_filter


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def normalize_name(text):
    return re.sub(r"[^a-z]", "", str(text).lower())


def normalize_column_name(text):
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def find_best_column(df, role):
    candidates = {
        "region": [
            "region", "roi", "roiname", "regionname", "label", "areaname",
            "name", "atlasregion", "atlaslabel", "harvardoxford",
            "brainregion", "parcel", "id", "index"
        ],
        "correlation": [
            "correlation", "corr", "r", "pearsonr", "spearmanr", "rho",
            "coef", "coefficient", "corrcoef"
        ],
        "pvalue": [
            "pvalue", "p", "pval", "p_value", "p-value", "padj", "p_adj",
            "p-adj", "q", "qvalue", "q_value", "q-value", "fdr", "pfdr",
            "bh", "p_fdr", "p_bh"
        ]
    }

    normalized_map = {
        normalize_column_name(column): column
        for column in df.columns
    }

    for candidate in candidates[role]:
        key = normalize_column_name(candidate)

        if key in normalized_map:
            return normalized_map[key]

        for normalized_column, original_column in normalized_map.items():
            if key in normalized_column:
                return original_column

    return None


template = r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")
OUTPUT_DIR = "flatmaps_wk6"
P_THRESHOLD = 0.05

os.environ["CORTEXFOLDER"] = DB_ROOT
os.makedirs(OUTPUT_DIR, exist_ok=True)

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = list(atlas.labels) if hasattr(atlas, "labels") else []
name_to_index = {
    normalize_name(label): index
    for index, label in enumerate(atlas_labels)
}

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape


def process_one_csv(csv_path):
    print(f"[Info] Processing: {csv_path}")

    try:
        corr_df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        corr_df = pd.read_csv(csv_path)

    normalized_columns = [
        normalize_column_name(column)
        for column in corr_df.columns
    ]

    if "region" not in normalized_columns:
        corr_df = corr_df.reset_index()

    region_column = find_best_column(corr_df, "region")
    correlation_column = find_best_column(corr_df, "correlation")
    p_value_column = find_best_column(corr_df, "pvalue")

    if region_column is None or correlation_column is None:
        raise ValueError(
            f"Required columns are missing: region={region_column}, "
            f"correlation={correlation_column}, file={csv_path}"
        )

    selected_columns = [region_column, correlation_column]

    if p_value_column is not None:
        selected_columns.append(p_value_column)

    sub_df = corr_df[selected_columns].copy()

    rename_map = {
        region_column: "region",
        correlation_column: "correlation"
    }

    if p_value_column is not None:
        rename_map[p_value_column] = "p_value"

    sub_df = sub_df.rename(columns=rename_map)
    sub_df["region"] = sub_df["region"].astype(str).str.strip()
    sub_df["correlation"] = pd.to_numeric(sub_df["correlation"], errors="coerce")

    if "p_value" in sub_df.columns:
        sub_df["p_value"] = pd.to_numeric(sub_df["p_value"], errors="coerce")
    else:
        sub_df["p_value"] = 0.0

    correlation_map = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in sub_df.iterrows():
        region = str(row["region"]).strip()
        correlation = row["correlation"]
        p_value = row["p_value"]

        if pd.isna(correlation) or pd.isna(p_value) or p_value > P_THRESHOLD:
            value = np.nan
        else:
            value = float(correlation)

        match = re.match(r"(?:region_)?(\d+)", region)

        if match:
            atlas_index = int(match.group(1))
        else:
            region_clean = re.sub(r"^region_\d+_", "", region)
            region_clean = re.sub(r"^region_", "", region_clean)
            atlas_index = name_to_index.get(normalize_name(region_clean), None)

        if atlas_index is not None:
            correlation_map[atlas_data == atlas_index] = value

    mni_corr_img = nib.Nifti1Image(correlation_map, atlas_img.affine)

    subject_img = image.resample_img(
        mni_corr_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="continuous"
    )

    data = subject_img.get_fdata().astype(np.float32)

    if data.shape != xfm_ref_shape:
        if data.shape[::-1] == xfm_ref_shape:
            data = data.transpose(2, 1, 0)
        else:
            raise ValueError(
                f"Data shape mismatch: {data.shape} != {xfm_ref_shape}, "
                f"file={csv_path}"
            )

    data[data == 0] = np.nan

    filled_data = data.copy()
    filled_data = generic_filter(
        filled_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )
    filled_data = generic_filter(
        filled_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )

    vmax = np.nanmax(np.abs(data)) if np.isfinite(data).any() else 1.0

    fig = plt.figure(figsize=(12, 6), dpi=300)

    volume = cortex.Volume(
        filled_data,
        SUBJECT,
        XFM,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax
    )

    cortex.quickflat.make_figure(
        volume,
        with_curvature=False,
        with_sulci=False,
        with_rois=False,
        nanmean=True,
        fig=fig,
        linewidth=3,
        labelsize="20pt",
        linecolor="black"
    )

    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{base_name}_flatmap.png")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved: {output_path}")


CSV_PATHS = [
    "correlations_dim_wk6.csv"
]

for csv_path in CSV_PATHS:
    if not os.path.isfile(csv_path):
        print(f"[Warning] File does not exist. Skipping: {csv_path}")
        continue

    try:
        process_one_csv(csv_path)
    except Exception as error:
        print(f"[Error] Failed to process {csv_path}: {error}")