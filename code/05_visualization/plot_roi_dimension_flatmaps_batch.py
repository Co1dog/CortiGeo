"""
This script batch-generates Pycortex flatmaps from ROI-level intrinsic dimension
CSV files. It robustly detects region and dimension columns, maps ROI-level
dimension values to the Harvard-Oxford cortical atlas, resamples the mapped
volume into Pycortex subject space, fills small missing areas using local
nearest-neighbor interpolation, and saves high-resolution flatmaps with a custom
soft colormap for each input CSV file.
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

from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, image
from scipy.ndimage import generic_filter


CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "cool_to_warm_soft",
    ["#7395dc", "#8390d8", "#dee3fe", "#fdfdfd", "#fdf3e0", "#fae8c1", "#f6d287"],
    N=256
)


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
        "dimension": [
            "intrinsicdimension", "intrinsic_dim", "dimension", "dim", "id",
            "intrinsicd", "intrinsic", "idimension", "manifolddim", "latentdim"
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
OUTPUT_DIR = "flatmaps_dim"

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
        df = pd.read_csv(csv_path, sep=None, engine="python")
    except Exception:
        df = pd.read_csv(csv_path)

    normalized_columns = [
        normalize_column_name(column)
        for column in df.columns
    ]

    if "region" not in normalized_columns:
        df = df.reset_index()

    region_column = find_best_column(df, "region")
    dimension_column = find_best_column(df, "dimension")

    if region_column is None or dimension_column is None:
        raise ValueError(
            f"Required columns are missing: region={region_column}, "
            f"dimension={dimension_column}, file={csv_path}"
        )

    sub_df = df[[region_column, dimension_column]].copy()
    sub_df = sub_df.rename(
        columns={
            region_column: "region",
            dimension_column: "dimension"
        }
    )

    sub_df["region"] = sub_df["region"].astype(str).str.strip()
    sub_df["dimension"] = pd.to_numeric(sub_df["dimension"], errors="coerce")

    dimension_map = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in sub_df.iterrows():
        region = str(row["region"]).strip()
        dimension = row["dimension"]

        if pd.isna(dimension):
            continue

        match = re.match(r"(?:region_)?(\d+)", region)

        if match:
            atlas_index = int(match.group(1))
        else:
            region_clean = re.sub(r"^region_\d+_", "", region)
            region_clean = re.sub(r"^region_", "", region_clean)
            atlas_index = name_to_index.get(normalize_name(region_clean), None)

        if atlas_index is not None:
            dimension_map[atlas_data == atlas_index] = float(dimension)

    mni_dimension_img = nib.Nifti1Image(dimension_map, atlas_img.affine)

    subject_img = image.resample_img(
        mni_dimension_img,
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

    vmax = np.nanmax(filled_data) if np.isfinite(filled_data).any() else 1.0
    vmin = 10.0

    fig = plt.figure(figsize=(12, 6), dpi=300)

    volume = cortex.Volume(
        filled_data,
        SUBJECT,
        XFM,
        cmap=CUSTOM_CMAP,
        vmin=vmin,
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
    output_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_flatmap_cool_to_warm_soft.png"
    )

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"[OK] Saved: {output_path}")


CSV_PATHS = [
    r"各脑区维度结果/dimensions_wk2_diff.csv",
    r"各脑区维度结果/dimensions_wk2_diff_sub-s105.csv",
    r"各脑区维度结果/dimensions_wk2_diff_sub-s107.csv",
    r"各脑区维度结果/dimensions_wk2_diff_sub-s122.csv",
    r"各脑区维度结果/dimensions_wk2_raw.csv",
    r"各脑区维度结果/dimensions_wk2_raw_sub-s105.csv",
    r"各脑区维度结果/dimensions_wk2_raw_sub-s107.csv",
    r"各脑区维度结果/dimensions_wk2_raw_sub-s122.csv"
]

for csv_path in CSV_PATHS:
    if not os.path.isfile(csv_path):
        print(f"[Warning] File does not exist. Skipping: {csv_path}")
        continue

    try:
        process_one_csv(csv_path)
    except Exception as error:
        print(f"[Error] Failed to process {csv_path}: {error}")