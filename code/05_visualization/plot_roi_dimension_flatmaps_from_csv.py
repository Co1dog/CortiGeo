"""
This script generates Pycortex flatmaps from ROI-level intrinsic dimension CSV
files. It loads one or more CSV files containing region and dimension columns,
matches each ROI name to the Harvard-Oxford cortical atlas, maps dimension values
back into atlas voxels, resamples the resulting volume into Pycortex subject
space, fills small missing regions using local nearest-neighbor interpolation,
and saves a flatmap for each input CSV. The script automatically selects a
sequential or diverging colormap based on the value range.
"""

import os
import re
import tempfile
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import cortex

from nilearn import datasets, image
from scipy.ndimage import generic_filter


template = r'inkscape "{svg}" --export-type=png --export-filename="{png}" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")
OUTPUT_DIR = "flatmaps"

CSV_LIST = [
    "dimensions_week2_raw_sub-s105.csv",
    "dimensions_week2_diff_sub-s105.csv"
]

os.environ["CORTEXFOLDER"] = DB_ROOT
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def normalize_name(text):
    text = str(text).lower()

    for token in ["left", "right", "hemisphere"]:
        text = text.replace(token, " ")

    text = text.replace("_", " ")

    return "".join(
        character
        for character in text
        if character.isalnum()
    )


atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = atlas.labels

name_to_indices = {}

for index, label in enumerate(atlas_labels):
    if index == 0:
        continue

    key = normalize_name(label)
    name_to_indices.setdefault(key, []).append(index)


def find_label_indices(csv_region):
    if str(csv_region).startswith("region_"):
        name_part = str(csv_region).split("region_", 1)[-1]
    else:
        name_part = str(csv_region)

    key = normalize_name(name_part)

    if key in name_to_indices:
        return name_to_indices[key]

    candidate_indices = [
        indices
        for label_key, indices in name_to_indices.items()
        if key in label_key or label_key in key
    ]

    if candidate_indices:
        return sorted(
            {
                index
                for indices in candidate_indices
                for index in indices
            }
        )

    return []


ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape

for csv_path in CSV_LIST:
    if not os.path.exists(csv_path):
        print(f"[Skipped] CSV not found: {csv_path}")
        continue

    df = pd.read_csv(csv_path)

    if "region" not in df.columns or "dimension" not in df.columns:
        print(f"[Skipped] {csv_path} does not contain 'region' and 'dimension' columns")
        continue

    csv_values = pd.to_numeric(df["dimension"], errors="coerce").to_numpy()
    csv_min = np.nanmin(csv_values)
    csv_max = np.nanmax(csv_values)

    if not np.isfinite(csv_min) or not np.isfinite(csv_max):
        print(f"[Skipped] {csv_path}: all dimension values are NaN")
        continue

    if csv_min == csv_max:
        epsilon = 1e-6 if csv_max == 0 else abs(csv_max) * 0.05
        vmin_csv = csv_min - epsilon
        vmax_csv = csv_max + epsilon
    else:
        vmin_csv = csv_min
        vmax_csv = csv_max

    if vmin_csv >= 0:
        cmap_name = "inferno"
        vmin_for_plot = vmin_csv
        vmax_for_plot = vmax_csv
    else:
        vmax_symmetric = max(abs(vmin_csv), abs(vmax_csv))
        cmap_name = "RdBu_r"
        vmin_for_plot = -vmax_symmetric
        vmax_for_plot = vmax_symmetric

    dimension_map = np.zeros_like(atlas_data, dtype=np.float32)
    unmatched_regions = []

    for _, row in df.iterrows():
        roi_name = str(row["region"])

        try:
            value = float(row["dimension"])
        except Exception:
            continue

        label_indices = find_label_indices(roi_name)

        if not label_indices:
            unmatched_regions.append(roi_name)
            continue

        for label_index in label_indices:
            dimension_map[atlas_data == label_index] = value

    if unmatched_regions:
        print(f"[Info] Unmatched ROI labels in {csv_path}:")

        for unmatched_region in unmatched_regions[:10]:
            print(f"   - {unmatched_region}")

        if len(unmatched_regions) > 10:
            print(f"   ... {len(unmatched_regions)} unmatched regions in total")

    mni_dimension_img = nib.Nifti1Image(
        dimension_map,
        atlas_img.affine
    )

    subject_img = image.resample_img(
        mni_dimension_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="continuous",
        force_resample=True,
        copy_header=True
    )

    data = subject_img.get_fdata().astype(np.float32)

    if data.shape != xfm_ref_shape:
        if data.shape[::-1] == xfm_ref_shape:
            data = data.transpose(2, 1, 0)
        else:
            raise ValueError(f"Data shape mismatch: {data.shape} != {xfm_ref_shape}")

    data[data == 0] = np.nan

    if not np.isfinite(data).any():
        print(f"[Warning] {csv_path}: mapped volume is fully NaN")
        continue

    filled_data = data.copy()

    for _ in range(3):
        filled_data = generic_filter(
            filled_data,
            fill_nan_with_nearest,
            size=3,
            mode="mirror"
        )

    fig = plt.figure(figsize=(12, 6), dpi=300)

    volume = cortex.Volume(
        filled_data,
        SUBJECT,
        XFM,
        cmap=cmap_name,
        vmin=vmin_for_plot,
        vmax=vmax_for_plot
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

    stem = os.path.splitext(os.path.basename(csv_path))[0]
    output_path = os.path.join(OUTPUT_DIR, f"{SUBJECT}_{stem}.png")

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(
        f"Saved: {output_path} | "
        f"vmin={vmin_for_plot:.4g}, vmax={vmax_for_plot:.4g}, cmap={cmap_name}"
    )