"""
This script generates a significance-aware Pycortex flatmap for ROI-level
correlation results. It loads a CSV file containing brain-region correlations
and p-values, maps each region to the Harvard-Oxford cortical atlas, assigns
significant regions their correlation coefficients, assigns non-significant
regions to zero so they appear gray, resamples the resulting statistical volume
into Pycortex subject space, applies local smoothing only to valid regions, and
saves a high-resolution cortical flatmap.
"""

import os
import re
import tempfile
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cortex

from nilearn import datasets, image
from scipy.ndimage import generic_filter
from matplotlib.colors import ListedColormap


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def normalize_name(text):
    return re.sub(r"[^a-z]", "", str(text).lower())


def create_custom_colormap():
    rdbu = plt.cm.get_cmap("RdBu_r")

    color_list = []

    n_negative = 128
    for index in range(n_negative):
        color_list.append(rdbu(index / (2 * n_negative)))

    color_list.append([0.85, 0.85, 0.85, 1.0])

    n_positive = 128
    for index in range(n_positive):
        color_list.append(rdbu((n_negative + index + 1) / (2 * n_negative)))

    custom_cmap = ListedColormap(color_list)
    custom_cmap.set_bad(color="white", alpha=0)

    return custom_cmap


template = r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
CSV_PATH = "correlations_dim_wk2_new.csv"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(
    DB_ROOT,
    SUBJECT,
    "transforms",
    XFM,
    "reference.nii.gz"
)

P_THRESHOLD = 0.05

os.environ["CORTEXFOLDER"] = DB_ROOT

corr_df = pd.read_csv(CSV_PATH)

if "region" not in corr_df.columns:
    corr_df = (
        pd.read_csv(CSV_PATH, index_col=0)
        .reset_index()
        .rename(columns={"index": "region"})
    )

if "p-value" not in corr_df.columns and "p_value" in corr_df.columns:
    corr_df = corr_df.rename(columns={"p_value": "p-value"})

if "correlation" not in corr_df.columns:
    raise ValueError("The CSV file must contain a 'correlation' column.")

if "p-value" not in corr_df.columns:
    raise ValueError("The CSV file must contain a 'p-value' or 'p_value' column.")

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = list(atlas.labels) if hasattr(atlas, "labels") else []
name_to_index = {
    normalize_name(label): index
    for index, label in enumerate(atlas_labels)
}

correlation_map = np.empty_like(atlas_data, dtype=np.float32)
correlation_map[:] = np.nan

significant_region_count = 0

for _, row in corr_df.iterrows():
    region = str(row["region"])
    correlation = row["correlation"]
    p_value = row["p-value"]

    match = re.match(r"region_(\d+)_", region)

    if match:
        atlas_index = int(match.group(1))
    else:
        region_clean = re.sub(r"^region_(?:\d+_)?", "", region)
        atlas_index = name_to_index.get(normalize_name(region_clean), None)

    if atlas_index is None:
        continue

    if pd.isna(correlation) or pd.isna(p_value):
        value = 0.0
    elif float(p_value) > P_THRESHOLD:
        value = 0.0
    else:
        value = float(correlation)
        significant_region_count += 1

    correlation_map[atlas_data == atlas_index] = value

mni_corr_img = nib.Nifti1Image(
    correlation_map,
    atlas_img.affine
)

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape

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
        raise ValueError(f"Data shape mismatch: {data.shape} != {xfm_ref_shape}")

filled_data = data.copy()

mask_to_smooth = ~np.isnan(data) & (data != 0)

if np.any(mask_to_smooth):
    temp_data = data.copy()
    temp_data[~mask_to_smooth] = np.nan

    smoothed_data = generic_filter(
        temp_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )

    smoothed_data = generic_filter(
        smoothed_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )

    filled_data[mask_to_smooth] = smoothed_data[mask_to_smooth]

custom_cmap = create_custom_colormap()

significant_values = filled_data[
    (~np.isnan(filled_data)) & (filled_data != 0)
]

if len(significant_values) > 0:
    vmax = np.max(np.abs(significant_values))
    vmax = max(vmax, 0.1)
else:
    vmax = 0.5

fig = plt.figure(figsize=(12, 6), dpi=300)

volume = cortex.Volume(
    filled_data,
    SUBJECT,
    XFM,
    cmap=custom_cmap,
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

os.makedirs("flatmaps", exist_ok=True)

output_path = f"flatmaps/{os.path.splitext(os.path.basename(CSV_PATH))[0]}_flatmap.png"

fig.savefig(
    output_path,
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print(f"Saved: {output_path}")
print(f"Number of significant regions: {significant_region_count}")
print(f"Number of significant voxels: {len(significant_values)}")
print(f"Correlation range: {-vmax:.3f} to {vmax:.3f}")