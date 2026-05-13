"""
This script computes a voxel-wise intrinsic dimension map for one subject and
one BOLD fMRI video. It loads a 4D BOLD image, identifies nonzero brain voxels,
estimates intrinsic dimension from each voxel's standardized temporal-difference
signal, saves the resulting voxel-wise dimension map as a NIfTI file, and
visualizes the map as a Pycortex cortical flatmap.
"""

import os
import tempfile
import warnings
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import cortex
import skdim

from sklearn.preprocessing import StandardScaler
from nilearn.image import resample_to_img


warnings.filterwarnings("ignore")

os.environ["CORTEX_INKSCAPE_CMD"] = (
    r'inkscape "{svg}" --export-type=png --export-filename="{png}" --export-area-drawing'
)
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")
os.environ["CORTEXFOLDER"] = DB_ROOT

OUTPUT_DIR = "whole_brain_dimension"
SCORE_CSV = "subject_weekly_data2.csv"
BASE_DIR = r"E:\ThinkLikeExperts"
WEEK_ID = 2
VIDEO_ID = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_voxel_dimension(time_series):
    if np.allclose(time_series, time_series[0]):
        return np.nan

    time_series_norm = StandardScaler().fit_transform(
        time_series.reshape(-1, 1)
    ).ravel()

    temporal_diff = np.diff(time_series_norm).reshape(-1, 1)

    if temporal_diff.shape[0] < 10:
        return np.nan

    n_neighbors = min(10, temporal_diff.shape[0] - 1)

    try:
        estimator = skdim.id.MLE(k=n_neighbors)
        dimensions = estimator.fit_transform(temporal_diff)
        dimension = float(np.mean(dimensions))

        return dimension if np.isfinite(dimension) else np.nan

    except Exception:
        return np.nan


df = pd.read_csv(SCORE_CSV)
subject_id = df["participant_id"].values[0]

template_path = os.path.join(
    BASE_DIR,
    f"sub-{subject_id}",
    f"ses-wk{WEEK_ID}",
    "func",
    f"sub-{subject_id}_ses-wk{WEEK_ID}_task-vid{VIDEO_ID}_bold.nii.gz"
)

template_img = nib.load(template_path)
data = template_img.get_fdata()

brain_mask = np.any(data != 0, axis=-1)
dimension_map = np.full(brain_mask.shape, np.nan)

total_voxels = np.sum(brain_mask)
processed_count = 0

for x in range(brain_mask.shape[0]):
    for y in range(brain_mask.shape[1]):
        for z in range(brain_mask.shape[2]):
            if brain_mask[x, y, z]:
                dimension_map[x, y, z] = calculate_voxel_dimension(data[x, y, z, :])
                processed_count += 1

                if processed_count % 1000 == 0:
                    progress = processed_count / total_voxels * 100
                    print(f"Processed {processed_count}/{total_voxels} ({progress:.1f}%)")

valid_dimensions = dimension_map[~np.isnan(dimension_map)]

if valid_dimensions.size == 0:
    print("No valid dimensions were computed. Please check the data and settings.")
    raise SystemExit

vmax = np.nanpercentile(valid_dimensions, 95)
vmin = np.nanpercentile(valid_dimensions, 5)

adjusted_map = np.copy(dimension_map)
adjusted_map[np.isnan(adjusted_map)] = 0
adjusted_map = np.clip(adjusted_map, vmin, vmax)

ref_img = nib.load(REF_IMG_PATH)

dimension_img = nib.Nifti1Image(adjusted_map, template_img.affine)
resampled_img = resample_to_img(
    dimension_img,
    ref_img,
    interpolation="continuous"
)

resampled_data = np.transpose(resampled_img.get_fdata(), (2, 1, 0))

volume = cortex.Volume(
    resampled_data,
    SUBJECT,
    XFM,
    ref_img.affine,
    cmap="viridis",
    vmin=vmin,
    vmax=vmax
)

fig = plt.figure(figsize=(12, 6), dpi=300)

cortex.quickflat.make_figure(
    volume,
    with_curvature=False,
    with_sulci=False,
    with_rois=False,
    with_labels=False,
    with_colorbar=True,
    colorbar_location="right",
    fig=fig,
    linewidth=2,
    labelsize="20pt"
)

fig.text(
    0.5,
    0.06,
    "Whole Brain Voxel-wise Intrinsic Dimensionality",
    fontsize=17,
    fontweight="bold",
    ha="center",
    va="center",
    transform=fig.transFigure
)

flatmap_path = os.path.join(OUTPUT_DIR, "whole_brain_dimension_flatmap.png")
plt.savefig(flatmap_path, dpi=300, bbox_inches="tight")
plt.close()

nifti_path = os.path.join(OUTPUT_DIR, "voxel_dimensions.nii.gz")
nib.save(
    nib.Nifti1Image(dimension_map, template_img.affine, template_img.header),
    nifti_path
)

print("\nDimensionality statistics:")
print(f"Total voxels processed: {total_voxels}")
print(f"Valid dimension calculations: {valid_dimensions.size}")
print(f"Mean dimension: {np.mean(valid_dimensions):.2f}")
print(f"Median dimension: {np.median(valid_dimensions):.2f}")
print(f"Minimum dimension: {np.min(valid_dimensions):.2f}")
print(f"Maximum dimension: {np.max(valid_dimensions):.2f}")
print(f"5th percentile: {vmin:.2f}")
print(f"95th percentile: {vmax:.2f}")

print("\nResults saved to:")
print(f"- Flatmap: {flatmap_path}")
print(f"- NIfTI: {nifti_path}")