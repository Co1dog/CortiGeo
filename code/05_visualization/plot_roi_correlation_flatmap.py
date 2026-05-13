"""
This script generates a Pycortex cortical flatmap from ROI-level correlation
results. It loads a correlation CSV file, maps region-wise Pearson correlation
values back to the Harvard-Oxford cortical atlas, resamples the resulting
statistical volume into the Pycortex subject space, fills small missing regions
using local nearest-neighbor interpolation, and saves the final correlation
flatmap as a high-resolution image.
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


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    if len(non_nan) > 0:
        return non_nan[0]

    return np.nan


template = r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
CSV_PATH = "correlations_lifetime.csv"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(
    DB_ROOT,
    SUBJECT,
    "transforms",
    XFM,
    "reference.nii.gz"
)

os.environ["CORTEXFOLDER"] = DB_ROOT

corr_df = pd.read_csv(CSV_PATH, index_col=0)

regions_to_hide = [
    "region_13_InferiorTemporalGyrusanteriordivision",
    "region_45_TemporalFusiformCortexanteriordivision",
    "region_27_OccipitalPole"
]

corr_df.loc[regions_to_hide, "correlation"] = np.nan

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()

corr_map = np.zeros_like(atlas_data, dtype=np.float32)

for roi_name, correlation in corr_df["correlation"].items():
    match = re.match(r"region_(\d+)_", roi_name)

    if match:
        region_index = int(match.group(1))
        corr_map[atlas_data == region_index] = correlation

mni_corr_img = nib.Nifti1Image(corr_map, atlas_img.affine)

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

data[data == 0] = np.nan

filled_data = data.copy()
filled_data[filled_data == 0] = np.nan

filled_data = generic_filter(
    filled_data,
    fill_nan_with_nearest,
    size=3,
    mode="mirror"
)

for _ in range(2):
    filled_data = generic_filter(
        filled_data,
        fill_nan_with_nearest,
        size=3,
        mode="mirror"
    )

vmax = np.nanmax(np.abs(data))

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

os.makedirs("flatmaps", exist_ok=True)

fig.savefig("flatmaps/subj04_corr.png", dpi=300)
plt.close(fig)