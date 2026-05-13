"""
This script generates a Pycortex cortical flatmap for ROI-level intrinsic
dimension results. It loads a CSV file containing region-wise intrinsic
dimension values, matches each region to the Harvard-Oxford cortical atlas,
maps the dimension values back into atlas space, resamples the resulting volume
into the Pycortex subject space, and saves a high-resolution flatmap showing the
spatial distribution of intrinsic dimension across cortical regions.
"""

import os
import tempfile
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cortex

from nilearn import datasets, image


template = r'inkscape "{svg}" --export-type=png --export-filename="{png}" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")
CSV_PATH = "brain_region_dimensions/region_dimensions.csv"
OUTPUT_DIR = "flatmaps"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "brain_regions_dimension_flatmap.png")

os.environ["CORTEXFOLDER"] = DB_ROOT

dimensions_df = pd.read_csv(CSV_PATH)

atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr25-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = atlas["labels"]

label_to_index = {}

for index, label in enumerate(atlas_labels):
    if label:
        processed_label = (
            label.replace("-", "")
            .replace(" ", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
        )
        label_to_index[processed_label] = index

dimension_map = np.zeros_like(atlas_data, dtype=np.float32)

print("Mapping regions:")

for _, row in dimensions_df.iterrows():
    region_name = row["Region"]
    dimension = row["Intrinsic_Dimension"]

    if region_name in label_to_index:
        region_index = label_to_index[region_name]
        dimension_map[atlas_data == region_index] = dimension
        print(f"Successfully mapped {region_name} with dimension {dimension}")
    else:
        print(f"Warning: no matching atlas region found for {region_name}")

if np.all(dimension_map == 0):
    print("Error: no data was mapped to the atlas")
    print("Available atlas labels:", atlas_labels)
    raise ValueError("No data was mapped to the atlas")

mni_dimension_img = nib.Nifti1Image(dimension_map, atlas_img.affine)

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape

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
        raise ValueError(f"Data shape mismatch: {data.shape} != {xfm_ref_shape}")

data[data == 0] = np.nan
vmax = np.nanmax(data)

fig = plt.figure(figsize=(12, 6), dpi=300)

volume = cortex.Volume(
    data,
    SUBJECT,
    XFM,
    cmap="viridis",
    vmin=0,
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
    labelsize="20pt"
)

fig.text(
    0.5,
    0.06,
    "Intrinsic Dimension",
    fontsize=17,
    fontweight="bold",
    ha="center",
    va="center",
    transform=fig.transFigure
)

plt.colorbar(
    plt.cm.ScalarMappable(
        norm=plt.Normalize(vmin=0, vmax=vmax),
        cmap="viridis"
    ),
    ax=plt.gca(),
    label="Dimension",
    shrink=0.7
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Flatmap saved to: {OUTPUT_PATH}")

mapped_voxels = np.sum(dimension_map > 0)

print("\nMapping statistics:")
print(f"Total atlas labels: {len(atlas_labels)}")
print(f"Mapped voxels: {mapped_voxels}")
print(f"Maximum dimension value: {vmax:.2f}")