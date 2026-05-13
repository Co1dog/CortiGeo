"""
This script generates weekly Pycortex value flatmaps for ROI-level mean
intrinsic dimension and mean Wasserstein distance. For each week, it loads the
corresponding ROI summary CSV files, maps region-level values back to the
Harvard-Oxford cortical atlas, resamples the mapped volume into Pycortex subject
space, fills small missing regions using local nearest-neighbor interpolation,
clips values by percentile for stable visualization, and saves high-resolution
flatmaps for both mean dimension and mean Wasserstein distance.
"""

import os
import re
import tempfile

import cortex
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from nilearn import datasets, image
from scipy.ndimage import generic_filter


CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    "cool_to_warm_soft",
    ["#7395dc", "#8390d8", "#dee3fe", "#e8e3dc", "#fdf3e0", "#fae8c1", "#f6d287"],
    N=256
)

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")

os.environ["CORTEXFOLDER"] = DB_ROOT
os.environ["CORTEX_INKSCAPE_CMD"] = (
    r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" '
    r"--export-area-drawing"
)
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def normalize_name(text):
    return re.sub(r"[^a-z]", "", str(text).lower())


def filter_regions(df):
    excluded_regions = [
        "inferior temporal gyrus anterior division",
        "temporal fusiform cortex anterior division",
        "occipital pole"
    ]

    def is_excluded(region_name):
        region_name = str(region_name).lower().replace(",", "").strip()
        return any(region in region_name for region in excluded_regions)

    return df[~df["region"].apply(is_excluded)].copy()


atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
atlas_labels = list(atlas.labels)

name_to_index = {
    normalize_name(label): index
    for index, label in enumerate(atlas_labels)
}

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape


def make_value_flatmap(
    df,
    value_column,
    output_path,
    clip_lower_pct=45,
    clip_upper_pct=80
):
    df = filter_regions(df)

    value_map = np.zeros_like(atlas_data, dtype=np.float32)

    for _, row in df.iterrows():
        region = str(row["region"]).strip()
        value = float(row[value_column]) if not pd.isna(row[value_column]) else np.nan

        match = re.match(r"(?:region_)?(\d+)", region)

        if match:
            atlas_index = int(match.group(1))
        else:
            region_clean = re.sub(r"^region_\d+_", "", region)
            region_clean = re.sub(r"^region_", "", region_clean)
            atlas_index = name_to_index.get(normalize_name(region_clean), None)

        if atlas_index is not None:
            value_map[atlas_data == atlas_index] = value

    mni_value_img = nib.Nifti1Image(value_map, atlas_img.affine)

    subject_img = image.resample_img(
        mni_value_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="continuous"
    )

    data = subject_img.get_fdata().astype(np.float32)

    if data.shape != xfm_ref_shape:
        if data.shape[::-1] == xfm_ref_shape:
            data = data.transpose(2, 1, 0)
        elif data.transpose(1, 0, 2).shape == xfm_ref_shape:
            data = data.transpose(1, 0, 2)
        else:
            raise ValueError(
                f"Data shape mismatch: got {data.shape}, expected {xfm_ref_shape}"
            )

    data[data == 0] = np.nan
    nan_mask = np.isnan(data)

    filled_data = generic_filter(
        data,
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
    filled_data[nan_mask] = np.nan

    finite_values = filled_data[np.isfinite(filled_data)]

    if finite_values.size == 0:
        vmin_raw, vmax_raw = 0.0, 1.0
    else:
        lower_q, upper_q = np.nanpercentile(
            finite_values,
            [clip_lower_pct, clip_upper_pct]
        )

        if lower_q == upper_q:
            delta = 1e-6 if lower_q == 0 else abs(lower_q) * 1e-6
            lower_q = lower_q - delta
            upper_q = upper_q + delta

        vmin_raw, vmax_raw = lower_q, upper_q

    if not np.isfinite(vmin_raw) or not np.isfinite(vmax_raw):
        if finite_values.size == 0:
            vmin_raw, vmax_raw = 0.0, 1.0
        else:
            vmin_raw = np.nanmin(finite_values)
            vmax_raw = np.nanmax(finite_values)

    display_data = filled_data.copy()
    finite_mask = np.isfinite(display_data)
    display_data[finite_mask] = np.clip(
        display_data[finite_mask],
        vmin_raw,
        vmax_raw
    )
    display_data[~np.isfinite(filled_data)] = np.nan

    fig = plt.figure(figsize=(14, 6), dpi=300)

    volume = cortex.Volume(
        display_data,
        SUBJECT,
        XFM,
        cmap=CUSTOM_CMAP,
        vmin=vmin_raw,
        vmax=vmax_raw
    )

    cortex.quickflat.make_figure(
        volume,
        with_curvature=True,
        with_sulci=False,
        with_rois=False,
        with_colorbar=False,
        curvature_brightness=0.9,
        curvature_contrast=0.04,
        nanmean=True,
        fig=fig,
        linewidth=3,
        labelsize="20pt",
        linecolor="black"
    )

    plt.subplots_adjust(top=0.92, bottom=0.12)

    scalar_map = plt.cm.ScalarMappable(
        cmap=CUSTOM_CMAP,
        norm=plt.Normalize(vmin=vmin_raw, vmax=vmax_raw)
    )
    scalar_map.set_array([])

    colorbar = fig.colorbar(
        scalar_map,
        ax=fig.axes,
        orientation="horizontal",
        fraction=0.046,
        pad=0.08
    )

    colorbar.locator = MaxNLocator(nbins=3)
    colorbar.update_ticks()
    colorbar.ax.tick_params(labelsize=30, width=1.5)

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved: {output_path}")


for week_id in range(1, 6):
    dimension_csv = f"voxelwise_results/mean_voxelwise_brain_regions_dimensions_wk{week_id}.csv"
    wasserstein_csv = f"voxelwise_results/roi_wasserstein_week{week_id}.csv"

    output_dir = f"flatmaps_value_week{week_id}"
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n===== Processing Week {week_id} =====")

    if os.path.exists(dimension_csv):
        dimension_df = pd.read_csv(dimension_csv)

        if "mean_dimension" in dimension_df.columns:
            make_value_flatmap(
                dimension_df,
                "mean_dimension",
                output_path=os.path.join(
                    output_dir,
                    f"mean_dimension_flatmap_week{week_id}.png"
                )
            )
        else:
            print(f"[Skipped] Missing column 'mean_dimension': {dimension_csv}")
    else:
        print(f"[Missing] File does not exist: {dimension_csv}")

    if os.path.exists(wasserstein_csv):
        wasserstein_df = pd.read_csv(wasserstein_csv)

        if "mean_wasserstein" in wasserstein_df.columns:
            make_value_flatmap(
                wasserstein_df,
                "mean_wasserstein",
                output_path=os.path.join(
                    output_dir,
                    f"mean_wasserstein_flatmap_week{week_id}.png"
                )
            )
        else:
            print(f"[Skipped] Missing column 'mean_wasserstein': {wasserstein_csv}")
    else:
        print(f"[Missing] File does not exist: {wasserstein_csv}")

print("\nAll value flatmaps have been generated.")