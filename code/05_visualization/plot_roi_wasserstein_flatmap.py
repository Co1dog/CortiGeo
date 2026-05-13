"""
This script generates a cortical flatmap visualization for ROI-level Wasserstein
distance results from fMRI analyses. It loads a region-wise CSV file containing
mean Wasserstein values, matches each anatomical region to the Harvard-Oxford
cortical atlas, maps the ROI values back into atlas space, resamples the volume
into the Pycortex subject space, handles excluded and missing regions, applies a
soft custom colormap, and saves a publication-style flatmap with a horizontal
colorbar.
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator, FuncFormatter


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    non_nan = non_nan[non_nan != 0]

    return non_nan[0] if len(non_nan) > 0 else np.nan


def norm_name(text):
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


def check_region_matching(df, atlas_labels, name_to_index, value_column):
    print("\n" + "=" * 60)
    print("Region matching report")
    print("=" * 60)

    matched_regions = []
    unmatched_regions = []

    for _, row in df.iterrows():
        region = str(row["region"]).strip()
        key = norm_name(region)
        atlas_index = name_to_index.get(key, None)

        if atlas_index is not None:
            matched_regions.append((region, key, atlas_index, row[value_column]))
        else:
            unmatched_regions.append((region, key, row[value_column]))

    print(f"Matched regions: {len(matched_regions)}")
    print(f"Unmatched regions: {len(unmatched_regions)}")
    print(f"Total regions: {len(df)}")
    print(f"Matching rate: {len(matched_regions) / len(df) * 100:.1f}%")

    if matched_regions:
        print("\nSuccessfully matched regions:")
        for region, key, atlas_index, value in sorted(
            matched_regions,
            key=lambda item: item[3],
            reverse=True
        ):
            atlas_label = atlas_labels[atlas_index]
            print(
                f"  {region:50s} -> "
                f"Atlas[{atlas_index}]: {atlas_label:40s} ({value:.6f})"
            )

    if unmatched_regions:
        print("\nUnmatched regions:")
        for region, key, value in sorted(
            unmatched_regions,
            key=lambda item: item[2],
            reverse=True
        ):
            possible_matches = []

            for index, atlas_label in enumerate(atlas_labels):
                normalized_atlas = norm_name(atlas_label)

                if any(word in normalized_atlas for word in key.split() if len(word) > 3):
                    possible_matches.append((index, atlas_label))

                if key in normalized_atlas or normalized_atlas in key:
                    possible_matches.append((index, atlas_label))

            possible_match_text = ""

            if possible_matches:
                possible_match_text = "Possible matches: " + ", ".join(
                    [
                        f"Atlas[{index}]: {label[:30]}"
                        for index, label in possible_matches[:3]
                    ]
                )

            print(f"  {region:50s} -> '{key}' ({value:.6f})")

            if possible_match_text:
                print(f"    {possible_match_text}")

    return len(matched_regions), len(unmatched_regions)


custom_cmap = LinearSegmentedColormap.from_list(
    "cool_to_warm_soft",
    ["#7395dc", "#8390d8", "#dee3fe", "#e8e3dc", "#fdf3e0", "#fae8c1", "#f6d287"],
    N=256
)

template = r'inkscape \"{svg}\" --export-type=png --export-filename=\"{png}\" --export-area-drawing'
os.environ["CORTEX_INKSCAPE_CMD"] = template
os.environ["CORTEX_SVGTEMP"] = tempfile.gettempdir()

SUBJECT = "subj01"
XFM = "full"
DB_ROOT = r"C:\Users\Admin\anaconda3\envs\myenv\share\pycortex\db"
REF_IMG_PATH = os.path.join(DB_ROOT, SUBJECT, "transforms", XFM, "reference.nii.gz")
os.environ["CORTEXFOLDER"] = DB_ROOT

print("Loading Harvard-Oxford Atlas...")
atlas = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
atlas_img = atlas.maps if isinstance(atlas.maps, nib.Nifti1Image) else nib.load(atlas.maps)
atlas_data = atlas_img.get_fdata()
labels = list(atlas.labels)
name2idx = {norm_name(label): index for index, label in enumerate(labels)}

print(f"Number of atlas labels: {len(labels)}")
print("\nFirst 20 atlas labels before and after normalization:")

for index in range(min(20, len(labels))):
    label = labels[index]
    normalized_label = norm_name(label)
    print(f"  {index:2d}. {label:50s} -> '{normalized_label}'")

ref_img = nib.load(REF_IMG_PATH)
xfm_ref_shape = cortex.db.get_xfm(SUBJECT, XFM).shape

print(f"Atlas shape: {atlas_data.shape}")
print(f"Reference image shape: {ref_img.shape}")
print(f"Transform shape: {xfm_ref_shape}")


def make_value_flatmap(df, value_column, output_path, clip_lower_pct=2, clip_upper_pct=98):
    df = filter_regions(df)

    print(f"  Processing {len(df)} brain regions...")

    value_map = np.zeros_like(atlas_data, dtype=np.float32)

    matched_count = 0

    for _, row in df.iterrows():
        region = str(row["region"]).strip()
        value = float(row[value_column]) if not pd.isna(row[value_column]) else np.nan

        key = norm_name(region)
        atlas_index = name2idx.get(key, None)

        if atlas_index is not None:
            mask = atlas_data == atlas_index
            value_map[mask] = value
            matched_count += 1

    print(f"    Successfully matched {matched_count}/{len(df)} regions")

    print("  Marking excluded regions...")

    excluded_regions = [
        "inferior temporal gyrus anterior division",
        "temporal fusiform cortex anterior division",
        "occipital pole"
    ]

    for region in excluded_regions:
        key = norm_name(region)
        atlas_index = name2idx.get(key, None)

        if atlas_index is not None:
            mask = atlas_data == atlas_index
            value_map[mask] = -9999
            print(f"    Excluded region: {region}")

    print("  Resampling into subject space...")

    mni_value_img = nib.Nifti1Image(value_map, atlas_img.affine)

    subject_img = image.resample_img(
        mni_value_img,
        target_affine=ref_img.affine,
        target_shape=ref_img.shape,
        interpolation="nearest"
    )

    data = subject_img.get_fdata().astype(np.float32)

    if data.shape != xfm_ref_shape:
        print(f"  Adjusting data shape: {data.shape} -> {xfm_ref_shape}")

        if data.shape[::-1] == xfm_ref_shape:
            data = data.transpose(2, 1, 0)
        elif data.transpose(1, 0, 2).shape == xfm_ref_shape:
            data = data.transpose(1, 0, 2)
        else:
            raise ValueError(
                f"Data shape mismatch: got {data.shape}, expected {xfm_ref_shape}"
            )

    data[data == 0] = np.nan
    data[data == -9999] = np.nan
    nan_mask = np.isnan(data)

    excluded_mask = np.zeros_like(data, dtype=bool)

    for region in excluded_regions:
        key = norm_name(region)
        atlas_index = name2idx.get(key, None)

        if atlas_index is not None:
            atlas_excluded_mask = atlas_data == atlas_index
            excluded_img = nib.Nifti1Image(
                atlas_excluded_mask.astype(float),
                atlas_img.affine
            )

            excluded_resampled = image.resample_img(
                excluded_img,
                target_affine=ref_img.affine,
                target_shape=ref_img.shape,
                interpolation="nearest"
            )

            excluded_data = excluded_resampled.get_fdata()

            if excluded_data.shape != data.shape:
                print(
                    f"    Adjusting excluded-region shape: "
                    f"{excluded_data.shape} -> {data.shape}"
                )

                if excluded_data.shape[::-1] == data.shape:
                    excluded_data = excluded_data.transpose(2, 1, 0)
                elif excluded_data.transpose(1, 0, 2).shape == data.shape:
                    excluded_data = excluded_data.transpose(1, 0, 2)
                else:
                    print("    Warning: unable to adjust excluded-region shape; skipping")
                    continue

            excluded_mask |= excluded_data > 0

    print("  Filling NaN values...")

    filled = generic_filter(data, fill_nan_with_nearest, size=3, mode="mirror")
    filled = generic_filter(filled, fill_nan_with_nearest, size=3, mode="mirror")

    filled[excluded_mask] = np.nan
    filled[nan_mask] = np.nan

    finite_values = filled[np.isfinite(filled)]

    if finite_values.size == 0:
        vmin_raw = 0.0
        vmax_raw = 1.0
    else:
        lower_q, upper_q = np.nanpercentile(
            finite_values,
            [clip_lower_pct, clip_upper_pct]
        )

        if lower_q == upper_q:
            delta = 1e-6 if lower_q == 0 else abs(lower_q) * 1e-6
            lower_q = lower_q - delta
            upper_q = upper_q + delta

        vmin_raw = lower_q
        vmax_raw = upper_q

    if not np.isfinite(vmin_raw) or not np.isfinite(vmax_raw):
        if finite_values.size == 0:
            vmin_raw = 0.0
            vmax_raw = 1.0
        else:
            vmin_raw = np.nanmin(finite_values)
            vmax_raw = np.nanmax(finite_values)

    print(f"  Value range: [{vmin_raw:.6f}, {vmax_raw:.6f}]")

    display_data = filled.copy()

    display_data[np.isfinite(display_data)] = np.clip(
        display_data[np.isfinite(display_data)],
        vmin_raw,
        vmax_raw
    )

    display_data[~np.isfinite(filled)] = np.nan

    print("  Generating flatmap...")

    fig = plt.figure(figsize=(14, 6), dpi=300)

    volume = cortex.Volume(
        display_data,
        SUBJECT,
        XFM,
        cmap=custom_cmap,
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
        cmap=custom_cmap,
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

    def format_func(value, position):
        return f"{value:.2f}"

    colorbar.ax.xaxis.set_major_formatter(FuncFormatter(format_func))

    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"  [OK] Saved: {output_path}\n")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    csv_file = os.path.join(
        script_dir,
        "voxelwise_results",
        "roi_wasserstein_week5.csv"
    )

    output_dir = os.path.join(script_dir, "flatmap_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"Script directory: {script_dir}")
    print(f"CSV file: {csv_file}")
    print(f"Output directory: {output_dir}\n")

    if not os.path.exists(csv_file):
        print(f"[Error] File does not exist: {csv_file}")
        return

    print("Loading Wasserstein result CSV file...")
    df = pd.read_csv(csv_file)

    print(f"DataFrame shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    if "region" not in df.columns or "mean_wasserstein" not in df.columns:
        print("[Error] CSV file must contain 'region' and 'mean_wasserstein' columns")
        print(f"Available columns: {list(df.columns)}")
        return

    filtered_df = filter_regions(df)
    check_region_matching(filtered_df, labels, name2idx, "mean_wasserstein")

    print("=" * 60)
    print("Generating Wasserstein distribution flatmap")
    print("=" * 60 + "\n")

    output_path = os.path.join(
        output_dir,
        "voxelwise_brain_wasserstein_week5_flatmap.png"
    )

    make_value_flatmap(
        df,
        "mean_wasserstein",
        output_path=output_path,
        clip_lower_pct=10,
        clip_upper_pct=90
    )

    print("=" * 60)
    print("[Done] All flatmaps have been generated")
    print("=" * 60)


if __name__ == "__main__":
    main()