"""
This script visualizes ROI-level Procrustes correlation results on a brain map.
It reads a CSV file containing region-wise correlation coefficients, maps each
ROI correlation value back to the Harvard-Oxford cortical atlas, saves the
resulting correlation volume as a NIfTI image, displays a middle-slice heat map,
and visualizes the correlation map over an MNI152 anatomical background.
"""

import re
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

from nilearn import datasets, plotting


CORRELATION_CSV = "analysis_results/correlations_procrustes.csv"
OUTPUT_NII = "analysis_results/correlation_map_from_csv.nii.gz"


def main():
    corr_df = pd.read_csv(CORRELATION_CSV, index_col=0)

    harvard_oxford = datasets.fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-2mm"
    )

    if isinstance(harvard_oxford.maps, nib.Nifti1Image):
        atlas_img = harvard_oxford.maps
    else:
        atlas_img = nib.load(harvard_oxford.maps)

    atlas_data = atlas_img.get_fdata()

    corr_map = np.zeros_like(atlas_data, dtype=np.float32)

    for region_name, correlation_value in corr_df["correlation"].items():
        match = re.match(r"region_(\d+)_", region_name)

        if match:
            region_index = int(match.group(1))
            corr_map[atlas_data == region_index] = correlation_value

    corr_nii = nib.Nifti1Image(
        corr_map,
        affine=atlas_img.affine
    )

    nib.save(corr_nii, OUTPUT_NII)

    mid_z = corr_map.shape[2] // 2

    plt.figure(figsize=(8, 6))

    plt.imshow(
        corr_map[:, :, mid_z].T,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        origin="lower",
        interpolation="nearest",
        aspect="equal"
    )

    plt.colorbar(label="Pearson r")
    plt.title("ROI correlation")
    plt.tight_layout()
    plt.show()

    finite_corr = corr_df["correlation"].dropna().values

    if finite_corr.size > 0:
        vmax = np.nanmax(np.abs(finite_corr))
    else:
        vmax = 1.0

    if vmax == 0:
        vmax = 1.0

    plotting.plot_stat_map(
        corr_nii,
        bg_img=datasets.load_mni152_template(),
        display_mode="ortho",
        cmap="RdBu_r",
        threshold=0,
        vmax=vmax
    )

    plotting.show()


if __name__ == "__main__":
    main()