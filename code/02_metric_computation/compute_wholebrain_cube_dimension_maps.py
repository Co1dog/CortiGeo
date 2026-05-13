"""
This script computes whole-brain local intrinsic dimension maps for multiple
aligned and masked fMRI files. For each 4D BOLD image, it converts the signal
into temporal-difference activation patterns, slides a 4 x 4 x 4 voxel cube
through the brain volume, estimates local intrinsic dimension using the MLE
estimator, saves the resulting dimension map as a NumPy file, and generates both
surface-view and Pycortex flatmap visualizations for each subject, session, and
video.
"""

import os
import glob
import re
import traceback
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import multiprocessing as mp
import cortex

from nilearn.image import mean_img, resample_to_img
from nilearn.plotting import plot_img_on_surf
from nilearn.datasets import load_mni152_template
from skdim.id import MLE
from scipy.ndimage import generic_filter


shared_data = None


def fill_nan_with_nearest(values):
    center = values[len(values) // 2]

    if not np.isnan(center) and center != 0:
        return center

    non_nan = values[~np.isnan(values)]
    nonzero_values = non_nan[non_nan != 0]

    if len(nonzero_values) > 0:
        return nonzero_values[0]

    return np.nan


def init_worker(data_np):
    global shared_data
    shared_data = data_np


def process_cube(coord):
    x, y, z = coord

    cube = shared_data[x:x + 4, y:y + 4, z:z + 4, :]

    if x == 0 and y == 0 and z == 0:
        print(f"Debug: cube shape: {cube.shape}")

    if np.count_nonzero(cube[:, :, :, 0] == 0) > 60:
        return None

    cube_reordered = np.transpose(cube, (3, 0, 1, 2))

    if x == 0 and y == 0 and z == 0:
        print(f"Debug: reordered cube shape: {cube_reordered.shape}")

    time_points = cube_reordered.shape[0]
    cube_flattened = cube_reordered.reshape(time_points, -1)

    try:
        dimension = MLE().fit_transform(
            cube_flattened,
            n_neighbors=10
        )
        return x, y, z, dimension

    except Exception as error:
        if x == 0 and y == 0 and z == 0:
            print(f"Debug: MLE error: {error}")
        return None


def process_single_subject(file_path):
    print(f"Processing file: {file_path}")

    match = re.search(r"sub-(s\d+)_ses-(wk\d)_task-(vid\d)", file_path)

    if not match:
        print(f"Could not parse subject information from {file_path}")
        return

    subject_id, session, video = match.groups()

    img = nib.load(file_path)
    data = img.get_fdata()

    print(f"Original data shape: {data.shape}")

    data = data[:, :, :, 1:] - data[:, :, :, :-1]

    print(f"Data shape after temporal differencing: {data.shape}")

    dimension_result = np.zeros_like(data)

    coords = [
        (x, y, z)
        for x in range(0, data.shape[0] - 3)
        for y in range(0, data.shape[1] - 3)
        for z in range(0, data.shape[2] - 3)
    ]

    print(f"Number of coordinates to process: {len(coords)}")

    with mp.Pool(
        processes=mp.cpu_count(),
        initializer=init_worker,
        initargs=(data,)
    ) as pool:
        results = pool.map(process_cube, coords)

    valid_results = [
        result
        for result in results
        if result is not None
    ]

    print(f"Number of valid results: {len(valid_results)}")

    for item in valid_results:
        x, y, z, dimension = item
        dimension_result[x:x + 4, y:y + 4, z:z + 4, :] = dimension

    output_base = "whole_brain_dimension_visualization"

    for subdir in ["dimension_results", "surface_views", "flatmaps"]:
        os.makedirs(os.path.join(output_base, subdir), exist_ok=True)

    npy_filename = f"sub-{subject_id}_{session}_{video}_dimension_diff.npy"
    np.save(
        os.path.join(output_base, "dimension_results", npy_filename),
        dimension_result
    )

    print(f"Saved dimension result to {npy_filename}")

    print("Generating surface visualization...")

    new_img = nib.Nifti1Image(
        dimension_result,
        affine=img.affine,
        header=img.header
    )

    mni152_2mm_template = load_mni152_template(resolution=2)

    new_img_resampled = resample_to_img(
        new_img,
        target_img=mni152_2mm_template,
        interpolation="continuous"
    )

    mean_modified = mean_img(new_img_resampled)

    vmin = np.percentile(dimension_result, 2)
    vmax = np.percentile(dimension_result, 98)

    plt.figure(figsize=(15, 5))

    plot_img_on_surf(
        stat_map=mean_modified,
        views=["lateral", "medial", "dorsal"],
        hemispheres=["left", "right"],
        colorbar=True,
        inflate=False,
        bg_on_data=True,
        cmap="RdYlBu_r",
        vmin=vmin,
        vmax=vmax,
        title=f"Surface Plot: {subject_id} {session} {video}"
    )

    surface_filename = f"surface_sub-{subject_id}_{session}_{video}.png"
    plt.savefig(os.path.join(output_base, "surface_views", surface_filename))
    plt.close()

    print(f"Saved surface plot to {surface_filename}")

    print("Generating flatmap visualization...")

    try:
        pycortex_subject = "subj01"
        xfm = "full"

        ref_shape = cortex.db.get_xfm(pycortex_subject, xfm).reference.shape
        print(f"Reference shape for flatmap: {ref_shape}")

        data_for_flatmap = mean_modified.get_fdata()
        print(f"Original data shape for flatmap: {data_for_flatmap.shape}")

        scale_factors = np.array(ref_shape) / np.array(data_for_flatmap.shape)
        new_affine = np.diag(list(scale_factors) + [1.0])

        temp_img = nib.Nifti1Image(data_for_flatmap, affine=new_affine)
        ref_img = nib.Nifti1Image(np.zeros(ref_shape), affine=np.eye(4))

        resampled_data = resample_to_img(
            temp_img,
            ref_img,
            interpolation="linear"
        ).get_fdata()

        print(f"Resampled data shape: {resampled_data.shape}")

        resampled_data = resampled_data.transpose(2, 1, 0)
        resampled_data[resampled_data == 0] = np.nan

        filled_data = generic_filter(
            resampled_data,
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

        fig = plt.figure(figsize=(12, 6), dpi=300)

        volume = cortex.Volume(
            filled_data,
            pycortex_subject,
            xfm,
            cmap="OrRd",
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
            linewidth=2,
            labelsize="20pt",
            linecolor="black",
            title=f"Flatmap: {subject_id} {session} {video}"
        )

        flatmap_filename = f"flatmap_sub-{subject_id}_{session}_{video}.png"
        plt.savefig(os.path.join(output_base, "flatmaps", flatmap_filename))
        plt.close()

        print(f"Saved flatmap to {flatmap_filename}")

    except Exception as error:
        print(f"Error creating flatmap for {subject_id}: {error}")
        traceback.print_exc()

    print(f"Completed processing for {subject_id} {session} {video}")


def main():
    os.makedirs("whole_brain_dimension_visualization", exist_ok=True)

    base_path = "ThinkLikeExperts"

    pattern = os.path.join(
        base_path,
        "sub-*",
        "ses-*",
        "func",
        "aligned_masked_sub-*_ses-*_task-*_bold.nii.gz"
    )

    files = glob.glob(pattern)

    if not files:
        print("No matching files found")
        return

    print(f"Found {len(files)} files to process")

    for file_path in files:
        print(f"\nProcessing {file_path}")
        process_single_subject(file_path)


if __name__ == "__main__":
    main()