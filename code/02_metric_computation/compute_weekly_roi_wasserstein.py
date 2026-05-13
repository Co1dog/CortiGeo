"""
This script computes weekly ROI-level Wasserstein distance summaries from
multi-video fMRI data. For each subject and recording week, it loads all
available video-specific BOLD images, converts them into temporal-difference
signals, and computes local Wasserstein distances between every pair of videos
using sliding 4 x 4 x 4 voxel cubes. The resulting voxel-wise Wasserstein maps
are cached, aggregated within Harvard-Oxford cortical atlas regions, averaged
across video pairs and subjects, and saved as weekly ROI-level CSV files.
"""

import os
import traceback
from itertools import combinations
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import nibabel as nib
import ot

from tqdm import tqdm
from nilearn.image import resample_to_img
from nilearn.datasets import fetch_atlas_harvard_oxford
from sklearn.preprocessing import StandardScaler


shared_data_pairs = None


def init_worker(data_pairs):
    global shared_data_pairs
    shared_data_pairs = data_pairs


def wasserstein_distance_between_cubes(cube_1, cube_2):
    try:
        if cube_1.shape[0] < 3 or cube_2.shape[0] < 3:
            return np.nan

        combined_data = np.vstack([cube_1, cube_2])
        combined_data = StandardScaler().fit_transform(combined_data)

        if np.isnan(combined_data).any():
            return np.nan

        cube_1_norm = combined_data[:cube_1.shape[0]]
        cube_2_norm = combined_data[cube_1.shape[0]:]

        diff_1 = cube_1_norm[1:] - cube_1_norm[:-1]
        diff_2 = cube_2_norm[1:] - cube_2_norm[:-1]

        cost_matrix = ot.dist(diff_1, diff_2)

        weights_1 = np.ones(len(diff_1)) / len(diff_1)
        weights_2 = np.ones(len(diff_2)) / len(diff_2)

        wasserstein_distance = float(ot.emd2(weights_1, weights_2, cost_matrix))
        return wasserstein_distance

    except Exception:
        traceback.print_exc()
        return np.nan


def process_cube_pair(coord):
    x, y, z = coord

    try:
        data_pairs = shared_data_pairs

        cube_1 = data_pairs[0][x:x + 4, y:y + 4, z:z + 4, :]
        cube_2 = data_pairs[1][x:x + 4, y:y + 4, z:z + 4, :]

        cube_1 = np.transpose(cube_1, (3, 0, 1, 2))
        cube_2 = np.transpose(cube_2, (3, 0, 1, 2))

        cube_1 = cube_1.reshape(cube_1.shape[0], -1)
        cube_2 = cube_2.reshape(cube_2.shape[0], -1)

        wasserstein_distance = wasserstein_distance_between_cubes(cube_1, cube_2)

        if not np.isfinite(wasserstein_distance):
            return None

        return x, y, z, wasserstein_distance

    except Exception:
        traceback.print_exc()
        return None


def calculate_wasserstein_distance(data_1, data_2, coords, save_path, n_workers=None):
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    wasserstein_result = np.zeros_like(data_1)

    init_worker([data_1, data_2])

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        results = executor.map(process_cube_pair, coords)

        for result in tqdm(
            results,
            total=len(coords),
            desc="Computing Wasserstein"
        ):
            if result is not None:
                x, y, z, wasserstein_distance = result
                wasserstein_result[x:x + 4, y:y + 4, z:z + 4, :] = wasserstein_distance

    try:
        np.save(save_path, wasserstein_result)
    except Exception:
        traceback.print_exc()

    return wasserstein_result


def calculate_roi_wasserstein(wasserstein_result, atlas_img, atlas_labels, reference_img):
    if not hasattr(atlas_img, "get_fdata"):
        atlas_img = nib.load(atlas_img)

    atlas_shape = atlas_img.get_fdata().shape

    if atlas_shape != wasserstein_result.shape[:3]:
        atlas_img = resample_to_img(
            atlas_img,
            reference_img,
            interpolation="nearest"
        )
        atlas_data = atlas_img.get_fdata()
    else:
        atlas_data = atlas_img.get_fdata()

    atlas_data = np.rint(atlas_data).astype(np.int32)
    unique_labels = np.unique(atlas_data)

    roi_results = {}

    for label in unique_labels:
        if label == 0:
            continue

        roi_mask = atlas_data == label
        roi_values = wasserstein_result[roi_mask]

        if roi_values.size == 0:
            continue

        finite_values = roi_values[np.isfinite(roi_values)]

        if finite_values.size == 0:
            continue

        mean_wasserstein = float(np.mean(finite_values))
        roi_results[str(int(label))] = mean_wasserstein

    return roi_results


def get_existing_subjects(base_dir, start=102, end=129):
    subjects = []

    for subject_index in range(start, end + 1):
        subject = f"sub-s{subject_index}"
        subject_dir = os.path.join(base_dir, subject)

        if os.path.exists(subject_dir):
            subjects.append(subject)

    return subjects


def get_fmri_file_path(base_dir, subject, week, video):
    return os.path.join(
        base_dir,
        subject,
        f"ses-wk{week}",
        "func",
        f"{subject}_ses-wk{week}_task-vid{video}_bold.nii.gz"
    )


def get_region_name_from_index(atlas_labels, index):
    try:
        return atlas_labels[int(index)]
    except Exception:
        try:
            return atlas_labels[int(index) - 1]
        except Exception:
            return f"Region_{index}"


def compute_weekly_group_csv(base_dir, output_root):
    atlas = fetch_atlas_harvard_oxford("cort-maxprob-thr0-2mm")
    atlas_img = atlas.maps
    atlas_labels = atlas.labels

    subjects = get_existing_subjects(base_dir)

    os.makedirs(output_root, exist_ok=True)

    for week in range(1, 6):
        per_subject_region_means = {}

        for subject in subjects:
            video_paths = {}

            for video in range(1, 6):
                fmri_path = get_fmri_file_path(base_dir, subject, week, video)

                if os.path.exists(fmri_path):
                    video_paths[video] = fmri_path

            if len(video_paths) < 2:
                continue

            reference_img = nib.load(list(video_paths.values())[0])
            data_shape = reference_img.shape

            coords = [
                (x, y, z)
                for x in range(0, data_shape[0] - 3, 4)
                for y in range(0, data_shape[1] - 3, 4)
                for z in range(0, data_shape[2] - 3, 4)
            ]

            pair_roi_values = {}
            pair_count = 0

            for (video_1, path_1), (video_2, path_2) in combinations(
                video_paths.items(),
                2
            ):
                try:
                    data_1 = nib.load(path_1).get_fdata()
                    data_2 = nib.load(path_2).get_fdata()
                except Exception:
                    continue

                if (
                    data_1.ndim < 4
                    or data_2.ndim < 4
                    or data_1.shape[3] < 2
                    or data_2.shape[3] < 2
                ):
                    continue

                data_1 = data_1[:, :, :, 1:] - data_1[:, :, :, :-1]
                data_2 = data_2[:, :, :, 1:] - data_2[:, :, :, :-1]

                cache_dir = os.path.join(output_root, "cache")
                os.makedirs(cache_dir, exist_ok=True)

                cache_name = (
                    f"{subject}_week{week}_video{video_1}_video{video_2}_"
                    f"wasserstein.npy"
                )
                cache_path = os.path.join(cache_dir, cache_name)

                if os.path.exists(cache_path):
                    wasserstein_map = np.load(cache_path)
                else:
                    wasserstein_map = calculate_wasserstein_distance(
                        data_1,
                        data_2,
                        coords,
                        cache_path
                    )

                roi_wasserstein = calculate_roi_wasserstein(
                    wasserstein_map,
                    atlas_img,
                    atlas_labels,
                    reference_img
                )

                pair_count += 1

                for region_index, value in roi_wasserstein.items():
                    pair_roi_values.setdefault(region_index, []).append(value)

            if pair_count == 0:
                continue

            subject_region_mean = {}

            for region_index, values in pair_roi_values.items():
                subject_region_mean[region_index] = float(np.mean(values))

            per_subject_region_means[subject] = subject_region_mean

        region_values = {}

        for subject, region_dict in per_subject_region_means.items():
            for region_index, value in region_dict.items():
                region_values.setdefault(region_index, []).append(value)

        region_group_mean = {}

        for region_index, values in region_values.items():
            region_group_mean[region_index] = float(np.mean(values))

        rows = []

        for region_index, mean_value in sorted(
            region_group_mean.items(),
            key=lambda item: int(float(item[0]))
        ):
            try:
                index = int(float(region_index))
            except Exception:
                index = region_index

            region_name = get_region_name_from_index(atlas_labels, index)

            rows.append(
                {
                    "region": region_name,
                    "mean_wasserstein": mean_value
                }
            )

        df = pd.DataFrame(rows, columns=["region", "mean_wasserstein"])

        output_csv = os.path.join(
            output_root,
            f"roi_wasserstein_week{week}.csv"
        )

        df.to_csv(output_csv, index=False, encoding="utf-8")


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    base_dir = r"E:\ThinkLikeExperts"
    output_root = os.path.join(script_dir, "wasserstein_weekly")

    compute_weekly_group_csv(base_dir, output_root)


if __name__ == "__main__":
    main()