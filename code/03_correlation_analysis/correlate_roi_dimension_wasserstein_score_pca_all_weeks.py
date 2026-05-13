"""
This script performs a PCA-controlled multi-week ROI-level analysis of intrinsic
dimension, within-subject video-to-video Wasserstein distance, and their
relationships with learning performance. For each week and ROI, it loads
subject-video fMRI ROI time-series data, standardizes voxel activity, computes
temporal-difference signals, applies PCA to harmonize feature dimensionality,
estimates intrinsic dimension using the MLE estimator, and computes Wasserstein
distances between videos from the same subject. The script then saves ROI-level
dimension-score, Wasserstein-score, and dimension-Wasserstein correlation
summaries for each week.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import ot
import skdim

from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


BASE_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_ROOT = "wasserstein_rois_adaptive_mixed"
WEEKS = range(1, 6)
VIDEOS = range(1, 6)
N_NEIGHBORS = 100

os.makedirs(OUTPUT_ROOT, exist_ok=True)

score_df = pd.read_csv(SCORE_CSV)
scores = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values
score_map = dict(zip(subject_ids, scores))


def wasserstein_distance(data_1, data_2):
    combined_data = np.vstack([data_1, data_2])
    combined_data = StandardScaler().fit_transform(combined_data)

    normalized_1 = combined_data[:len(data_1)]
    normalized_2 = combined_data[len(data_1):]

    weights_1 = np.ones(len(normalized_1)) / len(normalized_1)
    weights_2 = np.ones(len(normalized_2)) / len(normalized_2)

    cost_matrix = ot.dist(normalized_1, normalized_2)
    return float(ot.emd2(weights_1, weights_2, cost_matrix))


def get_roi_file_path(base_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


def get_region_names(base_dir, week_id):
    subjects = [
        subject
        for subject in os.listdir(base_dir)
        if subject.startswith("sub-")
    ]

    if not subjects:
        raise RuntimeError(f"No sub-* directories were found under {base_dir}")

    for subject in subjects:
        roi_dir = os.path.join(
            base_dir,
            subject,
            f"ses-wk{week_id}",
            "func",
            "regions"
        )

        if os.path.isdir(roi_dir):
            return sorted(
                {
                    file_name.split("_sub-")[0]
                    for file_name in os.listdir(roi_dir)
                    if file_name.endswith(".npy")
                }
            )

    return []


def collect_shape_statistics(base_dir, week_id, subject_ids):
    voxel_counts = []
    time_counts = []

    for subject_id in subject_ids:
        subject_dir = os.path.join(
            base_dir,
            f"sub-{subject_id}",
            f"ses-wk{week_id}",
            "func",
            "regions"
        )

        if not os.path.exists(subject_dir):
            continue

        for file_name in os.listdir(subject_dir):
            if not file_name.endswith(".npy"):
                continue

            array = np.load(os.path.join(subject_dir, file_name))

            if array.ndim == 2:
                voxel_counts.append(array.shape[1])
                time_counts.append(array.shape[0])

    return voxel_counts, time_counts


def safe_pearsonr(x_values, y_values):
    x_values = np.asarray(x_values, dtype=float)
    y_values = np.asarray(y_values, dtype=float)

    valid_mask = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid_mask]
    y_values = y_values[valid_mask]

    if x_values.size < 3 or y_values.size < 3:
        return np.nan, np.nan

    try:
        return pearsonr(x_values, y_values)
    except Exception:
        return np.nan, np.nan


for week_id in WEEKS:
    output_dir = os.path.join(OUTPUT_ROOT, f"week{week_id}")
    os.makedirs(output_dir, exist_ok=True)

    voxel_counts, time_counts = collect_shape_statistics(
        BASE_DIR,
        week_id,
        subject_ids
    )

    if not voxel_counts:
        print(f"[Week {week_id}] No valid ROI arrays found. Skipping.")
        continue

    n_components_auto = int(np.min(voxel_counts))
    time_steps_auto = int(np.min(time_counts))

    roi_names = get_region_names(BASE_DIR, week_id)

    if not roi_names:
        print(f"[Week {week_id}] No ROI names found. Skipping.")
        continue

    dimension_results = []
    wasserstein_results = []
    dimension_wasserstein_results = []

    for roi_name in tqdm(roi_names, desc=f"[Week {week_id}]"):
        all_dimensions = []
        all_dimension_scores = []
        all_wasserstein = []
        all_wasserstein_scores = []

        for subject_id in subject_ids:
            for video_id in VIDEOS:
                file_path = get_roi_file_path(
                    BASE_DIR,
                    subject_id,
                    week_id,
                    video_id,
                    roi_name
                )

                if not os.path.exists(file_path):
                    continue

                try:
                    voxel_data = np.load(file_path)

                    if (
                        voxel_data.ndim != 2
                        or voxel_data.shape[0] < 5
                        or voxel_data.shape[1] < 2
                    ):
                        continue

                    voxel_norm = StandardScaler().fit_transform(voxel_data)
                    voxel_diff = voxel_norm[1:] - voxel_norm[:-1]

                    n_components = min(
                        n_components_auto,
                        voxel_diff.shape[0],
                        voxel_diff.shape[1]
                    )

                    if n_components < 1:
                        continue

                    voxel_diff_pca = PCA(n_components=n_components).fit_transform(
                        voxel_diff
                    )

                    time_steps = min(time_steps_auto, voxel_diff_pca.shape[0])
                    voxel_diff_pca = voxel_diff_pca[:time_steps]

                    if voxel_diff_pca.shape[0] < 3:
                        continue

                    n_neighbors = max(
                        2,
                        min(N_NEIGHBORS, voxel_diff_pca.shape[0] - 1)
                    )

                    dimension = skdim.id.MLE().fit_transform(
                        voxel_diff_pca,
                        n_neighbors=n_neighbors
                    )
                    dimension = float(np.asarray(dimension).mean())

                    if np.isfinite(dimension):
                        all_dimensions.append(dimension)
                        all_dimension_scores.append(score_map.get(subject_id, np.nan))

                except Exception:
                    continue

        for subject_id in subject_ids:
            subject_dir = os.path.join(
                BASE_DIR,
                f"sub-{subject_id}",
                f"ses-wk{week_id}",
                "func",
                "regions"
            )

            video_files = {
                video_id: os.path.join(
                    subject_dir,
                    f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
                )
                for video_id in VIDEOS
            }

            existing_videos = [
                video_id
                for video_id, path in video_files.items()
                if os.path.exists(path)
            ]

            if len(existing_videos) < 2:
                continue

            for video_1, video_2 in combinations(existing_videos, 2):
                try:
                    voxel_data_1 = np.load(video_files[video_1])
                    voxel_data_2 = np.load(video_files[video_2])

                    if voxel_data_1.shape[0] < 3 or voxel_data_2.shape[0] < 3:
                        continue

                    combined_raw = np.vstack([voxel_data_1, voxel_data_2])
                    combined_scaled = StandardScaler().fit_transform(combined_raw)

                    n_components = min(
                        n_components_auto,
                        combined_scaled.shape[0],
                        combined_scaled.shape[1]
                    )

                    if n_components < 1:
                        continue

                    combined_pca = PCA(n_components=n_components).fit_transform(
                        combined_scaled
                    )

                    split_index = voxel_data_1.shape[0]

                    data_1 = combined_pca[:split_index]
                    data_2 = combined_pca[split_index:]

                    diff_1 = data_1[1:] - data_1[:-1]
                    diff_2 = data_2[1:] - data_2[:-1]

                    time_steps = min(
                        time_steps_auto,
                        diff_1.shape[0],
                        diff_2.shape[0]
                    )

                    diff_1 = diff_1[:time_steps]
                    diff_2 = diff_2[:time_steps]

                    if diff_1.shape[0] < 3 or diff_2.shape[0] < 3:
                        continue

                    distance = wasserstein_distance(diff_1, diff_2)

                    if np.isfinite(distance):
                        all_wasserstein.append(distance)
                        all_wasserstein_scores.append(score_map.get(subject_id, np.nan))

                except Exception:
                    continue

        if len(all_dimensions) >= 3:
            mean_dimension = float(np.nanmean(all_dimensions))
            r_dimension, p_dimension = safe_pearsonr(
                all_dimensions,
                all_dimension_scores
            )
        else:
            mean_dimension, r_dimension, p_dimension = np.nan, np.nan, np.nan

        if len(all_wasserstein) >= 3:
            mean_wasserstein = float(np.nanmean(all_wasserstein))
            r_wasserstein, p_wasserstein = safe_pearsonr(
                all_wasserstein,
                all_wasserstein_scores
            )
        else:
            mean_wasserstein, r_wasserstein, p_wasserstein = np.nan, np.nan, np.nan

        if len(all_dimensions) >= 3 and len(all_wasserstein) >= 3:
            n_pairs = min(len(all_dimensions), len(all_wasserstein))
            r_dimension_wasserstein, p_dimension_wasserstein = safe_pearsonr(
                np.asarray(all_dimensions[:n_pairs], dtype=float),
                np.asarray(all_wasserstein[:n_pairs], dtype=float)
            )
        else:
            r_dimension_wasserstein, p_dimension_wasserstein = np.nan, np.nan

        dimension_results.append(
            {
                "region": roi_name,
                "mean_dimension": mean_dimension,
                "r": r_dimension,
                "p": p_dimension
            }
        )

        wasserstein_results.append(
            {
                "region": roi_name,
                "mean_wasserstein": mean_wasserstein,
                "r": r_wasserstein,
                "p": p_wasserstein
            }
        )

        dimension_wasserstein_results.append(
            {
                "region": roi_name,
                "r": r_dimension_wasserstein,
                "p": p_dimension_wasserstein
            }
        )

    pd.DataFrame(dimension_results).to_csv(
        os.path.join(output_dir, f"roi_dimension_correlations_week{week_id}.csv"),
        index=False
    )

    pd.DataFrame(wasserstein_results).to_csv(
        os.path.join(output_dir, f"roi_wasserstein_correlations_week{week_id}.csv"),
        index=False
    )

    pd.DataFrame(dimension_wasserstein_results).to_csv(
        os.path.join(output_dir, f"roi_wasserstein_dimension_correlations_week{week_id}.csv"),
        index=False
    )

print("Done.")