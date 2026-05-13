"""
This script performs a multi-week ROI-level analysis of intrinsic dimension,
within-subject video-to-video Wasserstein distance, and their relationships with
learning performance. For each week and ROI, it computes temporal-difference
intrinsic dimension from subject-video fMRI data, computes Wasserstein distances
between video pairs from the same subject, correlates intrinsic dimension with
exam score, correlates Wasserstein distance with exam score, and estimates an
exploratory correlation between intrinsic dimension and log-transformed
Wasserstein distance. The script supports both regular video tasks and Week 6
recap tasks, and saves three CSV result files for each week.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import ot
import skdim

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm


BASE_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
WEEKS = range(1, 7)
N_NEIGHBORS = 100
EPSILON_LOG = 1e-8

VIDEOS_BY_WEEK = {
    1: [str(index) for index in range(1, 6)],
    2: [str(index) for index in range(1, 6)],
    3: [str(index) for index in range(1, 6)],
    4: [str(index) for index in range(1, 6)],
    5: [str(index) for index in range(1, 6)],
    6: ["wk1recap", "wk2recap", "wk3recap", "wk4recap", "wk5recap"]
}


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


def standardize_time_series(voxel_data):
    voxel_data = np.asarray(voxel_data, dtype=float)
    normalized_data = np.zeros_like(voxel_data)

    scaler = StandardScaler()

    for voxel_index in range(voxel_data.shape[1]):
        column = voxel_data[:, voxel_index]

        if np.isfinite(column).all() and np.std(column) > 1e-10:
            normalized_data[:, voxel_index] = scaler.fit_transform(
                column.reshape(-1, 1)
            ).ravel()
        else:
            normalized_data[:, voxel_index] = 0.0

    return normalized_data


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

    raise RuntimeError(
        f"No ROI directory was found under sub-*/ses-wk{week_id}/func/regions"
    )


def get_candidate_file_path(roi_name, subject_id, week_id, video_label):
    subject_dir = os.path.join(
        BASE_DIR,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(subject_dir):
        return None

    candidate_names = [
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_label}_bold.npy",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-{video_label}_bold.npy",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_label}.npy",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-{video_label}.npy",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_{video_label}.npy"
    ]

    for file_name in candidate_names:
        file_path = os.path.join(subject_dir, file_name)

        if os.path.exists(file_path):
            return file_path

    return None


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
    print(f"\n=== Processing Week {week_id} ===")

    output_dir = f"wasserstein_rois_week{week_id}_video_with_scores_log"
    os.makedirs(output_dir, exist_ok=True)

    try:
        roi_names = get_region_names(BASE_DIR, week_id)
    except RuntimeError as error:
        print(error)
        continue

    print(f"[Week {week_id}] Found {len(roi_names)} ROIs")

    dimension_results = []
    wasserstein_results = []
    dimension_wasserstein_results = []

    video_labels = VIDEOS_BY_WEEK.get(
        week_id,
        [str(index) for index in range(1, 6)]
    )

    for roi_name in tqdm(roi_names, desc=f"[Week {week_id}]"):
        all_dimensions = []
        all_dimension_scores = []

        all_wasserstein = []
        all_wasserstein_scores = []

        for subject_id in subject_ids:
            for video_label in video_labels:
                file_path = get_candidate_file_path(
                    roi_name,
                    subject_id,
                    week_id,
                    video_label
                )

                if file_path is None:
                    continue

                try:
                    voxel_data = np.load(file_path)

                    if voxel_data.ndim != 2 or voxel_data.shape[0] < 5 or voxel_data.shape[1] < 2:
                        continue

                    normalized_data = standardize_time_series(voxel_data)
                    temporal_diff = normalized_data[1:] - normalized_data[:-1]

                    try:
                        dimension = skdim.id.MLE().fit_transform(
                            temporal_diff,
                            n_neighbors=N_NEIGHBORS
                        )
                    except Exception:
                        dimension = skdim.id.MLE().fit_transform(temporal_diff)

                    dimension = float(np.asarray(dimension).mean())

                    if np.isfinite(dimension):
                        all_dimensions.append(dimension)
                        all_dimension_scores.append(score_map.get(subject_id, np.nan))

                except Exception:
                    continue

        for subject_id in subject_ids:
            video_paths = {}

            for video_label in video_labels:
                file_path = get_candidate_file_path(
                    roi_name,
                    subject_id,
                    week_id,
                    video_label
                )

                if file_path is not None:
                    video_paths[video_label] = file_path

            if len(video_paths) < 2:
                continue

            for video_1, video_2 in combinations(video_paths.keys(), 2):
                try:
                    voxel_data_1 = np.load(video_paths[video_1])
                    voxel_data_2 = np.load(video_paths[video_2])

                    if voxel_data_1.shape[0] < 3 or voxel_data_2.shape[0] < 3:
                        continue

                    normalized_1 = standardize_time_series(voxel_data_1)
                    normalized_2 = standardize_time_series(voxel_data_2)

                    diff_1 = normalized_1[1:] - normalized_1[:-1]
                    diff_2 = normalized_2[1:] - normalized_2[:-1]

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
            dimensions_for_correlation = np.asarray(all_dimensions[:n_pairs], dtype=float)
            wasserstein_for_correlation = np.asarray(all_wasserstein[:n_pairs], dtype=float)
            log_wasserstein = np.log(wasserstein_for_correlation + EPSILON_LOG)

            r_dimension_wasserstein, p_dimension_wasserstein = safe_pearsonr(
                dimensions_for_correlation,
                log_wasserstein
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
        os.path.join(output_dir, f"roi_wasserstein_dimension_log_correlations_week{week_id}.csv"),
        index=False
    )

    print(f"[Week {week_id}] Saved three result files to {output_dir}")

print("\nAll done.")