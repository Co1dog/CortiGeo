"""
This script computes ROI-level correlations between within-subject multi-video
intrinsic dimension and learning performance for one fMRI recording week. For
each ROI and subject, it loads all available video-specific ROI time-series
files, applies feature-wise standardization, computes temporal-difference
activation patterns, estimates intrinsic dimension with the MLE estimator, and
averages valid dimension estimates across videos. The subject-level mean
dimension values are then correlated with exam scores, and the resulting
ROI-level statistics are saved to a CSV file.
"""

import os
import numpy as np
import pandas as pd
import skdim

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm


BASE_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_DIR = "dimension_rois_week1_video_mean_with_scores"

WEEK_ID = 1
VIDEOS = range(1, 6)
N_NEIGHBORS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)


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

    first_subject = subjects[0]
    roi_dir = os.path.join(
        base_dir,
        first_subject,
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(roi_dir):
        raise RuntimeError(f"ROI directory not found: {roi_dir}")

    return sorted(
        {
            file_name.split("_")[0]
            for file_name in os.listdir(roi_dir)
            if file_name.endswith(".npy")
        }
    )


score_df = pd.read_csv(SCORE_CSV)

scores = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values
score_map = dict(zip(subject_ids, scores))

roi_names = get_region_names(BASE_DIR, WEEK_ID)

print(f"Found {len(roi_names)} ROIs")

results = []

for roi_name in tqdm(roi_names, desc=f"Computing ROI dimensions for Week {WEEK_ID}"):
    dimension_values = []
    score_values = []

    for subject_id in subject_ids:
        dimensions_per_video = []

        for video_id in VIDEOS:
            file_path = get_roi_file_path(
                BASE_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )

            if not os.path.exists(file_path):
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
                    dimensions_per_video.append(dimension)

            except Exception as error:
                print(
                    f"[Week {WEEK_ID}] Error in ROI {roi_name}, "
                    f"sub-{subject_id}, video {video_id}: {error}"
                )
                continue

        if dimensions_per_video:
            dimension_values.append(float(np.mean(dimensions_per_video)))
            score_values.append(score_map[subject_id])

    if len(dimension_values) >= 3:
        dimension_array = np.asarray(dimension_values, dtype=float)
        score_array = np.asarray(score_values, dtype=float)

        valid_mask = np.isfinite(dimension_array) & np.isfinite(score_array)
        dimension_array = dimension_array[valid_mask]
        score_array = score_array[valid_mask]

        if dimension_array.size >= 3:
            mean_dimension = float(np.mean(dimension_array))
            r_value, p_value = pearsonr(dimension_array, score_array)
        else:
            mean_dimension, r_value, p_value = np.nan, np.nan, np.nan
    else:
        mean_dimension, r_value, p_value = np.nan, np.nan, np.nan

    results.append(
        {
            "region": roi_name,
            "mean_dimension": round(mean_dimension, 5) if np.isfinite(mean_dimension) else None,
            "r": round(r_value, 4) if np.isfinite(r_value) else None,
            "p": f"{p_value:.4g}" if np.isfinite(p_value) else None
        }
    )

result_df = pd.DataFrame(
    results,
    columns=["region", "mean_dimension", "r", "p"]
)

output_csv = os.path.join(
    OUTPUT_DIR,
    f"roi_dimension_correlations_week{WEEK_ID}_video_mean.csv"
)

result_df.to_csv(output_csv, index=False)

print(f"Dimension results saved to: {output_csv}")