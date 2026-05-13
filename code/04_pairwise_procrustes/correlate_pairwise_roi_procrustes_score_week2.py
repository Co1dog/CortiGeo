"""
This script computes ROI-level correlations between pairwise Procrustes neural
similarity and learning performance for Week 2 fMRI data. For each ROI, it loads
ROI-level fMRI time-series data from all available videos, standardizes each
voxel time series, computes temporal-difference activation patterns, measures
pairwise subject similarity using Procrustes disparity, averages similarity
values across videos for each subject pair, and correlates the pairwise neural
similarity with the summed exam scores of the two subjects. The final output is
a CSV file containing one correlation coefficient and p-value for each ROI.
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from scipy.spatial import procrustes


BASE_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEOS = range(1, 7)
EPSILON = 1e-12
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_CSV = f"correlations_procrustes_week{WEEK_ID}_all_rois.csv"


def standardize_time_series(data):
    data = np.asarray(data, dtype=float)
    normalized_data = np.zeros_like(data)

    scaler = StandardScaler()

    for voxel_index in range(data.shape[1]):
        column = data[:, voxel_index]

        if np.isfinite(column).all() and np.std(column) > 1e-10:
            normalized_data[:, voxel_index] = scaler.fit_transform(
                column.reshape(-1, 1)
            ).ravel()

    return normalized_data


def get_roi_file_path(base_dir, subject_id, week_id, video_id, region):
    return os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{region}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


def get_region_names(base_dir, subject_id, week_id):
    example_regions_path = os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(example_regions_path):
        raise FileNotFoundError(
            f"[Week {week_id}] ROI directory not found: {example_regions_path}"
        )

    region_files = [
        file_name
        for file_name in os.listdir(example_regions_path)
        if file_name.endswith(".npy")
    ]

    return sorted(
        {
            file_name.split("_sub-")[0]
            for file_name in region_files
        }
    )


score_df = pd.read_csv(SCORE_CSV)

scores = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values

regions = get_region_names(
    BASE_DIR,
    subject_ids[0],
    WEEK_ID
)

results = []

print(f"Total ROIs: {len(regions)}, total subjects: {len(subject_ids)}")

for region_index, region in enumerate(regions, start=1):
    print(f"[{region_index}/{len(regions)}] Processing ROI: {region}")

    pair_similarities = []
    pair_scores = []

    n_subjects = len(subject_ids)

    for i in range(n_subjects):
        subject_i = subject_ids[i]

        for j in range(i + 1, n_subjects):
            subject_j = subject_ids[j]
            video_similarities = []

            for video_id in VIDEOS:
                path_i = get_roi_file_path(
                    BASE_DIR,
                    subject_i,
                    WEEK_ID,
                    video_id,
                    region
                )

                path_j = get_roi_file_path(
                    BASE_DIR,
                    subject_j,
                    WEEK_ID,
                    video_id,
                    region
                )

                if not (os.path.exists(path_i) and os.path.exists(path_j)):
                    continue

                try:
                    data_i = np.load(path_i)
                    data_j = np.load(path_j)

                    if data_i.ndim != 2 or data_j.ndim != 2:
                        continue

                    if not np.isfinite(data_i).all() or not np.isfinite(data_j).all():
                        continue

                    if (
                        data_i.shape[0] < 3
                        or data_j.shape[0] < 3
                        or data_i.shape[1] < 2
                        or data_j.shape[1] < 2
                    ):
                        continue

                    normalized_i = standardize_time_series(data_i)
                    normalized_j = standardize_time_series(data_j)

                    diff_i = normalized_i[1:] - normalized_i[:-1]
                    diff_j = normalized_j[1:] - normalized_j[:-1]

                    if diff_i.shape != diff_j.shape:
                        continue

                    _, _, disparity = procrustes(diff_i.T, diff_j.T)
                    similarity = 1.0 / (disparity + EPSILON)

                    if np.isfinite(similarity):
                        video_similarities.append(similarity)

                except Exception:
                    continue

            if video_similarities:
                pair_similarities.append(float(np.mean(video_similarities)))
                pair_scores.append(float(scores[i] + scores[j]))

    r_value = np.nan
    p_value = np.nan

    if len(pair_similarities) >= 3:
        x = np.asarray(pair_similarities, dtype=float)
        y = np.asarray(pair_scores, dtype=float)

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if x.size >= 3:
            r_value, p_value = pearsonr(x, y)

    results.append(
        {
            "region": f"region_{region}",
            "correlation": None if not np.isfinite(r_value) else float(r_value),
            "p_value": None if not np.isfinite(p_value) else float(p_value)
        }
    )

pd.DataFrame(
    results,
    columns=["region", "correlation", "p_value"]
).to_csv(OUTPUT_CSV, index=False)

print(f"[Week {WEEK_ID}] Results saved to: {OUTPUT_CSV}")