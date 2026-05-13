"""
This script computes ROI-level correlations between within-subject video-to-video
Wasserstein distance and learning performance for one fMRI recording week. For
each ROI and subject, it loads all available video-specific ROI time-series
files, computes temporal-difference activation patterns, measures Wasserstein
distances between every pair of videos from the same subject, and correlates the
resulting video-pair distances with the subject's exam score. The final output
is a CSV file containing the mean Wasserstein distance and correlation statistics
for each ROI.
"""

import os
from itertools import combinations

import numpy as np
import pandas as pd
import ot

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from tqdm import tqdm


ROOT_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_DIR = "wasserstein_rois_week1_video_pairs_with_scores"

WEEK_ID = 1
VIDEOS = [1, 2, 3, 4, 5]

os.makedirs(OUTPUT_DIR, exist_ok=True)


def wasserstein_distance(data_1, data_2):
    combined_data = np.vstack([data_1, data_2])
    combined_data = StandardScaler().fit_transform(combined_data)

    normalized_1 = combined_data[:len(data_1)]
    normalized_2 = combined_data[len(data_1):]

    weights_1 = np.ones(len(normalized_1)) / len(normalized_1)
    weights_2 = np.ones(len(normalized_2)) / len(normalized_2)

    cost_matrix = ot.dist(normalized_1, normalized_2)
    return float(ot.emd2(weights_1, weights_2, cost_matrix))


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


def get_region_names(root_dir, week_id):
    subjects = [
        subject
        for subject in os.listdir(root_dir)
        if subject.startswith("sub-")
    ]

    if not subjects:
        raise RuntimeError(f"No sub-* directories were found under {root_dir}")

    first_subject = subjects[0]
    roi_dir = os.path.join(
        root_dir,
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

roi_names = get_region_names(ROOT_DIR, WEEK_ID)

print(f"Found {len(roi_names)} ROIs")

results = []

for roi_name in tqdm(roi_names, desc="Computing ROI Wasserstein distances"):
    wasserstein_values = []
    score_values = []

    for subject_id in subject_ids:
        video_paths = {
            video_id: get_roi_file_path(
                ROOT_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )
            for video_id in VIDEOS
        }

        existing_videos = [
            video_id
            for video_id, path in video_paths.items()
            if os.path.exists(path)
        ]

        if len(existing_videos) < 2:
            continue

        for video_1, video_2 in combinations(existing_videos, 2):
            path_1 = video_paths[video_1]
            path_2 = video_paths[video_2]

            try:
                voxel_data_1 = np.load(path_1)
                voxel_data_2 = np.load(path_2)

                if voxel_data_1.shape[0] < 3 or voxel_data_2.shape[0] < 3:
                    continue

                diff_1 = voxel_data_1[1:] - voxel_data_1[:-1]
                diff_2 = voxel_data_2[1:] - voxel_data_2[:-1]

                distance = wasserstein_distance(diff_1, diff_2)

                if np.isfinite(distance):
                    wasserstein_values.append(distance)
                    score_values.append(score_map[subject_id])

            except Exception as error:
                print(
                    f"[Week {WEEK_ID}] Error in ROI {roi_name}, "
                    f"sub-{subject_id}, videos {video_1}-{video_2}: {error}"
                )
                continue

    if len(wasserstein_values) >= 3:
        wasserstein_array = np.asarray(wasserstein_values, dtype=float)
        score_array = np.asarray(score_values, dtype=float)

        valid_mask = np.isfinite(wasserstein_array) & np.isfinite(score_array)
        wasserstein_array = wasserstein_array[valid_mask]
        score_array = score_array[valid_mask]

        if wasserstein_array.size >= 3:
            mean_wasserstein = float(np.mean(wasserstein_array))
            r_value, p_value = pearsonr(wasserstein_array, score_array)
        else:
            mean_wasserstein, r_value, p_value = np.nan, np.nan, np.nan
    else:
        mean_wasserstein, r_value, p_value = np.nan, np.nan, np.nan

    results.append(
        {
            "region": roi_name,
            "mean_wasserstein": round(mean_wasserstein, 5) if np.isfinite(mean_wasserstein) else None,
            "r": round(r_value, 4) if np.isfinite(r_value) else None,
            "p": f"{p_value:.4g}" if np.isfinite(p_value) else None
        }
    )

result_df = pd.DataFrame(
    results,
    columns=["region", "mean_wasserstein", "r", "p"]
)

output_csv = os.path.join(
    OUTPUT_DIR,
    f"roi_wasserstein_correlations_week{WEEK_ID}_video_pairs.csv"
)

result_df.to_csv(output_csv, index=False)

print(f"Results saved to: {output_csv}")