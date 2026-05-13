"""
This script computes ROI-level lag-1 autocorrelation features from fMRI
time-series data for one subject, week, and video. For each ROI, it loads the
ROI-level voxel time-series matrix, applies feature-wise standardization,
computes voxel-wise lag-1 Pearson autocorrelation, averages voxel-level
correlations using Fisher-z transformation, and performs a one-sample t-test
against zero on the Fisher-z values. The final output is a CSV file containing
ROI-level autocorrelation estimates and p-values.
"""

import os
import numpy as np
import pandas as pd

from scipy.stats import pearsonr, ttest_1samp
from sklearn.preprocessing import StandardScaler


BASE_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEO_ID = 1
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_CSV = f"correlations_lag1_week{WEEK_ID}_video{VIDEO_ID}.csv"


def fisher_z(correlation, epsilon=1e-12):
    correlation = np.clip(correlation, -1 + epsilon, 1 - epsilon)
    return 0.5 * np.log((1 + correlation) / (1 - correlation))


def inverse_fisher_z(z_value):
    return np.tanh(z_value)


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
        raise FileNotFoundError(f"ROI directory not found: {example_regions_path}")

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

subject_ids = score_df["participant_id"].astype(str).values
target_subject = subject_ids[0]

print(f"Using subject: sub-{target_subject}, Week {WEEK_ID}, Video {VIDEO_ID}")

regions = get_region_names(
    BASE_DIR,
    target_subject,
    WEEK_ID
)

results = []

for region_index, region in enumerate(regions, start=1):
    print(f"[{region_index}/{len(regions)}] ROI: {region}")

    roi_path = get_roi_file_path(
        BASE_DIR,
        target_subject,
        WEEK_ID,
        VIDEO_ID,
        region
    )

    if not os.path.exists(roi_path):
        continue

    try:
        voxel_data = np.load(roi_path)

        if voxel_data.ndim != 2 or voxel_data.shape[0] < 3 or voxel_data.shape[1] < 1:
            continue

        if not np.isfinite(voxel_data).all():
            continue

        normalized_data = standardize_time_series(voxel_data)

        voxel_correlations = []

        for voxel_index in range(normalized_data.shape[1]):
            time_series = normalized_data[:, voxel_index]

            if not np.isfinite(time_series).all():
                continue

            if np.std(time_series[:-1]) <= 1e-12 or np.std(time_series[1:]) <= 1e-12:
                continue

            correlation, _ = pearsonr(time_series[:-1], time_series[1:])

            if np.isfinite(correlation):
                voxel_correlations.append(correlation)

        if not voxel_correlations:
            continue

        z_values = np.asarray(
            [
                fisher_z(correlation)
                for correlation in voxel_correlations
            ],
            dtype=float
        )

        z_mean = float(np.mean(z_values))
        roi_correlation = float(inverse_fisher_z(z_mean))

        _, p_value = ttest_1samp(
            z_values,
            popmean=0.0,
            alternative="two-sided"
        )

        roi_p_value = float(p_value) if np.isfinite(p_value) else np.nan

        results.append(
            {
                "region": f"region_{region}",
                "correlation": roi_correlation,
                "p_value": roi_p_value
            }
        )

    except Exception as error:
        print(f"Error while processing {region}: {error}")
        continue

result_df = pd.DataFrame(
    results,
    columns=["region", "correlation", "p_value"]
)

result_df.to_csv(OUTPUT_CSV, index=False)

print(f"Done. Results saved to: {OUTPUT_CSV}. Total ROIs: {len(result_df)}")