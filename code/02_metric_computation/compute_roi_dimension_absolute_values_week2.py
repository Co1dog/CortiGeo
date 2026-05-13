"""
This script computes absolute ROI-level intrinsic dimension values for Week 2
fMRI data. For each Harvard-Oxford ROI, subject, and video, it loads the
ROI-level fMRI time series, applies feature-wise standardization, estimates
intrinsic dimension on both the standardized raw signal and the temporal-
difference signal using the MLE estimator, averages valid estimates across
subjects and videos, and saves separate CSV files for raw-signal and
difference-signal intrinsic dimension values.
"""

import os
import numpy as np
import pandas as pd
import skdim

from sklearn.preprocessing import StandardScaler


OUTPUT_DIR = "brain_region_dimension_absolute_values"
BASE_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
WEEK_ID = 2
VIDEOS = [1, 2, 3, 4, 5]
N_NEIGHBORS = 10

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


def estimate_dimension(data, n_neighbors):
    if data.shape[0] < n_neighbors + 1:
        return None

    if not np.isfinite(data).all():
        return None

    try:
        dimension = skdim.id.MLE().fit_transform(
            data,
            n_neighbors=n_neighbors
        )
        dimension = float(np.asarray(dimension).mean())

        if np.isfinite(dimension):
            return dimension

    except Exception:
        return None

    return None


def get_roi_file_path(base_dir, subject_id, week_id, video_id, region):
    return os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{region}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


score_df = pd.read_csv(SCORE_CSV)
subject_ids = score_df["participant_id"].astype(str).values

print(subject_ids)

example_subject = f"sub-{subject_ids[0]}"
example_regions_path = os.path.join(
    BASE_DIR,
    example_subject,
    f"ses-wk{WEEK_ID}",
    "func",
    "regions"
)

if not os.path.isdir(example_regions_path):
    print(f"[Week {WEEK_ID}] Warning: region directory not found: {example_regions_path}")
    raise SystemExit(0)

region_files = [
    file_name
    for file_name in os.listdir(example_regions_path)
    if file_name.endswith(".npy")
]

regions = sorted(
    {
        file_name.split("_sub-")[0]
        for file_name in region_files
    }
)

raw_results = []
diff_results = []

for region_index, region in enumerate(regions, start=1):
    print(f"[Week {WEEK_ID}] Computing region {region_index}/{len(regions)}: {region}")

    raw_dimensions = []
    diff_dimensions = []

    for video_id in VIDEOS:
        for subject_id in subject_ids:
            file_path = get_roi_file_path(
                BASE_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                region
            )

            if not os.path.exists(file_path):
                continue

            try:
                voxel_data = np.load(file_path)

                if voxel_data.ndim != 2 or voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                    continue

                if not np.isfinite(voxel_data).all():
                    continue

                normalized_data = standardize_time_series(voxel_data)

                raw_dimension = estimate_dimension(
                    normalized_data,
                    N_NEIGHBORS
                )

                if raw_dimension is not None:
                    raw_dimensions.append(raw_dimension)

                temporal_diff = normalized_data[1:] - normalized_data[:-1]

                diff_dimension = estimate_dimension(
                    temporal_diff,
                    N_NEIGHBORS
                )

                if diff_dimension is not None:
                    diff_dimensions.append(diff_dimension)

            except Exception:
                continue

    if raw_dimensions:
        raw_results.append(
            {
                "region": f"region_{region}",
                "dimension": float(np.mean(raw_dimensions))
            }
        )

    if diff_dimensions:
        diff_results.append(
            {
                "region": f"region_{region}",
                "dimension": float(np.mean(diff_dimensions))
            }
        )

raw_csv = os.path.join(
    OUTPUT_DIR,
    f"dimensions_week{WEEK_ID}_raw.csv"
)

diff_csv = os.path.join(
    OUTPUT_DIR,
    f"dimensions_week{WEEK_ID}_diff.csv"
)

pd.DataFrame(
    raw_results,
    columns=["region", "dimension"]
).to_csv(raw_csv, index=False)

pd.DataFrame(
    diff_results,
    columns=["region", "dimension"]
).to_csv(diff_csv, index=False)

print(f"[Week {WEEK_ID}] Saved raw-signal dimensions to: {raw_csv}")
print(f"[Week {WEEK_ID}] Saved difference-signal dimensions to: {diff_csv}")