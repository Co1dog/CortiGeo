"""
This script computes the relationship between intrinsic dimension and H1
persistent-homology lifetime for each ROI across subjects. For a selected week,
it loads ROI-level fMRI time-series data from multiple videos, standardizes each
voxel time series, computes temporal-difference activation patterns, estimates
intrinsic dimension with the MLE estimator, and extracts the mean positive H1
lifetime using persistent homology. For each ROI, subject-level values are
averaged across videos and correlated across subjects. The script saves a
region-wise summary CSV and optional scatter plots with regression lines.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skdim

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from ripser import ripser


ROOT_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEOS = [1, 2, 3, 4, 5]
OUTPUT_DIR = "dimension_lifetime_roi_correlation_week2"
SAVE_PER_REGION_PLOT = True
SCORE_CSV = "subject_weekly_data2.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_standardize_time_series(data):
    data = np.asarray(data, dtype=float)
    normalized_data = np.zeros_like(data, dtype=float)

    scaler = StandardScaler()

    for voxel_index in range(data.shape[1]):
        column = data[:, voxel_index]

        if np.isfinite(column).all() and np.std(column) > 1e-10:
            normalized_data[:, voxel_index] = scaler.fit_transform(
                column.reshape(-1, 1)
            ).ravel()
        else:
            normalized_data[:, voxel_index] = 0.0

    return normalized_data


def compute_dimension(normalized_data):
    if normalized_data.shape[0] < 10 or normalized_data.shape[1] < 2:
        return None

    temporal_diff = normalized_data[1:] - normalized_data[:-1]

    if not np.isfinite(temporal_diff).all():
        return None

    try:
        n_neighbors = max(2, min(10, temporal_diff.shape[0] - 1))
        dimension = skdim.id.MLE().fit_transform(
            temporal_diff,
            n_neighbors=n_neighbors
        )
        dimension = float(np.asarray(dimension).mean())

        if np.isfinite(dimension):
            return dimension

    except Exception:
        return None

    return None


def compute_mean_h1_lifetime(normalized_data):
    if normalized_data.shape[0] < 10 or normalized_data.shape[1] < 2:
        return None

    temporal_diff = normalized_data[1:] - normalized_data[:-1]

    if not np.isfinite(temporal_diff).all():
        return None

    try:
        diagrams = ripser(temporal_diff)["dgms"]

        if len(diagrams) < 2:
            return None

        h1_diagram = diagrams[1]

        if h1_diagram.size == 0:
            return None

        lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
        lifetimes = lifetimes[np.isfinite(lifetimes)]
        lifetimes = lifetimes[lifetimes > 0]

        if lifetimes.size == 0:
            return None

        return float(np.mean(lifetimes))

    except Exception:
        return None


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


def get_roi_names(root_dir, subject_id, week_id):
    roi_dir = os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(roi_dir):
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")

    return sorted(
        {
            file_name.split("_sub-")[0]
            for file_name in os.listdir(roi_dir)
            if file_name.endswith(".npy")
        }
    )


score_df = pd.read_csv(SCORE_CSV)
subject_ids = score_df["participant_id"].astype(str).values

first_subject = subject_ids[0]
roi_names = get_roi_names(ROOT_DIR, first_subject, WEEK_ID)

records = []

for region_index, roi_name in enumerate(roi_names, start=1):
    subject_dimensions = []
    subject_lifetimes = []

    for subject_id in subject_ids:
        video_dimensions = []
        video_lifetimes = []

        for video_id in VIDEOS:
            file_path = get_roi_file_path(
                ROOT_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )

            if not os.path.exists(file_path):
                continue

            try:
                voxel_data = np.load(file_path)

                if voxel_data.ndim != 2 or voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                    continue

                if not np.isfinite(voxel_data).all():
                    continue

                normalized_data = safe_standardize_time_series(voxel_data)

                dimension_value = compute_dimension(normalized_data)
                lifetime_value = compute_mean_h1_lifetime(normalized_data)

                if dimension_value is not None:
                    video_dimensions.append(dimension_value)

                if lifetime_value is not None:
                    video_lifetimes.append(lifetime_value)

            except Exception:
                continue

        if video_dimensions and video_lifetimes:
            subject_dimensions.append(float(np.mean(video_dimensions)))
            subject_lifetimes.append(float(np.mean(video_lifetimes)))

    if len(subject_dimensions) >= 3 and len(subject_lifetimes) >= 3:
        x = np.asarray(subject_dimensions, dtype=float)
        y = np.asarray(subject_lifetimes, dtype=float)

        valid_mask = np.isfinite(x) & np.isfinite(y)
        x = x[valid_mask]
        y = y[valid_mask]

        if x.size < 3 or y.size < 3:
            print(
                f"[Skipped] ROI {roi_name}: too few valid subjects "
                f"(dimension={x.size}, lifetime={y.size})"
            )
            continue

        r_value, p_value = pearsonr(x, y)

        records.append(
            {
                "region": f"region_{region_index}_{roi_name}",
                "correlation": round(float(r_value), 3),
                "p_value": round(float(p_value), 4)
            }
        )

        if SAVE_PER_REGION_PLOT:
            plt.figure(figsize=(6, 4))

            sns.regplot(
                x=x,
                y=y,
                scatter_kws={"alpha": 0.7}
            )

            plt.xlabel("Intrinsic dimension across videos")
            plt.ylabel("Mean H1 lifetime across videos")
            plt.title(
                f"{roi_name} - Week {WEEK_ID}\n"
                f"r = {r_value:.3f}, p = {p_value:.4g}, n = {len(x)}"
            )

            plt.tight_layout()

            output_path = os.path.join(
                OUTPUT_DIR,
                f"{roi_name}_week{WEEK_ID}_dimension_vs_lifetime.png"
            )

            plt.savefig(output_path, dpi=300)
            plt.close()

        print(
            f"[OK] ROI {roi_name}: "
            f"r = {r_value:.3f}, p = {p_value:.4g}, n = {len(x)}"
        )

    else:
        print(
            f"[Skipped] ROI {roi_name}: too few valid subjects "
            f"(dimension={len(subject_dimensions)}, lifetime={len(subject_lifetimes)})"
        )

output_csv = os.path.join(
    OUTPUT_DIR,
    f"correlations_dimension_lifetime_week{WEEK_ID}.csv"
)

pd.DataFrame(
    records,
    columns=["region", "correlation", "p_value"]
).to_csv(output_csv, index=False)

print(f"\nSaved results to: {output_csv}")