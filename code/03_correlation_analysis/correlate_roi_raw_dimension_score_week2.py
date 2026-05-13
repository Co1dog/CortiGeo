"""
This script computes ROI-level correlations between raw-signal intrinsic
dimension and learning performance for Week 2 fMRI data. For each ROI, subject,
and video, it loads the ROI-level fMRI time series, applies feature-wise
standardization without temporal differencing, estimates intrinsic dimension
using the MLE estimator with a fixed neighborhood size, and correlates the
resulting dimension values with exam scores. The script saves regression plots
for each ROI and a summary CSV containing correlation coefficients and p-values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skdim

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


BASE_DIR = "ThinkLikeExpertsROIs"
SCORE_CSV = "subject_weekly_data2.csv"
WEEK_ID = 2
N_NEIGHBORS = 10
VIDEOS = [1, 2, 3, 4, 5]

OUTPUT_FIG_DIR = f"brain_region_plots_week{WEEK_ID}_k{N_NEIGHBORS}_raw"
OUTPUT_CSV = f"correlations_dimension_week{WEEK_ID}_k{N_NEIGHBORS}_raw.csv"

os.makedirs(OUTPUT_FIG_DIR, exist_ok=True)


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
    if data.ndim != 2 or data.shape[0] < n_neighbors + 1 or data.shape[1] < 2:
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


def get_region_names(base_dir, subject_id, week_id):
    example_regions_path = os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(example_regions_path):
        print(f"[Week {week_id}] Warning: region directory not found: {example_regions_path}")
        raise SystemExit(0)

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

scores = score_df["score"].values
subject_ids = score_df["participant_id"].astype(str).values

print(scores)
print(subject_ids)

regions = get_region_names(
    BASE_DIR,
    subject_ids[0],
    WEEK_ID
)

results = []

for region in regions:
    score_result = []
    dimension_result = []

    for video_id in VIDEOS:
        for subject_index, subject_id in enumerate(subject_ids):
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

                if voxel_data.ndim != 2 or voxel_data.shape[0] < N_NEIGHBORS + 1 or voxel_data.shape[1] < 2:
                    print(
                        f"[Week {WEEK_ID}] Warning: data shape too small "
                        f"for {file_path}: {voxel_data.shape}"
                    )
                    continue

                if not np.isfinite(voxel_data).all():
                    print(f"[Week {WEEK_ID}] Warning: invalid values found in {file_path}")
                    continue

                normalized_data = standardize_time_series(voxel_data)

                dimension = estimate_dimension(
                    normalized_data,
                    N_NEIGHBORS
                )

                if dimension is None:
                    print(f"[Week {WEEK_ID}] Warning: invalid dimension estimate for {file_path}")
                    continue

                dimension_result.append(dimension)
                score_result.append(scores[subject_index])

            except Exception as error:
                print(f"[Week {WEEK_ID}] Error while processing {file_path}: {error}")
                continue

    if dimension_result:
        dimension_result = np.asarray(dimension_result, dtype=float)
        score_result = np.asarray(score_result, dtype=float)

        valid_mask = np.isfinite(dimension_result) & np.isfinite(score_result)
        dimension_result = dimension_result[valid_mask]
        score_result = score_result[valid_mask]

        if len(dimension_result) < 3:
            print(f"[Week {WEEK_ID}] Warning: too few valid samples for {region}. Skipping.")
            continue

        r_value, p_value = pearsonr(dimension_result, score_result)

        results.append(
            {
                "region": f"region_{region}",
                "correlation": float(r_value),
                "p_value": float(p_value)
            }
        )

        plt.figure(figsize=(10, 6))

        sns.regplot(
            x=dimension_result,
            y=score_result,
            scatter_kws={"alpha": 0.7},
            line_kws={"color": "navy"}
        )

        region_display_name = region.replace("_", " ")

        plt.title(
            f"Week {WEEK_ID} - Region: {region_display_name} (raw), "
            f"k={N_NEIGHBORS}",
            fontsize=15
        )

        plt.xlabel("Intrinsic dimension of raw signal")
        plt.ylabel("Score")

        text = (
            f"Pearson r = {r_value:.3f}, p = {p_value:.3g}\n"
            f"n = {len(dimension_result)}"
        )

        plt.text(
            0.05,
            0.95,
            text,
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

        plt.tight_layout()

        fig_filename = os.path.join(
            OUTPUT_FIG_DIR,
            f"{region}_week{WEEK_ID}_k{N_NEIGHBORS}_raw.png"
        )

        plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[Week {WEEK_ID}] Region: {region_display_name}, k={N_NEIGHBORS} (raw)")
        print(f"[Week {WEEK_ID}] Correlation coefficient: {r_value:.3f}")
        print(f"[Week {WEEK_ID}] P-value: {p_value:.3g}")
        print(f"[Week {WEEK_ID}] Number of valid samples: {len(dimension_result)}")
        print(f"[Week {WEEK_ID}] Figure saved as: {fig_filename}")

results_df = pd.DataFrame(
    results,
    columns=["region", "correlation", "p_value"]
)

results_df.to_csv(OUTPUT_CSV, index=False)

print(f"[Week {WEEK_ID}] Results saved to: {OUTPUT_CSV}")
