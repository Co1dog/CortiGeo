"""
This script computes ROI-level intrinsic dimension and score correlations for
Week 6 recap fMRI sessions. For each ROI, subject, and recap task, it loads the
ROI-level fMRI time series, applies feature-wise standardization, computes
temporal-difference activation patterns, estimates intrinsic dimension using the
MLE estimator, and correlates the resulting dimension values with exam scores.
The script saves ROI-level regression plots, a correlation summary CSV, and a
separate CSV containing the raw dimension values for all valid ROI observations.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import skdim

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr


WEEK_ID = 6
N_NEIGHBORS = 10
SCORE_CSV = "subject_weekly_data2.csv"
BASE_DIR = "ThinkLikeExpertsROIs"
OUTPUT_FIG_DIR = f"brain_region_plots_week{WEEK_ID}"

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


def get_recap_file_path(base_dir, subject_id, week_id, recap_id, region):
    return os.path.join(
        base_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{region}_sub-{subject_id}_ses-wk{week_id}_task-wk{recap_id}recap_bold.npy"
    )


score_df = pd.read_csv(SCORE_CSV)

scores = score_df["score"].values
subject_ids = score_df["participant_id"].astype(str).values

print(scores)
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
    raise FileNotFoundError(
        f"[Week {WEEK_ID}] Directory not found: {example_regions_path}"
    )

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

correlation_results = []
dimension_rows = []

for region in regions:
    score_result = []
    dimension_result = []

    for recap_id in range(1, 6):
        for subject_index, subject_id in enumerate(subject_ids):
            file_path = get_recap_file_path(
                BASE_DIR,
                subject_id,
                WEEK_ID,
                recap_id,
                region
            )

            if not os.path.exists(file_path):
                continue

            try:
                voxel_data = np.load(file_path)

                if voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                    print(
                        f"[Week {WEEK_ID}] Warning: data shape too small "
                        f"for {file_path}: {voxel_data.shape}"
                    )
                    continue

                if not np.isfinite(voxel_data).all():
                    print(f"[Week {WEEK_ID}] Warning: invalid values found in {file_path}")
                    continue

                normalized_data = standardize_time_series(voxel_data)
                temporal_diff = normalized_data[1:] - normalized_data[:-1]

                if not np.isfinite(temporal_diff).all():
                    print(
                        f"[Week {WEEK_ID}] Warning: invalid values found after "
                        f"temporal differencing in {file_path}"
                    )
                    continue

                dimension = skdim.id.MLE().fit_transform(
                    temporal_diff,
                    n_neighbors=N_NEIGHBORS
                )

                dimension = float(np.asarray(dimension).mean())

                if not np.isfinite(dimension):
                    print(f"[Week {WEEK_ID}] Warning: invalid dimension estimate for {file_path}")
                    continue

                dimension_result.append(dimension)
                score_result.append(float(scores[subject_index]))

                dimension_rows.append(
                    {
                        "region": f"region_{region}",
                        "dimension": dimension
                    }
                )

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

        correlation_results.append(
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

        plt.title(
            f"Week {WEEK_ID} - Region: {region.replace('_', ' ')}, "
            f"k={N_NEIGHBORS}",
            fontsize=15
        )

        plt.xlabel("Intrinsic dimension")
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

        figure_path = os.path.join(
            OUTPUT_FIG_DIR,
            f"{region}_week{WEEK_ID}_k{N_NEIGHBORS}.png"
        )

        plt.savefig(figure_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"\n[Week {WEEK_ID}] Region: {region.replace('_', ' ')}, k={N_NEIGHBORS}")
        print(f"[Week {WEEK_ID}] Correlation coefficient: {r_value:.3f}")
        print(f"[Week {WEEK_ID}] P-value: {p_value:.3g}")
        print(f"[Week {WEEK_ID}] Number of valid samples: {len(dimension_result)}")
        print(f"[Week {WEEK_ID}] Figure saved as: {figure_path}")

correlation_df = pd.DataFrame(
    correlation_results,
    columns=["region", "correlation", "p_value"]
)

correlation_csv = f"correlations_dimension_week{WEEK_ID}.csv"
correlation_df.to_csv(correlation_csv, index=False)

print(f"[Week {WEEK_ID}] Correlation results saved to: {correlation_csv}")

dimension_df = pd.DataFrame(
    dimension_rows,
    columns=["region", "dimension"]
)

dimension_csv = f"dimensions_week{WEEK_ID}.csv"
dimension_df.to_csv(dimension_csv, index=False)

print(f"[Week {WEEK_ID}] Dimension values saved to: {dimension_csv}")