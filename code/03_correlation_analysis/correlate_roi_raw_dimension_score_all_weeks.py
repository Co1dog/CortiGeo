"""
This script computes multi-week ROI-level correlations between raw fMRI
intrinsic dimension and learning performance. For each selected week, ROI,
subject, and video, it loads ROI-level fMRI time-series data, applies
feature-wise z-score normalization without temporal differencing, estimates
intrinsic dimension using the MLE estimator with a fixed neighborhood size, and
correlates the resulting dimension values with exam scores. The script saves
weekly regression plots, weekly ROI-level correlation CSV files, weekly mean
dimension CSV files, and an all-week summary CSV.
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
WEEKS = [2, 3, 4, 5]
N_NEIGHBORS = 10
VIDEOS = range(1, 7)
OUTPUT_ROOT = "all_weeks_raw_dimension_score_correlation"
SCORE_CSV = "subject_weekly_data2.csv"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def safe_zscore_columns(data):
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
        return []

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

print("Scores:", scores)
print("Subjects:", subject_ids)

all_weeks_rows = []

for week_id in WEEKS:
    print(f"\n========== Starting Week {week_id} ==========")

    week_output_dir = os.path.join(OUTPUT_ROOT, f"week{week_id}")
    plot_dir = os.path.join(week_output_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    regions = get_region_names(
        BASE_DIR,
        subject_ids[0],
        week_id
    )

    if not regions:
        print(f"[Week {week_id}] No available ROI files. Skipping this week.")
        continue

    print(f"[Week {week_id}] Number of candidate regions: {len(regions)}")

    correlation_rows = []
    dimension_summary_rows = []

    for region in regions:
        score_values = []
        dimension_values = []

        for video_id in VIDEOS:
            for subject_index, subject_id in enumerate(subject_ids):
                file_path = get_roi_file_path(
                    BASE_DIR,
                    subject_id,
                    week_id,
                    video_id,
                    region
                )

                if not os.path.exists(file_path):
                    continue

                try:
                    voxel_data = np.load(file_path)

                    if voxel_data.ndim != 2 or voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                        print(
                            f"[Week {week_id}] Warning: abnormal data shape "
                            f"for {file_path}: {voxel_data.shape}"
                        )
                        continue

                    if not np.isfinite(voxel_data).all():
                        print(f"[Week {week_id}] Warning: NaN or Inf found in {file_path}")
                        continue

                    normalized_data = safe_zscore_columns(voxel_data)

                    dimension = skdim.id.MLE().fit_transform(
                        normalized_data,
                        n_neighbors=N_NEIGHBORS
                    )

                    dimension = float(np.asarray(dimension).mean())

                    if not np.isfinite(dimension):
                        print(f"[Week {week_id}] Warning: invalid dimension estimate for {file_path}")
                        continue

                    dimension_values.append(dimension)
                    score_values.append(float(scores[subject_index]))

                except Exception as error:
                    print(f"[Week {week_id}] Error while processing {file_path}: {error}")
                    continue

        if dimension_values:
            dimension_array = np.asarray(dimension_values, dtype=float)
            score_array = np.asarray(score_values, dtype=float)

            valid_mask = np.isfinite(dimension_array) & np.isfinite(score_array)
            dimension_array = dimension_array[valid_mask]
            score_array = score_array[valid_mask]

            if len(dimension_array) < 3:
                print(
                    f"[Week {week_id}] Warning: too few valid samples for "
                    f"{region} (n={len(dimension_array)}). Skipping."
                )
                continue

            r_value, p_value = pearsonr(dimension_array, score_array)
            mean_dimension = float(np.mean(dimension_array))

            correlation_record = {
                "week": week_id,
                "region": f"region_{region}",
                "correlation": float(r_value),
                "p_value": float(p_value),
                "n": int(len(dimension_array))
            }

            correlation_rows.append(correlation_record)
            all_weeks_rows.append(correlation_record)

            dimension_summary_rows.append(
                {
                    "region": f"region_{region}",
                    "dimension": mean_dimension,
                    "n": int(len(dimension_array))
                }
            )

            plt.figure(figsize=(10, 6))

            sns.regplot(
                x=dimension_array,
                y=score_array,
                scatter_kws={"alpha": 0.7},
                line_kws={"color": "navy"}
            )

            region_display_name = region.replace("_", " ")

            plt.title(
                f"Week {week_id} - Region: {region_display_name}, "
                f"k={N_NEIGHBORS} (Raw fMRI)",
                fontsize=15
            )
            plt.xlabel("Intrinsic dimension")
            plt.ylabel("Score")

            text = (
                f"Pearson r = {r_value:.3f}, p = {p_value:.3g}\n"
                f"n = {len(dimension_array)}"
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
                plot_dir,
                f"{region}_week{week_id}_k{N_NEIGHBORS}_raw.png"
            )

            plt.savefig(figure_path, dpi=300, bbox_inches="tight")
            plt.close()

            print(
                f"[Week {week_id}] {region_display_name}: "
                f"n={len(dimension_array)}, mean_dimension={mean_dimension:.3f}, "
                f"r={r_value:.3f}, p={p_value:.3g}"
            )
            print(f"[Week {week_id}] Figure saved to: {figure_path}")

        else:
            print(f"[Week {week_id}] Warning: no valid dimension values for {region}")

    correlation_df = pd.DataFrame(
        correlation_rows,
        columns=["week", "region", "correlation", "p_value", "n"]
    )

    correlation_csv = os.path.join(
        week_output_dir,
        f"correlations_dimension_week{week_id}_k{N_NEIGHBORS}_raw.csv"
    )

    correlation_df.to_csv(correlation_csv, index=False)

    dimension_summary_df = pd.DataFrame(
        dimension_summary_rows,
        columns=["region", "dimension", "n"]
    )

    dimension_csv = os.path.join(
        week_output_dir,
        f"mean_dimensions_week{week_id}_k{N_NEIGHBORS}_raw.csv"
    )

    dimension_summary_df.to_csv(dimension_csv, index=False)

    print(f"[Week {week_id}] Saved {len(correlation_df)} correlation rows")
    print(f"[Week {week_id}] Saved {len(dimension_summary_df)} mean-dimension rows")
    print(f"[Week {week_id}] Correlation CSV: {correlation_csv}")
    print(f"[Week {week_id}] Dimension CSV: {dimension_csv}")
    print(f"========== Finished Week {week_id} ==========\n")

if all_weeks_rows:
    all_df = pd.DataFrame(
        all_weeks_rows,
        columns=["week", "region", "correlation", "p_value", "n"]
    )

    all_csv = os.path.join(
        OUTPUT_ROOT,
        f"correlations_dimension_all_weeks_k{N_NEIGHBORS}_raw_summary.csv"
    )

    all_df.to_csv(all_csv, index=False)
    print(f"[All Weeks] Summary saved to: {all_csv}")
else:
    print("[All Weeks] No valid results were generated.")