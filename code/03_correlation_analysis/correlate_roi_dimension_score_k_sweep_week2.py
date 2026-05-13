"""
This script performs a k-sensitivity analysis for ROI-level intrinsic dimension
and learning performance correlations in Week 2 fMRI data. For each predefined
neighborhood size, it scans all available Harvard-Oxford ROI time-series files,
standardizes voxel activity, computes temporal-difference activation patterns,
estimates intrinsic dimension using the MLE estimator, and correlates each ROI's
intrinsic dimension values with exam scores. The script saves region-wise
correlation CSV files and regression plots for each tested neighborhood size.
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
K_VALUES = [3, 5, 30, 50]

df = pd.read_csv(SCORE_CSV)

scores = df["score"].values
subject_ids = df["participant_id"].values

print(scores)
print(subject_ids)

for k_value in K_VALUES:
    output_fig_dir = f"brain_region_plots_week{WEEK_ID}_k{k_value}"
    os.makedirs(output_fig_dir, exist_ok=True)

    example_subject = f"sub-{subject_ids[0]}"
    example_regions_path = os.path.join(
        BASE_DIR,
        example_subject,
        f"ses-wk{WEEK_ID}",
        "func",
        "regions"
    )

    if not os.path.isdir(example_regions_path):
        print(
            f"[Week {WEEK_ID}] Warning: region directory not found: "
            f"{example_regions_path}. Skipping this week."
        )
        continue

    region_files = [
        file_name
        for file_name in os.listdir(example_regions_path)
        if file_name.endswith(".npy")
    ]

    regions = sorted(
        list(
            set(
                file_name.split("_sub-")[0]
                for file_name in region_files
            )
        )
    )

    results = []

    for region in regions:
        score_result = []
        dimension_result = []

        try:
            for video_id in range(1, 6):
                for subject_index in range(len(subject_ids)):
                    file_path = os.path.join(
                        BASE_DIR,
                        f"sub-{subject_ids[subject_index]}",
                        f"ses-wk{WEEK_ID}",
                        "func",
                        "regions",
                        (
                            f"{region}_sub-{subject_ids[subject_index]}"
                            f"_ses-wk{WEEK_ID}_task-vid{video_id}_bold.npy"
                        )
                    )

                    if os.path.exists(file_path):
                        try:
                            voxel_data = np.load(file_path)

                            if voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                                print(
                                    f"[Week {WEEK_ID}] Warning: data shape too small "
                                    f"for {file_path}: {voxel_data.shape}"
                                )
                                continue

                            if np.any(np.isnan(voxel_data)) or np.any(np.isinf(voxel_data)):
                                print(
                                    f"[Week {WEEK_ID}] Warning: invalid values found "
                                    f"in {file_path}"
                                )
                                continue

                            voxel_data_norm = np.zeros_like(voxel_data)
                            scaler = StandardScaler()

                            for feature_index in range(voxel_data.shape[1]):
                                if np.std(voxel_data[:, feature_index]) > 1e-10:
                                    voxel_data_norm[:, feature_index] = scaler.fit_transform(
                                        voxel_data[:, feature_index].reshape(-1, 1)
                                    ).ravel()

                            voxel_diff = voxel_data_norm[1:] - voxel_data_norm[:-1]

                            if np.any(np.isnan(voxel_diff)) or np.any(np.isinf(voxel_diff)):
                                print(
                                    f"[Week {WEEK_ID}] Warning: invalid values found "
                                    f"after temporal differencing in {file_path}"
                                )
                                continue

                            dimension = skdim.id.MLE().fit_transform(
                                voxel_diff,
                                n_neighbors=k_value
                            )

                            if not np.isfinite(dimension):
                                print(
                                    f"[Week {WEEK_ID}] Warning: invalid dimension "
                                    f"estimate for {file_path}"
                                )
                                continue

                            dimension_result.append(dimension)
                            score_result.append(scores[subject_index])

                        except Exception as error:
                            print(
                                f"[Week {WEEK_ID}] Error while processing "
                                f"{file_path}: {error}"
                            )
                            continue

            if len(dimension_result) > 0:
                valid_mask = np.isfinite(dimension_result)
                dimension_result = np.array(dimension_result)[valid_mask]
                score_result = np.array(score_result)[valid_mask]

                if len(dimension_result) < 3:
                    print(
                        f"[Week {WEEK_ID}] Warning: too few valid samples for "
                        f"{region}. Skipping."
                    )
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
                    f"Week {WEEK_ID} - Region: {region_display_name}, k={k_value}",
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

                fig_filename = os.path.join(
                    output_fig_dir,
                    f"{region}_week{WEEK_ID}_k{k_value}.png"
                )

                plt.savefig(fig_filename, dpi=300, bbox_inches="tight")
                plt.close()

                print(f"\n[Week {WEEK_ID}] Region: {region_display_name}, k={k_value}")
                print(f"[Week {WEEK_ID}] Correlation coefficient: {r_value:.3f}")
                print(f"[Week {WEEK_ID}] P-value: {p_value:.3g}")
                print(f"[Week {WEEK_ID}] Number of valid samples: {len(dimension_result)}")
                print(f"[Week {WEEK_ID}] Figure saved as: {fig_filename}")

        except Exception as error:
            print(f"[Week {WEEK_ID}] Error while processing region {region}: {error}")
            continue

    results_df = pd.DataFrame(
        results,
        columns=["region", "correlation", "p_value"]
    )

    output_csv = f"correlations_dimension_week{WEEK_ID}_k{k_value}.csv"
    results_df.to_csv(output_csv, index=False)

    print(f"[Week {WEEK_ID}] Results saved to: {output_csv}")