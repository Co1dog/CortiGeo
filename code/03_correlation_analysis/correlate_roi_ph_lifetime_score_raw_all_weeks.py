"""
This script computes multi-week ROI-level correlations between raw fMRI
persistent-homology H1 lifetime features and learning performance. For each
week, ROI, subject, and video, it loads ROI-level fMRI time-series data, applies
feature-wise z-score normalization without temporal differencing, computes H1
persistence diagrams from the raw normalized point cloud, extracts the mean
positive H1 lifetime, averages lifetime values across videos for each subject,
and correlates the subject-level lifetime feature with exam scores. The script
saves weekly regression plots, weekly correlation CSV files, and an all-week
summary CSV.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

ROOT_DIR = "ThinkLikeExpertsROIs"
WEEKS = [1, 2, 3, 4, 5]
VIDEOS = [1, 2, 3, 4, 5, 6]
OUTPUT_ROOT = "ph_lifetime_raw_all_weeks"
SCORE_CSV = "subject_weekly_data2.csv"

os.makedirs(OUTPUT_ROOT, exist_ok=True)


def safe_zscore_columns(data):
    data = np.asarray(data, dtype=float)
    normalized_data = np.zeros_like(data)

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


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


score_df = pd.read_csv(SCORE_CSV)

scores = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values
score_map = dict(zip(subject_ids, scores))

roi_names = None

for probe_week in WEEKS:
    probe_subject = subject_ids[0]
    probe_roi_dir = os.path.join(
        ROOT_DIR,
        f"sub-{probe_subject}",
        f"ses-wk{probe_week}",
        "func",
        "regions"
    )

    if os.path.isdir(probe_roi_dir):
        roi_names = sorted(
            {
                file_name.split("_sub-")[0]
                for file_name in os.listdir(probe_roi_dir)
                if file_name.endswith(".npy")
            }
        )
        break

if roi_names is None or len(roi_names) == 0:
    raise FileNotFoundError(
        "No ROI .npy files were found for the example subject in any selected week."
    )

all_weeks_rows = []

for week_id in WEEKS:
    print(f"\n========== Starting Week {week_id} ==========")

    week_output_dir = os.path.join(OUTPUT_ROOT, f"week{week_id}")
    plot_dir = os.path.join(week_output_dir, "plots")

    os.makedirs(plot_dir, exist_ok=True)

    week_roi_dir = os.path.join(
        ROOT_DIR,
        f"sub-{subject_ids[0]}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(week_roi_dir):
        print(f"[Week {week_id}] ROI directory not found: {week_roi_dir}. Skipping this week.")
        continue

    correlation_rows = []

    for region_index, roi_name in enumerate(roi_names, start=1):
        mean_lifetimes = []
        valid_scores = []

        for subject_id in subject_ids:
            per_video_lifetimes = []

            for video_id in VIDEOS:
                roi_file_path = get_roi_file_path(
                    ROOT_DIR,
                    subject_id,
                    week_id,
                    video_id,
                    roi_name
                )

                if not os.path.exists(roi_file_path):
                    continue

                try:
                    voxel_data = np.load(roi_file_path)

                    if voxel_data.ndim != 2 or voxel_data.shape[0] < 10 or voxel_data.shape[1] < 2:
                        continue

                    if not np.isfinite(voxel_data).all():
                        continue

                    normalized_data = safe_zscore_columns(voxel_data)

                    diagrams = ripser(normalized_data)["dgms"]

                    if len(diagrams) < 2 or diagrams[1].size == 0:
                        continue

                    h1_diagram = diagrams[1]
                    lifetimes = h1_diagram[:, 1] - h1_diagram[:, 0]
                    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]

                    if lifetimes.size > 0:
                        per_video_lifetimes.append(float(np.mean(lifetimes)))

                except Exception:
                    continue

            if per_video_lifetimes:
                mean_lifetimes.append(float(np.mean(per_video_lifetimes)))
                valid_scores.append(float(score_map[subject_id]))

        if len(mean_lifetimes) < 3:
            continue

        mean_lifetimes = np.asarray(mean_lifetimes, dtype=float)
        valid_scores = np.asarray(valid_scores, dtype=float)

        valid_mask = np.isfinite(mean_lifetimes) & np.isfinite(valid_scores)
        mean_lifetimes = mean_lifetimes[valid_mask]
        valid_scores = valid_scores[valid_mask]

        if len(mean_lifetimes) < 3:
            continue

        r_value, p_value = pearsonr(mean_lifetimes, valid_scores)

        plt.figure(figsize=(6, 4))

        sns.regplot(
            x=mean_lifetimes,
            y=valid_scores
        )

        plt.xlabel("Mean H1 lifetime across videos (raw fMRI)")
        plt.ylabel("Score")
        plt.title(
            f"{roi_name} - Week {week_id}\n"
            f"r = {r_value:.2f}, p = {p_value:.4f}"
        )

        plt.tight_layout()

        figure_path = os.path.join(
            plot_dir,
            f"{roi_name}_week{week_id}_all_videos_raw.png"
        )

        plt.savefig(figure_path, dpi=300)
        plt.close()

        record = {
            "week": week_id,
            "region": f"region_{region_index}_{roi_name}",
            "correlation": round(float(r_value), 3),
            "p_value": round(float(p_value), 4),
            "n": int(len(mean_lifetimes))
        }

        correlation_rows.append(record)
        all_weeks_rows.append(record)

    correlation_df = pd.DataFrame(
        correlation_rows,
        columns=["week", "region", "correlation", "p_value", "n"]
    )

    week_csv = os.path.join(
        week_output_dir,
        f"correlations_lifetime_week{week_id}_raw.csv"
    )

    correlation_df.to_csv(week_csv, index=False)

    print(f"[Week {week_id}] Saved {len(correlation_df)} correlation rows to: {week_csv}")
    print(f"========== Finished Week {week_id} ==========\n")

if all_weeks_rows:
    all_df = pd.DataFrame(
        all_weeks_rows,
        columns=["week", "region", "correlation", "p_value", "n"]
    )

    all_csv = os.path.join(
        OUTPUT_ROOT,
        "correlations_lifetime_all_weeks_raw_summary.csv"
    )

    all_df.to_csv(all_csv, index=False)
    print(f"[All Weeks] Summary saved to: {all_csv}")
else:
    print("[All Weeks] No valid results were generated.")