"""
This script computes multi-week ROI-level correlations between pairwise
persistent-homology Wasserstein distance and learning performance using raw fMRI
representations. For each week, ROI, subject pair, and video, it loads ROI-level
fMRI time-series data without temporal differencing, computes H1 persistence
diagrams, measures Wasserstein distance between subjects' persistence diagrams,
averages distances across videos for each subject pair, and correlates the
pairwise topological distance with the average exam score of the subject pair.
The script saves ROI-level regression plots, weekly CSV files, and an all-week
summary CSV.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from persim import wasserstein
from scipy.stats import pearsonr


plt.rcParams["text.usetex"] = False

warnings.filterwarnings(
    "ignore",
    message="The input point cloud has more columns than rows; did you mean to transpose?"
)

ROOT_DIR = "ThinkLikeExpertsROIs"
WEEKS = [1, 2, 3, 4, 5]
VIDEOS = [1, 2, 3, 4, 5, 6]
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_DIR = "pairwise_persistence_wasserstein_raw_results"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def get_roi_names(root_dir, subject_id, week_id):
    roi_dir = os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions"
    )

    if not os.path.isdir(roi_dir):
        print(f"[Week {week_id}] ROI directory not found. Skipping: {roi_dir}")
        return []

    roi_names = sorted(
        {
            file_name.split("_sub-")[0]
            for file_name in os.listdir(roi_dir)
            if file_name.endswith(".npy")
        }
    )

    if not roi_names:
        print(f"[Week {week_id}] No ROI .npy files found in: {roi_dir}")

    return roi_names


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

first_subject = subject_ids[0]
all_records = []

for week_id in WEEKS:
    roi_names = get_roi_names(
        ROOT_DIR,
        first_subject,
        week_id
    )

    if not roi_names:
        continue

    print(f"\n========== Starting Week {week_id}, ROIs: {len(roi_names)} ==========")

    week_records = []

    for roi_index, roi_name in enumerate(roi_names, start=1):
        roi_output_dir = os.path.join(OUTPUT_DIR, roi_name)
        os.makedirs(roi_output_dir, exist_ok=True)

        pair_distances = []
        pair_scores = []

        n_subjects = len(subject_ids)
        total_pairs = n_subjects * (n_subjects - 1) // 2

        print(
            f"[Week {week_id}][ROI {roi_index}/{len(roi_names)}] "
            f"Starting {roi_name}, total pairs={total_pairs}"
        )

        pair_counter = 0

        for i in range(n_subjects):
            subject_i = subject_ids[i]

            for j in range(i + 1, n_subjects):
                subject_j = subject_ids[j]
                pair_counter += 1

                video_distances = []
                valid_video_count = 0

                for video_id in VIDEOS:
                    path_i = get_roi_file_path(
                        ROOT_DIR,
                        subject_i,
                        week_id,
                        video_id,
                        roi_name
                    )

                    path_j = get_roi_file_path(
                        ROOT_DIR,
                        subject_j,
                        week_id,
                        video_id,
                        roi_name
                    )

                    if not (os.path.exists(path_i) and os.path.exists(path_j)):
                        continue

                    try:
                        data_i = np.load(path_i)
                        data_j = np.load(path_j)

                        if data_i.ndim != 2 or data_j.ndim != 2:
                            continue

                        if data_i.shape[0] < 3 or data_j.shape[0] < 3:
                            continue

                        if not np.isfinite(data_i).all() or not np.isfinite(data_j).all():
                            continue

                        diagrams_i = ripser(data_i, maxdim=1)["dgms"]
                        diagrams_j = ripser(data_j, maxdim=1)["dgms"]

                        h1_i = diagrams_i[1] if len(diagrams_i) > 1 else np.empty((0, 2))
                        h1_j = diagrams_j[1] if len(diagrams_j) > 1 else np.empty((0, 2))

                        if h1_i.size == 0 or h1_j.size == 0:
                            continue

                        distance = wasserstein(h1_i, h1_j)

                        if np.isfinite(distance):
                            video_distances.append(float(distance))
                            valid_video_count += 1

                    except Exception:
                        continue

                if video_distances:
                    pair_distances.append(float(np.mean(video_distances)))
                    pair_scores.append(
                        (score_map[subject_i] + score_map[subject_j]) / 2.0
                    )

                if pair_counter % 10 == 0 or pair_counter == total_pairs:
                    print(
                        f"  - ROI {roi_name}: pair {pair_counter}/{total_pairs}, "
                        f"current pair valid videos={valid_video_count}, "
                        f"cumulative valid pairs={len(pair_distances)}",
                        flush=True
                    )

        r_value = np.nan
        p_value = np.nan
        n_pairs = 0

        if len(pair_distances) >= 3:
            x = np.asarray(pair_distances, dtype=float)
            y = np.asarray(pair_scores, dtype=float)

            valid_mask = np.isfinite(x) & np.isfinite(y)
            x = x[valid_mask]
            y = y[valid_mask]

            if x.size >= 3 and y.size >= 3:
                r_value, p_value = pearsonr(x, y)
                n_pairs = x.size

                plt.figure(figsize=(6, 5))
                sns.set(style="white")

                sns.regplot(
                    x=x,
                    y=y,
                    scatter_kws={"s": 80, "alpha": 0.8},
                    line_kws={"linewidth": 3}
                )

                plt.xlabel("Persistence Wasserstein distance across videos", fontsize=12)
                plt.ylabel("Pair average score", fontsize=12)

                p_text = "p < 0.05" if p_value < 0.05 else f"p = {p_value:.3f}"

                plt.text(
                    0.03,
                    0.97,
                    f"r = {r_value:.3f}\n{p_text}\nN = {n_pairs}",
                    transform=plt.gca().transAxes,
                    fontsize=11,
                    va="top",
                    bbox=dict(
                        facecolor="#f0f0f0",
                        edgecolor="none",
                        boxstyle="round,pad=0.3"
                    )
                )

                plt.title(f"{roi_name} - Week {week_id}", fontsize=13, weight="bold")
                sns.despine(top=True, right=True)

                plt.tight_layout()

                output_png = os.path.join(
                    roi_output_dir,
                    f"{roi_name}_week{week_id}_pairwise_persistence_wasserstein_raw.png"
                )

                plt.savefig(output_png, dpi=300)
                plt.close()

        record = {
            "region": roi_name,
            "week": week_id,
            "r": None if pd.isna(r_value) else round(float(r_value), 4),
            "p": None if pd.isna(p_value) else f"{p_value:.4g}",
            "n_pairs": int(n_pairs)
        }

        week_records.append(record)
        all_records.append(record)

        print(
            f"[Week {week_id}] Completed ROI {roi_name}: "
            f"valid_pairs={record['n_pairs']}, r={record['r']}, p={record['p']}"
        )

    week_df = pd.DataFrame(
        week_records,
        columns=["region", "week", "r", "p", "n_pairs"]
    )

    week_csv = os.path.join(
        OUTPUT_DIR,
        f"correlations_persistence_wasserstein_pairs_week{week_id}_raw.csv"
    )

    week_df.to_csv(week_csv, index=False)
    print(f"[Week {week_id}] Results saved to: {week_csv}")

summary_df = pd.DataFrame(
    all_records,
    columns=["region", "week", "r", "p", "n_pairs"]
)

summary_csv = os.path.join(
    OUTPUT_DIR,
    "correlations_persistence_wasserstein_pairs_all_weeks_raw.csv"
)

summary_df.to_csv(summary_csv, index=False)

print(f"Done. Summary results saved to: {summary_csv}")