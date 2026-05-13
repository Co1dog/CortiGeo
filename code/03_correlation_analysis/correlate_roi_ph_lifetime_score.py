"""
This script analyzes the relationship between persistent-homology lifetime
features and learning performance in ROI-level fMRI representations. For each
ROI in a selected recording week, it loads subject-specific fMRI time-series
data across multiple videos, computes temporal-difference activation patterns,
extracts H1 persistence diagrams using persistent homology, calculates the mean
positive H1 lifetime for each subject, and correlates these topological features
with exam scores. The script saves ROI-level scatter plots and a summary CSV
containing correlation coefficients and p-values.
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from scipy.stats import pearsonr


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

ROOT_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEOS = [1, 2, 3, 4]
SCORE_CSV = "subject_weekly_data2.csv"
OUTPUT_DIR = "ph_lifetime_results"

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
        raise FileNotFoundError(f"ROI directory not found: {roi_dir}")

    roi_names = sorted(
        {
            file_name.split("_sub-")[0]
            for file_name in os.listdir(roi_dir)
            if file_name.endswith(".npy")
        }
    )

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


def compute_mean_h1_lifetime(voxel_data):
    if voxel_data.shape[0] < 3:
        return None

    temporal_diff = voxel_data[1:] - voxel_data[:-1]
    diagrams = ripser(temporal_diff)["dgms"]

    if len(diagrams) <= 1:
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


def plot_lifetime_score_correlation(
    mean_lifetimes,
    scores,
    roi_name,
    week_id,
    r_value,
    p_value,
    output_dir
):
    plt.figure(figsize=(6, 4))

    sns.regplot(
        x=mean_lifetimes,
        y=scores
    )

    plt.xlabel("Mean H1 lifetime across videos")
    plt.ylabel("Score")
    plt.title(
        f"{roi_name} - Week {week_id}\n"
        f"r = {r_value:.2f}, p = {p_value:.4f}"
    )

    plt.tight_layout()

    output_path = os.path.join(
        output_dir,
        f"{roi_name}_week{week_id}_all_videos.png"
    )

    plt.savefig(output_path, dpi=300)
    plt.close()


score_df = pd.read_csv(SCORE_CSV)

scores_all = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values
score_map = dict(zip(subject_ids, scores_all))

first_subject = subject_ids[0]
roi_names = get_roi_names(ROOT_DIR, first_subject, WEEK_ID)

correlation_records = []

for region_index, roi_name in enumerate(roi_names, start=1):
    mean_lifetimes = []
    valid_scores = []

    for subject_id in subject_ids:
        per_video_lifetimes = []

        for video_id in VIDEOS:
            roi_file_path = get_roi_file_path(
                ROOT_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )

            if not os.path.exists(roi_file_path):
                continue

            voxel_data = np.load(roi_file_path)
            mean_lifetime = compute_mean_h1_lifetime(voxel_data)

            if mean_lifetime is not None:
                per_video_lifetimes.append(mean_lifetime)

        if per_video_lifetimes:
            mean_lifetimes.append(float(np.mean(per_video_lifetimes)))
            valid_scores.append(score_map[subject_id])

    if len(mean_lifetimes) < 3:
        continue

    mean_lifetimes = np.asarray(mean_lifetimes, dtype=float)
    valid_scores = np.asarray(valid_scores, dtype=float)

    valid_mask = np.isfinite(mean_lifetimes) & np.isfinite(valid_scores)
    mean_lifetimes = mean_lifetimes[valid_mask]
    valid_scores = valid_scores[valid_mask]

    if mean_lifetimes.size < 3:
        continue

    r_value, p_value = pearsonr(mean_lifetimes, valid_scores)

    plot_lifetime_score_correlation(
        mean_lifetimes,
        valid_scores,
        roi_name,
        WEEK_ID,
        r_value,
        p_value,
        OUTPUT_DIR
    )

    correlation_records.append(
        {
            "region": f"region_{region_index}_{roi_name}",
            "correlation": round(float(r_value), 3),
            "p_value": round(float(p_value), 4)
        }
    )

correlation_df = pd.DataFrame(correlation_records)

output_csv = os.path.join(
    OUTPUT_DIR,
    f"correlations_lifetime_week{WEEK_ID}.csv"
)

correlation_df.to_csv(output_csv, index=False)

print(f"Saved PH lifetime correlation results to: {output_csv}")