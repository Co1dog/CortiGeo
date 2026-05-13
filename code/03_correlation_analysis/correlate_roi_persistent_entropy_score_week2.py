"""
This script analyzes the relationship between ROI-level persistent entropy and
learning performance for Week 2 fMRI data. For each ROI, subject, and selected
video, it loads ROI-level fMRI time-series data, computes temporal-difference
activation patterns, extracts H1 persistence diagrams using persistent homology,
computes persistent entropy from positive finite lifetimes, averages entropy
values across videos for each subject, and correlates subject-level entropy
features with exam scores. The script saves ROI-level regression plots and a
summary CSV containing correlation coefficients, p-values, and sample counts.
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from ripser import ripser
from scipy.stats import pearsonr


mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

warnings.filterwarnings(
    "ignore",
    message="The input point cloud has more columns than rows"
)

ROOT_DIR = "ThinkLikeExpertsROIs"
WEEK_ID = 2
VIDEOS = [1, 2, 3, 4]
HOMOLOGY_DIMENSION = 1
OUTPUT_DIR = "persistent_entropy_results"
SCORE_CSV = "subject_weekly_data2.csv"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def compute_persistent_entropy(diagram):
    if diagram is None or diagram.size == 0:
        return np.nan

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    finite_mask = np.isfinite(births) & np.isfinite(deaths)
    births = births[finite_mask]
    deaths = deaths[finite_mask]

    lifetimes = deaths - births
    lifetimes = lifetimes[np.isfinite(lifetimes) & (lifetimes > 0)]

    if lifetimes.size == 0:
        return np.nan

    lifetime_sum = lifetimes.sum()

    if not np.isfinite(lifetime_sum) or lifetime_sum <= 0:
        return np.nan

    probabilities = lifetimes / lifetime_sum
    entropy = -np.sum(probabilities * np.log(probabilities))

    return float(entropy)


def get_roi_file_path(root_dir, subject_id, week_id, video_id, roi_name):
    return os.path.join(
        root_dir,
        f"sub-{subject_id}",
        f"ses-wk{week_id}",
        "func",
        "regions",
        f"{roi_name}_sub-{subject_id}_ses-wk{week_id}_task-vid{video_id}_bold.npy"
    )


print("Loading subject score table...")

score_df = pd.read_csv(SCORE_CSV)

scores = score_df["score"].astype(float).values
subject_ids = score_df["participant_id"].astype(str).values
score_map = dict(zip(subject_ids, scores))

print(f"Number of subjects: {len(subject_ids)}")

first_subject = subject_ids[0]
first_roi_dir = os.path.join(
    ROOT_DIR,
    f"sub-{first_subject}",
    f"ses-wk{WEEK_ID}",
    "func",
    "regions"
)

if not os.path.isdir(first_roi_dir):
    raise FileNotFoundError(f"ROI directory not found: {first_roi_dir}")

roi_names = sorted(
    {
        file_name.split("_sub-")[0]
        for file_name in os.listdir(first_roi_dir)
        if file_name.endswith(".npy")
    }
)

print(f"Number of ROIs: {len(roi_names)}")

correlation_records = []

for roi_index, roi_name in enumerate(roi_names, start=1):
    print(f"\n=== ROI {roi_index}/{len(roi_names)}: {roi_name} ===")

    subject_entropies = []
    subject_scores = []

    for subject_id in subject_ids:
        print(f"Processing subject {subject_id}")

        per_video_entropies = []

        for video_id in VIDEOS:
            roi_file_path = get_roi_file_path(
                ROOT_DIR,
                subject_id,
                WEEK_ID,
                video_id,
                roi_name
            )

            if not os.path.exists(roi_file_path):
                print(f"Missing file. Skipping video {video_id}: {roi_file_path}")
                continue

            voxel_data = np.load(roi_file_path)

            if voxel_data.shape[0] < 2:
                print(f"Time series too short. Skipping video {video_id}")
                continue

            voxel_diff = voxel_data[1:] - voxel_data[:-1]

            try:
                result = ripser(
                    voxel_diff,
                    maxdim=HOMOLOGY_DIMENSION
                )

                diagrams = result["dgms"]

                if len(diagrams) <= HOMOLOGY_DIMENSION:
                    print(f"No diagram for homology dimension {HOMOLOGY_DIMENSION}")
                    continue

                diagram = diagrams[HOMOLOGY_DIMENSION]

                if diagram is None or diagram.size == 0:
                    print(f"Empty diagram for video {video_id}")
                    continue

                entropy = compute_persistent_entropy(diagram)

                if np.isfinite(entropy):
                    per_video_entropies.append(float(entropy))
                    print(
                        f"Video {video_id}: entropy={entropy:.6f}, "
                        f"bars={diagram.shape[0]}"
                    )
                else:
                    print(f"Video {video_id}: entropy is not finite. Skipping.")

            except Exception as error:
                print(f"Video {video_id}: computation failed: {error}")
                continue

        if per_video_entropies:
            mean_entropy = float(np.mean(per_video_entropies))
            subject_entropies.append(mean_entropy)
            subject_scores.append(score_map[subject_id])

            print(
                f"Subject {subject_id}: mean entropy={mean_entropy:.6f} "
                f"from {len(per_video_entropies)} videos"
            )
        else:
            print(f"Subject {subject_id}: no valid videos. Skipping.")

    if len(subject_entropies) < 3:
        print("Too few valid subjects. Skipping correlation analysis.")
        continue

    subject_entropies = np.asarray(subject_entropies, dtype=float)
    subject_scores = np.asarray(subject_scores, dtype=float)

    r_value, p_value = pearsonr(subject_entropies, subject_scores)

    print(
        f"Pearson r={r_value:.3f}, "
        f"p={p_value:.4g}, n={len(subject_entropies)}"
    )

    plt.figure(figsize=(6, 4))

    sns.regplot(
        x=subject_entropies,
        y=subject_scores
    )

    plt.xlabel("Mean persistent entropy across videos")
    plt.ylabel("Score")
    plt.title(
        f"{roi_name} - Week {WEEK_ID}\n"
        f"r = {r_value:.2f}, p = {p_value:.4f}"
    )

    plt.tight_layout()

    figure_name = f"{roi_name}_week{WEEK_ID}_entropy_all_videos.png"
    plt.savefig(os.path.join(OUTPUT_DIR, figure_name), dpi=300)
    plt.close()

    correlation_records.append(
        {
            "region": f"region_{roi_index}_{roi_name}",
            "correlation": round(float(r_value), 3),
            "p_value": round(float(p_value), 4),
            "n_subjects": int(len(subject_entropies))
        }
    )

if correlation_records:
    correlation_df = pd.DataFrame(correlation_records)
    output_csv = os.path.join(
        OUTPUT_DIR,
        f"correlations_entropy_week{WEEK_ID}.csv"
    )
    correlation_df.to_csv(output_csv, index=False)
    print(f"\nSaved correlation results to: {output_csv}")
else:
    print("\nNo correlation results were exported.")